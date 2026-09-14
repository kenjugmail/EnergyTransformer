from dataclasses import asdict
import json
import re
import pytest
from energy_real import diagnostics
from energy_real.reasoning import compare,parser
from energy_real.service import ServiceUnavailable


class FixtureService:
    """Test-only deterministic parser/reader. Never reported as live model quality."""
    def __init__(self,episodes):
        self.edits={e.source_text:e for episode in episodes for e in episode.edits}
        self.calls=[];self.bad_query=False;self.scoring_offline=False

    def health(self):return {'model':'fixture-v2-only','decode_mode':'test'}

    def parse(self,question):
        if question.startswith('Follow '):
            match=re.fullmatch(r'Follow ([\w/]+) from ([\w-]+)\.',question)
            return {'subject':match[2],'relations':match[1].split('/')}
        if question.startswith('Which organization'):
            return {'subject':re.search(r'owns (.+)\?',question)[1],'relations':['owned_by','works_at']}
        if question.startswith('In which place'):
            return {'subject':re.search(r'owns (.+)\?',question)[1],'relations':['owned_by','works_at','located_in']}
        return {'subject':re.search(r'associated with (.+)\.',question)[1],'relations':['located_in']}

    def structured(self,instruction,text):
        self.calls.append(('structured',instruction,text))
        if instruction.startswith('Interpret'):
            if self.bad_query:return {'subject':'invented_subject','relations':['located_in']}
            return self.parse(text)
        edit=self.edits[text]
        return {'edits':[{k:v for k,v in asdict(edit).items() if k!='source_text'}]}

    def chat(self,messages,max_tokens=256):
        payload=json.loads(messages[1]['content']);self.calls.append(('chat',payload))
        query=self.parse(payload['question']);active={}
        for row in payload['records']:
            if 'subject' not in row:row={**asdict(self.edits[row['source_text']]),**row}
            key=(row['subject'],row['relation'])
            if row['operation']=='set':active[key]=row
            elif row['operation']=='delete':active.pop(key,None)
        current=query['subject'];ids=[]
        for relation in query['relations']:
            fact=active.get((current,relation))
            if fact is None:current=None;break
            ids.append(fact['fact_id']);current=fact['object']
        return {'text':json.dumps({'answer':current,'support_fact_ids':ids}),'model':'fixture-v2-only','usage':{'prompt_tokens':123}}

    def score(self,question,options):
        if self.scoring_offline:raise ServiceUnavailable('fixture scoring outage')
        return {'answer':'A'}


def configuration(tmp_path,*extra):
    return parser().parse_args(['evaluate','--benchmark','v2','--question-mode','natural','--gold-evidence',
                               '--per-family','1','--out',str(tmp_path/'result.json'),*extra])


@pytest.mark.parametrize('mode',['oracle_structured','model_proposals'])
def test_natural_queries_are_interpreted_and_gold_is_separate(tmp_path,mode):
    args=configuration(tmp_path,'--input-mode',mode)
    service=FixtureService(diagnostics.generate('test',1,args.seed))
    result=compare(args,service)
    assert result['status']=='completed' and len(result['rows'])==48
    assert result['summary']['exact_memory']['accuracy']==1
    assert result['summary']['gold_evidence_diagnostic']['accuracy']==1
    assert all(r['query_interpretation_correct'] for r in result['rows'])
    assert all(r['diagnostic_only']==(r['arm']=='gold_evidence_diagnostic') for r in result['rows'])
    assert all(not r['failures']['edit_interpretation'] for r in result['rows'])
    interpretation_calls=[call for call in service.calls if call[0]=='structured' and call[1].startswith('Interpret')]
    assert len(interpretation_calls)==12 and all(not call[2].startswith('Follow') for call in interpretation_calls)
    if mode=='model_proposals':
        recent=[r for r in result['rows'] if r['arm']=='recent_context']
        assert all('subject' not in fact for row in recent for fact in row['provided_evidence'])


def test_wrong_model_path_is_not_replaced_with_oracle_path(tmp_path):
    args=configuration(tmp_path)
    service=FixtureService(diagnostics.generate('test',1,args.seed));service.bad_query=True
    result=compare(args,service)
    exact=[r for r in result['rows'] if r['arm']=='exact_memory']
    assert all(r['failures']['query_interpretation'] for r in exact)
    assert all(r['provided_evidence']==[] for r in exact)
    assert result['summary']['exact_memory']['accuracy']<1
    assert result['summary']['gold_evidence_diagnostic']['accuracy']==1
    gold=[r for r in result['rows'] if r['arm']=='gold_evidence_diagnostic']
    assert not any(r['failures']['query_interpretation'] for r in gold)


def test_natural_query_scoring_outage_retains_trace_and_generated_answer(tmp_path):
    args=configuration(tmp_path,'--score')
    service=FixtureService(diagnostics.generate('test',1,args.seed));service.scoring_offline=True
    result=compare(args,service)
    assert result['status']=='pending_service' and len(result['rows'])==1
    row=result['rows'][0]
    assert row['valid_response'] and row['multiple_choice']['status']=='pending'
    assert row['question_mode']=='natural' and row['query_interpretation_correct']
    assert 'episode_manifest' in result
    assert json.loads(args.out.read_text())==result


def test_request_deadline_caps_timeout_and_restores_service_setting():
    from energy_real.reasoning import BudgetedService,ReasoningBudgetExpired
    import time
    class Timed:
        timeout=60.
        def health(self):return self.timeout
    raw=Timed();bounded=BudgetedService(raw,time.perf_counter()+.2)
    assert 0<bounded.health()<=.2 and raw.timeout==60.
    bounded.deadline=time.perf_counter()-1
    with pytest.raises(ReasoningBudgetExpired):bounded.health()
