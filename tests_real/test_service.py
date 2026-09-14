from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import re
import threading
import pytest
from energy_real.service import LocalModel,ServiceUnavailable
from energy_real.episodic import FactStore,FactEdit
from energy_real.reasoning import compare,parser


@pytest.fixture
def server():
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,obj):
            raw=json.dumps(obj).encode();self.send_response(200)
            self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            assert self.path=='/health'
            self.send({'model':'fixture-only','decode_mode':'bypass','adapters':{'passes':3}})
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path,payload))
            if self.path=='/v1/score':
                self.send({'answer':'A','used_loop':True,'probs':[1.,0.]});return
            assert self.path=='/v1/chat/completions'
            instruction=payload['messages'][0]['content'];text=payload['messages'][1]['content']
            if instruction.startswith('Propose'):
                out={'edits':[{'operation':'set','subject':'parcel','relation':'located_in','object':'locker'}]}
            else:
                query=json.loads(text);facts={}
                for row in query['records']:
                    key=(row['subject'],row['relation'])
                    if row['operation']=='set':facts[key]=row
                    elif row['operation']=='delete':facts.pop(key,None)
                match=re.fullmatch(r'Follow ([\w/]+) from ([\w-]+)\.',query['question'])
                current=match[2];ids=[]
                for relation in match[1].split('/'):
                    fact=facts.get((current,relation))
                    if fact is None:current=None;break
                    ids.append(fact['fact_id']);current=fact['object']
                out={'answer':current,'support_fact_ids':ids}
            self.send({'model':'fixture-only','choices':[{'message':{'content':json.dumps(out)}}]})
    httpd=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    yield LocalModel(f'http://127.0.0.1:{httpd.server_port}',timeout=2),seen
    httpd.shutdown();thread.join();httpd.server_close()


def test_http_contracts_and_proposals_do_not_write(server,tmp_path):
    service,seen=server
    assert service.health()['model']=='fixture-only'
    store=FactStore(tmp_path/'facts.sqlite',service=service)
    proposal=store.propose('The parcel is in the locker.')
    assert proposal['status']=='needs_review' and store.revision==0 and not store.snapshot()[1]
    edits=[FactEdit(**e) for e in proposal['edits']]
    with pytest.raises(PermissionError):store.apply(edits,expected_revision=0,request_id='r')
    store.apply(edits,expected_revision=0,request_id='r',authorized=True)
    assert service.score('Question',['yes','no'])['answer']=='A'
    chat_payload=next(p for route,p in seen if route=='/v1/chat/completions')
    assert chat_payload['temperature']==0 and chat_payload['max_tokens']==256


def test_three_arm_comparison_uses_only_supplied_evidence(server,tmp_path):
    service,seen=server
    args=parser().parse_args(['evaluate','--out',str(tmp_path/'result.json'),'--per-family','1','--score'])
    result=compare(args,service)
    assert result['status']=='completed' and len(result['rows'])==30
    assert result['summary']['exact_memory']['accuracy']==1
    assert result['summary']['recent_context']['accuracy']<1
    assert all(row['context_tokens'] is None for row in result['rows'])
    for row in result['rows']:
        assert 'multiple_choice' in row
        selected=json.loads(row['raw_response'])['support_fact_ids']
        assert set(selected)<={f['fact_id'] for f in row['provided_evidence']}


def test_unavailable_service_is_pending_not_measured(tmp_path):
    class Offline:
        def health(self):raise ServiceUnavailable('fixture offline')
    args=parser().parse_args(['evaluate','--out',str(tmp_path/'pending.json')])
    result=compare(args,Offline())
    assert result['status']=='pending_service' and result['rows']==[]


def test_scoring_outage_preserves_generated_answer_and_metadata(server,tmp_path):
    service,_=server
    def fail_score(*args):raise ServiceUnavailable('scorer went offline')
    service.score=fail_score
    args=parser().parse_args(['evaluate','--out',str(tmp_path/'interrupted.json'),'--per-family','1','--score'])
    result=compare(args,service)
    assert result['status']=='pending_service' and len(result['rows'])==1
    row=result['rows'][0]
    assert row['valid_response'] and row['response_model']=='fixture-only'
    assert row['multiple_choice']['status']=='pending'
    assert result['summary'][row['arm']]['episodes']==1
    assert result['summary'][row['arm']]['scored_episodes']==0
    assert result['decoding']=={'temperature':0.,'top_p':1.,'max_tokens':256}
    assert result['seed']==args.seed and result['energy_kind']==args.energy_kind
    assert json.loads(args.out.read_text())==result


@pytest.mark.parametrize('flag,value',[('--recent-events','0'),('--per-family','0'),('--max-seconds','nan'),('--timeout','inf')])
def test_invalid_reasoning_limits_fail_before_service(tmp_path,flag,value):
    class Unused:
        def health(self):raise AssertionError('Invalid arguments reached service')
    args=parser().parse_args(['evaluate','--out',str(tmp_path/'invalid.json'),flag,value])
    with pytest.raises(ValueError):compare(args,Unused())
    assert not args.out.exists()
