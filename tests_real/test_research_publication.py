import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import pytest
import torch
from energy_real import diagnostics, recall_study, stages
from energy_real.artifacts import export_bundle, resolve_artifact
from energy_real.episodic import EnergyRetriever
from energy_real.retrieval import rank_candidates, accepts
from energy_real.reasoning import compare
from energy_real.reporting import build
from energy_real.runtime import atomic_json, sha256
from energy_real.service import ServiceUnavailable
from test_reasoning_v2 import FixtureService, configuration


def selection_fixture(tmp_path):
    model=recall_study.new_model('precision',16,.001,11)
    checkpoint=tmp_path/'weights.pt'
    recall_study.save_controller(model,checkpoint,11,0)
    atomic_json({'version':diagnostics.VERSION},tmp_path/'datasets.json')
    atomic_json({'status':'frozen_for_test','dataset_sha256':sha256(tmp_path/'datasets.json'),
                 'selected':[{'kind':'precision','seed':11,'dimension':16,'prior':.001,'capacity':256,
                              'controller':str(checkpoint),'initial_controller':str(checkpoint),
                              'controller_sha256':sha256(checkpoint),'initial_sha256':sha256(checkpoint),
                              'validation':{'selected':{'score_threshold':.15,'margin_threshold':0.}}}]},tmp_path/'selection.json')
    return tmp_path/'selection.json'


def test_bundle_relocation_and_external_artifact_root(tmp_path):
    original=selection_fixture(tmp_path)
    frozen=original.read_bytes()
    exported=export_bundle(original,tmp_path/'bundle')
    relocated=tmp_path/'moved'
    shutil.move(exported.parent,relocated)
    (tmp_path/'weights.pt').unlink()
    config=recall_study.selected_memory(relocated/'selection.json')
    assert config['controller'].is_relative_to(relocated)
    assert original.read_bytes()==frozen
    metadata=tmp_path/'metadata';metadata.mkdir()
    for name in ('selection.json','datasets.json'):
        shutil.copyfile(relocated/name,metadata/name)
    assert recall_study.selected_memory(metadata/'selection.json',artifact_root=relocated)==config
    config['controller'].write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        recall_study.selected_memory(relocated/'selection.json')
    config['controller'].unlink()
    with pytest.raises(FileNotFoundError,match='artifact-root'):
        recall_study.selected_memory(relocated/'selection.json')
    with pytest.raises(ValueError,match='escapes'):
        resolve_artifact('../outside.pt','hash',metadata/'selection.json',metadata)


def test_ranking_ties_and_practical_study_parity():
    episodes=diagnostics.generate('test',1)
    for ep in episodes:
        model=recall_study.new_model('precision',16,.01,11)
        readout=recall_study.Readout(ep,model)
        retriever=EnergyRetriever(readout.events)
        subject,relation=ep.query.subject,ep.query.relations[0]
        expected=readout.candidates(subject,relation)
        retriever.read(subject,relation)
        got=retriever.last_diagnostic
        assert got['score']==expected['score'] and got['margin']==expected['margin']
        assert got['candidate_fact_ids']==[e['fact_id'] for e in expected['candidates']]
    row=rank_candidates(torch.tensor([1.,1.,1.]),{2:{'id':2},0:{'id':0},1:{'id':1}})
    assert row['candidates']==[{'id':0},{'id':1}] and row['margin']==0
    assert accepts(row,.5,0) and not accepts(row,.5,.01)
    assert rank_candidates(torch.tensor([float('nan')]),{0:{}})['score'] is None


@pytest.mark.parametrize('failure',['generation','scoring'])
def test_reasoning_resume_reuses_successful_calls_and_rows(tmp_path,failure):
    args=configuration(tmp_path,'--input-mode','model_proposals','--score')
    service=FixtureService(diagnostics.generate('test',1,args.seed))
    if failure=='scoring':service.scoring_offline=True
    else:
        normal=service.chat
        count=[0]
        def interrupted(*a,**kw):
            count[0]+=1
            if count[0]==3:raise ServiceUnavailable('interrupted generation')
            return normal(*a,**kw)
        service.chat=interrupted
    first=compare(args,service)
    assert first['status']=='pending_service' and first['rows']
    old_rows={ (r['episode_id'],r['arm']):r for r in first['rows'] }
    old_calls=list(service.calls)
    args.resume=True;service.scoring_offline=False
    if failure=='generation':service.chat=normal
    done=compare(args,service)
    assert done['status']=='completed' and len(done['rows'])==48
    assert len({(r['episode_id'],r['arm']) for r in done['rows']})==48
    assert done['elapsed_s']>=first['elapsed_s']>0
    for row in done['rows']:
        old=old_rows.get((row['episode_id'],row['arm']))
        if old:
            assert {k:v for k,v in row.items() if k!='multiple_choice'}=={k:v for k,v in old.items() if k!='multiple_choice'}
    # Every successful interpretation or generation is executed only once per scope.
    new=service.calls[len(old_calls):]
    assert all(call not in new for call in old_calls if call[0]=='structured')
    before=len(service.calls)
    again=compare(args,service)
    assert len(service.calls)==before and again['rows']==done['rows']


@pytest.mark.parametrize('changed',['seed','score','source','model','budget','dataset','checkpoint'])
def test_resume_rejects_changed_identity_without_overwriting(tmp_path,changed,monkeypatch):
    args=configuration(tmp_path,'--score')
    service=FixtureService(diagnostics.generate('test',1,args.seed));service.scoring_offline=True
    compare(args,service);args.resume=True
    if changed=='seed':args.seed+=1
    elif changed=='score':args.score=False
    elif changed=='budget':args.max_seconds+=1
    elif changed=='source':monkeypatch.setattr('energy_real.reasoning.source_identity',lambda:'changed')
    elif changed=='model':service.health=lambda:{'model':'different'}
    else:
        value=json.loads(args.out.read_text())
        if changed=='dataset':value['episode_manifest']['sha256']='changed'
        else:value['controller_sha256']='changed'
        atomic_json(value,args.out)
    before=args.out.read_bytes()
    with pytest.raises(ValueError,match='identity mismatch'):compare(args,service)
    assert args.out.read_bytes()==before


def test_resume_preserves_budget_when_health_remains_unavailable(tmp_path):
    args=configuration(tmp_path)
    service=FixtureService([])
    def unavailable():raise ServiceUnavailable('offline')
    service.health=unavailable
    first=compare(args,service);args.resume=True
    second=compare(args,service)
    assert second['elapsed_s']>=first['elapsed_s']>0 and second['status']=='pending_service'


def test_report_uses_explicit_evidence_and_preserves_inputs(tmp_path):
    study=tmp_path/'study';study.mkdir()
    atomic_json({'status':'budget_exhausted','source_sha256':'fixture','records':[],
                 'test_results':[],'elapsed_s':1.,'gpu_seconds':0},study/'summary.json')
    ledger=tmp_path/'ledger.json';atomic_json({'stages':{}},ledger)
    log=tmp_path/'suite.log';log.write_text('2 passed, 1 skipped in 1.0s\n')
    original=(study/'summary.json').read_bytes()
    result=build(study,ledger,tmp_path/'report',{'fixture':log})
    assert result['status']=='budget_exhausted' and result['tests']['fixture']['passed']==2
    assert result['verification_status']=='passed' and (study/'summary.json').read_bytes()==original
    with pytest.raises(FileExistsError):build(study,ledger,tmp_path/'report',{'fixture':log})


def test_stage_service_retry_preserves_completed_commands(tmp_path,monkeypatch):
    from energy_real import service
    class Healthy:
        def __init__(self,*a,**kw):pass
        def health(self):return {'model':'fixture'}
    monkeypatch.setattr(service,'LocalModel',Healthy)
    monkeypatch.setattr(stages,'source_identity',lambda:'fixture')
    monkeypatch.setattr(stages,'gpu_processes',lambda:[])
    script="""import json,pathlib,sys
p=pathlib.Path(sys.argv[sys.argv.index('--out')+1])
resumed='--resume' in sys.argv
status='completed' if p.name=='first.json' or resumed else 'pending_service'
p.write_text(json.dumps({'status':status,'max_seconds':60,'resumed':resumed}))
"""
    monkeypatch.setattr(stages,'commands',lambda name,out,remaining:[
        [sys.executable,'-c',script,'--out',str(out/f'{name}.json'),'--max-seconds','60'] for name in ('first','second')])
    a=SimpleNamespace(out=tmp_path/'stages',stages=['reasoning'],wait_idle=False,max_wait_seconds=0,prior_ledger=[])
    first=stages.run(a)['stages']['reasoning']
    assert first['status']=='pending_service' and first['completed_commands']==[0]
    artifact=a.out/'reasoning/attempt_0/first.json';before=artifact.stat().st_mtime_ns
    second=stages.run(a)['stages']['reasoning']
    assert second['status']=='completed' and second['completed_commands']==[0,1]
    assert second['used_s']>=first['used_s'] and artifact.stat().st_mtime_ns==before
    assert json.loads((artifact.parent/'second.json').read_text())['resumed']
