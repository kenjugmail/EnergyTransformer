import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from energy_campaign.budget import Ledger
from energy_campaign.spec import config,choose_budget,FAMILIES
from energy_campaign.data import Documents,prepare
from energy_campaign.acceptance import kernel_audit,model_audit
from energy_real.model import Config,LM
from energy_real.summaries import energy_layer,forward_with_summaries
from energy_forecast.probe import trace_states,load_energy_model
from energy_campaign.joint import JointForecastLM,frozen_targets


def tiny():return LM(Config(kind='hybrid',width=16,layers=3,heads=2,slots=4,ff_width=32,chunk=8,attention_window=8))


def test_ledger_crash_counts_reservation_and_no_duplicate(tmp_path):
    ledger=Ledger(tmp_path/'ledger.json',{'a':100})
    key=ledger.reserve('a',80,{'job':1});assert ledger.remaining('a')==20
    with pytest.raises(RuntimeError):Ledger(ledger.path,{'a':100}).reserve('a',10,{})
    ledger.settle(key,30,'failed');assert ledger.remaining('a')==70
    with pytest.raises(ValueError):ledger.settle(key,1,'completed')
    with pytest.raises(RuntimeError):ledger.reserve('a',71,{})


def test_parameter_match_and_selection():
    with torch.device('meta'):
        for size in range(4):
            counts=[LM(config(f,size)).parameters_count() for f in FAMILIES]
            assert (max(counts)-min(counts))/min(counts)<.005
    times={(f,i):.0001 for f in FAMILIES for i in range(4)}
    overhead={k:1 for k in times};result=choose_budget(times,overhead)
    assert result['sizes']==list(range(4)) and result['estimated_seconds']*1.2<=32400


def test_hybrid_trace_causality_gradients_and_loader(tmp_path):
    torch.manual_seed(1);model=tiny();x=torch.randint(256,(2,19))
    assert energy_layer(model.config)==1
    y,s,z=forward_with_summaries(model,x)
    torch.testing.assert_close(y,model(x)[0],atol=2e-5,rtol=2e-5)
    x2=x.clone();x2[:,12:]=0
    torch.testing.assert_close(z[:,:12],trace_states(model,x2)[:,:12],rtol=0,atol=0)
    z.square().mean().backward();assert model.blocks[1].mixer.projection.weight.grad.abs().sum()>0
    from dataclasses import asdict
    path=tmp_path/'model.pt';torch.save({'config':asdict(model.config),'weights':model.state_dict()},path)
    assert load_energy_model(path).config.kind=='hybrid'


def test_forecast_zero_residual_and_targets_do_not_leak():
    torch.manual_seed(2);model=tiny();joint=JointForecastLM(model,torch.zeros(4),torch.ones(4))
    x=torch.randint(256,(1,81));teacher=copy.deepcopy(model).eval()
    target=frozen_targets(teacher,x,joint.center,joint.scale)
    y,_,aux=joint(x,future_targets=target)
    expected=forward_with_summaries(model,x)[0]
    torch.testing.assert_close(y,expected,rtol=0,atol=0)
    target2=target.clone();target2[torch.isfinite(target2)]+=20
    y2,_,aux2=joint(x,future_targets=target2)
    torch.testing.assert_close(y,y2,rtol=0,atol=0);assert aux!=aux2
    aux.backward();assert joint.forecast.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())


def test_public_data_no_cross_document_and_coverage_resume(tmp_path):
    rng=np.random.default_rng(3)
    records=[{'id':str(i),'text':' '.join(str(x) for x in rng.integers(0,1000000,300))} for i in range(100)]
    records.append(records[0]);out=tmp_path/'data';m=prepare(records,out,limit=300000)
    assert sum(len(v) for v in m['documents'].values())==100
    groups=[{d['group'] for d in docs} for docs in m['documents'].values()]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    d=Documents(out);x,y=d.batch(rng,4,512);assert torch.equal(x[:,1:],y[:,:-1]);assert d.unique_targets>0
    saved=json.loads(json.dumps(d.coverage));fresh=Documents(out);fresh.restore_coverage(saved);assert d.unique_targets==fresh.unique_targets
    with pytest.raises(FileExistsError):prepare(records,out)


def test_current_native_acceptance_cpu():
    assert len(kernel_audit('cpu'))==8
    assert len(model_audit('cpu'))==3


def test_substage_budget_cannot_borrow(tmp_path):
    ledger=Ledger(tmp_path/'l.json')
    with pytest.raises(RuntimeError):ledger.reserve('backbone',3601,{'job':{'phase':'calibration'}})


def test_stale_cpu_acceptance_rejected(tmp_path):
    from energy_campaign.runner import validate_acceptance
    from energy_campaign.spec import source_identity
    (tmp_path/'accept.json').write_text(json.dumps({'status':'CPU_DIAGNOSTIC_ONLY','source_sha256':source_identity(),'runtime':{'torch':str(torch.__version__)}}))
    with pytest.raises(ValueError):validate_acceptance(tmp_path,{'action':'train'})


def test_fourier_rotation_can_preserve_function():
    from energy_forecast.model import ForecastEnergy,ForecastConfig
    torch.manual_seed(6);plain=ForecastEnergy(ForecastConfig(variables=4));spectral=ForecastEnergy(ForecastConfig(variables=4,spectral=True))
    state=plain.state_dict();state.pop('time_basis');spectral.load_state_dict(state,strict=False)
    with torch.no_grad():
        w=plain.encoder[0].weight.reshape(96,16,4)
        spectral.encoder[0].weight.copy_(torch.einsum('otv,kt->okv',w,spectral.time_basis).reshape(96,64))
    x=torch.randn(2,16,4)
    torch.testing.assert_close(plain(x).mean,spectral(x).mean,atol=1e-5,rtol=1e-5)


def test_scaling_refuses_checkpoint_pseudoreplication():
    from energy_campaign.analysis import fit_scaling,paired_interval
    assert fit_scaling([])['status']=='unresolved'
    with pytest.raises(ValueError):fit_scaling([{'metric':'nll','independent_run':False}])
    interval=paired_interval([{'seed':s,'group':g,'difference':-.1} for s in range(3) for g in range(4)],100)
    assert interval['high']<0


def test_busy_device_does_not_start_or_charge(tmp_path,monkeypatch):
    from energy_campaign import runner
    monkeypatch.setattr(runner,'doctor',lambda:{'other_compute_processes':[{'pid':123,'name':'foreign'}],'available_host_bytes':100*2**30,'status':'WAITING_FOR_EXCLUSIVE_GPU'})
    result=runner.execute(tmp_path,'acceptance',60,{'action':'accept','out':str(tmp_path/'accept.json')})
    assert result['status']=='WAITING_FOR_EXCLUSIVE_GPU'
    assert not (tmp_path/'budget.json').exists()


def test_budget_supervisor_charges_failed_child(tmp_path,monkeypatch):
    from energy_campaign import runner
    monkeypatch.setattr(runner,'doctor',lambda:{'other_compute_processes':[],'available_host_bytes':100*2**30})
    class Child:
        pid=999999
        def wait(self,timeout=None):return 1
        def poll(self):return 1
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:Child())
    result=runner.execute(tmp_path,'acceptance',60,{'action':'accept','out':str(tmp_path/'accept.json')})
    assert result['status']=='failed'
    entry=json.loads((tmp_path/'budget.json').read_text())['entries'][0]
    assert entry['status']=='failed' and entry['charged']>0


def test_recall_answers_are_not_in_query_position():
    from energy_campaign.recall import batch
    x,y=batch(np.random.default_rng(4),8,128,'cpu')
    assert (x[:,-2]==203).all() and (x[:,-1]<17).all() and (y>=32).all()
    for row,target in zip(x,y):
        pairs={int(row[i+1]):int(row[i+2]) for i in range(0,12,3)}
        assert pairs[int(row[-1])]==int(target)
