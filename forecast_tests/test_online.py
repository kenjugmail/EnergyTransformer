from pathlib import Path
import json,os,subprocess,sys
import pytest
import torch
from energy_forecast.conditioning import blend,GaussianMixture
from energy_forecast.online import ObservedForecastSession,_payload_digest
from energy_forecast.session import ForecastSession
ROOT=Path(__file__).resolve().parents[1]
TEACHER=ROOT/'checkpoints/teacher_973k.pt'
HEAD=ROOT/'results/repair/fits/anchored_seed11.pt'
GATE=ROOT/'results/online_gate/gate.json'


def setup_function():torch.set_num_threads(1)


def session(gated=True):
    if not TEACHER.exists() or not HEAD.exists() or (gated and not GATE.exists()):pytest.skip('Standalone model/evaluation artifacts required')
    return ObservedForecastSession(TEACHER,HEAD,gate_path=GATE if gated else None)


def tokens(n):return torch.tensor([list((b'Genuine arriving observations update this forecast. ' * 3)[:n])],dtype=torch.long)


def test_online_only_matured_observations_and_absolute_horizons():
    s=session();s.observe(tokens(20));s.issue();s.observe(tokens(2))
    assert s.status()['observed_horizons']==[1]
    assert s.status()['pending'][0]=={'origin_horizon':4,'absolute_target_step':24,'bytes_remaining':2}
    s.observe(tokens(14));assert s.status()['observed_horizons']==[1,4,16]
    assert s.status()['pending']==[{'origin_horizon':64,'absolute_target_step':84,'bytes_remaining':48}]
    # Validation rejects the far-only correction; exact fallback.
    assert torch.equal(s.distribution().mean,s.distribution(policy='original').mean)
    assert not hasattr(s,'add_observation')


def test_no_altered_teacher_logits():
    online=session();base=ForecastSession(TEACHER,HEAD);x=tokens(96)
    assert torch.equal(online.observe(x[:,:20]),base.observe(x[:,:20]));online.issue()
    for a,b in [(20,21),(21,24),(24,36),(36,84)]:
        assert torch.equal(online.observe(x[:,a:b]),base.observe(x[:,a:b]))
        if online.status()['status']=='active':online.distribution();online.distribution(policy='exact_conditional')
    assert online.status()['status']=='complete'
    with pytest.raises(ValueError):online.distribution()


def test_future_tokens_do_not_change_current_forecast():
    a=session();b=session();x=tokens(26)
    a.observe(x[:,:20]);b.observe(x[:,:20]);a.issue();b.issue()
    a.observe(x[:,20:24]);b.observe(x[:,20:24]);before=a.distribution().payload()
    for k in before:assert torch.equal(before[k],b.distribution().payload()[k])
    b.observe(torch.tensor([[0,1]],dtype=torch.long))
    for k in before:assert torch.equal(before[k],a.distribution().payload()[k])


def test_tracked_memory_bounded_and_return_values_are_defensive():
    s=session();s.observe(tokens(20));s.issue();size=s.raw_state_bytes();assert size['frozen_forecast']==25368
    f=s.distribution();expected=s.distribution().mean.clone();f.means.zero_()
    assert torch.equal(s.distribution().mean,expected)
    with pytest.raises(ValueError):s.issue()
    s.issue(replace=True);s.observe(tokens(130));size=s.raw_state_bytes()
    assert size['observed_evidence']==256 and size['history']==512
    s.clear();assert s.raw_state_bytes()['frozen_forecast']==0


def test_bad_late_token_rejected_before_any_update():
    s=session();s.observe(tokens(20));steps=s.steps
    with pytest.raises(ValueError):s.observe(torch.tensor([[32,97,999]],dtype=torch.long))
    assert s.steps==steps
    base=ForecastSession(TEACHER,HEAD);base.observe(tokens(20))
    with pytest.raises(ValueError):base.observe(torch.tensor([[32,97,999]],dtype=torch.long))
    assert base.steps==20


def test_no_gate_defaults_to_prior():
    s=session(gated=False);s.observe(tokens(20));s.issue();s.observe(tokens(1))
    assert s.status()['gate_weight']==0
    assert torch.equal(s.distribution().mean,s.distribution(policy='original').mean)


def test_blended_density_is_normalized_pool():
    torch.manual_seed(4)
    mu=torch.randn(2,3,4,dtype=torch.float64);chol=torch.eye(4,dtype=torch.float64)[None,None].expand(2,3,-1,-1)
    a=GaussianMixture(torch.randn(2,3,dtype=torch.float64),mu,chol)
    b=GaussianMixture(a.logits,mu+2,chol*.7);mix=blend(a,b,.3);y=torch.randn(2,4,dtype=torch.float64)
    expected=-torch.logaddexp(-a.energy(y)+torch.tensor(.7).double().log(),-b.energy(y)+torch.tensor(.3).double().log())
    torch.testing.assert_close(mix.energy(y),expected,atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(mix.mean,.7*a.mean+.3*b.mean)


def test_restore_consistency_rejects_before_mutation(tmp_path):
    s=session();s.observe(tokens(20));s.issue();s.observe(tokens(4));p=tmp_path/'good.pt';s.save(p)
    saved=torch.load(p,weights_only=True);saved['payload']['ticket']['observations']['16']=torch.zeros(1,8,dtype=torch.float64)
    saved['sha256']=_payload_digest(saved['payload']);bad=tmp_path/'bad.pt';torch.save(saved,bad)
    previous=s.status()
    with pytest.raises(ValueError):s.restore(bad)
    assert s.status()==previous
    saved=torch.load(p,weights_only=True);saved['payload']['base']['history'][:,-1]+=1
    saved['sha256']=_payload_digest(saved['payload']);torch.save(saved,bad)
    with pytest.raises(ValueError):s.restore(bad)
    assert s.status()==previous


def test_fresh_process_matches_online_continuation(tmp_path):
    s=session();s.observe(tokens(20));s.issue();s.observe(tokens(4));path=tmp_path/'state.pt';s.save(path)
    continuation=tokens(12);logits=s.observe(continuation);forecast=s.distribution().payload();out=tmp_path/'child.pt'
    code=f'''import torch
from energy_forecast.online import ObservedForecastSession
torch.set_num_threads(1)
s=ObservedForecastSession({str(TEACHER)!r},{str(HEAD)!r},gate_path={str(GATE)!r})
s.restore({str(path)!r})
logits=s.observe(torch.tensor({continuation.tolist()!r},dtype=torch.long))
torch.save({{'logits':logits,'forecast':s.distribution().payload(),'status':s.status()}},{str(out)!r})
'''
    env=dict(os.environ,PYTHONPATH=str(ROOT/'reference')+os.pathsep+str(ROOT))
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True)
    result=torch.load(out,weights_only=True)
    assert torch.equal(logits,result['logits'])
    for k in forecast:assert torch.equal(forecast[k],result['forecast'][k])
    assert s.status()==result['status']


def test_history_owns_only_its_declared_storage():
    s=session();s.observe(tokens(96))
    h=s._base.history
    assert h.untyped_storage().nbytes()==h.numel()*h.element_size()==512


def test_wrong_gate_identity_rejected(tmp_path):
    session()  # requires the parent artifacts
    gate=json.loads(GATE.read_text());gate['heads']={'other':'0'*64}
    p=tmp_path/'bad_gate.json';p.write_text(json.dumps(gate))
    with pytest.raises(ValueError):ObservedForecastSession(TEACHER,HEAD,gate_path=p)
