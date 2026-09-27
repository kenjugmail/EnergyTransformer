from pathlib import Path
import copy,hashlib,json,subprocess,sys
import pytest
import torch
from energy_forecast.refine_model import *
from energy_forecast.refine_session import RefinementSession
from energy_forecast.refine_train import fit
from energy_forecast.probe import load_energy_model

ROOT=Path(__file__).resolve().parents[1]
TEACHER=ROOT/'checkpoints/teacher_973k.pt'
HEAD=ROOT/'results/refinement/fits/energy_memory_seed11/best.pt'

@pytest.fixture
def session():
    if not TEACHER.exists() or not HEAD.exists():pytest.skip('Bundled teacher/experiment checkpoint required')
    torch.set_num_threads(1);s=RefinementSession(TEACHER,HEAD,steps=4)
    yield s;s.close()


def test_teacher_memory_unchanged_under_identical_bytes(session):
    torch.manual_seed(912);x=torch.randint(256,(1,32));session.observe(x)
    base=load_energy_model(TEACHER);state=None
    with torch.no_grad():
        for i in range(x.shape[1]):_,state=base(x[:,i:i+1],state)
    for a,b in zip(session.state,state):
        for p,q in zip(a,b):assert torch.equal(p,q)
    payload=sum(t.untyped_storage().nbytes() for layer in session.state for t in layer)
    assert payload==41984


def test_future_inputs_do_not_change_prefix(session):
    torch.manual_seed(93);x=torch.randint(256,(1,31));y=x.clone();y[:,23:]=(y[:,23:]+17)%256
    a=session.observe(x);other=RefinementSession(TEACHER,HEAD,steps=4)
    try:b=other.observe(y)
    finally:other.close()
    assert torch.equal(a[:,:23],b[:,:23])


def test_invalid_input_does_not_partially_advance(session):
    session.observe(torch.tensor([[1,2,3]]));before=copy.deepcopy(session.state);n=session.steps
    with pytest.raises(ValueError):session.observe(torch.tensor([[4,500]]))
    assert session.steps==n
    assert all(torch.equal(a,b) for x,y in zip(before,session.state) for a,b in zip(x,y))


def test_fresh_process_restore_exact(session,tmp_path):
    session.observe(torch.tensor([list(b'The energy memory has context.')]))
    session.save(tmp_path/'state.pt');expected=session.observe(torch.tensor([list(b' New bytes.')]))
    code=f"""import torch
from energy_forecast.refine_session import RefinementSession
torch.set_num_threads(1)
s=RefinementSession({str(TEACHER)!r},{str(HEAD)!r},steps=4)
s.restore({str(tmp_path/'state.pt')!r})
x=s.observe(torch.tensor([list(b' New bytes.')]))
torch.save({{'logits':x,'state':s.state}}, {str(tmp_path/'out.pt')!r})
"""
    subprocess.run([sys.executable,'-c',code],check=True,cwd=ROOT)
    actual=torch.load(tmp_path/'out.pt',weights_only=True)
    assert torch.equal(expected,actual['logits'])
    assert all(torch.equal(a,b) for x,y in zip(session.state,actual['state']) for a,b in zip(x,y))


def test_wrong_policy_restore_rejected_without_mutation(session,tmp_path):
    session.observe(torch.tensor([[1,2,3]]));session.save(tmp_path/'state.pt')
    data=torch.load(tmp_path/'state.pt',weights_only=True);data['depth']=8;torch.save(data,tmp_path/'bad.pt')
    with pytest.raises(ValueError):session.restore(tmp_path/'bad.pt')
    assert session.steps==3


def test_bad_precision_snapshot_rejected(session,tmp_path):
    session.observe(torch.tensor([[1,2,3]]));session.save(tmp_path/'state.pt')
    d=torch.load(tmp_path/'state.pt',weights_only=True);d['state'][0][0].fill_(-1);torch.save(d,tmp_path/'bad.pt')
    with pytest.raises(ValueError):session.restore(tmp_path/'bad.pt')
    assert session.steps==3


def test_batch_adapter_matches_single_stream(session):
    x=torch.tensor([list(b'A causal frozen energy test.')]);session.observe(x)
    h={};teacher=load_energy_model(TEACHER);hook=teacher.norm.register_forward_hook(lambda m,a,o:h.update(x=o.detach()))
    state=None
    with torch.no_grad():
        for t in range(x.shape[1]):_,state=teacher(x[:,t:t+1],state)
        memory=compact_energy_read(state)
        a=session.adapter(h['x'][:,0],memory,teacher.head.weight)
    # Reissue from the same prefix in an independent session for exact serial order.
    other=RefinementSession(TEACHER,HEAD,steps=4)
    try:b=other.observe(x)[:,-1]
    finally:other.close();hook.remove()
    assert torch.equal(a,b)


def small_data(path):
    torch.manual_seed(927)
    def part(n):return {'h':torch.randn(n,8),'m':torch.randn(n,8),'y':torch.randint(9,(n,))}
    data={'train':part(30),'validation':part(12),'weight':torch.randn(9,8),
        'metadata':{'width':8,'memory_width':8,'memory_center':torch.zeros(8),'memory_scale':torch.ones(8),'teacher_sha256':'fixture'}}
    torch.save(data,path);return data


def test_fresh_process_training_resume(tmp_path):
    data_path=tmp_path/'data.pt';d=small_data(data_path)
    a=fit(d,data_path,tmp_path/'complete','energy_memory',19,steps=8,batch=4)
    fit(d,data_path,tmp_path/'resume','energy_memory',19,steps=8,batch=4,stop_after=3)
    code=f"""import torch
from energy_forecast.refine_train import fit
d=torch.load({str(data_path)!r},weights_only=True)
fit(d,{str(data_path)!r},{str(tmp_path/'resume')!r},'energy_memory',19,steps=8,batch=4,resume=True)
"""
    subprocess.run([sys.executable,'-c',code],check=True,cwd=ROOT)
    left=torch.load(tmp_path/'complete/last.pt',weights_only=True);right=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    assert left['sample_hash']==right['sample_hash'] and left['history']==right['history']
    for k in left['weights']:assert torch.equal(left['weights'][k],right['weights'][k])
    assert torch.equal(left['rng'],right['rng']) and torch.equal(left['generator'],right['generator'])
    for key,state in left['optimizer']['state'].items():
        for name,val in state.items():
            if isinstance(val,torch.Tensor):assert torch.equal(val,right['optimizer']['state'][key][name])
            else:assert val==right['optimizer']['state'][key][name]


def test_zero_step_checkpoint_and_backwards_resume(tmp_path):
    p=tmp_path/'data.pt';d=small_data(p)
    fit(d,p,tmp_path/'run','energy_memory',19,steps=8,batch=4,stop_after=0)
    assert (tmp_path/'run/last.pt').exists()
    fit(d,p,tmp_path/'run','energy_memory',19,steps=8,batch=4,resume=True,stop_after=3)
    previous=(tmp_path/'run/last.pt').read_bytes()
    with pytest.raises(ValueError,match='Backwards'):fit(d,p,tmp_path/'run','energy_memory',19,steps=8,batch=4,resume=True,stop_after=2)
    assert previous==(tmp_path/'run/last.pt').read_bytes()
