import json,subprocess,sys
from pathlib import Path
import pytest
import torch
from energy_forecast.probe import load_energy_model,trace_states
from energy_forecast.session import ForecastSession
ROOT=Path(__file__).resolve().parents[1]
TEACHER=ROOT/'checkpoints/teacher_973k.pt'
HEAD=ROOT/'results/native_runs/spectral_seed11/best.pt'

@pytest.fixture
def session():
    if not TEACHER.exists() or not HEAD.exists():pytest.skip('Standalone artifacts not present')
    torch.set_num_threads(1);return ForecastSession(TEACHER,HEAD)

def test_current_kernel_checkpoint_compatibility(session):
    assert session.model.parameters_count()==972896
    assert session.model.config.kind=='energy'

def test_sidecar_does_not_change_language_outputs(session):
    ids=torch.tensor([list(b'The forecast does not replace actual bytes.')])
    m=load_energy_model(TEACHER);s=None;outs=[]
    with torch.no_grad():
        for i in range(ids.shape[1]):
            o,s=m(ids[:,i:i+1],s);outs.append(o)
    actual=session.observe(ids);assert torch.equal(actual,torch.cat(outs,1))
    before=[(p.clone(),ss.clone()) for p,ss in session.state]
    f=session.forecast();f.refine(f.mean,steps=2)
    assert all(torch.equal(p,a) and torch.equal(ss,b) for (p,ss),(a,b) in zip(session.state,before))

def test_real_teacher_future_perturbation_causality(session):
    torch.manual_seed(182);a=torch.randint(256,(1,35));b=a.clone();b[:,20:]=torch.randint(256,(1,15))
    left=trace_states(session.model,a);right=trace_states(session.model,b)
    assert torch.equal(left[:,:20],right[:,:20])

def test_fresh_process_forecast_and_language_restart(session,tmp_path):
    prefix=torch.tensor([list(b'The evolving energy function predicts ')]);continuation=torch.tensor([list(b'future')])
    session.observe(prefix);snapshot=tmp_path/'session.pt';session.save(snapshot)
    a=session.observe(continuation);energy=session.forecast().energy(torch.zeros(1,32));dest=tmp_path/'answer.pt'
    code=f"from energy_forecast.session import ForecastSession;import torch;torch.set_num_threads(1);s=ForecastSession({str(TEACHER)!r},{str(HEAD)!r});s.restore({str(snapshot)!r});a=s.observe(torch.tensor({continuation.tolist()!r}));e=s.forecast().energy(torch.zeros(1,32));torch.save((a,e),{str(dest)!r})"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    b,other=torch.load(dest,weights_only=True)
    assert torch.equal(a,b) and torch.equal(energy,other)

def test_wrong_teacher_head_rejected(tmp_path):
    if not HEAD.exists():pytest.skip('No head artifacts')
    checkpoint=torch.load(TEACHER,weights_only=True);checkpoint['selected_step']=-100
    file=tmp_path/'wrong.pt';torch.save(checkpoint,file)
    with pytest.raises(ValueError):ForecastSession(file,HEAD)
