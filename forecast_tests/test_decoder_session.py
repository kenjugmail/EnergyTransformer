from pathlib import Path
import json,subprocess,sys
import pytest
import torch
from energy_forecast.decoder_bridge import DecoderBridgeSession
from energy_forecast.probe import load_energy_model
ROOT=Path(__file__).resolve().parents[1]

def paths(kind='hidden',seed=11):
    return ROOT/'checkpoints/teacher_973k.pt',ROOT/f'results/decoder_bridge/fits/{kind}_seed{seed}/best.pt',ROOT/f'results/repair/fits/anchored_seed{seed}.pt'

def session(kind='hidden',seed=11):
    p,a,h=paths(kind,seed)
    if not all(x.exists() for x in (p,a,h)):pytest.skip('Local trained study artifacts unavailable')
    torch.set_num_threads(1);return DecoderBridgeSession(p,a,h)

def same(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))

@pytest.mark.parametrize('kind',['hidden','mean','energy_read'])
def test_teacher_state_is_unchanged(kind):
    s=session(kind);m=load_energy_model(paths()[0]);x=torch.arange(64).remainder(256)[None];state=None;baseline=[]
    with torch.no_grad():
        for i in range(x.shape[1]):
            y,state=m(x[:,i:i+1],state);baseline.append(y)
    y=s.observe(x)
    assert same(state,s.state)
    assert torch.equal(y[:,:15],torch.cat(baseline,1)[:,:15])
    if kind=='hidden':assert s.history is None
    else:assert s.history.untyped_storage().nbytes()==512

@pytest.mark.parametrize('kind',['hidden','energy_read'])
def test_future_change_does_not_change_prefix(kind):
    a=session(kind);b=session(kind);x=torch.arange(64)[None];changed=x.clone();changed[:,40:]+=1
    assert torch.equal(a.observe(x)[:,:40],b.observe(changed)[:,:40])

@pytest.mark.parametrize('kind',['hidden','mean','energy_read'])
def test_fresh_process_inference(kind,tmp_path):
    s=session(kind);s.observe(torch.arange(32)[None]);snap=tmp_path/'state.pt';s.save(snap)
    target=s.observe(torch.arange(32,64)[None]);expected=s.state
    teacher,adapter,head=paths(kind)
    dest=tmp_path/'out.pt'
    code=f"import torch;from energy_forecast.decoder_bridge import DecoderBridgeSession;torch.set_num_threads(1);s=DecoderBridgeSession({str(teacher)!r},{str(adapter)!r},{str(head)!r});s.restore({str(snap)!r});y=s.observe(torch.arange(32,64)[None]);torch.save({{'logits':y,'state':s.state}},{str(dest)!r})"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    out=torch.load(dest,weights_only=True)
    assert torch.equal(target,out['logits']);assert same(expected,out['state'])

def test_invalid_tokens_do_not_partially_advance():
    s=session('energy_read');s.observe(torch.arange(24)[None]);state=s.state;n=s.steps;history=s.history.clone()
    with pytest.raises(ValueError):s.observe(torch.tensor([[2,3,256]]))
    assert s.steps==n and same(state,s.state) and torch.equal(history,s.history)

def test_invalid_snapshot_is_atomic(tmp_path):
    s=session('energy_read');s.observe(torch.arange(24)[None]);snap=tmp_path/'state.pt';s.save(snap)
    d=torch.load(snap,weights_only=True);d['history'][0,-1,0]+=1;torch.save(d,snap)
    old=s.state;oldh=s.history.clone()
    with pytest.raises(ValueError):s.restore(snap)
    assert same(s.state,old) and torch.equal(s.history,oldh)

def test_wrong_forecast_rejected():
    p,a,_=paths('mean');_,_,wrong=paths('mean',22)
    if not a.exists():pytest.skip('Trained artifacts unavailable')
    with pytest.raises(ValueError):DecoderBridgeSession(p,a,wrong)

def test_folded_mean_needs_no_head():
    p,a,h=paths('mean');folded=ROOT/'results/decoder_bridge/folded_mean_seed11.pt'
    if not folded.exists():pytest.skip('Conversion not present')
    old=session('mean');new=DecoderBridgeSession(p,folded)
    tokens=torch.arange(80)[None]
    torch.testing.assert_close(old.observe(tokens),new.observe(tokens),rtol=1e-5,atol=2e-5)

def test_invalid_normalization_rejected_at_load(tmp_path):
    p,a,_=paths('hidden')
    if not a.exists():pytest.skip('Local artifacts unavailable')
    d=torch.load(a,weights_only=True);d['normalization']['scale'][0]=0
    invalid=tmp_path/'invalid.pt';torch.save(d,invalid)
    with pytest.raises(ValueError):DecoderBridgeSession(p,invalid)
