import pytest
import torch
from torch.nn import functional as F
from energy_forecast.decoder_bridge import DecoderConfig,DecoderAdapter,energy_query_read,distribution_moments,KINDS
from energy_forecast.model import EnergyLandscape

def fun(b=2,k=3,d=32,dtype=torch.float64):
    torch.manual_seed(23001)
    lower=torch.tril(torch.randn(k,d,d,dtype=dtype)*.03,-1)+torch.eye(d,dtype=dtype)[None]
    return EnergyLandscape(torch.randn(b,k,dtype=dtype),torch.randn(b,k,d,dtype=dtype),torch.rand(b,k,d,dtype=dtype)+.5,lower)

@pytest.mark.parametrize('kind',KINDS)
def test_initial_identity(kind):
    torch.manual_seed(14);m=DecoderAdapter(DecoderConfig(kind)).double();f=fun()
    h=torch.randn(2,160,dtype=torch.float64);x=torch.randn(2,16,8,dtype=torch.float64);w=torch.randn(256,160,dtype=torch.float64)
    assert torch.equal(m(h,x,w,f,f.mean),F.linear(h,w))

@pytest.mark.parametrize('kind',KINDS)
def test_finite_parameter_gradients(kind):
    torch.manual_seed(12);m=DecoderAdapter(DecoderConfig(kind)).double()
    with torch.no_grad():m.last.weight.normal_(std=.001)
    h=torch.randn(2,160,dtype=torch.float64);x=torch.randn(2,16,8,dtype=torch.float64);f=fun()
    value=m.delta(h,x,f,f.mean).square().sum();value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    if kind=='energy_read':assert m.query.weight.grad.abs().sum()>0

def test_force_matches_independent_autograd():
    f=fun(d=6);q=torch.randn(2,2,6,dtype=torch.float64,requires_grad=True)
    features=energy_query_read(f,q)
    grad=torch.stack([torch.autograd.grad(f.energy(q[:,i]).sum(),q,retain_graph=True)[0][:,i] for i in range(2)],1)
    _,v=distribution_moments(f)
    torch.testing.assert_close(features[...,6:],torch.tanh(-grad*v.sqrt()[:,None]),rtol=1e-10,atol=1e-10)

def test_query_gradcheck():
    f=fun(b=1,d=4);q=torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:energy_query_read(f,z),(q,),atol=1e-5)

def test_no_mutation():
    f=fun(d=6);q=torch.randn(2,2,6,dtype=torch.float64)
    before=[t.clone() for t in (f.logits,f.means,f.scales,f.correlation_cholesky,q)]
    energy_query_read(f,q)
    assert all(torch.equal(a,b) for a,b in zip(before,(f.logits,f.means,f.scales,f.correlation_cholesky,q)))

def test_parameter_budget():
    counts=[sum(p.numel() for p in DecoderAdapter(DecoderConfig(kind)).parameters()) for kind in KINDS]
    assert max(counts)-min(counts)<200

def test_constant_horizon_has_no_context_mean_signal():
    m=DecoderAdapter(DecoderConfig('mean'));h=torch.zeros(2,160);x=torch.zeros(2,16,8)
    mean=torch.randn(2,32);mean[:,24:]=1
    out=m.raw_features(h,x,mean=mean)
    assert torch.equal(out[0,24:32],out[1,24:32])

@pytest.mark.parametrize('kwargs',[{'history':0},{'queries':3},{'kind':'bad'}])
def test_config_rejects_invalid(kwargs):
    args={'kind':'energy_read',**kwargs}
    with pytest.raises(ValueError):DecoderAdapter(DecoderConfig(**args))
