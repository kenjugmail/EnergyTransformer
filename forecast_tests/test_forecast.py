import math,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal,MixtureSameFamily,Categorical
from energy_forecast.model import ForecastConfig,ForecastEnergy,EnergyLandscape,real_fourier_basis
from energy_forecast.probe import aligned_windows,state_sketch

def small(**kwargs):return ForecastEnergy(ForecastConfig(history=8,variables=2,horizons=(1,4),width=16,**kwargs))

def test_horizon_validation():
    for h in [(0,1),(4,1),(1,1)]:
        with pytest.raises(ValueError):ForecastConfig(horizons=h).validate()

def test_target_indices_and_doc_end():
    t=torch.arange(20).float()[:,None];x,y,anchors=aligned_windows(t,4,(1,4),stride=3)
    assert torch.equal(x[:,-1,0],anchors.float())
    assert torch.equal(y[:,:,0],anchors[:,None]+torch.tensor([1,4]))
    assert y.max()<20

def test_short_doc_no_crossing():assert aligned_windows(torch.ones(4,2),4,(1,4)) is None

@pytest.mark.parametrize('n',[1,3,8,16])
def test_fourier_is_invertible_past_time_change(n):
    a=real_fourier_basis(n,dtype=torch.float64)
    torch.testing.assert_close(a@a.T,torch.eye(n,dtype=torch.float64),atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('correlated',[False,True])
def test_normalized_energy_matches_torch_density(correlated):
    torch.manual_seed(0);m=small(correlated=correlated).double()
    if correlated:
        with torch.no_grad():m.raw_correlation.add_(torch.randn_like(m.raw_correlation)*.2)
    x=torch.randn(3,8,2,dtype=torch.float64);f=m(x);y=torch.randn(3,4,dtype=torch.float64)
    scale=f.scales[...,None]*f.correlation_cholesky[None]
    dist=MixtureSameFamily(Categorical(logits=f.logits),MultivariateNormal(f.means,scale_tril=scale))
    torch.testing.assert_close(f.energy(y),-dist.log_prob(y),atol=1e-10,rtol=1e-10)

def test_single_gaussian_minimum_and_curvature():
    m=small(components=1).double();f=m(torch.randn(2,8,2,dtype=torch.float64));z=f.means[:,0].detach().requires_grad_()
    g,=torch.autograd.grad(f.energy(z).sum(),z);assert g.abs().max()<1e-12

def test_multimodal_energy_is_not_quadratic():
    # Two equally likely separated normalized Gaussian components.
    f=EnergyLandscape(torch.zeros(1,2,dtype=torch.float64),torch.tensor([[[-2.],[2.]]],dtype=torch.float64),torch.full((1,2,1),.4,dtype=torch.float64),torch.ones(2,1,1,dtype=torch.float64))
    middle=torch.zeros(1,1,dtype=torch.float64,requires_grad=True)
    g,=torch.autograd.grad(f.energy(middle).sum(),middle,create_graph=True)
    h,=torch.autograd.grad(g.sum(),middle)
    assert h.item()<0
    assert f.energy(torch.tensor([[2.]],dtype=torch.float64))<f.energy(middle)

def test_cross_variable_curvature_not_independent():
    m=small(components=1).double()
    with torch.no_grad():m.raw_correlation[0,1,0]=.7
    f=m(torch.randn(1,8,2,dtype=torch.float64));z=torch.randn(1,4,dtype=torch.float64,requires_grad=True)
    h=torch.autograd.functional.hessian(lambda zz:f.energy(zz).sum(),z).reshape(4,4)
    assert h[0,1].abs()>.01

def test_candidate_energy_gradcheck():
    m=small().double();f=m(torch.randn(2,8,2,dtype=torch.float64)).detached()
    y=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(f.energy,(y,))

def test_parameter_and_history_gradients():
    m=small();x=torch.randn(3,8,2,requires_grad=True);loss=m.loss(x,torch.randn(3,2,2));loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert m.output.weight.grad.abs().sum()>0

def test_sampling_covariance():
    f=EnergyLandscape(torch.zeros(1,1),torch.zeros(1,1,2),torch.ones(1,1,2),torch.tensor([[[1.,0.],[.6,.8]]]))
    samples=f.sample(15000,generator=torch.Generator().manual_seed(11))[0]
    assert abs(float(torch.cov(samples.T)[0,1])-.6)<.04

def test_refinement_fixed_energy_nonincrease_and_no_weight_grads():
    m=small();f=m(torch.randn(3,8,2));z,es=f.refine(torch.randn(3,4)*4,steps=5)
    assert (es[:,1:]<=es[:,:-1]+1e-6).all()
    assert torch.isfinite(z).all() and all(p.grad is None for p in m.parameters())

def test_no_future_in_forward_signature():
    import inspect
    assert list(inspect.signature(ForecastEnergy.forward).parameters)==['self','history']
    m=small();prefix=torch.randn(2,8,2);a=m(prefix).means.clone()
    unused_future=torch.randn(2,64,2)*1e8
    assert torch.equal(a,m(prefix).means)

def test_nan_history_rejected():
    with pytest.raises(ValueError):small()(torch.full((1,8,2),float('nan')))

def test_sketch_does_not_modify_state():
    p=torch.rand(2,4,8)+.25;s=torch.randn(2,4,8,6);before=(p.clone(),s.clone());z=state_sketch(((p,s),))
    assert z.shape==(2,8) and torch.equal(p,before[0]) and torch.equal(s,before[1])

def test_roundtrip(tmp_path):
    m=small();x=torch.randn(2,8,2);expected=m(x).energy(torch.zeros(2,4))
    torch.save(m.state_dict(),tmp_path/'m.pt');n=small();n.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True))
    assert torch.equal(expected,n(x).energy(torch.zeros(2,4)))

def test_enwik8_pilot_blocks_do_not_overlap_or_cross_split():
    from energy_forecast.prepare_enwik8 import block_starts
    for lo,hi in [(0,90000),(90000,95000),(95000,100000)]:
        starts=block_starts(lo,hi,128,16,112)
        assert starts.min()>=lo and (starts+128).max()<=hi
        assert np.all(np.diff(starts)>=128)
