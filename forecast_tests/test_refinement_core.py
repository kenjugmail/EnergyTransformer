import copy
from pathlib import Path
import subprocess,sys,json
import pytest
import torch
from energy_forecast.refine_model import RefinementAdapter,RefinementConfig,compact_energy_read,KINDS

torch.set_num_threads(1)

def fixture(kind='energy_memory',dtype=torch.float64):
    torch.manual_seed(4)
    model=RefinementAdapter(RefinementConfig(kind=kind,width=8,memory_width=8,latent=8 if kind=='energy_full' else 4,parameter_budget=420)).to(dtype)
    h=torch.randn(3,8,dtype=dtype);m=torch.randn(3,8,dtype=dtype);w=torch.randn(9,8,dtype=dtype)
    return model,h,m,w

@pytest.mark.parametrize('kind',KINDS)
def test_zero_initial_residual_matches_teacher(kind):
    model,h,m,w=fixture(kind)
    assert torch.equal(model(h,m,w),h@w.T)


def test_gradient_is_scalar_energy_gradient():
    model,h,m,w=fixture();c=model.landscape(model.context_input(h,m));z=torch.randn(3,4,dtype=h.dtype,requires_grad=True)
    reference=torch.autograd.grad(model.energy(z,c).sum(),z,create_graph=True)[0]
    torch.testing.assert_close(model.gradient(z,c),reference,rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(lambda z:model.energy(z,c),(z,))


def test_unrolled_parameter_derivatives_include_through_gradient():
    model,h,m,w=fixture();model.output.weight.data.normal_(std=.1)
    clone=copy.deepcopy(model);expected=[];z=h.new_zeros(3,4,requires_grad=True)
    c=clone.landscape(clone.context_input(h,m))
    for _ in range(4):
        grad=torch.autograd.grad(clone.energy(z,c).sum(),z,create_graph=True)[0]
        z=z-clone.config.step_size*grad
    clone.logits(h,z,w).square().mean().backward()
    model(h,m,w).square().mean().backward()
    for (name,p),(other,q) in zip(model.named_parameters(),clone.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-10,atol=1e-11)


def test_energy_can_be_nonconvex_and_is_coercive():
    model,h,m,w=fixture();model.directions.data.fill_(2.)
    c=(torch.ones(1,model.units,dtype=h.dtype)*2,torch.zeros(1,model.units,dtype=h.dtype))
    z=torch.zeros(4,dtype=h.dtype)
    H=torch.autograd.functional.hessian(lambda v:model.energy(v[None],c).sum(),z)
    assert torch.linalg.eigvalsh(H).min()<0
    assert model.energy(torch.full((1,4),1e6,dtype=h.dtype),c)>0


def test_guarded_solve_never_accepts_energy_increase():
    model,h,m,w=fixture();model.output.weight.data.normal_();model.directions.data*=5
    before=[p.detach().clone() for p in model.parameters()]
    result=model.solve(h,m,w,max_steps=16,tolerance=1e-5)
    assert ((result['energies'][:,1:]-result['energies'][:,:-1])<=0).all()
    assert (result['steps']<=16).all()
    assert all(torch.equal(p,b) for p,b in zip(model.parameters(),before))


def test_fixed_energy_during_solver():
    model,h,m,w=fixture();count=[0];original=model.landscape
    def counted(c):count[0]+=1;return original(c)
    model.landscape=counted
    model.solve(h,m,w,max_steps=5)
    assert count==[1]


def test_batch_independence_and_extra_steps():
    model,h,m,w=fixture();model.output.weight.data.normal_()
    torch.testing.assert_close(model(h,m,w),torch.cat([model(h[i:i+1],m[i:i+1],w) for i in range(3)]),rtol=1e-12,atol=1e-12)
    outputs=model.trajectory(h,m,w,steps=8)
    torch.testing.assert_close(outputs[3],model(h,m,w,steps=4))
    assert torch.equal(model(h,m,w,steps=0),h@w.T)


def test_memory_intervention_is_distinct_but_hidden_control_invariant():
    for kind in ('energy_memory','energy_hidden'):
        model,h,m,w=fixture(kind);model.output.weight.data.normal_()
        before=model(h,m,w);after=model(h,m+5,w)
        if kind=='energy_hidden':assert torch.equal(before,after)
        else:assert not torch.equal(before,after)


def test_compact_energy_read_does_not_mutate_and_has_declared_shape():
    p=torch.rand(2,4,16,dtype=torch.float64)+.25;s=torch.randn(2,4,16,40,dtype=torch.float64)
    before=(p.clone(),s.clone());read=compact_energy_read(((p,s),))
    assert read.shape==(2,64) and torch.isfinite(read).all()
    assert torch.equal(p,before[0]) and torch.equal(s,before[1])
    r2=compact_energy_read(((p,s+1),));assert not torch.equal(read,r2)

@pytest.mark.parametrize('steps',[-1,65,True,1.5])
def test_bad_steps_fail(steps):
    model,h,m,w=fixture()
    with pytest.raises(ValueError):model(h,m,w,steps=steps)

@pytest.mark.parametrize('tol',[-1,float('nan'),float('inf')])
def test_bad_solver_tolerances_fail(tol):
    model,h,m,w=fixture()
    with pytest.raises(ValueError):model.solve(h,m,w,tolerance=tol)


def test_hessian_is_symmetric():
    model,h,m,w=fixture();c=model.landscape(model.context_input(h[:1],m[:1]))
    z=torch.randn(4,dtype=torch.float64)
    H=torch.autograd.functional.hessian(lambda z:model.energy(z[None],c).sum(),z)
    torch.testing.assert_close(H,H.T,rtol=1e-12,atol=1e-12)


def test_full_width_energy_gradient_and_parameter_gradient():
    model,h,m,w=fixture('energy_full');model.amplitude.data.fill_(.4)
    fixed=model.landscape(model.context_input(h,m));z=torch.randn(3,8,dtype=h.dtype,requires_grad=True)
    reference=torch.autograd.grad(model.energy(z,fixed).sum(),z,create_graph=True)[0]
    torch.testing.assert_close(reference,model.gradient(z,fixed),atol=1e-12,rtol=1e-12)
    model(h,m,w).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('kind',['recurrent','energy_hidden','energy_memory','energy_full'])
def test_final_only_projection_matches_all_step_projection(kind):
    model,h,m,w=fixture(kind)
    if kind=='energy_full':model.amplitude.data.fill_(.4)
    else:model.output.weight.data.normal_()
    assert torch.equal(model(h,m,w,steps=8),model.trajectory(h,m,w,steps=8)[-1])


def test_empty_batch_rejected():
    model,h,m,w=fixture()
    with pytest.raises(ValueError):model(h[:0],m[:0],w)


def test_boolean_step_size_rejected():
    with pytest.raises(ValueError):RefinementAdapter(RefinementConfig(step_size=True))
