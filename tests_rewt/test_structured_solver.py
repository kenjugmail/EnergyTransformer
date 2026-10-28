import pytest
import torch
from rewt.advanced_kernels import prepare_solver
from rewt.structured_solver import prepare_structured
from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM


@pytest.mark.parametrize('geometry', ['raw', 'centered'])
@pytest.mark.parametrize('weight', [0., .04, 4.])
def test_structured_elimination_matches_dense_solve_and_gradients(geometry, weight):
    torch.manual_seed(22181)
    model = ArchitectureEnergyLM(ArchitectureConfig(width=8, hidden=19, modes=7, history=8,
        recurrent_type='gru', compatibility_geometry='anchor_relative', wave_geometry='anchor_relative',
        initial_carry_scale=0., output_geometry=geometry, decoder_weight=weight)).double()
    with torch.no_grad():
        model.raw_precision.normal_()
    f, p = model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid()
    dense = prepare_solver(model, f, p, 'preconditioned')
    actual = prepare_structured(model, f, p)
    for key in ('lift', 'gram', 'step', 'upper'):
        torch.testing.assert_close(actual[key], dense[key], atol=1e-11, rtol=1e-10)
    params = model.raw_frequency, model.raw_precision, model.decoder.weight
    da = torch.autograd.grad(actual['lift'].square().sum(), params, retain_graph=True)
    db = torch.autograd.grad(dense['lift'].square().sum(), params)
    for a, b in zip(da, db):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


def test_structured_model_execution_and_gradients():
    from dataclasses import replace
    torch.manual_seed(22207)
    cfg=ArchitectureConfig(width=8,hidden=19,modes=5,history=8,position_period=16,
        recurrent_type='gru',context_kernel=3,compatibility_geometry='anchor_relative',
        wave_geometry='anchor_relative',initial_carry_scale=0.,output_geometry='centered',
        solver='preconditioned',rounds=4,step_size=.75,backbone='scan',delta_size=4)
    a=ArchitectureEnergyLM(cfg).double();b=ArchitectureEnergyLM(replace(cfg,preconditioner_factorization='structured')).double()
    b.load_state_dict(a.state_dict());tokens=torch.randint(0,256,(2,17))
    x,y=a(tokens),b(tokens);torch.testing.assert_close(x.logits,y.logits,atol=1e-11,rtol=1e-10)
    x.logits.square().mean().backward();y.logits.square().mean().backward()
    for n,p in a.named_parameters():
        q=dict(b.named_parameters())[n]
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-9,msg=n)
