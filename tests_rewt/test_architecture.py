from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from rewt.advanced_kernels import (affine_scan, delta_chunk, delta_memory, prepare_solver,
                                  reduced_refine, core_matrix)
from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM
from rewt.model import Conditioning, EnergyConfig, JointEnergyLM
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM


torch.set_num_threads(1)


def config(**kw):
    args = dict(width=8, hidden=19, modes=5, history=8, position_period=16,
                context_kernel=3, recurrent_type='gru', cache_history=True,
                output_geometry='centered', compatibility_geometry='anchor_relative',
                wave_geometry='anchor_relative', initial_carry_scale=0.)
    args.update(kw)
    return ArchitectureConfig(**args)


def check_grads(a, b, tol=1e-9):
    for name, p in a.named_parameters():
        q = dict(b.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=tol, rtol=tol, msg=name)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('solver', ['krylov2', 'certified'])
def test_exact_execution_and_all_parameter_gradients(packed, solver):
    torch.manual_seed(22151)
    cfg = config(solver=solver, packed=packed)
    base = {k: v for k, v in vars(cfg).items() if k in PositionMemoryConfig.__dataclass_fields__}
    reference = PositionMemoryEnergyLM(PositionMemoryConfig(**base)).double()
    candidate = ArchitectureEnergyLM(cfg).double()
    candidate.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 256, (2, 19))
    a, b = reference(tokens, details=True), candidate(tokens, details=True)
    torch.testing.assert_close(a.logits, b.logits, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(a.details['energy_trace'], b.details['energy_trace'], atol=1e-11, rtol=1e-10)
    a.logits.square().mean().backward(); b.logits.square().mean().backward()
    check_grads(reference, candidate)
    # The optimized certified two-step lift (trace disabled) follows the same
    # recurrence, including its parameter derivatives.
    candidate.zero_grad()
    plain = candidate(tokens, trace_energy=False)
    torch.testing.assert_close(plain.logits, b.logits, atol=1e-11, rtol=1e-10)
    plain.logits.square().mean().backward()
    check_grads(reference, candidate)


@pytest.mark.parametrize('support', [False, True])
@pytest.mark.parametrize('geometry', ['absolute', 'anchor_relative'])
@pytest.mark.parametrize('activation', ['fourier', 'mlp'])
def test_general_directional_values_and_gradients(support, geometry, activation):
    torch.manual_seed(22152)
    cfg = EnergyConfig(width=6, hidden=15, modes=7, history=5, support=support,
        rounds=3, step_size=4., context_kernel=2, output_geometry='centered',
        compatibility_geometry=geometry, wave_geometry=geometry, activation=activation,
        schedule='coarse_to_fine', diffusion_samples=1, diffusion_levels=1)
    a = JointEnergyLM(cfg).double()
    b = JointEnergyLM(replace(cfg, refinement_backend='directional')).double()
    b.load_state_dict(a.state_dict())
    tokens = torch.randint(0, 256, (2, 5))
    noise = torch.randn(2, 5, a.dimensions, dtype=torch.float64) * .1
    x, y = a(tokens, details=True, initial_noise=noise), b(tokens, details=True, initial_noise=noise)
    torch.testing.assert_close(x.logits, y.logits, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(x.details['energy_trace'], y.details['energy_trace'], atol=2e-10, rtol=2e-9)
    x.logits.square().mean().backward(); y.logits.square().mean().backward()
    check_grads(a, b, 2e-8)


@pytest.mark.parametrize('rounds', [0, 1, 2, 4, 12])
def test_preconditioned_recurrence_matches_full_space_and_descends(rounds):
    torch.manual_seed(22153)
    model = ArchitectureEnergyLM(config(solver='preconditioned', step_size=20.)).double()
    frequency, precision = model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid()
    ctx = prepare_solver(model, frequency, precision, 'preconditioned')
    anchor = model.canonical_state(torch.randn(3, model.dimensions, dtype=torch.float64))
    amp = torch.randn(3, model.config.modes, dtype=torch.float64).tanh() * .5 / model.config.modes**.5
    actual, trace = reduced_refine(model, anchor, amp, frequency, precision, ctx, 'preconditioned', rounds, trace=True)
    cond = Conditioning(anchor, amp, None, None, {}, frequency, precision)
    expected = anchor
    for _ in range(rounds):
        grad = model.gradient(expected, cond)
        expected = expected - ctx['step'] * torch.linalg.solve(core_matrix(model, frequency, precision), grad.T).T
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    energies = torch.stack(trace, -1)
    assert bool((energies[:, 1:] <= energies[:, :-1] + 1e-12).all())
    pa = torch.autograd.grad(actual.square().sum(), (model.raw_precision, model.raw_frequency, model.decoder.weight), retain_graph=True, allow_unused=True)
    pb = torch.autograd.grad(expected.square().sum(), (model.raw_precision, model.raw_frequency, model.decoder.weight), allow_unused=True) if rounds else pa
    for a, b in zip(pa, pb):
        if a is not None:
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_scan_zero_gates_and_chunk_delta_have_correct_gradients():
    torch.manual_seed(22154)
    a = torch.rand(2, 17, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    h = torch.randn(2, 5, dtype=torch.float64)
    expected = []
    for t in range(17):
        h = a[:, t] * h + b[:, t]
        expected.append(h)
    # Reconstruct initial seed for the parallel recurrence.
    torch.manual_seed(22154)
    _ = torch.rand(2, 17, 5, dtype=torch.float64)
    _ = torch.randn_like(a)
    initial = torch.randn(2, 5, dtype=torch.float64)
    result = affine_scan(a, b, initial)
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(result.square().sum(), (a, b), retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), (a, b))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(affine_scan(torch.zeros_like(a), b, initial), b)
    k = F.normalize(torch.randn(2, 17, 4, dtype=torch.float64), dim=-1).requires_grad_()
    q = F.normalize(torch.randn_like(k), dim=-1).requires_grad_()
    v = torch.randn(2, 17, 5, dtype=torch.float64, requires_grad=True)
    beta = torch.rand(2, 17, 1, dtype=torch.float64, requires_grad=True)
    initial = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    memory = initial
    expected = []
    for t in range(17):
        error = v[:, t] - (k[:, t, :, None] * memory).sum(-2)
        memory = memory + beta[:, t, :, None] * k[:, t, :, None] * error[:, None]
        expected.append((q[:, t, :, None] * memory).sum(-2))
    result, final = delta_chunk(k, v, beta, q, initial)
    expected = torch.stack(expected, 1)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(final, memory, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(result.square().sum() + final.square().sum(), (k, q, v, beta, initial), retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum() + memory.square().sum(), (k, q, v, beta, initial))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('options', [dict(), dict(backbone='scan'),
    dict(backbone='scan', feedback_stride=4), dict(backbone='scan', delta_size=4),
    dict(backbone='scan', feedback_stride=4, delta_size=4, patch_stride=3),
    dict(patch_stride=3), dict(backbone='scan', output_rank=4, mode_layout='hybrid')])
def test_architecture_causal_chunk_continuation_and_training(options):
    torch.manual_seed(22155)
    model = ArchitectureEnergyLM(config(**options, solver='preconditioned', step_size=.7)).double()
    tokens = torch.randint(0, 256, (2, 27))
    changed = tokens.clone(); changed[:, 14:] = (changed[:, 14:] + 17) % 256
    full = model(tokens, details=True)
    other = model(changed)
    torch.testing.assert_close(full.logits[:, :14], other.logits[:, :14], atol=1e-11, rtol=1e-10)
    state, parts = None, []
    for chunk in (tokens[:, :3], tokens[:, 3:11], tokens[:, 11:12], tokens[:, 12:]):
        out = model(chunk, state)
        state = out.state; parts.append(out.logits)
    torch.testing.assert_close(full.logits, torch.cat(parts, 1), atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(full.state.memory, state.memory, atol=1e-10, rtol=1e-9)
    assert state.count == tokens.shape[1]
    assert (full.details['energy_trace'][..., 1:] <= full.details['energy_trace'][..., :-1] + 1e-12).all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(tokens, state_gradient_interval=8).logits.square().mean() + .1 * model.diversity_loss()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()


def test_delta_overwrites_same_key_and_retains_orthogonal_key():
    keys = torch.tensor([[[1., 0.], [0., 1.], [1., 0.]]])
    values = torch.tensor([[[3.], [5.], [7.]]])
    reads, state = delta_chunk(keys, values, torch.ones(1, 3, 1), keys, torch.zeros(1, 2, 1))
    torch.testing.assert_close(reads.flatten(), torch.tensor([3., 5., 7.]))
    torch.testing.assert_close(state.flatten(), torch.tensor([7., 5.]))


@pytest.mark.parametrize('recurrence', ['gru', 'mlp'])
@pytest.mark.parametrize('backend', ['reference', 'krylov2'])
def test_direct_packed_execution_preserves_outputs_gradients_chunks(recurrence, backend):
    torch.manual_seed(22197)
    cfg = PositionMemoryConfig(width=8, hidden=19, modes=5, history=8, position_period=8,
        context_kernel=3, recurrent_type=recurrence, cache_history=True, output_geometry='centered',
        compatibility_geometry='anchor_relative', wave_geometry='anchor_relative', initial_carry_scale=0.,
        refinement_backend=backend)
    a = PositionMemoryEnergyLM(cfg).double()
    b = PositionMemoryEnergyLM(replace(cfg, packed_projections=True)).double()
    b.load_state_dict(a.state_dict());tokens=torch.randint(0,256,(2,19))
    x,y=a(tokens),b(tokens)
    torch.testing.assert_close(x.logits,y.logits,atol=1e-11,rtol=1e-10)
    x.logits.square().mean().backward();y.logits.square().mean().backward();check_grads(a,b)
    one=b(tokens[:,:7]);two=b(tokens[:,7:],one.state)
    torch.testing.assert_close(y.logits,torch.cat((one.logits,two.logits),1),atol=1e-11,rtol=1e-10)


def test_directional_arbitrary_frequency_and_partial_coordinate_mask():
    torch.manual_seed(22199)
    cfg=EnergyConfig(width=6,hidden=13,modes=5,output_geometry='centered',rounds=4,step_size=.02)
    a=JointEnergyLM(cfg).double();b=JointEnergyLM(replace(cfg,refinement_backend='directional')).double();b.load_state_dict(a.state_dict())
    state=a.initial_state(2);cond,_=a.condition(torch.tensor([1,2]),state)
    # Deliberately include a frequency component along the output gauge.
    cond.frequency=torch.randn_like(a.raw_frequency)*.1
    mask=torch.ones(a.dimensions,dtype=torch.float64);mask[::3]=0
    x,_,tx=a.refine(cond.anchor+.1,cond,trace=True,active_mask=mask)
    y,_,ty=b.refine(cond.anchor+.1,cond,trace=True,active_mask=mask)
    torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    torch.testing.assert_close(torch.stack(tx),torch.stack(ty),atol=1e-11,rtol=1e-10)


def test_scan_counters_count_batched_operations_and_round_fallback():
    model=ArchitectureEnergyLM(config(backbone='scan'))
    tokens=torch.ones(2,7,dtype=torch.long)
    output=model(tokens)
    assert output.details['cost']['gradient_calls']==2
    assert output.details['cost']['energy_calls']==11
    assert output.details['refinement_positions']==7
    assert model(tokens,rounds=3).details['refinement_backend']=='reference'
    certified=ArchitectureEnergyLM(config(backbone='scan',solver='certified'))
    assert certified(tokens).details['cost']['energy_calls']==0
