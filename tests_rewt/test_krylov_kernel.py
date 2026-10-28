from dataclasses import replace

import pytest
import torch

from rewt.kernels import core_quadratic_action, krylov2_refine, prepare_krylov2
from rewt.model import Conditioning, EnergyConfig, JointEnergyLM
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM


def config(**kwargs):
    options = dict(width=8, hidden=17, modes=5, history=8, context_kernel=3,
                   recurrent_type='gru', output_geometry='centered',
                   compatibility_geometry='anchor_relative', wave_geometry='anchor_relative',
                   initial_carry_scale=0., cache_history=True, early_backtrack=True)
    options.update(kwargs)
    return EnergyConfig(**options)


@pytest.mark.parametrize('geometry', ['raw', 'centered'])
@pytest.mark.parametrize('step_size', [.25, 8., 1e8])
def test_mode_space_matches_both_rounds_and_all_parameter_gradients(geometry, step_size):
    torch.set_num_threads(1)
    torch.manual_seed(22102)
    reference = JointEnergyLM(config(output_geometry=geometry, step_size=step_size)).double()
    kernel = JointEnergyLM(replace(reference.config, refinement_backend='krylov2')).double()
    kernel.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 256, (3, 13))
    a = reference(tokens, details=True, state_gradient_interval=4)
    b = kernel(tokens, details=True, state_gradient_interval=4)
    assert b.details['refinement_backend'] == 'krylov2'
    for x, y in [(a.logits, b.logits), (a.state.joint, b.state.joint),
                 (a.details['energy_trace'], b.details['energy_trace'])]:
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-11)
    assert a.details['cost']['accepted'] == b.details['cost']['accepted']
    if step_size == 1e8:
        assert b.details['cost']['accepted'] == 0
    assert (b.details['energy_trace'][..., 1:] <= b.details['energy_trace'][..., :-1] + 1e-12).all()
    actual = kernel.energy(b.details['joint'], b.details['conditioning'])
    torch.testing.assert_close(actual, b.details['energy_trace'][..., -1], rtol=1e-10, atol=1e-11)
    for m, o in [(reference, a), (kernel, b)]:
        (o.logits.square().mean() + .1 * o.state.joint.square().mean()).backward()
    for (name, p), (other, q) in zip(reference.named_parameters(), kernel.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-10, msg=name)


def test_hessian_action_and_its_parameter_derivatives_match_autograd():
    torch.manual_seed(22103)
    model = JointEnergyLM(config()).double()
    frequency, precision = model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid()
    z = torch.randn(4, model.dimensions, dtype=torch.float64, requires_grad=True)
    cond = Conditioning(torch.zeros_like(z), None, None, None, {}, frequency, precision)
    expected = torch.autograd.grad(model.quadratic(z, cond).sum(), z, create_graph=True)[0]
    actual = core_quadratic_action(model, z, precision)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for p in (model.raw_precision, model.decoder.weight):
        a = torch.autograd.grad(actual.square().sum(), p, retain_graph=True)[0]
        b = torch.autograd.grad(expected.square().sum(), p, retain_graph=True)[0]
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


def test_nonfinite_candidate_energy_retains_the_anchor():
    torch.manual_seed(22112)
    model = JointEnergyLM(config()).double()
    frequency, precision = model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid()
    cond, _ = model.condition(torch.tensor([1, 2]), model.initial_state(2),
                              constants=(frequency, precision))
    cond.amplitudes = torch.full_like(cond.amplitudes, float('nan'))
    expected, _, _ = model.refine(cond.anchor, cond)
    actual, _, accepted = krylov2_refine(cond.anchor, cond.amplitudes, model.phase, frequency,
                                         *prepare_krylov2(model, frequency, precision),
                                         output_start=model.blocks['output'].start, centered=True)
    assert torch.isfinite(actual).all()
    assert not accepted.any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [dict(rounds=0), dict(rounds=1), dict(rounds=3),
                                    dict(update_states=False), dict(update_output=False),
                                    dict(initial_noise=True)])
def test_interventions_fall_back_to_reference(kwargs):
    torch.manual_seed(22104)
    a = JointEnergyLM(config()).double()
    b = JointEnergyLM(replace(a.config, refinement_backend='krylov2')).double()
    b.load_state_dict(a.state_dict())
    tokens = torch.randint(0, 256, (2, 4))
    if 'initial_noise' in kwargs:
        kwargs = dict(initial_noise=torch.randn(2, 4, a.dimensions, dtype=torch.float64) * .1)
    x, y = a(tokens, **kwargs), b(tokens, **kwargs)
    assert y.details['refinement_backend'] == 'reference'
    torch.testing.assert_close(x.logits, y.logits, rtol=0, atol=0)


def test_mode_space_preserves_causality_chunks_and_optimizer_updates():
    torch.manual_seed(22105)
    options = dict(vars(config(refinement_backend='krylov2')))
    model = PositionMemoryEnergyLM(PositionMemoryConfig(**options, position_period=8)).double()
    tokens = torch.randint(0, 256, (2, 25))
    changed = tokens.clone()
    changed[:, 14:] = (changed[:, 14:] + 37) % 256
    for _ in range(2):
        whole = model(tokens)
        other = model(changed)
        first = model(tokens[:, :6])
        second = model(tokens[:, 6:17], first.state)
        final = model(tokens[:, 17:], second.state)
        torch.testing.assert_close(whole.logits[:, :14], other.logits[:, :14], rtol=0, atol=0)
        torch.testing.assert_close(whole.logits, torch.cat((first.logits, second.logits, final.logits), 1),
                                   rtol=1e-10, atol=1e-11)
        assert final.state.count == 25
        # Rebuild mode matrices after a real optimizer mutation; no stale graph.
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer.zero_grad()
        whole.logits.square().mean().backward()
        optimizer.step()


@pytest.mark.parametrize('kwargs', [dict(support=True), dict(activation='mlp'),
                                    dict(initial_carry_scale=.05), dict(wave_geometry='absolute'),
                                    dict(compatibility_geometry='absolute'),
                                    dict(halting_tolerance=.01), dict(schedule='coarse_to_fine')])
def test_unsupported_mode_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError, match='krylov2 requires'):
        JointEnergyLM(config(refinement_backend='krylov2', **kwargs))
