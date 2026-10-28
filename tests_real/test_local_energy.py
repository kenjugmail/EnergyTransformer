import inspect

import pytest
import torch
from torch.nn import functional as F

from energy_real.local_energy import LocalConfig, LocalEnergyNetwork, gradient_alignment


@pytest.mark.parametrize('activation', ['linear', 'tanh', 'relu'])
@pytest.mark.parametrize('use_memory', [False, True])
def test_local_state_derivative_matches_independent_autograd(activation, use_memory):
    torch.manual_seed(39)
    model = LocalEnergyNetwork(LocalConfig(5, 3, width=4, depth=3, activation=activation,
                                           memory_keys=3 if use_memory else 0,
                                           memory_values=2 if use_memory else 0)).double()
    x = torch.randn(2, 5, dtype=torch.double)
    target = torch.randn(2, 3, dtype=torch.double)
    states = torch.randn(3, 2, 4, dtype=torch.double, requires_grad=True)
    duals = torch.randn_like(states)
    memory = torch.randn(2, 3, 2, dtype=torch.double) if use_memory else None
    energy = model.energy(x, target, states, memory, rho=.7, duals=duals).sum()
    expected, = torch.autograd.grad(energy, states)
    actual, _ = model.state_gradient(x, target, states, memory, rho=.7, duals=duals)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_pc_is_zero_dual_and_sample_updates_are_batch_independent():
    torch.manual_seed(17)
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=5, depth=3)).double()
    x = torch.randn(3, 4, dtype=torch.double)
    target = torch.randn(3, 3, dtype=torch.double)
    for dual_lr in (0., 1.):
        batch = model.relax(x, target, steps=13, dual_lr=dual_lr)
        single = model.relax(x[:1], target[:1], steps=13, dual_lr=dual_lr)
        torch.testing.assert_close(batch.states[:, :1], single.states)
        torch.testing.assert_close(batch.duals[:, :1], single.duals)
        assert not batch.states.requires_grad and not batch.duals.requires_grad
        if not dual_lr:
            assert not batch.duals.any()


def test_duals_reset_and_final_step_has_no_dual_update():
    torch.manual_seed(18)
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=5, depth=3))
    x, y = torch.randn(2, 4), torch.randn(2, 3)
    first = model.relax(x, y, steps=1, dual_lr=1.)
    assert not first.duals.any()
    a = model.relax(x, y, steps=12, dual_lr=1.)
    model.relax(x * 2, y * 3, steps=12, dual_lr=1.)
    b = model.relax(x, y, steps=12, dual_lr=1.)
    torch.testing.assert_close(a.states, b.states, rtol=0, atol=0)
    torch.testing.assert_close(a.duals, b.duals, rtol=0, atol=0)


def test_linear_pc_alm_converges_to_bp_gradient():
    torch.manual_seed(25)
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=4, depth=3, activation='linear')).double()
    x, labels = torch.randn(4, 4, dtype=torch.double), torch.tensor([0, 1, 2, 0])
    score = gradient_alignment(model, x, labels, method='pcalm', steps=1500, state_lr=.1, dual_lr=.5)
    assert score['cosine'] > .999999
    assert score['relative_error'] < 1e-5
    pc = gradient_alignment(model, x, labels, method='pc', steps=1500, state_lr=.1)
    assert pc['relative_error'] > .01


def test_plain_energy_relaxation_decreases_energy():
    torch.manual_seed(88)
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=4, depth=3))
    x, target = torch.randn(3, 4), torch.randn(3, 3)
    result = model.relax(x, target, steps=30, state_lr=.05, trace=True)
    energies = [r['energy'] for r in result.trace] + [float(result.energy.mean())]
    assert all(b <= a + 1e-6 for a, b in zip(energies, energies[1:]))


def test_energy_prediction_has_no_labels_and_chunking_does_not_change_answers():
    torch.manual_seed(89)
    model = LocalEnergyNetwork(LocalConfig(4, 5, width=4, depth=3))
    x = torch.randn(3, 4)
    assert 'labels' not in inspect.signature(model.predict).parameters
    all_at_once = model.predict(x, method='energy', steps=12, candidate_chunk=5)
    chunked = model.predict(x, method='energy', steps=12, candidate_chunk=1)
    torch.testing.assert_close(chunked, all_at_once)
    energies = model.candidate_energies(x, steps=12)
    assert (energies >= 0).all()
    loss = model.energy_loss(x, torch.tensor([0, 1, 4]), steps=12)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_local_training_does_not_use_autograd_through_activity_history():
    torch.manual_seed(24)
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=4, depth=3))
    x, labels = torch.randn(2, 4), torch.tensor([1, 2])
    loss, result = model.local_loss(x, labels, method='pcalm', steps=12)
    assert result.states.grad_fn is None
    assert result.duals.grad_fn is None
    grads = torch.autograd.grad(loss, list(model.parameters()))
    target = F.one_hot(labels, 3).float()
    # Changing only block 1's parameters cannot alter block 0's local gradient
    # when settled states/duals are fixed.
    with torch.no_grad():
        model.weights[0].add_(.3)
    independent_loss = model.energy(x, target, result.states, duals=result.duals).mean()
    changed = torch.autograd.grad(independent_loss, list(model.parameters()))
    names = [n for n, _ in model.named_parameters()]
    torch.testing.assert_close(grads[names.index('input.weight')], changed[names.index('input.weight')])


def test_nonfinite_settings_and_expired_budget_fail():
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=4, depth=2))
    x, target = torch.randn(2, 4), torch.randn(2, 3)
    with pytest.raises(ValueError):
        model.relax(x, target, state_lr=float('nan'))
    def expired():
        raise TimeoutError('budget expired')
    with pytest.raises(TimeoutError):
        model.relax(x, target, check_budget=expired)


def test_energy_probe_features_ignore_evaluation_labels():
    from energy_real.local_energy_analysis import features
    from energy_real.local_energy_data import Dataset
    from energy_real.local_energy_study import FitConfig
    class Unlimited:
        def check(self):
            pass
    torch.manual_seed(93)
    model = LocalEnergyNetwork(LocalConfig(4, 3, width=4, depth=2))
    x = torch.randn(4, 4)
    a = Dataset(x, torch.tensor([0, 1, 2, 0]), list('abcd'), ['test'] * 4)
    b = Dataset(x, torch.tensor([2, 0, 1, 2]), list('abcd'), ['test'] * 4)
    config = FitConfig('graph', 'energy', 11, 4, 2, False, 2, 2, 8)
    torch.testing.assert_close(features(model, a, config, Unlimited()), features(model, b, config, Unlimited()), rtol=0, atol=0)
