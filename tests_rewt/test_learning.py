from dataclasses import replace

import pytest
import torch

from rewt.data import task_batch
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.tasks import generate_tasks


@pytest.fixture(autouse=True)
def small_runtime():
    torch.set_num_threads(1)
    torch.manual_seed(9271)


def config(**kwargs):
    return EnergyConfig(width=8, hidden=20, modes=4, history=8, auxiliary_hidden=8,
                        horizons=(1, 2), output_geometry="centered",
                        compatibility_geometry="anchor_relative", **kwargs)


@pytest.mark.parametrize("support,activation", [(False, "fourier"), (False, "mlp"), (True, "fourier")])
def test_anchor_relative_quadratic_has_its_minimum_at_the_proposal_and_exact_gradient(support, activation):
    model = JointEnergyLM(config(support=support, activation=activation)).double()
    cond, _ = model.condition(torch.tensor([41, 53]), model.initial_state(2))
    anchor = cond.anchor.detach().requires_grad_()
    # Re-centering float64 logits can leave a squared rounding residual.
    torch.testing.assert_close(model.quadratic(anchor, cond), torch.zeros(2, dtype=torch.double), rtol=0, atol=1e-25)
    g = torch.autograd.grad(model.quadratic(anchor, cond).sum(), anchor)[0]
    torch.testing.assert_close(g, torch.zeros_like(g), rtol=0, atol=1e-13)
    candidate = torch.randn_like(anchor).requires_grad_()
    expected = torch.autograd.grad(model.energy(candidate, cond).sum(), candidate)[0]
    torch.testing.assert_close(model.gradient(candidate, cond), expected, rtol=2e-9, atol=2e-9)


def test_contextual_memory_is_causal_cached_and_chunkable_after_history_rollover():
    c = config(context_kernel=3, recurrent_type="gru", cache_history=True)
    cached, direct = JointEnergyLM(c).double().eval(), JointEnergyLM(replace(c, cache_history=False)).double().eval()
    direct.load_state_dict(cached.state_dict())
    tokens = torch.randint(256, (2, 24))
    changed = tokens.clone()
    changed[:, 11:] = (changed[:, 11:] + 71) % 256
    with torch.inference_mode():
        whole = cached(tokens, details=True)
        alternative = cached(changed)
        direct_output = direct(tokens)
        first = cached(tokens[:, :7])
        second = cached(tokens[:, 7:18], state=first.state)
        third = cached(tokens[:, 18:], state=second.state)
    torch.testing.assert_close(whole.logits[:, :11], alternative.logits[:, :11], rtol=0, atol=0)
    torch.testing.assert_close(whole.logits, direct_output.logits, rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(whole.logits, torch.cat((first.logits, second.logits, third.logits), 1), rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(whole.state.observed, cached.embedding(tokens[:, -10:]), rtol=0, atol=0)
    assert (whole.details['energy_trace'][..., 1:] <= whole.details['energy_trace'][..., :-1]+1e-9).all()


def test_contextual_keys_distinguish_equal_bytes_with_different_preceding_context():
    model = JointEnergyLM(config(context_kernel=3))
    tokens = torch.tensor([list(b"K=3;X=3")])
    encoded = model.encode_observed(model.embedding(tokens))
    assert not torch.allclose(encoded[:, 2], encoded[:, 6])
    torch.testing.assert_close(model.embedding(tokens)[:, 2], model.embedding(tokens)[:, 6], rtol=0, atol=0)


def test_gated_carry_retains_interior_values_without_editing_the_joint_state():
    model = JointEnergyLM(config(recurrent_type="gru"))
    state = model.initial_state(2)
    state.joint[:, model.blocks['recurrent']] = .7
    before = state.joint.clone()
    retained = model.recurrent_condition(state)
    torch.testing.assert_close(retained, torch.full_like(retained, .7), rtol=0, atol=0)
    model.condition(torch.tensor([2, 3]), state)
    torch.testing.assert_close(state.joint, before, rtol=0, atol=0)


@pytest.mark.parametrize('support', [False, True])
@pytest.mark.parametrize('activation', ['fourier', 'mlp'])
def test_relative_waves_have_exact_gradient_and_ignore_common_anchor_translation(support, activation):
    model = JointEnergyLM(config(support=support, activation=activation,
        wave_geometry='anchor_relative',
        initial_carry_scale=0.)).double().eval()
    state = model.initial_state(2)
    cond, _ = model.condition(torch.tensor([2, 3]), state)
    candidate = torch.randn_like(cond.anchor).requires_grad_()
    expected = torch.autograd.grad(model.energy(candidate, cond).sum(), candidate)[0]
    torch.testing.assert_close(model.gradient(candidate, cond), expected, rtol=2e-9, atol=2e-9)
    displacement = model.canonical_state(torch.randn_like(candidate))
    shifted = replace(cond, anchor=cond.anchor + displacement)
    torch.testing.assert_close(model.energy(candidate + displacement, shifted),
                               model.energy(candidate, cond), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(model.gradient(candidate + displacement, shifted),
                               model.gradient(candidate, cond), rtol=2e-12, atol=2e-12)


def test_zero_extra_carry_initializes_at_conditioned_anchor_and_refines_one_fixed_energy():
    model = JointEnergyLM(config(recurrent_type='gru', context_kernel=3,
        wave_geometry='anchor_relative',
        initial_carry_scale=0.)).double().eval()
    output = model(torch.randint(256, (2, 16)), details=True)
    torch.testing.assert_close(output.details['initial_joint'],
                               output.details['conditioning'].anchor, rtol=0, atol=2e-15)
    energies = output.details['energy_trace']
    assert (energies[..., 1:] <= energies[..., :-1] + 1e-12).all()


@pytest.mark.parametrize('settings', [dict(wave_geometry='unknown'),
    dict(initial_carry_scale=-1.), dict(initial_carry_scale=float('nan'))])
def test_invalid_relative_wave_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        JointEnergyLM(config(**settings))


def test_semantic_weights_emphasize_answer_values_and_leave_prompt_padding_ignored():
    tasks = generate_tasks('train', 6, number_range=20, seed=831)
    ordinary, weighted = task_batch(tasks, range(6)), task_batch(tasks, range(6), semantic_weight=4.)
    torch.testing.assert_close(ordinary.tokens, weighted.tokens, rtol=0, atol=0)
    torch.testing.assert_close(ordinary.targets, weighted.targets, rtol=0, atol=0)
    assert ordinary.token_weights is None
    assert (weighted.token_weights[~weighted.valid] == 0).all()
    assert (weighted.token_weights[weighted.valid] >= 1).all()
    for row, task in enumerate(tasks):
        first = len(task.prompt)-1
        assert (weighted.token_weights[row, first:first+len('steps=')] == 1).all()
        answer = len(task.prompt)+task.response.index(';ans=')+len(';ans=')-1
        assert weighted.token_weights[row, answer] == 4
    torch.testing.assert_close(weighted.to('cpu').token_weights, weighted.token_weights, rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0., -1., float('nan'), float('inf')])
def test_invalid_semantic_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        task_batch(generate_tasks('train', 3), [0], semantic_weight=weight)
