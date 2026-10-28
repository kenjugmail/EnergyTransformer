from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from rewt.model import (EnergyConfig, JointEnergyLM, TransformerConfig,
                        TransformerLM, matched_configurations, build_model)


@pytest.fixture(autouse=True)
def deterministic_small_runtime():
    torch.set_num_threads(1)
    torch.manual_seed(401)


def tiny(**kwargs):
    return JointEnergyLM(EnergyConfig(width=8, hidden=16, modes=4, history=4,
        auxiliary_hidden=8, horizons=(1, 2), diffusion_samples=2, **kwargs))


@pytest.mark.parametrize("support,activation", [(False, "fourier"), (False, "mlp"), (True, "fourier")])
def test_analytic_derivative_matches_independent_autograd(support, activation):
    model = tiny(support=support, activation=activation).double()
    state = model.initial_state(2)
    cond, _ = model.condition(torch.tensor([3, 8]), state)
    candidate = torch.randn_like(state.joint).requires_grad_()
    expected = torch.autograd.grad(model.energy(candidate, cond).sum(), candidate)[0]
    actual = model.gradient(candidate, cond)
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("schedule", ["all", "coarse_to_fine"])
def test_backtracking_never_accepts_higher_full_energy(schedule):
    model = tiny(support=True, step_size=100., backtracks=12, schedule=schedule)
    state = model.initial_state(3)
    cond, _ = model.condition(torch.tensor([2, 7, 4]), state)
    candidate = torch.randn_like(state.joint) * 3
    refined, _, trace = model.refine(candidate, cond, rounds=8, trace=True)
    energy = torch.stack(trace, -1)
    assert (energy[:, 1:] <= energy[:, :-1] + 1e-5).all()
    assert (refined != candidate).any()


def test_inference_weights_are_fixed_and_forecasts_are_not_observed_bytes():
    model = tiny(support=True).eval()
    tokens = torch.randint(256, (2, 5))
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.inference_mode():
        output = model(tokens, details=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    torch.testing.assert_close(output.state.observed, model.embedding(tokens[:, -4:]), rtol=0, atol=0)
    assert output.state.count == 5
    with pytest.raises(TypeError):
        model(tokens, future_targets=tokens)


def test_future_changes_cannot_change_prefix_predictions():
    model = tiny(support=True).eval()
    tokens = torch.randint(256, (2, 7))
    other = tokens.clone()
    other[:, 4:] = (other[:, 4:] + 91) % 256
    noise = torch.randn(2, 7, 2, 8)
    with torch.inference_mode():
        a = model(tokens, noise=noise).logits
        b = model(other, noise=noise).logits
    torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=0)


def test_chunking_and_document_reset_preserve_predictions():
    model = tiny(support=True).eval()
    tokens = torch.randint(256, (2, 9))
    noise = torch.randn(2, 9, 2, 8)
    with torch.inference_mode():
        whole = model(tokens, noise=noise)
        first = model(tokens[:, :4], noise=noise[:, :4])
        second = model(tokens[:, 4:], first.state, noise=noise[:, 4:])
        reset = model(tokens[:, 4:], noise=noise[:, 4:])
        reset_again = model(tokens[:, 4:], noise=noise[:, 4:])
    torch.testing.assert_close(whole.logits, torch.cat((first.logits, second.logits), 1), rtol=0, atol=0)
    torch.testing.assert_close(reset.logits, reset_again.logits, rtol=0, atol=0)
    assert not torch.equal(reset.logits, second.logits)


def test_every_full_component_receives_training_gradients():
    model = tiny(support=True)
    tokens, targets = torch.randint(256, (2, 6)), torch.randint(256, (2, 6))
    valid = torch.ones_like(targets, dtype=torch.bool)
    output = model(tokens, details=True)
    loss = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
    loss = loss + model.ranking_loss(tokens, targets, output, valid)
    losses = model.auxiliary_losses(output, targets, valid=valid, teacher_hidden=torch.randn(2, 6, 8))
    loss = loss + sum(losses.values())
    loss.backward()
    expected = ["prior", "embedding", "query", "key", "value", "decoder", "output_prior", "amplitude",
                "raw_frequency", "phase", "raw_precision", "recurrent_proposal", "forecast",
                "forecast_bridge", "history_projection", "denoiser", "specialists", "bridge",
                "specialist_links", "router", "critic"]
    for prefix in expected:
        grads = [p.grad for name, p in model.named_parameters() if name == prefix or name.startswith(prefix + ".")]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), prefix
        assert sum(float(g.abs().sum()) for g in grads) > 0, prefix


def test_state_and_output_only_updates_have_correct_domains():
    model = tiny(support=True)
    state = model.initial_state(2)
    cond, _ = model.condition(torch.tensor([7, 9]), state)
    initial = torch.randn_like(state.joint)
    cut = model.blocks["output"].start
    a, _, _ = model.refine(initial, cond, update_output=False)
    b, _, _ = model.refine(initial, cond, update_states=False)
    torch.testing.assert_close(a[:, cut:], initial[:, cut:], rtol=0, atol=0)
    torch.testing.assert_close(b[:, :cut], initial[:, :cut], rtol=0, atol=0)
    assert (a[:, :cut] != initial[:, :cut]).any() and (b[:, cut:] != initial[:, cut:]).any()


def test_removed_workspaces_stay_zero_and_history_ablation_changes_conditioning():
    model = tiny(support=True)
    tokens = torch.randint(256, (2, 5))
    a = model(tokens, details=True, disabled=("forecast", "specialists"))
    assert not a.details["joint"][..., model.blocks["forecast"]].count_nonzero()
    for i in range(model.config.specialists):
        assert not a.details["joint"][..., model.blocks[f"specialist_{i}"]].count_nonzero()
    b = model(tokens)
    c = model(tokens, disabled=("history",))
    assert not torch.equal(b.logits, c.logits)


@pytest.mark.parametrize("removed,prefix", [("forecast", "forecast."), ("diffusion", "denoiser."),
                                           ("critic", "critic."), ("router", "router.")])
def test_disabled_proposal_cannot_reenter_through_communication_anchor(removed, prefix):
    model = tiny(support=True).eval()
    tokens = torch.randint(256, (2, 4))
    with torch.inference_mode():
        before = model(tokens, disabled=(removed,)).logits
        for name, parameter in model.named_parameters():
            if name.startswith(prefix):
                parameter.add_(10)
        after = model(tokens, disabled=(removed,)).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("activation", ["mlp", "fourier"])
def test_long_window_with_high_gain_proposals_remains_finite(activation):
    model = tiny(activation=activation)
    with torch.no_grad():
        model.prior[-1].weight.mul_(20)
    tokens = torch.randint(256, (2, 128))
    output = model(tokens, details=True)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.details["joint"]).all()
    loss = F.cross_entropy(output.logits.flatten(0, 1), tokens.flatten())
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_frequency_is_finite_and_energy_control_counts_match():
    model = tiny()
    with torch.no_grad():
        model.raw_frequency.zero_()
    model(torch.tensor([[1, 2, 3]])).logits.sum().backward()
    assert torch.isfinite(model.raw_frequency.grad).all()
    configs = matched_configurations()
    counts = {arm: build_model(arm, cfg).parameters_count() for arm, cfg in configs.items()}
    assert max(abs(count - 250_000) for count in counts.values()) / 250_000 < .005
    assert counts["neural_energy"] == counts["fourier_rewt"]


def test_transformer_is_causal_and_streaming_matches_batch():
    model = TransformerLM(TransformerConfig(width=8, layers=1, heads=2, feedforward=16, context=16)).eval()
    tokens = torch.randint(256, (2, 8))
    other = tokens.clone()
    other[:, 4:] = (other[:, 4:] + 1) % 256
    with torch.inference_mode():
        whole = model(tokens)
        first = model(tokens[:, :4])
        second = model(tokens[:, 4:], first.state)
        mutated = model(other)
    torch.testing.assert_close(whole.logits, torch.cat((first.logits, second.logits), 1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(whole.logits[:, :4], mutated.logits[:, :4], atol=0, rtol=0)
