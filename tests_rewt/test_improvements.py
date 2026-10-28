from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from rewt.data import task_batch
from rewt.experiment import curriculum_batch, initialization_noise, new_model, train
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.optim import GradientFilter
from rewt.tasks import corrupt_trace, generate_tasks, process_labels
from tests_rewt.test_experiment import TinyCorpus,plan_for
from rewt.improve import nonoverlapping_offsets


@pytest.mark.parametrize("method", ["clip", "ema", "fourier", "ema_residual", "fourier_residual"])
def test_large_finite_gradients_are_clipped_without_norm_or_fft_power_overflow(method):
    p=torch.nn.Parameter(torch.zeros(3))
    filtering=GradientFilter([p],method=method,history=8)
    gradient=torch.tensor([1e20,-2e20,3e20])
    assert not torch.isfinite(gradient.norm())  # Reproduces the failed-run reduction.
    expected=(gradient.double()/gradient.double().norm()).float()
    for _ in range(8):
        p.grad=gradient.clone()
        report=filtering.apply()
        assert math.isfinite(report['raw_gradient_norm'])
        assert report['applied_gradient_norm']<=1.000001
        if report['high_frequency_fraction'] is not None:
            assert math.isfinite(report['high_frequency_fraction'])
        torch.testing.assert_close(p.grad,expected,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize("value", [float('inf'),float('nan')])
def test_actual_nonfinite_gradient_components_are_rejected(value):
    p=torch.nn.Parameter(torch.zeros(3));p.grad=torch.tensor([0.,value,1.])
    with pytest.raises(FloatingPointError,match='component'):
        GradientFilter([p]).apply()


def test_temporal_gradient_boundary_retains_full_forward_memory():
    torch.manual_seed(720)
    model=improved(support=True).train()
    tokens=torch.randint(256,(2,12));noise=torch.randn(2,12,2,8)
    state=model.initial_state(2)
    state.joint=torch.randn_like(state.joint,requires_grad=True)
    whole=model(tokens,state,noise=noise)
    truncated=model(tokens,state,noise=noise,state_gradient_interval=4)
    torch.testing.assert_close(whole.logits,truncated.logits,rtol=0,atol=0)
    torch.testing.assert_close(whole.state.joint,truncated.state.joint,rtol=0,atol=0)
    assert truncated.state.count==12
    full_credit=torch.autograd.grad(whole.logits[:,-1].square().sum(),state.joint)[0]
    truncated_credit=torch.autograd.grad(truncated.logits[:,-1].square().sum(),state.joint,allow_unused=True)[0]
    assert full_credit.abs().sum()>0
    assert truncated_credit is None or truncated_credit.abs().max()==0
    model.eval()
    with torch.inference_mode():
        a=model(tokens,noise=noise)
        b=model(tokens,noise=noise,state_gradient_interval=4)
    torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)


def improved(**kwargs):
    torch.set_num_threads(1)
    return JointEnergyLM(EnergyConfig(width=8, hidden=16, modes=4, history=4,
        auxiliary_hidden=8, horizons=(1, 2), output_geometry="centered",
        normalize_frequency_blocks=True, cache_history=True, early_backtrack=True, **kwargs))


@pytest.mark.parametrize("support,activation", [(False, "fourier"), (False, "mlp"), (True, "fourier")])
def test_centered_energy_gradient_and_shift_invariance(support, activation):
    torch.manual_seed(640)
    model = improved(support=support, activation=activation).double()
    with torch.no_grad():
        model.raw_precision.normal_()  # Unequal coordinate precision needs the full projection chain rule.
    cond, _ = model.condition(torch.tensor([7, 9]), model.initial_state(2))
    candidate = torch.randn(2, model.dimensions, dtype=torch.double, requires_grad=True)
    expected = torch.autograd.grad(model.energy(candidate, cond).sum(), candidate)[0]
    torch.testing.assert_close(model.gradient(candidate, cond), expected, atol=2e-9, rtol=2e-9)
    shifted = candidate.detach().clone()
    shifted[:, model.blocks["output"]] += 31
    torch.testing.assert_close(model.energy(candidate, cond), model.energy(shifted, cond), atol=1e-9, rtol=1e-9)
    gradient = model.gradient(candidate, cond)[:, model.blocks["output"]]
    assert gradient.mean(-1).abs().max() < 1e-12
    revised, _, trace = model.refine(candidate, cond, rounds=8, trace=True)
    assert revised[:, model.blocks["output"]].mean(-1).abs().max() < 1e-12
    trajectory = torch.stack(trace, -1)
    assert (trajectory[:, 1:] <= trajectory[:, :-1] + 1e-9).all()


def test_history_cache_is_causal_and_matches_uncached_and_chunked_calls():
    torch.manual_seed(641)
    fast = improved(support=True).eval()
    reference = JointEnergyLM(replace(fast.config, cache_history=False)).eval()
    reference.load_state_dict(fast.state_dict())
    tokens = torch.randint(256, (2, 9))
    noise = torch.randn(2, 9, 2, 8)
    with torch.inference_mode():
        whole = fast(tokens, noise=noise)
        expected = reference(tokens, noise=noise)
        first = fast(tokens[:, :4], noise=noise[:, :4])
        second = fast(tokens[:, 4:], first.state, noise=noise[:, 4:])
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 19) % 256
        alternate = fast(changed, noise=noise)
    torch.testing.assert_close(whole.logits, expected.logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(whole.logits, torch.cat((first.logits, second.logits), 1), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(whole.logits[:, :4], alternate.logits[:, :4], atol=0, rtol=0)
    targets=torch.randint(256,(2,9))
    for model in (fast,reference):
        logits=model(tokens,noise=noise).logits
        torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten()).backward()
    for (name,a),(_,b) in zip(fast.named_parameters(),reference.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5,msg=name)


def test_early_backtracking_preserves_updates_and_reduces_energy_calls():
    torch.manual_seed(642)
    fast = improved(backtracks=6).double()
    slow = JointEnergyLM(replace(fast.config, early_backtrack=False)).double()
    slow.load_state_dict(fast.state_dict())
    cond, _ = fast.condition(torch.tensor([3, 2]), fast.initial_state(2))
    candidate = fast.canonical_state(torch.randn(2, fast.dimensions, dtype=torch.double))
    actual, fast_cost, _ = fast.refine(candidate, cond, rounds=3)
    expected, slow_cost, _ = slow.refine(candidate, cond, rounds=3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fast_cost["energy_calls"] < slow_cost["energy_calls"]


def test_corruption_is_label_free_reproducible_and_shared_between_core_and_full():
    core, full = improved(), improved(support=True)
    tokens = torch.randint(256, (2, 6))
    recipe = dict(corruption_probability=1., output_noise=.5, state_noise=.05)
    a, fraction = initialization_noise(core, tokens, seed=643, recipe=recipe)
    b, _ = initialization_noise(full, tokens, seed=643, recipe=recipe)
    repeat, _ = initialization_noise(core, tokens, seed=643, recipe=recipe)
    torch.testing.assert_close(a, repeat, rtol=0, atol=0)
    for name in core.blocks:
        torch.testing.assert_close(a[..., core.blocks[name]], b[..., full.blocks[name]], rtol=0, atol=0)
    assert fraction == 1
    assert not torch.equal(core(tokens).logits, core(tokens, initial_noise=a).logits)
    with pytest.raises(TypeError):
        core(tokens, future_targets=tokens)


def test_first_invalid_step_labels_align_with_unchanged_teacher_forced_prefix():
    tasks = generate_tasks("train", 60, number_range=20)
    batch = task_batch(tasks, range(len(tasks)), process_supervision=True, process_seed=644)
    assert (batch.process_negatives >= 0).sum() == len(tasks)
    for index, task in enumerate(tasks):
        wrong = corrupt_trace(task, seed=644)
        assert process_labels(task, task.response)["first_error"] is None
        assert wrong["labels"][-1] == -1
        assert len(wrong["labels"]) == wrong["first_error"] + 1
        assert len(wrong["response"]) == len(task.response)
        offset = wrong["response_byte_offset"]
        assert wrong["response"][:offset] == task.response[:offset]
        at = len(task.prompt) + offset - 1
        assert batch.process_negatives[index, at] == ord(wrong["response"][offset])
        assert batch.targets[index, at] == ord(task.response[offset])
        assert batch.valid[index, at]
    for stage in (1, 2, 3):
        chosen = curriculum_batch(tasks, np.random.default_rng(645), 9, stage=stage,
                                  process_seed=645, process_supervision=True)
        assert (chosen.process_negatives >= 0).sum() == 9


@pytest.mark.parametrize("method", ["fourier_residual", "ema_residual"])
def test_residual_filter_keeps_raw_gradient_and_can_resume_exactly(method):
    p = torch.nn.Parameter(torch.zeros(4))
    filtering = GradientFilter([p], method=method, history=8, residual_gain=.25,
                               ema_decay=0, max_norm=100)
    for t in range(7):
        p.grad = torch.full_like(p, 2 + (-1) ** t)
        filtering.apply()
    q = torch.nn.Parameter(torch.zeros(4))
    resumed = GradientFilter([q], method=method, history=8, residual_gain=.25,
                            ema_decay=0, max_norm=100)
    resumed.load_state_dict(filtering.state_dict())
    p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    filtering.apply()
    resumed.apply()
    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    torch.testing.assert_close(p.grad, torch.full_like(p, 1.5 if method == "fourier_residual" else 1.25))


def test_repair_hard_negative_and_process_losses_train_through_centered_energy():
    torch.manual_seed(646)
    model = improved(support=True)
    tokens, targets = torch.randint(256, (2, 8)), torch.randint(256, (2, 8))
    negatives = (targets + 1) % 256
    output = model(tokens, initial_noise=torch.randn(2, 8, model.dimensions)*.2, details=True,
                   trace_energy=False)
    loss = model.ranking_loss(tokens, targets, output, torch.ones_like(targets,dtype=torch.bool), hard_negatives=True)
    loss += model.repair_loss(output, targets, torch.ones_like(targets,dtype=torch.bool))
    loss += model.process_loss(output, targets, negatives)
    loss.backward()
    for prefix in ("raw_frequency", "raw_precision", "amplitude", "output_prior", "prior"):
        grads = [p.grad for name,p in model.named_parameters() if name == prefix or name.startswith(prefix+".")]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_halting_is_bounded_retention_and_never_a_higher_energy_update():
    torch.manual_seed(647)
    model=improved(halting_tolerance=100.,minimum_refinement_rounds=1).double()
    cond,_=model.condition(torch.tensor([7,1]),model.initial_state(2))
    initial=model.canonical_state(torch.randn(2,model.dimensions,dtype=torch.double))
    final,cost,trace=model.refine(initial,cond,rounds=8,trace=True)
    expected,_,_=model.refine(initial,cond,rounds=1)
    torch.testing.assert_close(final,expected,rtol=0,atol=0)
    assert cost['gradient_calls']==1 and len(trace)==9
    energy=torch.stack(trace,-1)
    assert (energy[:,1:]<=energy[:,:-1]+1e-9).all()


def test_revised_recipe_and_residual_filter_resume_the_same_optimizer_path(tmp_path):
    torch.manual_seed(648)
    config=improved().config
    plan=plan_for(tmp_path)
    plan['settings']['budget_milestones']=[2,8]
    recipe=dict(variable_rounds=True,hard_negatives=True,repair_weight=.1,
                corruption_probability=.5,trace_training_energy=False,residual_gain=.25)
    full=new_model('fourier_rewt',config.__dict__,649)
    full,summary=train(full,'fourier_rewt',649,'language',plan,TinyCorpus(),tmp_path/'full',
                       updates=8,recipe=recipe,method='fourier_residual')
    part=new_model('fourier_rewt',config.__dict__,649)
    train(part,'fourier_rewt',649,'language',plan,TinyCorpus(),tmp_path/'part',
          updates=4,recipe=recipe,method='fourier_residual')
    (tmp_path/'part'/'training.json').unlink()
    resumed=new_model('fourier_rewt',config.__dict__,123)
    resumed,other=train(resumed,'fourier_rewt',649,'language',plan,TinyCorpus(),tmp_path/'part',
                        updates=8,recipe=recipe,method='fourier_residual')
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)
    assert summary['best_step']==other['best_step']
    assert (tmp_path/'full'/'budget_2.pt').exists() and (tmp_path/'full'/'budget_8.pt').exists()


def test_fresh_windows_and_task_exclusions_do_not_reuse_previous_evaluation_cases():
    class Offsets:
        def offsets(self,*args):
            return list(range(1000,5000,100))
    fresh=nonoverlapping_offsets(Offsets(),'test',4,128,650,[1000,1400])
    assert all(all(abs(x-old)>128 for old in (1000,1400)) for x in fresh)
    assert all(abs(x-y)>128 for i,x in enumerate(fresh) for y in fresh[i+1:])
    old=generate_tasks('test',30)
    new=generate_tasks('test',60,seed=1729,number_range=20,exclude_ids=[t.id for t in old])
    assert not {t.id for t in old}&{t.id for t in new}
