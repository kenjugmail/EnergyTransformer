import pytest
import torch
from torch.nn import functional as F

from rewt.fast_energy import optimize_energy
from rewt.refinement_training import refinement_loss
from rewt.streaming import StreamingConfig, StreamingLM


def model(conditioned=False):
    torch.manual_seed(1729)
    return StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4, vocab_size=517,
                                      conditioned_phase=conditioned))


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('conditioned', [False, True])
def test_trajectory_agrees_with_inference_at_every_depth_and_has_causal_gradients(optimized, conditioned):
    m = model(conditioned)
    if optimized:
        optimize_energy(m)
    tokens = torch.randint(517, (2, 9))
    states = m.refinement_states(tokens, 8)
    for step in (0, 1, 2, 4, 8):
        torch.testing.assert_close(F.linear(states[step], m.embedding.weight), m(tokens, rounds=step)[0])
    changed = tokens.clone(); changed[:, 5:] = (changed[:, 5:] + 3) % 517
    torch.testing.assert_close(m.refinement_states(changed, 8)[-1][:, :5], states[-1][:, :5])
    loss, details = refinement_loss(m, tokens, torch.randint(517, tokens.shape))
    loss.backward()
    assert details['monotonic_penalty'] >= 0
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert m.energy_head.amplitude.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('rounds', [2, 8, 32])
@pytest.mark.parametrize('optimized', [False, True])
def test_conditioned_phase_preserves_energy_descent_and_cached_predictions(rounds, optimized, tmp_path):
    from dataclasses import asdict
    from rewt.streaming import load_streaming
    from rewt.streaming_decode import StreamingSession
    from rewt.fast_decode import FastDecodeSession
    m = model(True).eval()
    tokens = torch.randint(517, (2, 13))
    with torch.inference_mode():
        expected, _, energy = m(tokens, rounds=rounds, details=True)
        assert (energy.diff(dim=-1) <= 2e-6).all()
        path = tmp_path/'conditioned.pt'
        torch.save(dict(arm='streaming_lm', config=asdict(m.config), weights=m.state_dict()), path)
        restored, _ = load_streaming(path)
        torch.testing.assert_close(restored(tokens, rounds=rounds)[0], expected)
        if optimized: optimize_energy(m)
        ordinary = StreamingSession(m, batch=2, capacity=32, rounds=rounds)
        fast = FastDecodeSession(m, batch=2, capacity=32, rounds=rounds, backend='eager')
        torch.testing.assert_close(ordinary.consume(tokens[:, :5]), expected[:, :5])
        fast.consume(tokens[:, :5]); fast.prepare()
        torch.testing.assert_close(ordinary.consume(tokens[:, 5:]), expected[:, 5:], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(fast.consume(tokens[:, 5:]), expected[:, 5:], atol=3e-6, rtol=3e-5)


def test_single_depth_objective_matches_normal_ce():
    m = model(); tokens = torch.randint(517, (2, 9)); targets = torch.randint(517, tokens.shape)
    expected = F.cross_entropy(m(tokens, rounds=4)[0].flatten(0, 1), targets.flatten())
    actual, details = refinement_loss(m, tokens, targets, steps=(4,))
    torch.testing.assert_close(actual, expected)
    assert details['monotonic_penalty'] == 0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_conditioned_single_sequence_single_token_incremental_decode(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    m=model(True).to(device).eval();tokens=torch.tensor([[1,300,516,20]],device=device)
    with torch.inference_mode():
        full=m(tokens,rounds=8)[0];state=None;parts=[]
        for token in tokens.unbind(1):
            logits,state,_=m(token[:,None],state,rounds=8,return_state=True)
            parts.append(logits)
        torch.testing.assert_close(torch.cat(parts,1),full,atol=3e-6,rtol=3e-5)


@pytest.mark.parametrize('steps', [(), (2, 0), (0, 0), (-1, 2), (True, 2)])
def test_invalid_training_depths(steps):
    with pytest.raises(ValueError, match='steps'):
        refinement_loss(model(), torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long), steps=steps)


def test_comparison_metrics_match_existing_word_evaluator():
    from types import SimpleNamespace
    import numpy as np
    from rewt.data import Batch
    from rewt.word_lm import evaluate_words
    from scripts.study_rewt_word_refinement import evaluate_depths
    tokens = torch.tensor([[1, 44, 516, 7, 0, 12, 3, 0, 99, 1]])
    corpus = SimpleNamespace(tokenizer=SimpleNamespace(unk_id=0),
                             word_masks={'test':np.array([0,1,1,1,1,0,1,1,1,0], dtype=np.uint8)})
    corpus.batch = lambda split, rng, count, context, starts: Batch(tokens[:, :context], tokens[:, 1:context+1])
    m = model(True).eval()
    expected = evaluate_words(m, corpus, 'test', [0], 9, rounds=8)
    actual = evaluate_depths(m, corpus, 'test', [0], 9, [8])['8']
    for key in ['token_nll','next_word_accuracy','next_word_top5_accuracy','word_vocabulary_coverage']:
        assert actual[key] == pytest.approx(expected[key], abs=1e-6)
