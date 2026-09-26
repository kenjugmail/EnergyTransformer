import copy
from dataclasses import replace

import pytest
import torch

from energy_real.model import Config, LM, SoftmaxMixer


@pytest.mark.parametrize('kind', ['softmax', 'hybrid'])
@pytest.mark.parametrize('window', [1, 7, 64])
def test_window_streaming_reset_causality_and_storage(kind, window):
    torch.manual_seed(182)
    cfg = Config(kind=kind, width=16, layers=3, heads=2, slots=4,
                 ff_width=32, chunk=8, attention_window=window)
    model = LM(cfg).eval()
    x = torch.randint(256, (2, 41))
    reset = torch.zeros_like(x, dtype=torch.bool)
    reset[0, 14] = True
    reset[1, 22] = True
    with torch.no_grad():
        whole, _ = model(x, reset=reset)
        parts, state = [], None
        for lo, hi in [(0, 3), (3, 17), (17, 18), (18, 41)]:
            saved = copy.deepcopy(state)
            y, new = model(x[:, lo:hi], state, reset[:, lo:hi])
            if saved is not None:
                assert all(torch.equal(a, b) for old, original in zip(saved, state)
                           for a, b in zip(old, original))
            parts.append(y)
            state = new
        torch.testing.assert_close(whole, torch.cat(parts, 1), atol=3e-5, rtol=3e-5)
        for b, start in [(0, 14), (1, 22)]:
            fresh, _ = model(x[b:b+1, start:])
            torch.testing.assert_close(whole[b:b+1, start:], fresh, atol=3e-5, rtol=3e-5)
        altered = x.clone()
        altered[:, 27:] = (altered[:, 27:] + 1) % 256
        future, _ = model(altered, reset=reset)
        torch.testing.assert_close(whole[:, :27], future[:, :27], atol=3e-5, rtol=3e-5)
        for index, layer in enumerate(state):
            if cfg.mixer_kind(index) == 'softmax':
                assert layer[0].shape[2] == min(window, 41)
                assert layer[3].eq(41).all()
                assert all(a.untyped_storage().nbytes() == a.numel() * a.element_size() for a in layer)


def test_local_attention_matches_independent_window_reference_and_gradients():
    torch.manual_seed(41)
    cfg = Config(kind='softmax', width=16, heads=2, layers=1, attention_window=5)
    local = SoftmaxMixer(cfg).double()
    full = SoftmaxMixer(replace(cfg, attention_window=0)).double()
    full.load_state_dict(local.state_dict())
    x = torch.randn(1, 13, 16, dtype=torch.double, requires_grad=True)
    y, _ = local(x)
    # Recompute every query using an ordinary full-attention window. RoPE is
    # translation invariant, so restarting its origin preserves the result.
    expected = torch.cat([full(x[:, max(0, i-4):i+1])[0][:, -1:] for i in range(13)], 1)
    torch.testing.assert_close(y, expected, atol=2e-7, rtol=2e-7)
    ga = torch.autograd.grad(y.square().sum(), (x, *local.parameters()))
    gb = torch.autograd.grad(expected.square().sum(), (x, *full.parameters()))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_window_checkpointing_and_memory_reader_composition():
    from energy_real.fact_memory import FactMemory
    torch.manual_seed(90)
    cfg = Config(kind='hybrid', width=16, layers=3, heads=2, slots=4,
                 ff_width=32, attention_window=4, memory_dim=8)
    plain = LM(cfg)
    checkpointed = LM(replace(cfg, activation_checkpointing=True))
    checkpointed.load_state_dict(plain.state_dict())
    memory = FactMemory(torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
    x = torch.randint(256, (2, 11))
    a, _ = plain(x, memory=memory)
    b, _ = checkpointed(x, memory=memory)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for p, q in zip(plain.parameters(), checkpointed.parameters()):
        assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize('kw', [dict(kind='hybrid'), dict(kind='hybrid', attention_window=4, layers=2),
                               dict(attention_window=3), dict(attention_window=-1), dict(attention_every=0)])
def test_invalid_windows(kw):
    with pytest.raises(ValueError):
        LM(Config(**kw))


def test_window_cache_validation():
    model = LM(Config(kind='softmax', width=16, heads=2, layers=1, attention_window=4))
    x = torch.ones(2, 9, dtype=torch.long)
    _, state = model(x)
    bad = (*state[0][:3], torch.zeros(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match='position'):
        model(x, (bad,))
