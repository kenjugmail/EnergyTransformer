import pytest
import torch

from energy_real.model import Config, LM
from energy_real.static_cache import allocated_state_bytes, with_static_cache


def model():
    torch.manual_seed(710)
    return LM(Config(kind='softmax', width=16, layers=2, heads=2, ff_width=32)).eval()


@pytest.mark.parametrize('segmented', [False, True])
@torch.no_grad()
def test_static_cache_matches_full_and_chunked_attention(segmented):
    original = model()
    static = with_static_cache(original, 41)
    x = torch.randint(256, (2, 41))
    reset = torch.zeros_like(x, dtype=torch.bool) if segmented else None
    if segmented:
        reset[0, 6] = reset[1, 17] = True
    expected, _ = original(x, reset=reset)
    state, parts, pointers = None, [], None
    for lo, hi in ((0, 4), (4, 5), (5, 17), (17, 18), (18, 41)):
        output, state = static(x[:, lo:hi], state, None if reset is None else reset[:, lo:hi])
        parts.append(output)
        current = [a.data_ptr() for layer in state for a in layer]
        if pointers is not None:
            assert current == pointers
        pointers = current
    torch.testing.assert_close(torch.cat(parts, 1), expected, atol=3e-6, rtol=3e-6)
    assert allocated_state_bytes(static, state) == 2 * 2 * 41 * (16 * 2 * 4 + 8)
    with pytest.raises(ValueError, match='capacity'):
        static(x[:, :1], state)
    assert all(block.mixer.length == 41 for block in static.blocks)
    fresh, _ = static(x[:, :7])
    reference, _ = original(x[:, :7])
    torch.testing.assert_close(fresh, reference)
    assert static.state_dict().keys() == original.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, static.state_dict()[key])


def test_static_cache_rejects_gradient_and_stale_state():
    static = with_static_cache(model(), 10)
    x = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(RuntimeError, match='disabled gradients'):
        static(x)
    with torch.no_grad():
        _, first = static(x)
        static(x, first)
        with pytest.raises(ValueError, match='stale'):
            static(x, first)


@torch.no_grad()
def test_reset_prefix_remains_isolated_when_later_calls_omit_reset():
    original = model()
    static = with_static_cache(original, 12)
    x = torch.randint(256, (2, 12))
    resets = torch.zeros_like(x, dtype=torch.bool)
    resets[0, 3] = resets[1, 5] = True
    expected, _ = original(x, reset=resets)
    _, state = static(x[:, :6], reset=resets[:, :6])
    parts = []
    for index in range(6, 12):
        logits, state = static(x[:, index:index+1], state)
        parts.append(logits)
    torch.testing.assert_close(torch.cat(parts, 1), expected[:, 6:], atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        with_static_cache(model(), capacity)
