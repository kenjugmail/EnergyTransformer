import copy
from dataclasses import replace
import pytest
import torch
from energy_real.model import Config, LM
from energy_real.kernel import chunk_gla


def test_backend_cannot_switch_on_future_inputs():
    torch.manual_seed(27)
    u = torch.randn(1, 16, 2, 8)
    wk, v, state = torch.randn_like(u), torch.randn(1, 16, 2, 6), torch.randn(1, 2, 8, 6)
    f = torch.full_like(u, .99)
    future = f.clone(); future[:, 12:] = 1e-4
    a, _, _ = chunk_gla(u, wk, v, f.log(), state, 16)
    b, _, _ = chunk_gla(u, wk, v, future.log(), state, 16)
    assert torch.equal(a[:, :12], b[:, :12])
    with pytest.raises(ValueError, match='fixed backend'):
        chunk_gla(u, wk, v, f.log(), state, 16, force='auto')
    with pytest.raises(ValueError, match='validity domain'):
        chunk_gla(u, wk, v, future.log(), state, 16, force='matmul')


@pytest.mark.parametrize('kind', ['energy', 'softmax'])
def test_resets_with_cached_prefix_per_batch_and_immutable_state(kind):
    torch.manual_seed(5)
    model = LM(Config(kind=kind, width=32, layers=2, heads=2, slots=8, ff_width=64, chunk=8)).eval()
    x = torch.randint(256, (2, 33))
    reset = torch.zeros_like(x, dtype=torch.bool)
    reset[0, 8] = True; reset[1, 18] = True; reset[0, 24] = True
    with torch.no_grad():
        whole, _ = model(x, reset=reset)
        first, state = model(x[:, :13], reset=reset[:, :13])
        saved = copy.deepcopy(state)
        second, _ = model(x[:, 13:], state, reset[:, 13:])
        torch.testing.assert_close(whole, torch.cat((first, second), 1), atol=3e-5, rtol=3e-5)
        for b, start in [(0, 24), (1, 18)]:
            fresh, _ = model(x[b:b+1, start:])
            torch.testing.assert_close(whole[b:b+1, start:], fresh, atol=3e-5, rtol=3e-5)
        for old, new in zip(saved, state):
            assert all(torch.equal(a, b) for a, b in zip(old, new))


@pytest.mark.parametrize('updates', [{'heads': 0}, {'heads': -2}, {'width': 0}, {'width': 1.5}, {'chunk': 0}, {'kind': 'delta'}])
def test_invalid_configuration_rejected_before_allocation(updates):
    with pytest.raises(ValueError):
        LM(replace(Config(), **updates))


def test_state_and_reset_shape_validation():
    model = LM(Config(width=32, layers=1, heads=2, ff_width=64))
    x = torch.ones(2, 8, dtype=torch.long)
    for state in [(), ((torch.ones(1), torch.ones(1)),)]:
        with pytest.raises(ValueError):
            model(x, state)
    with pytest.raises(ValueError):
        model(x, reset=torch.ones_like(x))


@pytest.mark.parametrize('backend', ['dmat', 'matmul'])
def test_gradients_include_log_decay_both_reads_and_final_state(backend):
    torch.manual_seed(3)
    shapes = [(1, 7, 2, 3), (1, 7, 2, 3), (1, 7, 2, 4), (1, 7, 2, 3), (1, 2, 3, 4)]
    values = [torch.randn(s, dtype=torch.double) for s in shapes]
    values[3] = -values[3].abs() * .1
    args = tuple(x.requires_grad_() for x in values)
    def fn(*xs):
        from energy_real.kernel import read_again
        y, state, ctx = chunk_gla(*xs, chunk=4, force=backend)
        return y, read_again(ctx, xs[0] * .3), state
    assert torch.autograd.gradcheck(fn, args, fast_mode=True)
