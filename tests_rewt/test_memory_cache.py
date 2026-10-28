from dataclasses import replace

import pytest
import torch

from rewt.memory import MemoryConfig, MemoryEnergyLM
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM


@pytest.mark.parametrize('positions', [False, True])
@pytest.mark.parametrize('heads,kernel', [(1, 3), (2, 3), (2, 1)])
def test_cached_memory_matches_uncached_predictions_and_parameter_gradients(positions, heads, kernel):
    torch.set_num_threads(1)
    torch.manual_seed(4101)
    cls = PositionMemoryEnergyLM if positions else MemoryEnergyLM
    cfg = PositionMemoryConfig if positions else MemoryConfig
    options = dict(width=8, hidden=13, modes=3, history=8, attention_heads=heads,
                   context_kernel=kernel, recurrent_type='gru', output_geometry='centered',
                   compatibility_geometry='anchor_relative', wave_geometry='anchor_relative',
                   initial_carry_scale=0.)
    if positions:
        options['position_period'] = 8
    plain = cls(cfg(**options)).double()
    cached = cls(replace(plain.config, cache_history=True)).double()
    cached.load_state_dict(plain.state_dict())
    tokens = torch.randint(0, 128, (2, 23))
    a = plain(tokens, details=True, state_gradient_interval=5)
    b = cached(tokens, details=True, state_gradient_interval=5)
    torch.testing.assert_close(a.logits, b.logits, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(a.state.observed, b.state.observed, rtol=0, atol=0)
    torch.testing.assert_close(a.details['features'], b.details['features'], rtol=1e-11, atol=1e-11)
    assert a.details['cost'] == b.details['cost']
    assert (b.details['energy_trace'][..., 1:] <= b.details['energy_trace'][..., :-1] + 1e-12).all()
    a.logits.square().mean().backward()
    b.logits.square().mean().backward()
    for (name, p), (other_name, q) in zip(plain.named_parameters(), cached.named_parameters()):
        assert name == other_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-9, atol=1e-11, msg=name)


@pytest.mark.parametrize('positions', [False, True])
def test_cached_memory_preserves_causality_and_chunk_rollover(positions):
    torch.manual_seed(4102)
    cls = PositionMemoryEnergyLM if positions else MemoryEnergyLM
    cfg = PositionMemoryConfig if positions else MemoryConfig
    options = dict(width=8, hidden=13, modes=3, history=8, attention_heads=2,
                   context_kernel=3, recurrent_type='gru', cache_history=True)
    if positions:
        options['position_period'] = 8
    model = cls(cfg(**options)).double().eval()
    x = torch.randint(0, 128, (2, 29))
    altered = x.clone()
    altered[:, 15:] = (altered[:, 15:] + 11) % 128
    with torch.inference_mode():
        whole = model(x)
        changed = model(altered)
        first = model(x[:, :5])
        second = model(x[:, 5:18], state=first.state)
        last = model(x[:, 18:], state=second.state)
    torch.testing.assert_close(whole.logits[:, :15], changed.logits[:, :15], rtol=0, atol=0)
    torch.testing.assert_close(whole.logits, torch.cat((first.logits, second.logits, last.logits), 1),
                               rtol=1e-11, atol=1e-11)
    assert last.state.count == x.shape[1]
    torch.testing.assert_close(whole.state.observed, last.state.observed, rtol=0, atol=0)
