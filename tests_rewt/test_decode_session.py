from dataclasses import replace

import pytest
import torch

from rewt.decode import EnergyDecodeSession
from rewt.memory import MemoryConfig, MemoryEnergyLM
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM


def make_model(positions=True, kernel=3, history=8, backend='krylov2'):
    cls = PositionMemoryEnergyLM if positions else MemoryEnergyLM
    cfg = PositionMemoryConfig if positions else MemoryConfig
    options = dict(width=8, hidden=17, modes=5, history=history, context_kernel=kernel,
                   recurrent_type='gru', output_geometry='centered', attention_heads=2,
                   compatibility_geometry='anchor_relative', wave_geometry='anchor_relative',
                   initial_carry_scale=0., cache_history=True, refinement_backend=backend)
    if positions:
        options['position_period'] = max(8, history)
    return cls(cfg(**options)).double().eval()


@pytest.mark.parametrize('positions,kernel,history', [(True, 3, 8), (False, 3, 8),
                                                     (True, 1, 8), (False, 1, 1)])
@pytest.mark.parametrize('backend', ['reference', 'krylov2'])
@pytest.mark.parametrize('prompt_length', [1, 19])
def test_decode_cache_matches_forward_and_rollover(positions, kernel, history, backend, prompt_length):
    torch.set_num_threads(1)
    torch.manual_seed(22107)
    model = make_model(positions, kernel, history, backend)
    tokens = torch.randint(0, 256, (2, prompt_length + 19))
    session = EnergyDecodeSession(model)
    with torch.inference_mode():
        expected = model(tokens)
        first = session.prefill(tokens[:, :prompt_length])
        pieces = [first.logits]
        for t in range(prompt_length, tokens.shape[1]):
            out = session.step(tokens[:, t:t + 1])
            pieces.append(out.logits)
        actual = torch.cat(pieces, 1)
        torch.testing.assert_close(actual, expected.logits, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(out.state.joint, expected.state.joint, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(out.state.observed, expected.state.observed, rtol=0, atol=0)
        assert out.state.count == tokens.shape[1]
        assert session.keys.shape[1] == history
        # Prefill is an explicit reset, including positional offset and caches.
        again = session.prefill(tokens[:, :prompt_length])
        torch.testing.assert_close(again.logits, first.logits, rtol=0, atol=0)


@pytest.mark.parametrize('change', ['parameter', 'buffer', 'dtype', 'mode', 'config'])
def test_decode_cache_rejects_mutated_models(change):
    model = make_model()
    session = EnergyDecodeSession(model)
    session.prefill(torch.tensor([[1, 2, 3]]))
    if change == 'parameter':
        with torch.no_grad():
            model.decoder.weight.add_(.01)
    elif change == 'buffer':
        model.mode_mask.zero_()
    elif change == 'dtype':
        model.float()
    elif change == 'mode':
        model.train()
    else:
        model.config = replace(model.config, rounds=1)
    with pytest.raises(RuntimeError, match='invalidated'):
        session.step(torch.tensor([[4]]))
    with pytest.raises(RuntimeError, match='invalidated'):
        session.prefill(torch.tensor([[4]]))


def test_decode_round_intervention_and_invalid_input():
    model = make_model()
    session = EnergyDecodeSession(model, rounds=0)
    with pytest.raises(RuntimeError, match='prefill'):
        session.step(torch.tensor([[1]]))
    with torch.inference_mode():
        a = model(torch.tensor([[1, 2, 3]]), rounds=0)
        b = session.prefill(torch.tensor([[1, 2]]))
        b = session.step(torch.tensor([[3]]))
    torch.testing.assert_close(a.logits[:, -1:], b.logits, rtol=1e-10, atol=1e-11)
    assert b.details['refinement_backend'] == 'reference'
    with pytest.raises(ValueError, match='0..255'):
        session.step(torch.tensor([[256]]))
    with pytest.raises(ValueError, match='one int64 byte'):
        session.step(torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match='eval'):
        EnergyDecodeSession(model.train())
