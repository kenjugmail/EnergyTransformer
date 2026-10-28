from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from rewt.lfm_conversion import (ConversionConfig, ConversionEnergy, ConvertedDeltaMemory,
                                  ConvertedLM, WindowAttention, load_converted)


def config(**kwargs):
    values = dict(vocab_size=31, width=16, hidden=24, heads=2, kv_heads=1, head_dim=8,
                  donor_layers=('conv', 'full_attention', 'sliding_attention'), window=8,
                  conv_width=3, chunk=4, modes=4, pad_token_id=0, eos_token_id=2)
    values.update(kwargs)
    return ConversionConfig(**values)


@pytest.mark.parametrize('window,conv_width', [(8, 3), (1, 1)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_converted_model_causal_chunk_parity_and_bounded_owned_state(window, conv_width, dtype):
    torch.manual_seed(271101)
    c = config(window=window, conv_width=conv_width)
    model = ConvertedLM(c).to(dtype).eval()
    tokens = torch.randint(c.vocab_size, (2, 43))
    with torch.inference_mode():
        expected, _ = model(tokens)
        changed = tokens.clone(); changed[:, 21:] = (changed[:, 21:]+1) % c.vocab_size
        altered, _ = model(changed)
        torch.testing.assert_close(expected[:, :21], altered[:, :21], atol=1e-6, rtol=1e-5)
        pieces = []; state = None
        for first, last in [(0, 1), (1, 13), (13, 14), (14, 29), (29, 43)]:
            output, state = model(tokens[:, first:last], state, return_state=True)
            pieces.append(output)
        # The deliberately unscaled random embedding amplifies FP32 reduction
        # differences. Float64 separately checks the algebra through wraps.
        tolerance = 3e-5 if dtype == torch.float32 else 5e-11
        torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=tolerance, rtol=tolerance)
        previous = [(a.clone(), b.clone()) for a, b in state.layers]
        extra, new_state = model(tokens[:, :1], state, return_state=True)
        for old, preserved in zip(state.layers, previous):
            for a, b in zip(old, preserved):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert state.allocated_bytes() == new_state.allocated_bytes()
        assert new_state.position == 44
        assert state.layers[1][0].shape[1] == c.kv_heads


def test_converted_delta_matches_independent_sequential_values_and_gradients():
    torch.manual_seed(271102)
    c = config(); module = ConvertedDeltaMemory(c).double()
    x = torch.randn(2, 17, c.width, dtype=torch.float64, requires_grad=True)
    output, _ = module(x)
    features = F.silu(module.conv(F.pad(x.transpose(1, 2), (c.conv_width-1, 0))).transpose(1, 2))
    q, k, v = [p(features).view(2, 17, c.heads, c.head_dim) for p in (module.q_proj, module.k_proj, module.v_proj)]
    q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
    beta, decay = module.gates(features).view(2, 17, 2, c.heads).unbind(2)
    beta, decay = beta.sigmoid(), .9+.1*decay.sigmoid()
    memory = x.new_zeros(2, c.heads, c.head_dim, c.head_dim); rows = []
    for t in range(17):
        memory = decay[:, t, :, None, None]*memory
        error = v[:, t]-torch.einsum('bhd,bhde->bhe', k[:, t], memory)
        memory = memory+beta[:, t, :, None, None]*k[:, t, :, :, None]*error[:, :, None, :]
        rows.append(torch.einsum('bhd,bhde->bhe', q[:, t], memory))
    read = torch.stack(rows, 1)
    expected = module.o_proj(module.norm(read).flatten(2)*F.silu(module.gate_proj(features)))
    torch.testing.assert_close(output, expected, atol=1e-11, rtol=1e-10)
    parameters = (x, module.q_proj.weight, module.k_proj.weight, module.v_proj.weight, module.gates.weight)
    probe = torch.randn_like(output)
    actual_grad = torch.autograd.grad((output*probe).sum(), parameters, retain_graph=True)
    expected_grad = torch.autograd.grad((expected*probe).sum(), parameters)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)


def test_energy_identity_initialization_can_learn():
    energy = ConversionEnergy(config())
    x = torch.randn(2, 7, 16)
    output = energy(x)
    torch.testing.assert_close(output, x, atol=0, rtol=0)
    (output*torch.randn_like(output)).sum().backward()
    assert torch.isfinite(energy.amplitude.weight.grad).all()
    assert energy.amplitude.weight.grad.abs().max() > 0
    with torch.no_grad():
        energy.amplitude.weight.add_(-.01*energy.amplitude.weight.grad)
    assert not torch.equal(energy(x), x)


def test_conversion_state_validation_and_standalone_reload(tmp_path):
    model = ConvertedLM(config()).eval()
    tokens = torch.randint(31, (1, 13))
    with torch.no_grad():
        expected, state = model(tokens, return_state=True)
    path = tmp_path/'student.pt'; model.save(path, metadata={'step': 4})
    loaded, metadata = load_converted(path)
    loaded.eval()
    with torch.no_grad():
        actual, _ = loaded(tokens)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert metadata == {'step': 4}
    with pytest.raises(ValueError, match='another model'):
        loaded(tokens[:, :1], state)
    with torch.no_grad():
        model.energy.phase.add_(.1)
    with pytest.raises(ValueError, match='weights changed'):
        model(tokens[:, :1], state)
    with pytest.raises(ValueError, match='vocabulary'):
        model(torch.tensor([[31]]))


def test_freezing_and_checkpointed_training_preserve_gradients():
    torch.manual_seed(271103)
    model = ConvertedLM(config()).train()
    model.trainable_conversion_parameters()
    tokens = torch.randint(31, (1, 13))
    logits, _ = model(tokens); logits.square().mean().backward()
    before = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True); model.gradient_checkpointing = True
    logits2, _ = model(tokens); logits2.square().mean().backward()
    torch.testing.assert_close(logits, logits2, atol=0, rtol=0)
    for n, p in model.named_parameters():
        if n in before:
            torch.testing.assert_close(before[n], p.grad, atol=1e-7, rtol=1e-5)
    assert model.embedding.weight.grad is None
    assert model.blocks[0].feed_forward.w1.weight.grad is None
    assert model.blocks[0].mixer.gates.weight.grad is not None
    with pytest.raises(ValueError, match='uncached'):
        model(tokens, return_state=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_bfloat16_autocast_training_and_float32_memory():
    torch.manual_seed(271104)
    model = ConvertedLM(config()).cuda().to(torch.bfloat16)
    model.trainable_conversion_parameters()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.float()
    tokens = torch.randint(31, (2, 13), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output, state = model(tokens, return_state=True)
        loss = F.cross_entropy(output.flatten(0, 1), tokens.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert state.layers[0][1].dtype == torch.float32
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_config_rejects_incompatible_architecture():
    with pytest.raises(ValueError, match='head dimensions'):
        ConvertedLM(config(heads=3))
    with pytest.raises(ValueError, match='positive'):
        ConvertedLM(config(window=0))
    with pytest.raises(ValueError, match='unsupported donor'):
        ConvertedLM(config(donor_layers=('unknown',)))
