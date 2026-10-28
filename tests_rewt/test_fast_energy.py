import copy

import pytest
import torch

from rewt.fast_energy import FastLatentEnergy, optimize_energy
from rewt.streaming import LatentEnergy, StreamingConfig, StreamingLM
from rewt.streaming_decode import StreamingSession


@pytest.mark.parametrize('rounds', [0, 1, 2, 5])
def test_fast_energy_matches_outputs_input_and_weight_gradients(rounds):
    torch.manual_seed(813)
    original = LatentEnergy(StreamingConfig(width=16, heads=2, modes=4)).double()
    fast = FastLatentEnergy(copy.deepcopy(original))
    x = torch.randn(2, 7, 16, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, _ = original(x, rounds)
    actual, _ = fast(y, rounds)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for a, b in zip((x, *original.parameters()), (y, *fast.parameters())):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('rounds', [0, 1, 2, 5])
def test_prepared_projection_matches_reference_and_rejects_stale_weights(rounds):
    torch.manual_seed(19)
    head = LatentEnergy(StreamingConfig(width=16, heads=2, modes=4)).double()
    fast = FastLatentEnergy(head)
    prepared = fast.prepare_inference(rounds)
    x = torch.randn(2, 7, 16, dtype=torch.double)
    with torch.no_grad():
        torch.testing.assert_close(prepared(x), head(x, rounds)[0], atol=1e-12, rtol=1e-11)
        fast.raw_precision.add_(.1)
        with pytest.raises(ValueError, match='weights changed'):
            prepared(x)
    with pytest.raises(ValueError, match='no_grad'):
        prepared(x)


def test_optimizer_checkpoint_trace_and_session_compatibility():
    torch.manual_seed(7)
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4)).eval()
    original = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    optimize_energy(model)
    assert not model.energy_head.training
    assert before_ids == [id(p) for p in model.parameters()]
    assert list(original.state_dict()) == list(model.state_dict())
    x = torch.randint(256, (2, 23))
    with torch.inference_mode():
        expected, _, trace = original(x, details=True)
        actual, _, fast_trace = model(x, details=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(fast_trace, trace, atol=0, rtol=0)
        session = StreamingSession(model, batch=2, capacity=24)
        parts = [session.consume(x[:, :7]), session.consume(x[:, 7:22]), session.consume(x[:, 22:])]
        torch.testing.assert_close(torch.cat(parts, 1), expected, atol=2e-6, rtol=2e-5)
        stored = parts[-1].clone()
        session.consume(x[:, :1])
        torch.testing.assert_close(parts[-1], stored, atol=0, rtol=0)
    # Existing optimizer updates shared parameters, and ordinary checkpoints load.
    model.train(); model(x)[0].square().mean().backward(); optimizer.step()
    original.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(ValueError, match='changed'):
        session.consume(x[:, :1])
    model.eval(); session = StreamingSession(model, batch=2, capacity=24)
    with torch.no_grad():
        model.energy_head.phase.add_(.01)
    with pytest.raises(ValueError, match='weights'):
        session.consume(x[:, :1])


def test_compiled_energy_forward_and_backward_cpu():
    torch.manual_seed(11)
    head = LatentEnergy(StreamingConfig(width=16, heads=2, modes=4))
    fast = FastLatentEnergy(copy.deepcopy(head), compiled=True)
    x = torch.randn(2, 7, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = head(x, 2)[0]; actual = fast(y, 2)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    expected.square().mean().backward(); actual.square().mean().backward()
    for a, b in zip((x, *head.parameters()), (y, *fast.parameters())):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
    with torch.inference_mode():
        prepared = fast.prepare_inference(2)
        torch.testing.assert_close(prepared(x.detach()), expected.detach(), atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('width,modes', [(16, 4), (64, 16), (60, 13), (128, 32)])
def test_fused_cuda_energy_shapes_parity_and_ownership(width, modes):
    pytest.importorskip('triton')
    torch.manual_seed(981)
    head = LatentEnergy(StreamingConfig(width=width, heads=2, modes=modes)).cuda().eval()
    fast = FastLatentEnergy(head, fused=True)
    with torch.inference_mode():
        prepared = fast.prepare_inference(2)
        for length in (1, 27, 3072):
            x = torch.randn(2, length, width, device='cuda')
            actual = prepared(x)
            torch.testing.assert_close(actual, head(x, 2)[0], atol=4e-6, rtol=4e-5)
            saved = actual.clone()
            prepared(x * 2)
            torch.testing.assert_close(saved, actual, atol=0, rtol=0)
        with pytest.raises(ValueError, match='width'):
            prepared(torch.randn(1, 1, width + 1, device='cuda'))
        with pytest.raises(ValueError, match='float32'):
            prepared(x.double())


def test_fused_energy_rejects_unsupported_cpu_preparation():
    head = LatentEnergy(StreamingConfig(width=16, heads=2, modes=4))
    with pytest.raises(ValueError, match='CUDA float32'):
        FastLatentEnergy(head, fused=True).prepare_inference(2)


@pytest.mark.parametrize('length', [1, 16, 17, 128])
def test_cpu_small_batch_boundary_strides_and_output_ownership(length):
    torch.manual_seed(75)
    head = LatentEnergy(StreamingConfig(width=16, heads=2, modes=7))
    fast = FastLatentEnergy(head)
    with torch.no_grad():
        prepared = fast.prepare_inference(2)
        x = torch.randn(1, length * 2, 16)[:, ::2]
        actual = prepared(x); saved = actual.clone()
        torch.testing.assert_close(actual, head(x, 2)[0], atol=2e-6, rtol=2e-5)
        prepared(x + 1)
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)
        assert actual.data_ptr() != x.data_ptr()
