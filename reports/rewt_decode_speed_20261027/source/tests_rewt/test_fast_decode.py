import pytest
import torch

from rewt.fast_decode import FastDecodeSession
from rewt.streaming import StreamingConfig, StreamingLM
from rewt.streaming_decode import StreamingSession


@pytest.mark.parametrize('hybrid,window', [(False, 0), (False, 8), (True, 8), (True, 1)])
@pytest.mark.parametrize('rounds', [0, 1, 2])
def test_ring_and_manual_delta_match_reference_through_wraps(hybrid, window, rounds):
    torch.manual_seed(819)
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4,
                                       hybrid=hybrid, window=window, rounds=rounds)).eval()
    tokens = torch.randint(256, (2, 61))
    reference = StreamingSession(model, batch=2, capacity=64)
    fast = FastDecodeSession(model, batch=2, capacity=64, backend='eager')
    for a, b in [(0, 3), (3, 13)]:
        torch.testing.assert_close(fast.consume(tokens[:, a:b]), reference.consume(tokens[:, a:b]))
    fast.prepare()
    storage = fast.allocated_state_bytes()
    for a, b in [(13, 14), (14, 29), (29, 61)]:
        expected = reference.consume(tokens[:, a:b])
        actual = fast.consume(tokens[:, a:b])
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    assert fast.allocated_state_bytes() == storage
    assert fast.position == 61


def check_generation(backend, device, hybrid=True, window=8):
    torch.manual_seed(36)
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4,
                                       hybrid=hybrid, window=window)).to(device).eval()
    prompt = torch.randint(256, (2, 13), device=device)
    reference = StreamingSession(model, batch=2, capacity=50)
    scores = reference.consume(prompt); expected = []
    for _ in range(20):
        token = scores[:, -1].argmax(-1, keepdim=True)
        expected.append(token); scores = reference.consume(token)
    fast = FastDecodeSession(model, batch=2, capacity=50, backend=backend)
    old = fast.consume(prompt); saved = old.clone()
    actual = torch.cat((fast.generate(7), fast.generate(13)), 1)
    torch.testing.assert_close(actual, torch.cat(expected, 1), rtol=0, atol=0)
    torch.testing.assert_close(old, saved, rtol=0, atol=0)
    # Consume is still available after graph generation, with owned outputs.
    extra = torch.randint(256, (2, 3), device=device)
    output = fast.consume(extra); preserved = output.clone()
    torch.testing.assert_close(output, reference.consume(extra), atol=3e-6, rtol=3e-5)
    fast.generate(1)
    torch.testing.assert_close(output, preserved, rtol=0, atol=0)


def test_compiled_cpu_generation_and_continuation():
    check_generation('compile', 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('hybrid,window', [(False, 0), (False, 8), (True, 8)])
def test_graph_cuda_generation_and_owned_outputs(hybrid, window):
    check_generation('graph', 'cuda', hybrid, window)


def test_fast_session_validates_inputs_and_stale_weights():
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4)).eval()
    session = FastDecodeSession(model, capacity=8, backend='eager')
    with pytest.raises(ValueError, match='prompt'):
        session.generate(1)
    with pytest.raises(ValueError, match='vocabulary'):
        session.consume(torch.tensor([[256]]))
    with pytest.raises(ValueError, match='session_batch'):
        session.consume(torch.ones(2, 1, dtype=torch.long))
    session.consume(torch.ones(1, 7, dtype=torch.long))
    with pytest.raises(ValueError, match='capacity'):
        session.generate(2)
    assert session.position == 7
    session.generate(1)
    with torch.no_grad():
        model.energy_head.phase.add_(.01)
    with pytest.raises(ValueError, match='weights'):
        session.generate(1)


def test_fast_sampling_reproducibility():
    model = StreamingLM(StreamingConfig(width=16, heads=2, hidden=24, modes=4)).eval()
    outputs = []
    for _ in range(2):
        session = FastDecodeSession(model, capacity=32, backend='eager')
        session.consume(torch.ones(1, 4, dtype=torch.long))
        outputs.append(session.generate(20, temperature=.8, top_k=12, generator=torch.Generator().manual_seed(69)))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
