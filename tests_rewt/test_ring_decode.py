import pytest
import torch

from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM
from rewt.ring_decode import RingDecodeSession


def make_model(solver, history=5, kernel=3):
    args = dict(width=8, hidden=19, modes=5, history=history, position_period=8,
        context_kernel=kernel, recurrent_type='gru', output_geometry='centered',
        compatibility_geometry='anchor_relative', wave_geometry='anchor_relative',
        initial_carry_scale=0., cache_history=True)
    if solver in ('preconditioned', 'certified'):
        return ArchitectureEnergyLM(ArchitectureConfig(**args, solver=solver)).double().eval()
    return PositionMemoryEnergyLM(PositionMemoryConfig(**args, refinement_backend=solver)).double().eval()


@pytest.mark.parametrize('solver', ['reference', 'krylov2', 'certified', 'preconditioned'])
@pytest.mark.parametrize('history,kernel,prefix', [(1, 1, 1), (5, 3, 2), (5, 3, 15)])
@torch.no_grad()
def test_ring_wrap_position_wrap_and_old_output_ownership(solver, history, kernel, prefix):
    torch.manual_seed(22171)
    model = make_model(solver, history, kernel)
    tokens = torch.randint(0, 256, (2, prefix + 23))
    expected = model(tokens).logits
    session = RingDecodeSession(model)
    outputs = [session.prefill(tokens[:, :prefix]).logits]
    pointers = session.keys.data_ptr(), session.values.data_ptr(), session.joint.data_ptr()
    for t in range(prefix, tokens.shape[1]):
        outputs.append(session.step(tokens[:, t:t+1]))
        assert pointers == (session.keys.data_ptr(), session.values.data_ptr(), session.joint.data_ptr())
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-10, rtol=1e-9)
    assert int(session.count) == tokens.shape[1]


@torch.no_grad()
def test_ring_invalidation_and_shape_errors():
    model = make_model('krylov2')
    session = RingDecodeSession(model)
    with pytest.raises(RuntimeError):
        session.step(torch.ones(1, 1, dtype=torch.long))
    session.prefill(torch.ones(1, 4, dtype=torch.long))
    for bad in (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 1), torch.tensor([[256]])):
        with pytest.raises(ValueError):
            session.step(bad)
    model.phase.add_(.1)
    with pytest.raises(RuntimeError):
        session.step(torch.ones(1, 1, dtype=torch.long))


@torch.no_grad()
def test_architecture_inference_cache_invalidates():
    model = make_model('preconditioned').prepare_inference()
    tokens = torch.ones(1, 4, dtype=torch.long)
    original = model(tokens).logits
    model.phase.add_(.1)
    with pytest.raises(RuntimeError):
        model(tokens)
    changed = model.prepare_inference()(tokens).logits
    assert not torch.equal(original, changed)


def test_legacy_session_cannot_silently_change_architecture_solver():
    from rewt.decode import EnergyDecodeSession
    with pytest.raises(ValueError,match='architecture models'):
        EnergyDecodeSession(make_model('preconditioned'))
