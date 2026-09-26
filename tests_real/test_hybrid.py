from dataclasses import asdict, replace
import copy

import pytest
import torch

from energy_real.episodic import FactEdit, FactStore
from energy_real.fact_memory import FactMemory, snapshot_memory
from energy_real.graph_program import GraphProgram, Instruction, execute
from energy_real.hybrid import HybridSession, augment_checkpoint
from energy_real.local_lm import pcalm_loss
from energy_real.model import Config, LM


def config(**kw):
    return Config(width=16, layers=2, heads=2, slots=4, ff_width=32, chunk=4, **kw)


def fact(subject='a', relation='r', obj='b', id='1'):
    return dict(operation='set', subject=subject, relation=relation, object=obj, fact_id=id)


@pytest.mark.parametrize('kind', ['energy', 'softmax'])
def test_zero_gate_upgrade_preserves_checkpoint_and_chunked_outputs(tmp_path, kind):
    torch.manual_seed(11)
    original = LM(config(kind=kind)).eval()
    source, target = tmp_path / 'base.pt', tmp_path / 'hybrid.pt'
    torch.save({'config': asdict(original.config), 'weights': original.state_dict()}, source)
    saved = augment_checkpoint(source, target, memory_dim=8)
    model = LM(Config(**saved['config'])).eval()
    model.load_state_dict(saved['weights'], strict=True)
    memory = snapshot_memory([fact()], 8)
    tokens = torch.randint(256, (1, 9))
    with torch.no_grad():
        before = original(tokens)[0]
        after = model(tokens, memory=memory)[0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        for block in model.blocks:
            block.fact_reader.gate.fill_(.5)
        whole, _ = model(tokens, memory=memory)
        prefix, state = model(tokens[:, :4], memory=memory)
        suffix, _ = model(tokens[:, 4:], state, memory=memory)
        torch.testing.assert_close(whole, torch.cat((prefix, suffix), 1), rtol=3e-5, atol=3e-5)
    with pytest.raises(FileExistsError):
        augment_checkpoint(source, target)


def test_optional_readers_preserve_shared_initialization():
    torch.manual_seed(3); a = LM(config())
    torch.manual_seed(3); b = LM(config(memory_dim=8))
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name]), name


def test_memory_empty_validation_and_reader_gradients():
    model = LM(config(memory_dim=8))
    x = torch.ones(1, 4, dtype=torch.long)
    empty = snapshot_memory([], 8)
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x, memory=empty)[0])
        for block in model.blocks:
            block.fact_reader.gate.fill_(.2)
    logits, _ = model(x, memory=snapshot_memory([fact(), fact('b', obj='c', id='2')], 8))
    logits.square().mean().backward()
    for block in model.blocks:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.fact_reader.parameters())
        assert block.fact_reader.value.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='capacity'):
        snapshot_memory([fact(), fact()], 8, capacity=1)
    with pytest.raises(ValueError, match='finite'):
        model(x, memory=FactMemory(torch.full((1, 1, 8), float('nan')), empty.values, empty.mask))
    with pytest.raises(ValueError):
        model(x, memory=snapshot_memory([fact()], 4))


def test_program_branches_provenance_and_missing_premise():
    steps = (Instruction('entity', ('a',)), Instruction('follow', ('0', 'r')),
             Instruction('entity', ('c',)), Instruction('follow', ('2', 'r')),
             Instruction('intersect', ('1', '3')))
    program = GraphProgram(steps)
    facts = [fact(), fact('c', obj='b', id='2')]
    answer = execute(program, facts, 7)
    assert answer['answers'] == ['b'] and answer['revision'] == 7
    assert {f['fact_id'] for f in answer['facts']} == {'1', '2'}
    assert execute(program, facts[:1], 8)['abstained']
    unequal = [fact(), fact('c', obj='d', id='2')]
    assert execute(program, unequal, 8)['answers'] == []
    union = GraphProgram((*steps[:-1], Instruction('union', ('1', '3'))))
    assert execute(union, unequal, 8)['answers'] == ['b', 'd']
    with pytest.raises(ValueError, match='budget'):
        execute(union, unequal, 8, max_bindings=1)
    assert execute(GraphProgram.path('a', ('r', 'r')), [fact(), fact('b', obj='z', id='2')], 1)['answers'] == ['z']


@pytest.mark.parametrize('payload', [
    {'steps': [{'operation': 'exec', 'arguments': ['rm']}]},
    {'steps': [{'operation': 'follow', 'arguments': ['0', 'r']}]},
    {'steps': [{'operation': 'entity', 'arguments': ['a'], 'code': 'bad'}]},
    {'steps': []},
])
def test_program_rejects_arbitrary_code_forward_refs_and_malformed_inputs(payload):
    with pytest.raises(ValueError):
        GraphProgram.parse(payload)


def test_session_correction_deletion_hold_and_failed_calls(tmp_path):
    store = FactStore(tmp_path / 'facts.sqlite')
    model = LM(config(memory_dim=8))
    with torch.no_grad():
        for block in model.blocks: block.fact_reader.gate.fill_(.5)
    session = HybridSession(model, store)
    query = GraphProgram.path('a', ('r',))
    def commit(op, obj):
        return store.apply([FactEdit(op, 'a', 'r', obj, f'{op}:{obj}')],
                           expected_revision=store.revision, request_id=str(store.revision), authorized=True)
    commit('set', 'b')
    x = torch.ones(1, 3, dtype=torch.long)
    assert session.forward(x)[1]['state_reset']
    assert not session.forward(x)[1]['state_reset']
    commit('set', 'c')
    logits, info = session.forward(x)
    assert info['state_reset'] and session.answer(query)['answers'] == ['c']
    with torch.no_grad():
        fresh = model(x, memory=session.memory)[0]
    assert torch.equal(logits, fresh)
    commit('hold', 'wrong')
    assert session.answer(query)['answers'] == ['c']
    commit('delete', None)
    assert session.answer(query)['abstained']
    _, info = session.forward(x)
    assert info['fact_count'] == 0 and info['state_reset']
    previous = copy.deepcopy(session.state)
    with pytest.raises((IndexError, RuntimeError)):
        session.forward(torch.full_like(x, 999))
    assert all(torch.equal(a, b) for aa, bb in zip(previous, session.state) for a, b in zip(aa, bb))
    reopened = HybridSession(model, FactStore(store.path))
    assert reopened.answer(query)['abstained']


def test_native_pcalm_local_gradients_and_deadline():
    torch.manual_seed(7)
    model = LM(config(memory_dim=8))
    with torch.no_grad():
        for b in model.blocks: b.fact_reader.gate.fill_(.2)
    tokens, targets = torch.randint(256, (1, 4)), torch.randint(256, (1, 4))
    result = pcalm_loss(model, tokens, targets, memory=snapshot_memory([fact()], 8), steps=4)
    assert all(p.grad is None for p in model.parameters())
    assert result.loss.isfinite() and result.residual_rms > 0
    result.loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    calls = []
    def deadline():
        calls.append(1)
        if len(calls) == 3: raise TimeoutError('deadline')
    with pytest.raises(TimeoutError):
        pcalm_loss(model, tokens, targets, check_budget=deadline)
    with pytest.raises(ValueError):
        pcalm_loss(model, tokens, targets, state_lr=float('nan'))


def test_native_local_boundary_rule_matches_bp_at_linear_equilibrium():
    from types import SimpleNamespace
    class LinearBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, bias=False)
            with torch.no_grad(): self.linear.weight.copy_(torch.eye(3) * .5)
        def forward(self, x, state, reset, memory):
            return self.linear(x), None
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(memory_dim=0)
            self.embedding = torch.nn.Embedding(5, 3)
            self.blocks = torch.nn.ModuleList([LinearBlock(), LinearBlock()])
            self.norm = torch.nn.Identity()
            self.head = torch.nn.Linear(3, 5, bias=False)
    torch.manual_seed(4)
    model = Model()
    x, y = torch.tensor([[1, 2]]), torch.tensor([[2, 3]])
    state = model.embedding(x)
    for block in model.blocks: state = block(state, None, None, None)[0]
    bp = torch.nn.functional.cross_entropy(model.head(state).flatten(0, 1), y.flatten())
    expected = torch.autograd.grad(bp, tuple(model.parameters()))
    local = pcalm_loss(model, x, y, steps=256, state_lr=.2, dual_lr=.1)
    actual = torch.autograd.grad(local.loss, tuple(model.parameters()))
    assert local.residual_rms < 1e-5
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)


def test_full_native_adapter_study_freezes_backbone_and_selections(tmp_path):
    from types import SimpleNamespace
    from energy_real.hybrid_study import run
    from energy_real.runtime import sha256
    source = tmp_path / 'base.pt'
    model = LM(config())
    torch.save({'config': asdict(model.config), 'weights': model.state_dict()}, source)
    before = sha256(source)
    args = SimpleNamespace(out=tmp_path / 'study', steps=1, batch=2, seeds=[11], checkpoint=source,
                           max_seconds=60, budget_ledger=tmp_path / 'budget.json', prior_ledger=None)
    result = run(args)
    assert result['status'] == 'completed'
    assert result['graph_program_test']['accuracy'] == 1
    assert result['records'][0]['backbone_unchanged']
    assert sha256(source) == before
    assert len(result['tests'][0]['memory']['correct']) == 600
    assert (args.out / 'selection.json').stat().st_mtime_ns <= (args.out / 'test_data.json').stat().st_mtime_ns
