"""Native LM + exact graph workspace + learned per-block fact readers."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .episodic import FactStore
from .fact_memory import snapshot_memory
from .graph_program import GraphProgram, execute
from .model import Config, LM
from .runtime import atomic_checkpoint, sha256


def augment_checkpoint(source, out, *, memory_dim=32, memory_capacity=256):
    """Explicit weight-only conversion; never claim optimizer-compatible resume."""
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if type(memory_dim) is not int or memory_dim < 1:
        raise ValueError('Positive memory dimension required')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    old = Config(**saved['config'])
    if old.memory_dim:
        raise ValueError('Checkpoint already has memory readers')
    original = LM(old)
    original.load_state_dict(saved['weights'], strict=True)
    config = replace(old, memory_dim=memory_dim, memory_capacity=memory_capacity)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = LM(config)
    state = model.state_dict()
    state.update(original.state_dict())
    model.load_state_dict(state, strict=True)
    payload = {'config': asdict(config), 'weights': model.state_dict(),
               'initialization': 'zero-gated fact readers; original native weights preserved',
               'parent_sha256': sha256(source), 'training': 'weight_only_warm_start'}
    atomic_checkpoint(payload, out)
    return payload


class HybridSession:
    """Single-user inference session. Store commits remain explicitly authorized.

    Reads/programs use a single SQLite snapshot. Any committed revision change
    discards recurrent state before processing more tokens, preventing stale
    neural history from surviving an edit. This intentionally loses old token
    context; callers can prefill an appropriate current prompt again.
    """
    def __init__(self, model: LM, store: FactStore):
        if not model.config.memory_dim:
            raise ValueError('Hybrid sessions require a memory-enabled native LM')
        self.model, self.store = model.eval(), store
        self.state = None
        self.revision = None
        self.memory = None

    @torch.no_grad()
    def forward(self, tokens, reset=None):
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError('One conversation per hybrid session required')
        revision, facts, _ = self.store.snapshot()
        changed = revision != self.revision
        memory = self.memory
        if changed:
            memory = snapshot_memory(facts, self.model.config.memory_dim,
                                     self.model.config.memory_capacity).to(tokens.device)
        elif memory.keys.device != tokens.device:
            memory = memory.to(tokens.device)
        logits, state = self.model(tokens, None if changed else self.state, reset, memory=memory)
        self.state = tuple(tuple(t.detach() for t in layer) for layer in state)
        self.revision, self.memory = revision, memory
        return logits, {'revision': revision, 'state_reset': changed, 'fact_count': len(facts),
                        'fact_memory_bytes': memory.bytes}

    def answer(self, program):
        if not isinstance(program, GraphProgram):
            program = GraphProgram.parse(program)
        revision, facts, _ = self.store.snapshot()
        return execute(program, facts, revision, max_bindings=self.model.config.memory_capacity)

    def reset(self):
        self.state = self.revision = self.memory = None

    @classmethod
    def load(cls, checkpoint, store, *, device='cpu'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = LM(Config(**saved['config']))
        model.load_state_dict(saved['weights'], strict=True)
        return cls(model.to(device), store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    upgrade = commands.add_parser('augment')
    upgrade.add_argument('--checkpoint', type=Path, required=True)
    upgrade.add_argument('--out', type=Path, required=True)
    upgrade.add_argument('--memory-dim', type=int, default=32)
    upgrade.add_argument('--memory-capacity', type=int, default=256)
    query = commands.add_parser('query')
    query.add_argument('--store', type=Path, required=True)
    query.add_argument('--program', type=Path, required=True)
    generate = commands.add_parser('generate')
    generate.add_argument('--checkpoint', type=Path, required=True)
    generate.add_argument('--store', type=Path, required=True)
    generate.add_argument('--prompt', required=True)
    generate.add_argument('--tokens', type=int, default=32)
    args = parser.parse_args()
    if args.action == 'augment':
        augment_checkpoint(args.checkpoint, args.out, memory_dim=args.memory_dim, memory_capacity=args.memory_capacity)
        print(json.dumps({'checkpoint': str(args.out), 'status': 'created'}))
    elif args.action == 'query':
        if not args.store.exists():
            raise FileNotFoundError(args.store)
        store = FactStore(args.store)
        revision, facts, _ = store.snapshot()
        print(json.dumps(execute(GraphProgram.parse(json.loads(args.program.read_text())), facts, revision)))
    else:
        if not args.store.exists():
            raise FileNotFoundError(args.store)
        if not args.prompt or len(args.prompt.encode('utf-8')) > 4096 or not 1 <= args.tokens <= 256:
            raise ValueError('Use a 1–4096-byte prompt and 1–256 generated bytes')
        session = HybridSession.load(args.checkpoint, FactStore(args.store))
        if session.model.config.vocab != 256:
            raise ValueError('Text generation requires a byte-vocabulary checkpoint')
        torch.set_num_threads(2)
        tokens = torch.tensor([list(args.prompt.encode('utf-8'))], dtype=torch.long)
        raw, revisions = bytearray(), []
        for _ in range(args.tokens):
            logits, info = session.forward(tokens)
            tokens = logits[:, -1].argmax(-1, keepdim=True)
            raw.append(int(tokens.item()))
            revisions.append(info['revision'])
        print(json.dumps({'generated_text': raw.decode('utf-8', errors='replace'), 'generated_bytes': list(raw),
                          'revisions': revisions, 'decoding': 'greedy',
                          'scope': 'Neural text; use graph programs for exact evidence-backed answers.'}))


if __name__ == '__main__':
    main()
