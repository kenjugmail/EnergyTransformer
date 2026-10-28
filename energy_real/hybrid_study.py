"""Bounded adaptation of native-model fact readers, with frozen original weights.

This is a controlled integration test on supplied one-hop symbol queries. It
does not train a natural-language parser or measure general language gains.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .diagnostics import active_records, events_for, generate
from .episodes import oracle
from .fact_memory import FactMemory, snapshot_memory
from .graph_program import GraphProgram, execute
from .hybrid import augment_checkpoint
from .local_energy_study import Budget
from .model import Config, LM
from .runtime import atomic_checkpoint, atomic_json, environment, sha256, source_identity


def examples(count, seed):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        subject, relation = rng.choice('ab'), rng.choice('rs')
        facts = []
        for s in 'ab':
            for r in 'rs':
                facts.append({'operation': 'set', 'subject': s, 'relation': r,
                              'object': rng.choice('ABCDEF'), 'fact_id': f'{seed}:{index}:{s}:{r}'})
        family = ('set', 'replace', 'delete', 'hold')[index % 4]
        target = next(f for f in facts if (f['subject'], f['relation']) == (subject, relation))
        history = [dict(f) for f in facts]
        if family == 'replace':
            target['object'] = rng.choice([c for c in 'ABCDEF' if c != target['object']])
            target['fact_id'] += ':updated'
            history.append(dict(target))
        elif family == 'delete':
            history.append({**target, 'operation': 'delete', 'object': None})
            facts.remove(target)
        elif family == 'hold':
            history.append({**target, 'operation': 'hold', 'object': rng.choice('ABCDEF')})
        # Replay supplies causal event semantics; target is evaluator-only.
        active = {}
        for event in history:
            address = (event['subject'], event['relation'])
            if event['operation'] == 'set': active[address] = event
            elif event['operation'] == 'delete': active.pop(address, None)
        current = sorted(active.values(), key=lambda f: (f['subject'], f['relation']))
        expected = active.get((subject, relation), {}).get('object', '?')
        rows.append({'id': f'{seed}:{index}', 'family': family, 'subject': subject, 'relation': relation,
                     'prompt': f'{subject}{relation}?', 'facts': current, 'events': history, 'expected': expected})
    return rows


def pack(rows, dimension, capacity):
    keys, values = torch.zeros(len(rows), 4, dimension), torch.zeros(len(rows), 4, dimension)
    mask = torch.zeros(len(rows), 4, dtype=torch.bool)
    for i, row in enumerate(rows):
        memory = snapshot_memory(row['facts'], dimension, capacity)
        n = memory.keys.shape[1]
        keys[i, :n], values[i, :n], mask[i, :n] = memory.keys[0], memory.values[0], memory.mask[0]
    return (torch.tensor([list(r['prompt'].encode()) for r in rows]),
            torch.tensor([ord(r['expected']) for r in rows]), FactMemory(keys, values, mask))


def take(data, index):
    x, y, memory = data
    return x[index], y[index], FactMemory(memory.keys[index], memory.values[index], memory.mask[index])


@torch.no_grad()
def score(model, data, budget, *, use_memory=True):
    hits, losses, predicted = [], [], []
    for start in range(0, len(data[1]), 32):
        budget.check()
        x, y, memory = take(data, slice(start, start + 32))
        logits, _ = model(x, memory=memory if use_memory else None)
        output = logits[:, -1].float()
        prediction = output.argmax(-1)
        hits.extend(prediction.eq(y).tolist())
        losses.extend(F.cross_entropy(output, y, reduction='none').tolist())
        predicted.extend(prediction.tolist())
    return {'accuracy': sum(hits) / len(hits), 'nll': sum(losses) / len(losses),
            'correct': hits, 'prediction': predicted}


def backbone_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if '.fact_reader.' not in name:
            digest.update(name.encode() + b'\0' + tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    if args.out.exists(): raise FileExistsError(args.out)
    if not 1 <= args.steps <= 1000 or not 1 <= args.batch <= 32 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Bounded steps, batch size, and unique seeds required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    identity = source_identity()
    budget = Budget(args.budget_ledger, args.max_seconds, args.prior_ledger)
    status = 'failed'
    records, tests = [], []
    try:
        args.out.mkdir(parents=True)
        initial = augment_checkpoint(args.checkpoint, args.out / 'initial.pt', memory_dim=32)
        config = Config(**initial['config'])
        if config.vocab != 256:
            raise ValueError('Expected a byte-vocabulary native checkpoint')
        training, validation = examples(1024, 591001), examples(256, 591002)
        atomic_json({'train': training, 'valid': validation,
                     'test_recipe': {'count': 600, 'seed': 591003},
                     'scope': 'New independent fact assignments over fixed symbols. Query grammar and edit semantics supplied.'}, args.out / 'data.json')
        train, valid = pack(training, 32, config.memory_capacity), pack(validation, 32, config.memory_capacity)
        atomic_json({'source_sha256': identity, 'parent_sha256': sha256(args.checkpoint), 'config': asdict(config),
                     'seeds': args.seeds, 'steps': args.steps, 'batch': args.batch, 'lr': .003,
                     'selection': 'Minimum validation CE including initialization; all seeds frozen before test.',
                     'data_sha256': sha256(args.out / 'data.json')}, args.out / 'plan.json')
        for seed in args.seeds:
            budget.check()
            torch.manual_seed(seed)
            model = LM(config)
            model.load_state_dict(initial['weights'], strict=True)
            expected_hash = backbone_hash(model)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_('.fact_reader.' in name)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=.003)
            generator = torch.Generator().manual_seed(seed + 9000)
            best = score(model, valid, budget)
            initial_score = {k: v for k, v in best.items() if k not in ('correct', 'prediction')}
            selected = {k: v.clone() for k, v in model.state_dict().items() if '.fact_reader.' in k}
            best_step, history = 0, []
            begin = time.perf_counter()
            for step in range(1, args.steps + 1):
                budget.check()
                indices = torch.randint(len(training), (args.batch,), generator=generator)
                x, y, memory = take(train, indices)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x, memory=memory)
                loss = F.cross_entropy(logits[:, -1].float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                if step % max(1, args.steps // 4) == 0 or step == args.steps:
                    result = score(model, valid, budget)
                    history.append({'step': step, 'loss': float(loss.detach()), 'valid_accuracy': result['accuracy'], 'valid_nll': result['nll']})
                    print(json.dumps({'seed': seed, **history[-1]}), flush=True)
                    if result['nll'] < best['nll']:
                        best, best_step = result, step
                        selected = {k: v.clone() for k, v in model.state_dict().items() if '.fact_reader.' in k}
            assert backbone_hash(model) == expected_hash
            weights = model.state_dict(); weights.update(selected)
            model.load_state_dict(weights, strict=True)
            path = args.out / f'seed{seed}.pt'
            atomic_checkpoint({'config': asdict(config), 'weights': model.state_dict(), 'source_sha256': identity,
                               'parent_sha256': sha256(args.checkpoint), 'best_step': best_step,
                               'scope': 'One-hop symbolic-query adapter; original native weights frozen.'}, path)
            records.append({'seed': seed, 'best_step': best_step, 'updates': args.steps,
                            'initial_validation': initial_score, 'validation': {k: v for k, v in best.items() if k not in ('correct', 'prediction')},
                            'trainable_parameters': sum(p.numel() for p in parameters), 'total_parameters': model.parameters_count(),
                            'backbone_sha256': expected_hash, 'backbone_unchanged': True, 'checkpoint_sha256': sha256(path),
                            'elapsed_s': time.perf_counter() - begin, 'history': history})
        # Freeze all choices before creating or evaluating the test set.
        selected_seed = min(records, key=lambda r: (r['validation']['nll'], r['seed']))['seed']
        atomic_json({'source_sha256': identity, 'selected_seed': selected_seed, 'records': records}, args.out / 'selection.json')
        testing = examples(600, 591003)
        atomic_json(testing, args.out / 'test_data.json')
        data = pack(testing, 32, config.memory_capacity)
        for record in records:
            saved = torch.load(args.out / f"seed{record['seed']}.pt", weights_only=True)
            model = LM(config); model.load_state_dict(saved['weights'], strict=True)
            active = score(model, data, budget)
            disabled = score(model, data, budget, use_memory=False)
            changed = (data[0], data[1], FactMemory(data[2].keys.roll(1, 0), data[2].values.roll(1, 0), data[2].mask.roll(1, 0)))
            shuffled = score(model, changed, budget)
            groups = {family: sum(v for v, row in zip(active['correct'], testing) if row['family'] == family) /
                      sum(row['family'] == family for row in testing) for family in ('set', 'replace', 'delete', 'hold')}
            tests.append({'seed': record['seed'], 'memory': active, 'memory_disabled': disabled,
                          'memory_shuffled': shuffled, 'groups': groups})
        exact = []
        for episode in generate('test', 30, seed=591004):
            budget.check()
            facts = active_records(events_for(episode))
            answer = execute(GraphProgram.path(episode.query.subject, episode.query.relations), list(facts.values()), 1)
            expected, support = oracle(episode)
            correct = answer['abstained'] if expected is None else not answer['abstained'] and answer['answers'] == [expected]
            exact.append({'id': episode.id, 'family': episode.family, 'correct': correct,
                          'abstained': answer['abstained'], 'evidence': [f['fact_id'] for f in answer['facts']]})
        if source_identity() != identity:
            raise ValueError('Source changed during integration study')
        result = {'status': 'completed', 'source_sha256': identity, 'environment': environment(torch.device('cpu')),
                  'parent_sha256': sha256(args.checkpoint), 'selected_seed': selected_seed, 'records': records, 'tests': tests,
                  'graph_program_test': {'accuracy': sum(r['correct'] for r in exact) / len(exact), 'rows': exact},
                  'gpu_seconds': 0, 'scope': 'Reader integration and one-hop symbol binding; no general language quality claim.'}
        atomic_json(result, args.out / 'results.json')
        status = 'completed'
        return result
    finally:
        budget.finish(status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--max-seconds', type=float, default=1800)
    p.add_argument('--budget-ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    p.add_argument('--prior-ledger', type=Path, default=Path('reports/publication/device_stages.json'))
    run(p.parse_args())
