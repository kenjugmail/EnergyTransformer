"""Learned energy factors over discrete entity paths, with no learned embeddings.

Scalar feature weights are trained by exact sum-product over all latent path
assignments. Inference messages are explicit log masses for named entity IDs.
Event parsing, latest-write semantics, features, and path composition are
supplied inductive biases; this does not discover a DSL or parse language.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .local_energy_data import execute, graph_dataset
from .local_energy_study import Budget
from .memory_eval import paired_interval
from .runtime import atomic_checkpoint, atomic_json, environment, sha256, source_identity


FEATURES = ('active_relation', 'missing_to_unknown', 'stale_set', 'hypothetical',
            'other_relation', 'answer_unknown', 'unknown_absorbing')


def encode_factors(episodes):
    if not episodes:
        raise ValueError('Nonempty episodes required')
    entities = episodes[0]['entities']
    if any(e['entities'] != entities for e in episodes):
        raise ValueError('A batch must have one entity-domain size')
    hops = max(len(e['path']) for e in episodes)
    features = torch.zeros(len(episodes), hops, entities + 1, entities + 1, len(FEATURES))
    active_steps = torch.zeros(len(episodes), hops, dtype=torch.bool)
    starts = torch.tensor([e['start'] for e in episodes])
    for b, episode in enumerate(episodes):
        _, _, active = execute(episode['events'], episode['start'], (), entities)
        for h, relation in enumerate(episode['path']):
            active_steps[b, h] = True
            features[b, h, :, entities, 5] = 1
            features[b, h, entities, entities, 6] = 1
            for source in range(entities):
                if (source, relation) in active:
                    features[b, h, source, active[(source, relation)], 0] = 1
                else:
                    features[b, h, source, entities, 1] = 1
                other = active.get((source, 1 - relation))
                if other is not None:
                    features[b, h, source, other, 4] = 1
            for operation, source, rel, obj in episode['events']:
                if rel != relation or obj is None:
                    continue
                if operation == 'set' and active.get((source, relation)) != obj:
                    features[b, h, source, obj, 2] = 1
                elif operation == 'hold':
                    features[b, h, source, obj, 3] = 1
    return features, active_steps, starts


class DiscreteEnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(len(FEATURES)) * .01)

    def forward(self, features, active_steps, starts):
        # Minus local energy: one interpretable weight per supplied feature.
        scores = features @ self.weights
        states = scores.shape[-1]
        message = torch.full((len(starts), states), -torch.inf, device=scores.device, dtype=scores.dtype)
        message.scatter_(1, starts[:, None], 0.)
        for h in range(scores.shape[1]):
            following = torch.logsumexp(message[:, :, None] + scores[:, h], dim=1)
            message = torch.where(active_steps[:, h, None], following, message)
        # logsumexp over the explicit path assignments ending at each answer.
        return message

    def candidate_free_energies(self, features, active_steps, starts):
        return -self(features, active_steps, starts)


def score(model, encoded, data):
    with torch.no_grad():
        logits = model(*encoded)
        predicted = logits.argmax(-1)
        loss = -logits.log_softmax(-1)[torch.arange(len(data.y)), data.y]
    unknown = encoded[0].shape[2] - 1
    hits = predicted.eq(data.y)
    return {'accuracy': float(hits.float().mean()), 'nll': float(loss.mean()),
            'unsupported_answers': int((data.y.eq(unknown) & predicted.ne(unknown)).sum()),
            'groups': {g: float(hits[[v == g for v in data.groups]].float().mean()) for g in sorted(set(data.groups))},
            'rows': [{'id': identifier, 'group': group, 'expected': int(y), 'prediction': int(p),
                      'correct': bool(c), 'nll': float(n)}
                     for identifier, group, y, p, c, n in zip(data.ids, data.groups, data.y, predicted, hits, loss)]}


def run(args):
    if not 1 <= args.steps <= 2000 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use 1–2000 updates and unique seeds')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    budget = Budget(args.budget_ledger, args.max_seconds, args.prior_ledger)
    identity = source_identity()
    status = 'running'
    records, tests = [], []
    try:
        datasets = {split: graph_dataset(split, count, seed) for split, count, seed in
                    [('train', args.train_count, 481001), ('valid', args.valid_count, 481002), ('test', args.test_count, 481003)]}
        encoded = {split: encode_factors(data.episodes) for split, data in datasets.items()}
        atomic_json({'source_sha256': identity, 'features': list(FEATURES), 'steps': args.steps,
                     'seeds': args.seeds, 'optimizer': 'Adam(lr=0.05, weight_decay=0.001)', 'batch': 64,
                     'datasets': {split: d.manifest() for split, d in datasets.items()},
                     'scope': 'Supplied feature vocabulary and execution semantics; learned seven scalar weights. No learned embeddings.'}, args.out / 'plan.json')
        for seed in args.seeds:
            budget.check()
            started = time.perf_counter()
            torch.manual_seed(seed)
            model = DiscreteEnergyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=.05, weight_decay=.001)
            generator = torch.Generator().manual_seed(seed + 7000)
            initial = score(model, encoded['valid'], datasets['valid'])
            best, selected_step = initial, 0
            best_weights = model.weights.detach().clone()
            atomic_checkpoint({'weights': model.weights.detach().clone()}, args.out / f'initial_seed{seed}.pt')
            history = []
            for step in range(1, args.steps + 1):
                budget.check()
                index = torch.randint(len(datasets['train'].y), (64,), generator=generator)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(*(part[index] for part in encoded['train'])), datasets['train'].y[index])
                loss.backward()
                if not torch.isfinite(loss) or not torch.isfinite(model.weights.grad).all():
                    raise FloatingPointError('Nonfinite discrete-factor learning')
                optimizer.step()
                if step % max(1, args.steps // 4) == 0 or step == args.steps:
                    validation = score(model, encoded['valid'], datasets['valid'])
                    if (validation['accuracy'], -validation['nll']) > (best['accuracy'], -best['nll']):
                        best, selected_step = validation, step
                        best_weights = model.weights.detach().clone()
                    history.append({'step': step, 'loss': float(loss.detach()), 'validation_accuracy': validation['accuracy']})
            checkpoint = args.out / f'best_seed{seed}.pt'
            atomic_checkpoint({'weights': best_weights}, checkpoint)
            record = {'seed': seed, 'steps': args.steps, 'best_step': selected_step, 'initial_validation': initial,
                      'selected_validation': best, 'weights': dict(zip(FEATURES, best_weights.tolist())),
                      'checkpoint': str(checkpoint.resolve()), 'sha256': sha256(checkpoint),
                      'initial_sha256': sha256(args.out / f'initial_seed{seed}.pt'),
                      'elapsed_s': time.perf_counter() - started, 'history': history}
            records.append(record)
            atomic_json({'status': status, 'records': records}, args.out / 'progress.json')
        atomic_json({'status': 'frozen_before_test', 'source_sha256': identity, 'records': records}, args.out / 'selection.json')
        for record in records:
            model = DiscreteEnergyModel()
            for name in ('initial', 'best'):
                path = args.out / f'{name}_seed{record["seed"]}.pt'
                expected_sha = record['initial_sha256'] if name == 'initial' else record['sha256']
                if sha256(path) != expected_sha:
                    raise ValueError('Discrete checkpoint changed')
                with torch.no_grad():
                    model.weights.copy_(torch.load(path, weights_only=True)['weights'])
                result = score(model, encoded['test'], datasets['test'])
                if name == 'initial':
                    initial = result
            timings = []
            for i in range(11):
                tick = time.perf_counter()
                with torch.no_grad():
                    model(*(part[:1] for part in encoded['test']))
                if i: timings.append(time.perf_counter() - tick)
            tests.append({'seed': record['seed'], 'initial': initial, 'trained': result,
                          'single_example_s_excluding_factor_encoding': timings,
                          'learning_interval': paired_interval([r['correct'] for r in result['rows']],
                                                               [r['correct'] for r in initial['rows']])})
        if source_identity() != identity:
            raise ValueError('Source changed during discrete study')
        status = 'completed'
        report = {'status': status, 'source_sha256': identity, 'environment': environment('cpu'),
                  'selection_sha256': sha256(args.out / 'selection.json'),
                  'parameters': 7, 'records': records, 'tests': tests,
                  'scope': 'Exact marginalization over discrete paths. Seven learned scalar costs; supplied features, parser, and graph composition. No general language claim.'}
        atomic_json(report, args.out / 'results.json')
        return report
    finally:
        budget.finish(status if status == 'completed' else 'failed')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    p.add_argument('--train-count', type=int, default=4096)
    p.add_argument('--valid-count', type=int, default=512)
    p.add_argument('--test-count', type=int, default=1200)
    p.add_argument('--budget-ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    p.add_argument('--prior-ledger', type=Path, default=Path('reports/publication/device_stages.json'))
    p.add_argument('--max-seconds', type=float, default=300)
    return p


if __name__ == '__main__':
    result = run(parser().parse_args())
    print(json.dumps({'status': result['status'], 'accuracy': [r['trained']['accuracy'] for r in result['tests']]}))
