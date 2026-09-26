"""Frozen-model feature probes, compact reports, and portable inference.

Probes use separate supervised ridge readouts of intermediate states. They do
not modify the tested network or select its weights. Energy-model features are
conditioned on the model's predicted answer, never a gold answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from .local_energy_data import Dataset, execute
from .local_energy_study import Budget, FitConfig, load_model, make_data
from .runtime import atomic_json, sha256, source_identity


def frozen_record(study: Path, name: str):
    selection = json.loads((study / 'selection.json').read_text())
    if selection['source_sha256'] != source_identity():
        raise ValueError('Source differs from frozen study')
    if selection['datasets_sha256'] != sha256(study / 'datasets.json'):
        raise ValueError('Frozen dataset recipe changed')
    record = next((r for r in selection['records'] if FitConfig(**r['config']).name == name), None)
    if record is None:
        raise ValueError(f'Unknown frozen fit: {name}')
    # Relocation works when the study directory moves with its fit folders.
    checkpoint = study / name / 'best.pt'
    if sha256(checkpoint) != record['best_sha256']:
        raise ValueError('Frozen model checkpoint changed')
    return record, checkpoint


@torch.no_grad()
def features(model, data, config, budget, batch=32):
    collected = []
    for start in range(0, len(data.y), batch):
        budget.check()
        x, _, memory = data.take(slice(start, start + batch))
        if not config.memory:
            memory = None
        if config.method == 'energy':
            scores = model.predict(x, memory, method='energy', steps=config.settle_steps,
                                   state_lr=config.state_lr, rho=config.rho,
                                   candidate_chunk=config.candidate_chunk, check_budget=budget.check)
            prediction = F.one_hot(scores.argmax(-1), model.config.outputs).to(x.dtype)
            states = model.relax(x, prediction, memory, steps=config.settle_steps,
                                 state_lr=config.state_lr, rho=config.rho, check_budget=budget.check).states
        else:
            states = model.feedforward(x, memory, return_states=True)[1]
        collected.append(states)
    return torch.cat(collected, dim=1)


def first_hop_targets(data):
    return torch.tensor([execute(e['events'], e['start'], e['path'][:1], e['entities'])[0] for e in data.episodes])


def probe_features(train_states, test_states, targets, expected, groups, classes):
    result = []
    for layer in range(len(train_states)):
        a, b = train_states[layer].double(), test_states[layer].double()
        mean, std = a.mean(0), a.std(0).clamp_min(1e-5)
        a, b = (a - mean) / std, (b - mean) / std
        a = torch.cat((a, torch.ones(len(a), 1, dtype=a.dtype)), -1)
        b = torch.cat((b, torch.ones(len(b), 1, dtype=b.dtype)), -1)
        # Fixed ridge penalty; no test tuning or model weight updates.
        penalty = .1 * torch.eye(a.shape[1], dtype=a.dtype)
        penalty[-1, -1] = 0.
        weights = torch.linalg.solve(a.T @ a + penalty, a.T @ F.one_hot(targets, classes).double())
        prediction = (b @ weights).argmax(-1)
        hits = prediction.eq(expected)
        result.append({'layer': layer, 'accuracy': float(hits.float().mean()),
                       'groups': {g: float(hits[[v == g for v in groups]].float().mean()) for g in sorted(set(groups))}})
    return result


def feature_probes(study, out, budget, *, train_count=512):
    plan = json.loads((study / 'plan.json').read_text())
    args = argparse.Namespace(**plan['recipes'])
    args.fashion_data, args.byte_data = Path(args.fashion_data), Path(args.byte_data)
    datasets = make_data('graph', args)
    manifest = json.loads((study / 'datasets.json').read_text())['graph']
    for split, data in datasets.items():
        if data.manifest() != manifest[split]:
            raise ValueError('Probe data differs from frozen study')
    train, test = datasets['train'], datasets['test']
    train = Dataset(train.x[:train_count], train.y[:train_count], train.ids[:train_count], train.groups[:train_count],
                    train.memory[:train_count], train.episodes[:train_count])
    targets, expected = first_hop_targets(train), first_hop_targets(test)
    selection = json.loads((study / 'selection.json').read_text())
    results = []
    for record in selection['records']:
        config = FitConfig(**record['config'])
        if config.task != 'graph':
            continue
        _, selected_path = frozen_record(study, config.name)
        row = {'config': record['config']}
        for name in ('initial', 'best'):
            budget.check()
            path = selected_path.with_name(f'{name}.pt')
            if sha256(path) != record[f'{name}_sha256']:
                raise ValueError('Probe checkpoint checksum failed')
            model = load_model(path)
            a = features(model, train, config, budget)
            b = features(model, test, config, budget)
            row[name] = probe_features(a, b, targets, expected, test.groups, model.config.outputs)
        results.append(row)
        atomic_json({'status': 'running', 'train_count': len(train.y), 'results': results}, out)
        print(json.dumps({'probe': config.name, 'first_hop_initial_last_layer': row['initial'][-1]['accuracy'],
                          'first_hop_trained_last_layer': row['best'][-1]['accuracy']}), flush=True)
    report = {'status': 'completed', 'source_sha256': source_identity(), 'selection_sha256': sha256(study / 'selection.json'),
              'train_count': len(train.y), 'test_count': len(test.y), 'ridge': .1,
              'scope': 'Diagnostic first-hop ridge probes at every layer. Predicted-label states for EBM; no gold conditioning. Network weights remain frozen. Probes themselves use first-hop supervision.',
              'results': results}
    atomic_json(report, out)
    return report


def compact_report(study, out, *, probes=None):
    summary = json.loads((study / 'summary.json').read_text())
    comparisons = json.loads((study / 'comparisons.json').read_text())
    diagnostics = json.loads((study / 'diagnostics.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('Only completed studies can produce a final report')
    out.mkdir(parents=True, exist_ok=True)
    records = {FitConfig(**r['config']).name: r for r in summary['records']}
    groups = {}
    for test in summary['test_results']:
        config = test['config']
        key = (config['task'], config['method'], config['depth'], config['width'], config['memory'])
        groups.setdefault(key, []).append(test)
    rows = []
    for key, tests in groups.items():
        task, method, depth, width, memory = key
        per_seed = []
        for test in tests:
            record = records[FitConfig(**test['config']).name]
            per_seed.append({'seed': test['config']['seed'], 'initial_accuracy': test['initial']['accuracy'],
                             'accuracy': test['trained']['accuracy'], 'bits_per_target': test['trained']['bits_per_target'],
                             'best_step': record['best_step'], 'updates': record['updates'], 'examples_seen': record['examples_seen'],
                             'training_elapsed_s': record['elapsed_s'], 'learning_interval': test['learning_interval'],
                             'groups': test['trained']['groups'],
                             'unsupported_answers': test['trained'].get('unsupported_answers')})
        diag = next(d for d in diagnostics if (d['config']['task'], d['config']['method'], d['config']['depth'],
                    d['config']['width'], d['config']['memory']) == key)
        record = records[FitConfig(**tests[0]['config']).name]
        rows.append({'task': task, 'method': method, 'depth': depth, 'width': width, 'memory': memory,
                     'mean_accuracy': statistics.mean(r['accuracy'] for r in per_seed),
                     'mean_initial_accuracy': statistics.mean(r['initial_accuracy'] for r in per_seed),
                     'mean_bits_per_target': statistics.mean(r['bits_per_target'] for r in per_seed),
                     'parameter_count': record['parameter_count'], 'parameter_bytes': record['parameter_bytes'],
                     'cpu_single_example_ms': 1000 * diag['single_example_median_s'],
                     'latency_seed': diag['config']['seed'],
                     'per_seed': per_seed})
    result = {'status': summary['status'], 'source_sha256': summary['source_sha256'],
              'selection_sha256': sha256(study / 'selection.json'), 'environment': summary['environment'],
              'completed_fits': summary['completed_fits'], 'gpu_seconds': summary['gpu_seconds'],
              'rows': rows, 'comparisons': comparisons}
    if (study / 'discrete_graph.json').exists():
        graph = json.loads((study / 'discrete_graph.json').read_text())
        result['discrete_graph'] = {'accuracy': graph['accuracy'], 'scope': graph['scope'],
                                   'mean_elapsed_ms': 1000 * statistics.mean(r['elapsed_s'] for r in graph['rows'])}
    if probes is not None:
        result['feature_probes'] = json.loads(probes.read_text())
    atomic_json(result, out / 'results.json')
    lines = ['# Local learning, energy modules, and full energy predictors', '',
             f"Completed {summary['completed_fits']} fits on CPU; all models trained from scratch. "
             'This is a bounded prototype study, not a reproduction of the Sakana results or a general-purpose language model.', '',
             'All checkpoint and temperature selections used validation only and were frozen before test evaluation. '
             'Training seeds are paired within each architecture. Initializations remain eligible for selection.', '']
    for task in ('fashion', 'graph', 'bytes'):
        subset = [r for r in rows if r['task'] == task]
        if not subset:
            continue
        lines += [f'## {task}', '', '| Method | Hidden layers | Memory | Initial accuracy | Trained accuracy | Bits/target | Parameters | CPU ms/example |',
                  '|---|---:|---|---:|---:|---:|---:|---:|']
        for row in subset:
            lines.append(f"| {row['method']} | {row['depth']} | {'yes' if row['memory'] else 'no'} | "
                         f"{row['mean_initial_accuracy']:.2%} | {row['mean_accuracy']:.2%} | {row['mean_bits_per_target']:.3f} | "
                         f"{row['parameter_count']:,} | {row['cpu_single_example_ms']:.3f} |")
        lines.append('')
    lines += ['## Interpretation limits', '',
              '- BP, PC, and PC-ALM use the same MSE objective and architecture. BP-CE is an additional byte likelihood control.',
              '- The full EBM uses candidate energy minimization and a contrastive cross-entropy objective plus 0.1 positive energy. Its objective differs from PC-ALM.',
              '- Finite EBM solves use approximate envelope gradients; stationarity is measured. They do not backpropagate through the solver.',
              '- The graph writer and event semantics are supplied. Memory readers are learned. All neural arms receive the complete current fact table.',
              '- The discrete graph control has supplied rules, no trained perception, and no learned hidden vector state.',
              '- The byte model predicts one of all 256 bytes from eight preceding bytes. Its tiny corpus windows and test count do not support broad language claims.',
              '- Reported intervals resample paired examples after averaging three seeds. Per-seed effects are reported separately; this is not a population estimate over many training seeds.',
              '- Latencies are CPU implementation measurements. Different inference budgets and candidate enumeration are charged explicitly; no hardware efficiency promotion is made.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return result


def generate_bytes(study, fit, prompt, count, budget):
    record, path = frozen_record(study, fit)
    config = FitConfig(**record['config'])
    if config.task != 'bytes' or config.memory:
        raise ValueError('Select a byte-model fit')
    model = load_model(path)
    context = model.config.inputs // 8
    raw = bytearray(prompt.encode('utf-8'))
    new = bytearray()
    for _ in range(count):
        budget.check()
        window = bytes(raw[-context:]).rjust(context, b'\x00')
        tokens = torch.tensor(list(window))[None]
        x = (((tokens[:, :, None] >> torch.arange(8)) & 1).float() * 2 - 1).flatten(1)
        method = 'bp' if config.method == 'bp_ce' else config.method
        score = model.predict(x, method=method, steps=config.settle_steps, state_lr=config.state_lr,
                              rho=config.rho, candidate_chunk=config.candidate_chunk, check_budget=budget.check)
        token = int(score.argmax(-1))
        raw.append(token); new.append(token)
    return {'fit': fit, 'prompt': prompt, 'generated_bytes': list(new),
            'generated_text': new.decode('utf-8', errors='replace'), 'decoding': 'greedy', 'context_bytes': context}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('probes', 'report', 'generate'))
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--probes', type=Path)
    p.add_argument('--fit')
    p.add_argument('--prompt', default='The world ')
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--budget-ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    p.add_argument('--prior-ledger', type=Path, default=Path('reports/publication/device_stages.json'))
    p.add_argument('--max-seconds', type=float, default=900)
    return p


def main(args):
    torch.set_num_threads(2)
    if args.action == 'report':
        return compact_report(args.study, args.out, probes=args.probes)
    if args.out.exists():
        raise FileExistsError(args.out)
    budget = Budget(args.budget_ledger, args.max_seconds, args.prior_ledger)
    status = 'running'
    try:
        if args.action == 'probes':
            result = feature_probes(args.study, args.out, budget)
        else:
            if args.tokens < 1 or args.tokens > 256 or not args.fit:
                raise ValueError('A byte fit and 1–256 generated tokens are required')
            result = generate_bytes(args.study, args.fit, args.prompt, args.tokens, budget)
            atomic_json(result, args.out)
        status = 'completed'
        return result
    finally:
        budget.finish(status if status == 'completed' else 'failed')


if __name__ == '__main__':
    main(parser().parse_args())
