"""Extra diagnostics on frozen weights; no selection or parameter updates."""
import argparse
import json
from pathlib import Path

import torch

from energy_real.local_energy import gradient_alignment
from energy_real.local_energy_analysis import frozen_record, generate_bytes
from energy_real.local_energy_study import Budget, FitConfig, load_model, make_data
from energy_real.runtime import atomic_json, sha256, source_identity


def inspect(study, out, budget):
    plan = json.loads((study / 'plan.json').read_text())
    args = argparse.Namespace(**plan['recipes'])
    args.fashion_data, args.byte_data = Path(args.fashion_data), Path(args.byte_data)
    datasets = {task: make_data(task, args)['valid'] for task in ('fashion', 'graph')}
    expected = json.loads((study / 'datasets.json').read_text())
    for task, data in datasets.items():
        if data.manifest() != expected[task]['valid']:
            raise ValueError('Validation data changed')
    selection = json.loads((study / 'selection.json').read_text())
    rows = []
    for record in selection['records']:
        config = FitConfig(**record['config'])
        if config.seed != 11 or config.method != 'bp' or config.task not in datasets:
            continue
        _, path = frozen_record(study, config.name)
        model = load_model(path)
        x, labels, memory = datasets[config.task].take(slice(0, 8))
        if not config.memory: memory = None
        for steps in (config.settle_steps, 2 * config.settle_steps):
            row = {'reference_fit': config.name, 'steps': steps, 'checkpoint_sha256': sha256(path)}
            for method in ('pc', 'pcalm'):
                budget.check()
                try:
                    row[method] = gradient_alignment(model, x, labels, memory, method=method, steps=steps,
                        state_lr=config.state_lr, dual_lr=config.dual_lr, rho=config.rho, check_budget=budget.check)
                except FloatingPointError as exc:
                    row[method] = {'status': 'unstable', 'reason': str(exc)}
            rows.append(row)
    generations = []
    for method in ('bp_ce', 'energy'):
        fit = f'bytes_{method}_d3_w32_mem0_seed11'
        generations.append(generate_bytes(study, fit, 'The world ', 32, budget))
    result = {'source_sha256': source_identity(), 'selection_sha256': sha256(study / 'selection.json'),
              'scope': 'PC versus PC-ALM derivatives on identical frozen BP-trained weights and eight validation examples. Extra iterations are diagnostic, not selected.',
              'common_weight_diagnostics': rows, 'byte_generations': generations}
    atomic_json(result, out)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--budget-ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    p.add_argument('--prior-ledger', type=Path, default=Path('reports/publication/device_stages.json'))
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.out.exists(): raise FileExistsError(args.out)
    budget = Budget(args.budget_ledger, 300, args.prior_ledger)
    status = 'failed'
    try:
        inspect(args.study, args.out, budget)
        status = 'completed'
    finally:
        budget.finish(status)
