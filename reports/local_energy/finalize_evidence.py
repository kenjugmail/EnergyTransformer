"""Export the fresh byte study and verify relocation of all frozen models."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from energy_real.local_energy_analysis import compact_report, frozen_record, generate_bytes
from energy_real.local_energy_study import Budget, FitConfig, load_model, make_data
from energy_real.memory_eval import paired_interval
from energy_real.runtime import atomic_json, sha256, source_identity


def main():
    torch.set_num_threads(2)
    root = Path('reports/local_energy')
    study = Path('runs/local_energy_byte_followup')
    out = root / 'byte_followup'
    result = compact_report(study, out)
    summary = json.loads((study / 'summary.json').read_text())
    tests = {method: sorted([t for t in summary['test_results'] if t['config']['method'] == method],
                           key=lambda t: t['config']['seed']) for method in ('energy', 'bp_ce')}
    arrays = {method: np.array([[r['nll'] for r in t['trained']['rows']] for t in rows]) / np.log(2)
              for method, rows in tests.items()}
    result['energy_minus_bp_ce_bits'] = {
        'paired_interval': paired_interval(arrays['energy'].mean(0).tolist(), arrays['bp_ce'].mean(0).tolist()),
        'per_seed_differences': (arrays['energy'].mean(1) - arrays['bp_ce'].mean(1)).tolist(),
        'scope': '1,000 nonoverlapping targets on fresh corpus segment; text correlation limits this interval.'}
    recipe = json.loads((study / 'plan.json').read_text())['recipes']
    args = argparse.Namespace(**recipe)
    args.byte_data = Path(args.byte_data)
    data = make_data('bytes', args)
    manifest = json.loads((study / 'datasets.json').read_text())['bytes']
    assert all(data[s].manifest() == manifest[s] for s in data)
    baselines = []
    for seed in recipe['seeds']:
        generator = torch.Generator().manual_seed(seed + 7000)
        order = torch.randperm(len(data['train'].y), generator=generator)
        offset = 0
        counts = torch.ones(256, dtype=torch.double)
        for _ in range(recipe['byte_steps']):
            if offset + recipe['byte_batch'] > len(order):
                order = torch.randperm(len(data['train'].y), generator=generator)
                offset = 0
            labels = data['train'].y[order[offset:offset + recipe['byte_batch']]]
            counts += torch.bincount(labels, minlength=256)
            offset += recipe['byte_batch']
        probability = counts / counts.sum()
        bits = -probability[data['test'].y].log2()
        baselines.append({'seed': seed, 'bits_per_target': float(bits.mean()),
                          'accuracy': float(data['test'].y.eq(counts.argmax()).float().mean()),
                          'target_exposures': recipe['byte_steps'] * recipe['byte_batch']})
    result['unigram_reference'] = {
        'scope': 'Post hoc fixed Laplace-1 unigram; exactly the same ordered training target exposures, no test tuning.',
        'mean_bits_per_target': float(np.mean([r['bits_per_target'] for r in baselines])), 'per_seed': baselines}
    atomic_json(result, out / 'results.json')
    bundle = Path('runs/local_energy_byte_bundle')
    bundle.mkdir(exist_ok=True)
    selection = json.loads((study / 'selection.json').read_text())
    compact = {k: copy.deepcopy(v) for k, v in selection.items() if k != 'records'}
    compact['records'] = []
    files = {}
    for record in selection['records']:
        name = FitConfig(**record['config']).name
        entry = {k: copy.deepcopy(record[k]) for k in ('config', 'model_config', 'source_sha256', 'best_step',
                                                      'updates', 'parameter_count', 'best_sha256', 'initial_sha256')}
        compact['records'].append(entry)
        (bundle / name).mkdir(exist_ok=True)
        for checkpoint in ('initial', 'best'):
            src, dst = study / name / f'{checkpoint}.pt', bundle / name / f'{checkpoint}.pt'
            assert sha256(src) == record[f'{checkpoint}_sha256']
            shutil.copy2(src, dst)
            files[str(dst.relative_to(bundle))] = sha256(dst)
    atomic_json(compact, bundle / 'selection.json')
    for name in ('datasets.json', 'plan.json'):
        shutil.copy2(study / name, bundle / name)
    for name in ('selection.json', 'datasets.json', 'plan.json'):
        files[name] = sha256(bundle / name)
        shutil.copy2(bundle / name, out / name)
    atomic_json({'source_sha256': source_identity(), 'files': files}, bundle / 'bundle.json')
    shutil.copy2(bundle / 'bundle.json', out / 'bundle.json')
    budget = Budget(Path('runs/local_energy_budget.json'), 300, Path('reports/publication/device_stages.json'))
    status = 'failed'
    try:
        generations = [generate_bytes(study, f'bytes_{method}_d3_w32_mem0_seed11', 'The world ', 32, budget)
                       for method in ('bp_ce', 'energy')]
        atomic_json(generations, out / 'generations.json')
        verified = []
        for origin, exported in ((Path('runs/local_energy_main'), Path('runs/local_energy_bundle')),
                                 (study, bundle)):
            moved = exported.with_name(exported.name + '_relocated')
            if exported.exists():
                exported.rename(moved)
            hashes = json.loads((moved / 'bundle.json').read_text())['files']
            assert all(sha256(moved / p) == h for p, h in hashes.items())
            selected = json.loads((moved / 'selection.json').read_text())
            for record in selected['records']:
                budget.check()
                config = FitConfig(**record['config'])
                _, old = frozen_record(origin, config.name)
                _, new = frozen_record(moved, config.name)
                a, b = load_model(old), load_model(new)
                assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
                # Numerical inference as well as checksum equality after moving.
                x = torch.zeros(1, a.config.inputs)
                memory = (torch.zeros(1, a.config.memory_keys, a.config.memory_values)
                          if a.config.memory_keys else None)
                with torch.no_grad():
                    expected, actual = a.feedforward(x, memory), b.feedforward(x, memory)
                assert torch.equal(expected, actual) and torch.isfinite(actual).all()
                verified.append({'study': str(origin), 'fit': config.name, 'sha256': sha256(new)})
        atomic_json({'source_sha256': source_identity(), 'verified_neural_models': len(verified),
                     'checks': 'All exported file hashes; all neural selected weights and numerical feedforward outputs match originals after directory relocation.',
                     'records': verified}, root / 'relocation.json')
        status = 'completed'
    finally:
        budget.finish(status)
    for row in result['rows']:
        print(row['method'], row['mean_bits_per_target'], row['mean_accuracy'])
    print(json.dumps(result['energy_minus_bp_ce_bits']))
    print(json.dumps(result['unigram_reference']))
    print(json.dumps(generations))


if __name__ == '__main__':
    main()
