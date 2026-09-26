"""Assemble compact reports and portable model bundles from frozen studies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from energy_real.local_energy_analysis import compact_report
from energy_real.local_energy_study import FitConfig, comparison
from energy_real.memory_eval import paired_interval
from energy_real.runtime import atomic_json, sha256


def assemble(study, discrete, out, bundle):
    summary = json.loads((study / 'summary.json').read_text())
    result = compact_report(study, out, probes=study / 'feature_probes.json')
    if (study / 'discrete_graph.json').exists():
        import statistics
        fixed = json.loads((study / 'discrete_graph.json').read_text())
        result['discrete_graph']['groups'] = {g: statistics.mean(r['correct'] for r in fixed['rows'] if r['group'] == g)
                                            for g in sorted({r['group'] for r in fixed['rows']})}
    for depth in sorted({r['config']['depth'] for r in summary['records'] if r['config']['task'] == 'fashion'}):
        result['comparisons'].append(comparison(summary['test_results'], 'fashion', 'pcalm', 'pc', depth=depth))
    for memory in (False, True):
        result['comparisons'].append(comparison(summary['test_results'], 'graph', 'pcalm', 'pc', memory_a=memory, memory_b=memory))
    # Likelihood is the byte model's primary validation metric. A positive
    # difference here means the EBM has higher loss (worse) than BP-CE.
    by_method = {method: sorted([t for t in summary['test_results'] if t['config']['task'] == 'bytes'
                  and t['config']['method'] == method], key=lambda t: t['config']['seed']) for method in ('energy', 'bp_ce')}
    if all(by_method.values()):
        arrays = {}
        for method, tests in by_method.items():
            arrays[method] = [[r['nll'] for r in t['trained']['rows']] for t in tests]
        import numpy as np
        a, b = np.array(arrays['energy']) / np.log(2), np.array(arrays['bp_ce']) / np.log(2)
        result['byte_energy_minus_bp_ce_bits'] = {'paired_interval': paired_interval(a.mean(0).tolist(), b.mean(0).tolist()),
                                                 'per_seed_differences': (a.mean(1) - b.mean(1)).tolist(),
                                                 'scope': '256 nonoverlapping target windows; residual text correlation limits this interval.'}
    discrete_result = json.loads((discrete / 'results.json').read_text())
    if discrete_result['status'] != 'completed':
        raise ValueError('Discrete study is incomplete')
    trimmed = copy.deepcopy(discrete_result)
    for record in trimmed['records']:
        for field in ('initial_validation', 'selected_validation'):
            record[field].pop('rows', None)
    for test in trimmed['tests']:
        for field in ('initial', 'trained'):
            test[field].pop('rows', None)
    trimmed['full_results_sha256'] = sha256(discrete / 'results.json')
    atomic_json(trimmed, out / 'discrete_results.json')
    result['learned_discrete'] = trimmed
    atomic_json(result, out / 'results.json')
    # Keep both original and compact inference manifests. Binaries remain in
    # ignored runs/. Each model has a relative folder with its best and initial.
    if bundle.exists():
        raise FileExistsError(bundle)
    bundle.mkdir(parents=True)
    selection = json.loads((study / 'selection.json').read_text())
    compact = {k: v for k, v in selection.items() if k != 'records'}
    compact['original_selection_sha256'] = sha256(study / 'selection.json')
    compact['records'] = []
    files = {}
    for record in selection['records']:
        name = FitConfig(**record['config']).name
        entry = {key: copy.deepcopy(record[key]) for key in
                 ('config', 'model_config', 'source_sha256', 'best_step', 'updates', 'parameter_count',
                  'best_sha256', 'initial_sha256')}
        entry['checkpoint'] = f'{name}/best.pt'
        for field in ('initial_validation', 'selected_validation'):
            entry[field] = {k: v for k, v in record[field].items() if k not in ('rows',)}
        compact['records'].append(entry)
        (bundle / name).mkdir()
        for checkpoint in ('initial', 'best'):
            source = study / name / f'{checkpoint}.pt'
            if sha256(source) != record[f'{checkpoint}_sha256']:
                raise ValueError('Checkpoint checksum changed before export')
            target = bundle / name / f'{checkpoint}.pt'
            shutil.copy2(source, target)
            files[str(target.relative_to(bundle))] = sha256(target)
    shutil.copy2(study / 'datasets.json', bundle / 'datasets.json')
    shutil.copy2(study / 'plan.json', bundle / 'plan.json')
    atomic_json(compact, bundle / 'selection.json')
    for name in ('datasets.json', 'plan.json', 'selection.json'):
        files[name] = sha256(bundle / name)
    for record in discrete_result['records']:
        for name in ('initial', 'best'):
            source = discrete / f'{name}_seed{record["seed"]}.pt'
            target = bundle / 'discrete' / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            files[str(target.relative_to(bundle))] = sha256(target)
    manifest = {'source_sha256': summary['source_sha256'], 'files': files,
                'original_neural_selection_sha256': sha256(study / 'selection.json'),
                'original_discrete_selection_sha256': sha256(discrete / 'selection.json')}
    atomic_json(manifest, bundle / 'bundle.json')
    atomic_json(manifest, out / 'bundle.json')
    atomic_json(compact, out / 'selection.json')
    shutil.copy2(study / 'datasets.json', out / 'datasets.json')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--discrete', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args()
    assemble(args.study, args.discrete, args.out, args.bundle)
