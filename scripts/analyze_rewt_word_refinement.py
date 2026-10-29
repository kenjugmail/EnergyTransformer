"""Aggregate a frozen matched-seed comparison without selecting on test."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rewt.data import file_sha256
from rewt.experiment import save_json


def paired_interval(differences, seed=291999, repetitions=10000):
    """Paired bootstrap over both training seeds and shared held-out documents."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or not np.isfinite(differences).all():
        raise ValueError('Finite [training seed, document] differences required')
    rng = np.random.default_rng(seed); seeds, documents = differences.shape; means = []
    for start in range(0, repetitions, 100):
        n = min(100, repetitions-start)
        si = rng.integers(seeds, size=(n,seeds,1))
        di = rng.integers(documents, size=(n,1,documents))
        means.extend(differences[si,di].mean(axis=(1,2)).tolist())
    return dict(mean=float(differences.mean()), ci95=np.quantile(means,[.025,.975]).tolist(),
                seed_means=differences.mean(1).tolist(), training_seeds=seeds,
                documents=documents, bootstrap_repetitions=repetitions, bootstrap_seed=seed,
                method='Paired resampling of training seeds and documents; few training seeds limit uncertainty estimates.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, default=ROOT/'reports/rewt_word_refinement_20261029')
    a = p.parse_args(); protocol_path = a.report/'final_test_protocol.json'
    protocol = json.loads(protocol_path.read_text()); grouped = {}; models = {}
    for key, spec in protocol['models'].items():
        result = json.loads((a.report/'test'/f'{key}.json').read_text())
        if (result['protocol_sha256'] != file_sha256(protocol_path)
                or result['checkpoint_sha256'] != protocol['checkpoint_sha256'][key]
                or result['offsets'] != protocol['test_offsets']):
            raise ValueError('Evaluation is outside the frozen protocol')
        depth = result['selected_rounds']; score = result['scores'][str(depth)]
        models[key] = dict(arm=spec['arm'], seed=spec['seed'], selected_rounds=depth,
                           score={k:v for k,v in score.items() if not k.startswith('per_document')})
        grouped.setdefault(spec['arm'], {})[spec['seed']] = result
    aggregates = {}
    for arm, results in grouped.items():
        selected = [r['scores'][str(r['selected_rounds'])] for _,r in sorted(results.items())]
        means = {metric:float(np.mean([r[metric] for r in selected])) for metric in
                 ['token_nll','next_word_accuracy','next_word_top5_accuracy','word_label_nll','known_word_nll']}
        means['token_perplexity'] = math.exp(means['token_nll'])
        means['nll_seed_std'] = float(np.std([r['token_nll'] for r in selected],ddof=1)) if len(selected)>1 else None
        means['seeds'] = len(selected); aggregates[arm] = means
    comparisons = {}
    for arm in ('refined','hybrid'):
        if arm not in grouped: continue
        seeds = sorted(set(grouped[arm]) & set(grouped['transformer']))
        differences = []
        for seed in seeds:
            def values(name):
                row = grouped[name][seed]
                return np.asarray(row['scores'][str(row['selected_rounds'])]['per_document'])[:,0]
            differences.append(values(arm)-values('transformer'))
        comparisons[arm+'_minus_transformer'] = paired_interval(differences)
    if 'refined' in grouped and 'hybrid' in grouped:
        delta = []
        for seed in sorted(set(grouped['refined']) & set(grouped['hybrid'])):
            r, h = grouped['refined'][seed], grouped['hybrid'][seed]
            delta.append(np.asarray(r['scores'][str(r['selected_rounds'])]['per_document'])[:,0] -
                         np.asarray(h['scores'][str(h['selected_rounds'])]['per_document'])[:,0])
        comparisons['refined_minus_hybrid'] = paired_interval(delta)
    curves = {}
    if 'refined' in grouped:
        for early in (0,2):
            delta = [np.asarray(r['scores'][str(r['selected_rounds'])]['per_document'])[:,0] -
                     np.asarray(r['scores'][str(early)]['per_document'])[:,0] for r in grouped['refined'].values()]
            comparisons[f'refined_selected_minus_{early}'] = paired_interval(delta)
        for depth in ('0','1','2','4','8','16','32'):
            values = [r['scores'][depth]['token_nll'] for r in grouped['refined'].values()]
            curves[depth] = dict(mean=float(np.mean(values)), minimum=float(min(values)), maximum=float(max(values)),
                                values=values)
        for early, late in [(0,2),(2,8),(8,32),(0,32)]:
            delta = [np.asarray(r['scores'][str(late)]['per_document'])[:,0] -
                     np.asarray(r['scores'][str(early)]['per_document'])[:,0] for r in grouped['refined'].values()]
            comparisons[f'refined_{late}_minus_{early}'] = paired_interval(delta)
    result = dict(protocol_sha256=file_sha256(protocol_path), models=models, aggregates=aggregates,
                  comparisons=comparisons, refinement_curve=curves,
                  inference_depth_selection='Frozen validation choice; test curves are diagnostics only.',
                  scope='Whole-word TinyStories crops, 64-token context, fixed vocabulary; not general LLM superiority.')
    save_json(a.report/'comparison.json', result)
    print(json.dumps(dict(aggregates=aggregates, comparisons=comparisons),indent=2))


if __name__ == '__main__': main()
