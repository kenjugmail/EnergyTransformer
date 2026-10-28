"""Audit saved generations and summarize the frozen broader-task study."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.tasks import score_response
from train_rewt_curriculum import read_tasks


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def describe(rows):
    metrics = ('valid_format', 'answer_correct', 'steps_correct', 'both_correct')
    return dict(count=len(rows), semantic_identities=len({r['id'] for r in rows}),
                **{k: dict(correct=sum(r[k] for r in rows),
                           fraction=sum(r[k] for r in rows) / len(rows)) for k in metrics})


def paired_uncertainty(left, right, seed=7101, repeats=5000):
    """Stratified semantic-ID bootstrap; both wordings stay together."""
    lhs = {(r['id'], r['template']): r for r in left}
    rhs = {(r['id'], r['template']): r for r in right}
    assert lhs.keys() == rhs.keys()
    groups = defaultdict(lambda: defaultdict(list))
    for key, row in lhs.items():
        groups[row['family'], row['steps']][row['id']].append(
            int(row['both_correct']) - int(rhs[key]['both_correct']))
    rng = np.random.default_rng(seed)
    draws, point = [], []
    for identities in groups.values():
        values = np.array([np.mean(v) for v in identities.values()])
        point.append(float(values.mean()))
        indices = rng.integers(len(values), size=(repeats, len(values)))
        draws.append(values[indices].mean(1))
    distribution = np.mean(draws, axis=0)
    return dict(complete_accuracy_difference=float(np.mean(point)),
                conditional_95_percent_interval=np.quantile(distribution, [.025, .975]).tolist(),
                repetitions=repeats, seed=seed,
                scope='Fixed selected checkpoints; test-sample uncertainty only. Semantic identities are resampled within family/depth, with paired wording and model outputs kept together. Does not cover training-seed or model-selection uncertainty.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    study, test, out = (p.resolve() for p in (args.study_dir, args.test_dir, args.output_dir))
    plan = json.loads((test / 'plan.json').read_text())
    results = json.loads((test / 'results.json').read_text())
    training = json.loads((study / 'results.json').read_text())
    study_plan = json.loads((study / 'plan.json').read_text())
    main_training = json.loads((Path(study_plan['base_study']) / 'results.json').read_text())
    tasks = read_tasks(test / 'tasks_test.jsonl')
    expected = {(t.id, t.template): t for t in tasks}
    assert len(expected) == len(tasks) == plan['task_count']
    summaries, outputs = {}, {}
    for result in results:
        name = result['case']['name']
        assert result['checkpoint_sha256'] == plan['checkpoints_sha256'][name]
        assert result['unchanged_weights'] and result['unchanged_checkpoint']
        path = test / f'{name}_outputs.jsonl'
        rows = read_rows(path)
        assert len(rows) == len(expected)
        assert {(r['id'], r['template']) for r in rows} == expected.keys()
        for row in rows:
            task = expected[row['id'], row['template']]
            assert row['prompt'] == task.prompt
            assert bytes.fromhex(row['generated_hex']).decode('ascii', errors='replace') == row['response']
            assert score_response(task, row['response']) == {k: row[k] for k in score_response(task, row['response'])}
        grouped = defaultdict(list)
        for row in rows:
            grouped[f"{row['family']}/{row['steps']}/{row['difficulty']}"].append(row)
        groups = {k: describe(v) for k, v in sorted(grouped.items())}
        macro = statistics.mean(g['both_correct']['fraction'] for g in groups.values())
        assert abs(macro - result['summary']['macro_complete']) < 1e-12
        by_family = {f: describe([r for r in rows if r['family'] == f])
                     for f in ('arithmetic', 'rules', 'recall')}
        by_family_wording = {f'{f}/{w}': describe([r for r in rows if r['family'] == f and r['difficulty'] == w])
                            for f in ('arithmetic', 'rules', 'recall')
                            for w in ('familiar_format', 'withheld_wording')}
        summaries[name] = dict(all=describe(rows), macro_complete=macro, groups=groups,
                               by_family=by_family, by_family_wording=by_family_wording,
                               generation_and_checking_cpu_seconds=result['generation_and_checking_cpu_seconds'],
                               forward_only_profile=result['profile'],
                               outputs_sha256=file_sha256(path))
        outputs[name] = rows
    cold = {}
    for prefix in ('energy_', 'transformer_'):
        names = sorted(n for n in summaries if n.startswith(prefix))
        assert len(names) == 3
        cold[prefix.rstrip('_')] = dict(
            cases=names, macro_seed_scores=[summaries[n]['macro_complete'] for n in names],
            mean_macro=statistics.mean(summaries[n]['macro_complete'] for n in names),
            family_wording_seed_scores={k: [summaries[n]['by_family_wording'][k]['both_correct']['fraction']
                                            for n in names] for k in summaries[names[0]]['by_family_wording']})
    ledger = [dict(name=r['case']['name'], warm=r['case']['warm'],
                   inference_parameters=r['parameters'], auxiliary_parameters=r['auxiliary_parameters'],
                   source_best_step=r['case']['base_record']['best_step'], selected_additional_step=r['best_step'],
                   charged_updates=study_plan['updates'] + study_plan['base_plan']['updates'],
                   selected_validation_complete=r['best_validation_macro'],
                   training_cpu_seconds=r['total_training_cpu_seconds'],
                   validation_cpu_seconds=r['total_validation_cpu_seconds'],
                   processed_response_tokens=r['processed_response_tokens'] + r['case']['base_record']['processed_response_tokens'],
                   maximum_raw_gradient_norm=max(r['maximum_raw_gradient_norm'], r['case']['base_record']['maximum_raw_gradient_norm']),
                   checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256']) for r in training]
    for seed in (2101, 2102, 2103):
        matched = [r for r in ledger if not r['warm'] and str(seed) in r['name']]
        assert len(matched) == 3
        assert len({r['processed_response_tokens'] for r in matched}) == 1
    comparisons = {name: paired_uncertainty(outputs['selected_warm'], outputs[name])
                   for name in ('selected_warm_zero_rounds', 'previous_warm_parent')}
    record = dict(test_plan_sha256=file_sha256(test / 'plan.json'),
                  test_results_sha256=file_sha256(test / 'results.json'), cases=summaries,
                  cold_seed_results=cold, completed_fit_ledger=ledger, paired_comparisons=comparisons,
                  aggregate_executed_training_cpu_seconds=sum(r['training_cpu_seconds'] for r in main_training+training),
                  aggregate_executed_validation_cpu_seconds=sum(r['validation_cpu_seconds'] for r in main_training+training),
                  cost_scope='Ledger entries are per-model main-plus-termination lineage costs; absolute-wave warm alternatives share their main fits, so summing ledger entries would double-count them. Aggregate executed totals count each main and termination fit once. Warm parent-lineage training, data preparation, test evaluation, compilation probes, smoke runs and interrupted harness work are excluded. Cold budgets match updates and examples, not CPU time. Both Transformer learning-rate screen candidates are retained.',
                  inference_scope='Raw-byte prompts and fixed learned model weights. Auxiliary heads, task payloads and symbolic checkers are absent from generation.')
    save_json(out / 'summary.json', record)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    families = ('arithmetic', 'rules', 'recall')
    for ax, wording in zip(axes, ('familiar_format', 'withheld_wording')):
        for offset, (kind, color) in zip((-.17, .17), (('energy', '#276c9f'), ('transformer', '#db7736'))):
            values = np.array([cold[kind]['family_wording_seed_scores'][f'{f}/{wording}'] for f in families]) * 100
            positions = np.arange(3) + offset
            ax.bar(positions, values.mean(1), width=.3, color=color, alpha=.75,
                   label=f'{kind.capitalize()}, 3 cold seeds')
            for column in range(3):
                ax.scatter(positions + (column-1)*.05, values[:, column], color=color, s=18, edgecolor='white', linewidth=.5)
        ax.set_xticks(np.arange(3), ['Addition\n1–3 steps', 'Rules\n1–3 steps', 'Recall\n4/8/16 distractors'])
        ax.set_ylim(0, 105)
        ax.set_title('Familiar formats' if wording == 'familiar_format' else 'New held-out wording')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Whole trace + final answer correct (%)')
    axes[1].legend(loc='upper left', fontsize=8)
    fig.suptitle('Fresh broader-task test: matched examples and updates, unequal CPU cost', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'cold_seed_results.png', dpi=180)
    fig.savefig(out / 'cold_seed_results.pdf')
    plt.close(fig)
    print(json.dumps({k: v['mean_macro'] for k, v in cold.items()}))


if __name__ == '__main__':
    main()
