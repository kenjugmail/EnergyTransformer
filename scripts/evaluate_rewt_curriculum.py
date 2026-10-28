"""Lock validation selections before scoring fresh broader-task generations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256
from rewt.evaluate import refinement_audit, runtime_profile
from rewt.experiment import save_json, save_lines
from rewt.position_memory import load_learning_checkpoint
from rewt.tasks import Task
from train_rewt_curriculum import evaluate, read_tasks


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def select(study):
    results = json.loads((study / 'results.json').read_text())
    warm_cases = sorted((r for r in results if r['case']['warm']), key=lambda r: r['case']['name'])
    warm = max(warm_cases, key=lambda r: r['best_validation_macro'])
    rates = sorted({r['case']['learning_rate'] for r in results if r['case']['family'] == 'transformer'})
    rate_scores = {rate: statistics.mean(r['best_validation_macro'] for r in results
                                        if r['case']['family'] == 'transformer' and r['case']['learning_rate'] == rate)
                   for rate in rates}
    chosen_rate = max(rates, key=lambda rate: rate_scores[rate])
    cases = [dict(name='selected_warm', checkpoint=warm['checkpoint'], rounds=2, selection=warm)]
    cases.append(dict(name='selected_warm_zero_rounds', checkpoint=warm['checkpoint'], rounds=0, selection=warm,
                      scope='Same trained weights; a refinement counterfactual, not a separate architecture.'))
    for result in results:
        case = result['case']
        if case['warm']:
            continue
        if case['family'] == 'energy' or case['learning_rate'] == chosen_rate:
            cases.append(dict(name=case['name'], checkpoint=result['checkpoint'],
                              rounds=2 if case['family'] == 'energy' else None, selection=result))
    plan = json.loads((study / 'plan.json').read_text())
    parent = plan.get('parent', plan.get('base_plan', {}).get('parent'))
    cases.append(dict(name='previous_warm_parent', checkpoint=parent['path'], rounds=2,
                      scope='Previous selected task checkpoint, scored on the same fresh tasks without training.'))
    return cases, dict(selected_transformer_learning_rate=chosen_rate, validation_rate_scores=rate_scores,
                       selected_warm=warm['case']['name'])


def score_case(case, plan):
    torch.set_num_threads(1)
    path = Path(case['checkpoint'])
    model, _ = load_learning_checkpoint(path)
    before, file_before = state_hash(model), file_sha256(path)
    assert file_before == plan['checkpoints_sha256'][case['name']]
    tasks = read_tasks(plan['tasks_path'])
    kwargs = {} if case['rounds'] is None else dict(rounds=case['rounds'])
    cpu, wall = time.process_time(), time.perf_counter()
    summary, rows = evaluate(model, tasks, 6101) if not kwargs else evaluate_with_rounds(model, tasks, kwargs)
    generation_cpu, generation_wall = time.process_time()-cpu, time.perf_counter()-wall
    tokens = torch.tensor([list(tasks[0].prompt.encode('ascii'))], dtype=torch.long)
    audit = None if case['rounds'] == 0 else refinement_audit(model, tokens)
    probe = torch.tensor([list(bytes.fromhex(plan['forward_profile_prompt_hex']))], dtype=torch.long)
    assert probe.shape == (1, 32)
    profile = None if case['rounds'] == 0 else runtime_profile(model, probe, repeats=3)
    after, file_after = state_hash(model), file_sha256(path)
    assert before == after and file_before == file_after
    out = Path(plan['output_dir'])
    save_lines(out / f'{case["name"]}_outputs.jsonl', rows)
    result = dict(case=case, summary=summary, inference_parameters=model.parameters_count(),
                  checkpoint_sha256=file_before, state_sha256_before=before, state_sha256_after=after,
                  unchanged_weights=True, unchanged_checkpoint=True, energy_audit=audit, profile=profile)
    result['generation_and_checking_cpu_seconds'] = generation_cpu
    result['generation_and_checking_wall_seconds'] = generation_wall
    result['generation_cost_scope'] = 'All prompt prefill, greedy generation until newline or declared limit, and post-generation checking. Excludes checkpoint loading, energy audit, forward-only profiling and output writes. Wall time includes worker contention; CPU time is process CPU.'
    result['forward_profile_prompt_hex'] = plan['forward_profile_prompt_hex']
    save_json(out / f'{case["name"]}.json', result)
    return result


def evaluate_with_rounds(model, tasks, kwargs):
    from rewt.evaluate import reasoning_evaluation
    summary, rows = reasoning_evaluation(model, tasks, seed=6101, **kwargs)
    groups = {}
    for family in ('arithmetic', 'rules', 'recall'):
        for depth in ((4, 8, 16) if family == 'recall' else (1, 2, 3)):
            for kind in ('familiar_format', 'withheld_wording'):
                subset = [r for r in rows if r['family'] == family and r['steps'] == depth and r['difficulty'] == kind]
                if subset:
                    groups[f'{family}/{depth}/{kind}'] = sum(r['both_correct'] for r in subset)/len(subset)
    return dict(macro_complete=statistics.mean(groups.values()), groups=groups, summary=summary), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    study, out = args.study_dir.resolve(), args.output_dir.resolve()
    if (out / 'plan.json').exists():
        raise FileExistsError('the frozen test is already planned; use its saved outputs')
    cases, choice = select(study)
    source_plan = json.loads((study / 'plan.json').read_text())
    for name, sha in source_plan['source_sha256'].items():
        assert file_sha256(ROOT / 'rewt' / name) == sha
    for case in cases:
        selection = case.get('selection')
        if selection:
            assert file_sha256(Path(case['checkpoint'])) == selection['checkpoint_sha256']
    parent = source_plan.get('parent', source_plan.get('base_plan', {}).get('parent'))
    assert file_sha256(Path(parent['path'])) == parent['sha256']
    tasks_path = study / 'tasks_test.jsonl'
    train, valid, tasks = (read_tasks(study / f'tasks_{s}.jsonl') for s in ('train', 'valid', 'test'))
    previous = set(source_plan.get('excluded_previous_test_identities',
                                   source_plan.get('base_plan', {}).get('excluded_previous_test_identities', [])))
    ids = {t.id for t in tasks}
    assert ids.isdisjoint({t.id for t in train+valid} | previous)
    assert {t.template for t in tasks} <= {0, 1, 2, 3, 4, 7, 8}
    assert all(t.template not in (7, 8) for t in train+valid)
    plan = dict(created=datetime.now(timezone.utc).isoformat(), cases=cases, selection=choice,
                study=str(study), study_results_sha256=file_sha256(study / 'results.json'),
                tasks_path=str(tasks_path), tasks_sha256=file_sha256(tasks_path),
                task_count=len(tasks), semantic_identities=len(ids), output_dir=str(out),
                forward_profile_prompt_hex=tasks[0].prompt.encode('ascii')[:32].ljust(32, b' ').hex(),
                forward_profile_scope='Single-row 32-byte prompt prefix, padded with spaces if needed. Same raw input for every profiled model. Forward-only CPU timing after warmup; not cached decode throughput.',
                checkpoints_sha256={c['name']: file_sha256(Path(c['checkpoint'])) for c in cases},
                source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
                script_sha256=file_sha256(Path(__file__)),
                policy='Greedy raw-byte generation, max40 bytes within128-byte window; checker only after generation. All selections locked before fresh generation. Validation ties use lexicographic warm-case name and smaller Transformer learning rate. Whole trace plus final answer must match.',
                prior_test_identity_exclusions=len(previous))
    save_json(out / 'plan.json', plan)
    save_lines(out / 'tasks_test.jsonl', [t.to_dict() for t in tasks])
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(score_case, case, plan) for case in cases]
        results = [future.result() for future in as_completed(futures)]
    assert file_sha256(tasks_path) == plan['tasks_sha256']
    for name, sha in plan['source_sha256'].items():
        assert file_sha256(ROOT / 'rewt' / name) == sha
    save_json(out / 'results.json', results)
    print(json.dumps({r['case']['name']: r['summary']['macro_complete'] for r in results}))


if __name__ == '__main__':
    main()
