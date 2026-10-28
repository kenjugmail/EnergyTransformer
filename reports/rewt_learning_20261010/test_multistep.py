"""Fresh longer-chain evaluation of the validation-selected continuation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from rewt.data import file_sha256, task_batch
from rewt.evaluate import reasoning_evaluation, refinement_audit, runtime_profile
from rewt.experiment import save_json, save_lines
from rewt.position_memory import load_learning_checkpoint
from rewt.tasks import Task, task_manifest

PARENT = Path(__file__).resolve().parent
OUT = PARENT / 'multistep_test'


def read_tasks(path):
    return [Task(**json.loads(line)) for line in path.read_text().splitlines()]


def weights_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.shape).encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def groups(rows):
    result = {}
    for label, templates in [('known', (0, 1)), ('withheld_wording', (2,))]:
        for family in ('arithmetic', 'rules', 'recall'):
            for depth in sorted({r['steps'] for r in rows if r['family'] == family}):
                selected = [r for r in rows if r['template'] in templates and
                            r['family'] == family and r['steps'] == depth]
                if selected:
                    result[f'{label}/{family}/{depth}'] = dict(count=len(selected),
                        complete_correct=sum(r['both_correct'] for r in selected),
                        answer_correct=sum(r['answer_correct'] for r in selected),
                        complete_accuracy=sum(r['both_correct'] for r in selected) / len(selected))
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise FileExistsError('Keep the original locked fresh test results.')
    assert (PARENT / 'multistep_balanced/results.json').exists(), 'Finish the declared curriculum first.'
    candidates = []
    folders = ['multistep', 'multistep_position', 'multistep_huber', 'multistep_balanced']
    expected_valid = file_sha256(PARENT / 'multistep_balanced/tasks_valid.jsonl')
    for folder in folders:
        assert file_sha256(PARENT / folder / 'tasks_valid.jsonl') == expected_valid
        for path in sorted((PARENT / folder).glob('*.json')):
            record = json.loads(path.read_text())
            if 'best_validation_macro' in record:
                candidates.append(dict(experiment=folder, **record))
    selected = max(candidates, key=lambda r: r['best_validation_macro'])
    training = json.loads((PARENT / selected['experiment'] / 'plan.json').read_text())
    already_opened = {t.id for t in read_tasks(PARENT / 'foundation/tasks_test.jsonl')}
    prior = set(training['previous_test_exclusions']) | already_opened
    tasks = [t for t in read_tasks(PARENT / selected['experiment'] / 'tasks_test.jsonl') if t.id not in prior]
    assert len(tasks) == 960
    test_ids = {t.id for t in tasks}
    assert test_ids.isdisjoint(prior)
    for folder in folders:
        for split in ('train', 'valid'):
            assert test_ids.isdisjoint({t.id for t in read_tasks(PARENT / folder / f'tasks_{split}.jsonl')})
    assert file_sha256(Path(selected['checkpoint'])) == selected['checkpoint_sha256']
    plan = dict(created=datetime.now(timezone.utc).isoformat(), selected=selected, candidates=candidates,
        selection='Maximum known-wording validation macro across family/depth groups, using the same validation tasks for every candidate. No fresh test output is used for selection.',
        excluded_test_identities=sorted(prior), task_manifest=task_manifest(tasks),
        primary_policy=2, counterfactual_policy=0, generation_seed=2011,
        model_inputs='Raw ASCII prompts only; annotations, task payloads, expected traces, heads and solvers enter no inference call.',
        comparison='Zero refinement uses the same trained weights, not a separately trained control.',
        limitations='A selected continuation of one initialization lineage. Position-aware candidates have 6144 additional inference parameters. Independent-seed replication and broad instruction following remain unestablished.',
        source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
        script_sha256=file_sha256(Path(__file__)))
    save_json(OUT / 'plan.json', plan)
    save_lines(OUT / 'tasks_test.jsonl', [t.to_dict() for t in tasks])
    model, checkpoint = load_learning_checkpoint(selected['checkpoint'])
    model.eval()
    before = weights_sha(model)
    policies = []
    for rounds in (2, 0):
        start = time.process_time()
        summary, rows = reasoning_evaluation(model, tasks, seed=plan['generation_seed'], rounds=rounds)
        result = dict(rounds=rounds, summary=summary, groups=groups(rows),
                      cpu_seconds=time.process_time() - start)
        save_lines(OUT / f'rounds_{rounds}_outputs.jsonl', rows)
        policies.append(result)
        print(json.dumps(result), flush=True)
    batch = task_batch(tasks, range(4))
    audit = refinement_audit(model, batch.tokens)
    profile = runtime_profile(model, batch.tokens[:2, :32], repeats=3)
    after = weights_sha(model)
    assert before == after and file_sha256(Path(selected['checkpoint'])) == selected['checkpoint_sha256']
    assert audit['all_accepted_states_nonincreasing']
    for name, digest in plan['source_sha256'].items():
        assert file_sha256(ROOT / 'rewt' / name) == digest
    save_json(OUT / 'results.json', dict(selected=selected, policies=policies,
        parameters=model.parameters_count(), energy_audit=audit, runtime_profile=profile,
        weights_before=before, weights_after=after, weights_unchanged=before == after))


if __name__ == '__main__':
    main()
