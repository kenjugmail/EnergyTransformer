"""Preregistered cold-start architecture comparison on disjoint semantic tasks.

Run --prepare, then --train, then --test. Test files never enter fit or selection.
Each case shares per-seed initialization of common tensors, exact batches, target
labels and update count. Model size and measured CPU work are reported separately.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM, load_architecture_checkpoint
from rewt.curriculum import generate_curriculum, render_task, training_annotations
from rewt.data import task_batch, file_sha256
from rewt.experiment import save_checkpoint, save_json, save_lines, event
from rewt.tasks import Task, task_manifest
from train_rewt_curriculum import auxiliary_losses, evaluate

OUT = ROOT / 'reports/rewt_architecture_20261022'
RUNS = ROOT / 'runs/rewt_architecture_20261022'
VARIANTS = {
    'control': {},
    'certified': dict(solver='certified', step_size=.75),
    'precond2': dict(solver='preconditioned', step_size=.75),
    'precond4': dict(solver='preconditioned', step_size=.75, rounds=4),
    'precond8': dict(solver='preconditioned', step_size=.75, rounds=8),
    'scan': dict(backbone='scan'),
    'feedback8': dict(backbone='scan', feedback_stride=8),
    'delta': dict(backbone='scan', delta_size=16),
    'patch4': dict(backbone='scan', patch_stride=4),
    'diversity': dict(),
    'hybrid_modes': dict(mode_layout='hybrid'),
    'lowrank16': dict(backbone='scan', output_rank=16),
    'combined': dict(backbone='scan', delta_size=16, feedback_stride=8,
                     solver='preconditioned', step_size=.75, rounds=4),
}


def read_tasks(path):
    return [Task(**json.loads(line)) for line in Path(path).read_text().splitlines()]


def prepare(updates):
    path = OUT / 'plan.json'
    if path.exists():
        raise FileExistsError('preserve existing study; use --train or --test')
    excluded = set(); previous = []
    for p in (ROOT / 'reports').rglob('tasks_test.jsonl'):
        if OUT in p.parents:
            continue
        ids = {t.id for t in read_tasks(p)}
        excluded |= ids
        previous.append(dict(path=str(p), sha256=file_sha256(p), identities=len(ids)))
    tasks = {s: generate_curriculum(s, n, seed=20261022,
             exclude_ids=excluded if s == 'test' else ()) for s, n in [('train', 600), ('valid', 9), ('test', 18)]}
    ids = {s: {t.id for t in rows} for s, rows in tasks.items()}
    assert ids['test'].isdisjoint(excluded | ids['train'] | ids['valid'])
    assert ids['train'].isdisjoint(ids['valid'])
    for s, rows in tasks.items():
        save_lines(OUT / f'tasks_{s}.jsonl', [t.to_dict() for t in rows])
    cfg = ArchitectureConfig(width=48, hidden=736, modes=32, history=128, position_period=128,
        context_kernel=3, recurrent_type='gru', cache_history=True, output_geometry='centered',
        normalize_frequency_blocks=True, compatibility_geometry='anchor_relative', wave_geometry='anchor_relative',
        initial_carry_scale=0., early_backtrack=True, attention_heads=2)
    cases = [dict(name=f'{variant}_{seed}', variant=variant, seed=seed, overrides=overrides,
                  diversity_weight=.03 if variant == 'diversity' else 0.)
             for variant, overrides in VARIANTS.items() for seed in (22101, 22102, 22103)]
    plan = dict(updates=updates, batch_size=24, seeds=[22101, 22102, 22103], cases=cases,
        config=asdict(cfg), validation_every=256, state_gradient_interval=32,
        learning_rate=.001, task_manifests={s: task_manifest(rows) for s, rows in tasks.items()},
        previous_tests=previous, excluded_previous_test_identities=len(excluded),
        data_sha256={s: file_sha256(OUT / f'tasks_{s}.jsonl') for s in tasks},
        selection='Per-seed checkpoint selected by validation macro complete accuracy; lower validation response CE breaks ties. All preregistered variants reported on fresh test once training is complete.',
        protocol='Cold starts; common tensors initialized identically per seed. Same per-step batches and supervision. All three reasoning depths sampled from step 256. Known and held-out wording evaluated separately. Training-only auxiliary heads never enter generation.',
        limitations='Same width/hidden/modes, not exact parameter matching for changed architectures. Equal updates/tokens, not equal compute. CPU process time measured under concurrent single-thread workers; isolated inference benchmarks separate.',
        source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in [*sorted((ROOT/'rewt').glob('*.py')), Path(__file__)]},
        test_not_evaluated=True)
    save_json(path, plan)
    return plan


def make_model(case, plan):
    torch.manual_seed(case['seed'])
    model = ArchitectureEnergyLM(ArchitectureConfig(**dict(plan['config'], **case['overrides'])))
    torch.manual_seed(case['seed'] + 100)
    d = model.config.width
    heads = nn.ModuleDict(dict(number=nn.Linear(d, 1), addition=nn.Linear(3*d, 1),
                              query=nn.Linear(d, 256), progress=nn.Linear(3*d, 4)))
    return model, heads


def choose_batch(train, pools, seed, step):
    rng = np.random.default_rng(seed * 1000000 + step + 221022)
    maximum = 1 if step < 128 else 2 if step < 256 else 3
    chosen = []
    for family in ('arithmetic', 'rules', 'recall'):
        depths = (4, 8, 16) if family == 'recall' else tuple(range(1, maximum + 1))
        for j in range(8):
            t = train[int(rng.choice(pools[family, depths[j % len(depths)]]))]
            chosen.append(render_task(t.payload, 'train', int(rng.integers(5)), seed=int(rng.integers(2**32))))
    rng.shuffle(chosen)
    return chosen


@torch.inference_mode()
def validation(model, tasks, seed):
    model.eval().prepare_inference()
    score, rows = evaluate(model, tasks, seed)
    nll = count = 0
    for i in range(0, len(tasks), 24):
        batch = task_batch(tasks, range(i, min(i + 24, len(tasks))))
        logits = model(batch.tokens).logits
        nll += float(F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100, reduction='sum'))
        count += int(batch.valid.sum())
    score['response_ce'] = nll / count
    return score, rows


def fit(case, plan):
    torch.set_num_threads(1)
    directory = RUNS / case['name']; directory.mkdir(parents=True, exist_ok=True)
    result_path = OUT / f'{case["name"]}.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    if (directory / 'last.pt').exists():
        raise RuntimeError('partial run exists; inspect before restarting')
    model, heads = make_model(case, plan)
    params = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=plan['learning_rate'], betas=(.9, .95), weight_decay=.01)
    train, valid = read_tasks(OUT/'tasks_train.jsonl'), read_tasks(OUT/'tasks_valid.jsonl')
    pools = {(f, d): [i for i, t in enumerate(train) if t.family == f and t.steps == d]
             for f in ('arithmetic', 'rules', 'recall') for d in ((4, 8, 16) if f == 'recall' else (1, 2, 3))}
    best = (-1., -float('inf')); best_step = 0
    cpu = validation_cpu = 0.; maximum = 0.; tokens = 0; trace = []
    event('architecture_started', case=case['name'], parameters=model.parameters_count())
    for step in range(plan['updates']):
        start = time.process_time()
        tasks = choose_batch(train, pools, case['seed'], step)
        batch = task_batch(tasks, range(len(tasks)), semantic_weight=16.)
        # Termination and separators receive explicit training weight, identical
        # across arms. No symbolic check participates in the model forward.
        for row, task in enumerate(tasks):
            for b in training_annotations(task)[0]:
                batch.token_weights[row, b.end] = 16.
            batch.token_weights[row, len(task.prompt + task.response) - 2] = 32.
        model.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        rate = plan['learning_rate'] * min(1., (step+1)/32) * (.3 + .7*(1+math.cos(math.pi*step/plan['updates']))/2)
        for group in optimizer.param_groups:
            group['lr'] = rate
        output = model(batch.tokens, details=True, trace_energy=False, state_gradient_interval=32)
        per_byte = F.cross_entropy(output.logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100, reduction='none').reshape_as(batch.targets)
        ce = (per_byte * batch.token_weights).sum() / batch.token_weights.sum()
        aux = auxiliary_losses(model, heads, output, batch.tokens, tasks)
        ramp = min(1., (step+1)/256)
        weights = dict(pointer=1., query=.5, copy=.5, number=.15*ramp, addition=.15*ramp, progress=1., binding=.2)
        diversity = model.diversity_loss()
        loss = ce + sum(weights[k]*v for k, v in aux.items()) + case['diversity_weight'] * diversity
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{case["name"]}/{step} objective')
        loss.backward()
        grad = float(nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        optimizer.step()
        cpu += time.process_time() - start; maximum = max(maximum, grad); tokens += int(batch.valid.sum())
        row = dict(step=step+1, loss=float(loss.detach()), ce=float(ce.detach()), diversity=float(diversity.detach()), gradient=grad)
        if (step+1) % plan['validation_every'] == 0 or step+1 == plan['updates']:
            start = time.process_time(); score, rows = validation(model, valid, case['seed'])
            validation_cpu += time.process_time() - start
            rank = (score['macro_complete'], -score['response_ce'])
            saved = dict(arm='architecture_energy', config=asdict(model.config), seed=case['seed'],
                         weights=model.state_dict(), step=step+1, selection=score, case=case)
            if rank > best:
                best, best_step = rank, step+1
                save_checkpoint(directory/'best.pt', saved)
                save_lines(OUT/f'{case["name"]}_validation_outputs.jsonl', rows)
            save_checkpoint(directory/'last.pt', dict(saved, optimizer=optimizer.state_dict(), training_heads=heads.state_dict()))
            row['validation'] = score
            event('architecture_validation', case=case['name'], step=step+1, complete=score['macro_complete'], best=best[0], ce=score['response_ce'])
        elif (step+1) % 64 == 0:
            event('architecture_progress', case=case['name'], step=step+1, ce=row['ce'])
        trace.append(row)
        if (step+1) % 64 == 0:
            save_lines(directory/'trace.jsonl', trace)
    save_lines(directory/'trace.jsonl', trace)
    result = dict(case=case, parameters=model.parameters_count(), auxiliary_parameters=sum(p.numel() for p in heads.parameters()),
                  best_step=best_step, validation_complete=best[0], validation_ce=-best[1], training_cpu_seconds=cpu,
                  validation_cpu_seconds=validation_cpu, maximum_gradient=maximum, response_tokens=tokens,
                  checkpoint=str(directory/'best.pt'), checkpoint_sha256=file_sha256(directory/'best.pt'))
    save_json(result_path, result)
    return result


def test_case(case):
    torch.set_num_threads(1)
    result = json.loads((OUT / f'{case["name"]}.json').read_text())
    model, _ = load_architecture_checkpoint(result['checkpoint'])
    before = {n: t.clone() for n, t in model.state_dict().items()}
    score, rows = validation(model, read_tasks(OUT / 'tasks_test.jsonl'), case['seed'])
    assert all(torch.equal(t, before[n]) for n, t in model.state_dict().items())
    save_lines(OUT / f'{case["name"]}_test_outputs.jsonl', rows)
    result['test'] = score; result['inference_weights_unchanged'] = True
    save_json(OUT / f'{case["name"]}_test.json', result)
    event('architecture_test', case=case['name'], complete=score['macro_complete'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true'); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--updates', type=int, default=1024); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--variants', nargs='*')
    args = parser.parse_args(); torch.set_num_threads(1)
    plan = prepare(args.updates) if args.prepare else json.loads((OUT/'plan.json').read_text())
    cases = [c for c in plan['cases'] if not args.variants or c['variant'] in args.variants]
    if args.smoke:
        tasks = read_tasks(OUT/'tasks_train.jsonl')[::600]
        batch = task_batch(tasks, range(len(tasks)))
        for case in cases[::3]:
            model, heads = make_model(case, plan)
            begin = time.process_time()
            output = model(batch.tokens, details=True, trace_energy=False, state_gradient_interval=32)
            loss = output.logits.square().mean() + sum(auxiliary_losses(model, heads, output, batch.tokens, tasks).values())
            loss.backward()
            event('architecture_smoke', case=case['name'], seconds=time.process_time()-begin, loss=float(loss.detach()))
    if args.train or args.test:
        if args.test and not all((OUT/f'{c["name"]}.json').exists() for c in plan['cases']):
            raise RuntimeError('all preregistered fits must finish before opening test')
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(fit, c, plan) if args.train else pool.submit(test_case, c) for c in cases]
            results = [f.result() for f in as_completed(futures)]
        save_json(OUT/('training_results.json' if args.train else 'test_results.json'), results)


if __name__ == '__main__':
    main()
