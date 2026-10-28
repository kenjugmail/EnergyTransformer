"""CPU-compute-matched comparison with equal validation-only LR search.

Timing is process CPU time for forward/backward/optimizer updates, not FLOPs
or accelerator compute. Data loading, validation, and checkpoint I/O are
excluded equally. Inputs share a minibatch-stream prefix, but faster models
receive more bytes. Selection is frozen before fresh held-out evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from .bench import balanced_order
from .competitive_study import configuration, fixed_data, paired_interval, score
from .data import Enwik8, LOCAL_CORPUS_SHA256
from .local_energy_study import Budget
from .model import Config, LM
from .runtime import atomic_checkpoint, atomic_json, environment, sha256, source_identity

ARMS = ('softmax', 'local_softmax', 'hybrid')


def compute_lr(progress, peak):
    return peak * min(1., max(progress, 1e-4) / .04) * (.1 + .45 * (1 + math.cos(math.pi * min(progress, 1.))))


def objective_for(model):
    # A separate closure cell per model prevents late binding between fits.
    def objective(x, y):
        return F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten())
    return objective


def fit_group(specs, seconds, training, valid, budget, out, stage, *,
              quantum=0., compile_model=False, foreach=False):
    jobs = {}
    for arm, seed, lr in specs:
        name = f'{stage}_{arm}_{seed}_{lr:g}'
        torch.manual_seed(seed)
        cfg = replace(configuration(arm), chunk=8)
        model = LM(cfg)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': .1},
            {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.}],
            lr=lr, betas=(.9, .95), foreach=foreach)
        objective = objective_for(model)
        setup_wall = 0.
        if compile_model:
            budget.check()
            objective = torch.compile(objective, fullgraph=True)
            # Warm compilation and autograd without updating any parameter or
            # consuming the training minibatch stream. Record the one-off cost.
            # Match Enwik8.batch's overlapping, noncontiguous input/target
            # views; otherwise the first real batch triggers recompilation.
            warm = torch.zeros(8, 129, dtype=torch.long)
            begin = time.perf_counter()
            objective(warm[:, :-1], warm[:, 1:]).backward()
            model.zero_grad(set_to_none=True)
            setup_wall = time.perf_counter() - begin
            budget.check()
        jobs[name] = dict(arm=arm, seed=seed, lr=lr, model=model, optimizer=optimizer,
                          rng=np.random.default_rng(seed + 91100), cpu_s=0., wall_s=0.,
                          steps=0, next_fraction=.25, best_bpb=float('inf'), history=[],
                          objective=objective, compile_setup_wall_s=setup_wall, max_update_cpu_s=0.)
    round_index = 0
    scheduler = []
    while any(job['cpu_s'] < seconds for job in jobs.values()):
        for name in balanced_order(list(jobs), round_index):
            job = jobs[name]
            if job['cpu_s'] >= seconds:
                continue
            target = min(seconds, (round_index + 1) * quantum) if quantum else seconds
            while job['cpu_s'] < target:
                budget.check()
                x, y = training.batch(job['rng'], 8, 128)
                model, optimizer = job['model'], job['optimizer']
                model.train()
                begin_cpu, begin_wall = time.process_time(), time.perf_counter()
                progress = (job['cpu_s'] + job['cpu_s'] / max(job['steps'], 1)) / seconds
                for group in optimizer.param_groups:
                    group['lr'] = compute_lr(progress, job['lr'])
                optimizer.zero_grad(set_to_none=True)
                loss = job['objective'](x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(name)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                update_cpu = time.process_time() - begin_cpu
                job['cpu_s'] += update_cpu
                job['max_update_cpu_s'] = max(job['max_update_cpu_s'], update_cpu)
                job['wall_s'] += time.perf_counter() - begin_wall
                job['steps'] += 1
                if job['cpu_s'] >= seconds * job['next_fraction']:
                    result = score(model, valid, budget)
                    record = {k: job[k] for k in ('steps', 'cpu_s', 'wall_s')}
                    record.update(valid_bpb=result['bpb'], last_train_nll=float(loss.detach()),
                                  host_load_1m=os.getloadavg()[0])
                    job['history'].append(record)
                    if result['bpb'] < job['best_bpb']:
                        job['best_bpb'] = result['bpb']
                        job['best'] = record
                        atomic_checkpoint({'config': asdict(model.config), 'weights': model.state_dict(),
                                           'seed': job['seed'], 'steps': job['steps'], 'cpu_s': job['cpu_s']},
                                          out / f'{name}.pt')
                    atomic_json(job['history'], out / f'{name}_history.json')
                    job['next_fraction'] += .25
                    print(json.dumps({'fit': name, **record}), flush=True)
                if not quantum:
                    break
        scheduler.append({'round': round_index, 'cpu_s': {name: job['cpu_s'] for name, job in jobs.items()}})
        round_index += 1
    atomic_json(scheduler, out / f'{stage}_scheduler.json')
    rows = []
    for name, job in jobs.items():
        path = out / f'{name}.pt'
        rows.append({**{k: job[k] for k in ('arm', 'seed', 'lr', 'steps', 'cpu_s', 'wall_s', 'best')},
                     'training_bytes': job['steps'] * 1024,
                     'compile_setup_wall_s': job['compile_setup_wall_s'],
                     'max_update_cpu_s': job['max_update_cpu_s'],
                     'parameters': job['model'].parameters_count(),
                     'checkpoint': str(path), 'checkpoint_sha256': sha256(path)})
    atomic_json(rows, out / f'{stage}_fits.json')
    return rows


def run(args):
    thread_environment = {key: os.environ.get(key) for key in
                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    if any(value != '1' for value in thread_environment.values()):
        raise ValueError('Launch with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1')
    if args.out.exists():
        raise FileExistsError(args.out)
    if not all(math.isfinite(s) and 1 <= s <= 600 for s in (args.pilot_seconds, args.fit_seconds)):
        raise ValueError('Fit durations must be within 1..600 CPU seconds')
    if not math.isfinite(args.quantum) or not 0 <= args.quantum <= 5:
        raise ValueError('Quantum must be within 0..5 CPU seconds')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError('Distinct nonnegative training seeds required')
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError('Requested CPU is unavailable')
        os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    identity = source_identity()
    budget = Budget(args.ledger, args.max_seconds, Path('reports/publication/device_stages.json'))
    status = 'failed'
    try:
        args.out.mkdir(parents=True)
        plan = {'source_sha256': identity, 'environment': environment('cpu'),
                'corpus_sha256': LOCAL_CORPUS_SHA256, 'threads': 1, 'thread_environment': thread_environment,
                'batch': 8, 'seq': 128,
                'arms': {a: asdict(replace(configuration(a), chunk=8)) for a in ARMS},
                'cpu_affinity': sorted(os.sched_getaffinity(0)), 'quantum_cpu_s': args.quantum,
                'compiled_training': args.compile, 'optimizer_foreach': args.foreach,
                'compile_cost': 'Recorded one-off warmup wall time; excluded from steady-state update budget. No parameter updates during warmup.',
                'scheduler': 'Fixed CPU-time blocks' if args.quantum else 'One update per arm per round',
                'pilot_seed': args.pilot_seed, 'learning_rates': [.001, .002, .004],
                'pilot_cpu_seconds_per_fit': args.pilot_seconds,
                'main_seeds': args.seeds, 'main_cpu_seconds_per_fit': args.fit_seconds,
                'validation_seed': args.validation_seed, 'test_seed': args.test_seed,
                'selection': 'Equal three-rate CPU-budget search per arm; pilot validation selects LR. Minimum quarter-budget validation loss selects each main checkpoint. Freeze all selections before test.',
                'schedule': '4% CPU-budget warmup, cosine to 10% of peak; AdamW betas .9,.95; matrix decay .1; gradient clip 1.',
                'timing': 'process CPU seconds for updates; data, validation and I/O excluded equally. Not FLOP matched. Wall time and tokens also reported.',
                'gate': 'Hybrid mean heldout bpb beats both controls at 128 and 512; conditional paired-window 95% upper bounds below zero.',
                'scope': 'Small CPU byte models, one corpus, random initialization. Does not establish broad Transformer superiority.'}
        atomic_json(plan, args.out / 'plan.json')
        training = Enwik8(args.data, 'train', expected_sha256=LOCAL_CORPUS_SHA256)
        validation = fixed_data(Enwik8(args.data, 'valid'), 128, 128, args.validation_seed)
        atomic_checkpoint({'x': validation[0], 'y': validation[1]}, args.out / 'validation.pt')
        options = dict(quantum=args.quantum, compile_model=args.compile, foreach=args.foreach)
        pilots = fit_group([(a, args.pilot_seed, lr) for a in ARMS for lr in plan['learning_rates']],
                           args.pilot_seconds, training, (validation[0][:48], validation[1][:48]),
                           budget, args.out, 'pilot', **options)
        rates = {a: min((p for p in pilots if p['arm'] == a), key=lambda p: p['best']['valid_bpb'])['lr'] for a in ARMS}
        atomic_json({'rates': rates, 'pilot_fits_sha256': sha256(args.out / 'pilot_fits.json')}, args.out / 'rates.json')
        print(json.dumps({'selected_learning_rates': rates}), flush=True)
        fits = []
        for seed in plan['main_seeds']:
            fits.extend(fit_group([(a, seed, rates[a]) for a in ARMS], args.fit_seconds,
                                  training, validation, budget, args.out, f'main_{seed}', **options))
        selection = {'source_sha256': identity, 'plan_sha256': sha256(args.out / 'plan.json'), 'rates': rates, 'fits': fits}
        atomic_json(selection, args.out / 'selection.json')
        if source_identity() != identity:
            raise RuntimeError('Source changed before test')
        tests = []
        corpus = Enwik8(args.data, 'test')
        for length in (128, 512):
            data = fixed_data(corpus, 256, length, args.test_seed)
            atomic_checkpoint({'x': data[0], 'y': data[1]}, args.out / f'test_{length}.pt')
            for fit in fits:
                if sha256(fit['checkpoint']) != fit['checkpoint_sha256']:
                    raise RuntimeError('Checkpoint changed')
                saved = torch.load(fit['checkpoint'], weights_only=True)
                model = LM(Config(**saved['config']))
                model.load_state_dict(saved['weights'])
                tests.append({'arm': fit['arm'], 'seed': fit['seed'], **score(model, data, budget)})
                atomic_json(tests, args.out / 'test_rows.json')
        quality = {}
        gate = True
        for length in (128, 512):
            group = [r for r in tests if r['context'] == length]
            quality[length] = {a: {'mean_bpb': statistics.mean(r['bpb'] for r in group if r['arm'] == a),
                                   'seed_bpb': [r['bpb'] for r in group if r['arm'] == a]} for a in ARMS}
            for control in ARMS[:2]:
                differences = []
                for candidate in (r for r in group if r['arm'] == 'hybrid'):
                    base = next(r for r in group if r['arm'] == control and r['seed'] == candidate['seed'])
                    differences.append((np.asarray(candidate['window_nll']) - base['window_nll']) / math.log(2))
                interval = paired_interval(np.mean(differences, axis=0))
                quality[length]['hybrid_minus_' + control] = interval
                gate = gate and interval['interval95'][1] < 0
        results = {'status': 'completed', 'source_sha256': identity, 'quality': quality,
                   'quality_gate_passed': gate, 'selection_sha256': sha256(args.out / 'selection.json')}
        atomic_json(results, args.out / 'results.json')
        print(json.dumps(results), flush=True)
        status = 'completed'
    finally:
        budget.finish(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/enwik8'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--pilot-seconds', type=float, default=20)
    parser.add_argument('--fit-seconds', type=float, default=90)
    parser.add_argument('--quantum', type=float, default=0, help='CPU seconds per interleaved block; zero rotates single updates')
    parser.add_argument('--cpu', type=int, help='Pin this process to one allowed CPU')
    parser.add_argument('--compile', action='store_true', help='Compile and warm each training objective before timing')
    parser.add_argument('--foreach', action='store_true', help='Use foreach AdamW for every arm')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--pilot-seed', type=int, default=17)
    parser.add_argument('--validation-seed', type=int, default=91101)
    parser.add_argument('--test-seed', type=int, default=91102)
    run(parser.parse_args())
