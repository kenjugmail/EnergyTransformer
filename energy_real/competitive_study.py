"""CPU-bounded, token/parameter-matched language and inference comparison.

Four controls share corpus splits, minibatches, schedule, evaluation windows,
and seed IDs. This is a small-model experiment, not a scale-up claim. All
validation selections are committed before any test predictions are made.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from .bench import balanced_order, tensor_bytes
from .data import Enwik8, LOCAL_CORPUS_SHA256
from .local_energy_study import Budget
from .model import Config, LM
from .runtime import atomic_checkpoint, atomic_json, environment, sha256, source_identity
from .train import learning_rate, matched_ff

ARMS = ('energy', 'softmax', 'local_softmax', 'hybrid')


def configuration(arm, width=64, layers=3, window=64):
    base_ff = width * 3
    soft_ff = matched_ff(width, 2, 8, base_ff)
    fraction = 1 if arm in ('softmax', 'local_softmax') else (layers // 3) / layers if arm == 'hybrid' else 0
    return Config(kind='softmax' if arm == 'local_softmax' else arm, width=width, layers=layers,
                  heads=2, slots=8, ff_width=base_ff + round((soft_ff - base_ff) * fraction),
                  chunk=32, attention_window=window if arm in ('local_softmax', 'hybrid') else 0)


def fixed_data(corpus, count, length, seed):
    return corpus.batch(np.random.default_rng(seed), count, length)


@torch.no_grad()
def score(model, data, budget):
    model.eval()
    x, y = data
    nll, hits = [], []
    for lo in range(0, len(x), 8):
        budget.check()
        output, _ = model(x[lo:lo+8])
        loss = F.cross_entropy(output.flatten(0, 1), y[lo:lo+8].flatten(), reduction='none')
        nll.extend(loss.reshape(len(output), -1).mean(1).tolist())
        hits.extend(output.argmax(-1).eq(y[lo:lo+8]).float().mean(1).tolist())
    return {'bpb': statistics.mean(nll) / math.log(2), 'accuracy': statistics.mean(hits),
            'window_nll': nll, 'window_accuracy': hits, 'windows': len(x), 'context': x.shape[1]}


def paired_interval(differences, seed=1873):
    values = np.asarray(differences)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'interval95': np.quantile(draws, [.025, .975]).tolist()}


@torch.no_grad()
def profile(models, budget, out, rounds=6):
    rows = []
    generator = torch.Generator().manual_seed(72630)
    for length in (128, 1024, 4096):
        tokens = torch.randint(256, (1, length + 16), generator=generator)
        for turn in range(rounds):
            for arm in balanced_order(list(ARMS), turn):
                budget.check()
                model = models[arm].eval()
                model(tokens[:, :64])  # identical warmup work in every arm
                start = time.perf_counter()
                _, state = model(tokens[:, :length])
                prefill_s = time.perf_counter() - start
                before = tensor_bytes(state)
                start = time.perf_counter()
                for position in range(length, length + 16):
                    _, state = model(tokens[:, position:position+1], state)
                decode_s = time.perf_counter() - start
                rows.append({'arm': arm, 'context': length, 'round': turn,
                             'prefill_s': prefill_s, 'decode_ms': decode_s * 1000 / 16,
                             'state_bytes': tensor_bytes(state), 'prefill_state_bytes': before,
                             'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters())})
        atomic_json(rows, out / 'profile_rows.json')
        print(json.dumps({'profile_context_completed': length}), flush=True)
    return rows


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 1 <= args.steps <= 3000 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Bounded steps and unique seeds required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    budget = Budget(args.budget_ledger, args.max_seconds, args.prior_ledger)
    status = 'failed'
    identity = source_identity()
    try:
        args.out.mkdir(parents=True)
        training = Enwik8(args.data, 'train', expected_sha256=LOCAL_CORPUS_SHA256)
        valid = fixed_data(Enwik8(args.data, 'valid'), 128, 128, 72601)
        atomic_checkpoint({'x': valid[0], 'y': valid[1]}, args.out / 'validation.pt')
        plan = {'source_sha256': identity, 'corpus_sha256': training.file_identity(),
                'environment': environment('cpu'), 'arms': {arm: asdict(configuration(arm)) for arm in ARMS},
                'seeds': args.seeds, 'steps': args.steps, 'batch': 8, 'seq': 128,
                'tokens_per_fit': args.steps*8*128, 'lr': .002, 'warmup': 40,
                'optimizer': 'AdamW, betas=(.9,.95), weight_decay=.1 for matrices; 0 otherwise',
                'validation': {'seed': 72601, 'windows': 128, 'context': 128, 'every': 100},
                'test': {'seed': 72602, 'windows': 256, 'contexts': [128, 512]},
                'selection': 'Minimum validation bpb including initialization, separately per arm/seed; freeze all before test.',
                'quality_gate': 'Hybrid mean test bpb <= full softmax with paired window bootstrap upper bound <= 0.',
                'scope': 'Fresh random initialization; CPU float32; matched tokens and parameters, measured unequal compute. No fact memory in any arm.'}
        atomic_json(plan, args.out / 'plan.json')
        fits = []
        for seed in args.seeds:
            models, optimizers, best, histories, timers = {}, {}, {}, {}, {}
            rng = np.random.default_rng(seed + 72000)
            for arm in ARMS:
                torch.manual_seed(seed)
                model = LM(configuration(arm))
                models[arm] = model
                optimizers[arm] = torch.optim.AdamW([
                    {'params': [p for p in model.parameters() if p.ndim >= 2], 'weight_decay': .1},
                    {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0.}],
                    lr=.002, betas=(.9,.95), foreach=False)
                result = score(model, valid, budget)
                best[arm] = {'step': 0, 'bpb': result['bpb']}
                histories[arm] = [{'step': 0, 'valid_bpb': result['bpb'], 'train_s': 0.}]
                timers[arm] = 0.
                atomic_checkpoint({'config': asdict(model.config), 'weights': model.state_dict(),
                                   'step': 0, 'seed': seed}, args.out / f'{arm}_{seed}.pt')
            for step in range(1, args.steps + 1):
                x, y = training.batch(rng, 8, 128)
                losses = {}
                for arm in balanced_order(list(ARMS), step):
                    budget.check()
                    model, optimizer = models[arm], optimizers[arm]
                    model.train()
                    start = time.perf_counter()
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate(step, args.steps, 40, .002)
                    optimizer.zero_grad(set_to_none=True)
                    output, _ = model(x)
                    loss = F.cross_entropy(output.flatten(0, 1), y.flatten())
                    if not torch.isfinite(loss): raise FloatingPointError(arm)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    timers[arm] += time.perf_counter() - start
                    losses[arm] = float(loss.detach())
                if step % 100 == 0 or step == args.steps:
                    for arm in ARMS:
                        result = score(models[arm], valid, budget)
                        histories[arm].append({'step': step, 'valid_bpb': result['bpb'],
                                               'train_s': timers[arm], 'last_train_nll': losses[arm]})
                        if result['bpb'] < best[arm]['bpb']:
                            best[arm] = {'step': step, 'bpb': result['bpb']}
                            atomic_checkpoint({'config': asdict(models[arm].config),
                                'weights': models[arm].state_dict(), 'step': step, 'seed': seed},
                                args.out / f'{arm}_{seed}.pt')
                    atomic_json(histories, args.out / f'history_{seed}.json')
                    print(json.dumps({'seed': seed, 'step': step, 'valid_bpb': {a: histories[a][-1]['valid_bpb'] for a in ARMS}}), flush=True)
            for arm in ARMS:
                path = args.out / f'{arm}_{seed}.pt'
                fits.append({'arm': arm, 'seed': seed, 'parameters': models[arm].parameters_count(),
                             'best': best[arm], 'train_s': timers[arm], 'checkpoint': str(path),
                             'checkpoint_sha256': sha256(path)})
            atomic_json(fits, args.out / 'fits.json')
        selection = {'source_sha256': identity, 'plan_sha256': sha256(args.out/'plan.json'), 'fits': fits,
                     'selected_seeds': {arm: min([f for f in fits if f['arm']==arm], key=lambda f:f['best']['bpb'])['seed'] for arm in ARMS}}
        atomic_json(selection, args.out / 'selection.json')
        if source_identity() != identity: raise ValueError('Study source changed before test')
        tests = []
        corpus = Enwik8(args.data, 'test')
        for length in (128, 512):
            data = fixed_data(corpus, 256, length, 72602)
            atomic_checkpoint({'x': data[0], 'y': data[1]}, args.out / f'test_{length}.pt')
            for fit in fits:
                if sha256(fit['checkpoint']) != fit['checkpoint_sha256']: raise ValueError('Checkpoint changed')
                saved = torch.load(fit['checkpoint'], weights_only=True)
                model = LM(Config(**saved['config']))
                model.load_state_dict(saved['weights'])
                tests.append({'arm': fit['arm'], 'seed': fit['seed'], **score(model, data, budget)})
                atomic_json(tests, args.out / 'test_rows.json')
            print(json.dumps({'test_context_completed': length}), flush=True)
        profiles = {}
        for arm in ARMS:
            saved = torch.load(args.out / f'{arm}_{selection["selected_seeds"][arm]}.pt', weights_only=True)
            profiles[arm] = LM(Config(**saved['config']))
            profiles[arm].load_state_dict(saved['weights'])
        rows = profile(profiles, budget, args.out)
        results = summarize(fits, tests, rows)
        results.update(status='completed', source_sha256=identity,
                       selection_sha256=sha256(args.out/'selection.json'))
        atomic_json(results, args.out / 'results.json')
        status = 'completed'
        print(json.dumps(results), flush=True)
        return results
    finally:
        budget.finish(status)


def summarize(fits, tests, profile_rows):
    quality = {}
    for length in sorted({r['context'] for r in tests}):
        group = [r for r in tests if r['context']==length]
        quality[length] = {arm: {'mean_bpb': statistics.mean(r['bpb'] for r in group if r['arm']==arm),
                                'seed_bpb': [r['bpb'] for r in group if r['arm']==arm]} for arm in ARMS}
        for control in ('energy', 'softmax', 'local_softmax'):
            differences = []
            for candidate in [r for r in group if r['arm']=='hybrid']:
                base = next(r for r in group if r['arm']==control and r['seed']==candidate['seed'])
                differences.append((np.array(candidate['window_nll'])-base['window_nll'])/math.log(2))
            # Resample windows, averaging across the same three seeds first.
            quality[length]['hybrid_minus_'+control] = paired_interval(np.mean(differences, axis=0))
    efficiency = {}
    for length in sorted({r['context'] for r in profile_rows}):
        efficiency[length] = {}
        for arm in ARMS:
            group = [r for r in profile_rows if r['context']==length and r['arm']==arm]
            efficiency[length][arm] = {key: statistics.median(r[key] for r in group)
                for key in ('prefill_s','decode_ms','state_bytes','prefill_state_bytes','parameter_bytes')}
    return {'quality': quality, 'efficiency': efficiency,
            'train_s': {arm: statistics.mean(f['train_s'] for f in fits if f['arm']==arm) for arm in ARMS},
            'parameters': {arm: next(f['parameters'] for f in fits if f['arm']==arm) for arm in ARMS},
            'interval_scope': 'Paired test-window bootstrap conditional on the tested training seeds; overlapping corpus crops can be dependent.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/enwik8'))
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seeds', type=int, nargs='+', default=[11,22,33])
    p.add_argument('--max-seconds', type=float, default=5400)
    p.add_argument('--budget-ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    p.add_argument('--prior-ledger', type=Path, default=Path('reports/publication/device_stages.json'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
