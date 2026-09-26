"""Bounded training studies for local credit, energy modules, and full EBMs.

Run ``python -m energy_real.local_energy_study --help``. All test evaluations
follow a frozen selection file. Checkpoints and full datasets stay under runs/.
Completed fits can be resumed under identical source/config/data identities;
an interrupted fit restarts from its committed optimizer/RNG checkpoint.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from .local_energy import LocalConfig, LocalEnergyNetwork, gradient_alignment, model_payload
from .local_energy_data import Dataset, byte_datasets, discrete_graph_energy, fashion_datasets, graph_dataset
from .memory_eval import paired_interval
from .runtime import atomic_checkpoint, atomic_json, environment, sha256, source_identity


VERSION = 'local_energy_study_v1'
SEEDS = (11, 22, 33)
TEMPERATURES = (.03, .1, .3, 1.)


class BudgetExpired(RuntimeError):
    pass


class Budget:
    """Single-process deadline plus durable cumulative charges across invocations."""
    def __init__(self, ledger: Path, seconds: float, prior_ledger: Path | None):
        self.path = ledger.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = None
        inherited = 0.
        if prior_ledger is not None:
            payload = json.loads(prior_ledger.read_text())
            inherited = sum(stage['used_s'] for stage in payload['stages'].values())
            prior = {'path': str(prior_ledger.resolve()), 'sha256': sha256(prior_ledger), 'used_s': inherited}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data['prior'] != prior:
                raise ValueError('Budget prior ledger changed')
            # A crashed process is charged its full reserved allocation. An
            # ordinary interruption is caught by run() and charges actual time.
            for entry in self.data['runs']:
                if entry['status'] == 'running':
                    entry['status'] = 'unclosed_charged_reservation'
                    entry['elapsed_s'] = entry['reserved_s']
        else:
            self.data = {'version': VERSION, 'ceiling_s': 8 * 3600, 'prior': prior, 'runs': []}
        used = inherited + sum(entry['elapsed_s'] for entry in self.data['runs'])
        if not math.isfinite(seconds) or seconds <= 0 or seconds > self.data['ceiling_s'] - used:
            raise ValueError(f'Requested run exceeds remaining eight-hour budget ({self.data["ceiling_s"] - used:.1f}s)')
        self.entry = {'status': 'running', 'reserved_s': seconds, 'elapsed_s': 0.,
                      'started_unix': time.time(), 'source_sha256': source_identity()}
        self.data['runs'].append(self.entry)
        self.start = time.perf_counter()
        self.deadline = self.start + seconds
        atomic_json(self.data, self.path)

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise BudgetExpired('Run deadline reached')

    def finish(self, status):
        self.entry.update(status=status, elapsed_s=time.perf_counter() - self.start)
        self.data['total_used_s'] = (self.data['prior']['used_s'] if self.data['prior'] else 0.) + sum(
            entry['elapsed_s'] for entry in self.data['runs'])
        atomic_json(self.data, self.path)
        self.lock.close()


@dataclass(frozen=True)
class FitConfig:
    task: str
    method: str
    seed: int
    width: int
    depth: int
    memory: bool
    steps: int
    batch: int
    settle_steps: int
    state_lr: float = .2
    dual_lr: float = 1.
    rho: float = 1.
    weight_lr: float = .001
    temperature: float = .1
    positive_weight: float = .1
    activation: str = 'tanh'
    candidate_chunk: int = 16

    @property
    def name(self):
        return f'{self.task}_{self.method}_d{self.depth}_w{self.width}_mem{int(self.memory)}_seed{self.seed}'


def spec(args):
    seeds = tuple(args.seeds)
    configs = []
    for task in args.tasks:
        if task == 'fashion':
            for depth in args.fashion_depths:
                methods = ('bp', 'pc', 'pcalm', 'energy') if depth == min(args.fashion_depths) else ('bp', 'pc', 'pcalm')
                for seed in seeds:
                    for method in methods:
                        configs.append(FitConfig(task, method, seed, args.fashion_width, depth, False,
                                                 args.fashion_steps, args.batch, 2 * depth))
        elif task == 'graph':
            for seed in seeds:
                for memory in (False, True):
                    for method in ('bp', 'pc', 'pcalm', 'energy'):
                        configs.append(FitConfig(task, method, seed, args.graph_width, args.graph_depth, memory,
                                                 args.graph_steps, args.batch, 4 * args.graph_depth))
        elif task == 'bytes':
            for seed in seeds:
                for method in ('bp', 'bp_ce', 'pcalm', 'energy'):
                    configs.append(FitConfig(task, method, seed, args.byte_width, args.byte_depth, False,
                                             args.byte_steps, args.byte_batch, 4 * args.byte_depth,
                                             candidate_chunk=32))
        else:
            raise ValueError(f'Unknown task: {task}')
    return configs


def predict_scores(model, dataset, config, budget, *, batch=128):
    scores = []
    started = time.perf_counter()
    # Small candidate chunks bound memory even for the 256-way byte EBM.
    if config.method == 'energy':
        batch = min(batch, 32 if config.task != 'bytes' else 8)
    for begin in range(0, len(dataset.y), batch):
        budget.check()
        x, _, memory = dataset.take(slice(begin, begin + batch))
        if not config.memory:
            memory = None
        with torch.no_grad():
            if config.method == 'energy':
                score = -model.candidate_energies(x, memory, steps=config.settle_steps,
                    state_lr=config.state_lr, rho=config.rho, candidate_chunk=config.candidate_chunk,
                    check_budget=budget.check)
            else:
                score = model.feedforward(x, memory)
        if not torch.isfinite(score).all():
            raise FloatingPointError('Nonfinite prediction scores')
        scores.append(score)
    return torch.cat(scores), time.perf_counter() - started


def summarize_scores(scores, dataset, temperature):
    logp = (scores / temperature).log_softmax(-1)
    predictions = scores.argmax(-1)
    nll = -logp[torch.arange(len(dataset.y)), dataset.y]
    correct = predictions.eq(dataset.y)
    result = {'accuracy': float(correct.float().mean()), 'nll': float(nll.mean()),
            'bits_per_target': float(nll.mean() / math.log(2)), 'temperature': temperature,
            'count': len(dataset.y),
            'groups': {group: {'count': sum(g == group for g in dataset.groups),
                              'accuracy': float(correct[[g == group for g in dataset.groups]].float().mean())}
                       for group in sorted(set(dataset.groups))},
            'rows': [{'id': identifier, 'group': group, 'expected': int(target), 'prediction': int(prediction),
                      'correct': bool(hit), 'nll': float(loss)}
                     for identifier, group, target, prediction, hit, loss in zip(
                         dataset.ids, dataset.groups, dataset.y, predictions, correct, nll)]}
    if dataset.memory is not None:
        unknown = dataset.memory.shape[-1] - 1
        result['expected_unknown'] = int(dataset.y.eq(unknown).sum())
        result['unsupported_answers'] = int((dataset.y.eq(unknown) & predictions.ne(unknown)).sum())
        result['unnecessary_abstentions'] = int((dataset.y.ne(unknown) & predictions.eq(unknown)).sum())
    return result


def validate(model, data, config, budget):
    scores, elapsed = predict_scores(model, data, config, budget)
    candidates = [summarize_scores(scores, data, temp) for temp in TEMPERATURES]
    best = min(candidates, key=lambda r: (r['nll'], -r['temperature']))
    best['elapsed_s'] = elapsed
    return best


def _rank(result, task):
    # No test metric enters either checkpoint or temperature selection.
    return (-result['nll'], result['accuracy']) if task == 'bytes' else (result['accuracy'], -result['nll'])


def train_fit(config, train, valid, out, budget, identity, *, resume=False):
    out.mkdir(parents=True, exist_ok=True)
    network_config = LocalConfig(train.x.shape[1], 256 if config.task == 'bytes' else
                                (10 if config.task == 'fashion' else train.memory.shape[-1]),
                                config.width, config.depth, config.activation,
                                train.memory.shape[1] if config.memory else 0,
                                train.memory.shape[2] if config.memory else 0)
    fit_identity = {'version': VERSION, 'config': asdict(config), 'model_config': asdict(network_config),
                    'source_sha256': identity, 'train': train.manifest(), 'valid': valid.manifest()}
    identity_path = out / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != fit_identity:
        raise ValueError(f'Fit identity changed: {out}')
    atomic_json(fit_identity, identity_path)
    if (out / 'result.json').exists():
        if not resume:
            raise FileExistsError(out)
        result = json.loads((out / 'result.json').read_text())
        for name in ('best', 'initial'):
            if sha256(out / f'{name}.pt') != result[f'{name}_sha256']:
                raise ValueError(f'Changed checkpoint: {out}/{name}.pt')
        return result
    torch.manual_seed(config.seed)
    model = LocalEnergyNetwork(network_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.weight_lr)
    generator = torch.Generator().manual_seed(config.seed + 7000)
    order = torch.randperm(len(train.y), generator=generator)
    offset = 0
    history = []
    step = 0
    inherited_s = 0.
    train_start = time.perf_counter()
    if (out / 'last.pt').exists():
        if not resume:
            raise FileExistsError('Interrupted fit exists; use --resume')
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if last['identity'] != fit_identity:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(last['model']['state_dict'])
        optimizer.load_state_dict(last['optimizer'])
        generator.set_state(last['generator'])
        order, offset, step = last['order'], last['offset'], last['step']
        history, best, best_step, initial = last['history'], last['best'], last['best_step'], last['initial']
        best_model = last['best_model']
        # A best.pt written just before a failed checkpoint commit is an
        # uncommitted tail. Restore the exact committed selection on resume.
        atomic_checkpoint(best_model, out / 'best.pt')
        inherited_s = last['elapsed_s']
    else:
        initial = validate(model, valid, config, budget)
        best, best_step = initial, 0
        best_model = copy.deepcopy(model_payload(model))
        atomic_checkpoint(model_payload(model), out / 'initial.pt')
        atomic_checkpoint(model_payload(model), out / 'best.pt')
    def save():
        atomic_checkpoint({'identity': fit_identity, 'model': model_payload(model), 'optimizer': optimizer.state_dict(),
                           'generator': generator.get_state(), 'order': order, 'offset': offset, 'step': step,
                           'history': history, 'best': best, 'best_step': best_step, 'initial': initial,
                           'best_model': best_model,
                           'elapsed_s': inherited_s + time.perf_counter() - train_start}, out / 'last.pt')
        atomic_json({'config': asdict(config), 'step': step, 'best_step': best_step,
                     'best_accuracy': best['accuracy'], 'best_nll': best['nll'],
                     'elapsed_s': inherited_s + time.perf_counter() - train_start,
                     'history': history}, out / 'progress.json')
    save()
    period = max(1, config.steps // 4)
    try:
        while step < config.steps:
            budget.check()
            if offset + config.batch > len(order):
                order = torch.randperm(len(train.y), generator=generator)
                offset = 0
            batch_indices = order[offset:offset + config.batch]
            x, labels, memory = train.take(batch_indices)
            if not config.memory:
                memory = None
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            if config.method == 'energy':
                loss = model.energy_loss(x, labels, memory, steps=config.settle_steps, state_lr=config.state_lr,
                                         rho=config.rho, temperature=config.temperature,
                                         positive_weight=config.positive_weight, candidate_chunk=config.candidate_chunk,
                                         check_budget=budget.check)
                settled = None
            elif config.method == 'bp_ce':
                loss = F.cross_entropy(model.feedforward(x, memory), labels)
                settled = None
            else:
                loss, settled = model.local_loss(x, labels, memory, method=config.method,
                    steps=config.settle_steps, state_lr=config.state_lr, dual_lr=config.dual_lr,
                    rho=config.rho, check_budget=budget.check)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameter gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
            optimizer.step()
            step += 1
            offset += len(batch_indices)
            if step % period == 0 or step == config.steps:
                update_s = time.perf_counter() - tick
                validation = validate(model, valid, config, budget)
                if _rank(validation, config.task) > _rank(best, config.task):
                    best, best_step = validation, step
                    best_model = copy.deepcopy(model_payload(model))
                    atomic_checkpoint(model_payload(model), out / 'best.pt')
                history.append({'step': step, 'loss': float(loss.detach()), 'gradient_norm': norm,
                                'last_update_s': update_s, 'validation_accuracy': validation['accuracy'],
                                'validation_nll': validation['nll'],
                                'residual_rms': None if settled is None else settled.residual_rms,
                                'stationarity_rms': None if settled is None else settled.stationarity_rms})
                save()
                print(json.dumps({'fit': config.name, **history[-1]}), flush=True)
    except BaseException:
        # If an update was interrupted, restore the preceding committed state
        # on resume; do not commit partially accumulated gradients as a step.
        raise
    result = {'config': asdict(config), 'model_config': asdict(network_config), 'status': 'completed',
              'source_sha256': identity, 'best_step': best_step, 'updates': step,
              'examples_seen': step * config.batch,
              'initial_validation': initial, 'selected_validation': best,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
              'elapsed_s': inherited_s + time.perf_counter() - train_start,
              'initial_sha256': sha256(out / 'initial.pt'), 'best_sha256': sha256(out / 'best.pt'),
              'checkpoint': str((out / 'best.pt').resolve()), 'history': history}
    atomic_json(result, out / 'result.json')
    return result


def load_model(path):
    payload = torch.load(path, weights_only=True, map_location='cpu')
    model = LocalEnergyNetwork(LocalConfig(**payload['config']))
    model.load_state_dict(payload['state_dict'])
    return model


def evaluate_fit(record, data, budget):
    config = FitConfig(**record['config'])
    result = {'config': record['config'], 'initial': None, 'trained': None}
    for name, field, valfield in (('initial', 'initial', 'initial_validation'), ('best', 'trained', 'selected_validation')):
        path = Path(record['checkpoint']).with_name(f'{name}.pt')
        if sha256(path) != record[f'{name}_sha256']:
            raise ValueError('Checkpoint modified after selection')
        model = load_model(path)
        score, elapsed = predict_scores(model, data, config, budget)
        result[field] = summarize_scores(score, data, record[valfield]['temperature'])
        result[field]['evaluation_s'] = elapsed
    a, b = result['trained']['rows'], result['initial']['rows']
    result['learning_interval'] = paired_interval([r['correct'] for r in a], [r['correct'] for r in b])
    return result


def comparison(tests, task, method_a, method_b, *, memory_a=False, memory_b=False, depth=None):
    def rows(method, memory):
        return sorted([t for t in tests if t['config']['task'] == task and t['config']['method'] == method
                       and t['config']['memory'] == memory and (depth is None or t['config']['depth'] == depth)],
                      key=lambda t: t['config']['seed'])
    a, b = rows(method_a, memory_a), rows(method_b, memory_b)
    if not a or len(a) != len(b):
        return None
    if [r['config']['seed'] for r in a] != [r['config']['seed'] for r in b]:
        raise ValueError('Unpaired training seeds')
    for ra, rb in zip(a, b):
        if [r['id'] for r in ra['trained']['rows']] != [r['id'] for r in rb['trained']['rows']]:
            raise ValueError('Unpaired evaluation examples')
    correct_a = np.array([[r['correct'] for r in t['trained']['rows']] for t in a], dtype=float)
    correct_b = np.array([[r['correct'] for r in t['trained']['rows']] for t in b], dtype=float)
    seed_differences = (correct_a.mean(1) - correct_b.mean(1)).tolist()
    interval = paired_interval(correct_a.mean(0).tolist(), correct_b.mean(0).tolist())
    return {'task': task, 'a': method_a, 'b': method_b, 'memory_a': memory_a, 'memory_b': memory_b,
            'depth': depth, 'paired_interval': interval, 'per_seed_differences': seed_differences,
            'replicated_five_point_gain': interval['difference'] >= .05 and interval['low'] > 0
                                        and min(seed_differences) > 0,
            'uncertainty_scope': 'Paired examples after averaging matched training seeds; three-seed differences reported separately. Byte windows may remain correlated.'}


def diagnostics(record, dataset, budget):
    config = FitConfig(**record['config'])
    model = load_model(record['checkpoint'])
    x, y, memory = dataset.take(slice(0, 8))
    if not config.memory:
        memory = None
    result = {'config': record['config'], 'parameter_bytes': record['parameter_bytes'],
              'persistent_memory_bytes_per_episode': 0 if not config.memory else
                  4 * (model.config.memory_keys ** 2 + model.config.memory_keys * model.config.memory_values),
              'scope': 'CPU implementation; no GPU efficiency claim. Persistent memory excludes input/event storage, which is shared.'}
    if config.method in ('pc', 'pcalm'):
        result['gradient_alignment'] = gradient_alignment(model, x, y, memory, method=config.method,
            steps=config.settle_steps, state_lr=config.state_lr, dual_lr=config.dual_lr, rho=config.rho,
            check_budget=budget.check)
        target = F.one_hot(y, model.config.outputs).float()
        result['credit_trace'] = model.relax(x, target, memory, steps=config.settle_steps,
            state_lr=config.state_lr, dual_lr=config.dual_lr if config.method == 'pcalm' else 0.,
            rho=config.rho, trace=True, check_budget=budget.check).trace
    if config.method == 'energy':
        with torch.no_grad():
            _, result['solver'] = model.candidate_energies(x[:2], None if memory is None else memory[:2],
                steps=config.settle_steps, state_lr=config.state_lr, rho=config.rho,
                candidate_chunk=config.candidate_chunk, return_diagnostics=True, check_budget=budget.check)
        # Test-time iteration sweep uses frozen weights/temperature and is
        # diagnostic only. It cannot change the selected headline model.
        result['iteration_sweep'] = []
        subset = Dataset(dataset.x[:32], dataset.y[:32], dataset.ids[:32], dataset.groups[:32],
                         None if dataset.memory is None else dataset.memory[:32])
        for steps in (1, 4, 8, 16, 32):
            alternative = FitConfig(**{**asdict(config), 'settle_steps': steps})
            score, elapsed = predict_scores(model, subset, alternative, budget)
            metrics = summarize_scores(score, subset, record['selected_validation']['temperature'])
            result['iteration_sweep'].append({'steps': steps, 'accuracy': metrics['accuracy'],
                                              'nll': metrics['nll'], 'elapsed_s': elapsed})
    timings = []
    method = 'bp' if config.method == 'bp_ce' else config.method
    for i in range(9):
        budget.check()
        tick = time.perf_counter()
        model.predict(x[:1], None if memory is None else memory[:1], method=method,
                      steps=config.settle_steps, state_lr=config.state_lr, rho=config.rho,
                      candidate_chunk=config.candidate_chunk, check_budget=budget.check)
        if i:
            timings.append(time.perf_counter() - tick)
    result['single_example_median_s'] = statistics.median(timings)
    result['single_example_timings_s'] = timings
    return result


def make_data(task, args):
    if task == 'fashion':
        return fashion_datasets(args.fashion_data, train_count=args.fashion_train, valid_count=args.fashion_valid,
                                test_count=args.fashion_test)
    if task == 'bytes':
        return byte_datasets(args.byte_data, context=args.byte_context, train_count=args.byte_train,
                             valid_count=args.byte_valid, test_count=args.byte_test)
    return {split: graph_dataset(split, count, seed) for split, count, seed in
            [('train', args.graph_train, 481001), ('valid', args.graph_valid, 481002), ('test', args.graph_test, 481003)]}


def run(args):
    if args.threads < 1 or min(args.fashion_steps, args.graph_steps, args.byte_steps) < 1:
        raise ValueError('Positive training steps and threads required')
    if any(step > 2000 for step in (args.fashion_steps, args.graph_steps, args.byte_steps)):
        raise ValueError('This bounded study allows at most 2000 updates per fit')
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError('Unique tasks and seeds required')
    if min(args.batch, args.byte_batch) < 1:
        raise ValueError('Positive batch sizes required')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.out = args.out.resolve()
    if args.out.exists() and not args.resume:
        raise FileExistsError('Use a fresh run directory or --resume')
    args.out.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    configs = spec(args)
    budget = Budget(args.budget_ledger, args.max_seconds, args.prior_ledger)
    budget.entry['out'] = str(args.out)
    records, tests, status = [], [], 'running'
    started = time.perf_counter()
    data = {}
    def progress():
        atomic_json({'status': status, 'version': VERSION, 'source_sha256': identity,
                     'environment': environment('cpu'), 'planned_fits': len(configs),
                     'completed_fits': len(records), 'evaluated_fits': len(tests),
                     'elapsed_s': time.perf_counter() - started, 'gpu_seconds': 0,
                     'records': records, 'test_results': tests}, args.out / 'summary.json')
    try:
        recipes = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                   if k not in ('resume', 'out', 'max_seconds', 'budget_ledger', 'prior_ledger')}
        plan = {'version': VERSION, 'source_sha256': identity, 'recipes': recipes,
                'fits': [asdict(c) for c in configs],
                'selection': 'Validation accuracy then NLL; bytes use NLL then accuracy. Includes initialization. Temperature uses validation NLL only.',
                'scope': 'CPU bounded prototypes; structured graph inputs; 8-byte context model is not a general-purpose LM.'}
        if (args.out / 'plan.json').exists() and json.loads((args.out / 'plan.json').read_text()) != plan:
            raise ValueError('Study source/configuration changed; cannot resume')
        atomic_json(plan, args.out / 'plan.json')
        for task in args.tasks:
            budget.check()
            data[task] = make_data(task, args)
            batch = args.byte_batch if task == 'bytes' else args.batch
            if len(data[task]['train'].y) < batch:
                raise ValueError('Training set must contain at least one full batch')
        manifests = {task: {split: d.manifest() for split, d in parts.items()} for task, parts in data.items()}
        if (args.out / 'datasets.json').exists() and json.loads((args.out / 'datasets.json').read_text()) != manifests:
            raise ValueError('Study data changed; cannot resume')
        atomic_json(manifests, args.out / 'datasets.json')
        progress()
        for config in configs:
            budget.check()
            result = train_fit(config, data[config.task]['train'], data[config.task]['valid'],
                               args.out / config.name, budget, identity, resume=args.resume)
            records.append(result)
            progress()
        if source_identity() != identity:
            raise ValueError('Source changed during training')
        selection = {'status': 'frozen_before_test', 'source_sha256': identity,
                     'datasets_sha256': sha256(args.out / 'datasets.json'), 'records': records}
        if (args.out / 'selection.json').exists() and json.loads((args.out / 'selection.json').read_text()) != selection:
            raise ValueError('Frozen selection changed')
        atomic_json(selection, args.out / 'selection.json')
        if args.training_only:
            status = 'training_completed'
            progress()
            return json.loads((args.out / 'summary.json').read_text())
        for record in records:
            config = FitConfig(**record['config'])
            output = args.out / config.name / 'test.json'
            if output.exists() and args.resume:
                result = json.loads(output.read_text())
                if result['selection_sha256'] != sha256(args.out / 'selection.json'):
                    raise ValueError('Test identity mismatch')
            else:
                result = evaluate_fit(record, data[config.task]['test'], budget)
                result['selection_sha256'] = sha256(args.out / 'selection.json')
                atomic_json(result, output)
            tests.append(result); progress()
            print(json.dumps({'tested': config.name, 'accuracy': result['trained']['accuracy'],
                              'initial_accuracy': result['initial']['accuracy'],
                              'bits_per_target': result['trained']['bits_per_target']}), flush=True)
        diagnostics_rows = []
        for record in records:
            if record['config']['seed'] == min(args.seeds):
                diagnostics_rows.append(diagnostics(record, data[record['config']['task']]['test'], budget))
        atomic_json(diagnostics_rows, args.out / 'diagnostics.json')
        if 'graph' in data:
            discrete = []
            for episode, label in zip(data['graph']['test'].episodes, data['graph']['test'].y):
                budget.check()
                tick = time.perf_counter(); answer = discrete_graph_energy(episode); elapsed = time.perf_counter() - tick
                discrete.append({'id': episode['id'], 'group': episode['group'], 'expected': int(label),
                                 'correct': answer['answer'] == int(label), 'elapsed_s': elapsed, **answer})
            atomic_json({'scope': 'Fixed supplied-rule discrete energy control; no learned extraction or rule discovery',
                         'accuracy': statistics.mean(r['correct'] for r in discrete), 'rows': discrete}, args.out / 'discrete_graph.json')
        comparisons = []
        for task in args.tasks:
            depths = args.fashion_depths if task == 'fashion' else [None]
            for depth in depths:
                for method in ('pc', 'pcalm', 'energy'):
                    for memory in ((False, True) if task == 'graph' else (False,)):
                        item = comparison(tests, task, method, 'bp', memory_a=memory, memory_b=memory, depth=depth)
                        if item: comparisons.append(item)
            if task == 'graph':
                for method in ('bp', 'pc', 'pcalm', 'energy'):
                    comparisons.append(comparison(tests, task, method, method, memory_a=True))
            if task == 'bytes':
                comparisons.append(comparison(tests, task, 'energy', 'bp_ce'))
        atomic_json(comparisons, args.out / 'comparisons.json')
        if source_identity() != identity:
            raise ValueError('Source changed during evaluation')
        status = 'completed'
    except BudgetExpired as exc:
        status = 'budget_exhausted'
        atomic_json({'status': status, 'reason': str(exc)}, args.out / 'incomplete.json')
    except BaseException as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        atomic_json({'status': status, 'reason': f'{type(exc).__name__}: {exc}'}, args.out / 'incomplete.json')
        raise
    finally:
        progress()
        budget.finish(status)
    return json.loads((args.out / 'summary.json').read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--training-only', action='store_true', help='Diagnostic pilots: fit and validate without evaluating test examples')
    p.add_argument('--tasks', nargs='+', choices=('fashion', 'graph', 'bytes'), default=['fashion', 'graph', 'bytes'])
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-seconds', type=float, default=7200)
    p.add_argument('--budget-ledger', type=Path, default=Path('runs/local_energy_budget.json'))
    p.add_argument('--prior-ledger', type=Path, default=Path('reports/publication/device_stages.json'))
    p.add_argument('--fashion-data', type=Path, default=Path('data/fashion_mnist'))
    p.add_argument('--fashion-depths', nargs='+', type=int, default=[8, 32])
    p.add_argument('--fashion-width', type=int, default=32)
    p.add_argument('--fashion-steps', type=int, default=400)
    p.add_argument('--fashion-train', type=int, default=6000)
    p.add_argument('--fashion-valid', type=int, default=1000)
    p.add_argument('--fashion-test', type=int, default=2000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--graph-depth', type=int, default=4)
    p.add_argument('--graph-width', type=int, default=48)
    p.add_argument('--graph-steps', type=int, default=1000)
    p.add_argument('--graph-train', type=int, default=4096)
    p.add_argument('--graph-valid', type=int, default=512)
    p.add_argument('--graph-test', type=int, default=1200)
    p.add_argument('--byte-data', type=Path, default=Path('data/enwik8'))
    p.add_argument('--byte-context', type=int, default=8)
    p.add_argument('--byte-depth', type=int, default=3)
    p.add_argument('--byte-width', type=int, default=32)
    p.add_argument('--byte-steps', type=int, default=200)
    p.add_argument('--byte-batch', type=int, default=8)
    p.add_argument('--byte-train', type=int, default=4096)
    p.add_argument('--byte-valid', type=int, default=128)
    p.add_argument('--byte-test', type=int, default=256)
    return p


if __name__ == '__main__':
    from .stages import interruption_handler
    with interruption_handler():
        result = run(parser().parse_args())
    print(json.dumps({key: value for key, value in result.items() if key not in ('records', 'test_results')}), flush=True)
