"""Controlled broader-task training, with validation-only model selection.

All labels, prompt rendering and symbolic checks are outside model inference.
Cold energy and Transformer controls share examples and supervision targets.
Warm continuations are reported separately from random-initialization controls.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
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
from rewt.curriculum import generate_curriculum, render_task, training_annotations
from rewt.data import file_sha256, task_batch
from rewt.evaluate import reasoning_evaluation
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.model import TransformerConfig, TransformerLM
from rewt.optim import GradientFilter
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM
from rewt.tasks import Task, task_manifest


def read_tasks(path):
    return [Task(**json.loads(line)) for line in Path(path).read_text().splitlines()]


def prepare(args):
    out, runs = args.output_dir.resolve(), args.run_dir.resolve()
    if (out / 'plan.json').exists():
        raise FileExistsError('use a fresh output directory; completed or partial studies are preserved')
    out.mkdir(parents=True, exist_ok=True)
    parent = ROOT / 'runs/rewt_learning_20261010/multistep_balanced/progress/best.pt'
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    cfg = dict(saved['config'], cache_history=True)
    excluded = set()
    prior = []
    for path in (ROOT / 'reports').rglob('tasks_test.jsonl'):
        if out in path.parents:
            continue
        ids = {json.loads(line)['id'] for line in path.read_text().splitlines() if line}
        excluded.update(ids)
        prior.append(dict(path=str(path), sha256=file_sha256(path), identities=len(ids)))
    tasks = {split: generate_curriculum(split, 2000 if split == 'train' else 36 if split == 'valid' else 72,
                                       seed=2029, exclude_ids=excluded if split == 'test' else ())
             for split in ('train', 'valid', 'test')}
    ids = {s: {t.id for t in rows} for s, rows in tasks.items()}
    assert ids['train'].isdisjoint(ids['valid'] | ids['test']) and ids['valid'].isdisjoint(ids['test'])
    assert ids['test'].isdisjoint(excluded)
    for split, rows in tasks.items():
        save_lines(out / f'tasks_{split}.jsonl', [t.to_dict() for t in rows])
    cases = [dict(name=f'warm_heads{heads}', seed=2101, family='energy', heads=heads,
                  warm=True, learning_rate=.0005) for heads in (1, 2)]
    for seed in (2101, 2102, 2103):
        cases.append(dict(name=f'energy_{seed}', seed=seed, family='energy', heads=2,
                          warm=False, learning_rate=.001))
        for rate in (.001, .003):
            cases.append(dict(name=f'transformer_{seed}_lr{rate:g}', seed=seed, family='transformer',
                              heads=4, warm=False, learning_rate=rate))
    target = PositionMemoryEnergyLM(PositionMemoryConfig(**cfg)).parameters_count()
    candidates = [TransformerConfig(width=80, layers=2, heads=4, feedforward=h, context=128)
                  for h in range(465, 480)]
    trcfg = min(candidates, key=lambda c: abs(TransformerLM(c).parameters_count()-target))
    plan = dict(created=datetime.now(timezone.utc).isoformat(), updates=args.updates, batch_size=24,
                validation_every=256, state_gradient_interval=32, cases=cases,
                energy_config=cfg, transformer_config=asdict(trcfg),
                parent=dict(path=str(parent), sha256=file_sha256(parent),
                            prior_training='The completed task-model lineage; warm results are not equal-budget cold comparisons.'),
                task_manifests={s: task_manifest(rows) for s, rows in tasks.items()},
                previous_test_files=prior, excluded_previous_test_identities=sorted(excluded),
                source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
                script_sha256=file_sha256(Path(__file__)), output_dir=str(out), run_dir=str(runs),
                selection='Mean validation complete accuracy over family/depth/known-or-withheld-wording groups. Cold Transformer rate chosen by mean of three seed validation scores. Warm head count chosen on validation only. Fresh test templates 7/8 never enter validation or training.',
                curriculum='Arithmetic input values 0..63. First 256 updates one step, next 512 up to two, remainder up to three. Recall 4/8/16 distractors, requested key randomly placed. Balanced depth groups and five randomly rendered training wordings.',
                supervision='Common response CE, causal operand/fact pointer labels, numeric input and running-result Huber labels, observed-byte copy labels, source-symbol labels, remaining-step labels and observed rule-symbol query/key alignment. Architecture-specific hidden tensors/heads are reported; no auxiliary heads participate in inference.',
                fairness='Cold controls share seeds, batches, rendering randomness, target labels, updates and approximately256K inference parameters. Different learning-rate presets and representation dimensions; actual compute and auxiliary parameter counts reported. Warm continuations carry previous training cost.',
                test_not_evaluated=True)
    save_json(out / 'plan.json', plan)
    source = out / 'source'
    source.mkdir(exist_ok=True)
    for path in (ROOT / 'rewt').glob('*.py'):
        (source / path.name).write_bytes(path.read_bytes())
    (out / 'training_script.py').write_bytes(Path(__file__).read_bytes())
    return plan


def make_model(case, plan):
    torch.manual_seed(case['seed'])
    if case['family'] == 'transformer':
        model = TransformerLM(TransformerConfig(**plan['transformer_config']))
        d, features = model.config.width, model.config.width
    else:
        cfg = PositionMemoryConfig(**dict(plan['energy_config'], attention_heads=case['heads']))
        model = PositionMemoryEnergyLM(cfg)
        if case['warm']:
            saved = torch.load(plan['parent']['path'], map_location='cpu', weights_only=True)
            model.load_state_dict(saved['weights'], strict=True)
        d, features = model.config.width, 3 * model.config.width
    torch.manual_seed(case['seed'] + 100)
    heads = nn.ModuleDict(dict(number=nn.Linear(d, 1), addition=nn.Linear(features, 1),
                               query=nn.Linear(d, 256), progress=nn.Linear(features, 4)))
    return model, heads


def supervision_views(model, output, tokens):
    if isinstance(model, TransformerLM):
        details = output.details
        features = source = details['hidden']
        encoded = details['attention_inputs']
        query, keys, values = (details[f'attention_{name}'] for name in ('query', 'key', 'value'))
    else:
        d, h = model.config.width, model.config.attention_heads
        features = output.details['features']
        source = features[..., d:2*d]
        encoded = model.encode_tokens(tokens)
        shape = (*tokens.shape, h, d//h)
        query = model.query(features[..., :d] + source).reshape(shape)
        keys, values = model.key(encoded).reshape(shape), model.value(encoded).reshape(shape)
    return features, source, encoded, query, keys, values


def auxiliary_losses(model, heads, output, tokens, tasks):
    features, source, encoded, queries, keys, values = supervision_views(model, output, tokens)
    d, h = encoded.shape[-1], queries.shape[-2]
    ptr_rows, ptr_ats, ptr_targets = [], [], []
    num_rows, num_ats, num_targets = [], [], []
    sum_rows, sum_ats, sum_targets = [], [], []
    query_rows, query_ats, query_targets = [], [], []
    count_rows, count_ats, count_targets = [], [], []
    copy_rows, copy_ats, copy_targets = [], [], []
    align_queries, align_keys, align_query_targets, align_key_targets = [], [], [], []
    for row, task in enumerate(tasks):
        boundaries, numbers = training_annotations(task)
        for b in boundaries:
            for at in range(b.query, b.end):
                ptr_rows.append(row); ptr_ats.append(at)
                ptr_targets.append([b.keys[(head * len(b.keys)) // h] for head in range(h)] if h > 1 else b.keys)
                if b.numeric is not None:
                    sum_rows.append(row); sum_ats.append(at); sum_targets.append(b.numeric)
            query_rows.append(row); query_ats.append(b.query); query_targets.append(b.source)
            count_rows.extend([row, row]); count_ats.extend([b.query, b.end])
            count_targets.extend([b.remaining, b.remaining-1])
            if task.family != 'arithmetic':
                copy_rows.append(row); copy_ats.append(b.keys[0]); copy_targets.append(int(tokens[row, b.keys[0]]))
            if task.family == 'rules':
                align_queries.append(queries[row, b.query].flatten())
                align_query_targets.append(model.embedding(tokens.new_tensor(b.source)).detach())
        if task.family == 'rules':
            import re
            for match in re.finditer(r'([A-Z])>([A-Z])', task.prompt):
                align_keys.append(keys[row, match.end()-1].flatten())
                align_key_targets.append(model.embedding(tokens.new_tensor(ord(match[1]))).detach())
        for at, value in numbers:
            num_rows.append(row); num_ats.append(at); num_targets.append(value)
    q = queries[ptr_rows, ptr_ats]
    k = keys[ptr_rows]
    scores = (k * q[:, None]).sum(-1) / math.sqrt(d / h)
    positions = torch.arange(tokens.shape[1])[None, :, None]
    ends = torch.tensor(ptr_ats)[:, None, None]
    scores = scores.masked_fill(positions > ends, -torch.inf)
    logp = scores.log_softmax(1)
    pointer = torch.stack([-logp[i, list(target)].mean() if h == 1 else
                           -logp[i, torch.tensor(target), torch.arange(h)].mean()
                           for i, target in enumerate(ptr_targets)]).mean()
    result = dict(pointer=pointer,
                  query=F.cross_entropy(heads['query'](source[query_rows, query_ats]), torch.tensor(query_targets)),
                  progress=F.cross_entropy(heads['progress'](features[count_rows, count_ats]), torch.tensor(count_targets)))
    if num_rows:
        observed = encoded[num_rows, num_ats]
        retrieved = values[num_rows, num_ats].flatten(1)
        inputs = torch.cat((observed, retrieved), 0)
        target = torch.tensor(num_targets + num_targets, dtype=inputs.dtype)
        result['number'] = F.smooth_l1_loss(heads['number'](inputs).squeeze(-1) * 63, target)
        result['addition'] = F.smooth_l1_loss(heads['addition'](features[sum_rows, sum_ats]).squeeze(-1) * 252,
                                             torch.tensor(sum_targets, dtype=features.dtype))
    if copy_rows:
        result['copy'] = F.cross_entropy(model.decoder(values[copy_rows, copy_ats].flatten(1)), torch.tensor(copy_targets))
    if align_queries:
        lhs = torch.stack(align_queries + align_keys)
        result['binding'] = (1-F.cosine_similarity(lhs, torch.stack(align_query_targets + align_key_targets), dim=-1)).mean()
    return result


def evaluate(model, tasks, seed):
    summary, rows = reasoning_evaluation(model, tasks, seed=seed)
    groups = {}
    for family in ('arithmetic', 'rules', 'recall'):
        for depth in ((4, 8, 16) if family == 'recall' else (1, 2, 3)):
            for kind in ('familiar_format', 'withheld_wording'):
                subset = [r for r in rows if r['family'] == family and r['steps'] == depth and r['difficulty'] == kind]
                if subset:
                    groups[f'{family}/{depth}/{kind}'] = sum(r['both_correct'] for r in subset) / len(subset)
    return dict(macro_complete=sum(groups.values())/len(groups), groups=groups, summary=summary), rows


def choose_batch(tasks, pools, case, step):
    rng = np.random.default_rng(case['seed'] * 1000000 + step + 994)
    maximum = 1 if step < 256 else 2 if step < 768 else 3
    chosen = []
    for family in ('arithmetic', 'rules', 'recall'):
        depths = (4, 8, 16) if family == 'recall' else tuple(range(1, maximum+1))
        for j in range(8):
            task = tasks[int(rng.choice(pools[family, depths[j % len(depths)]]))]
            chosen.append(render_task(task.payload, 'train', int(rng.integers(5)), seed=int(rng.integers(2**32))))
    rng.shuffle(chosen)
    return chosen


def fit(case, plan):
    torch.set_num_threads(1)
    out, directory = Path(plan['output_dir']), Path(plan['run_dir']) / case['name']
    directory.mkdir(parents=True, exist_ok=True)
    model, heads = make_model(case, plan)
    parameters = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=case['learning_rate'], betas=(.9, .95), weight_decay=.01)
    clip = GradientFilter(parameters, method='clip', max_norm=1., spike_threshold=5.)
    train, valid = read_tasks(out / 'tasks_train.jsonl'), read_tasks(out / 'tasks_valid.jsonl')
    pools = {(f, depth): [i for i, t in enumerate(train) if t.family == f and t.steps == depth]
             for f in ('arithmetic', 'rules', 'recall') for depth in ((4, 8, 16) if f == 'recall' else (1, 2, 3))}
    trace, best, best_step, maximum_gradient, targets = [], -1., 0, 0., 0
    training_cpu = validation_cpu = 0.
    event('curriculum_started', case=case['name'], parameters=model.parameters_count(),
          auxiliary_parameters=sum(p.numel() for p in heads.parameters()), updates=plan['updates'])
    initial_validation = None
    if case['warm']:
        begin = time.process_time()
        initial_validation, initial_rows = evaluate(model, valid, case['seed'])
        validation_cpu += time.process_time()-begin
        best = initial_validation['macro_complete']
        save_checkpoint(directory / 'best.pt', dict(arm='position_memory_energy', config=asdict(model.config),
                        seed=case['seed'], weights=model.state_dict(), training_heads=heads.state_dict(),
                        step=0, selection=initial_validation, case=case))
        save_lines(out / f'{case["name"]}_validation_outputs.jsonl', initial_rows)
        event('curriculum_initial_validation', case=case['name'], macro=best, groups=initial_validation['groups'])
    for step in range(plan['updates']):
        begin = time.process_time()
        tasks = choose_batch(train, pools, case, step)
        batch = task_batch(tasks, range(len(tasks)), semantic_weight=16.)
        for row, task in enumerate(tasks):
            for b in training_annotations(task)[0]:
                batch.token_weights[row, b.end] = 16.
        model.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        rate = case['learning_rate'] * min(1., (step+1)/32) * (.3+.7*(1+math.cos(math.pi*step/plan['updates']))/2)
        for group in optimizer.param_groups:
            group['lr'] = rate
        output = model(batch.tokens, details=True, trace_energy=False,
                       state_gradient_interval=plan['state_gradient_interval'])
        per_byte = F.cross_entropy(output.logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100,
                                   reduction='none').reshape_as(batch.targets)
        ce = (per_byte * batch.token_weights).sum() / batch.token_weights.sum()
        auxiliary = auxiliary_losses(model, heads, output, batch.tokens, tasks)
        ramp = min(1., (step+1)/256)
        weights = dict(pointer=1., query=.5, copy=.5, number=.15*ramp, addition=.15*ramp, progress=1., binding=.2)
        loss = ce + sum(weights[key]*value for key, value in auxiliary.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{case["name"]}/{step}: nonfinite objective')
        loss.backward(); diagnostic = clip.apply(); optimizer.step()
        training_cpu += time.process_time()-begin
        maximum_gradient = max(maximum_gradient, diagnostic['raw_gradient_norm'])
        targets += int(batch.valid.sum())
        row = dict(step=step+1, ce=float(ce.detach()), loss=float(loss.detach()),
                   auxiliary={k: float(v.detach()) for k, v in auxiliary.items()}, **diagnostic)
        if (step+1) % plan['validation_every'] == 0 or step+1 == plan['updates']:
            begin = time.process_time()
            score, rows = evaluate(model, valid, case['seed']+step)
            validation_cpu += time.process_time()-begin
            row['validation'] = score
            saved = dict(arm='transformer' if case['family'] == 'transformer' else 'position_memory_energy',
                         config=asdict(model.config), seed=case['seed'], weights=model.state_dict(),
                         training_heads=heads.state_dict(), step=step+1, selection=score, case=case)
            if score['macro_complete'] > best:
                best, best_step = score['macro_complete'], step+1
                save_checkpoint(directory / 'best.pt', saved)
                save_lines(out / f'{case["name"]}_validation_outputs.jsonl', rows)
            save_checkpoint(directory / 'last.pt', dict(saved, optimizer=optimizer.state_dict()))
            event('curriculum_validation', case=case['name'], step=step+1, macro=score['macro_complete'],
                  best=best, groups=score['groups'], maximum_gradient=maximum_gradient)
        elif (step+1) % 64 == 0:
            event('curriculum_progress', case=case['name'], step=step+1, ce=row['ce'], loss=row['loss'])
        trace.append(row)
        if (step+1) % 64 == 0:
            save_lines(directory / 'trace.jsonl', trace)
    save_lines(directory / 'trace.jsonl', trace)
    result = dict(case=case, parameters=model.parameters_count(), auxiliary_parameters=sum(p.numel() for p in heads.parameters()),
                  best_step=best_step, best_validation_macro=best, training_cpu_seconds=training_cpu,
                  validation_cpu_seconds=validation_cpu, processed_response_tokens=targets,
                  maximum_raw_gradient_norm=maximum_gradient, checkpoint=str(directory / 'best.pt'),
                  checkpoint_sha256=file_sha256(directory / 'best.pt'), initial_validation=initial_validation,
                  test_not_evaluated=True)
    save_json(out / f'{case["name"]}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/rewt_polish_20261010_v2/curriculum')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/rewt_polish_20261010_v2/curriculum')
    parser.add_argument('--updates', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.smoke:
        saved = torch.load(ROOT / 'runs/rewt_learning_20261010/multistep_balanced/progress/best.pt', map_location='cpu', weights_only=True)
        plan = dict(energy_config=dict(saved['config'], cache_history=True),
                    transformer_config=asdict(TransformerConfig(width=80, layers=2, heads=4, feedforward=470, context=128)))
        tasks = generate_curriculum('train', 1)
        for family in ('energy', 'transformer'):
            model, heads = make_model(dict(family=family, warm=False, heads=2, seed=2101), plan)
            batch = task_batch(tasks, range(len(tasks)), semantic_weight=16.)
            output = model(batch.tokens, details=True, trace_energy=False, state_gradient_interval=32)
            auxiliary = auxiliary_losses(model, heads, output, batch.tokens, tasks)
            loss = sum(auxiliary.values()) + output.logits.square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            event('curriculum_smoke_passed', family=family, tasks=len(tasks), parameters=model.parameters_count(),
                  losses={k: float(v.detach()) for k, v in auxiliary.items()})
        return
    plan = prepare(args)
    if args.prepare_only:
        event('curriculum_prepared', cases=len(plan['cases']))
        return
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(fit, case, plan) for case in plan['cases']]
        results = [future.result() for future in as_completed(futures)]
    assert file_sha256(Path(plan['parent']['path'])) == plan['parent']['sha256']
    for name, digest in plan['source_sha256'].items():
        assert file_sha256(ROOT / 'rewt' / name) == digest
    save_json(Path(plan['output_dir']) / 'results.json', results)


if __name__ == '__main__':
    main()
