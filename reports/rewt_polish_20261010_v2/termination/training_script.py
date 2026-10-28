"""Continue a completed curriculum with stronger learned response termination.

The model receives the same raw bytes. Only training losses change: syntax
weights, final-newline weights and a training-only zero-remaining-steps label.
Absolute-wave warm controls test an existing energy setting at equal budgets.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.curriculum import training_annotations
from rewt.data import file_sha256, task_batch
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.model import TransformerConfig, TransformerLM
from rewt.optim import GradientFilter
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM
from train_rewt_curriculum import (auxiliary_losses, choose_batch, evaluate, make_model,
                                   read_tasks, supervision_views)


def prepare(args):
    base, out, runs = args.base_study.resolve(), args.output_dir.resolve(), args.run_dir.resolve()
    if (out / 'plan.json').exists():
        raise FileExistsError('a fresh output directory is required')
    results = json.loads((base / 'results.json').read_text())
    base_plan = json.loads((base / 'plan.json').read_text())
    cases = []
    for record in results:
        case = dict(record['case'], base_record=record,
                    learning_rate=record['case']['learning_rate'] * .5)
        cases.append(case)
        if case['warm']:
            cases.append(dict(case, name=case['name'] + '_absolute_wave', wave_geometry='absolute'))
    out.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid', 'test'):
        shutil.copyfile(base / f'tasks_{split}.jsonl', out / f'tasks_{split}.jsonl')
    plan = dict(base_plan, created=datetime.now(timezone.utc).isoformat(), base_plan=base_plan,
                base_study=str(base), base_results_sha256=file_sha256(base / 'results.json'),
                cases=cases, updates=args.updates, output_dir=str(out), run_dir=str(runs),
                source_sha256={p.name: file_sha256(p) for p in (ROOT / 'rewt').glob('*.py')},
                script_sha256=file_sha256(Path(__file__)),
                recipe='Continue each validation-selected checkpoint with a fresh AdamW optimizer and half its preceding learning rate; all shared supervision unchanged except syntax CE weight4, final-newline CE weight32, and equal weighting of trace-progress and final-answer zero-progress labels. Warm absolute-wave controls differ only in the existing wave coordinate setting.',
                source_selection='All source checkpoints and auxiliary heads selected from the completed preceding validation study, before fresh test scoring. Source best step can differ; charged budget includes all previous training updates.',
                test_not_evaluated=True)
    save_json(out / 'plan.json', plan)
    source = out / 'source'
    source.mkdir(exist_ok=True)
    for path in (ROOT / 'rewt').glob('*.py'):
        (source / path.name).write_bytes(path.read_bytes())
    (out / 'training_script.py').write_bytes(Path(__file__).read_bytes())
    return plan


def instantiate(case, plan):
    saved = torch.load(case['base_record']['checkpoint'], map_location='cpu', weights_only=True)
    assert file_sha256(Path(case['base_record']['checkpoint'])) == case['base_record']['checkpoint_sha256']
    torch.manual_seed(case['seed'])
    if case['family'] == 'transformer':
        model = TransformerLM(TransformerConfig(**saved['config']))
    else:
        config = dict(saved['config'])
        if case.get('wave_geometry'):
            config['wave_geometry'] = case['wave_geometry']
        model = PositionMemoryEnergyLM(PositionMemoryConfig(**config))
    model.load_state_dict(saved['weights'], strict=True)
    _, heads = make_model(case, plan)
    heads.load_state_dict(saved['training_heads'], strict=True)
    return model, heads


def termination_feedback(model, heads, output, tokens, tasks):
    auxiliary = auxiliary_losses(model, heads, output, tokens, tasks)
    features = supervision_views(model, output, tokens)[0]
    ends = [len((t.prompt + t.response).encode('ascii'))-2 for t in tasks]
    indices = torch.arange(len(tasks))
    assert (tokens[indices, torch.tensor(ends)] != 10).all()
    end_loss = F.cross_entropy(heads['progress'](features[indices, ends]), torch.zeros(len(tasks), dtype=torch.long))
    auxiliary['progress'] = .5*auxiliary['progress'] + .5*end_loss
    return auxiliary


def fit(case, plan):
    torch.set_num_threads(1)
    out, directory = Path(plan['output_dir']), Path(plan['run_dir']) / case['name']
    directory.mkdir(parents=True, exist_ok=True)
    model, heads = instantiate(case, plan)
    parameters = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=case['learning_rate'], betas=(.9, .95), weight_decay=.01)
    clip = GradientFilter(parameters, method='clip', max_norm=1., spike_threshold=5.)
    train, valid = read_tasks(out / 'tasks_train.jsonl'), read_tasks(out / 'tasks_valid.jsonl')
    pools = {(f, n): [i for i, t in enumerate(train) if t.family == f and t.steps == n]
             for f in ('arithmetic', 'rules', 'recall') for n in ((4, 8, 16) if f == 'recall' else (1, 2, 3))}
    begin = time.process_time()
    initial, rows = evaluate(model, valid, case['seed'])
    initial_validation_cpu = time.process_time()-begin
    best, best_step = initial['macro_complete'], 0
    arm = 'transformer' if case['family'] == 'transformer' else 'position_memory_energy'
    def snapshot(step, score):
        return dict(arm=arm, config=asdict(model.config), seed=case['seed'], weights=model.state_dict(),
                    training_heads=heads.state_dict(), step=step, selection=score, case=case,
                    source_checkpoint_sha256=case['base_record']['checkpoint_sha256'],
                    source_best_step=case['base_record']['best_step'])
    save_checkpoint(directory / 'best.pt', snapshot(0, initial))
    save_lines(out / f'{case["name"]}_validation_outputs.jsonl', rows)
    trace, training_cpu, validation_cpu, maximum, target_tokens = [], 0., initial_validation_cpu, 0., 0
    event('termination_started', case=case['name'], initial_macro=best, updates=plan['updates'])
    for step in range(plan['updates']):
        begin = time.process_time()
        tasks = choose_batch(train, pools, case, step + plan['base_plan']['updates'])
        batch = task_batch(tasks, range(len(tasks)), semantic_weight=16.)
        batch.token_weights = torch.where(batch.token_weights == 1, 4., batch.token_weights)
        for row, task in enumerate(tasks):
            for boundary in training_annotations(task)[0]:
                batch.token_weights[row, boundary.end] = 16.
            end = len((task.prompt + task.response).encode('ascii'))-2
            assert batch.targets[row, end] == 10
            batch.token_weights[row, end] = 32.
        model.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        rate = case['learning_rate'] * (.3+.7*(1+math.cos(math.pi*step/plan['updates']))/2)
        for group in optimizer.param_groups:
            group['lr'] = rate
        output = model(batch.tokens, details=True, trace_energy=False, state_gradient_interval=plan['state_gradient_interval'])
        per_byte = F.cross_entropy(output.logits.flatten(0, 1), batch.targets.flatten(), ignore_index=-100,
                                   reduction='none').reshape_as(batch.targets)
        ce = (per_byte*batch.token_weights).sum()/batch.token_weights.sum()
        auxiliary = termination_feedback(model, heads, output, batch.tokens, tasks)
        weights = dict(pointer=1., query=.5, copy=.5, number=.15, addition=.15, progress=1., binding=.2)
        loss = ce + sum(weights[k]*v for k, v in auxiliary.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{case["name"]}/{step}: nonfinite objective')
        loss.backward(); diagnostic = clip.apply(); optimizer.step()
        training_cpu += time.process_time()-begin
        maximum = max(maximum, diagnostic['raw_gradient_norm'])
        target_tokens += int(batch.valid.sum())
        row = dict(step=step+1, ce=float(ce.detach()), loss=float(loss.detach()),
                   auxiliary={k: float(v.detach()) for k, v in auxiliary.items()}, **diagnostic)
        if (step+1) % 256 == 0 or step+1 == plan['updates']:
            begin = time.process_time()
            score, rows = evaluate(model, valid, case['seed']+step)
            validation_cpu += time.process_time()-begin
            row['validation'] = score
            if score['macro_complete'] > best:
                best, best_step = score['macro_complete'], step+1
                save_checkpoint(directory / 'best.pt', snapshot(step+1, score))
                save_lines(out / f'{case["name"]}_validation_outputs.jsonl', rows)
            save_checkpoint(directory / 'last.pt', dict(snapshot(step+1, score), optimizer=optimizer.state_dict()))
            event('termination_validation', case=case['name'], step=step+1, macro=score['macro_complete'],
                  best=best, groups=score['groups'], maximum_gradient=maximum)
        elif (step+1) % 64 == 0:
            event('termination_progress', case=case['name'], step=step+1, ce=row['ce'])
        trace.append(row)
        if (step+1) % 64 == 0:
            save_lines(directory / 'trace.jsonl', trace)
    save_lines(directory / 'trace.jsonl', trace)
    base = case['base_record']
    record = dict(case=case, initial_validation=initial, best_step=best_step, best_validation_macro=best,
                  training_cpu_seconds=training_cpu, validation_cpu_seconds=validation_cpu,
                  total_training_cpu_seconds=training_cpu+base['training_cpu_seconds'],
                  total_validation_cpu_seconds=validation_cpu+base['validation_cpu_seconds'],
                  processed_response_tokens=target_tokens, parameters=model.parameters_count(),
                  auxiliary_parameters=sum(p.numel() for p in heads.parameters()), maximum_raw_gradient_norm=maximum,
                  checkpoint=str(directory / 'best.pt'), checkpoint_sha256=file_sha256(directory / 'best.pt'),
                  test_not_evaluated=True)
    save_json(out / f'{case["name"]}.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-study', type=Path, default=ROOT / 'reports/rewt_polish_20261010_v2/curriculum')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/rewt_polish_20261010_v2/termination')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/rewt_polish_20261010_v2/termination')
    parser.add_argument('--updates', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = prepare(args)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(fit, case, plan) for case in plan['cases']]
        results = [future.result() for future in as_completed(futures)]
    for name, sha in plan['source_sha256'].items():
        assert file_sha256(ROOT / 'rewt' / name) == sha
    for case in plan['cases']:
        assert file_sha256(Path(case['base_record']['checkpoint'])) == case['base_record']['checkpoint_sha256']
    save_json(Path(plan['output_dir']) / 'results.json', results)


if __name__ == '__main__':
    main()
