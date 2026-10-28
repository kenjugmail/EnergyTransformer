"""Paired validation-only continuations of the same foundation checkpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.data import file_sha256, task_batch
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.optim import GradientFilter
from rewt.tasks import Task

PARENT = Path(__file__).resolve().parent
OUT = PARENT/'stability'
RUNS = ROOT/'runs/rewt_learning_20261010/stability'
spec = importlib.util.spec_from_file_location('learning_foundation', PARENT/'foundation.py')
foundation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foundation)


def prepare(updates):
    OUT.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT/'foundation/combined.json').read_text())
    cases = {}
    for name, relative, interval in [('original',False,0), ('relative',True,0),
                                     ('original_credit16',False,16), ('relative_credit16',True,16)]:
        config = dict(parent['config'])
        if relative:
            config.update(wave_geometry='anchor_relative', initial_carry_scale=0.)
        cases[name] = dict(config=config, state_gradient_interval=interval)
    plan = dict(created=datetime.now(timezone.utc).isoformat(), seed=1041, updates=updates,
        cases=cases, checkpoint=parent['checkpoint'], checkpoint_sha256=parent['checkpoint_sha256'],
        parent_step=parent['best_step'], learning_rate=.001, semantic_weight=4., batch_size=24,
        validation_every=128, optimizer='Fresh AdamW state for every paired continuation.',
        tasks_parent=str(PARENT/'foundation'), selection='Best validation macro complete accuracy.',
        test_not_evaluated=True, source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
        script_sha256=file_sha256(Path(__file__)))
    assert file_sha256(Path(plan['checkpoint'])) == plan['checkpoint_sha256']
    save_json(OUT/'plan.json',plan)
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'rewt').glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
    return plan


def fit(name, settings, plan):
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    model=JointEnergyLM(EnergyConfig(**settings['config']))
    model.load_state_dict(torch.load(plan['checkpoint'],weights_only=False,map_location='cpu')['weights'])
    optimizer=torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],betas=(.9,.95),weight_decay=.01)
    clipping=GradientFilter(model.parameters(),method='clip',max_norm=1.,spike_threshold=5.)
    tasks={split:[Task(**json.loads(line)) for line in (Path(plan['tasks_parent'])/f'tasks_{split}.jsonl').read_text().splitlines()]
           for split in ('train','valid')}
    pools={family:[i for i,t in enumerate(tasks['train']) if t.family==family]
           for family in ('arithmetic','rules','recall')}
    directory=RUNS/name;directory.mkdir(parents=True,exist_ok=True)
    model.eval();initial,_=foundation.evaluate(model,tasks['valid'],1043)
    event('stability_started',case=name,validation=initial,updates=plan['updates'])
    trace=[];best=-1.;best_step=0;start=time.process_time();max_norm=0.
    for step in range(plan['updates']):
        rng=np.random.default_rng(plan['seed']*1000000+step+994)
        indices=[int(rng.choice(pools[family])) for family in pools for _ in range(plan['batch_size']//3)]
        rng.shuffle(indices)
        batch=task_batch(tasks['train'],indices,semantic_weight=plan['semantic_weight'])
        model.train();optimizer.zero_grad(set_to_none=True)
        output=model(batch.tokens,state_gradient_interval=settings['state_gradient_interval'])
        losses=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,
                               reduction='none').reshape_as(batch.targets)
        loss=(losses*batch.token_weights).sum()/batch.token_weights.sum()
        if not torch.isfinite(loss):raise FloatingPointError(f'{name}/{step}: nonfinite loss')
        loss.backward();g=clipping.apply();optimizer.step();max_norm=max(max_norm,g['raw_gradient_norm'])
        row=dict(step=step+1,loss=float(loss.detach()),**g)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            scores,outputs=foundation.evaluate(model,tasks['valid'],plan['seed']+step)
            row['validation']=scores
            if scores['id_macro_complete']>best:
                best=scores['id_macro_complete'];best_step=step+1
                save_checkpoint(directory/'best.pt',dict(arm='fourier_rewt',config=asdict(model.config),
                    seed=plan['seed'],weights=model.state_dict(),step=step+1,selection=scores))
                save_lines(OUT/f'{name}_validation_outputs.jsonl',outputs)
            save_checkpoint(directory/'last.pt',dict(arm='fourier_rewt',config=asdict(model.config),
                seed=plan['seed'],weights=model.state_dict(),step=step+1,optimizer=optimizer.state_dict()))
            event('stability_validation',case=name,step=step+1,loss=row['loss'],
                  id_macro=scores['id_macro_complete'],families=scores['id_family_complete'],max_raw_norm=max_norm)
        elif (step+1)%64==0:
            event('stability_progress',case=name,step=step+1,loss=row['loss'],raw_norm=g['raw_gradient_norm'])
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=name,best_step=best_step,best_id_macro_complete=best,
        maximum_raw_gradient_norm=max_norm,checkpoint=str(directory/'best.pt'),
        checkpoint_sha256=file_sha256(directory/'best.pt'),config=asdict(model.config),
        initial_validation=initial,test_not_evaluated=True,cpu_seconds=time.process_time()-start,
        state_gradient_interval=settings['state_gradient_interval'])
    save_json(OUT/f'{name}.json',result);event('stability_complete',case=name,best=best,max_raw_norm=max_norm)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=512)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();torch.set_num_threads(1)
    plan=prepare(args.updates)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,name,settings,plan) for name,settings in plan['cases'].items()]
        results=[f.result() for f in as_completed(futures)]
    assert file_sha256(Path(plan['checkpoint']))==plan['checkpoint_sha256']
    for name,sha in plan['source_sha256'].items():assert file_sha256(ROOT/'rewt'/name)==sha
    save_json(OUT/'results.json',results)


if __name__=='__main__':main()
