"""Learn one-step skills before expanding reasoning depth; select on validation."""
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
from rewt.evaluate import reasoning_evaluation, refinement_audit
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.optim import GradientFilter
from rewt.tasks import Task, generate_tasks, semantic_identity, semantic_partition, task_manifest

OUT = Path(__file__).resolve().parent / 'foundation'
RUNS = ROOT / 'runs/rewt_learning_20261010/foundation'
PARENT = ROOT / 'reports/rewt_learning_20261010'


def configuration_cases():
    spec = importlib.util.spec_from_file_location('rewt_micro_configuration', PARENT/'micro.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.cases()
    result['memory_semantic'] = dict(config=result['memory']['config'], semantic_weight=4.)
    return result


def make_tasks(split, previous_ids):
    arithmetic=[]
    for start in range(32):
        for delta in range(1,32):
            payload=dict(family='arithmetic',start=start,deltas=[delta])
            identity=semantic_identity(payload)
            if semantic_partition(identity)!=split or identity in previous_ids:
                continue
            for template in ((0,1) if split=='train' else (0,1,2)):
                prompt=(f'A={start};B=A+{delta};B=?\n' if template==0 else
                        f'start {start};add {delta};result?\n' if template==1 else
                        f'sum {start}+{delta}=?\n')
                value=start+delta
                arithmetic.append(Task(identity,'arithmetic',split,template,1,prompt,
                    f'steps={value};ans={value}\n',payload,'heldout_template' if template==2 else 'heldout_instance'))
    generated=generate_tasks(split,9000 if split=='train' else 3000,seed=973,
                             number_range=32,exclude_ids=previous_ids)
    cap=1024 if split=='train' else 96
    selected=arithmetic
    for family in ('rules','recall'):
        for template_group in ([None] if split=='train' else [(0,1),(2,)]):
            eligible=[t for t in generated if t.family==family and
                      t.steps <= (4 if family=='recall' else 1) and
                      (template_group is None or t.template in template_group)]
            selected.extend(eligible[:cap])
    return selected


def prepare(updates, seed):
    OUT.mkdir(parents=True,exist_ok=True)
    exclusions=set()
    for name in ('rewt_pilot_20261009_v2','rewt_improvements_20261009','rewt_stabilized_20261009'):
        path=ROOT/'reports'/name/'tasks_test.jsonl'
        if path.exists(): exclusions.update(json.loads(line)['id'] for line in path.read_text().splitlines())
    tasks={split:make_tasks(split,exclusions if split=='test' else set()) for split in ('train','valid','test')}
    ids={split:{t.id for t in values} for split,values in tasks.items()}
    assert ids['train'].isdisjoint(ids['valid']|ids['test']) and ids['valid'].isdisjoint(ids['test'])
    assert ids['test'].isdisjoint(exclusions)
    for split,values in tasks.items(): save_lines(OUT/f'tasks_{split}.jsonl',[t.to_dict() for t in values])
    configs=configuration_cases()
    plan=dict(created=datetime.now(timezone.utc).isoformat(),updates=updates,seed=seed,
        cases=configs,initialization='New random weights; the 12-task micro fits are not reused.',
        source_sha256={f.name:file_sha256(f) for f in sorted((ROOT/'rewt').glob('*.py'))},
        script_sha256=file_sha256(Path(__file__)),configuration_script_sha256=file_sha256(PARENT/'micro.py'),
        task_manifests={split:task_manifest(values) for split,values in tasks.items()},
        previous_test_id_exclusions=sorted(exclusions),learning_rate=.003,batch_size=24,
        validation_every=128,selection='Best mean family complete accuracy on validation templates 0/1. Template 2 remains a separate transfer measurement.',
        scope='One-step arithmetic/rules, recall delays 1/4, number range 0-31; no test outputs used for selection.')
    save_json(OUT/'plan.json',plan)
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for f in (ROOT/'rewt').glob('*.py'): (snapshot/f.name).write_bytes(f.read_bytes())
    return plan,tasks


def evaluate(model,tasks,seed):
    summary,rows=reasoning_evaluation(model,tasks,seed=seed)
    known=[r for r in rows if r['template'] in (0,1)]
    family={name:sum(r['both_correct'] for r in known if r['family']==name)/
        max(1,sum(r['family']==name for r in known)) for name in ('arithmetic','rules','recall')}
    return dict(summary=summary,id_family_complete=family,id_macro_complete=sum(family.values())/3),rows


def fit(name,settings,plan,train_dicts,valid_dicts):
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    model=JointEnergyLM(EnergyConfig(**settings['config']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
    clipping=GradientFilter(model.parameters(),method='clip',max_norm=1.,spike_threshold=5.)
    tasks=[Task(**v) for v in train_dicts];validation=[Task(**v) for v in valid_dicts]
    pools={family:[i for i,t in enumerate(tasks) if t.family==family] for family in ('arithmetic','rules','recall')}
    directory=RUNS/name;directory.mkdir(parents=True,exist_ok=True)
    trace=[];best=-1.;best_step=0;start=time.process_time();max_norm=0.
    event('foundation_started',case=name,updates=plan['updates'],parameters=model.parameters_count())
    for step in range(plan['updates']):
        rng=np.random.default_rng(plan['seed']*1000000+step+994)
        indices=[int(rng.choice(pools[family])) for family in pools for _ in range(plan['batch_size']//3)]
        rng.shuffle(indices)
        batch=task_batch(tasks,indices,semantic_weight=settings['semantic_weight'])
        model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr']=plan['learning_rate']*min(1.,(step+1)/16)
        output=model(batch.tokens,rounds=0 if step<16 else 2)
        per_byte=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,
                                reduction='none').reshape_as(batch.targets)
        loss=per_byte[batch.valid].mean() if batch.token_weights is None else (
            per_byte*batch.token_weights).sum()/batch.token_weights.sum()
        if not torch.isfinite(loss): raise FloatingPointError(f'{name}/{step}: nonfinite loss')
        loss.backward();g=clipping.apply();optimizer.step();max_norm=max(max_norm,g['raw_gradient_norm'])
        row=dict(step=step+1,loss=float(loss.detach()),**g)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            scores,outputs=evaluate(model,validation,plan['seed']+step)
            row['validation']=scores
            if scores['id_macro_complete']>best:
                best=scores['id_macro_complete'];best_step=step+1
                save_checkpoint(directory/'best.pt',dict(arm='fourier_rewt',config=asdict(model.config),
                    seed=plan['seed'],weights=model.state_dict(),step=step+1,selection=scores))
                save_lines(OUT/f'{name}_validation_outputs.jsonl',outputs)
            save_checkpoint(directory/'last.pt',dict(arm='fourier_rewt',config=asdict(model.config),
                seed=plan['seed'],weights=model.state_dict(),step=step+1,
                optimizer=optimizer.state_dict(),filter=clipping.state_dict()))
            event('foundation_validation',case=name,step=step+1,loss=row['loss'],
                  id_macro=scores['id_macro_complete'],families=scores['id_family_complete'],max_raw_norm=max_norm)
        elif (step+1)%64==0:
            event('foundation_progress',case=name,step=step+1,loss=row['loss'],raw_norm=g['raw_gradient_norm'])
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=name,seed=plan['seed'],parameters=model.parameters_count(),updates=plan['updates'],
        best_step=best_step,best_id_macro_complete=best,maximum_raw_gradient_norm=max_norm,
        fitting_and_validation_cpu_seconds=time.process_time()-start,checkpoint=str(directory/'best.pt'),
        checkpoint_sha256=file_sha256(directory/'best.pt'),config=asdict(model.config),
        semantic_weight=settings['semantic_weight'],test_not_evaluated=True)
    save_json(OUT/f'{name}.json',result);event('foundation_complete',**result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=1024)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--seed',type=int,default=941)
    args=parser.parse_args();torch.set_num_threads(1)
    plan,tasks=prepare(args.updates,args.seed)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,name,settings,plan,[t.to_dict() for t in tasks['train']],
                 [t.to_dict() for t in tasks['valid']]) for name,settings in plan['cases'].items()]
        results=[future.result() for future in as_completed(futures)]
    for name,expected in plan['source_sha256'].items(): assert file_sha256(ROOT/'rewt'/name)==expected
    save_json(OUT/'results.json',results)


if __name__=='__main__':main()
