"""Training-only learning fixture; memorization here is not held-out success."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from rewt.data import task_batch
from rewt.evaluate import generate_batch, reasoning_evaluation, refinement_audit
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.optim import GradientFilter
from rewt.tasks import Task, task_manifest

OUT = Path(__file__).resolve().parent
RUNS = ROOT / "runs/rewt_learning_20261010/micro"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sized_config(**kwargs):
    defaults = dict(width=48, hidden=8, modes=32, history=16, rounds=2,
        cache_history=True, early_backtrack=True, output_geometry='centered',
        normalize_frequency_blocks=True)
    base = EnergyConfig(**{**defaults, **kwargs})
    one = JointEnergyLM(base).parameters_count()
    two = JointEnergyLM(replace(base, hidden=9)).parameters_count()
    return replace(base, hidden=max(8, round(8+(250000-one)/(two-one))))


def cases():
    return dict(
        control=dict(config=asdict(sized_config()), semantic_weight=1.),
        semantic=dict(config=asdict(sized_config()), semantic_weight=4.),
        relative=dict(config=asdict(sized_config(compatibility_geometry='anchor_relative')), semantic_weight=1.),
        memory=dict(config=asdict(sized_config(recurrent_type='gru', context_kernel=3, history=128)), semantic_weight=1.),
        combined=dict(config=asdict(sized_config(compatibility_geometry='anchor_relative',
            recurrent_type='gru', context_kernel=3, history=128)), semantic_weight=4.))


def fixture_tasks():
    all_tasks = [Task(**json.loads(line)) for line in
                 (ROOT/'reports/rewt_stabilized_20261009/tasks_train.jsonl').read_text().splitlines()]
    return [task for family in ('arithmetic','rules','recall') for task in
            [t for t in all_tasks if t.family==family and t.steps <= (4 if family=='recall' else 1)][:4]]


def fit(name, settings, tasks_dicts, updates):
    torch.set_num_threads(1)
    torch.manual_seed(941)
    model = JointEnergyLM(EnergyConfig(**settings['config']))
    tasks = [Task(**r) for r in tasks_dicts]
    batch = task_batch(tasks, range(len(tasks)), semantic_weight=settings['semantic_weight'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01,betas=(.9,.95))
    clip = GradientFilter(model.parameters(), method='clip', max_norm=1., spike_threshold=5.)
    records=[]
    start = time.process_time()
    event('micro_started',case=name,parameters=model.parameters_count(),updates=updates,training_tasks=len(tasks))
    for step in range(updates):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:
            g['lr']=.003*min(1.,(step+1)/16)
        output = model(batch.tokens, rounds=0 if step<8 else 2)
        losses=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,
                               reduction='none').reshape_as(batch.targets)
        loss=losses[batch.valid].mean() if batch.token_weights is None else (
            losses*batch.token_weights).sum()/batch.token_weights.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {name}/{step}')
        loss.backward(); diagnostics=clip.apply(); optimizer.step()
        records.append(dict(step=step+1,loss=float(loss.detach()),**diagnostics))
        if (step+1)%40==0:
            event('micro_progress',case=name,step=step+1,loss=float(loss.detach()),raw_norm=diagnostics['raw_gradient_norm'])
    model.eval()
    summary, outputs = reasoning_evaluation(model,tasks,seed=943)
    audit = refinement_audit(model,batch.tokens[:2,:16])
    directory=RUNS/name; directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'model.pt'
    save_checkpoint(checkpoint,dict(arm='fourier_rewt',config=asdict(model.config),seed=941,
                                   weights=model.state_dict(),step=updates,optimizer=optimizer.state_dict()))
    save_lines(directory/'trace.jsonl',records)
    save_lines(OUT/f'micro_{name}_outputs.jsonl',outputs)
    result=dict(case=name,config=asdict(model.config),semantic_weight=settings['semantic_weight'],
                parameters=model.parameters_count(),updates=updates,training_cpu_seconds=time.process_time()-start,
                training_only_fixture=True,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                maximum_raw_gradient_norm=max(r['raw_gradient_norm'] for r in records),summary=summary,energy_audit=audit)
    save_json(OUT/f'micro_{name}.json',result)
    event('micro_complete',case=name,complete=summary['all']['both_correct'],families=summary['by_family'])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    torch.set_num_threads(1)
    selected=cases(); tasks=fixture_tasks()
    assert len(tasks)==12
    protocol=dict(updates=args.updates,seeds=[941],cases=selected,task_manifest=task_manifest(tasks),
        learning_rate=.003,gradient_clip=1.,micro_script_sha256=sha(Path(__file__)),
        source_sha256={x.name:sha(x) for x in sorted((ROOT/'rewt').glob('*.py'))},
        scope='Training-only 12-task learnability fixture. No held-out improvement claimed.')
    save_json(OUT/'micro_protocol.json',protocol)
    snapshot=OUT/'micro_source';snapshot.mkdir(exist_ok=True)
    for x in (ROOT/'rewt').glob('*.py'): (snapshot/x.name).write_bytes(x.read_bytes())
    save_lines(OUT/'micro_tasks.jsonl',[t.to_dict() for t in tasks])
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,name,settings,[t.to_dict() for t in tasks],args.updates)
                 for name,settings in selected.items()]
        results=[future.result() for future in as_completed(futures)]
    for name,expected in protocol['source_sha256'].items(): assert sha(ROOT/'rewt'/name)==expected
    save_json(OUT/'micro_results.json',results)


if __name__=='__main__':
    main()
