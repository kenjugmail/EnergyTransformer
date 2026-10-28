"""A causal, learned continuation curriculum; annotations exist only in training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rewt.data import file_sha256, task_batch
from rewt.evaluate import reasoning_evaluation
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.memory import MemoryConfig, MemoryEnergyLM
from rewt.position_memory import PositionMemoryConfig, PositionMemoryEnergyLM
from rewt.optim import GradientFilter
from rewt.tasks import Task, generate_tasks, solve, task_manifest

PARENT=Path(__file__).resolve().parent
OUT=PARENT/'multistep_huber'
RUNS=ROOT/'runs/rewt_learning_20261010/multistep_huber'


def read_tasks(path):
    return [Task(**json.loads(line)) for line in path.read_text().splitlines()]


def prepare(updates):
    OUT.mkdir(parents=True,exist_ok=True)
    candidates=[PARENT/'precision/one_head.json',PARENT/'binding/continued.json',PARENT/'binding/aligned.json']
    records=[json.loads(p.read_text()) for p in candidates if p.exists()]
    parent=max(records,key=lambda r:r['best_id_macro_complete'])
    assert file_sha256(Path(parent['checkpoint']))==parent['checkpoint_sha256']
    tasks={split:read_tasks(PARENT/'foundation'/f'tasks_{split}.jsonl') for split in ('train','valid','test')}
    previous=set()
    for folder in ('rewt_pilot_20261009_v2','rewt_improvements_20261009','rewt_stabilized_20261009'):
        path=ROOT/'reports'/folder/'tasks_test.jsonl'
        if path.exists():previous.update(t.id for t in read_tasks(path))
    for split in tasks:
        extra=generate_tasks(split,18000 if split=='train' else 960,seed=1929,number_range=32,
            exclude_ids={t.id for t in tasks[split]}|(previous if split=='test' else set()))
        tasks[split].extend(extra)
    binding=PARENT/'binding/expanded_train_rules.jsonl'
    if binding.exists():
        ids={t.id for t in tasks['train']}
        tasks['train'].extend(t for t in read_tasks(binding) if t.id not in ids)
    ids={split:{t.id for t in values} for split,values in tasks.items()}
    assert ids['train'].isdisjoint(ids['valid']|ids['test']) and ids['valid'].isdisjoint(ids['test'])
    assert ids['test'].isdisjoint(previous)
    for split,values in tasks.items():save_lines(OUT/f'tasks_{split}.jsonl',[t.to_dict() for t in values])
    plan=dict(created=datetime.now(timezone.utc).isoformat(),seed=1781,updates=updates,
        parent=parent,cases=['continued','progress','workspace'],learning_rate=.0005,batch_size=24,
        state_gradient_interval=64,validation_every=256,semantic_weight=4.,value_weight=16.,
        curriculum='First 128 updates: one hop/recall <=4; then through 1536: <=2 hops/recall <=8; remainder: <=3 hops/recall <=16. Bounded numeric loss: .5 Huber in integer units instead of .25 squared integer error. Trace delimiter weight 8 in both arms; learned positions in the progress arm add 6144 active parameters (2.46%). All arithmetic inputs <32.',
        selection='Known-template validation macro over family/depth groups (arithmetic/rules 1-3; recall delays 1/4/8/16). Four hops and template 2 are separate transfer measurements.',
        training_only_auxiliary='All arms have the same training-only progress head and bounded numeric losses; workspace also supervises the accepted joint recurrent/attention/feedforward states instead of the pre-refinement conditioning features; the continued arm has no position embeddings, and the progress arm adds observed-token positions. Both: causal operand/fact attention labels, observed-number and running-sum regression, source-symbol classification, observed fact-value copying. Progress arm also predicts remaining trace steps at trace boundaries. All annotation code runs outside the model and generator.',
        task_manifests={s:task_manifest(v) for s,v in tasks.items()},previous_test_exclusions=sorted(previous),
        source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
        script_sha256=file_sha256(Path(__file__)),test_not_evaluated=True,shared_onehop_test_already_evaluated=True,primary_test_exclude_ids=sorted({t.id for t in read_tasks(PARENT/'foundation/tasks_test.jsonl')}))
    save_json(OUT/'plan.json',plan)
    source=OUT/'source';source.mkdir(exist_ok=True)
    for p in (ROOT/'rewt').glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
    return plan


def annotations(task):
    """Every key/source below is present in the prefix at its query position."""
    values,_=solve(task.payload)
    cursor=len(task.prompt)+len('steps=')-1
    numbers=[(m.end()-1,int(m[0])) for m in re.finditer(r'\d+',task.prompt)] if task.family=='arithmetic' else []
    boundaries=[];previous=None
    edges={m[1]:m.end()-1 for m in re.finditer(r'([A-Z])>([A-Z])',task.prompt)}
    recall=re.search(r'(?:K=|remember )\d',task.prompt) if task.family=='recall' else None
    for index,value in enumerate(values):
        if task.family=='arithmetic':
            keys=[numbers[0][0] if index==0 else previous,numbers[index+1][0]]
            source=task.payload['start'] if index==0 else int(values[index-1])
        elif task.family=='rules':
            source=ord(task.payload['start'] if index==0 else values[index-1])
            keys=[edges[chr(source)]]
        else:
            assert recall is not None
            keys=[recall.end()-1];source=48+task.payload['value']
        assert all(0<=key<=cursor for key in keys)
        boundaries.append(dict(query=cursor,end=cursor+len(value),keys=keys,source=source,
            numeric=int(value) if task.family=='arithmetic' else None,remaining=len(values)-index))
        previous=cursor+len(value)
        if task.family=='arithmetic':numbers.append((previous,int(value)))
        cursor+=len(value)+1
    return boundaries,numbers


def losses(model,heads,output,tokens,tasks,progress,workspace=False):
    features=output.details['features'];d=model.config.width
    encoded=model.encode_tokens(tokens) if isinstance(model,PositionMemoryEnergyLM) else model.encode_observed(model.embedding(tokens));zero=features.sum()*0
    pointers=[];copies=[];copy_labels=[];number_inputs=[];number_labels=[]
    sum_inputs=[];sum_labels=[];query_inputs=[];query_labels=[];count_inputs=[];count_labels=[]
    for row,task in enumerate(tasks):
        boundaries,numbers=annotations(task)
        keys=model.key(encoded[row]);values=model.value(encoded[row])
        for boundary in boundaries:
            position=boundary['query']
            for at in range(position,boundary['end']):
                q=model.query(features[row,at,:d]+features[row,at,d:2*d])
                scores=keys@q/math.sqrt(d)
                indices=torch.arange(len(scores));scores=scores.masked_fill((indices>at)|(indices<=at-model.config.history),-torch.inf)
                pointers.append(-scores.log_softmax(0)[boundary['keys']].mean())
                if boundary['numeric'] is not None:
                    sum_inputs.append(output.details['joint'][row,at,:3*d] if workspace else features[row,at]);sum_labels.append(boundary['numeric']/62.)
            query_inputs.append(features[row,position,d:2*d]);query_labels.append(boundary['source'])
            count_inputs.extend([features[row,position],features[row,boundary['end']]])
            count_labels.extend([boundary['remaining'],boundary['remaining']-1])
            if task.family!='arithmetic':
                key=boundary['keys'][0];copies.append(values[key]);copy_labels.append(int(tokens[row,key]))
        for position,value in numbers:
            number_inputs.extend([encoded[row,position],values[position]]);number_labels.extend([value/31.,value/31.])
    result={'pointer':torch.stack(pointers).mean(),
            'query':F.cross_entropy(heads['query'](torch.stack(query_inputs)),torch.tensor(query_labels))}
    if copies:result['copy']=F.cross_entropy(model.decoder(torch.stack(copies)),torch.tensor(copy_labels))
    if number_inputs:
        result['number']=F.smooth_l1_loss(heads['number'][0](torch.stack(number_inputs)).squeeze(-1)*31,torch.tensor(number_labels)*31)
        result['addition']=F.smooth_l1_loss(heads['sum'](torch.stack(sum_inputs)).squeeze(-1)*62,torch.tensor(sum_labels)*62)
    if progress:
        result['progress']=F.cross_entropy(heads['progress'](torch.stack(count_inputs)),torch.tensor(count_labels))
    return result


def evaluate(model,tasks,seed):
    summary,rows=reasoning_evaluation(model,tasks,seed=seed)
    groups={}
    for family in ('arithmetic','rules','recall'):
        for depth in ((1,4,8,16) if family=='recall' else (1,2,3)):
            chosen=[r for r in rows if r['family']==family and r['steps']==depth and r['template']<2]
            if chosen:groups[f'{family}/{depth}']=sum(r['both_correct'] for r in chosen)/len(chosen)
    return dict(summary=summary,known_depth_groups=groups,macro_complete=sum(groups.values())/len(groups)),rows


def make_model(plan,case):
    parent=torch.load(plan['parent']['checkpoint'],map_location='cpu',weights_only=True)
    model=(PositionMemoryEnergyLM(PositionMemoryConfig(**parent['config'],position_period=128)) if case in ('progress','workspace') else MemoryEnergyLM(MemoryConfig(**parent['config'])))
    loaded=model.load_state_dict(parent['weights'],strict=False)
    assert not loaded.unexpected_keys and loaded.missing_keys==(['position.weight'] if case in ('progress','workspace') else [])
    d=model.config.width
    heads=nn.ModuleDict({'number':nn.ModuleList([nn.Linear(d,1)]),'sum':nn.Linear(3*d,1),'query':nn.Linear(d,256)})
    heads.load_state_dict(parent['training_heads'])
    torch.manual_seed(plan.get('seed',1721)+1)
    heads['progress']=nn.Linear(3*d,4)
    return model,heads


def fit(case,plan):
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    model,heads=make_model(plan,case)
    parameters=list(model.parameters())+list(heads.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
    clip=GradientFilter(parameters,method='clip',max_norm=1.,spike_threshold=5.)
    tasks=read_tasks(OUT/'tasks_train.jsonl');valid=read_tasks(OUT/'tasks_valid.jsonl')
    directory=RUNS/case;directory.mkdir(parents=True,exist_ok=True)
    trace=[];best=-1.;best_step=0;max_norm=0.;cpu=time.process_time()
    event('multistep_started',case=case,updates=plan['updates'],parameters=model.parameters_count(),
        training_auxiliary_parameters=sum(p.numel() for p in heads.parameters()))
    for step in range(plan['updates']):
        depth=1 if step<128 else 2 if step<1536 else 3
        delay={1:4,2:8,3:16}[depth]
        pools={f:[i for i,t in enumerate(tasks) if t.family==f and t.steps<=(delay if f=='recall' else depth)]
               for f in ('arithmetic','rules','recall')}
        rng=np.random.default_rng(plan['seed']*1000000+step+994)
        indices=[int(rng.choice(pools[f])) for f in pools for _ in range(plan['batch_size']//3)]
        rng.shuffle(indices);chosen=[tasks[i] for i in indices]
        batch=task_batch(tasks,indices,semantic_weight=plan['semantic_weight'])
        for row,task in enumerate(chosen):
            boundaries,_=annotations(task)
            for b in boundaries:
                batch.token_weights[row,b['query']:b['end']]=plan['value_weight']
                batch.token_weights[row,b['end']]=8.
        model.train();heads.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr']=plan['learning_rate']*(.3+.7*(1+math.cos(math.pi*step/plan['updates']))/2)
        output=model(batch.tokens,details=True,trace_energy=False,state_gradient_interval=plan['state_gradient_interval'])
        per_byte=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,reduction='none').reshape_as(batch.targets)
        ce=(per_byte*batch.token_weights).sum()/batch.token_weights.sum()
        auxiliary=losses(model,heads,output,batch.tokens,chosen,True,workspace=case=='workspace')
        ramp=min(1.,(step+1)/128)
        weights={'pointer':1.,'query':.5,'copy':.5,'number':.5*ramp,'addition':.5*ramp,'progress':.5}
        loss=ce+sum(weights[k]*v for k,v in auxiliary.items())
        if not torch.isfinite(loss):raise FloatingPointError(f'{case}/{step}: nonfinite loss')
        loss.backward();g=clip.apply();optimizer.step();max_norm=max(max_norm,g['raw_gradient_norm'])
        row=dict(step=step+1,depth_limit=depth,recall_delay_limit=delay,ce=float(ce.detach()),
            loss=float(loss.detach()),auxiliary={k:float(v.detach()) for k,v in auxiliary.items()},**g)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            scores,outputs=evaluate(model,valid,plan['seed']+step);row['validation']=scores
            if scores['macro_complete']>best:
                best=scores['macro_complete'];best_step=step+1
                save_checkpoint(directory/'best.pt',dict(arm='position_memory_energy' if isinstance(model,PositionMemoryEnergyLM) else 'memory_energy',config=asdict(model.config),
                    seed=plan['seed'],weights=model.state_dict(),step=step+1,selection=scores,training_heads=heads.state_dict()))
                save_lines(OUT/f'{case}_validation_outputs.jsonl',outputs)
            save_checkpoint(directory/'last.pt',dict(arm='position_memory_energy' if isinstance(model,PositionMemoryEnergyLM) else 'memory_energy',config=asdict(model.config),seed=plan['seed'],
                weights=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,training_heads=heads.state_dict()))
            event('multistep_validation',case=case,step=step+1,macro=best,groups=scores['known_depth_groups'],max_raw_norm=max_norm)
        elif (step+1)%64==0:event('multistep_progress',case=case,step=step+1,ce=row['ce'],depth=depth)
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=case,seed=plan['seed'],parameters=model.parameters_count(),best_step=best_step,
        best_validation_macro=best,maximum_raw_gradient_norm=max_norm,cpu_seconds=time.process_time()-cpu,
        auxiliary_training_parameters=sum(p.numel() for p in heads.parameters()),checkpoint=str(directory/'best.pt'),
        checkpoint_sha256=file_sha256(directory/'best.pt'),config=asdict(model.config),test_not_evaluated=True)
    save_json(OUT/f'{case}.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.smoke:
        parent=json.loads((PARENT/'precision/one_head.json').read_text());plan={'parent':parent}
        tasks=[t for t in generate_tasks('train',300,seed=1929,number_range=32) if t.template<2]
        chosen=[next(t for t in tasks if t.family==f and t.steps==depth)
                for f,depth in [('arithmetic',1),('arithmetic',2),('arithmetic',3),('rules',1),('rules',2),('rules',3),('recall',16)]]
        model,heads=make_model(plan,'progress');batch=task_batch(chosen,range(len(chosen)),semantic_weight=4.)
        output=model(batch.tokens,details=True,trace_energy=False,state_gradient_interval=64)
        values=losses(model,heads,output,batch.tokens,chosen,True,workspace=True)
        total=sum(values.values());assert torch.isfinite(total);total.backward()
        event('multistep_smoke_passed',tasks=len(chosen),losses={k:float(v.detach()) for k,v in values.items()});return
    plan=prepare(args.updates)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,case,plan) for case in plan['cases']]
        results=[f.result() for f in as_completed(futures)]
    assert file_sha256(Path(plan['parent']['checkpoint']))==plan['parent']['checkpoint_sha256']
    for name,digest in plan['source_sha256'].items():assert file_sha256(ROOT/'rewt'/name)==digest
    save_json(OUT/'results.json',results)


if __name__=='__main__':main()
