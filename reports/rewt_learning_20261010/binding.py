"""Learn fact retrieval and numeric representations using training-only labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
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
from rewt.experiment import event, save_checkpoint, save_json, save_lines
from rewt.model import EnergyConfig, JointEnergyLM
from rewt.memory import MemoryConfig, MemoryEnergyLM
from rewt.optim import GradientFilter
from rewt.tasks import Task

PARENT=Path(__file__).resolve().parent
OUT=PARENT/'binding'
RUNS=ROOT/'runs/rewt_learning_20261010/binding'
spec=importlib.util.spec_from_file_location('grounding_foundation',PARENT/'foundation.py')
foundation=importlib.util.module_from_spec(spec);spec.loader.exec_module(foundation)


def prepare(updates):
    OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads((PARENT/'precision/one_head.json').read_text())
    parents={name:parent for name in ('continued','aligned')}
    config={**parent['config'],'wave_geometry':'anchor_relative','initial_carry_scale':0.,'cache_history':False}
    plan=dict(created=datetime.now(timezone.utc).isoformat(),seed=1551,updates=updates,
        cases={'continued':1,'aligned':1},config=config,learning_rate=.0005,batch_size=24,parents=parents,
        semantic_weight=4.,state_gradient_interval=16,validation_every=128,
        checkpoint=parent['checkpoint'],checkpoint_sha256=parent['checkpoint_sha256'],
        source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},
        script_sha256=file_sha256(Path(__file__)),tasks_parent=str(PARENT/'foundation'),
        parent_step=parent['best_step'],first_value_weight=16.,
        curriculum='Continue the completed range curriculum with all operands <32.',
        auxiliary='Paired continuations of the same precision checkpoint. Aligned arm adds weight .2 cosine alignment of rule query/key vectors to detached embeddings of their observed source symbols. Both use an expanded training-only pool of rule graphs and the same integer precision losses. No symbol parsing or alignment targets enter generation.',
        selection='Validation mean complete accuracy over the three known-template families.',test_not_evaluated=True)
    for record in parents.values():assert file_sha256(Path(record['checkpoint']))==record['checkpoint_sha256']
    save_json(OUT/'plan.json',plan)
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'rewt').glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
    return plan


def label_locations(task):
    """Annotations are constructed by the trainer; no parser enters inference."""
    query=len(task.prompt)+len('steps=')-1
    if task.family=='rules':
        match=next(m for m in re.finditer(r'([A-Z])>([A-Z])',task.prompt)
                   if m[1]==task.payload['start'])
        return query,match.end()-1
    if task.family=='recall':
        match=re.search(r'(?:K=|remember )\d',task.prompt)
        if match is None:raise ValueError('Unrecognized training recall format')
        return query,match.end()-1
    return query,None


def auxiliary_losses(model,heads,output,tokens,tasks,align=False):
    features=output.details['features']
    encoded=model.encode_observed(model.embedding(tokens))
    h=model.config.attention_heads;d=model.config.width//h
    pointer_queries=[];pointer_keys=[];pointer_ends=[];pointer_labels=[]
    input_features=[];input_values=[];sum_features=[];sum_values=[]
    copied_features=[];copied_values=[]
    for row,task in enumerate(tasks):
        query,key=label_locations(task)
        number_positions=[m.end()-1 for m in re.finditer(r'\d+',task.prompt)] if key is None else []
        locations=[key] if key is not None else number_positions
        value_length=task.response.index(';ans=')-len('steps=')
        for position in range(query,query+value_length):
            pointer_queries.append(model.query(features[row,position,:model.config.width]
                +features[row,position,model.config.width:2*model.config.width]).reshape(h,d))
            pointer_keys.append(model.key(encoded[row]).reshape(tokens.shape[1],h,d))
            pointer_ends.append(position);pointer_labels.append(locations)
        if key is not None:
            copied_features.append(model.value(encoded[row,key]))
            copied_values.append(ord(task.response[len('steps=')]))
        else:
            for match in re.finditer(r'\d+',task.prompt):
                input_features.append(encoded[row,match.end()-1])
                input_values.append(int(match[0])/31.)
                input_features.append(model.value(encoded[row,match.end()-1]))
                input_values.append(int(match[0])/31.)
            for position in range(query,query+value_length):
                sum_features.append(features[row,position])
                sum_values.append((task.payload['start']+task.payload['deltas'][0])/62.)
    zero=features.sum()*0
    pointer=number=addition=zero
    if pointer_queries:
        scores=(torch.stack(pointer_keys)*torch.stack(pointer_queries)[:,None]).sum(-1)/math.sqrt(d)
        positions=torch.arange(tokens.shape[1])[None,:,None]
        ends=torch.tensor(pointer_ends)[:,None,None]
        scores=scores.masked_fill((positions>ends)|(positions<=ends-model.config.history),-torch.inf)
        logp=scores.log_softmax(1)
        pointer=torch.stack([-logp[i,indices].mean() if h==1 or len(indices)==1 else
            -logp[i,torch.tensor(indices),torch.arange(h)].mean()
            for i,indices in enumerate(pointer_labels)]).mean()
    if input_features:
        values=torch.stack(input_features).reshape(-1,h,d)
        number=torch.stack([F.mse_loss(heads['number'][head](values[:,head]).squeeze(-1),torch.tensor(input_values))
                            for head in range(h)]).mean()
        addition=F.mse_loss(heads['sum'](torch.stack(sum_features)).squeeze(-1),torch.tensor(sum_values))
    result=dict(pointer=pointer,number=number,addition=addition)
    if copied_features:
        result['copy']=F.cross_entropy(model.decoder(torch.stack(copied_features)),torch.tensor(copied_values))
    query_features=[];query_values=[]
    for row,task in enumerate(tasks):
        position,_=label_locations(task)
        query_features.append(features[row,position,model.config.width:2*model.config.width])
        value=ord(task.payload['start']) if task.family=='rules' else (48+task.payload['value'] if task.family=='recall' else task.payload['start'])
        query_values.append(value)
    result['query']=F.cross_entropy(heads['query'](torch.stack(query_features)),torch.tensor(query_values))
    if align:
        alignments=[]
        for row,task in enumerate(tasks):
            if task.family!='rules':continue
            position,_=label_locations(task)
            target=model.embedding(torch.tensor(ord(task.payload['start']))).detach()
            q=model.query(encoded[row,position]+features[row,position,model.config.width:2*model.config.width])
            alignments.append(1-F.cosine_similarity(q,target,dim=0))
            for match in re.finditer(r'([A-Z])>([A-Z])',task.prompt):
                key=model.key(encoded[row,match.end()-1])
                source=model.embedding(torch.tensor(ord(match[1]))).detach()
                alignments.append(1-F.cosine_similarity(key,source,dim=0))
        result['binding']=torch.stack(alignments).mean() if alignments else zero
    return result


def fit(name,head_count,plan):
    torch.set_num_threads(1);torch.manual_seed(plan['seed'])
    record=plan['parents'][name]
    model=MemoryEnergyLM(MemoryConfig(**record['config']))
    parent=torch.load(record['checkpoint'],weights_only=False,map_location='cpu')
    model.load_state_dict(parent['weights'])
    d=model.config.width
    heads=nn.ModuleDict({'number':nn.ModuleList([nn.Linear(d//head_count,1) for _ in range(head_count)]),
                         'sum':nn.Linear(3*d,1),'query':nn.Linear(d,256)})
    heads.load_state_dict(parent['training_heads'])
    parameters=list(model.parameters())+list(heads.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],betas=(.9,.95),weight_decay=.01)
    clipping=GradientFilter(parameters,method='clip',max_norm=1.,spike_threshold=5.)
    tasks={split:[Task(**json.loads(line)) for line in (Path(plan['tasks_parent'])/f'tasks_{split}.jsonl').read_text().splitlines()]
           for split in ('train','valid')}
    extra_path=OUT/'expanded_train_rules.jsonl'
    extra=[Task(**json.loads(line)) for line in extra_path.read_text().splitlines()]
    known={t.id for t in tasks['train']}
    tasks['train'].extend(t for t in extra if t.id not in known)
    pools={family:[i for i,t in enumerate(tasks['train']) if t.family==family] for family in ('arithmetic','rules','recall')}
    directory=RUNS/name;directory.mkdir(parents=True,exist_ok=True)
    trace=[];best=-1.;best_step=0;max_norm=0.;start=time.process_time()
    event('grounding_started',case=name,updates=plan['updates'],parameters=model.parameters_count(),
          auxiliary_training_parameters=sum(p.numel() for p in heads.parameters()))
    for step in range(plan['updates']):
        limit=32
        rng=np.random.default_rng(plan['seed']*1000000+step+994)
        arithmetic=[i for i in pools['arithmetic'] if max(tasks['train'][i].payload['start'],*tasks['train'][i].payload['deltas'])<limit]
        indices=[int(rng.choice(arithmetic if family=='arithmetic' else pools[family]))
                 for family in pools for _ in range(plan['batch_size']//3)]
        rng.shuffle(indices);selected=[tasks['train'][i] for i in indices]
        batch=task_batch(tasks['train'],indices,semantic_weight=plan['semantic_weight'])
        for row,task in enumerate(selected):
            first=len(task.prompt)+len('steps=')-1
            length=task.response.index(';ans=')-len('steps=')
            batch.token_weights[row,first:first+length]=plan['first_value_weight']
        model.train();heads.train();optimizer.zero_grad(set_to_none=True)
        scale=.3+.7*(1+math.cos(math.pi*step/plan['updates']))/2
        for g in optimizer.param_groups:g['lr']=plan['learning_rate']*scale
        output=model(batch.tokens,details=True,trace_energy=False,state_gradient_interval=plan['state_gradient_interval'])
        per_byte=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,
                                 reduction='none').reshape_as(batch.targets)
        ce=(per_byte*batch.token_weights).sum()/batch.token_weights.sum()
        auxiliary_terms=auxiliary_losses(model,heads,output,batch.tokens,selected,align=name=='aligned')
        ramp=min(1.,(step+1)/128)
        weights={'pointer':1.,'number':.25*31**2*ramp,'addition':.25*62**2*ramp,'copy':.5,'query':.5,'binding':.2}
        loss=ce+sum(value*weights[key] for key,value in auxiliary_terms.items())
        if not torch.isfinite(loss):raise FloatingPointError(f'{name}/{step}: nonfinite loss')
        loss.backward();g=clipping.apply();optimizer.step();max_norm=max(max_norm,g['raw_gradient_norm'])
        row=dict(step=step+1,loss=float(loss.detach()),ce=float(ce.detach()),operand_limit=limit,
            auxiliary={k:float(v.detach()) for k,v in auxiliary_terms.items()},**g)
        row.update(number_rmse_units=math.sqrt(row['auxiliary']['number'])*31,
                   sum_rmse_units=math.sqrt(row['auxiliary']['addition'])*62)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            scores,outputs=foundation.evaluate(model,tasks['valid'],plan['seed']+step);row['validation']=scores
            if scores['id_macro_complete']>best:
                best=scores['id_macro_complete'];best_step=step+1
                save_checkpoint(directory/'best.pt',dict(arm='memory_energy',config=asdict(model.config),seed=plan['seed'],
                    weights=model.state_dict(),step=step+1,selection=scores,training_heads=heads.state_dict()))
                save_lines(OUT/f'{name}_validation_outputs.jsonl',outputs)
            save_checkpoint(directory/'last.pt',dict(arm='memory_energy',config=asdict(model.config),seed=plan['seed'],
                weights=model.state_dict(),step=step+1,optimizer=optimizer.state_dict(),training_heads=heads.state_dict()))
            event('grounding_validation',case=name,step=step+1,ce=row['ce'],auxiliary=row['auxiliary'],
                  number_rmse_units=row['number_rmse_units'],sum_rmse_units=row['sum_rmse_units'],
                  id_macro=scores['id_macro_complete'],families=scores['id_family_complete'],max_raw_norm=max_norm)
        elif (step+1)%64==0:event('grounding_progress',case=name,step=step+1,ce=row['ce'],auxiliary=row['auxiliary'])
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=name,best_step=best_step,best_id_macro_complete=best,maximum_raw_gradient_norm=max_norm,
        checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'),config=asdict(model.config),
        test_not_evaluated=True,cpu_seconds=time.process_time()-start,parameters=model.parameters_count(),
        auxiliary_training_parameters=sum(p.numel() for p in heads.parameters()))
    save_json(OUT/f'{name}.json',result);event('grounding_complete',case=name,best=best,max_raw_norm=max_norm)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=1024)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();torch.set_num_threads(1)
    plan=prepare(args.updates)
    from rewt.tasks import generate_tasks,task_manifest
    extra=[t for t in generate_tasks('train',36000,seed=1909,number_range=32) if t.family=='rules' and t.steps==1]
    save_lines(OUT/'expanded_train_rules.jsonl',[t.to_dict() for t in extra])
    plan['expanded_rule_training']=task_manifest(extra)
    save_json(OUT/'plan.json',plan)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,name,auxiliary,plan) for name,auxiliary in plan['cases'].items()]
        results=[f.result() for f in as_completed(futures)]
    for record in plan['parents'].values():assert file_sha256(Path(record['checkpoint']))==record['checkpoint_sha256']
    for name,sha in plan['source_sha256'].items():assert file_sha256(ROOT/'rewt'/name)==sha
    save_json(OUT/'results.json',results)


if __name__=='__main__':main()
