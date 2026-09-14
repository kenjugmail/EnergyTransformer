"""Bounded controller study on oracle-addressed symbolic value tokens.

This is NOT natural-text pretraining: operation/address fields are supplied.
The next-token objective predicts each SET's value BEFORE writing it. The
augmented arm adds delayed-query CE and preservation of unrelated reads.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from energy_transformer.geometric_memory import EditMemory
from energy_transformer.recall_data import episodes
from .runtime import atomic_json,atomic_checkpoint,source_identity
from .memory_eval import paired_interval

KINDS=('diagonal','precision','delta','preconditioned_delta','projected')
OBJECTIVES=('next_value_token','next_value_token_plus_memory')


def categorical(values):
    label=values.argmax(-1)+1
    return torch.where(values.abs().sum(-1)==0,torch.zeros_like(label),label)


def read_logits(values):
    # Zero memory represents unknown. A fixed null logit makes that explicit.
    return torch.cat((torch.full_like(values[...,:1],.5),values),-1)*4.


def batch(seed,batch_size=8,dimension=16,value_dim=16,nkeys=6,correlation=.4):
    b=episodes(seed,batch=batch_size,nkeys=nkeys,dimension=dimension,value_dim=value_dim,edits=4,gap=2,correlation=correlation)
    b.values=F.one_hot(b.values.argmax(-1),value_dim).float()
    target_labels=categorical(b.targets)
    b.targets=F.one_hot((target_labels-1).clamp_min(0),value_dim).float()*(target_labels!=0)[...,None]
    return b


def objective(model,b,augmented):
    state=model.initial(b.keys.shape[0]);next_losses=[];locality=[]
    addresses=torch.arange(b.queries.shape[1])[None]
    for t in range(b.keys.shape[1]):
        before=model.read(b.queries,state)
        prediction=model.read(b.keys[:,t:t+1],state)[:,0]
        active=b.operations[:,t]==0
        if active.any():
            target=b.values[:,t].argmax(-1)+1
            next_losses.append(F.cross_entropy(read_logits(prediction[active]),target[active]))
        state=model.step(b.keys[:,t],b.values[:,t],b.operations[:,t],state)
        if augmented:
            after=model.read(b.queries,state)
            unchanged=addresses!=b.key_ids[:,t,None]
            locality.append((after-before.detach()).square().mean(-1)[unchanged].mean())
    next_loss=torch.stack(next_losses).mean()
    final=model.read(b.queries,state)
    delayed=F.cross_entropy(read_logits(final).flatten(0,1),categorical(b.targets).flatten())
    local=torch.stack(locality).mean() if locality else next_loss.new_zeros(())
    return next_loss+(delayed+.1*local if augmented else 0), {'next_value_ce':next_loss.detach(),
             'delayed_ce':delayed.detach(),'locality_mse':local.detach()}


@torch.no_grad()
def evaluate(model,seed=700001,batches=4):
    correctness=[];families={}
    for name,nkeys,corr in [('ordinary',6,.4),('similar',6,.95),('overload',24,.4)]:
        rows=[]
        for i in range(batches):
            b=batch(seed+i,batch_size=8,nkeys=nkeys,correlation=corr)
            result,_,_=model(b.keys,b.values,b.operations,b.queries)
            pred=read_logits(result).argmax(-1)
            target=categorical(b.targets)
            # Cluster the primary bootstrap unit by episode, not individual query.
            rows.extend((pred==target).float().mean(-1).tolist())
        families[name]=sum(rows)/len(rows);correctness.extend(rows)
    return {'accuracy':sum(correctness)/len(correctness),'episode_accuracies':correctness,'families':families}


def fit(kind,objective_name,seed,steps,out,deadline,learning_rate=.003):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.manual_seed(seed)
    model=EditMemory(kind,16,16)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.001)
    initial=evaluate(model,batches=1)
    best=initial['accuracy'];best_step=0
    best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    history=[];started=time.perf_counter();completed=0;status='completed'
    for step in range(1,steps+1):
        if time.perf_counter()>=deadline:
            status='budget_exhausted';break
        model.train()
        b=batch(200000+seed*10000+step)
        loss,parts=objective(model,b,objective_name.endswith('plus_memory'))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite controller loss')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();completed=step
        if step%20==0 or step==steps:
            model.eval();score=evaluate(model,batches=1)
            row={'step':step,'loss':float(loss.detach()),'validation_accuracy':score['accuracy'],
                 **{k:float(v) for k,v in parts.items()}}
            history.append(row)
            if score['accuracy']>best:
                best=score['accuracy'];best_step=step
                best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights);model.eval()
    checkpoint={'kind':kind,'dimension':16,'value_dim':16,'seed':seed,'objective':objective_name,
                'weights':best_weights,'selected_step':best_step}
    atomic_checkpoint(checkpoint,out/'controller.pt')
    # Test is evaluated only after validation selects the immutable weights.
    test=evaluate(model,seed=900001)
    record={'status':status,'kind':kind,'objective':objective_name,'seed':seed,'steps':completed,
            'selected_step':best_step,'initial_validation_accuracy':initial['accuracy'],
            'best_validation_accuracy':best,'test':test,'history':history,'seconds':time.perf_counter()-started,
            'learning_rate':learning_rate,'train_batch':8,'source_sha256':source_identity()}
    atomic_json(record,out/'result.json')
    print(json.dumps({k:v for k,v in record.items() if k not in ('test','history')},sort_keys=True),flush=True)
    return record


def promotion(records):
    decisions=[]
    for objective_name in OBJECTIVES:
        grouped={kind:sorted([r for r in records if r['kind']==kind and r['objective']==objective_name
                             and r['status']=='completed'],key=lambda r:r['seed']) for kind in KINDS}
        energy=grouped['precision']
        controls=[k for k in ('delta','preconditioned_delta') if len(grouped[k])==3]
        if len(energy)!=3 or not controls:
            decisions.append({'objective':objective_name,'decision':'not_evaluated','reason':'Three completed seeds required'})
            continue
        # Select the stronger control by validation, never by test results.
        control=max(controls,key=lambda k:sum(r['best_validation_accuracy'] for r in grouped[k])/3)
        # Average matched seeds within each episode; retain per-seed differences.
        a=torch.tensor([r['test']['episode_accuracies'] for r in energy])
        b=torch.tensor([r['test']['episode_accuracies'] for r in grouped[control]])
        interval=paired_interval(a.mean(0).tolist(),b.mean(0).tolist())
        gains=(a.mean(1)-b.mean(1)).tolist()
        passed=interval['difference']>=.05 and interval['low']>0 and min(gains)>0
        decisions.append({'objective':objective_name,'control':control,'decision':'candidate_for_replication' if passed else 'retain_bounded_result',
                          'paired_interval':interval,'per_seed_differences':gains,
                          'efficiency_gate':'not evaluated: raw state and provenance storage are not equivalent'})
    return decisions


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    deadline=time.perf_counter()+args.max_seconds
    records=[]
    for seed in args.seeds:
        for kind in KINDS:
            for objective_name in OBJECTIVES:
                if time.perf_counter()>=deadline:break
                records.append(fit(kind,objective_name,seed,args.steps,args.out/f'{kind}_{objective_name}_seed{seed}',deadline))
                atomic_json({'status':'in_progress','records':records},args.out/'summary.json')
    result={'status':'completed' if len(records)==len(args.seeds)*len(KINDS)*2 and all(r['status']=='completed' for r in records) else 'budget_exhausted',
            'records':records,'decisions':promotion(records),'input_mode':'oracle operations and addresses',
            'objective_scope':'Next symbolic value token predicted before writing; not full-text language modelling.',
            'comparison':'Identical LR/update budgets across objectives and mechanisms; no hyperparameter search.',
            'equivalence_control':'projected and preconditioned_delta are algebraic reparameterizations.',
            'limitations':['Small synthetic controller study; cannot establish general reasoning or large-pretraining readiness.']}
    atomic_json(result,args.out/'summary.json')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--max-seconds',type=float,default=900)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.steps<1 or args.max_seconds<=0:raise ValueError('Positive experiment budget required')
    result=run(args)
    print(json.dumps({'status':result['status'],'decisions':result['decisions']},indent=2))


if __name__=='__main__':main()
