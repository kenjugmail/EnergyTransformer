"""Bounded, paired warm-start study of forecast conditioning objectives.

Forecasts are distributions over frozen-teacher summaries, never observed facts.
No decoder or teacher is trained. Saved checkpoints retain the calibrated v2
format; an old checkpoint-bound online gate cannot authorize a new checkpoint.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from .conditioning import GaussianMixture
from .experiment import load_head, atomic_save, digest
from .repair import split_validation, subset

KINDS=('joint', 'conditional', 'conditional_mse')


def source_hash():
    h=hashlib.sha256()
    for name in ('conditional_training.py','conditioning.py','model.py','calibrated.py'):
        p=Path(__file__).with_name(name);h.update(name.encode()+b'\0'+p.read_bytes())
    return h.hexdigest()


def objective_terms(f:GaussianMixture,y:torch.Tensor,variables:int):
    """All scores are proper conditional NLLs before optional MSE weighting.

    Stage zero evaluates the whole future; stage i evaluates only the remaining
    coordinates conditional on the first i complete, eventually observed horizons.
    Future values are targets here. They are NOT inference inputs before maturity.
    """
    if type(variables) is not int or variables<1 or f.dimension%variables:
        raise ValueError('Variable count must divide the future dimension')
    if y.shape!=f.mean.shape:raise ValueError('Target shape mismatch')
    nll=[f.energy(y)/f.dimension];mse=[(f.mean-y).square().mean(-1)]
    for n in range(variables,f.dimension,variables):
        post=f.condition_prefix(n,y[:,:n]).distribution
        nll.append(post.energy(y[:,n:])/(f.dimension-n))
        mse.append((post.mean-y[:,n:]).square().mean(-1))
    return torch.stack(nll,-1),torch.stack(mse,-1)


@torch.no_grad()
def measure(model,part):
    ns=[];ms=[]
    for start in range(0,len(part['x']),96):
        f=GaussianMixture.from_landscape(model(part['x'][start:start+96]))
        n,m=objective_terms(f,part['y'][start:start+96].flatten(1).double(),model.config.variables)
        ns.append(n);ms.append(m)
    n=torch.cat(ns).mean(0);m=torch.cat(ms).mean(0)
    # All training arms have exactly the same checkpoint-selection objective.
    score=.5*n[0]+.5*n[1:].mean()
    return {'selection_score':float(score),'nll_by_stage':n.tolist(),'mse_by_stage':m.tolist()}


def train(data_path,parents,out,kind,seed=11,steps=350,resume=False,stop_at=None):
    if kind not in KINDS:raise ValueError('Unknown objective')
    if type(steps) is not int or not 1<=steps<=2000:raise ValueError('Choose 1..2000 updates')
    if stop_at is not None and (type(stop_at) is not int or not 1<=stop_at<=steps):raise ValueError('Invalid stop_at')
    torch.set_num_threads(1);torch.manual_seed(seed)
    out=Path(out);parents=Path(parents)
    parent=parents/f'anchored_seed{seed}.pt';model,meta=load_head(parent)
    if not hasattr(model,'anchor'):raise ValueError('Mean-anchored parent required')
    data=torch.load(data_path,weights_only=True,map_location='cpu');data.pop('test',None)
    if meta['teacher_sha256']!=data['metadata']['teacher_sha256']:raise ValueError('Teacher identity mismatch')
    if meta['identity']['data_sha256']!=digest(data_path):raise ValueError('Parent data mismatch')
    v=data['metadata']['variables']
    if list(data['metadata']['horizons'])!=[1,4,16,64]:raise ValueError('This study uses four fixed horizons')
    anchor=torch.load(parents/'anchor.pt',weights_only=True,map_location='cpu')
    if any(not torch.equal(value,model.anchor.state_dict()[key]) for key,value in anchor['weights'].items()):
        raise ValueError('Crossfit means do not belong to the parent anchor')
    oof=anchor['oof_mean']
    if oof.shape!=data['train']['y'].flatten(1).shape:raise ValueError('Crossfit row mismatch')
    mask,_=split_validation(data['validation']['group']);valid=subset(data['validation'],mask)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+9262601)
    identity={'format':'conditional_study_v1','kind':kind,'seed':seed,'steps':steps,'batch':64,
              'data_sha256':digest(data_path),'parent_sha256':digest(parent),'anchor_sha256':digest(parents/'anchor.pt'),
              'source_sha256':source_hash(),'torch':str(torch.__version__),'device':'cpu'}
    start=0;best=float('inf');selected=0;logs=[];batch_hash=bytes(32);elapsed=0.
    # Never overwrite an existing study. All checkpoints include the exact saved
    # best model so a fresh-process resume restores checkpoint selection too.
    if resume:
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu')
        if saved['identity']!=identity:raise ValueError('Resume identity changed')
        model.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer'])
        gen.set_state(saved['generator']);torch.set_rng_state(saved['rng'])
        start=saved['step'];best=saved['best'];selected=saved['selected'];logs=saved['logs'];elapsed=saved['elapsed']
        batch_hash=bytes.fromhex(saved['batch_sha256']);best_model=saved['best_weights']
    else:
        out.mkdir(parents=True,exist_ok=False)
        score=measure(model,valid);best=score['selection_score'];logs=[{'step':0,**score}]
        best_model=copy.deepcopy(model.state_dict())
    t0=time.perf_counter();finish=steps if stop_at is None else stop_at
    for step in range(start+1,finish+1):
        model.train();idx=torch.randint(len(data['train']['x']),(64,),generator=gen)
        batch_hash=hashlib.sha256(batch_hash+idx.numpy().tobytes()).digest()
        x=data['train']['x'][idx];y=data['train']['y'][idx].flatten(1).double()
        f=GaussianMixture.from_landscape(model.landscape(x,oof[idx]))
        if kind=='joint':loss=f.energy(y).mean()/model.d
        else:
            n,m=objective_terms(f,y,v)
            loss=.5*n[:,0].mean()+.5*n[:,1:].mean()
            if kind=='conditional_mse':loss=loss+.25*m[:,1:].mean()
        loss=loss+1e-4*torch.tril(model.raw_correlation,-1).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        for g in optimizer.param_groups:g['lr']=2e-4*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.step()
        # Mid-run interruptions do not add a new validation/selection opportunity.
        if step%50==0 or step==steps:
            model.eval();score=measure(model,valid);logs.append({'step':step,'train_loss':float(loss.detach()),**score})
            if score['selection_score']<best:
                best=score['selection_score'];selected=step;best_model=copy.deepcopy(model.state_dict())
            print(kind,seed,step,score,flush=True)
        if step%50==0 or step==finish:
            atomic_save({'identity':identity,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
              'generator':gen.get_state(),'rng':torch.get_rng_state(),'step':step,'best':best,'selected':selected,
              'best_weights':best_model,'logs':logs,'batch_sha256':batch_hash.hex(),
              'elapsed':elapsed+time.perf_counter()-t0},out/'last.pt')
    result=copy.deepcopy(meta);result['weights']=best_model;result['identity']=identity
    result['conditional_study']={'selection_score':best,'selected_step':selected,'teacher_unchanged':True,
                                'online_default':'prior unless a NEW matching gate is supplied; raw conditioning is diagnostic'}
    atomic_save(result,out/'best.pt')
    record={**identity,'selected_step':selected,'steps_completed':finish,'selection_score':best,
            'batch_sha256':batch_hash.hex(),'seconds':elapsed+time.perf_counter()-t0,'history':logs,
            'validation_groups':valid['group'].unique().tolist()}
    (out/'training.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--parents',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--steps',type=int,default=350);p.add_argument('--resume',action='store_true');p.add_argument('--stop-at',type=int)
    a=p.parse_args();train(a.data,a.parents,a.out,a.kind,a.seed,a.steps,a.resume,a.stop_at)
if __name__=='__main__':main()
