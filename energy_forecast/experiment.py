"""Small reproducible head-only study. Does not train or replace the language model.

Document splits precede all window extraction. Targets are frozen teacher sketches.
Validation NLL selects checkpoints; final test is scored only by evaluate.py.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,time,os
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from .model import ForecastConfig,ForecastEnergy

KINDS={'diagonal':dict(components=1,correlated=False,spectral=False),
       'joint':dict(components=1,correlated=True,spectral=False),
       'mixture':dict(components=3,correlated=True,spectral=False),
       'spectral':dict(components=3,correlated=True,spectral=True)}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_save(data,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

@torch.no_grad()
def measure(model,part,batch=128):
    total=0.;mse=0.
    model.eval()
    for start in range(0,len(part['x']),batch):
        x,y=part['x'][start:start+batch],part['y'][start:start+batch]
        fun=model(x);total+=float(fun.energy(y.flatten(1)).sum());mse+=float((fun.mean-y.flatten(1)).square().sum())
    return {'nll_per_coordinate':total/(len(part['x'])*model.d),'mse':mse/(len(part['x'])*model.d)}

def train(data_path,out,kind,seed=11,steps=600,resume=False,device='cpu',stop_at=None):
    out=Path(out);torch.set_num_threads(1);torch.manual_seed(seed)
    data=torch.load(data_path,map_location='cpu',weights_only=True);meta=data['metadata']
    config=ForecastConfig(history=meta['history'],variables=meta['variables'],horizons=tuple(meta['horizons']),**KINDS[kind])
    model=ForecastEnergy(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    gen=torch.Generator().manual_seed(seed+20260926)
    identity={'data_sha256':digest(data_path),'kind':kind,'seed':seed,'steps':steps,'config':asdict(config),'torch':str(torch.__version__),'device':device}
    log=[];best=float('inf');best_step=0;start=0;batch_hash=bytes(32)
    if resume:
        old=torch.load(out/'last.pt',weights_only=True,map_location=device)
        if old['identity']!=identity:raise ValueError('Resume identities changed')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);gen.set_state(old['generator'].cpu())
        torch.set_rng_state(old['rng'].cpu());start=old['step'];best=old['best'];best_step=old['best_step'];log=old['log'];batch_hash=bytes.fromhex(old['batch_sha256'])
    else:
        out.mkdir(parents=True,exist_ok=False)
    for part in ('train','validation'):
        for name in ('x','y'):data[part][name]=data[part][name].to(device)
    t0=time.perf_counter();finish=min(steps,stop_at) if stop_at else steps
    for step in range(start+1,finish+1):
        model.train();idx=torch.randint(len(data['train']['x']),(64,),generator=gen)
        batch_hash=hashlib.sha256(batch_hash+idx.numpy().tobytes()).digest()
        idx=idx.to(device);x,y=data['train']['x'][idx],data['train']['y'][idx]
        optimizer.zero_grad(set_to_none=True)
        loss=model.loss(x,y)
        # Small regularizer on only the nontrivial covariance terms.
        if config.correlated:loss=loss+1e-4*torch.tril(model.raw_correlation,-1).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups:
            group['lr']=.001*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.step()
        if step%100==0 or step==finish:
            score=measure(model,data['validation']);log.append({'step':step,'train_loss':float(loss.detach()),**score})
            if score['nll_per_coordinate']<best:
                best=score['nll_per_coordinate'];best_step=step
                atomic_save({'format':'trajectory_energy_v1','config':asdict(config),'weights':model.state_dict(),
                    'normalization':{k:meta[k] for k in ('center','scale')},'identity':identity,'selected_step':step,
                    'teacher_sha256':meta.get('teacher_sha256'),'summary_only':True},out/'best.pt')
            atomic_save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'generator':gen.get_state(),
                'rng':torch.get_rng_state(),'step':step,'best':best,'best_step':best_step,'log':log,'batch_sha256':batch_hash.hex()},out/'last.pt')
            print(kind,seed,step,score,flush=True)
    record={'kind':kind,'seed':seed,'steps_completed':finish,'selected_step':best_step,'validation_nll':best,
            'parameters':sum(p.numel() for p in model.parameters()),'seconds_this_invocation':time.perf_counter()-t0,
            'batch_sha256':batch_hash.hex(),'identity':identity,'history':log}
    (out/'training.json').write_text(json.dumps(record,indent=2));return record

def load_head(path,device='cpu'):
    d=torch.load(path,weights_only=True,map_location=device)
    if d.get('format')=='calibrated_trajectory_v2':
        from .calibrated import load_calibrated
        return load_calibrated(path,device)
    if d.get('format')!='trajectory_energy_v1':raise ValueError('Unknown forecast checkpoint')
    c=dict(d['config']);c['horizons']=tuple(c['horizons'])
    m=ForecastEnergy(ForecastConfig(**c)).to(device);m.load_state_dict(d['weights']);m.eval();return m,d

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cpu');a=p.parse_args()
    train(a.data,a.out,a.kind,a.seed,a.steps,a.resume,a.device)
if __name__=='__main__':main()
