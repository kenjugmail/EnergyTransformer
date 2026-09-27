"""CPU-only paired candidate-refinement study; no inference-time target leakage.

Static teacher and current-memory features. CE is averaged over the four trained
iterates for recurrent/energy arms. This is a different cost from one MLP read.
All controls have approximately the same trainable parameter budget.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,math,time
import torch
from torch.nn import functional as F
from .refine_model import RefinementConfig,RefinementAdapter,KINDS,load_refinement
from .probe import checkpoint_hash
from .experiment import atomic_save

@torch.no_grad()
def per_example(model,part,weight,*,steps=4,batch=256,memory_mode='normal',policy=None):
    values=[];counts=[]
    for start in range(0,len(part['y']),batch):
        h=part['h'][start:start+batch];m=part['m'][start:start+batch]
        if memory_mode=='zero':m=model.memory_center.expand_as(m)
        elif memory_mode=='shuffle':
            idx=part['permutation'][start:start+batch];m=part['m'][idx]
        if policy is None:logits=model(h,m,weight,steps=steps)
        else:
            info=model.solve(h,m,weight,max_steps=16,tolerance=policy)
            logits=info['logits'];counts.append(info['steps'])
        values.append(F.cross_entropy(logits,part['y'][start:start+batch],reduction='none')/math.log(2))
    result=torch.cat(values)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite validation')
    return result,torch.cat(counts) if counts else None


def fit(data,path,out,kind,seed=11,*,steps=600,batch=128,resume=False,stop_after=None):
    if type(steps)is not int or steps<1 or type(batch)is not int or batch<1:raise ValueError('Positive schedule required')
    if stop_after is not None and not 0<=stop_after<=steps:raise ValueError('Invalid absolute stop')
    torch.set_num_threads(1);torch.manual_seed(seed)
    meta=data['metadata'];cfg=RefinementConfig(kind=kind,width=meta['width'],memory_width=meta['memory_width'],latent=meta['width'] if kind=='energy_full' else 32)
    model=RefinementAdapter(cfg)
    model.memory_center.copy_(meta['memory_center']);model.memory_scale.copy_(meta['memory_scale'])
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,foreach=False)
    rng=torch.Generator().manual_seed(seed+27000)
    identity={'kind':kind,'seed':seed,'steps':steps,'batch':batch,'lr':.001,'data_sha256':checkpoint_hash(path),
        'teacher_sha256':meta['teacher_sha256'],'config':asdict(cfg),'torch':str(torch.__version__),
        'sources':{p.name:checkpoint_hash(p) for p in (Path(__file__),Path(__file__).with_name('refine_model.py'))}}
    def export(step):
        return {'format':'energy_refinement_v1','config':asdict(cfg),'weights':{k:v.detach().clone() for k,v in model.state_dict().items()},
            'teacher_sha256':meta['teacher_sha256'],'identity':identity,'step':step}
    out=Path(out);hist=[];count=0;sample=bytes(32);best_step=0;time0=time.perf_counter()
    if resume:
        old=torch.load(out/'last.pt',weights_only=True,map_location='cpu')
        if old['identity']!=identity:raise ValueError('Changed run identity')
        count=old['step']
        if stop_after is not None and stop_after<count:raise ValueError('Backwards stop')
        model.load_state_dict(old['weights']);optim.load_state_dict(old['optimizer']);rng.set_state(old['generator'])
        torch.set_rng_state(old['rng']);hist=old['history'];sample=bytes.fromhex(old['sample_hash'])
        best=old['best'];best_step=old['best_step'];selected=old['selected']
    else:
        out.mkdir(parents=True,exist_ok=False)
        best=float(per_example(model,data['validation'],data['weight'])[0].mean())
        hist=[{'step':0,'validation_bpb':best}];selected=export(0)
    def save():
        atomic_save({'identity':identity,'step':count,'weights':model.state_dict(),'optimizer':optim.state_dict(),
            'generator':rng.get_state(),'rng':torch.get_rng_state(),'sample_hash':sample.hex(),
            'best':best,'best_step':best_step,'selected':selected,'history':hist},out/'last.pt')
        atomic_save(selected,out/'best.pt')
    if not resume:save()
    end=steps if stop_after is None else stop_after
    for count in range(count+1,end+1):
        idx=torch.randint(len(data['train']['y']),(batch,),generator=rng)
        sample=hashlib.sha256(sample+idx.numpy().tobytes()).digest()
        model.train();optim.zero_grad(set_to_none=True)
        trajectory=model.trajectory(data['train']['h'][idx],data['train']['m'][idx],data['weight'])
        loss=torch.stack([F.cross_entropy(logits,data['train']['y'][idx]) for logits in trajectory]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        for group in optim.param_groups:group['lr']=.001*(.2+.8*.5*(1+math.cos(math.pi*count/steps)))
        optim.step()
        if count%100==0 or count==steps:
            model.eval();valid=float(per_example(model,data['validation'],data['weight'])[0].mean())
            hist.append({'step':count,'validation_bpb':valid,'train_nats':float(loss.detach()),'grad_norm':float(gn)})
            if valid<best:best=valid;best_step=count;selected=export(count)
            print(kind,seed,hist[-1],flush=True)
        if count%100==0 or count==end:save()
    count=end;save()
    record={'identity':identity,'kind':kind,'seed':seed,'status':'completed' if count==steps else 'partial',
        'step':count,'selected_step':best_step,'validation_bpb':best,'sample_hash':sample.hex(),
        'parameters':sum(p.numel() for p in model.parameters()),'presented_targets':count*batch,
        'seconds_this_invocation':time.perf_counter()-time0,'history':hist}
    (out/'training.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kinds',nargs='+',choices=KINDS,default=list(KINDS));p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=128);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data=torch.load(args.data,weights_only=True,map_location='cpu');data.pop('test',None);data.pop('secondary',None)
    records=[]
    for seed in args.seeds:
        for kind in args.kinds:
            records.append(fit(data,args.data,out/f'{kind}_seed{seed}',kind,seed,steps=args.steps,batch=args.batch))
    # Freeze every selected checkpoint before reading final labels.
    files={f"{r['kind']}_seed{r['seed']}/best.pt":checkpoint_hash(out/f"{r['kind']}_seed{r['seed']}/best.pt") for r in records}
    means={k:sum(r['validation_bpb'] for r in records if r['kind']==k)/len(args.seeds) for k in args.kinds}
    frozen={'format':'refinement_selection_v1','files':files,'data_sha256':checkpoint_hash(args.data),'seeds':args.seeds,'kinds':args.kinds,
        'validation_means':means,'validation_winner':min(means,key=means.get),'default_depth':4,
        'selection':'fixed four-step validation checkpoints; separate validation-only inference-depth audit follows'}
    (out/'FROZEN.json').write_text(json.dumps(frozen,indent=2));(out/'training.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
