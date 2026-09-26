"""Matched trainable-budget decoder adapters on frozen causal teacher features.

CPU head-only study. Cached test/secondary labels are removed before fitting.
Step zero is eligible; all selected hashes are frozen before final evaluation.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, time
from dataclasses import asdict
import torch
from torch.nn import functional as F
from .decoder_bridge import DecoderAdapter, DecoderConfig, KINDS
from .model import EnergyLandscape
from .experiment import load_head, atomic_save
from .probe import checkpoint_hash

@torch.no_grad()
def forecast_cache(head, part, batch=512):
    rows={k:[] for k in ('mean','means','scales','logits')}; lower=None
    for start in range(0,len(part['h']),batch):
        x=part['x'][start:start+batch];fun=head(x)
        rows['mean'].append(head.anchor(x).detach())
        for k in ('means','scales','logits'):rows[k].append(getattr(fun,k).detach())
        lower=fun.correlation_cholesky.detach()
    return {**{k:torch.cat(v) for k,v in rows.items()},'lower':lower}

def select(cache, idx):
    if cache is None:return None,None
    return EnergyLandscape(cache['logits'][idx],cache['means'][idx],cache['scales'][idx],cache['lower']),cache['mean'][idx]

@torch.no_grad()
def feature_normalization(adapter, part, cache):
    if adapter.config.kind=='hidden':return
    total=[]
    for start in range(0,len(part['h']),512):
        sl=slice(start,start+512);fun,mean=select(cache,sl)
        total.append(adapter.raw_features(part['h'][sl],part['x'][sl],fun,mean))
    feature=torch.cat(total)
    adapter.feature_center.copy_(feature.mean(0))
    adapter.feature_scale.copy_(feature.std(0).clamp_min(.05))

@torch.no_grad()
def losses(adapter, part, weight, cache=None, *, batch=512, erase_side=False, permutation=None):
    rows=[]
    for start in range(0,len(part['h']),batch):
        idx=torch.arange(start,min(start+batch,len(part['h'])))
        side_idx=idx if permutation is None else permutation[idx]
        fun,mean=select(cache,side_idx)
        if adapter is None:logits=F.linear(part['h'][idx],weight)
        else:logits=adapter(part['h'][idx],part['x'][side_idx],weight,fun,mean,erase_side=erase_side)
        rows.append(F.cross_entropy(logits,part['y'][idx],reduction='none')/math.log(2))
    return torch.cat(rows)

def train_one(data, data_hash, cache, head_hash, out, kind, seed=11, steps=600, batch=128,
              peak_lr=.001, stop_after=None, resume=False):
    if kind not in KINDS or min(steps,batch)<1 or not math.isfinite(peak_lr) or peak_lr<=0:raise ValueError('Invalid training configuration')
    if stop_after is not None and not 0<=stop_after<=steps:raise ValueError('Invalid stop_after')
    out=Path(out);torch.set_num_threads(1);torch.manual_seed(seed)
    meta=data['metadata'];cfg=DecoderConfig(kind=kind,width=meta['width'],history=meta['history'],variables=meta['variables'],horizons=tuple(meta['horizons']))
    model=DecoderAdapter(cfg);feature_normalization(model,data['train'],cache['train'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=peak_lr,weight_decay=.01,foreach=False)
    generator=torch.Generator().manual_seed(19000+seed)
    identity={'data_sha256':data_hash,'head_sha256':head_hash,'teacher_sha256':meta['teacher_sha256'],
        'config':asdict(cfg),'seed':seed,'steps':steps,'batch':batch,'peak_lr':peak_lr,'torch':str(torch.__version__),
        'sources':{p.name:checkpoint_hash(p) for p in [Path(__file__),Path(__file__).with_name('decoder_bridge.py')]}}
    start=0;history=[];sample_hash=bytes(32);best_step=0
    def export(step):
        return {'format':'decoder_bridge_v1','config':asdict(cfg),'weights':model.state_dict(),'teacher_sha256':meta['teacher_sha256'],
            'head_sha256':head_hash if kind in ('mean','moments','energy_read') else None,
            'normalization':meta['normalization'],'identity':identity,'selected_step':step}
    if resume:
        old=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if old['identity']!=identity:raise ValueError('Changed resume identity')
        model.load_state_dict(old['weights']);optimizer.load_state_dict(old['optimizer'])
        generator.set_state(old['generator']);torch.set_rng_state(old['rng'])
        start=old['step'];best=old['best'];best_step=old['best_step'];history=old['history'];sample_hash=bytes.fromhex(old['sample_hash'])
        best_payload=old['best_payload'];atomic_save(best_payload,out/'best.pt')
    else:
        out.mkdir(parents=True,exist_ok=False)
        initial=float(losses(model,data['validation'],data['weight'],cache['validation']).mean())
        best=initial;history=[{'step':0,'validation_bpb':best}]
        best_payload=export(0);best_payload['weights']={k:v.clone() for k,v in best_payload['weights'].items()}
        atomic_save(best_payload,out/'best.pt');atomic_save(best_payload,out/'initial.pt')
    t0=time.perf_counter();finish=steps if stop_after is None else stop_after
    for step in range(start+1,finish+1):
        model.train();idx=torch.randint(len(data['train']['h']),(batch,),generator=generator)
        sample_hash=hashlib.sha256(sample_hash+idx.numpy().tobytes()).digest()
        fun,mean=select(cache['train'],idx)
        optimizer.zero_grad(set_to_none=True)
        logits=model(data['train']['h'][idx],data['train']['x'][idx],data['weight'],fun,mean)
        loss=F.cross_entropy(logits,data['train']['y'][idx])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        for group in optimizer.param_groups:group['lr']=peak_lr*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.step()
        if step%100==0 or step==steps:
            model.eval();valid=float(losses(model,data['validation'],data['weight'],cache['validation']).mean())
            history.append({'step':step,'validation_bpb':valid,'train_nats':float(loss.detach()),'gradient_norm':float(grad)})
            if valid<best:
                best=valid;best_step=step;best_payload=export(step)
                best_payload['weights']={k:v.detach().clone() for k,v in best_payload['weights'].items()}
                atomic_save(best_payload,out/'best.pt')
            print(kind,seed,history[-1],flush=True)
        if step%100==0 or step==finish:
            atomic_save({'identity':identity,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),
                'generator':generator.get_state(),'rng':torch.get_rng_state(),'step':step,'best':best,'best_step':best_step,
                'history':history,'sample_hash':sample_hash.hex(),'best_payload':best_payload},out/'last.pt')
    record={'kind':kind,'seed':seed,'steps_completed':finish,'selected_step':best_step,'validation_bpb':best,
        'parameters':sum(p.numel() for p in model.parameters()),'sample_hash':sample_hash.hex(),
        'presented_targets':finish*batch,'seconds_this_invocation':time.perf_counter()-t0,'history':history,'identity':identity}
    (out/'training.json').write_text(json.dumps(record,indent=2));return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--heads');p.add_argument('--out',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);p.add_argument('--kinds',nargs='+',choices=KINDS,default=list(KINDS))
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=128);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    torch.set_num_threads(1);data=torch.load(a.data,weights_only=True,map_location='cpu')
    data.pop('test',None);data.pop('secondary',None)
    data_hash=checkpoint_hash(a.data);out=Path(a.out)
    if out.exists() and not a.resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);records=[]
    for seed in a.seeds:
        hp = Path(a.heads)/f'anchored_seed{seed}.pt' if a.heads else None
        if hp is None:
            if any(k not in ('hidden','history') for k in a.kinds):
                raise ValueError('Forecast arms require --heads; hidden/history do not')
            cache={'train':None,'validation':None}
        else:
            head,hm=load_head(hp)
            if hm['teacher_sha256']!=data['metadata']['teacher_sha256']:raise ValueError('Teacher mismatch')
            for par in head.parameters():par.requires_grad_(False)
            cache={name:forecast_cache(head,data[name]) for name in ('train','validation')}
        for kind in a.kinds:
            rd=out/f'{kind}_seed{seed}'
            if a.resume and (rd/'training.json').exists() and json.loads((rd/'training.json').read_text())['steps_completed']==a.steps:
                records.append(json.loads((rd/'training.json').read_text()));continue
            records.append(train_one(data,data_hash,cache,checkpoint_hash(hp) if hp else None,rd,kind,seed,a.steps,a.batch,stop_after=a.stop_after,resume=a.resume and (rd/'last.pt').exists()))
    if all(r['steps_completed']==a.steps for r in records):
        files={str((out/f"{r['kind']}_seed{r['seed']}"/'best.pt').relative_to(out)):checkpoint_hash(out/f"{r['kind']}_seed{r['seed']}"/'best.pt') for r in records}
        means={kind:sum(r['validation_bpb'] for r in records if r['kind']==kind)/len([r for r in records if r['kind']==kind]) for kind in a.kinds}
        frozen={'format':'decoder_bridge_selection_v1','selection':'validation before final test','data_sha256':data_hash,'files':files,'validation_means':means,'validation_winner':min(means,key=means.get)}
        (out/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
    print('COMPLETED',len(records),flush=True)
if __name__=='__main__':main()
