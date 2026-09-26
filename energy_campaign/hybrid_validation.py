"""Actual-shape validation for the selected hybrid study; no full-pilot authorization."""
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from energy_real.model import LM
from energy_real.runtime import atomic_json
from .spec import config,source_identity
from .data import Documents


def relative(a,b):
    # Accumulate diagnostics in FP64; million-element FP32 reductions can distort
    # cosines/norms on this CPU runtime (including reported cosines above one).
    a,b=a.double(),b.double()
    return float((a-b).norm()/(b.norm()+1e-12))


def gradients(model,x,y,amp=False):
    model.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=amp):
        logits,state=model(x);loss=torch.nn.functional.cross_entropy(logits.float().flatten(0,1),y.flatten())
    loss.backward();g=torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
    if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite gradients')
    for i,s in enumerate(state):
        if model.config.mixer_kind(i)=='energy' and (s[0].dtype!=torch.float32 or s[1].dtype!=torch.float32 or not all(torch.isfinite(t).all() for t in s)):
            raise AssertionError('Energy state must remain finite FP32')
    return logits.detach(),float(loss.detach()),g.detach()


def check_model(model,x,y,streaming=False):
    logits,loss,g=gradients(model,x,y);b,bl,bg=gradients(model,x,y,True)
    row={'loss_abs':abs(bl-loss),'logits_relative':relative(b,logits),'gradient_relative':relative(bg,g),
         'gradient_cosine':float(torch.nn.functional.cosine_similarity(g.double(),bg.double(),dim=0)),'input_stride':list(x.stride())}
    row['passed']=row['loss_abs']<.15 and row['logits_relative']<.08 and row['gradient_relative']<.3 and row['gradient_cosine']>.95
    if streaming:
        with torch.no_grad():
            state=None;parts=[]
            for start in range(0,x.shape[1],17):
                z,state=model(x[:,start:start+17],state);parts.append(z)
            row['chunk_max_abs']=float((torch.cat(parts,1)-logits).abs().max())
            state=None;parts=[]
            for start in range(x.shape[1]):
                z,state=model(x[:,start:start+1],state);parts.append(z)
            row['token_max_abs']=float((torch.cat(parts,1)-logits).abs().max())
            split=x.shape[1]//2;altered=x.clone();altered[:,split:]=0
            row['future_max_abs']=float((model(altered)[0][:,:split]-logits[:,:split]).abs().max())
            reset=torch.zeros_like(x,dtype=torch.bool);reset[:,split]=True
            row['reset_max_abs']=float((model(x,reset=reset)[0][:,split:]-model(x[:,split:])[0]).abs().max())
        cp=LM(replace(model.config,activation_checkpointing=True)).to(x.device);cp.load_state_dict(model.state_dict())
        _,_,cg=gradients(cp,x,y);row['checkpoint_gradient_relative']=relative(cg,g)
        row['passed'] &= (max(row['chunk_max_abs'],row['token_max_abs'],row['reset_max_abs'])<.002
                          and row['future_max_abs']<1e-6 and row['checkpoint_gradient_relative']<2e-5)
        del cp,cg
    return row


def run(data,out,device='cpu',energy_projection_fp32=False):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);corpus=Documents(data);rows=[]
    torch.set_num_threads(1)
    for size in (0,1):
        for seed in (404,505,606):
            for family in ('hybrid','softmax','local_softmax'):
                torch.manual_seed(seed);cfg=replace(config(family,size,chunk=16),energy_projection_fp32=energy_projection_fp32 and family=='hybrid');model=LM(cfg).to(device)
                x,y=corpus.batch(np.random.default_rng(seed),1,512,device)
                row={'family':family,'size':size,'seed':seed,'config':asdict(cfg),
                     **check_model(model,x,y,streaming=family=='hybrid' and seed==404)}
                rows.append(row);print({k:v for k,v in row.items() if k!='config'},flush=True)
                if not row['passed']:
                    torch.save({'config':asdict(cfg),'weights':{k:v.cpu() for k,v in model.state_dict().items()},'x':x.cpu(),'y':y.cpu()},out/f'{family}_{size}_{seed}_replay.pt')
                atomic_json({'source':source_identity(),'device':device,'rows':rows,'status':'running','gpu_campaign_authorized':False},out/'results.json')
                del model,x,y
                if str(device).startswith('cuda'):torch.cuda.empty_cache()
    receipt={'source':source_identity(),'device':device,'rows':rows,'status':'passed' if all(r['passed'] for r in rows) else 'failed','gpu_campaign_authorized':False}
    atomic_json(receipt,out/'results.json');return receipt
