"""Separately supervised symbolic recall learning curves, never a language score."""
from pathlib import Path
import numpy as np
import torch
from energy_real.model import LM,Config
from energy_real.runtime import atomic_json
from .spec import FAMILIES,SEEDS


def batch(rng,n,length,device,task='associative'):
    if length<20:raise ValueError('Recall sequence too short')
    x=rng.integers(64,128,(n,length),dtype=np.int64);answers=[]
    for row in x:
        keys=rng.choice(np.arange(1,17),4,replace=False);values=rng.integers(32,64,4)
        if task=='copy':keys=keys[:1];values=values[:1]
        for i,(key,value) in enumerate(zip(keys,values)):row[3*i:3*i+3]=[200,key,value]
        chosen=int(rng.integers(len(keys)));row[-2:]=[203,keys[chosen]];answers.append(values[chosen])
    return torch.tensor(x,device=device),torch.tensor(answers,device=device)


def run(root,device='cuda',steps=100):
    root=Path(root);rows=[]
    for family in FAMILIES:
        for seed in SEEDS:
            path=root/f'matched_data/{family}_1_{seed}/best.pt'
            if not path.exists():continue
            saved=torch.load(path,map_location='cpu',weights_only=True)
            for task in ('copy','associative'):
                torch.manual_seed(seed);model=LM(Config(**saved['config'])).to(device);model.load_state_dict(saved['weights'])
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95));rng=np.random.default_rng(seed)
                for step in range(steps+1):
                    if step in (0,25,50,100):
                        model.eval()
                        with torch.no_grad():
                            for length in (128,512,2048):
                                x,y=batch(np.random.default_rng(8300+length),32,length,device,task)
                                score=float((model(x)[0][:,-1].argmax(-1)==y).float().mean())
                                rows.append({'family':family,'seed':seed,'task':task,'step':step,'context':length,'accuracy':score,'examples':32})
                        atomic_json({'status':'partial','rows':rows,'scope':'supervised synthetic adaptation; not general reasoning'},root/'recall.json')
                    if step==steps:break
                    model.train();x,y=batch(rng,8,128,device,task);optimizer.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(model(x)[0][:,-1],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                del model,optimizer
    atomic_json({'status':'completed','rows':rows,'scope':'supervised synthetic adaptation; not general reasoning'},root/'recall.json')
    return rows
