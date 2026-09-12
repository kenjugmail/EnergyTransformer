#!/usr/bin/env python3
"""Evaluate only after final fits. Retain independent episodes and per-query errors."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,random,hashlib
import numpy as np
import pandas as pd
import torch
from energy_transformer.geometric_memory import EditMemory
from energy_transformer.recall_attention import SoftmaxRecall
from energy_transformer.recall_data import episodes,EpisodeBatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v18'
CONDITIONS={'primary':{},'uncorrelated':{'correlation':0.0},'near_collinear':{'correlation':.95},'many_edits':{'edits':32},'at_capacity':{'nkeys':12},'over_capacity':{'nkeys':20},'long_hold':{'gap':1024}}

def load(path):
    c=torch.load(path,map_location='cpu',weights_only=True)
    m=SoftmaxRecall(c['dimension'],c['value_dim'],c['prior']) if c['kind']=='softmax' else EditMemory(c['kind'],c['dimension'],c['value_dim'],c['prior'])
    m.load_state_dict(c['weights']);m.eval();return m,c

def sliced(b,start,end):return EpisodeBatch(**{k:v[start:end] for k,v in vars(b).items()})

@torch.inference_mode()
def table_reference(b):
    answer=torch.zeros_like(b.targets)
    for i in range(len(answer)):
        memory={}
        for t in range(b.keys.shape[1]):
            op=int(b.operations[i,t]);k=b.keys[i,t].numpy().tobytes()
            if op==0:memory[k]=b.values[i,t].clone()
            if op==1:memory[k]=torch.zeros_like(b.values[i,t])
        for j in range(b.queries.shape[1]):answer[i,j]=memory.get(b.queries[i,j].numpy().tobytes(),torch.zeros_like(answer[i,j]))
    return answer

@torch.inference_mode()
def main():
    torch.set_num_threads(1);(OUT/'test_inputs').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    paths=sorted((ROOT/'checkpoints/v18').glob('*.pt'));paths=[p for p in paths if not p.name.startswith(('pilot_','projected_'))]
    assert len(paths)==21, f'Expected 21 completed final fits, got {len(paths)}'
    manifest={};tasks={}
    for i,(name,args) in enumerate(CONDITIONS.items()):
        b=episodes(18900000+i,batch=192,**args);tasks[name]=b
        file=OUT/f'test_inputs/{name}.npz';np.savez_compressed(file,**{k:v.numpy() for k,v in vars(b).items()})
        manifest[name]={'seed':18900000+i,'episodes':192,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),**args}
        reference=table_reference(b);assert torch.equal(reference,b.targets)
    (OUT/'test_manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    for path in paths:
        model,c=load(path);name=path.stem
        for condition,b in tasks.items():
            outputs=[];min_eig=[]
            for start in range(0,192,24):
                part=sliced(b,start,start+24)
                out,state,_=model(part.keys,part.values,part.operations,part.queries)
                if not torch.isfinite(out).all():raise RuntimeError(f'Nonfinite {name} {condition}')
                outputs.append(out)
                if c['kind'] in ('precision','euclidean','preconditioned_delta'):min_eig.append(float(torch.linalg.eigvalsh(state.precision).min()))
            prediction=torch.cat(outputs);mse=(prediction-b.targets).square().mean(-1)
            # Hindsight lower bound for this fixed linear query feature space.
            q=model.encode_key(b.queries).double()
            coef=torch.linalg.lstsq(q,b.targets.double(),driver='gelsd').solution
            oracle=((q@coef-b.targets.double())**2).mean((-1,-2))
            np.savez_compressed(OUT/f'predictions/{name}_{condition}.npz',output=prediction.numpy(),mse=mse.numpy(),changed=b.changed.numpy(),linear_oracle=oracle.numpy())
            for ep in range(192):
                changed=b.changed[ep];unchanged=~changed
                records.append({'model':c['kind'],'objective':c['objective'],'seed':c['seed'],'condition':condition,'episode':ep,
                                'mse':float(mse[ep].mean()),'changed_mse':float(mse[ep][changed].mean()) if changed.any() else np.nan,
                                'untouched_mse':float(mse[ep][unchanged].mean()),'fraction_rmse_below_point1':float((mse[ep]<.01).float().mean()),
                                'linear_oracle_mse':float(oracle[ep]),'min_precision_eigenvalue':min(min_eig) if min_eig else np.nan})
        print(name,'evaluated',flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'raw.csv',index=False)
    groups=df.groupby(['model','objective','condition'])
    summary=groups[['mse','changed_mse','untouched_mse','fraction_rmse_below_point1','linear_oracle_mse']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    pairs=[];rng=np.random.default_rng(18970000)
    for cond in CONDITIONS:
        base=df[(df.model=='precision')&(df.objective=='delayed')&(df.condition==cond)].pivot(index='seed',columns='episode',values='mse').values
        for other,obj in [('euclidean','delayed'),('diagonal','delayed'),('delta','delayed'),('preconditioned_delta','delayed'),('softmax','delayed'),('precision','immediate')]:
            b=df[(df.model==other)&(df.objective==obj)&(df.condition==cond)].pivot(index='seed',columns='episode',values='mse').values
            delta=(base-b).mean(0);idx=rng.integers(192,size=(5000,192));boot=delta[idx].mean(1)
            pairs.append({'condition':cond,'a':'precision_delayed','b':other+'_'+obj,'difference_mse':delta.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975)})
    pd.DataFrame(pairs).to_csv(OUT/'paired.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':main()
