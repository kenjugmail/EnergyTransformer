"""No-training replay of frozen forecasts as nearer observations actually arrive.

At cutoff s only targets with horizon <= s are available to the update. Metrics
score remaining, unobserved coordinates only. The original mean/mixture weights
were fitted previously; no final-test tuning or model promotion happens here.
The corpus and test domain were inspected in previous iterations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .conditioning import GaussianMixture
from .experiment import load_head, digest
from .repair_evaluate import RidgeGaussian


def block_interval(delta, groups, draws=3000):
    unique=torch.unique(groups)
    counts=np.array([(groups==g).sum().item() for g in unique])
    sums=np.array([delta[groups==g].sum().item() for g in unique])
    rng=np.random.default_rng(2692603)
    idx=rng.integers(len(unique),size=(draws,len(unique)))
    samples=sums[idx].sum(-1)/counts[idx].sum(-1)
    return {'difference':float(delta.mean()),'ci_low':float(np.quantile(samples,.025)),
            'ci_high':float(np.quantile(samples,.975)),'groups':len(unique)}


@torch.no_grad()
def run(data_path, fits, out):
    torch.set_num_threads(1)
    data_path,fits,out=Path(data_path),Path(fits),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    paths=[fits/f'anchored_seed{s}.pt' for s in (11,22,33)]
    protocol={'experiment':'fixed_origin_observed_coordinate_conditioning_v1',
       'data_sha256':digest(data_path),'heads':{p.name:digest(p) for p in paths},
       'gaussian_sha256':digest(fits/'ridge_gaussian.pt'),'cutoffs':[0,1,4,16],
       'states':'Original forecast at t; receive actual sketches at t+1/t+4/t+16; targets retain t+4/t+16/t+64 absolute times',
       'no_new_neural_fits':True,'no_test_selection':True,'reused_test_domain':True,
       'controls':['unchanged_prior','conditioned_correlated_gaussian','shuffled_near_observations'],
       'metrics':['standardized_mse','nll_per_remaining_coordinate','marginal_coverage90'],
       'dtype':'float64 conditioning; inherited heads float32','torch':str(torch.__version__)}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    data=torch.load(data_path,weights_only=True,map_location='cpu')
    x,y,groups=data['test']['x'],data['test']['y'],data['test']['group']
    horizons=data['metadata']['horizons'];v=data['metadata']['variables']
    if v!=8 or list(horizons)!=[1,4,16,64]:raise ValueError('This evaluation protocol requires 8 variables and horizons 1/4/16/64')
    heads=[(f'mixture_seed{s}',load_head(p)[0]) for s,p in zip((11,22,33),paths)]
    heads.append(('gaussian',RidgeGaussian(torch.load(fits/'ridge_gaussian.pt',weights_only=True))))
    records=[];raw={};started=time.perf_counter()
    for name,head in heads:
        # Fixed global permutation, deterministic and nonidentity; never used for
        # model fitting or ordinary forecast updates. Deliberate bad-evidence test.
        perm=torch.arange(len(x)).roll(137)
        for cutoff in protocol['cutoffs']:
            seen=[j for j,h in enumerate(horizons) if h<=cutoff]
            pending=[j for j,h in enumerate(horizons) if h>cutoff]
            observed_coords=[j*v+a for j in seen for a in range(v)]
            remaining_coords=[j*v+a for j in pending for a in range(v)]
            modes=['prior'] if not seen else ['prior','observed','shuffled']
            for mode in modes:
                parts=[]
                for start in range(0,len(x),96):
                    xx=x[start:start+96];yy=y[start:start+96].double().flatten(1)
                    fun=GaussianMixture.from_landscape(head(xx))
                    if mode=='prior':new=fun.marginal(remaining_coords)
                    else:
                        values=(y[perm[start:start+len(xx)]] if mode=='shuffled' else y[start:start+len(xx)]).double().flatten(1)
                        new=fun.condition(observed_coords,values[:,observed_coords]).distribution
                    target=yy[:,remaining_coords]
                    lower,upper=new.quantile(.05),new.quantile(.95)
                    parts.append({'mean':new.mean,'mse':(new.mean-target).square().reshape(-1,len(pending),v).mean(-1),
                        'nll':new.energy(target)/target.shape[1],
                        'coverage':((target>=lower)&(target<=upper)).double().reshape(-1,len(pending),v).mean(-1),
                        'width':(upper-lower).reshape(-1,len(pending),v).mean(-1)})
                merged={key:torch.cat([p[key] for p in parts]) for key in parts[0]}
                tag=f'{name}_{cutoff}_{mode}';raw[tag]={**merged,'group':groups,'remaining_horizons':[horizons[j] for j in pending]}
                row={'model':name,'cutoff':cutoff,'mode':mode,'n':len(x),'mse':float(merged['mse'].mean()),
                    'nll':float(merged['nll'].mean()),'coverage90':float(merged['coverage'].mean()),'width90':float(merged['width'].mean())}
                for i,t in enumerate([horizons[j] for j in pending]):row[f'mse_h{t}']=float(merged['mse'][:,i].mean())
                records.append(row);print(json.dumps(row),flush=True)
    torch.save(raw,out/'raw.pt')
    frame=pd.DataFrame(records);frame.to_csv(out/'records.csv',index=False)
    frame['family']=frame.model.str.replace(r'_seed\d+$','',regex=True)
    summary=frame.groupby(['family','cutoff','mode']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    pairs=[]
    for cutoff in (1,4,16):
        def agg(family,mode,metric):
            names=[key for key in raw if key.endswith(f'_{cutoff}_{mode}') and key.startswith(family)]
            return torch.stack([raw[n][metric] for n in names]).mean(0)
        for family_a,mode_a,family_b,mode_b in [('mixture','observed','mixture','prior'),('mixture','observed','gaussian','observed'),('mixture','shuffled','mixture','observed')]:
            for metric in ('mse','nll','coverage'):
                delta=agg(family_a,mode_a,metric)-agg(family_b,mode_b,metric)
                if delta.ndim==2:delta=delta.mean(-1)
                pairs.append({'a':f'{family_a}_{mode_a}','b':f'{family_b}_{mode_b}','cutoff':cutoff,'metric':metric,**block_interval(delta,groups)})
    pd.DataFrame(pairs).to_csv(out/'paired.csv',index=False)
    receipt={**protocol,'status':'completed','test_windows':len(x),'test_groups':len(groups.unique()),
             'rows':len(records),'seconds':time.perf_counter()-started}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(summary.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--fits',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.data,a.fits,a.out)

if __name__=='__main__':main()
