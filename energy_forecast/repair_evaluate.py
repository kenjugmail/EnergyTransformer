"""Score frozen repairs once, keeping point loss and density quality separate."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .calibrated import load_calibrated,HorizonAnchor
from .model import EnergyLandscape
from .experiment import load_head,digest
from .repair import group_interval,features,ridge_weights

@torch.no_grad()
def marginal_quantile(fun, q: float, iterations=40):
    if not 0<q<1:raise ValueError('Quantile level must be strictly between zero and one')
    sigma=fun.scales*fun.correlation_cholesky.square().sum(-1).sqrt()[None]
    lo=(fun.means-12*sigma).amin(1); hi=(fun.means+12*sigma).amax(1)
    for _ in range(iterations):
        mid=(lo+hi)/2
        z=(mid[:,None]-fun.means)/sigma
        cdf=((.5*(1+torch.erf(z/math.sqrt(2))))*fun.logits.softmax(-1)[...,None]).sum(1)
        lo=torch.where(cdf<q,mid,lo);hi=torch.where(cdf>=q,mid,hi)
    return (lo+hi)/2


class RidgeGaussian:
    def __init__(self,saved):
        m=saved['metadata'];self.anchor=HorizonAnchor(m['history'],m['variables'],tuple(m['horizons']))
        self.anchor.load_state_dict(saved['anchor']);self.chol=saved['chol']
    def __call__(self,x):
        mean=self.anchor(x);return EnergyLandscape(torch.zeros(len(x),1),mean[:,None],torch.ones(len(x),1,mean.shape[1]),self.chol[None])


@torch.no_grad()
def evaluate(data_path,runs,fit_path,out):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    fit_path=Path(fit_path);frozen=json.loads((fit_path/'FROZEN.json').read_text())
    if frozen['data_sha256']!=digest(data_path):raise ValueError('Changed final data')
    for name,value in frozen['files'].items():
        if digest(fit_path/name)!=value:raise ValueError('Changed frozen checkpoint: '+name)
    data=torch.load(data_path,weights_only=True,map_location='cpu');part=data['test'];meta=data['metadata']
    y=part['y'];x=part['x'];h,v=y.shape[1:];groups=part['group'];records=[];raw={}
    def score(name, predictor, density=False):
        predictions=[];nlls=[];energy_scores=[];coverages=[];widths=[]
        for start in range(0,len(x),128):
            xx=x[start:start+128]; yy=y[start:start+128].flatten(1)
            if density:
                fun=predictor(xx);predictions.append(fun.mean.reshape(-1,h,v));nlls.append(fun.energy(yy)/(h*v))
                # Independent paired trajectory draws: unbiased Monte Carlo energy score.
                gen=torch.Generator().manual_seed(269261+start)
                a=fun.sample(128,generator=gen);b=fun.sample(128,generator=gen)
                energy_scores.append(((a-yy[:,None]).norm(dim=-1).mean(1)-.5*(a-b).norm(dim=-1).mean(1))/math.sqrt(h*v))
                low=marginal_quantile(fun,.05);high=marginal_quantile(fun,.95)
                coverages.append(((yy>=low)&(yy<=high)).float().reshape(-1,h,v).mean(-1))
                widths.append((high-low).reshape(-1,h,v).mean(-1))
            else:predictions.append(predictor(xx).reshape(-1,h,v))
        pred=torch.cat(predictions);error=(pred-y).square().mean(-1);r={'prediction':pred,'mse_per_horizon':error,'group':groups}
        record={'name':name,'mse':float(error.mean()),'n':len(y),'groups':len(groups.unique())}
        for i,t in enumerate(meta['horizons']):record[f'mse_h{t}']=float(error[:,i].mean())
        if density:
            r.update(nll=torch.cat(nlls),energy_score=torch.cat(energy_scores),coverage=torch.cat(coverages),width=torch.cat(widths))
            record.update(nll=float(r['nll'].mean()),energy_score=float(r['energy_score'].mean()),coverage90=float(r['coverage'].mean()),width90=float(r['width'].mean()))
        raw[name]=r;torch.save(r,out/f'{name}.pt');records.append(record);print(record,flush=True)
    saved=torch.load(fit_path/'anchor.pt',weights_only=True)
    anchor=HorizonAnchor(meta['history'],v,tuple(meta['horizons']));anchor.load_state_dict(saved['weights'])
    score('constant',lambda xx:data['train']['y'].mean(0).flatten().expand(len(xx),-1))
    score('horizon_ridge_raw',lambda xx:(features(xx).float()@anchor.weight))
    score('gated_ridge',anchor)
    # Reconstruct the original limited-grid, persistence-residual ridge exactly.
    tx,vx=features(data['train']['x']),features(data['validation']['x'])
    last=lambda p:p['x'][:,-1,None,:].expand_as(p['y']).flatten(1).double()
    target=data['train']['y'].flatten(1).double()-last(data['train']);options=[]
    for a in (.1,1.,10.,100.):
        w=ridge_weights(tx,target,a)
        mse=((vx@w+last(data['validation']))-data['validation']['y'].flatten(1)).square().mean()
        options.append((float(mse),a,w))
    _,alpha,w=min(options,key=lambda z:z[0])
    score('original_ridge',lambda xx:((features(xx)@w)+xx[:,-1,:].repeat(1,h).double()).float())
    score('ridge_gaussian',RidgeGaussian(torch.load(fit_path/'ridge_gaussian.pt',weights_only=True)),True)
    for seed in (11,22,33):
        old,_=load_head(Path(runs)/f'spectral_seed{seed}/best.pt')
        score(f'original_spectral_seed{seed}',old,True)
        for family in ('recentered_uncal','recentered','anchored_uncal','anchored'):
            model,_=load_calibrated(fit_path/f'{family}_seed{seed}.pt')
            score(f'{family}_seed{seed}',model,True)
    frame=pd.DataFrame(records);frame.to_csv(out/'scores.csv',index=False)
    frame['family']=frame.name.str.replace(r'_seed\d+$','',regex=True)
    summary=frame.groupby('family').mean(numeric_only=True).reset_index();summary.to_csv(out/'summary.csv',index=False)
    def aggregated(family,metric):
        names=[name for name in raw if name==family or name.startswith(family+'_seed')]
        return torch.stack([raw[name][metric] for name in names]).mean(0)
    pairs=[]
    comparisons=[('gated_ridge','original_ridge'),('gated_ridge','horizon_ridge_raw'),('anchored','original_spectral'),
                 ('anchored','recentered'),('anchored','ridge_gaussian'),('anchored','anchored_uncal')]
    for left,right in comparisons:
        for metric in ('mse_per_horizon','nll','energy_score'):
            try:delta=aggregated(left,metric)-aggregated(right,metric)
            except KeyError:continue
            if delta.ndim==2:
                for i,t in enumerate(meta['horizons']):pairs.append({'a':left,'b':right,'metric':f'mse_h{t}',**group_interval(delta[:,i].numpy(),groups.numpy())})
                delta=delta.mean(-1)
            pairs.append({'a':left,'b':right,'metric':metric,**group_interval(delta.numpy(),groups.numpy())})
    pd.DataFrame(pairs).to_csv(out/'paired.csv',index=False)
    (out/'receipt.json').write_text(json.dumps({'frozen_sha256':digest(fit_path/'FROZEN.json'), 'data_sha256':digest(data_path),
        'records':len(records),'test_windows':len(y),'test_groups':len(groups.unique()),'old_ridge_alpha':alpha,
        'point_mean_of_all_recentered_variants':'gated_ridge','no_future_state_commits':True},indent=2))
    print(summary.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--parent-runs',required=True)
    p.add_argument('--fits',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    evaluate(a.data,a.parent_runs,a.fits,a.out)
if __name__=='__main__':main()
