"""Select cautious posterior weights using validation documents, never test targets.

An exploratory engineering follow-up to the retained raw-conditioning audit, not
preregistration or formal simultaneous risk control. One gate per revealed-prefix
stage pools the original remaining joint distribution with its exact posterior.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import pandas as pd
import torch
from .conditioning import GaussianMixture,blend
from .conditioning_audit import block_interval
from .experiment import load_head,digest
from .repair import split_validation


@torch.no_grad()
def select(data_path,fits,out):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False);fits=Path(fits)
    paths=[fits/f'anchored_seed{s}.pt' for s in (11,22,33)]
    grid=[0.,.05,.1,.25,.5,.75,1.]
    protocol={'kind':'exploratory_validation_gated_conditioning','selected_on':'original validation selection half only',
              'grid':grid,'choose':'lowest mean validation NLL among weights with negative upper 95% bootstrap bounds for MSE and NLL; else 0',
              'teacher_sha256':None,'heads':{p.name:digest(p) for p in paths},'data_sha256':digest(data_path),
              'prior_raw_audit':'Unmodified conditioning was already scored; gate is explicitly post-hoc. No neural retraining.',
              'calibration_half':'descriptive check only, never changes selection','test_role':'evaluation only'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    d=torch.load(data_path,weights_only=True)
    if d['metadata']['variables']!=8 or list(d['metadata']['horizons'])!=[1,4,16,64]:raise ValueError('This evaluation protocol requires 8 variables and horizons 1/4/16/64')
    sel,cal=split_validation(d['validation']['group'])
    protocol['teacher_sha256']=d['metadata']['teacher_sha256']
    heads=[load_head(p)[0] for p in paths];records=[];gates={}
    for cutoff,n in [(1,8),(4,16),(16,24)]:
        x=d['validation']['x'][sel];y=d['validation']['y'][sel].double().flatten(1);group=d['validation']['group'][sel]
        funcs=[GaussianMixture.from_landscape(m(x)) for m in heads]
        priors=[f.marginal(list(range(n,32))) for f in funcs]
        posts=[f.condition(list(range(n)),y[:,:n]).distribution for f in funcs]
        yy=y[:,n:];p_mse=torch.stack([(p.mean-yy).square().mean(-1) for p in priors]).mean(0)
        p_nll=torch.stack([p.energy(yy)/(32-n) for p in priors]).mean(0)
        candidates=[]
        for g in grid:
            fs=[blend(p,q,g) for p,q in zip(priors,posts)]
            mse=torch.stack([(f.mean-yy).square().mean(-1) for f in fs]).mean(0)
            nll=torch.stack([f.energy(yy)/(32-n) for f in fs]).mean(0)
            dm=block_interval(mse-p_mse,group);dn=block_interval(nll-p_nll,group)
            valid=g==0 or (dm['ci_high']<0 and dn['ci_high']<0)
            records.append({'cutoff':cutoff,'weight':g,'mse':float(mse.mean()),'nll':float(nll.mean()),'admissible':valid,
                            'mse_delta_upper':dm['ci_high'],'nll_delta_upper':dn['ci_high']})
            if valid:candidates.append((float(nll.mean()),g))
        gates[str(cutoff)]=min(candidates)[1]
    pd.DataFrame(records).to_csv(out/'selection.csv',index=False)
    saved={**protocol,'format':'forecast_conditioning_gate_v1','gates':gates,'selected_validation_groups':d['validation']['group'][sel].unique().tolist(),
           'calibration_validation_groups':d['validation']['group'][cal].unique().tolist(),
           'warning':'Empirical reuse-domain heuristic. Not calibrated conditional coverage or a guarantee under shift.'}
    (out/'gate.json').write_text(json.dumps(saved,indent=2))
    print(json.dumps(saved,indent=2),flush=True)
    evaluate_gated(d,heads,gates,out,'calibration_check',cal)
    # Selection has been serialized; remaining calls never change gates.
    evaluate_gated(d,heads,gates,out,'test',None)


@torch.no_grad()
def evaluate_gated(data,heads,gates,out,name,mask):
    part=data['test'] if name=='test' else data['validation']
    if mask is None:mask=torch.ones(len(part['x']),dtype=torch.bool)
    x,y,groups=part['x'][mask],part['y'][mask].double().flatten(1),part['group'][mask]
    records=[];raw={};pair=[]
    for cutoff,n in [(1,8),(4,16),(16,24)]:
        g=gates[str(cutoff)];mse_lists=[];nll_lists=[];baseline_mse=[];baseline_nll=[]
        for s,model in zip((11,22,33),heads):
            parts=[];base=[]
            for i in range(0,len(x),96):
                f=GaussianMixture.from_landscape(model(x[i:i+96]));yy=y[i:i+96]
                p=f.marginal(list(range(n,32)));q=f.condition(list(range(n)),yy[:,:n]).distribution
                mixed=blend(p,q,g);target=yy[:,n:]
                lo,hi=mixed.quantile(.05),mixed.quantile(.95)
                parts.append({'mean':mixed.mean,'mse':(mixed.mean-target).square().reshape(-1,(32-n)//8,8).mean(-1),
                              'nll':mixed.energy(target)/(32-n),'coverage':((target>=lo)&(target<=hi)).double().mean(-1)})
                base.append({'mse':(p.mean-target).square().mean(-1),'nll':p.energy(target)/(32-n)})
            vals={k:torch.cat([p[k] for p in parts]) for k in parts[0]};raw[f'seed{s}_{cutoff}']=dict(vals,group=groups)
            row={'seed':s,'cutoff':cutoff,'weight':g,'mse':float(vals['mse'].mean()),'nll':float(vals['nll'].mean()),'coverage90':float(vals['coverage'].mean())}
            for j,h in enumerate([h for h in (1,4,16,64) if h>cutoff]):row[f'mse_h{h}']=float(vals['mse'][:,j].mean())
            records.append(row);mse_lists.append(vals['mse'].mean(-1));nll_lists.append(vals['nll'])
            baseline_mse.append(torch.cat([b['mse'] for b in base]));baseline_nll.append(torch.cat([b['nll'] for b in base]))
        for metric,a,b in [('mse',mse_lists,baseline_mse),('nll',nll_lists,baseline_nll)]:
            pair.append({'cutoff':cutoff,'metric':metric,**block_interval(torch.stack(a).mean(0)-torch.stack(b).mean(0),groups)})
    torch.save(raw,out/f'{name}_raw.pt');frame=pd.DataFrame(records);frame.to_csv(out/f'{name}_records.csv',index=False)
    frame.groupby('cutoff').mean(numeric_only=True).reset_index().to_csv(out/f'{name}_summary.csv',index=False)
    pd.DataFrame(pair).to_csv(out/f'{name}_paired.csv',index=False)
    print(name,frame.groupby('cutoff').mean(numeric_only=True).to_string(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--fits',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();select(a.data,a.fits,a.out)
if __name__=='__main__':main()
