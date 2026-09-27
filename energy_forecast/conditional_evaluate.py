"""Evaluate frozen conditioning-study selections; keep failed training visible.

Original test data are reused development data. The optional separate corpus is
never used for fitting/selection. Models see exactly the same matured observations.
Only remaining target coordinates are scored; this is not origin-time foresight.
"""
from pathlib import Path
import argparse,json
import pandas as pd
import torch
from .experiment import load_head,digest
from .conditioning import GaussianMixture,blend
from .calibrated import HorizonAnchor
from .conditional_training import KINDS
from .repair import group_interval


@torch.no_grad()
def evaluate(data,parents,fits,gate,out,secondary=None):
    torch.set_num_threads(1);fits=Path(fits);parents=Path(parents);out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((fits/'FROZEN.json').read_text())
    if frozen['status']!='frozen_before_test':raise ValueError('Frozen selections required')
    for path,sha in frozen['heads'].items():
        p=fits.parent.parent/path if not Path(path).is_absolute() else Path(path)
        if not p.is_file():p=fits/Path(path).parent.name/Path(path).name
        if digest(p)!=sha:raise ValueError('Frozen checkpoint changed')
    base=torch.load(data,weights_only=True);datasets={'development':base}
    if secondary:datasets['separate_news']=torch.load(secondary,weights_only=True)
    gate=json.loads(Path(gate).read_text());gs=torch.load(parents/'ridge_gaussian.pt',weights_only=True)
    meta=base['metadata'];v=meta['variables'];D=v*len(meta['horizons'])
    a=HorizonAnchor(meta['history'],v,meta['horizons']);a.load_state_dict(gs['anchor'])
    # Check actual tensor equality, not checkpoint-container hash equality.
    promotions=[]
    for seed in (11,22,33):
        pm,pmeta=load_head(parents/f'anchored_seed{seed}.pt')
        for kind in KINDS:
            selected,smeta=load_head(fits/f'{kind}_seed{seed}/best.pt')
            same=all(torch.equal(t,pm.state_dict()[key]) for key,t in selected.state_dict().items())
            promotions.append({'kind':kind,'seed':seed,'selected_step':smeta['conditional_study']['selected_step'],'equals_parent_tensors':same})
    records=[];paired=[]
    for name,ds in datasets.items():
        if ds['metadata']['teacher_sha256']!=meta['teacher_sha256']:raise ValueError('Wrong teacher in evaluation data')
        for k in ('center','scale'):
            if not torch.equal(ds['metadata'][k],meta[k]):raise ValueError('Normalization differs')
        p=ds['test'];groups=p['group'];raw={}
        specs=[]
        for seed in (11,22,33):
            pm,_=load_head(parents/f'anchored_seed{seed}.pt')
            specs.extend([(f'parent_raw_{seed}',pm,'raw'),(f'parent_gated_{seed}',pm,'gated')])
            for kind in KINDS:
                sm,_=load_head(fits/f'{kind}_seed{seed}/best.pt');specs.append((f'{kind}_selected_{seed}',sm,'raw'))
                final=torch.load(fits/f'{kind}_seed{seed}/last.pt',weights_only=True)
                fm,_=load_head(parents/f'anchored_seed{seed}.pt');fm.load_state_dict(final['weights'])
                specs.append((f'{kind}_final_diagnostic_{seed}',fm,'raw'))
        specs.append(('gaussian_0',None,'raw'))
        for model_name,model,policy in specs:
            cols={stage:[] for stage in (0,1,4,16)}
            for start in range(0,len(p['x']),96):
                x=p['x'][start:start+96];y=p['y'][start:start+96].flatten(1).double()
                if model is None:
                    mu=a(x).double()[:,None,:];L=gs['chol'].double()[None,None].expand(len(x),1,D,D)
                    f=GaussianMixture(torch.zeros(len(x),1,dtype=torch.float64),mu,L)
                else:f=GaussianMixture.from_landscape(model(x))
                for stage,n in [(0,0),(1,v),(4,2*v),(16,3*v)]:
                    post=f if n==0 else f.condition_prefix(n,y[:,:n]).distribution
                    if policy=='gated' and n:
                        post=blend(f.marginal(list(range(n,D))),post,gate['gates'][str(stage)])
                    target=y[:,n:];lo,hi=post.quantile(.05,iterations=32),post.quantile(.95,iterations=32)
                    cols[stage].append({'mse':(post.mean-target).square().mean(-1),'nll':post.energy(target)/(D-n),
                         'coverage90':((lo<=target)&(target<=hi)).double().mean(-1),
                         'mean':post.mean,'variance':post.variance})
            for stage,items in cols.items():
                values={k:torch.cat([item[k] for item in items]) for k in items[0]}
                raw[f'{model_name}|{stage}']={**values,'group':groups}
                records.append({'dataset':name,'model':model_name,'stage':stage,
                 **{k:float(values[k].mean()) for k in ('mse','nll','coverage90')},'origins':len(groups),'documents':len(groups.unique())})
            print(name,model_name,'evaluated',flush=True)
        torch.save(raw,out/f'{name}_raw.pt')
        comparisons=[('parent_gated','parent_raw')]
        comparisons += [(f'{k}_selected','parent_raw') for k in KINDS]
        comparisons += [(f'{k}_final_diagnostic','parent_raw') for k in KINDS]
        for left,right in comparisons:
            for stage in (0,1,4,16):
                for metric in ('mse','nll'):
                    lv=torch.stack([raw[f'{left}_{s}|{stage}'][metric] for s in (11,22,33)]).mean(0)
                    rv=torch.stack([raw[f'{right}_{s}|{stage}'][metric] for s in (11,22,33)]).mean(0)
                    ci=group_interval((lv-rv).numpy(),groups.numpy(),draws=4000)
                    paired.append({'dataset':name,'a':left,'b':right,'stage':stage,'metric':metric,**ci})
    df=pd.DataFrame(records);df.to_csv(out/'records.csv',index=False)
    df['family']=df.model.str.replace(r'_\d+$','',regex=True)
    summary=df.groupby(['dataset','family','stage']).agg(mse=('mse','mean'),nll=('nll','mean'),coverage90=('coverage90','mean'),origins=('origins','first'),documents=('documents','first')).reset_index()
    summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired.csv',index=False)
    (out/'selection_audit.json').write_text(json.dumps({'checkpoints':promotions,'promotion':'retain_parent' if all(p['equals_parent_tensors'] for p in promotions) else 'see_validation_records',
     'note':'Final-weight outcomes are diagnostics, not promoted checkpoints. No family/step was selected on these tests.'},indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','parents','fits','gate','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--secondary')
    a=p.parse_args();evaluate(a.data,a.parents,a.fits,a.gate,a.out,a.secondary)
if __name__=='__main__':main()
