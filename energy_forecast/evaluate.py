"""Held-out forecast scoring, simple baselines and fixed-landscape MAP audit.

No generated point forecast replaces a real LM state; no language logits change.
Mean squared errors use train-standardized summary coordinates, NOT token loss.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from .experiment import load_head,digest

def bootstrap(delta,groups,seed=18822):
    unique=np.unique(groups);sums=np.array([delta[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique]);r=np.random.default_rng(seed)
    choices=r.integers(len(unique),size=(2000,len(unique)));values=sums[choices].sum(1)/counts[choices].sum(1)
    return {'mean':float(delta.mean()),'low':float(np.quantile(values,.025)),'high':float(np.quantile(values,.975)),'groups':len(unique)}

@torch.no_grad()
def predict(m,part):
    mu=[];nll=[]
    for start in range(0,len(part['x']),128):
        fun=m(part['x'][start:start+128]);y=part['y'][start:start+128]
        mu.append(fun.mean.reshape_as(y));nll.append(fun.energy(y.flatten(1))/m.d)
    return torch.cat(mu),torch.cat(nll)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--runs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    data=torch.load(a.data,weights_only=True,map_location='cpu');test=data['test'];rows=[];preds={};groups=test['group'].numpy();y=test['y']
    def record(name,mu,nll=None,parameters=0):
        error=(mu-y).square().mean((1,2));preds[name]={'mse':error.numpy(),'nll':None if nll is None else nll.numpy()}
        interval=bootstrap(error.numpy(),groups)
        row={'name':name,'parameters':parameters,'mse':float(error.mean()),'mse_low':interval['low'],'mse_high':interval['high'],'nll':None if nll is None else float(nll.mean()),'examples':len(y),'documents_or_episodes':len(np.unique(groups))}
        for i,h in enumerate(data['metadata']['horizons']):row[f'mse_h{h}']=float((mu[:,i]-y[:,i]).square().mean())
        rows.append(row)
        torch.save({'predicted_mean':mu,'per_example_mse':error,'nll_per_coordinate':nll,'group':test['group']},out/(name+'.pt'))
    record('persistence',test['x'][:,-1,None,:].expand_as(y))
    record('train_mean',data['train']['y'].mean(0,keepdim=True).expand_as(y))
    # Ridge chosen by validation MSE, never by test. History-only residual regression.
    def features(part):
        x=part['x'].flatten(1).double();return torch.cat([x,torch.ones(len(x),1,dtype=torch.float64)],1)
    tx,vx,xx=[features(data[k]) for k in ('train','validation','test')]
    def anchor(part):return part['x'][:,-1,None,:].expand_as(part['y']).flatten(1).double()
    target=data['train']['y'].flatten(1).double()-anchor(data['train']);prior=torch.eye(tx.shape[1],dtype=tx.dtype);prior[-1,-1]=0
    choices=[]
    for alpha in (.1,1.,10.,100.):
        w=torch.linalg.solve(tx.T@tx+alpha*prior,tx.T@target)
        err=((vx@w+anchor(data['validation']))-data['validation']['y'].flatten(1)).square().mean()
        choices.append((float(err),alpha,w))
    _,alpha,w=min(choices,key=lambda z:z[0]);record('ridge',(xx@w+anchor(test)).float().reshape_as(y),parameters=w.numel())
    (out/'ridge_selection.json').write_text(json.dumps({'selected_alpha':alpha,'validation_mse':[(v,b) for v,b,_ in choices]},indent=2))
    audits=[]
    for file in sorted(Path(a.runs).glob('*/best.pt')):
        m,d=load_head(file);name=file.parent.name
        if d['identity']['data_sha256']!=digest(a.data):raise ValueError('Wrong head/data pairing')
        mu,nll=predict(m,test);record(name,mu,nll,sum(p.numel() for p in m.parameters()))
        if name.endswith('11'):
            subset=min(192,len(y));x=test['x'][:subset];truth=y[:subset].flatten(1)
            with torch.no_grad():
                fun=m(x);components=fun.means.shape[1]
                energies=torch.stack([fun.energy(fun.means[:,k]) for k in range(components)],1)
                indices=energies.argmin(1);initial=fun.means[torch.arange(subset),indices]
            refined,history=fun.refine(initial,steps=6,backtracks=8)
            with torch.no_grad():
                # Conditional ranking against fixed offsets from other held-out examples.
                candidate_e=[fun.energy(truth)]
                for offset in (17,37,59,83,101,139,163):
                    candidates=y[(torch.arange(subset)+offset)%len(y)].flatten(1)
                    candidate_e.append(fun.energy(candidates))
                rank=torch.stack(candidate_e,1).argmin(1)==0
                audits.append({'model':name,'points':subset,'initial_mode_mse':float((initial-truth).square().mean()),
                    'refined_mode_mse':float((refined-truth).square().mean()),'conditional_mean_mse':float((fun.mean-truth).square().mean()),
                    'initial_energy':float(history[:,0].mean()),'final_energy':float(history[:,-1].mean()),
                    'positive_energy_transitions':int(((history[:,1:]-history[:,:-1])>1e-6).sum()),
                    'eight_candidate_ranking_accuracy':float(rank.float().mean())})
                torch.save({'initial':initial,'refined':refined,'energies':history,'targets':truth},out/(name+'_refinement.pt'))
        print(name,rows[-1],flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'scores.csv',index=False)
    df['family']=df.name.str.replace(r'_seed\d+$','',regex=True)
    mean=df.groupby('family').agg(mse=('mse','mean'),nll=('nll','mean'),parameters=('parameters','first'),seed_sd=('mse','std')).reset_index()
    mean.to_csv(out/'summary.csv',index=False)
    (out/'refinement.json').write_text(json.dumps(audits,indent=2))
    paired=[]
    for left,right in [('joint','diagonal'),('mixture','joint'),('spectral','mixture')]:
        if not all(f'{q}_seed{s}' in preds for q in (left,right) for s in (11,22,33)):continue
        for metric in ('mse','nll'):
            delta=np.mean([preds[f'{left}_seed{s}'][metric]-preds[f'{right}_seed{s}'][metric] for s in (11,22,33)],0)
            paired.append({'a':left,'b':right,'metric':metric,**bootstrap(delta,groups)})
    pd.DataFrame(paired).to_csv(out/'paired.csv',index=False)
    print(mean.to_string(index=False),flush=True)
if __name__=='__main__':main()
