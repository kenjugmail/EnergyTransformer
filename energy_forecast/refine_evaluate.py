"""Fixed-head depth, halting, and matched-memory interventions; freeze before test.

No test-based depth selection. Report all declared depths even when worse.
Bootstrap entire documents after averaging seeds; this is reused-domain evidence.
"""
from pathlib import Path
import argparse,json,math
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .refine_model import load_refinement
from .refine_train import per_example
from .probe import checkpoint_hash

DEPTHS=(0,1,2,4,8,16)
TOLS=(.0001,.001,.01)

def paired(a,b,groups):
    delta=np.asarray(a)-np.asarray(b);ids=np.unique(groups)
    sums=np.array([delta[groups==g].sum() for g in ids]);counts=np.array([(groups==g).sum() for g in ids])
    rng=np.random.default_rng(271203);ii=rng.integers(len(ids),size=(3000,len(ids)))
    stat=sums[ii].sum(1)/counts[ii].sum(1)
    return {'difference_bpb':float(delta.mean()),'ci_low':float(np.quantile(stat,.025)),
            'ci_high':float(np.quantile(stat,.975)),'groups':len(ids)}


def policy_list(kind):
    if kind in ('feedforward','memory_ff','latent_ff'):return [('fixed',1,None)]
    result=[('fixed',k,None) for k in DEPTHS]
    if kind.startswith('energy_'):result.extend(('adaptive',16,t) for t in TOLS)
    return result


def policy_key(p):return f'{p[0]}_{p[1]}_{p[2]}'


def cross_group_permutation(groups):
    # A one-to-one cyclic permutation exists when no group occupies >half rows.
    order=torch.argsort(groups,stable=True);count=torch.unique_consecutive(groups[order],return_counts=True)[1]
    shift=int(count.max())
    if 2*shift>len(groups):raise ValueError('Cannot construct cross-group permutation for dominant document')
    perm=torch.empty_like(order);perm[order]=order.roll(shift)
    if (groups==groups[perm]).any():raise AssertionError('Intervention must cross document groups')
    return perm


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--fits',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(1);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    fits=Path(args.fits);frozen=json.loads((fits/'FROZEN.json').read_text())
    if checkpoint_hash(args.data)!=frozen['data_sha256']:raise ValueError('Changed data')
    for name,digest in frozen['files'].items():
        if checkpoint_hash(fits/name)!=digest:raise ValueError('Changed selected weights')
    data=torch.load(args.data,weights_only=True,map_location='cpu')
    # The inference policy is chosen on validation. Neither final partition enters here.
    validation=[];chosen={};models={}
    for kind in frozen['kinds']:
        totals={policy_key(p):[] for p in policy_list(kind)}
        for seed in frozen['seeds']:
            model,meta=load_refinement(fits/f'{kind}_seed{seed}/best.pt');models[kind,seed]=model
            for policy in policy_list(kind):
                value,n=per_example(model,data['validation'],data['weight'],steps=policy[1],policy=policy[2])
                totals[policy_key(policy)].append(float(value.mean()))
                validation.append({'kind':kind,'seed':seed,'mode':policy[0],'steps':policy[1],'tolerance':policy[2],
                                   'bpb':float(value.mean()),'mean_updates':float(n.float().mean()) if n is not None else policy[1]})
        chosen[kind]=min(policy_list(kind),key=lambda p:np.mean(totals[policy_key(p)]))
    policy_manifest={'teacher':data['metadata']['teacher_sha256'],'selection_sha256':checkpoint_hash(fits/'FROZEN.json'),
        'policy':chosen,'validation_only':True,'warning':'Policies chosen on reused validation documents; no guaranteed optimal halting.'}
    (out/'INFERENCE_FROZEN.json').write_text(json.dumps(policy_manifest,indent=2))
    pd.DataFrame(validation).to_csv(out/'validation_depth.csv',index=False)
    print('FROZEN_INFERENCE',chosen,flush=True)
    rows=[];arrays={};mechanics=[]
    for name in ('test','secondary'):
        part=data[name];part['permutation']=cross_group_permutation(part['group']);groups=part['group'].numpy();arrays[name]={}
        base=F.cross_entropy(F.linear(part['h'],data['weight']),part['y'],reduction='none').numpy()/math.log(2)
        arrays[name]['base']=base
        rows.append({'dataset':name,'kind':'base','seed':0,'mode':'fixed','steps':0,'tolerance':None,'intervention':'normal','bpb':float(base.mean()),'targets':len(base)})
        np.savez_compressed(out/f'{name}_base.npz',bpb=base,group=groups,anchor=part['anchor'].numpy())
        for kind in frozen['kinds']:
            for seed in frozen['seeds']:
                model=models[kind,seed]
                for policy in policy_list(kind):
                    values,counts=per_example(model,part,data['weight'],steps=policy[1],policy=policy[2])
                    key=policy_key(policy);v=values.numpy();arrays[name].setdefault((kind,key,'normal'),[]).append(v)
                    rows.append({'dataset':name,'kind':kind,'seed':seed,'mode':policy[0],'steps':policy[1],
                        'tolerance':policy[2],'intervention':'normal','bpb':float(v.mean()),'targets':len(v),
                        'mean_updates':float(counts.float().mean()) if counts is not None else policy[1]})
                    np.savez_compressed(out/f'{name}_{kind}_{seed}_{key}.npz',bpb=v,group=groups,
                        updates=counts.numpy() if counts is not None else np.full(len(v),policy[1]))
                if model.config.uses_memory:
                    for intervention in ('zero','shuffle'):
                        v=per_example(model,part,data['weight'],steps=4,memory_mode=intervention)[0].numpy()
                        arrays[name].setdefault((kind,policy_key(('fixed',1 if kind in ('memory_ff','latent_ff') else 4,None)),intervention),[]).append(v)
                        rows.append({'dataset':name,'kind':kind,'seed':seed,'mode':'fixed','steps':1 if kind in ('memory_ff','latent_ff') else 4,
                            'tolerance':None,'intervention':intervention,'bpb':float(v.mean()),'targets':len(v)})
                if model.config.energy:
                    # Energy and language loss at each UNGUARDED trained update.
                    h,m=part['h'][:256],part['m'][:256];c=model.landscape(model.context_input(h,m))
                    z=torch.zeros(len(h),model.config.latent);old_e=model.energy(z,c)
                    old_loss=F.cross_entropy(model.logits(h,z,data['weight']),part['y'][:256],reduction='none')
                    for step in range(1,17):
                        z=z-model.config.step_size*model.gradient(z,c)
                        new_e=model.energy(z,c);loss=F.cross_entropy(model.logits(h,z,data['weight']),part['y'][:256],reduction='none')
                        mechanics.append({'dataset':name,'kind':kind,'seed':seed,'step':step,'n':len(h),
                            'energy_increases':int((new_e>old_e+1e-6).sum()),
                            'energy_down_loss_up':int(((new_e<old_e)&(loss>old_loss)).sum()),
                            'mean_energy':float(new_e.mean().detach()),'bpb':float(loss.mean().detach())/math.log(2)})
                        old_e=new_e;old_loss=loss
                print('scored',name,kind,seed,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'runs.csv',index=False)
    frame['tolerance']=frame.tolerance.fillna(-1)
    summary=frame.groupby(['dataset','kind','mode','steps','tolerance','intervention']).agg(
        mean_bpb=('bpb','mean'),sd_bpb=('bpb','std'),targets=('targets','first'),mean_updates=('mean_updates','mean')).reset_index()
    summary['perplexity']=2**summary.mean_bpb;summary.to_csv(out/'summary.csv',index=False)
    comparisons=[]
    for name,ar in arrays.items():
        groups=data[name]['group'].numpy()
        default={kind:np.mean(ar[(kind,policy_key(('fixed',1 if kind in ('feedforward','memory_ff','latent_ff') else 4,None)),'normal')],0) for kind in frozen['kinds']}
        for kind,v in default.items():
            for ref in ('base','feedforward','memory_ff','recurrent','energy_hidden','latent_ff'):
                if ref==kind or (ref!='base' and ref not in default):continue
                comparisons.append({'dataset':name,'a':kind,'b':ref,'condition':'default',**paired(v,ar['base'] if ref=='base' else default[ref],groups)})
            selected=np.mean(ar[kind,policy_key(tuple(chosen[kind])),'normal'],0)
            comparisons.append({'dataset':name,'a':kind,'b':kind,'condition':'validation_policy_minus_default',**paired(selected,v,groups)})
        for kind in ('energy_hidden','energy_memory','energy_full','recurrent'):
            if kind not in default:continue
            for depth in (1,2,8,16):
                vv=np.mean(ar[kind,policy_key(('fixed',depth,None)),'normal'],0)
                comparisons.append({'dataset':name,'a':f'{kind}@{depth}','b':f'{kind}@4','condition':'depth',**paired(vv,default[kind],groups)})
    pd.DataFrame(comparisons).to_csv(out/'paired.csv',index=False)
    pd.DataFrame(mechanics).to_csv(out/'energy_vs_loss.csv',index=False)
    receipt={'status':'completed','data_sha256':checkpoint_hash(args.data),'frozen_models_sha256':checkpoint_hash(fits/'FROZEN.json'),
        'inference_selection_sha256':checkpoint_hash(out/'INFERENCE_FROZEN.json'),'records':len(rows),
        'independent_targets':{s:len(data[s]['y']) for s in ('test','secondary')},
        'no_new_training_after_test':True}
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(summary.to_string(index=False))
if __name__=='__main__':main()
