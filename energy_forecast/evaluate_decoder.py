"""Score frozen decoder-adapter choices on identical next-byte labels.

Stored forecast heads and domains were used in earlier research. These scores are
sampled-position development evidence, not untouched whole-corpus LM evaluations.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from .decoder_bridge import load_adapter,KINDS
from .experiment import load_head
from .train_decoder import forecast_cache,losses
from .probe import checkpoint_hash
from .decoder_integrity import selected_runs, validate_head_metadata

def paired(a,b,groups,seed=21009):
    delta=np.asarray(a)-np.asarray(b);unique=np.unique(groups)
    count=np.array([(groups==g).sum() for g in unique]);total=np.array([delta[groups==g].sum() for g in unique])
    rng=np.random.default_rng(seed);idx=rng.integers(len(unique),size=(5000,len(unique)))
    draws=total[idx].sum(1)/count[idx].sum(1)
    return {'difference_bpb':float(delta.mean()),'ci_low':float(np.quantile(draws,.025)),
            'ci_high':float(np.quantile(draws,.975)),'document_groups':len(unique)}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--heads');p.add_argument('--fits',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv)
    torch.set_num_threads(1);fits=Path(a.fits);out=Path(a.out)
    frozen=json.loads((fits/'FROZEN.json').read_text())
    kinds,seeds,selected=selected_runs(frozen)
    if checkpoint_hash(a.data)!=frozen['data_sha256']:raise ValueError('Evaluation data identity changed')
    if 'study_sha256' in frozen and checkpoint_hash(fits/'STUDY.json')!=frozen['study_sha256']:
        raise ValueError('Study manifest changed')
    for _,_,name,digest in selected:
        if checkpoint_hash(fits/name)!=digest:raise ValueError('Selected checkpoint changed')
    data=torch.load(a.data,map_location='cpu',weights_only=True);records=[];arrays={}
    heads={}
    for seed in seeds:
        if a.heads:
            head,hm=load_head(Path(a.heads)/f'anchored_seed{seed}.pt')
            validate_head_metadata(hm,data['metadata'])
            heads[seed]=head
        elif any(k not in ('hidden','history') for k in kinds):
            raise ValueError('Forecast heads required')
    out.mkdir(parents=True,exist_ok=False)
    for dataset in [name for name in ('test','secondary') if name in data]:
        part=data[dataset];arrays[dataset]={};groups=part['group'].numpy()
        base=losses(None,part,data['weight']).numpy();arrays[dataset]['base']=base
        records.append({'dataset':dataset,'kind':'base','seed':0,'intervention':'none','bpb':float(base.mean()),'targets':len(base)})
        np.savez_compressed(out/f'{dataset}_base.npz',bpb=base,group=groups,anchor=part['anchor'].numpy())
        permutation=torch.arange(len(part['h'])).roll(len(part['h'])//2)
        if (part['group']==part['group'][permutation]).any():raise AssertionError('Shuffle must cross documents')
        for seed in seeds:
            if a.heads:
                cache=forecast_cache(heads[seed],part)
            else:
                if any(k not in ('hidden','history') for k in kinds):raise ValueError('Forecast heads required')
                cache=None
            for kind in kinds:
                m,meta=load_adapter(fits/f'{kind}_seed{seed}'/'best.pt')
                if meta['teacher_sha256']!=data['metadata']['teacher_sha256']:raise ValueError('Teacher mismatch')
                if kind in ('mean','moments','energy_read') and meta['head_sha256']!=checkpoint_hash(Path(a.heads)/f'anchored_seed{seed}.pt'):raise ValueError('Head mismatch')
                for intervention in ('none',) if kind=='hidden' else ('none','zero_side','shuffle_side'):
                    value=losses(m,part,data['weight'],cache,erase_side=intervention=='zero_side',
                        permutation=permutation if intervention=='shuffle_side' else None).numpy()
                    records.append({'dataset':dataset,'kind':kind,'seed':seed,'intervention':intervention,
                                    'bpb':float(value.mean()),'targets':len(value),'selected_step':meta['selected_step']})
                    np.savez_compressed(out/f'{dataset}_{kind}_seed{seed}_{intervention}.npz',bpb=value,group=groups,anchor=part['anchor'].numpy())
                    arrays[dataset].setdefault((kind,intervention),[]).append(value)
            print('scored',dataset,seed,flush=True)
    pd.DataFrame(records).to_csv(out/'runs.csv',index=False)
    summary=pd.DataFrame(records).groupby(['dataset','kind','intervention']).agg(mean_bpb=('bpb','mean'),sd_bpb=('bpb','std'),targets=('targets','first')).reset_index()
    summary['perplexity']=2**summary.mean_bpb;summary.to_csv(out/'summary.csv',index=False)
    comparisons=[]
    for dataset in arrays:
        base=arrays[dataset]['base'];groups=data[dataset]['group'].numpy()
        means={kind:np.mean(arrays[dataset][(kind,'none')],0) for kind in kinds}
        for kind in kinds:
            for other in ('base','hidden','history'):
                if kind==other or (other!='base' and other not in means):continue
                comparisons.append({'dataset':dataset,'a':kind,'b':other,**paired(means[kind],base if other=='base' else means[other],groups)})
            if kind!='hidden':
                for intervention in ('zero_side','shuffle_side'):
                    changed=np.mean(arrays[dataset][(kind,intervention)],0)
                    comparisons.append({'dataset':dataset,'a':f'{kind}:{intervention}','b':kind,**paired(changed,means[kind],groups)})
    pd.DataFrame(comparisons).to_csv(out/'paired.csv',index=False)
    (out/'EVALUATION.json').write_text(json.dumps({'data_sha256':checkpoint_hash(a.data),'selection_sha256':checkpoint_hash(fits/'FROZEN.json'),
        'candidate':'energy_read' if 'energy_read' in kinds else None,'seeds':seeds,'validation_winner':frozen['validation_winner'],'records':len(records),
        'unique_primary_origins':len(data['test']['y']),'unique_secondary_origins':len(data['secondary']['y']) if 'secondary' in data else 0,
        'warning':'Repeated domains. Pretrained forecasts carry extra frozen parameters and prior training. No held-out corpus test claim.'},indent=2))
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
