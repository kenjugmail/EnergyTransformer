"""Post-selection dense and longer-prefix audit; no parameters are fitted here."""
from pathlib import Path
import argparse,json,hashlib,time
import torch
import pandas as pd
import numpy as np
from .prepare_decoder import extract
from .probe import load_energy_model,checkpoint_hash
from .experiment import load_head,atomic_save
from .train_decoder import forecast_cache,losses
from .decoder_bridge import load_adapter,KINDS
from .evaluate_decoder import paired

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');p.add_argument('--corpus',required=True);p.add_argument('--secondary',required=True);p.add_argument('--max-bytes',type=int,default=1024);a=p.parse_args()
    root=Path(a.root);out=root/'results/decoder_bridge/dense_audit';out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((root/'results/decoder_bridge/fits/FROZEN.json').read_text())
    for rel,h in frozen['files'].items():
        if checkpoint_hash(root/'results/decoder_bridge/fits'/rel)!=h:raise ValueError('Changed selected file')
    meta=torch.load(root/'results/state_data.pt',weights_only=True)['metadata'];raw=Path(a.corpus).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['corpus_sha256']:raise ValueError('Wrong corpus')
    lookup={hashlib.sha256(s).hexdigest():s for s in raw.splitlines()};docs=[]
    for i,d in enumerate(meta['documents']):
        if d['split']=='test':docs.append({'text':lookup[d['sha256']],'sha256':d['sha256'],'group':i,'split':'test'})
    for j,b in enumerate(Path(a.secondary).read_bytes().splitlines()):
        if len(b)>=18:docs.append({'text':b,'sha256':hashlib.sha256(b).hexdigest(),'group':1000+j,'split':'secondary'})
    torch.set_num_threads(1);teacher=load_energy_model(root/'checkpoints/teacher_973k.pt')
    rows=extract(teacher,docs,stride=1,max_bytes=a.max_bytes,batch=12);parts={}
    for name in ('test','secondary'):
        rr=[r for r in rows if r['split']==name]
        part={k:torch.cat([r[k] for r in rr]) for k in ('h','x','y','group','anchor')}
        part['x']=(part['x']-meta['center'])/meta['scale'];parts[name]=part
    all_records=[];differences=[]
    for name,part in parts.items():
        base=losses(None,part,teacher.head.weight).numpy();by_kind={}
        for seed in (11,22,33):
            head,_=load_head(root/f'results/repair/fits/anchored_seed{seed}.pt');cache=forecast_cache(head,part)
            for kind in KINDS:
                adapter,_=load_adapter(root/f'results/decoder_bridge/fits/{kind}_seed{seed}/best.pt')
                value=losses(adapter,part,teacher.head.weight,cache).numpy();by_kind.setdefault(kind,[]).append(value)
                np.savez_compressed(out/f'{name}_{kind}_seed{seed}.npz',bpb=value,group=part['group'].numpy(),anchor=part['anchor'].numpy())
            print('dense scored',name,seed,flush=True)
        for region,mask in [('early',part['anchor'].numpy()<383),('late',part['anchor'].numpy()>=383),('all',np.ones(len(base),dtype=bool))]:
            if not mask.any():continue
            groups=part['group'].numpy()[mask]
            all_records.append({'dataset':name,'region':region,'kind':'base','mean_bpb':float(base[mask].mean()),'sd_bpb':0.,'targets':int(mask.sum()),'documents':len(np.unique(groups))})
            for kind,arr in by_kind.items():
                means=np.array([r[mask].mean() for r in arr]);all_records.append({'dataset':name,'region':region,'kind':kind,'mean_bpb':float(means.mean()),'sd_bpb':float(means.std(ddof=1)),'targets':int(mask.sum()),'documents':len(np.unique(groups))})
                averaged=np.mean(arr,0)[mask]
                for other in ('base','hidden'):
                    if kind==other:continue
                    ref=base[mask] if other=='base' else np.mean(by_kind[other],0)[mask]
                    differences.append({'dataset':name,'region':region,'a':kind,'b':other,**paired(averaged,ref,groups)})
    pd.DataFrame(all_records).to_csv(out/'summary.csv',index=False);pd.DataFrame(differences).to_csv(out/'paired.csv',index=False)
    (out/'PROVENANCE.json').write_text(json.dumps({'phase':'post-hoc audit after original held-out scoring; no retraining or selection','selection_sha256':checkpoint_hash(root/'results/decoder_bridge/fits/FROZEN.json'),'max_bytes':a.max_bytes,'stride':1,'same_document_domain':True},indent=2))
    print(pd.DataFrame(all_records).to_string(index=False))
if __name__=='__main__':main()
