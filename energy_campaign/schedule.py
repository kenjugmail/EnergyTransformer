"""Validation-only staged schedule generation; never promotes partial fits."""
from pathlib import Path
import json
import math
from dataclasses import asdict
from .spec import config,choose_budget,FAMILIES,SEEDS,source_identity
from energy_real.runtime import atomic_json


def profiles(path):
    d=json.loads(Path(path).read_text())
    if d['source_sha256']!=source_identity() or d['runtime']['device']!='cuda':raise ValueError('Fresh GPU profiles required')
    chosen={}
    for row in d['rows']:
        if row.get('status')=='failed':continue
        if row.get('effective_targets')!=8192 or row.get('batch')!=1 or row.get('length')!=512 or row.get('parity',{}).get('status')!='passed':continue
        key=(row['family'],row['size'])
        if key not in chosen or row['seconds_per_target']<chosen[key]['seconds_per_target']:chosen[key]=row
    if len(chosen)!=16:raise ValueError('Complete four-family, four-size profile grid required')
    return chosen


def freeze_file(value,path):
    value=json.loads(json.dumps(value));path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen selection differs')
    else:atomic_json(value,path)
    return value


def build(root,phase):
    root=Path(root).resolve();p=profiles(root/'profile.json');data=str(root/'fineweb');jobs=[]
    def job(f,i,seed,targets,lr,seconds,name,phase):
        row=p[f,i]
        return {'action':'train','family':f,'size':i,'seed':seed,'targets':targets,'lr':lr,
                'chunk':row['chunk'],'compiled':row['compiled'],'checkpointing':row['checkpointing'],'microbatch':1,'context':512,
                'seconds':max(1,seconds-45),'reservation_seconds':seconds,'phase':phase,
                'data':data,'out':str(root/name)}
    if phase=='calibration':
        # Equal data and candidate opportunities; fail if even 32 updates do not fit.
        available=3600;targets=2**18
        estimate=3*sum(r['seconds_per_target']*targets+r['warmup_seconds']+45 for r in p.values())*1.2
        if estimate>available:raise RuntimeError('Minimum LR calibration grid exceeds 1h; optimize first')
        for f,i in p:
            for lr in (.0003,.001,.003):
                seconds=1.2*(p[f,i]['seconds_per_target']*targets+p[f,i]['warmup_seconds']+45)
                jobs.append(job(f,i,1701,targets,lr,seconds,f'calibration/{f}_{i}_{lr}',phase))
    else:
        selected={}
        for f,i in p:
            rows=[]
            for lr in (.0003,.001,.003):
                path=root/f'calibration/{f}_{i}_{lr}/result.json'
                result=json.loads(path.read_text())
                if result['status']!='completed' or result['best_valid_bpb'] is None:raise ValueError('Calibration incomplete; cannot select')
                rows.append((result['best_valid_bpb'],lr))
            selected[f'{f}_{i}']=min(rows)[1]
        freeze_file({'source':source_identity(),'learning_rates':selected},root/'selection.json')
        if phase=='matched_data':
            budget=choose_budget({k:r['seconds_per_target'] for k,r in p.items()},
                                 {k:r['warmup_seconds']+45 for k,r in p.items()})
            if 1 not in budget['sizes']:raise RuntimeError('Replicated grid cannot supply the required 3.33M joint teachers within budget; audit allocation before training')
            freeze_file(budget,root/'matched_data_budget.json')
            for i in budget['sizes']:
                for f in FAMILIES:
                    for seed in SEEDS:
                        seconds=1.2*(budget['targets']*p[f,i]['seconds_per_target']+p[f,i]['warmup_seconds']+45)
                        jobs.append(job(f,i,seed,budget['targets'],selected[f'{f}_{i}'],seconds,f'matched_data/{f}_{i}_{seed}',phase))
        elif phase=='matched_time':
            for f in FAMILIES:
                for seed in SEEDS:
                    estimate=max(8192,int(600/p[f,2]['seconds_per_target']))
                    planned=2**math.ceil(math.log2(estimate))
                    j=job(f,2,seed,planned,selected[f'{f}_2'],600,f'matched_time/{f}_2_{seed}',phase)
                    j.update(eval_every=100,eval_bytes=65536);jobs.append(j)
        else:raise ValueError('Unknown phase')
    return freeze_file({'phase':phase,'source_sha256':source_identity(),'jobs':jobs},root/(phase+'_jobs.json'))
