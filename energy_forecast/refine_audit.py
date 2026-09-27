"""CPU numerical, geometry, and component-cost audits; no quality tuning."""
from pathlib import Path
import argparse,json,copy,time,random
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .refine_model import load_refinement
from .probe import checkpoint_hash
from .refine_train import per_example

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--fits',required=True)
    p.add_argument('--followup-fits',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    d=torch.load(a.data,weights_only=True,map_location='cpu');paths={}
    for folder in (Path(a.fits),Path(a.followup_fits)):
        for q in folder.glob('*/best.pt'):paths[q.parent.name]=q
    identity={name:checkpoint_hash(p) for name,p in paths.items()};models={n:load_refinement(p)[0] for n,p in paths.items()}
    # Prove final-only decoder projection retains the scored logits.
    parity=[]
    for name,m in models.items():
        h,z,w=d['test']['h'][:128],d['test']['m'][:128],d['weight']
        with torch.no_grad():
            full=m.trajectory(h,z,w,steps=4)[-1];fast=m(h,z,w,steps=4)
        parity.append({'model':name,'max_abs':float((full-fast).abs().max()),'exact':torch.equal(full,fast)})
    if not all(x['exact'] for x in parity):raise AssertionError('Final-only projection changed logits')
    # Curvature is an empirical query-state diagnostic, not a fractal certificate.
    geometry=[]
    for name,m in models.items():
        if not m.config.energy:continue
        with torch.no_grad():
            h=d['test']['h'][:32];memory=d['test']['m'][:32];fixed=m.landscape(m.context_input(h,memory))
            z=torch.zeros(len(h),m.config.latent);gain=m.amplitude if m.config.kind=='energy_full' else 1.
            for step in (0,4,16):
                if step:
                    for _ in range(step-(0 if step==4 else 4)):z=z-m.config.step_size*m.gradient(z,fixed)
                aa,bb=fixed;sig=torch.sigmoid(F.linear(z,m.directions)+bb)
                curvature=aa*sig*(1-sig)*gain*(4/m.units**.5)
                H=torch.eye(m.config.latent)[None]-torch.einsum('ji,bj,jk->bik',m.directions,curvature,m.directions)
                minimum=torch.linalg.eigvalsh(H)[:,0];grad=m.gradient(z,fixed).square().mean(-1).sqrt()
                geometry.append({'model':name,'step':step,'queries':len(h),'negative_curvature':int((minimum<0).sum()),
                    'small_gradient_negative_curvature':int(((minimum<0)&(grad<1e-3)).sum()),
                    'min_eigenvalue':float(minimum.min()),'mean_gradient_rms':float(grad.mean())})
    cases={name.replace('_seed11',''):models[name] for name in models if name.endswith('_seed11')}
    h=d['train']['h'][:128];memory=d['train']['m'][:128];weight=d['weight'];target=d['train']['y'][:128]
    timings=[];rng=random.Random(2799)
    # component-only measured with identical inputs, after all fits finished.
    for repeat in range(7):
        order=list(cases);rng.shuffle(order)
        for kind in order:
            m=cases[kind]
            for mode in ('inference_batch1','training_batch128'):
                train=mode.startswith('training');model=copy.deepcopy(m)
                optimizer=torch.optim.AdamW(model.parameters(),lr=0.,foreach=False) if train else None
                def call():
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        loss=torch.stack([F.cross_entropy(v,target) for v in model.trajectory(h,memory,weight)]).mean()
                        loss.backward();optimizer.step()
                    else:
                        with torch.no_grad():model(h[:1],memory[:1],weight,steps=4)
                for _ in range(3):call()
                n=30 if train else 100;t=time.perf_counter()
                for _ in range(n):call()
                timings.append({'kind':kind,'mode':mode,'repeat':repeat,'milliseconds':1000*(time.perf_counter()-t)/n})
    pd.DataFrame(timings).to_csv(out/'timing_raw.csv',index=False)
    pd.DataFrame(timings).groupby(['kind','mode']).agg(median_ms=('milliseconds','median'),min_ms=('milliseconds','min'),max_ms=('milliseconds','max')).reset_index().to_csv(out/'timing_summary.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'curvature.csv',index=False)
    if identity!={name:checkpoint_hash(p) for name,p in paths.items()}:raise AssertionError('Audit changed checkpoint bytes')
    (out/'audit.json').write_text(json.dumps({'forward_parity':parity,'checkpoint_hashes_unchanged':True,
        'teacher_unchanged':checkpoint_hash(Path(a.data).parents[2]/'checkpoints/teacher_973k.pt') if (Path(a.data).parents[2]/'checkpoints/teacher_973k.pt').exists() else 'see package manifest',
        'cost_scope':'Adapter only; excludes the frozen teacher and compact energy-read extraction. Training counts intermediate CE and AdamW at lr0.',
        'device':'cpu','threads':1,'timing_repetitions':7},indent=2))
    print('AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
