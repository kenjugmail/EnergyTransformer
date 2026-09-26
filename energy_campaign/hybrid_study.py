"""Bounded replicated hybrid/attention study on a shared GPU; one existing ledger."""
from pathlib import Path
import json,os,signal,subprocess,sys,time
from .spec import source_identity,config
from .budget import Ledger
from .shared_probe import supervise_shared
from energy_real.device import available_host_bytes,gpu_processes
from energy_real.runtime import atomic_json,sha256

FAMILIES=('softmax','local_softmax','hybrid')


def specification(root):
    return {'source':source_identity(),'corpus_sha256':sha256(Path(root)/'fineweb/manifest.json'),
            'families':FAMILIES,'sizes':[0,1],'seeds':[404,505,606],'steps':128,'targets':1048576,
            'context':512,'effective_targets':8192,'peak_lr':.001,'warmup_updates':8,
            'validation_targets':65536,'test_targets':262144,'max_torch_gib':4,'host_reserve_gib':16,
            'precision_policy':'BF16 outer; FP32 first energy projection and recurrent state',
            'hybrid_energy_projection_fp32':True,'shared_gpu':True,'timing_comparable':False,'automatic_promotion':False,
            'scope':'Independent fixed-LR replicated endpoints; not optimized scaling laws or a throughput benchmark'}


def train_job(root,out,job):
    import torch
    from energy_real.train import parser,run
    from .data import Documents
    cfg=config(job['family'],job['size'],chunk=16)
    args=parser().parse_args(['--out',str(out),'--shared-hybrid-study'])
    args.energy_projection_fp32=job['family']=='hybrid'
    args.kind=cfg.kind;args.width=cfg.width;args.layers=cfg.layers;args.ff_width=4*cfg.width
    args.heads=4;args.slots=16;args.chunk=16;args.attention_window=cfg.attention_window
    args.data=str(Path(root)/'fineweb');args.data_sha256=sha256(Path(args.data)/'manifest.json')
    args.seed=job['seed'];args.steps=job.get('steps',128);args.max_tokens=args.steps*8192
    args.batch=1;args.seq=512;args.accum=16;args.warmup=min(8,max(1,args.steps//8));args.lr=.001
    args.eval_every=max(1,args.steps//4);args.eval_bytes=65536 if args.steps==128 else 2048
    args.eval_batch=1;args.save_every=max(1,args.steps//8);args.max_seconds=500;args.max_cuda_gib=4
    args.device='cuda';args.amp='bfloat16';args.resume=job.get('resume',False);args.stop_after=job.get('stop_after')
    args.campaign_identity={'source':source_identity(),'protocol':specification(root),'deterministic':job.get('deterministic',False)}
    if job.get('deterministic'):torch.use_deterministic_algorithms(True)
    result=run(args,corpus_factory=Documents)
    expected='interrupted' if args.stop_after is not None else 'completed'
    if result['status']!=expected:raise RuntimeError(f'Fit ended {result["status"]}, expected {expected}; retained for audit')


def compare_restart(out):
    import torch
    a=torch.load(Path(out)/'restart_whole/last.pt',map_location='cpu',weights_only=True)
    b=torch.load(Path(out)/'restart_split/last.pt',map_location='cpu',weights_only=True)
    def same(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    keys=('weights','optimizer','torch_rng','cuda_rng','numpy_rng','coverage','step','tokens')
    checks={k:same(a[k],b[k]) for k in keys}
    if not all(checks.values()):raise AssertionError(f'GPU restart mismatch: {checks}')
    # A resumed committed log must contain each completed update exactly once.
    records=[json.loads(s) for s in (Path(out)/'restart_split/steps.jsonl').read_text().splitlines()]
    if [r['step'] for r in records if r.get('event')=='step']!=[1,2,3,4]:raise AssertionError('Duplicate/missing committed update')
    atomic_json({'status':'passed','checks':checks,'committed_updates':[1,2,3,4]},Path(out)/'restart_receipt.json')


def evaluate(root,out):
    import torch
    import numpy as np
    from energy_real.model import LM,Config
    from energy_real.train import evaluate as loss
    from .data import Documents
    out=Path(out);corpus=Documents(Path(root)/'fineweb','test');validation=Documents(Path(root)/'fineweb','valid');paths=sorted((out/'fits').glob('*/best.pt'))
    if len(paths)!=18:raise ValueError('All 18 fits must complete before final test evaluation')
    selection={str(p.relative_to(out)):sha256(p) for p in paths}
    atomic_json(selection,out/'test_selection.json');rows=[]
    for p in paths:
        if available_host_bytes()<16*2**30:raise RuntimeError('Host reserve breached')
        saved=torch.load(p,map_location='cpu',weights_only=True);model=LM(Config(**saved['config'])).cuda().eval();model.load_state_dict(saved['weights'])
        name=p.parent.name;family,size,seed=name.rsplit('_',2);numerics=None
        if family=='hybrid':
            from .hybrid_validation import check_model
            x,y=validation.batch(np.random.default_rng(308),1,512,'cuda')
            numerics=check_model(model,x,y,streaming=int(seed)==404)
            atomic_json(numerics,out/f'trained_numerics_{size}_{seed}.json')
            if not numerics['passed']:raise AssertionError('Trained hybrid failed numerical validation; preserve checkpoint for audit')
            model.zero_grad(set_to_none=True);del x,y
        metrics=loss(model,corpus,512,262144,torch.device('cuda'),batch=1,amp=True)
        if metrics['status']!='completed':raise RuntimeError('Incomplete held-out evaluation')
        fit=json.loads((p.parent/'result.json').read_text())
        rows.append({'family':family,'size':int(size),'seed':int(seed),'checkpoint_sha256':sha256(p),
                     'test_bpb':metrics['bpb'],'test_targets':metrics['targets'],'valid_bpb':fit['best_valid_bpb'],
                     'peak_reserved_bytes':fit['peak_reserved_bytes'],'trained_targets':fit['tokens'],'trained_numerics':numerics})
        atomic_json({'status':'partial','rows':rows},out/'evaluation.json');del model,saved;torch.cuda.empty_cache()
    atomic_json({'status':'completed','rows':rows,'promotion':False,'limits':'Only two sizes and one fixed data budget; no scaling exponent or speed claim'},out/'evaluation.json')


def worker(root,out,job):
    import torch
    from energy_real.device import set_memory_limit
    if available_host_bytes()<20*2**30:raise RuntimeError('Host headroom lost before launch')
    torch.set_num_threads(1);set_memory_limit('cuda',4)
    if job['action']=='validate':
        from .hybrid_validation import run
        result=run(Path(root)/'fineweb',Path(out)/'gpu_validation','cuda',energy_projection_fp32=True)
        if result['status']!='passed':raise AssertionError('Actual-shape validation failed; no training authorized')
    elif job['action']=='train':train_job(root,Path(out)/job['directory'],job)
    elif job['action']=='evaluate':evaluate(root,out)
    else:raise ValueError('Unknown hybrid-study job')


def run(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if available_host_bytes()<20*2**30:return {'status':'WAITING_FOR_HOST_HEADROOM','available_host_bytes':available_host_bytes()}
    cpu=json.loads((root/'cpu_verification.json').read_text())
    if cpu['status']!='passed' or cpu['source_sha256']!=source_identity():raise ValueError('Current CPU receipt required')
    spec=json.loads(json.dumps(specification(root)));p=out/'spec.json'
    if p.exists() and json.loads(p.read_text())!=spec:raise ValueError('Study identity changed')
    if not p.exists():atomic_json(spec,p)
    if (out/'stopped.json').exists():return {'status':'AUDIT_REQUIRED','record':str(out/'stopped.json')}
    progress=out/'progress.json';done=json.loads(progress.read_text()) if progress.exists() else {}
    jobs=[('validation','acceptance',600,{'action':'validate'}),
          ('restart_whole','acceptance',120,{'action':'train','family':'hybrid','size':1,'seed':1701,'steps':4,'deterministic':True,'directory':'restart_whole'}),
          ('restart_half','acceptance',120,{'action':'train','family':'hybrid','size':1,'seed':1701,'steps':4,'deterministic':True,'stop_after':2,'directory':'restart_split'}),
          ('restart_finish','acceptance',120,{'action':'train','family':'hybrid','size':1,'seed':1701,'steps':4,'deterministic':True,'resume':True,'directory':'restart_split'})]
    for size in (0,1):
        for seed in (404,505,606):
            for family in FAMILIES:
                name=f'{family}_{size}_{seed}'
                jobs.append((name,'backbone',540,{'action':'train','family':family,'size':size,'seed':seed,'directory':f'fits/{name}'}))
    jobs.append(('evaluate','verification',600,{'action':'evaluate'}))
    atomic_json({'jobs':jobs,'maximum_new_seconds':11280,'existing_ledger':str(root/'budget.json')},out/'jobs.json')
    ledger=Ledger(root/'budget.json')
    for name,stage,seconds,job in jobs:
        if name in done:continue
        if stage=='backbone' and not (out/'restart_receipt.json').exists():compare_restart(out)
        if available_host_bytes()<20*2**30:return {'status':'WAITING_FOR_HOST_HEADROOM','next_job':name}
        key=ledger.reserve(stage,seconds,{'source':source_identity(),'job':{'action':'shared_hybrid_study','name':name},'details':job})
        folder=out/'jobs'/key;folder.mkdir(parents=True)
        atomic_json({'root':str(root),'out':str(out),'job':job,'foreign_processes':gpu_processes()},folder/'job.json')
        started=time.monotonic();status='failed';reason=None;child=None
        try:
            with (folder/'output.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'-m','energy_campaign.hybrid_study','worker',str(folder/'job.json')],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                status,reason=supervise_shared(child,seconds,started)
        finally:
            if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
            elapsed=time.monotonic()-started;ledger.settle(key,elapsed,status)
            receipt={'status':status,'reason':reason,'seconds':elapsed,'folder':str(folder),'job':job}
            atomic_json(receipt,folder/'result.json')
        if status!='completed':atomic_json(receipt,out/'stopped.json');return {'status':'AUDIT_REQUIRED','name':name,'result':receipt}
        done[name]=receipt;atomic_json(done,progress)
    atomic_json({'status':'completed','source':source_identity(),'promotion':False},out/'completed.json')
    return {'status':'completed','out':str(out)}


if __name__=='__main__':
    if sys.argv[1]=='worker':
        d=json.loads(Path(sys.argv[2]).read_text());worker(d['root'],d['out'],d['job'])
    else:print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
