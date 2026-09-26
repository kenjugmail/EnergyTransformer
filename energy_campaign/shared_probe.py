"""Explicitly authorized shared-device diagnostic, separate from the exclusive pilot.

Charges the same ledger. Never grants production GPU acceptance or speed evidence.
"""
from pathlib import Path
import json,os,signal,subprocess,sys,time
from energy_real.runtime import atomic_json,sha256
from energy_real.device import available_host_bytes,gpu_processes
from .budget import Ledger
from .spec import source_identity,config,FAMILIES
from .runner import gpu_identity


def supervise_shared(child,seconds,start):
    reason=None
    while time.monotonic()-start<seconds-15:
        try:
            rc=child.wait(timeout=min(2.,max(.01,seconds-15-(time.monotonic()-start))))
            return ('completed' if rc==0 else 'failed'),reason
        except subprocess.TimeoutExpired:
            if available_host_bytes()<16*2**30:
                reason='host_memory_reserve_exhausted';break
    if reason is None:reason='diagnostic_time_limit'
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return ('interrupted' if reason=='host_memory_reserve_exhausted' else 'timeout'),reason


def execute(root,out,job,seconds,stage):
    if available_host_bytes()<20*2**30:raise RuntimeError('Need 20 GiB available before shared probe')
    ledger=Ledger(root/'budget.json');identity={'source':source_identity(),'job':job,'shared_device':True,'max_torch_gib':4,'host_reserve_gib':16}
    key=ledger.reserve(stage,seconds,identity);folder=out/key;folder.mkdir(parents=True)
    atomic_json(job,folder/'job.json');start=time.monotonic();child=None;status='failed';reason=None
    try:
        with (folder/'output.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'-m','energy_campaign.shared_probe','worker',str(folder/'job.json')],
                start_new_session=True,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            status,reason=supervise_shared(child,seconds,start)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        elapsed=time.monotonic()-start;ledger.settle(key,elapsed,status)
        result={'status':status,'reason':reason,'seconds':elapsed,'job':job,'log':str(folder/'output.log'),'reservation':key}
        atomic_json(result,folder/'result.json')
    return result


def worker(job):
    import torch
    from energy_real.device import set_memory_limit
    torch.set_num_threads(1)
    if available_host_bytes()<20*2**30:raise RuntimeError('Host headroom lost before GPU initialization')
    set_memory_limit('cuda',4)
    if job['action']=='accept':
        from .acceptance import kernel_audit,model_audit
        result={'kernel':kernel_audit('cuda'),'models':model_audit('cuda'),'status':'SHARED_EAGER_NUMERICAL_PASS',
                'source_sha256':source_identity(),'gpu_identity':gpu_identity(),'torch':str(torch.__version__),
                'gpu_campaign_authorized':False,'compiled':False,'restart_audited':False,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        atomic_json(result,job['out']);return
    from energy_real.train import parser,run
    from .data import Documents
    c=config(job['family'],job['size'],chunk=16)
    args=parser().parse_args(['--out',job['out'],'--shared-gpu-diagnostic'])
    args.kind=c.kind;args.width=c.width;args.layers=c.layers;args.heads=4;args.slots=16
    args.ff_width=4*c.width;args.attention_window=c.attention_window;args.chunk=16
    args.data=job['data'];args.data_sha256=sha256(Path(job['data'])/'manifest.json')
    args.batch=1;args.seq=512;args.accum=16;args.steps=8;args.max_tokens=65536
    args.seed=404;args.lr=.001;args.warmup=1;args.eval_every=2;args.eval_bytes=2048;args.eval_batch=1;args.save_every=2
    args.max_seconds=100;args.max_cuda_gib=4;args.amp='bfloat16';args.device='cuda'
    args.campaign_identity={'source':source_identity(),'scope':'shared preliminary one-seed fit; no performance claim','job':job}
    result=run(args,corpus_factory=Documents)
    if result['status']!='completed':raise RuntimeError('Incomplete preliminary fit; retained, not comparable to completed fits')


def main(root,out):
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    identity={'source_sha256':source_identity(),'gpu_identity':gpu_identity(),'foreign_processes_at_start':gpu_processes(),
              'available_host_bytes_at_start':available_host_bytes(),'torch_allocation_cap_gib':4,'host_reserve_gib':16,
              'seed':404,'updates':8,'targets_per_update':8192,'families':FAMILIES,'sizes':[0,1],
              'ledger':str(root/'budget.json'),'timing_comparable':False,'scope':'preliminary fitting and loss curves; no scaling-law claim'}
    atomic_json(identity,out/'manifest.json');rows=[]
    job={'action':'accept','out':str(out/'accept.json')}
    result=execute(root,out/'jobs',job,300,'acceptance');rows.append(result)
    if result['status']=='completed':
        for size in (0,1):
            for family in FAMILIES:
                job={'action':'train','family':family,'size':size,'data':str(root/'fineweb'),'out':str(out/f'{family}_{size}')}
                result=execute(root,out/'jobs',job,120,'backbone');rows.append(result)
                atomic_json({'status':'running','jobs':rows},out/'summary.json')
                if result['status']!='completed':break
            if result['status']!='completed':break
    summary={'status':'completed' if len(rows)==9 and all(r['status']=='completed' for r in rows) else 'stopped',
             'jobs':rows,'timing_comparable':False,'promotion':False}
    atomic_json(summary,out/'summary.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='worker':worker(json.loads(Path(sys.argv[2]).read_text()))
    else:main(sys.argv[1],sys.argv[2])
