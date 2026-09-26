"""Supervised subprocess execution; all GPU stages share one reservation ledger."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from energy_real.runtime import atomic_json
from energy_real.device import gpu_processes,available_host_bytes
from .budget import Ledger
from .spec import source_identity,campaign_spec


def gpu_identity():
    """Query without creating a CUDA context in the supervisor."""
    result=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],
                          capture_output=True,text=True,check=True,timeout=10)
    rows=[line.split(',') for line in result.stdout.splitlines() if line.strip()]
    if len(rows)!=1 or len(rows[0])!=3:raise RuntimeError('Expected one identifiable campaign GPU')
    return dict(zip(('name','uuid','driver'),(v.strip() for v in rows[0])))


def resource_violation(group):
    if available_host_bytes()<16*2**30:return 'host_memory_reserve_exhausted'
    for row in gpu_processes():
        try:owner=os.getpgid(row['pid'])
        except ProcessLookupError:continue
        if owner!=group:return 'foreign_gpu_process_detected'
    return None


def supervise(child,seconds,started):
    deadline=started+max(.01,seconds-30)
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:status,reason='timeout','reservation_deadline';break
        try:
            code=child.wait(timeout=min(5.,remaining))
            return ('completed' if code==0 else 'failed'),None
        except subprocess.TimeoutExpired:
            try:reason=resource_violation(child.pid)
            except Exception as error:reason=f'resource_monitor_failed: {type(error).__name__}: {error}'
            if reason:status='interrupted';break
    # Only our newly-created process group receives a signal; foreign jobs stay intact.
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=min(5.,max(.01,seconds-(time.monotonic()-started))))
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return status,reason


def doctor():
    import platform,torch
    rows=gpu_processes()
    return {'machine':platform.machine(),'torch':str(torch.__version__),'cuda_build':torch.version.cuda,
            'other_compute_processes':rows,'available_host_bytes':available_host_bytes(),
            'status':'WAITING_FOR_EXCLUSIVE_GPU' if rows else 'DEVICE_ACCEPTANCE_REQUIRED',
            'source_sha256':source_identity(),'gpu_identity':gpu_identity()}


def freeze(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'spec.json';spec=campaign_spec()
    # JSON normalizes tuples; compare its canonical representation.
    spec=json.loads(json.dumps(spec))
    if path.exists():
        if json.loads(path.read_text())!=spec:raise ValueError('Frozen campaign identity differs; create a new campaign')
    else:atomic_json(spec,path)
    return spec


def execute(root,stage,seconds,job):
    root=Path(root);freeze(root)
    if stage != {'accept':'acceptance','profile':'acceptance','train':'backbone','evaluate':'verification','forecast':'forecast','joint':'forecast','conversion':'conversion','conversion_evaluate':'conversion','joint_profile':'forecast','joint_evaluate':'forecast','verify':'verification'}.get(job.get('action')):
        raise ValueError('Job action must charge its designated stage')
    if job.get('device','cuda')!='cuda':raise ValueError('GPU supervisor requires CUDA jobs')
    current=doctor()
    if current['other_compute_processes']:
        atomic_json(current,root/'device_status.json');return current
    if current['available_host_bytes']<16*2**30:raise RuntimeError('Host memory reserve unavailable')
    if job['action'] not in ('accept','profile'):
        validate_acceptance(root,job)
    # Budget ownership lasts through the complete child process, including imports.
    ledger=Ledger(root/'budget.json');key=ledger.reserve(stage,seconds,{'source':source_identity(),'job':job})
    folder=root/'jobs'/key;folder.mkdir(parents=True);atomic_json(job,folder/'job.json')
    started=time.monotonic();status='failed';child=None;reason=None
    try:
        with (folder/'output.log').open('w') as output:
            child=subprocess.Popen([sys.executable,'-m','energy_campaign.worker',str(folder/'job.json')],
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            status,reason=supervise(child,seconds,started)
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        elapsed=time.monotonic()-started;ledger.settle(key,elapsed,status)
        atomic_json({'status':status,'reason':reason,'seconds':elapsed,'job':job,'reservation':key},folder/'result.json')
    return {'status':status,'reason':reason,'seconds':elapsed,'folder':str(folder)}


def validate_acceptance(root,job):
    from dataclasses import asdict
    from .spec import config
    acceptance=json.loads((Path(root)/'accept.json').read_text())
    import torch
    if (acceptance.get('status')!='NATIVE_GPU_ACCEPTANCE_PASS' or acceptance.get('source_sha256')!=source_identity()
        or acceptance['runtime'].get('torch')!=str(torch.__version__)
        or acceptance['runtime'].get('cuda')!=torch.version.cuda
        or acceptance['runtime'].get('machine')!=__import__('platform').machine()
        or acceptance['runtime'].get('gpu')!=gpu_identity()['name'] or not acceptance.get('compiled')
        or acceptance.get('gpu_identity')!=gpu_identity()
        or not acceptance.get('restart',{}).get('bitwise_weights_optimizer_rng_samples')):
        raise ValueError('Missing/stale/non-GPU native acceptance')
    profiles=json.loads((Path(root)/'profile.json').read_text())
    if (profiles.get('source_sha256')!=source_identity() or profiles['runtime']['device']!='cuda'
        or profiles['runtime'].get('gpu_identity')!=acceptance['gpu_identity']):raise ValueError('Profile identity mismatch')
    if job['action']=='train':
        cfg=asdict(config(job['family'],job['size'],job.get('chunk',16),job.get('checkpointing',False)))
        if not any(r.get('config')==cfg and r.get('status')!='failed' and r.get('compiled',False)==job.get('compiled',False)
                   and r.get('effective_targets')==8192 and r.get('batch')==job.get('microbatch',1)
                   and r.get('length')==job.get('context',512) and r.get('parity',{}).get('status')=='passed'
                   for r in profiles['rows']):
            raise ValueError('Exact training shape lacks a matching GPU profile')
