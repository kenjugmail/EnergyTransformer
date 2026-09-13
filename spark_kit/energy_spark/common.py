"""Local-only provenance, bounded resource checks and atomic checkpoints."""
from __future__ import annotations
import contextlib, hashlib, json, math, os, platform, resource, shutil, subprocess, tempfile, time
from pathlib import Path
from typing import Any
import torch

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3

def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def json_hash(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def source_identity()->str:
    files=sorted([*(ROOT/'energy_spark').glob('*.py'),*(ROOT/'spark').glob('*.sh'),*(ROOT/'configs/v22').glob('*.json')])
    return json_hash({p.relative_to(ROOT).as_posix():sha256(p) for p in files})

def atomic_json(value:Any,path:str|Path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        Path(tmp).unlink(missing_ok=True)

def atomic_checkpoint(value:Any,path:str|Path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:Path(tmp).unlink(missing_ok=True)

def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)

def host_memory()->dict:
    out={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            k,v=line.split(':',1)
            if k in ('MemTotal','MemAvailable','SwapTotal','SwapFree'):out[k]=int(v.strip().split()[0])*1024
    except (OSError,ValueError):pass
    return out

def metrics(device)->dict:
    d=torch.device(device);r={'host':host_memory(),'process_max_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024}
    if d.type=='cuda':
        free,total=torch.cuda.mem_get_info(d)
        r.update(cuda_free_reported=free,cuda_total_reported=total,allocated=torch.cuda.memory_allocated(d),reserved=torch.cuda.memory_reserved(d),
                 peak_allocated=torch.cuda.max_memory_allocated(d),peak_reserved=torch.cuda.max_memory_reserved(d))
    return r

def environment(device='cpu')->dict:
    d=torch.device(device)
    r={'python':platform.python_version(),'machine':platform.machine(),'platform':platform.platform(),'torch':str(torch.__version__),
       'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':str(d),'source_sha256':source_identity(),
       'image_digest':os.environ.get('ENERGY_IMAGE_DIGEST','not_container_recorded'),'host':host_memory()}
    if d.type=='cuda' and torch.cuda.is_available():
        props=torch.cuda.get_device_properties(d)
        r.update(gpu_name=props.name,capability=list(torch.cuda.get_device_capability(d)),gpu_total_memory=props.total_memory,
                 bf16_supported=bool(torch.cuda.is_bf16_supported()))
        try:
            r['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,power.draw,temperature.gpu','--format=csv,noheader'],
                       capture_output=True,text=True,timeout=5,check=False).stdout.strip()
        except (OSError,subprocess.TimeoutExpired):r['nvidia_smi']='unavailable'
    return r

def configure(device,*,seed=22,deterministic=True,fraction=.60):
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise RuntimeError('Single-device pilot only; distributed training is not validated')
    d=torch.device(device)
    if d.type not in ('cpu','cuda'):raise ValueError('Only CPU diagnostics and CUDA are supported')
    if d.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: no Spark result can be produced here')
    if not 0<fraction<=.7:raise ValueError('Initial allocator fraction must be >0 and <=0.70')
    torch.set_num_threads(1);torch.manual_seed(seed)
    if d.type=='cuda':
        if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU to this pilot')
        torch.cuda.set_device(d);torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(fraction,d)
        torch.cuda.reset_peak_memory_stats(d)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(deterministic)
    return d

class BudgetStop(RuntimeError):pass

class Budget:
    """Cooperative guard, not a system-wide hard memory or watchdog guarantee."""
    def __init__(self,device,*,seconds:float,host_reserve_gib=16,max_cuda_gib=48,min_disk_gib=2,out:Path|None=None):
        self.device=torch.device(device);self.start=time.monotonic();self.seconds=seconds
        self.reserve=host_reserve_gib*GIB;self.cap=max_cuda_gib*GIB;self.min_disk=min_disk_gib*GIB;self.out=out
    def check(self):
        if time.monotonic()-self.start>self.seconds:raise BudgetStop('Wall-clock budget reached at an update boundary')
        if self.device.type=='cuda':
            mem=host_memory()
            if mem.get('MemAvailable',float('inf'))<self.reserve:raise BudgetStop('Shared host/GPU memory reserve reached')
            if torch.cuda.memory_reserved(self.device)>self.cap:raise BudgetStop('CUDA reserved-memory budget reached')
        if self.out is not None and shutil.disk_usage(self.out).free<self.min_disk:raise BudgetStop('Disk reserve reached')

@contextlib.contextmanager
def exclusive_run(directory:Path):
    """Advisory process lock; file remains but lock is released on process exit."""
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process owns this run directory')
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
