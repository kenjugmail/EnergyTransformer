"""Read-only host checks. Does not pull images, change drivers, or start training."""
from __future__ import annotations
import hashlib,json,platform,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b'

def check(root:Path=ROOT,*,require_original=False)->dict:
    root=Path(root);model=root/'checkpoints/v21_energy_973k.pt'
    def command(args):
        if shutil.which(args[0]) is None:return {'available':False}
        try:
            r=subprocess.run(args,capture_output=True,text=True,timeout=8,check=False)
            return {'available':True,'returncode':r.returncode,'output':(r.stdout or r.stderr).strip()[:1200]}
        except subprocess.TimeoutExpired:return {'available':True,'error':'timeout'}
    digest=None
    if model.is_file():
        h=hashlib.sha256()
        with model.open('rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        digest=h.hexdigest()
    mem={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            name,value=line.split(':',1)
            if name in ('MemTotal','MemAvailable'):mem[name]=int(value.split()[0])*1024
    except (OSError,ValueError):pass
    docker=command(['docker','info','--format','{{.ServerVersion}}'])
    gpu=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'])
    issues=[]
    if platform.machine() not in ('aarch64','arm64'):issues.append('not_arm64_spark_host')
    if not docker.get('available') or docker.get('returncode')!=0:issues.append('docker_unavailable')
    if not gpu.get('available') or gpu.get('returncode')!=0:issues.append('nvidia_driver_unavailable')
    artifact_issues=[] if digest==EXPECTED else ['missing_or_mismatched_original_checkpoint']
    if require_original:issues.extend(artifact_issues)
    return {'format':'energy_spark_doctor_v23','machine':platform.machine(),'python':platform.python_version(),
        'docker':docker,'nvidia':gpu,'host_memory':mem,'disk_free_bytes':shutil.disk_usage(root).free,
        'artifact_issues':artifact_issues,'checkpoint_required':require_original,'checkpoint_present':model.is_file(),'checkpoint_matches':digest==EXPECTED,'issues':issues,
        'status':'HOST_CHECKS_PASSED_GPU_NOT_TESTED' if not issues else 'HOST_ACTION_NEEDED',
        'next':'Run bash spark/run.sh accept only after host issues are resolved. This is not CUDA validation.'}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-original',action='store_true');a=p.parse_args()
    r=check(require_original=a.require_original);print(json.dumps(r,indent=2));raise SystemExit(0 if not r['issues'] else 2)
