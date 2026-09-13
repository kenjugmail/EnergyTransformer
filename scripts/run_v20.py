#!/usr/bin/env python3
"""Execute the bounded recorded v20 fits. Does not provision remote hardware."""
from pathlib import Path
import subprocess,sys,os,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
def run(job):
    name,kind,aux,seed=job;out=ROOT/f'results/v20/{name}_seed{seed}'
    log=ROOT/f'results/v20/{name}_seed{seed}.log'
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with log.open('w') as f:
        subprocess.run([sys.executable,'scripts/train_router.py','--kind',kind,'--aux',str(aux),'--seed',str(seed),'--steps','1000','--out',str(out)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    return str(out)
if __name__=='__main__':
    jobs=[(n,k,a,s) for n,k,a in [('hard_aux','hard',1),('hard_noaux','hard',0),('soft_aux','soft',1),('attention_aux','attention',1)] for s in [11,22,33]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for out in pool.map(run,jobs):print('COMPLETE',out,flush=True)
