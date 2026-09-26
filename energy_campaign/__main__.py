import os
import sys
if "--cpu-diagnostic" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
from pathlib import Path
from .runner import doctor,freeze,execute
from .spec import STAGES


def main():
    p=argparse.ArgumentParser(description='Bounded native energy GPU campaign')
    p.add_argument('action',choices=['doctor','readiness','design','prepare','freeze','accept','profile','train','evaluate','summarize','schedule','forecast','joint','conversion','conversion_evaluate','run','joint_profile','joint_evaluate','verify'])
    p.add_argument('--root',type=Path,default=Path('runs/campaign'));p.add_argument('--job',type=Path)
    p.add_argument('--phase',choices=['calibration','matched_data','matched_time'])
    p.add_argument('--d0',type=int);p.add_argument('--selected-family',choices=['energy','hybrid'])
    p.add_argument('--seconds',type=float);p.add_argument('--stage',choices=list(STAGES));p.add_argument('--cpu-diagnostic',action='store_true')
    a=p.parse_args()
    if a.action=='doctor':result=doctor()
    elif a.action=='readiness':
        from .readiness import audit
        result=audit(a.root)
    elif a.action=='design':
        from .design import design
        result=design(a.d0,a.selected_family)
    elif a.action=='run':
        from .campaign import run
        result=run(a.root)
    elif a.action=='prepare':
        from .data import download
        result=download(a.root/'fineweb');result={k:v for k,v in result.items() if k!='documents'}
    elif a.action=='schedule':
        from .schedule import build
        result=build(a.root,a.phase)
    elif a.action=='freeze':result=freeze(a.root)
    elif a.action=='summarize':
        from .evaluation import summarize
        result=summarize(a.root,a.root/'summary.json')
    else:
        job=json.loads(a.job.read_text()) if a.job else {'action':a.action,'out':str(a.root/(a.action+'.json'))}
        if job.get('action')!=a.action:raise ValueError('Action differs from job')
        if a.cpu_diagnostic:
            if a.action not in ('accept','profile'):raise ValueError('CPU CLI diagnostics are acceptance/profile only')
            from .worker import run
            job['device']='cpu';job['small']=True;result=run(job)
        else:
            if a.seconds is None or a.stage is None:raise ValueError('Explicit GPU stage and reservation seconds required')
            result=execute(a.root,a.stage,a.seconds,job)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
