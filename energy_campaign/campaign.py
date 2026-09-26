"""Sequential, resumable campaign driver. Returns immediately on device contention."""
from pathlib import Path
import json
from energy_real.runtime import atomic_json,sha256
from .runner import doctor,freeze,execute
from .schedule import build
from .budget import Ledger
from .spec import SEEDS,source_identity


def run(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    status=doctor()
    if status['other_compute_processes']:
        atomic_json(status,root/'device_status.json');return status
    stop_path=root/'campaign_status.json'
    if stop_path.exists() and json.loads(stop_path.read_text()).get('status')=='stopped':
        return {'status':'AUDIT_REQUIRED_AFTER_FAILED_JOB','record':str(stop_path)}
    from .readiness import audit
    preparation=audit(root)
    atomic_json(preparation,root/'readiness.json')
    if preparation['status']!='passed':return {'status':'PREPARATION_AUDIT_FAILED','audit':preparation}
    freeze(root)
    verification=root/'cpu_verification.json'
    if not verification.exists():raise ValueError('CPU verification receipt missing')
    checks=json.loads(verification.read_text())
    if checks.get('source_sha256')!=source_identity() or checks.get('status')!='passed':raise ValueError('CPU verification stale or failed')
    if not (root/'fineweb/manifest.json').exists():raise FileNotFoundError('Prepare public corpus first')
    progress=root/'progress.json';done=json.loads(progress.read_text()) if progress.exists() else {}
    def dispatch(name,stage,seconds,job):
        key=__import__('hashlib').sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
        if name in done:
            if done[name]['job_sha256']!=key:raise ValueError('Completed job changed')
            return True
        result=execute(root,stage,seconds,job)
        if result['status']=='WAITING_FOR_EXCLUSIVE_GPU':
            return False  # Contention before launch is resumable, not a failed experiment.
        if result['status']!='completed':
            atomic_json({'status':'stopped','name':name,'result':result},root/'campaign_status.json');return False
        done[name]={'job_sha256':key,'result':result};atomic_json(done,progress);return True
    if not dispatch('accept','acceptance',1800,{'action':'accept','out':str(root/'accept.json')}):return {'status':'stopped_at_acceptance'}
    if not dispatch('profile','acceptance',min(5400,Ledger(root/'budget.json').remaining('acceptance')),{'action':'profile','out':str(root/'profile.json')}):return {'status':'stopped_at_profile'}
    for phase in ('calibration','matched_data','matched_time'):
        plan=build(root,phase)
        for i,j in enumerate(plan['jobs']):
            if not dispatch(f'{phase}_{i}','backbone',j['reservation_seconds'],j):return {'status':'stopped','phase':phase,'index':i}
    # Frozen teacher heads: existing 10.6M energy model, then one hybrid per seed.
    teachers=[('historical',str(Path('/home/kt/Documents/EnergyTransformer/runs/enwik8_energy_10m/best.pt')))]
    teachers += [(str(s),str(root/f'matched_data/hybrid_1_{s}/best.pt')) for s in SEEDS]
    for name,teacher in teachers:
        j={'action':'forecast','phase':'heads','checkpoint':teacher,'data':str(root/'fineweb'),'out':str(root/f'forecast/{name}'),'seconds':850,'steps':600}
        if not dispatch(f'forecast_{name}','forecast',900,j):return {'status':'stopped_at_forecast','teacher':name}
    j={'action':'joint_profile','phase':'joint','checkpoint':str(root/'matched_data/hybrid_1_404/best.pt'),
       'forecast_data':str(root/'forecast/404/data.pt'),'data':str(root/'fineweb'),'out':str(root/'joint_profile.json')}
    if not dispatch('joint_profile','forecast',300,j):return {'status':'stopped_at_joint_profile'}
    rate=json.loads((root/'joint_profile.json').read_text())['seconds_per_update']
    steps=next((2**p for p in range(10,2,-1) if 1.2*(2**p*rate+60)<(9000-300)/9),None)
    if steps is None:return {'status':'joint_minimum_does_not_fit','seconds_per_update':rate}
    from .schedule import freeze_file
    joint_jobs=[]
    for seed in SEEDS:
        for mode in ('continuation','residual','forecast'):
            joint_jobs.append({'action':'joint','phase':'joint','mode':mode,'seed':seed,'steps':steps,'seconds':900,
                              'checkpoint':str(root/f'matched_data/hybrid_1_{seed}/best.pt'),
                              'forecast_data':str(root/f'forecast/{seed}/data.pt'),'data':str(root/'fineweb'),'out':str(root/f'joint/{seed}_{mode}')})
    freeze_file({'jobs':joint_jobs,'profile_source':source_identity()},root/'joint_jobs.json')
    for i,j in enumerate(joint_jobs):
        if not dispatch(f'joint_{i}','forecast',960,j):return {'status':'stopped_at_joint','index':i}
    j={'action':'joint_evaluate','phase':'forecast_evaluate','root':str(root),'data':str(root/'fineweb'),'out':str(root/'joint_evaluation.json')}
    if not dispatch('joint_evaluate','forecast',1800,j):return {'status':'stopped_at_joint_evaluation'}
    base={'action':'conversion','donor':str(root/'donor'),'data':str(root/'fineweb'),'seconds':3540,'steps':1000000}
    for mode in ('transfer','recovery','lora'):
        j={**base,'mode':mode,'phase':mode,'out':str(root/f'conversion/{mode}')}
        if mode=='recovery':j['initial']=str(root/'conversion/transfer/best.pt')
        if not dispatch(f'conversion_{mode}','conversion',3600,j):return {'status':'stopped_at_conversion','phase':mode}
    j={'action':'conversion_evaluate','phase':'conversion_evaluate','out':str(root/'conversion/evaluation.json'),'donor':str(root/'donor'),'data':str(root/'fineweb'),
       'converted':str(root/'conversion/recovery/best.pt'),'lora':str(root/'conversion/lora/best.pt'),'untrained':str(root/'conversion/transfer/untrained.pt')}
    if not dispatch('conversion_evaluate','conversion',3600,j):return {'status':'stopped_at_conversion_evaluation'}
    j={'action':'verify','root':str(root),'data':str(root/'fineweb'),'out':str(root/'verification.json')}
    if not dispatch('verify','verification',7200,j):return {'status':'stopped_at_verification'}
    return {'status':'completed','automatic_promotion':False}
