"""Summarize local reports and optionally profile kernels. No remote telemetry."""
from __future__ import annotations
import argparse,json,time,statistics
from pathlib import Path
import torch
from .common import configure,atomic_json,environment,ROOT,sync,metrics
from .train import make_model,initialize,require_acceptance

def profile(config,out,device,acceptance):
    c=json.loads(config.read_text());d=configure(device)
    if d.type=='cuda':require_acceptance(acceptance,d,c)
    m=make_model(c,d);initialize(m,c);m.train();out.mkdir(parents=True,exist_ok=True)
    x=torch.randint(256,(1,c['sequence_length']),device=d);y=torch.randint(256,x.shape,device=d)
    opt=torch.optim.AdamW(m.parameters(),lr=0.,foreach=False)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=d.type,dtype=torch.bfloat16,enabled=c['amp']=='bfloat16'):o,_=m(x)
        torch.nn.functional.cross_entropy(o.float().flatten(0,1),y.flatten()).backward();opt.step()
    for _ in range(2):step()
    sync(d)
    acts=[torch.profiler.ProfilerActivity.CPU]
    if d.type=='cuda':acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts,record_shapes=True,profile_memory=True) as p:step();sync(d)
    p.export_chrome_trace(str(out/'trace.json'))
    key='self_cuda_time_total' if d.type=='cuda' else 'self_cpu_time_total'
    (out/'kernels.txt').write_text(p.key_averages().table(sort_by=key,row_limit=40))
    atomic_json({'environment':environment(d),'memory':metrics(d),'config':c,'scope':'One profiled training update; not a speed benchmark'},out/'profile.json')

def summarize(root:Path,out:Path):
    root=root.resolve();report={'format':'energy_spark_review_v22','acceptance':[],'runs':[],'serious_pretraining':'NOT_APPROVED_BY_THIS_KIT'}
    for p in root.rglob('acceptance.json'):
        a=json.loads(p.read_text());report['acceptance'].append({'file':str(p.relative_to(root)),'status':a['status'],'environment':a['environment']})
    for p in root.rglob('completed.json'):
        r=json.loads(p.read_text());r['file']=str(p.relative_to(root));report['runs'].append(r)
    report['next_action']='Review actual GPU numerical checks, fit, paired validation curves and training throughput; no automatic model promotion.'
    atomic_json(report,out);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('summarize');a.add_argument('--runs',type=Path,default=Path('runs'));a.add_argument('--out',type=Path,default=Path('runs/SUMMARY.json'))
    a=sub.add_parser('profile');a.add_argument('--config',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--device',default='cuda');a.add_argument('--acceptance',type=Path)
    a=p.parse_args()
    if a.command=='summarize':print(json.dumps(summarize(a.runs,a.out),indent=2))
    else:profile(a.config,a.out,a.device,a.acceptance)
