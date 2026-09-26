"""Internal one-job entry point, launched only by the budget supervisor."""
import json
from pathlib import Path
import sys
import time
import torch
from energy_real.device import require_idle_gpu,set_memory_limit,available_host_bytes
from energy_real.runtime import atomic_json,sha256
from .spec import source_identity,config


def run(job):
    torch.set_num_threads(1);device=job.get('device','cuda')
    if device=='cuda':require_idle_gpu();set_memory_limit(device,48.)
    action=job['action'];out=Path(job['out'])
    if action=='accept':
        from .acceptance import run
        return run(out,device)
    if action=='profile':
        from .profile import run
        return run(out,device,job.get('small',False))
    if action=='train':
        from energy_real.train import parser,run
        from .data import Documents
        cfg=config(job['family'],job['size'],job.get('chunk',16),job.get('checkpointing',False))
        data=Documents(job['data']);targets=job['targets'];batch=job.get('microbatch',1);seq=job.get('context',512)
        if 8192%(batch*seq):raise ValueError('Microbatch must divide effective batch')
        args=parser().parse_args(['--out',str(out)])
        args.compile=job.get('compiled',False);args.kind=cfg.kind;args.width=cfg.width;args.layers=cfg.layers;args.heads=cfg.heads;args.slots=cfg.slots
        args.ff_width=4*cfg.width;args.chunk=cfg.chunk;args.attention_window=cfg.attention_window
        args.activation_checkpointing=cfg.activation_checkpointing;args.data=job['data'];args.data_sha256=data.file_identity()
        args.batch=batch;args.seq=seq;args.accum=8192//(batch*seq);args.steps=targets//8192
        if args.steps<1 or targets%8192:raise ValueError('Target budget must be a positive multiple of 8192')
        args.max_tokens=targets;args.seed=job['seed'];args.lr=job.get('lr',.001);args.warmup=max(1,args.steps//20)
        args.eval_every=job.get('eval_every',max(1,args.steps//4));args.save_every=max(1,args.steps//16);args.eval_bytes=job.get('eval_bytes',65536)
        args.max_seconds=job['seconds'];args.device=device;args.amp='bfloat16' if device=='cuda' else 'none';args.resume=job.get('resume',False)
        args.campaign_identity={'source_sha256':source_identity(),'job':{k:v for k,v in job.items() if k!='resume'}}
        result=run(args,corpus_factory=Documents)
        allowed=('completed','budget_exhausted') if job.get('phase')=='matched_time' else ('completed',)
        if result['status'] not in allowed or result['best_valid_bpb'] is None:
            raise RuntimeError('Fit did not complete its comparison protocol; checkpoint retained')
        return result
    if action=='evaluate':
        from .evaluation import evaluate_checkpoint
        return evaluate_checkpoint(job['checkpoint'],job['data'],out,device,job.get('max_targets',1048576))
    if action=='joint_profile':
        from .forecast_study import profile_joint
        return profile_joint(job)
    if action=='joint_evaluate':
        from .forecast_study import evaluate_joint
        return evaluate_joint(job)
    if action=='verify':
        from .verification import run
        return run(job)
    if action=='forecast':
        from .forecast_study import head_study
        return head_study(job)
    if action=='joint':
        from .forecast_study import joint_train
        result=joint_train(job)
        receipt=json.loads((Path(job['out'])/'result.json').read_text())
        if receipt['status']!='completed':raise RuntimeError('Joint arm incomplete; paired comparison cannot proceed')
        return result
    if action=='conversion':
        from .conversion_study import train
        result=train(job)
        if result.get('selection_loss') is None:raise RuntimeError('Conversion produced no validated checkpoint')
        return result
    if action=='conversion_evaluate':
        from .conversion_study import evaluate
        return evaluate(job)
    raise ValueError(f'Unknown worker action {action}')

if __name__=='__main__':
    job=json.loads(Path(sys.argv[1]).read_text());run(job)
