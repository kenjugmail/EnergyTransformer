"""Resumable bounded single-device byte-LM training; all memory writes learned.

No remote compute, provisioning, data download, automatic budget increase or
checkpoint promotion. CPU diagnostics cannot unlock a CUDA pilot.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, signal, time
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from .model import ScaleLM, ScaleConfig
from .data import ByteCorpus
from .common import (ROOT, Budget, BudgetStop, atomic_json, atomic_checkpoint, configure, environment,
                     exclusive_run, json_hash, metrics, sha256, source_identity, sync)

FIELDS={'name','kind','vocab','width','layers','heads','slots','ff_width','prior','chunk','sequence_length','microbatch',
        'accumulation','steps','max_tokens','max_seconds','learning_rate','weight_decay','warmup','eval_every','eval_targets',
        'seed','amp','activation_checkpointing','allocator_fraction','max_cuda_gib','host_reserve_gib','init_checkpoint'}

def validate_config(c):
    if set(c)!=FIELDS:raise ValueError(f'Configuration fields differ: {set(c)^FIELDS}')
    for k in ('vocab','width','layers','heads','slots','ff_width','chunk','sequence_length','microbatch','accumulation','steps','max_tokens','max_seconds','warmup','eval_every','eval_targets','seed'):
        if type(c[k]) is not int or c[k]<1:raise ValueError(f'{k} must be a positive integer')
    if c['vocab']!=256 or c['kind'] not in ('energy','softmax'):raise ValueError('This pilot compares 256-byte energy and softmax models only')
    if c['amp'] not in ('none','bfloat16'):raise ValueError('Use float32 or BF16 outer arithmetic')
    for k in ('prior','learning_rate','weight_decay','allocator_fraction','max_cuda_gib','host_reserve_gib'):
        if isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<0:raise ValueError(f'Invalid {k}')
    if c['learning_rate']==0 or c['prior']==0 or not 0<c['allocator_fraction']<=.7:raise ValueError('Invalid rate/prior/allocator fraction')
    if c['sequence_length']>1024 or c['width']>512 or c['layers']>8 or c['steps']>2000 or c['max_tokens']>16_777_216 or c['max_seconds']>8*3600:
        raise ValueError('Outside bounded single-Spark pilot scope; make a separately reviewed plan')
    if c['width']%c['heads'] or (c['width']//c['heads'])%2:raise ValueError('Head count must divide width with an even head dimension')
    if c['slots']>256 or c['ff_width']>4096 or c['microbatch']>16 or c['accumulation']>64 or c['chunk']>1024:raise ValueError('Configuration exceeds bounded per-operation dimensions')
    targets=c['steps']*c['microbatch']*c['accumulation']*c['sequence_length']
    if targets>c['max_tokens']:raise ValueError('Configured updates exceed the token cap')
    if c['init_checkpoint'] is not None and (not isinstance(c['init_checkpoint'],str) or Path(c['init_checkpoint']).is_absolute() or '..' in Path(c['init_checkpoint']).parts):raise ValueError('Initialization must name a bundled relative checkpoint')
    if not isinstance(c['activation_checkpointing'],bool):raise ValueError('activation_checkpointing must be Boolean')
    return c

def make_model(c,device='cpu'):
    cc=ScaleConfig(kind=c['kind'],vocab=c['vocab'],width=c['width'],layers=c['layers'],heads=c['heads'],slots=c['slots'],
                   ff_width=c['ff_width'],prior=c['prior'],chunk=c['chunk'],activation_checkpointing=c['activation_checkpointing'],cpu_scan=False)
    return ScaleLM(cc).to(device)

def initialize(m,c):
    if not c['init_checkpoint']:return None
    p=ROOT/c['init_checkpoint'];d=torch.load(p,map_location='cpu',weights_only=True)
    if d.get('format')!='wiki_pilot_v21' or d['kind']!=c['kind'] or d['width']!=c['width'] or d['layers']!=c['layers'] or d['ff_width']!=c['ff_width']:
        raise ValueError('Initialization dimensions do not match. Larger models start from scratch; this is not automatic growth.')
    m.load_state_dict(d['weights'],strict=True);return sha256(p)

FIT_FIELDS=('kind','width','layers','heads','slots','ff_width','chunk','sequence_length','microbatch','amp','activation_checkpointing')

def require_acceptance(path:Path,device,config=None):
    r=json.loads(path.read_text())
    if r.get('status')!='PASS_FOR_BOUNDED_DEVICE_PILOT' or r.get('source_sha256')!=source_identity():raise ValueError('Need passing current-source GPU acceptance receipt')
    if r.get('checkpoint_sha256')!=sha256(ROOT/'checkpoints/v21_energy_973k.pt'):
        raise ValueError('Bundled warm-start checkpoint changed since acceptance')
    now=environment(device);prev=r['environment']
    for k in ('machine','torch','cuda_runtime','gpu_name','capability','image_digest'):
        if prev.get(k)!=now.get(k):raise ValueError(f'Device/runtime acceptance mismatch: {k}')
    completed=r.get('completed_unix')
    if isinstance(completed,bool) or not isinstance(completed,(int,float)) or not math.isfinite(completed):raise ValueError('Invalid acceptance timestamp')
    age=time.time()-completed
    if age < -300 or age>7*86400:raise ValueError('Acceptance receipt is future-dated or more than seven days old; rerun it')
    if config is not None:
        wanted={k:config[k] for k in FIT_FIELDS}
        if wanted not in r.get('fit_configurations',[]):raise ValueError('This exact architecture/context/microbatch was not fit-tested; rerun acceptance --include-large')
    return r

@torch.no_grad()
def evaluate(m,corpus,c,device):
    m.eval();nll=0.;n=0
    for x,y in corpus.validation_windows(c['sequence_length'],c['eval_targets']):
        x=x.to(device);y=y.to(device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=c['amp']=='bfloat16'):
            out,_=m(x)
        loss=F.cross_entropy(out.float().flatten(0,1),y.flatten(),reduction='sum')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
        nll+=float(loss);n+=y.numel()
    if n<1:raise ValueError('No validation targets')
    return {'bpb':nll/n/math.log(2),'targets':n}


def reconcile_step_log(directory:Path,saved_step:int)->dict:
    """Keep committed log entries; preserve crash-tail bytes in a separate file.

    A last.pt checkpoint is authoritative. Log writes can reach disk after the
    last checkpoint; replay must not silently count those updates twice.
    """
    directory=Path(directory);path=directory/'steps.jsonl'
    if not path.exists():return {'retained':0,'recovered_tail':False}
    lines=path.read_bytes().splitlines(keepends=True);keep=[];tail=[];previous=0;in_tail=False
    for line in lines:
        try:
            row=json.loads(line);step=row['step']
            if type(step) is not int or step<=previous:raise ValueError('non-increasing step')
        except (ValueError,TypeError,KeyError) as exc:
            if previous<saved_step:raise ValueError('Corrupt committed training log') from exc
            in_tail=True;tail.append(line);continue
        if step<=saved_step and not in_tail:
            keep.append(line);previous=step
        else:
            in_tail=True;tail.append(line)
    if tail:
        import tempfile
        fd,backup=tempfile.mkstemp(prefix='recovered_steps.',suffix='.jsonl',dir=directory)
        with os.fdopen(fd,'wb') as f:f.write(b''.join(tail));f.flush();os.fsync(f.fileno())
        fd,tmp=tempfile.mkstemp(prefix='steps.',suffix='.tmp',dir=directory)
        try:
            with os.fdopen(fd,'wb') as f:f.write(b''.join(keep));f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:Path(tmp).unlink(missing_ok=True)
    return {'retained':len(keep),'recovered_tail':bool(tail)}

def train(config:dict,data:Path,out:Path,*,device='cpu',resume=False,stop_after=None,acceptance:Path|None=None,cpu_diagnostic=False):
    c=validate_config(dict(config));d=configure(device,seed=c['seed'],fraction=c['allocator_fraction'])
    if d.type=='cuda':
        if acceptance is None:raise ValueError('A hardware acceptance receipt is required')
        require_acceptance(acceptance,d,c)
    elif not cpu_diagnostic:raise RuntimeError('CPU training is an explicit diagnostic only; use --cpu-diagnostic')
    if stop_after is not None and not 0<stop_after<=c['steps']:raise ValueError('Invalid requested stop boundary')
    train_data=ByteCorpus(data,'train');val_data=ByteCorpus(data,'validation')
    out=Path(out)
    with exclusive_run(out):
        last=out/'last.pt'
        if last.exists() and not resume:raise FileExistsError('Run exists: use --resume or a new output directory')
        if resume and not last.exists():raise FileNotFoundError('No last.pt to resume')
        m=make_model(c,d);initial_sha=initialize(m,c)
        opt=torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],foreach=False)
        env=environment(d)
        identity={'config':c,'data_identity':train_data.identity,'initial_sha256':initial_sha,'source_sha256':source_identity(),
                  'runtime':{k:env.get(k) for k in ('torch','machine','cuda_runtime','device','gpu_name','capability','image_digest')}}
        step=0;tokens=0;stream='';best=float('inf');history=[];elapsed=0.;train_seconds=0.
        if resume:
            old=torch.load(last,weights_only=True,map_location='cpu')
            if old.get('format')!='energy_spark_train_v22' or old['identity']!=identity:raise ValueError('Resume identity mismatch')
            m.load_state_dict(old['weights']);opt.load_state_dict(old['optimizer']);torch.set_rng_state(old['rng'])
            if d.type=='cuda':torch.cuda.set_rng_state(old['cuda_rng'],d)
            reconcile_step_log(out,old['step'])
            step=old['step'];tokens=old['tokens'];stream=old['stream'];best=old['best'];history=old['history'];elapsed=old['elapsed_seconds'];train_seconds=old['train_seconds']
        budget=Budget(d,seconds=max(0,c['max_seconds']-elapsed),host_reserve_gib=c['host_reserve_gib'],max_cuda_gib=c['max_cuda_gib'],out=out)
        stop_flag={'value':False}
        def stop_signal(*args):stop_flag['value']=True
        handlers={}
        try:
            for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,stop_signal)
        except ValueError:pass # unit tests may invoke from a thread
        atomic_json({'config':c,'identity':identity,'parameters':m.parameters_count(),'environment':env},out/'run.json')
        start=time.monotonic();last_saved_step=step
        def save_checkpoint():
            nonlocal best,last_saved_step
            score=evaluate(m,val_data,c,d)
            record={'step':step,'tokens':tokens,'validation_bpb':score['bpb'],'validation_targets':score['targets'],
                    'train_seconds':train_seconds,'elapsed_seconds':elapsed+time.monotonic()-start,'memory':metrics(d)}
            history.append(record)
            model={'format':'energy_spark_model_v22','config':asdict(m.config),'weights':m.state_dict(),'step':step,'data_identity':train_data.identity}
            if score['bpb']<best:best=score['bpb'];atomic_checkpoint(model,out/'best.pt')
            atomic_checkpoint({'format':'energy_spark_train_v22','identity':identity,'weights':m.state_dict(),'optimizer':opt.state_dict(),
               'step':step,'tokens':tokens,'stream':stream,'best':best,'history':history,'rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state(d) if d.type=='cuda' else None,'elapsed_seconds':elapsed+time.monotonic()-start,
               'train_seconds':train_seconds},last)
            last_saved_step=step;atomic_json(history,out/'history.json');print(json.dumps(record),flush=True)
        status='COMPLETE'
        try:
            if not resume:save_checkpoint() # initialization eligible for validation selection
            while step<c['steps']:
                budget.check();m.train();opt.zero_grad(set_to_none=True);current=step+1;loss_value=0.;sync(d);t0=time.perf_counter()
                for micro in range(c['accumulation']):
                    x,y=train_data.batch(c['seed'],current,micro,c['microbatch'],c['sequence_length'])
                    raw=x.numpy().tobytes()+y.numpy().tobytes();stream=hashlib.sha256(bytes.fromhex(stream)+raw).hexdigest()
                    x=x.to(d);y=y.to(d)
                    with torch.autocast(device_type=d.type,dtype=torch.bfloat16,enabled=c['amp']=='bfloat16'):
                        logits,state=m(x);loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten())/c['accumulation']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward();loss_value+=float(loss.detach());tokens+=y.numel()
                    del logits,state,loss
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                rate=c['learning_rate']*min(1.,current/c['warmup'])*(.15+.85*.5*(1+math.cos(math.pi*current/c['steps'])))
                for g in opt.param_groups:g['lr']=rate
                opt.step();sync(d);duration=time.perf_counter()-t0;train_seconds+=duration;step=current
                with (out/'steps.jsonl').open('a') as f:
                    f.write(json.dumps({'step':step,'loss':loss_value,'gradient_norm':float(norm),'lr':rate,'tokens':tokens,
                      'seconds':duration,'tokens_per_second':c['microbatch']*c['accumulation']*c['sequence_length']/duration})+'\n')
                if step%c['eval_every']==0 or step==c['steps'] or step==stop_after or stop_flag['value']:save_checkpoint()
                if step==stop_after or stop_flag['value']:status='STOPPED_AT_UPDATE_BOUNDARY';break
            if last_saved_step!=step:save_checkpoint()
        except BudgetStop as exc:
            status='STOPPED_BUDGET';
            if last_saved_step!=step:save_checkpoint()
            atomic_json({'reason':str(exc),'step':step},out/'stop.json')
        except Exception as exc:
            # Do not overwrite the last verified checkpoint after an OOM/partial update.
            atomic_json({'status':'FAILED','exception':type(exc).__name__,'message':str(exc),'last_saved_step':last_saved_step},out/'failure.json')
            raise
        finally:
            for sig,h in handlers.items():signal.signal(sig,h)
        result={'status':status,'step':step,'tokens':tokens,'stream_sha256':stream,'parameters':m.parameters_count(),
                'best_validation_bpb':best,'train_seconds':train_seconds,'source_sha256':source_identity(),'device':str(d),
                'mode':'cpu_diagnostic_only' if cpu_diagnostic else 'bounded_device_pilot','memory':metrics(d)}
        atomic_json(result,out/'completed.json');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--acceptance',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--cpu-diagnostic',action='store_true')
    a=p.parse_args();r=train(json.loads(a.config.read_text()),a.data,a.out,device=a.device,resume=a.resume,stop_after=a.stop_after,acceptance=a.acceptance,cpu_diagnostic=a.cpu_diagnostic);print(json.dumps(r,indent=2))
