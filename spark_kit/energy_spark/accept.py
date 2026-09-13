"""Run local DGX Spark acceptance. CPU mode is explicitly non-authorizing.

Checks first; optional bounded fit benchmarks second. CUDA execution is required
for a PASS_FOR_BOUNDED_DEVICE_PILOT receipt. No remote hardware is accessed.
"""
from __future__ import annotations
import argparse, contextlib, copy, json, math, os, statistics, subprocess, sys, time, traceback
from dataclasses import asdict,replace
from pathlib import Path
import torch
import torch.nn.functional as F
from .model import ScaleLM,ScaleConfig,transported_scan,detached_state,state_bytes
from .reference_model import ScaleLM as ReferenceLM,ScaleConfig as ReferenceConfig
from .common import ROOT,atomic_json,atomic_checkpoint,configure,environment,metrics,source_identity,sha256,sync,Budget

def relative(a,b):
    return float((a.float()-b.float()).norm()/(b.float().norm()+1e-12))

def load_native(device='cpu'):
    path=ROOT/'checkpoints/v21_energy_973k.pt';d=torch.load(path,map_location='cpu',weights_only=True)
    c=ScaleConfig(kind='energy',vocab=256,width=d['width'],layers=d['layers'],heads=4,slots=16,ff_width=d['ff_width'],chunk=128,cpu_scan=False)
    m=ScaleLM(c);m.load_state_dict(d['weights']);return m.to(device)

def state_tensors(s):return [t for layer in s for t in layer]

def forward_audit(device):
    m=load_native(device).eval();torch.manual_seed(22041)
    x=torch.randint(0,256,(2,49),device=device)
    with torch.no_grad():
        out,s=m(x);chunks=[];ss=None
        for i in range(x.shape[1]):
            y,ss=m(x[:,i:i+1],ss);chunks.append(y)
        serial=torch.cat(chunks,1);err=float((out-serial).abs().max())
        changed=x.clone();changed[:,30:]=torch.randint(0,256,changed[:,30:].shape,device=device)
        yy,_=m(changed);future=float((out[:,:30]-yy[:,:30]).abs().max())
        reset=torch.zeros_like(x,dtype=torch.bool);reset[:,24]=True
        split,_=m(x,reset=reset);fresh,_=m(x[:,24:]);reset_err=float((split[:,24:]-fresh).abs().max())
        frozen=[t.clone() for t in state_tensors(s)];_,_ =m(x[:,:1],s)
        no_mutation=all(torch.equal(a,b) for a,b in zip(frozen,state_tensors(s)))
        prior=min(float(layer[0].min()) for layer in s)
        raw_bytes=state_bytes(s)//x.shape[0]
        fp32=all(t.dtype==torch.float32 for t in state_tensors(s))
    assert err<2e-3 and future<1e-6 and reset_err<2e-3 and no_mutation and prior>=.25-1e-5 and fp32
    return {'stream_max_abs':err,'future_difference':future,'reset_max_abs':reset_err,'state_nonmutating':no_mutation,
            'precision_min':prior,'fp32_state':fp32,'raw_bytes_per_sequence':raw_bytes}

def scan_audit(device):
    torch.manual_seed(22042)
    f=(torch.rand(2,13,2,5,device=device)*.9).requires_grad_();b=torch.randn(2,13,2,5,4,device=device,requires_grad=True)
    ini=torch.randn(2,2,5,4,device=device,requires_grad=True);strides=torch.tensor([1,3],device=device)
    y=transported_scan(f,b,ini,strides);coeff=torch.randn_like(y);g=torch.autograd.grad((y*coeff).sum(),(f,b,ini))
    ff=f.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_();ii=ini.detach().clone().requires_grad_()
    prior=ii;history=[];idx=(torch.arange(5,device=device)[None]-strides[:,None])%5
    for t in range(13):
        old=prior.gather(2,idx[None,:,:,None].expand(2,-1,-1,4));prior=ff[:,t,...,None]*old+bb[:,t];history.append(prior)
    yy=torch.stack(history,1);gg=torch.autograd.grad((yy*coeff).sum(),(ff,bb,ii))
    error=max(float((a-c).abs().max()) for a,c in zip(g,gg));fw=float((y-yy).detach().abs().max())
    assert error<3e-5 and fw<3e-5
    return {'forward_max_abs':fw,'gradient_max_abs':error,'algorithm':'Torch affine scan versus sequential recurrence'}

def checkpoint_parity(device):
    torch.manual_seed(22043);c=ScaleConfig(vocab=256,width=32,layers=2,heads=2,slots=8,ff_width=64,chunk=16,cpu_scan=False)
    m=ScaleLM(c).to(device);mc=ScaleLM(replace(c,activation_checkpointing=True)).to(device);mc.load_state_dict(m.state_dict())
    x=torch.randint(256,(2,37),device=device);target=torch.randint(256,x.shape,device=device)
    outputs=[];grads=[]
    for mod in (m,mc):
        mod.train();y,_=mod(x);loss=F.cross_entropy(y.flatten(0,1),target.flatten());loss.backward()
        outputs.append(y.detach());grads.append(torch.cat([p.grad.flatten() for p in mod.parameters() if p.grad is not None]))
    e=relative(grads[1],grads[0]);assert e<2e-5
    return {'activation_recompute_gradient_relative':e,'logit_max_abs':float((outputs[0]-outputs[1]).abs().max())}

def precision_audit(device):
    cpu=load_native('cpu');gpu=load_native(device)
    torch.manual_seed(22044);x=torch.randint(256,(1,16));target=torch.randint(256,x.shape)
    def run(m,dev,amp):
        m.train();m.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch.device(dev).type,dtype=torch.bfloat16,enabled=amp):y,state=m(x.to(dev))
        loss=F.cross_entropy(y.float().flatten(0,1),target.to(dev).flatten());loss.backward()
        g=torch.cat([p.grad.float().flatten() for p in m.parameters() if p.grad is not None]).cpu()
        assert torch.isfinite(y).all() and torch.isfinite(g).all()
        assert all(t.dtype==torch.float32 for t in state_tensors(state))
        return y.detach().float().cpu(),float(loss.detach()),g
    cy,cl,cg=run(cpu,'cpu',False);gy,gl,gg=run(gpu,device,False)
    fp=relative(gy,cy);grad=relative(gg,cg)
    assert fp<.005 and abs(gl-cl)<.005 and grad<.02
    by,bl,bg=run(gpu,device,True);bc=F.cosine_similarity(bg,gg,dim=0).item()
    br=relative(by,gy);gr=relative(bg,gg)
    # Engineering tolerances, not an accuracy/correctness guarantee.
    assert br<.08 and abs(bl-gl)<.15 and bc>.95 and gr<.3
    return {'cpu_device_fp32_logit_relative':fp,'cpu_device_fp32_gradient_relative':grad,
            'bf16_logit_relative':br,'bf16_loss_absolute':abs(bl-gl),'bf16_gradient_relative':gr,'bf16_gradient_cosine':bc}

def independent_restart(device,out):
    m=load_native(device).eval();torch.manual_seed(22045)
    x=torch.randint(256,(1,17),device=device);new=torch.randint(256,(1,13),device=device)
    with torch.no_grad():_,s=m(x);y,expected=m(new,s)
    # Only coefficients and new tokens leave this process, never the old prompt.
    atomic_checkpoint({'state':detached_state(s),'new_tokens':new.cpu()},out/'restart_input.pt')
    cmd=[sys.executable,'-m','energy_spark.accept','--restart-child',str(out),'--device',str(device)]
    subprocess.run(cmd,check=True,cwd=ROOT,timeout=180)
    answer=torch.load(out/'restart_output.pt',map_location=device,weights_only=True)
    error=float((y-answer['logits']).abs().max());same=all(torch.equal(a,b) for a,b in zip(state_tensors(expected),state_tensors(answer['state'])))
    assert error==0 and same
    return {'fresh_process_max_abs':error,'state_bitwise_identical':same,'previous_prompt_provided':False}

def training_resume(device,out):
    from .data import fixtures
    from .train import train
    corpus=fixtures(out/'resume_fixture');c=json.loads((ROOT/'configs/v22/cpu_test.json').read_text());c['amp']='none'
    # GPU resume is run by a small self-contained child worker, without pretending
    # an unissued acceptance receipt already exists. The worker exercises AdamW,
    # RNG serialization and exact model/optimizer continuation on the real device.
    code_cmd=[sys.executable,'-m','energy_spark.accept','--resume-child',str(out),'--device',str(device)]
    subprocess.run(code_cmd+['--resume-mode','whole'],check=True,cwd=ROOT,timeout=240)
    subprocess.run(code_cmd+['--resume-mode','half'],check=True,cwd=ROOT,timeout=240)
    subprocess.run(code_cmd+['--resume-mode','finish'],check=True,cwd=ROOT,timeout=240)
    a=torch.load(out/'whole.pt',weights_only=True,map_location='cpu');b=torch.load(out/'finish.pt',weights_only=True,map_location='cpu')
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    assert equal(a,b),'Optimizer/model/RNG mismatch after process restart'
    return {'same_device_process_resume_bitwise':True,'updates':6,'scope':'tiny energy model AdamW/RNG; full trainer separately unit-tested'}


def overfit_audit(device):
    """A fixed random-token mapping only checks optimization plumbing, not language quality."""
    torch.manual_seed(22049)
    c=ScaleConfig(vocab=256,width=32,layers=2,heads=2,slots=8,ff_width=64,chunk=16,cpu_scan=False)
    m=ScaleLM(c).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.005,foreach=False)
    x=torch.randint(256,(2,24),device=device);y=torch.randint(256,x.shape,device=device)
    def loss():
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=True):o,_=m(x)
        return F.cross_entropy(o.float().flatten(0,1),y.flatten())
    initial=float(loss().detach())
    for _ in range(24):
        opt.zero_grad(set_to_none=True);l=loss();l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    final=float(loss().detach())
    assert math.isfinite(final) and final<initial-.25
    return {'initial_nll':initial,'final_nll':final,'updates':24,'scope':'fixed synthetic optimization check only'}

def benchmark(device,out,include_large=False):
    from .train import make_model,FIT_FIELDS
    configs=['continuation_973k']+(['pilot_3m','pilot_10m'] if include_large else [])
    rows=[];gen=torch.Generator().manual_seed(22046)
    for name in configs:
        c=json.loads((ROOT/f'configs/v22/{name}.json').read_text());n=c['sequence_length']
        candidates=[]
        for kind in ('energy','softmax'):
            cc={**c,'kind':kind,'init_checkpoint':None,'chunk':128 if kind=='energy' else n}
            if kind=='softmax':
                base=make_model({**c,'init_checkpoint':None}).parameters_count();tmp=make_model(cc).parameters_count()
                cc['ff_width']=max(1,cc['ff_width']+round((base-tmp)/(cc['layers']*(2*cc['width']+1))))
            candidates.append(cc)
        x=torch.randint(256,(1,n),generator=gen).to(device);y=torch.randint(256,(1,n),generator=gen).to(device)
        for c in candidates:
            budget=Budget(device,seconds=600,host_reserve_gib=c['host_reserve_gib'],max_cuda_gib=c['max_cuda_gib'])
            m=make_model(c,device);opt=torch.optim.AdamW(m.parameters(),lr=0.,foreach=False);durations=[]
            if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            for i in range(7):
                budget.check();m.train();opt.zero_grad(set_to_none=True);sync(device);t=time.perf_counter()
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=True):logits,_=m(x)
                loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten());loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();sync(device)
                if i>=2:durations.append(time.perf_counter()-t)
            row={'fit_configuration':{k:c[k] for k in FIT_FIELDS},'config':name,'kind':c['kind'],'parameters':m.parameters_count(),'sequence':n,'microbatch':1,'activation_checkpointing':True,
                 'seconds':durations,'median_update_seconds':statistics.median(durations),'tokens_per_second':n/statistics.median(durations),
                 'memory':metrics(device),'scope':'same random inputs and near-matched parameter counts; not equal-quality language scores'}
            rows.append(row);print(json.dumps(row),flush=True)
            del m,opt,logits,loss
            if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    atomic_json(rows,out/'benchmark.json');return rows

def run(out:Path,device='cuda',cpu_diagnostic=False,include_large=False):
    if not __debug__:raise RuntimeError('Acceptance requires Python assertions; do not use python -O')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'acceptance.json').exists():raise FileExistsError('Use a new acceptance directory; retain prior receipts')
    checks={};fits=[];t0=time.time();env=environment(device)
    if torch.device(device).type=='cuda' and not torch.cuda.is_available():
        r={'status':'NOT_RUN_CUDA_UNAVAILABLE','source_sha256':source_identity(),'environment':env,'checks':{},'completed_unix':time.time()}
        atomic_json(r,out/'acceptance.json');return r
    if torch.device(device).type=='cpu' and not cpu_diagnostic:raise RuntimeError('Use explicit --cpu-diagnostic; CPU cannot authorize a Spark pilot')
    status='CPU_VALIDATION_ONLY' if cpu_diagnostic else 'PASS_FOR_BOUNDED_DEVICE_PILOT'
    try:
        d=configure(device)
        if d.type=='cuda':
            Budget(d,seconds=1800).check()
            if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 device support is required for the supplied pilot')
            if env['machine'] not in ('aarch64','arm64') or 'GB10' not in env.get('gpu_name','').upper():
                status='PASS_OTHER_CUDA_DEVICE_NOT_SPARK' # useful record, but will not unlock a Spark pilot
        for name,fn in [('scan',lambda:scan_audit(d)),('causality_and_state',lambda:forward_audit(d)),
                        ('checkpoint_recompute',lambda:checkpoint_parity(d)),('precision',lambda:precision_audit(d)),
                        ('inference_restart',lambda:independent_restart(d,out)),('training_restart',lambda:training_resume(d,out)),('fixed_batch_learning',lambda:overfit_audit(d))]:
            checks[name]=fn();print('PASS',name,checks[name],flush=True)
        fits=benchmark(d,out,include_large=include_large)
    except Exception as exc:
        status='FAILED';checks['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    result={'status':status,'source_sha256':source_identity(),'environment':env,'checkpoint_sha256':sha256(ROOT/'checkpoints/v21_energy_973k.pt'),
            'checks':checks,'fit_configurations':[r['fit_configuration'] for r in fits],'elapsed_seconds':time.time()-t0,'completed_unix':time.time(),
            'meaning':'Numerical/device permission for a capped pilot only, never a language-quality or large-pretraining approval.'}
    atomic_json(result,out/'acceptance.json');print(json.dumps(result,indent=2));return result

def child_restart(path,device):
    configure(device);m=load_native(device).eval();d=torch.load(path/'restart_input.pt',map_location=device,weights_only=True)
    with torch.inference_mode():out,state=m(d['new_tokens'],d['state'])
    atomic_checkpoint({'logits':out,'state':state},path/'restart_output.pt')

def child_resume(path,device,mode):
    configure(device,seed=22047);c=ScaleConfig(vocab=256,width=32,layers=2,heads=2,slots=8,ff_width=64,chunk=16,cpu_scan=False)
    m=ScaleLM(c).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.002,foreach=False)
    g=torch.Generator().manual_seed(22048);x=torch.randint(256,(2,24),generator=g).to(device);y=torch.randint(256,(2,24),generator=g).to(device)
    start=0;end=3 if mode=='half' else 6
    if mode=='finish':
        data=torch.load(path/'half.pt',weights_only=True,map_location='cpu');m.load_state_dict(data['weights']);opt.load_state_dict(data['optimizer']);torch.set_rng_state(data['rng']);start=3
        if torch.device(device).type=='cuda':torch.cuda.set_rng_state(data['cuda_rng'],device)
    for i in range(start,end):
        opt.zero_grad(set_to_none=True);logits,_=m(x);F.cross_entropy(logits.flatten(0,1),y.flatten()).backward();opt.step()
    atomic_checkpoint({'weights':m.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None,'step':end},path/f'{mode}.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('runs/acceptance'));p.add_argument('--device',default='cuda');p.add_argument('--cpu-diagnostic',action='store_true');p.add_argument('--include-large',action='store_true')
    p.add_argument('--restart-child',type=Path);p.add_argument('--resume-child',type=Path);p.add_argument('--resume-mode',choices=['whole','half','finish']);a=p.parse_args()
    if a.restart_child:child_restart(a.restart_child,a.device)
    elif a.resume_child:child_resume(a.resume_child,a.device,a.resume_mode)
    else:
        r=run(a.out,a.device,a.cpu_diagnostic,a.include_large);sys.exit(0 if r['status'] in ('PASS_FOR_BOUNDED_DEVICE_PILOT','CPU_VALIDATION_ONLY') else 2)
