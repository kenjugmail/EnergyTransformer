"""Device acceptance of the CURRENT native implementation, not the Spark copy."""
from dataclasses import asdict,replace
import copy
import time
import torch
from energy_real.kernel import chunk_gla,read_again
from energy_real.model import Config,LM
from energy_real.runtime import atomic_json,environment
from .spec import source_identity


def relative(a,b):return float((a.float()-b.float()).norm()/(b.float().norm()+1e-12))


def kernel_audit(device):
    rows=[]
    for length in (1,17):
      for chunk in (8,16,32,64):
        torch.manual_seed(913)
        args=[torch.randn(2,length,2,4,device=device) for _ in range(3)]
        u,k,q=args;v=torch.randn(2,length,2,6,device=device);f=torch.rand_like(u).clamp(.0001,.999);s=torch.randn(2,2,4,6,device=device)
        values=[z.detach().requires_grad_() for z in (u,k,v,f.log(),s,q)]
        u,k,v,f,s,q=values
        y,final,ctx=chunk_gla(u,k,v,f,s,chunk);z=read_again(ctx,q)
        loss=y.square().mean()+z.square().mean()+final.square().mean();grad=torch.autograd.grad(loss,values)
        u,k,v,f,s,q=[z.detach().requires_grad_() for z in values];ref=[];ref2=[];state=s
        for i in range(length):
            state=f[:,i,...,None].exp()*state+k[:,i,...,None]*v[:,i,:,None,:]
            ref.append((u[:,i,...,None]*state).sum(-2));ref2.append((q[:,i,...,None]*state).sum(-2))
        a,b=torch.stack(ref,1),torch.stack(ref2,1)
        g=torch.autograd.grad(a.square().mean()+b.square().mean()+state.square().mean(),(u,k,v,f,s,q))
        err=max(float((x-y).abs().max().detach()) for x,y in [(y,a),(z,b),(final,state)])
        ge=max(float((x-y).abs().max()) for x,y in zip(grad,g))
        if err>=3e-5 or ge>=3e-5:raise AssertionError(('serial recurrence mismatch',err,ge))
        rows.append({'length':length,'chunk':chunk,'forward_max_abs':err,'gradient_max_abs':ge})
    return rows


def model_audit(device,kinds=('energy','hybrid','softmax')):
    if not kinds or any(k not in ('energy','hybrid','softmax') for k in kinds):raise ValueError('Unknown model audit family')
    rows=[]
    for kind in kinds:
        torch.manual_seed(914);c=Config(kind=kind,width=32,layers=3,heads=2,slots=4,ff_width=64,chunk=8,attention_window=8 if kind=='hybrid' else 0)
        m=LM(c).to(device);x=torch.randint(256,(2,33),device=device);target=x.roll(-1,1)
        with torch.no_grad():
            y,s=m(x);state=None;ys=[]
            for t in range(33):
                z,state=m(x[:,t:t+1],state);ys.append(z)
            stream=float((y-torch.cat(ys,1)).abs().max())
            altered=x.clone();altered[:,17:]=0;future=float((y[:,:17]-m(altered)[0][:,:17]).abs().max())
            reset=torch.zeros_like(x,dtype=torch.bool);reset[:,17]=True
            reset_error=float((m(x,reset=reset)[0][:,17:]-m(x[:,17:])[0]).abs().max())
        if stream>=.002 or future>=1e-6 or reset_error>=.002:raise AssertionError(('stream/causality/reset',kind,stream,future,reset_error))
        def gradients(model,amp=False):
            model.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=amp):
                y,state=model(x);loss=torch.nn.functional.cross_entropy(y.float().flatten(0,1),target.flatten())
            loss.backward();g=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            if not torch.isfinite(g).all():raise AssertionError('Nonfinite gradients')
            for i,v in enumerate(state):
                if c.mixer_kind(i)=='energy' and (v[0].dtype!=torch.float32 or v[1].dtype!=torch.float32 or (v[0]<c.prior-1e-5).any()):raise AssertionError('Invalid FP32 recurrent state')
            return y.detach(),float(loss.detach()),g.detach()
        y,loss,g=gradients(m);check=LM(replace(c,activation_checkpointing=True)).to(device);check.load_state_dict(m.state_dict());cy,cl,cg=gradients(check)
        cp=relative(cg,g)
        if cp>=2e-5:raise AssertionError('Activation checkpoint gradient mismatch')
        row={'kind':kind,'stream_max_abs':stream,'future_max_abs':future,'reset_max_abs':reset_error,'checkpoint_gradient_relative':cp}
        if torch.device(device).type=='cuda':
            cpu=copy.deepcopy(m).cpu();cpu.zero_grad(set_to_none=True)
            cpu_y,_=cpu(x.cpu());cpu_loss=torch.nn.functional.cross_entropy(cpu_y.flatten(0,1),target.cpu().flatten());cpu_loss.backward()
            cpu_grad=torch.cat([p.grad.flatten() for p in cpu.parameters() if p.grad is not None])
            fp=relative(y.cpu(),cpu_y.detach());fg=relative(g.cpu(),cpu_grad)
            if fp>=.005 or abs(loss-float(cpu_loss.detach()))>=.005 or fg>=.02:raise AssertionError('CPU/device FP32 mismatch')
            row.update(cpu_device_logits_relative=fp,cpu_device_gradient_relative=fg)
            by,bl,bg=gradients(m,True);cos=float(torch.nn.functional.cosine_similarity(g,bg,dim=0));br=relative(by,y);gr=relative(bg,g)
            if br>=.08 or abs(bl-loss)>=.15 or gr>=.3 or cos<=.95:raise AssertionError('BF16 failed inherited engineering tolerances')
            row.update(bf16_logits_relative=br,bf16_gradient_relative=gr,bf16_gradient_cosine=cos)
        rows.append(row)
    return rows


def run(out,device='cpu'):
    torch.set_num_threads(1);start=time.monotonic()
    receipt={'format':'native_campaign_acceptance_v1','source_sha256':source_identity(),'runtime':environment(device),
             'kernel':kernel_audit(device),'models':model_audit(device),'status':'CPU_DIAGNOSTIC_ONLY',
             'scope':'small numerical fixtures; production shape profiling and restart checks required separately'}
    # Do not promote a partial acceptance battery to GPU campaign authorization.
    if torch.device(device).type=='cuda':
        from .runner import gpu_identity
        receipt['gpu_identity']=gpu_identity()
        receipt['compiled']=compilation_audit(device)
        receipt['restart']=restart_audit(device,out)
        receipt['status']='NATIVE_GPU_ACCEPTANCE_PASS'
    else:
        receipt['status']='CPU_DIAGNOSTIC_ONLY'
    receipt['seconds']=time.monotonic()-start;atomic_json(receipt,out);return receipt


def compilation_audit(device):
    rows=[]
    for kind in ('energy','hybrid','softmax'):
        torch.manual_seed(12);c=Config(kind=kind,width=32,layers=3,heads=2,slots=4,ff_width=64,chunk=8,attention_window=8 if kind=='hybrid' else 0)
        model=LM(c).to(device);tokens=torch.randint(256,(2,34),device=device);x,y=tokens[:,:-1],tokens[:,1:]
        def objective(x,y):return torch.nn.functional.cross_entropy(model(x)[0].flatten(0,1),y.flatten())
        loss=objective(x,y);loss.backward();g=torch.cat([p.grad.flatten() for p in model.parameters()]);model.zero_grad(set_to_none=True)
        compiled=torch.compile(objective);cl=compiled(x,y);cl.backward();cg=torch.cat([p.grad.flatten() for p in model.parameters()])
        err=relative(cg,g)
        if abs(float(cl.detach()-loss.detach()))>=.005 or err>=.02:raise AssertionError('Compiled objective mismatch')
        rows.append({'kind':kind,'gradient_relative':err,'loss_abs':abs(float(cl.detach()-loss.detach())),'input_stride':list(x.stride())})
    return rows


def restart_audit(device,out):
    import os,subprocess,sys
    from pathlib import Path
    path=Path(out).parent/'restart';path.mkdir(exist_ok=False)
    for mode in ('whole','half','finish'):
        subprocess.run([sys.executable,'-m','energy_campaign.restart',str(path),mode,str(device)],check=True,
                       env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'},timeout=180)
    a=torch.load(path/'whole.pt',weights_only=True,map_location='cpu');b=torch.load(path/'finish.pt',weights_only=True,map_location='cpu')
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not equal(a,b):raise AssertionError('Fresh-process training resume mismatch')
    return {'bitwise_weights_optimizer_rng_samples':True,'updates':6}
