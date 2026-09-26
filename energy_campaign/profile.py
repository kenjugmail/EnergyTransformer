"""Real-shaped training and inference profiles; CUDA timings are synchronized."""
from dataclasses import asdict,replace
import gc
import statistics
import time
import torch
from energy_real.model import LM
from energy_real.runtime import sync,environment,atomic_json
from .spec import config,FAMILIES,source_identity


def profile_one(cfg,device='cpu',compile_model=False,steps=3,batch=1,length=512,effective_targets=8192):
    if effective_targets%(batch*length):raise ValueError('Profile batch must divide effective targets')
    accumulation=effective_targets//(batch*length)
    torch.manual_seed(991);model=LM(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=.1,foreach=False)
    # Match real overlapping input/target views, including aliasing and stride.
    tokens=torch.randint(256,(batch,length+1),device=device);x,y=tokens[:,:-1],tokens[:,1:]
    def objective(x,y):
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=torch.device(device).type=='cuda'):
            logits,_=model(x);return torch.nn.functional.cross_entropy(logits.float().flatten(0,1),y.flatten())
    loss_fn=torch.compile(objective) if compile_model else objective
    # Production configuration/strides/precision parity, before any optimizer update.
    parity_start=time.monotonic();reference=objective(x,y);reference.backward()
    expected=torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    optimizer.zero_grad(set_to_none=True);actual=loss_fn(x,y);actual.backward()
    observed=torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    relative=float((observed-expected).float().norm()/(expected.float().norm()+1e-12))
    difference=abs(float(actual.detach()-reference.detach()))
    if not torch.isfinite(observed).all() or difference>=.005 or relative>=.02:
        raise AssertionError(f'Production profile gradient parity failed: loss={difference}, gradient={relative}')
    parity={'status':'passed','loss_abs':difference,'gradient_relative':relative,
            'precision':'bf16_outer_fp32_state' if torch.device(device).type=='cuda' else 'fp32',
            'input_stride':list(x.stride()),'seconds':time.monotonic()-parity_start}
    del expected,observed,reference,actual
    def update():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            loss=loss_fn(x,y)/accumulation
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite profile loss')
            loss.backward();float(loss.detach())  # Match trainer's microbatch accounting/synchronization.
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();sync(device)
    if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats()
    t=time.monotonic();update();warmup=time.monotonic()-t;durations=[]
    for _ in range(steps):
        t=time.monotonic();update();durations.append(time.monotonic()-t)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,with_flops=True) as prof:
        update()
    operators=prof.key_averages()
    sdpa=[e.key for e in operators if 'scaled_dot_product' in e.key or 'flash_attention' in e.key]
    counted_flops=sum(e.flops for e in operators)
    row={'sdpa_operators':sdpa,'profiler_counted_flops':counted_flops,
         'flop_scope':'Partial profiler coverage; custom recurrence operations may be missing; not total arithmetic work',
         'config':asdict(cfg),'parameters':model.parameters_count(),'non_embedding_parameters':model.parameters_count()-model.embedding.weight.numel(),
         'compiled':compile_model,'warmup_seconds':warmup+parity['seconds'],'update_seconds':durations,
         'batch':batch,'length':length,'effective_targets':effective_targets,'accumulation':accumulation,'parity':parity,
         'seconds_per_target':statistics.median(durations)/effective_targets,
         'peak_allocated':torch.cuda.max_memory_allocated() if torch.device(device).type=='cuda' else None,
         'peak_reserved':torch.cuda.max_memory_reserved() if torch.device(device).type=='cuda' else None}
    del model,optimizer;gc.collect()
    return row


def run(out,device='cpu',small=False):
    torch.set_num_threads(1);rows=[]
    runtime=environment(device)
    if torch.device(device).type=='cuda':
        from .runner import gpu_identity
        runtime['gpu_identity']=gpu_identity()
    cases=[(f,i,c,a,cp) for i in range(4) for f in FAMILIES for c in ((8,16,32,64) if f in ('energy','hybrid') else (16,)) for a in (False,True) for cp in (False,True)]
    if small:cases=[('hybrid',0,8,False,False)]
    for f,i,c,a,cp in cases:
        cfg=config(f,i,c,cp)
        if small:cfg=replace(cfg,width=32,heads=2,slots=4,layers=3,ff_width=64)
        start=time.monotonic()
        try:row=profile_one(cfg,device,a,batch=1,length=32 if small else 512,effective_targets=64 if small else 8192)
        except (RuntimeError,ValueError) as e:row={'status':'failed','error':str(e)}
        row.update(family=f,size=i,chunk=c,checkpointing=cp,compiled=a,elapsed_seconds=time.monotonic()-start);rows.append(row)
        atomic_json({'source_sha256':source_identity(),'runtime':runtime,'rows':rows,'status':'partial'},out)
    atomic_json({'source_sha256':source_identity(),'runtime':runtime,'rows':rows,'status':'completed'},out)
    return rows
