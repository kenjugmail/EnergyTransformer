"""Balanced round-level next-token training, prefill, and decoding measurements."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from .model import LM, Config
from .train import matched_ff
from .runtime import atomic_json, environment, source_identity, sync


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, (tuple,list)): return sum(tensor_bytes(x) for x in value)
    return 0


def balanced_order(arms, round_index):
    # Rotations followed by their reverse balance both position and adjacency.
    n = len(arms)
    order = arms[round_index % n:] + arms[:round_index % n]
    return list(reversed(order)) if (round_index // n) % 2 else order


def make(arm, args, device):
    kind = 'softmax' if arm == 'softmax' else 'energy'
    ff = matched_ff(args.width, args.heads, args.slots, args.ff_width) if kind=='softmax' else args.ff_width
    cfg = Config(kind=kind, width=args.width, layers=args.layers, heads=args.heads,
                 slots=args.slots, ff_width=ff, chunk=args.chunk, backend=args.backend)
    if arm == 'reference_energy':
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'spark_kit'))
        from energy_spark.model import ScaleLM, ScaleConfig
        kw = asdict(cfg)
        for field in ('backend', 'memory_dim', 'memory_capacity', 'attention_window', 'attention_every'):
            kw.pop(field)
        return ScaleLM(ScaleConfig(**kw, cpu_scan=False)).to(device), cfg
    return LM(cfg).to(device), cfg


def measure(model, tokens, mode, args, device):
    amp = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type=='cuda' and args.amp=='bfloat16')
    x,y = tokens[:,:-1],tokens[:,1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False) if mode=='train' else None
    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with amp():
            logits,_=model(x)
            loss=F.cross_entropy(logits.float().flatten(0,1),y.flatten())
        loss.backward(); optimizer.step()
    model.train(mode=='train')
    if mode=='train':
        for _ in range(args.warmup): train_step()
    else:
        with torch.no_grad(), amp():
            for _ in range(args.warmup): model(x)
    sync(device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    state_size = 0
    t0=time.perf_counter()
    if mode=='train':
        for _ in range(args.steps): train_step()
        count=x.numel()*args.steps
    elif mode=='prefill':
        with torch.no_grad(), amp():
            for _ in range(args.steps): _,state=model(x)
        state_size=tensor_bytes(state); count=x.numel()*args.steps
    else:
        with torch.no_grad(), amp():
            logits,state=model(x)
            next_token=logits[:,-1].argmax(-1,keepdim=True)
            sync(device)
            t0=time.perf_counter()  # Excludes prompt prefill.
            for _ in range(args.new_tokens):
                logits,state=model(next_token,state)
                next_token=logits[:,-1].argmax(-1,keepdim=True)
            state_size=tensor_bytes(state)
        count=x.shape[0]*args.new_tokens
    sync(device)
    seconds=time.perf_counter()-t0
    return {'seconds':seconds,'tokens':count,'tokens_per_second':count/seconds,
            'latency_ms_per_token':1000*seconds/count, 'state_bytes':state_size,
            'model_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}


def run(args):
    if any(type(v) is not int or v < 1 for v in [*args.batches,*args.contexts,args.rounds,args.steps,args.warmup,args.new_tokens]):
        raise ValueError('Positive benchmark dimensions and repetition counts required')
    if args.max_seconds <= 0 or args.max_cuda_gib <= 0:
        raise ValueError('Positive time and memory budgets required')
    if args.out.exists(): raise FileExistsError('Benchmark output already exists')
    dev=torch.device(args.device)
    if dev.type=='cuda':
        from .device import require_idle_gpu,set_memory_limit
        require_idle_gpu();set_memory_limit(dev,args.max_cuda_gib)
    arms=['energy','softmax']+(['reference_energy'] if args.include_reference else [])
    rows=[]; started=time.perf_counter()
    result={'environment':environment(dev),'source_sha256':source_identity(),
            'backend':args.backend,'amp':args.amp,'rounds':rows,
            'fallback_count':0, 'note':'Fixed backend; invalid matmul spans are failures, never fallback.'}
    exhausted=False
    profiled=False
    for batch in args.batches:
        for length in args.contexts:
            failed=set()
            for mode in ['train','prefill','decode']:
                for round_index in range(args.rounds):
                    for arm in balanced_order(arms,round_index):
                        if (arm,mode) in failed: continue
                        if time.perf_counter()-started>=args.max_seconds:
                            exhausted=True;break
                        row={'arm':arm,'mode':mode,'batch':batch,'context':length,'round':round_index}
                        model=None
                        try:
                            torch.manual_seed(0)
                            model,cfg=make(arm,args,dev)
                            generator=torch.Generator(device=dev).manual_seed(17)
                            tokens=torch.randint(256,(batch,length+1),generator=generator,device=dev)
                            row.update({'config':asdict(cfg),'parameters':sum(p.numel() for p in model.parameters())})
                            row.update(measure(model,tokens,mode,args,dev))
                            if args.profile and not profiled and arm=='energy' and mode=='train':
                                activities=[torch.profiler.ProfilerActivity.CPU]
                                if dev.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
                                with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
                                    measure(model,tokens,mode,args,dev)
                                trace=args.out.with_suffix('.trace.json')
                                prof.export_chrome_trace(str(trace))
                                result['profile']={'trace':str(trace),'chunk_recurrences_per_forward':args.layers*((length+args.chunk-1)//args.chunk),
                                    'operators':[{'name':e.key,'calls':e.count,'cpu_time_us':e.cpu_time_total}
                                                 for e in prof.key_averages() if e.key in ('aten::item','aten::_local_scalar_dense','aten::cat','aten::einsum','aten::bmm','aten::cumsum')]}
                                profiled=True
                            row['status']='completed'
                            if row['peak_allocated_bytes'] and row['peak_allocated_bytes']>args.max_cuda_gib*2**30:
                                row['status']='memory_budget_exhausted';failed.add((arm,mode))
                        except (torch.OutOfMemoryError,ValueError) as exc:
                            row.update(status='shape_stopped',error=f'{type(exc).__name__}: {exc}')
                            failed.add((arm,mode))
                        finally:
                            del model
                            gc.collect()
                            if dev.type=='cuda':torch.cuda.empty_cache()
                        rows.append(row)
                        atomic_json({**result,'status':'in_progress'},args.out)
                        print(json.dumps(row),flush=True)
                    if exhausted:break
                if exhausted:break
            if exhausted:break
        if exhausted:break
    result['status']='budget_exhausted' if exhausted else 'completed'
    result['elapsed_s']=time.perf_counter()-started
    atomic_json(result,args.out)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--contexts',type=int,nargs='+',default=[512,2048,8192,16384])
    p.add_argument('--batches',type=int,nargs='+',default=[1,8])
    for name,value in [('width',384),('layers',6),('heads',4),('slots',16),('ff-width',1536),
                       ('chunk',64),('rounds',6),('steps',3),('warmup',2),('new-tokens',64)]:
        p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--backend',choices=['dmat','matmul'],default='dmat')
    p.add_argument('--amp',choices=['none','bfloat16'],default='bfloat16')
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-seconds',type=float,default=1800)
    p.add_argument('--max-cuda-gib',type=float,default=48)
    p.add_argument('--include-reference',action='store_true')
    p.add_argument('--profile',action='store_true')
    return p


if __name__=='__main__': run(parser().parse_args())
