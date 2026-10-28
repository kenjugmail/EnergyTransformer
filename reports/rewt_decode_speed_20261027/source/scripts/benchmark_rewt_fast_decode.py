"""Same-checkpoint generation timing, including equally optimized controls."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from rewt.fast_decode import FastDecodeSession
from rewt.fast_energy import optimize_energy
from rewt.streaming import load_streaming
from rewt.streaming_decode import StreamingSession


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utilization():
    result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return result.stdout.strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=9)
    a=p.parse_args()
    if a.output.exists():p.error('output exists')
    if a.repetitions<1:p.error('positive repetitions required')
    torch.set_num_threads(1);os.sched_setaffinity(0,{5} if 5 in os.sched_getaffinity(0) else {min(os.sched_getaffinity(0))})
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    variants=['transformer','window_transformer','hybrid','hybrid_energy']
    backends=['reference','eager','compile']+(['graph'] if a.device=='cuda' else [])
    models={};sources={}
    for variant in variants:
        path=ROOT/f'runs/rewt_streaming_20261026/language_{variant}_26301/best.pt'
        model,_=load_streaming(path);model.to(a.device).eval()
        optimize_energy(model,fused=a.device=='cuda' and variant=='hybrid_energy')
        models[variant]=model;sources[variant]=dict(path=str(path),sha256=sha(path))
    torch.manual_seed(271701);prompts={n:torch.randint(256,(1,n),device=a.device) for n in (128,8192)}
    result=dict(device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
                torch=str(torch.__version__),dtype='float32',tf32=False,threads=1,cpu_core=5,
                sources=sources,steps=32,prefill_chunk=128,repetitions=a.repetitions,
                utilization_before=utilization(),samples=[],preparation=[],parity=[],
                scope='Warmed greedy generation of 32 tokens returned to CPU. Same checkpoints and prompts for every backend; identical optimizations applied to all architecture controls. Reference uses the prior sampler (per-token checks and integer copies); new backends validate external prompts once and return all generated IDs together. Preparation, compilation and capture are timed separately; no retraining. Shared machine.')

    def sync():
        if a.device=='cuda':torch.cuda.synchronize()

    @torch.inference_mode()
    def prepare(variant,backend,length):
        started=time.perf_counter();model=models[variant]
        if backend=='reference':session=StreamingSession(model,capacity=length+32)
        else:session=FastDecodeSession(model,capacity=length+32,backend=backend)
        for start in range(0,length,128):scores=session.consume(prompts[length][:,start:start+128])
        sync();prefill=time.perf_counter()-started;started=time.perf_counter()
        if backend!='reference':session.prepare()
        sync();setup=time.perf_counter()-started
        return session,scores,prefill,setup

    @torch.inference_mode()
    def generate(session,scores,backend):
        if backend!='reference':return session.generate(32)[0].cpu().tolist()
        ids=[]
        for _ in range(32):
            token=scores[:,-1].argmax(-1);ids.append(int(token))
            scores=session.consume(token[:,None])
        return ids

    expected={}
    for length in prompts:
        for variant in variants:
            for backend in backends:
                session,scores,prefill,setup=prepare(variant,backend,length)
                ids=generate(session,scores,backend)
                if backend=='reference':expected[variant,length]=ids
                assert ids==expected[variant,length],(variant,backend,length)
                result['parity'].append(dict(variant=variant,backend=backend,prefix=length,equal=True,tokens=ids))
                result['preparation'].append(dict(variant=variant,backend=backend,prefix=length,prefill_seconds=prefill,setup_seconds=setup))
                del session,scores;gc.collect()
                print('warmed',variant,backend,length,round(setup,3),flush=True)
    cases=[(v,b,n) for n in prompts for v in variants for b in backends]
    for repetition in range(a.repetitions):
        for variant,backend,length in (cases if repetition%2==0 else cases[::-1]):
            session,scores,_,_=prepare(variant,backend,length)
            sync();begin=time.perf_counter();cpu=time.process_time()
            ids=generate(session,scores,backend)
            sync();elapsed=time.perf_counter()-begin;process=time.process_time()-cpu
            assert ids==expected[variant,length],(variant,backend,length,repetition)
            result['samples'].append(dict(variant=variant,backend=backend,prefix=length,repetition=repetition,
                                          wall_ms=elapsed*1000,cpu_ms=process*1000,state_bytes=session.allocated_state_bytes()))
            del session,scores
        gc.collect();print('repetition',repetition+1,flush=True)
    result['medians']={}
    for variant,backend,length in cases:
        rows=[r for r in result['samples'] if r['variant']==variant and r['backend']==backend and r['prefix']==length]
        result['medians'][f'{variant}/{backend}/{length}']={k:statistics.median(r[k] for r in rows) for k in ['wall_ms','cpu_ms','state_bytes']}
    result['utilization_after']=utilization()
    result['source_sha256']={name:sha(ROOT/name) for name in ['rewt/streaming.py','rewt/streaming_decode.py','rewt/fast_energy.py','rewt/fast_decode.py','scripts/benchmark_rewt_fast_decode.py']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['medians'],indent=2))


if __name__=='__main__':main()
