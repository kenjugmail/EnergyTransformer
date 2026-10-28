"""Alternating, preallocated-cache CPU timing for the frozen streaming study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.streaming import load_streaming
from rewt.streaming_decode import StreamingSession


def run_decode(model,tokens,prefix,steps):
    begin=time.process_time();wall=time.perf_counter()
    session=StreamingSession(model,capacity=prefix+steps)
    for start in range(0,prefix,128):session.consume(tokens[:,start:min(prefix,start+128)])
    prefill_cpu=time.process_time()-begin;prefill_wall=time.perf_counter()-wall
    begin=time.process_time();wall=time.perf_counter()
    for t in range(prefix,prefix+steps):session.consume(tokens[:,t:t+1])
    return dict(prefill_cpu_seconds=prefill_cpu,prefill_wall_seconds=prefill_wall,
                decode_cpu_seconds=time.process_time()-begin,decode_wall_seconds=time.perf_counter()-wall,
                allocated_state_bytes=session.allocated_state_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=ROOT/'reports/rewt_streaming_20261026');a=p.parse_args()
    plan=json.loads((a.study/'plan.json').read_text())
    if not (a.study/'training_results.json').exists():raise RuntimeError('finish training before isolated timing')
    torch.set_num_threads(1);allowed=os.sched_getaffinity(0);core=5 if 5 in allowed else min(allowed);os.sched_setaffinity(0,{core})
    variants=['transformer','window_transformer','hybrid','hybrid_energy'];models={};sources={}
    for name in variants:
        r=json.loads((a.study/f'language_{name}_26301.json').read_text());model,_=load_streaming(r['checkpoint']);models[name]=model.eval()
        sources[name]=dict(checkpoint=r['checkpoint'],sha256=file_sha256(r['checkpoint']))
    rng=torch.Generator().manual_seed(26999);decode=[];training=[]
    with torch.inference_mode():
        for model in models.values():run_decode(model,torch.randint(256,(1,192),generator=rng),128,32)
        for length in (128,512,2048,8192):
            tokens=torch.randint(256,(1,length+32),generator=rng)
            for repetition in range(5):
                order=variants if repetition%2==0 else variants[::-1]
                for name in order:
                    row=run_decode(models[name],tokens,length,32)
                    decode.append(dict(variant=name,prefix=length,repetition=repetition,**row))
    x=torch.randint(256,(24,128),generator=rng);target=torch.randint(256,(24,128),generator=rng)
    for model in models.values():
        model.train();model.zero_grad(set_to_none=True);logits,_,_=model(x);F.cross_entropy(logits.flatten(0,1),target.flatten()).backward()
    for repetition in range(7):
        for name in (variants if repetition%2==0 else variants[::-1]):
            model=models[name];model.zero_grad(set_to_none=True);begin=time.process_time();wall=time.perf_counter()
            logits,_,_=model(x);F.cross_entropy(logits.flatten(0,1),target.flatten()).backward()
            training.append(dict(variant=name,repetition=repetition,cpu_seconds=time.process_time()-begin,wall_seconds=time.perf_counter()-wall))
    summary={}
    for name in variants:
        summary[name]=dict(training_cpu_seconds=statistics.median(r['cpu_seconds'] for r in training if r['variant']==name),decode={})
        for length in (128,512,2048,8192):
            rows=[r for r in decode if r['variant']==name and r['prefix']==length]
            summary[name]['decode'][length]={k:statistics.median(r[k] for r in rows) for k in ['prefill_cpu_seconds','prefill_wall_seconds','decode_cpu_seconds','decode_wall_seconds','allocated_state_bytes']}
    output=dict(cpu_core=core,torch_version=str(torch.__version__),threads=1,prefill_chunk=128,generated_tokens=32,batch_decode=1,
        training_shape=[24,128],training_scope='forward, cross entropy, backward; optimizer excluded',
        timing_scope='CPU process and wall time; shared machine; no study workers. Preallocated full attention KV and bounded local/delta caches. All models consume the same random input bytes. Median alternating reps; session allocation included in prefill.',
        sources=sources,source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in [ROOT/'rewt/streaming.py',ROOT/'rewt/streaming_decode.py',Path(__file__)]},
        summary=summary,decode_samples=decode,training_samples=training)
    save_json(a.study/'benchmark.json',output);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
