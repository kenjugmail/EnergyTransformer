"""Bounded forecast extraction using the repository's fixed enwik8 split boundaries.

No text is uploaded. Frozen teacher, train-only normalization, disjoint byte blocks.
Blocks are NOT independently deduplicated Wikipedia articles. CUDA is opt-in and
uses the repository's idle-device guard. No language-model parameters are trained.
"""
import argparse,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .probe import load_energy_model,trace_states,aligned_windows,checkpoint_hash

SPLITS={'train':(0,90_000_000),'validation':(90_000_000,95_000_000),'test':(95_000_000,100_000_000)}

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def block_starts(lo,hi,width,count,seed):
    if min(width,count)<1 or width>hi-lo:raise ValueError('Invalid block budget')
    choices=np.arange((hi-lo)//width)
    if count>len(choices):raise ValueError('Too many disjoint blocks requested')
    return lo+np.sort(np.random.default_rng(seed).choice(choices,size=count,replace=False))*width

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--data-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--train-blocks',type=int,default=128);p.add_argument('--eval-blocks',type=int,default=32)
    p.add_argument('--block-bytes',type=int,default=384);p.add_argument('--batch',type=int,default=4);p.add_argument('--max-seconds',type=float,default=900.)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    if a.block_bytes<80 or a.batch<1 or not 0<a.max_seconds<=3600:raise ValueError('Invalid bounded extraction configuration')
    if a.train_blocks>2048 or a.eval_blocks>512:raise ValueError('Exceeds this bounded pilot; design a larger study explicitly')
    if Path(a.corpus).stat().st_size!=100_000_000:raise ValueError('Expected the fixed 100M-byte enwik8 corpus')
    if file_hash(a.corpus)!=a.data_sha256:raise ValueError('Corpus checksum mismatch')
    if a.device.startswith('cuda'):
        from energy_real.device import require_idle_gpu,set_memory_limit
        require_idle_gpu();set_memory_limit(torch.device(a.device),8.)
    torch.set_num_threads(1);t0=time.perf_counter();teacher=load_energy_model(a.checkpoint,a.device)
    raw=np.memmap(a.corpus,dtype=np.uint8,mode='r');result={};blocks=[]
    for si,(name,(lo,hi)) in enumerate(SPLITS.items()):
        n=a.train_blocks if name=='train' else a.eval_blocks
        starts=block_starts(lo,hi,a.block_bytes,n,8721+si);parts=[]
        for i in range(0,n,a.batch):
            if time.perf_counter()-t0>=a.max_seconds:raise TimeoutError('Extraction budget exhausted; no completed dataset written')
            positions=starts[i:i+a.batch]
            ids=torch.tensor(np.stack([raw[s:s+a.block_bytes].astype(np.int64) for s in positions]),device=a.device)
            trace=trace_states(teacher,ids).cpu()
            for j,s in enumerate(positions):
                x,y,anchors=aligned_windows(trace[j],16,(1,4,16,64))
                group=len(blocks);parts.append((x,y,torch.full((len(x),),group),anchors))
                blocks.append({'split':name,'offset':int(s),'bytes':a.block_bytes})
            print(name,min(i+a.batch,n),n,flush=True)
        result[name]={key:torch.cat([r[j] for r in parts]) for j,key in enumerate(('x','y','group','anchor'))}
    center=result['train']['x'].flatten(0,1).mean(0);scale=result['train']['x'].flatten(0,1).std(0).clamp_min(.001)
    for part in result.values():part['x']=(part['x']-center)/scale;part['y']=(part['y']-center)/scale
    result['metadata']={'format':'energy_forecast_data_v1','history':16,'horizons':[1,4,16,64],'variables':len(center),
        'center':center,'scale':scale,'teacher_sha256':checkpoint_hash(a.checkpoint),'model_config':asdict(teacher.config),
        'corpus_sha256':a.data_sha256,'splits':SPLITS,'blocks':blocks,'seconds':time.perf_counter()-t0,
        'source_note':'User-preserved enwik8. Fixed split byte blocks, not article-isolated. Acquisition provenance not independently audited.'}
    out.parent.mkdir(parents=True,exist_ok=True);torch.save(result,out)
    print('Saved',out,flush=True)
if __name__=='__main__':main()
