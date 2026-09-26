"""Resumable, local feature extraction for the actual Spark teacher.

Each atomic shard contains derived features and byte labels, not model updates.
No paid/remote job is launched. CUDA is explicit and uses the repository idle-GPU
guard. Check the same command on CPU fixtures before using a new device stack.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,time,inspect
from . import probe
from energy_real import kernel
import numpy as np
import torch
from .prepare_decoder import extract
from .probe import load_energy_model,checkpoint_hash
from .prepare_enwik8 import block_starts,SPLITS,file_hash
from .experiment import atomic_save


def validate_blocks(blocks,total):
    if type(total) is not int or total < 1:
        raise ValueError('Positive integer corpus size required')
    if not isinstance(blocks, (list, tuple)):
        raise ValueError('Blocks must be a sequence of dictionaries')
    for block in blocks:
        if (not isinstance(block, dict) or not {'split','offset','bytes'} <= set(block)
                or block['split'] not in ('train','validation','test')):
            raise ValueError('Invalid block schema or partition')
    if len(blocks)>3072:raise ValueError('Too many blocks for this bounded study')
    if not blocks or {b['split'] for b in blocks}!={'train','validation','test'}:
        raise ValueError('All three partitions need blocks')
    intervals=[]
    for b in blocks:
        if (type(b['offset']) is not int or type(b['bytes']) is not int or b['offset']<0
                or b['bytes']<18 or b['offset']+b['bytes']>total):
            raise ValueError('Invalid block range')
        intervals.append((b['offset'],b['offset']+b['bytes']))
    intervals.sort()
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Blocks overlap')


def records_digest(records):
    h=hashlib.sha256()
    for row in records:
        h.update(json.dumps({'split':row['split'],'document':row['document']},sort_keys=True).encode())
        for key in ('h','x','y','group','anchor'):
            t=row[key].detach().contiguous().cpu()
            h.update(str((key,str(t.dtype),tuple(t.shape))).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def build(teacher_path,corpus_path,blocks,shards,out,*,normalization=None,device='cpu',batch=2,stride=4,
          max_seconds=900.,resume=False,stop_after_shards=None,source_note='Fixed byte blocks, not article-independent.'):
    if (type(batch) is not int or batch<1 or type(stride) is not int or stride<1
            or type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or max_seconds<=0):
        raise ValueError('Invalid extraction limits')
    if stop_after_shards is not None and (type(stop_after_shards) is not int or stop_after_shards<1):raise ValueError('Invalid shard stop')
    torch.set_num_threads(1);shards=Path(shards);out=Path(out);corpus_path=Path(corpus_path)
    validate_blocks(blocks,corpus_path.stat().st_size)
    if out.exists():raise FileExistsError('Completed dataset already exists; do not overwrite it')
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):raise ValueError('Only CPU/CUDA supported')
    if device.type=='cuda':
        from energy_real.device import require_idle_gpu,set_memory_limit
        require_idle_gpu();set_memory_limit(device,8.)
    teacher=load_energy_model(teacher_path,device)
    norm_json=None if normalization is None else {k:v.cpu().tolist() for k,v in normalization.items()}
    if normalization is not None:
        if set(normalization)!={'center','scale'}:raise ValueError('Invalid normalization')
        if any(v.shape!=(2*teacher.config.heads,) or not torch.isfinite(v).all() for v in normalization.values()) or (normalization['scale']<=0).any():raise ValueError('Invalid normalization')
    identity={'format':'decoder_shards_v1','teacher_sha256':checkpoint_hash(teacher_path),'corpus_sha256':file_hash(corpus_path),
        'blocks':blocks,'batch':batch,'stride':stride,'device':str(device),'torch':str(torch.__version__),'normalization':norm_json,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'tf32_matmul':bool(torch.backends.cuda.matmul.allow_tf32),
        'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        'sources':{p.name:checkpoint_hash(p) for p in (Path(__file__),Path(__file__).with_name('prepare_decoder.py'),
                  Path(probe.__file__),Path(inspect.getfile(type(teacher))),Path(kernel.__file__))}}
    ih=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if resume:
        old=torch.load(shards/'identity.pt',weights_only=True)
        if old!=identity:raise ValueError('Extraction identity changed')
    else:
        shards.mkdir(parents=True,exist_ok=False);atomic_save(identity,shards/'identity.pt',overwrite=False)
    raw=np.memmap(corpus_path,dtype=np.uint8,mode='r');records=[];new_count=0;reused_count=0
    start=time.perf_counter();deadline=start+max_seconds
    for offset in range(0,len(blocks),batch):
        path=shards/f'{offset//batch:06d}.pt'
        if path.exists():
            payload=torch.load(path,weights_only=True,map_location='cpu')
            if payload['identity']!=ih or payload['start']!=offset or records_digest(payload['records'])!=payload['records_sha256']:
                raise ValueError('Corrupted or mismatched feature shard')
            records.extend(payload['records']);reused_count+=1;continue
        if time.perf_counter()>=deadline or (stop_after_shards is not None and new_count>=stop_after_shards):
            return {'status':'partial','new_shards':new_count,'reused_shards':reused_count,'identity':ih}
        docs=[]
        for j,b in enumerate(blocks[offset:offset+batch]):
            text=raw[b['offset']:b['offset']+b['bytes']].tobytes()
            docs.append({'text':text,'sha256':hashlib.sha256(text).hexdigest(),'split':b['split'],
                         'group':offset+j,'offset':b['offset'],'bytes':b['bytes']})
        try:rr=extract(teacher,docs,stride=stride,max_bytes=max(b['bytes'] for b in blocks),batch=batch,deadline=deadline)
        except TimeoutError:
            return {'status':'partial','new_shards':new_count,'reused_shards':reused_count,'identity':ih}
        payload={'identity':ih,'start':offset,'records':rr,'records_sha256':records_digest(rr)}
        atomic_save(payload,path,overwrite=False);records.extend(rr);new_count+=1
        print('published shard',offset//batch,flush=True)
    data={}
    for split in ('train','validation','test'):
        rr=[r for r in records if r['split']==split]
        data[split]={key:torch.cat([r[key] for r in rr]) for key in ('h','x','y','group','anchor')}
    if normalization is None:
        x=data['train']['x'].flatten(0,1)
        normalization={'center':x.mean(0),'scale':x.std(0).clamp_min(.001)}
    normalization={k:v.cpu() for k,v in normalization.items()}
    for split in ('train','validation','test'):data[split]['x']=(data[split]['x']-normalization['center'])/normalization['scale']
    data['weight']=teacher.head.weight.detach().cpu().clone()
    data['metadata']={'format':'decoder_probe_data_v1','teacher_sha256':identity['teacher_sha256'],
        'corpus_sha256':identity['corpus_sha256'],'normalization':normalization,'history':16,'variables':2*teacher.config.heads,
        'horizons':[1,4,16,64],'width':teacher.config.width,'stride':stride,'documents':[r['document'] for r in records],
        'cache_identity':ih,'warning':source_note,'seconds_this_invocation':time.perf_counter()-start}
    atomic_save(data,out,overwrite=False)
    receipt={'status':'completed','new_shards':new_count,'reused_shards':reused_count,'identity':ih,'data_sha256':checkpoint_hash(out),
             'counts':{k:len(data[k]['y']) for k in ('train','validation','test')}}
    (out.parent/(out.name+'.json')).write_text(json.dumps(receipt,indent=2));return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',required=True);p.add_argument('--corpus',required=True);p.add_argument('--data-sha256',required=True)
    p.add_argument('--forecast-data');p.add_argument('--shards',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--train-blocks',type=int,default=64);p.add_argument('--eval-blocks',type=int,default=16);p.add_argument('--block-bytes',type=int,default=512)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--stride',type=int,default=4);p.add_argument('--max-seconds',type=float,default=900.)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-shards',type=int);a=p.parse_args(argv)
    if Path(a.corpus).stat().st_size!=100_000_000 or file_hash(a.corpus)!=a.data_sha256:raise ValueError('Expected hash-verified 100M-byte enwik8')
    if not 1<=a.train_blocks<=2048 or not 1<=a.eval_blocks<=512 or not 80<=a.block_bytes<=4096:raise ValueError('Exceeds bounded cache study')
    normalization=None
    if a.forecast_data:
        meta=torch.load(a.forecast_data,weights_only=True,map_location='cpu')['metadata']
        if meta['teacher_sha256']!=checkpoint_hash(a.teacher) or meta['corpus_sha256']!=a.data_sha256:raise ValueError('Parent forecast identities differ')
        blocks=meta['blocks'];normalization={k:meta[k] for k in ('center','scale')}
    else:
        blocks=[]
        for i,(split,(lo,hi)) in enumerate(SPLITS.items()):
            n=a.train_blocks if split=='train' else a.eval_blocks
            for s in block_starts(lo,hi,a.block_bytes,n,91207+i):blocks.append({'split':split,'offset':int(s),'bytes':a.block_bytes})
    result=build(a.teacher,a.corpus,blocks,a.shards,a.out,normalization=normalization,device=a.device,batch=a.batch,
        stride=a.stride,max_seconds=a.max_seconds,resume=a.resume,stop_after_shards=a.stop_after_shards)
    print(json.dumps(result,indent=2))
    # A partial cache is restartable, but cannot be used for downstream training.
    # Distinct nonzero status keeps `extract && train` from using unfinished data.
    if result.get('status') == 'completed':
        return 0
    if result.get('status') == 'partial':
        return 75
    raise RuntimeError('Extraction returned an unknown status')
if __name__=='__main__':
    raise SystemExit(main())
