"""Frozen-teacher causal caches for a bounded decoder-adapter study.

No future summary is an adapter input. Next-byte targets are label-only. Existing
forecast document splits are preserved. Numeric caches stay local; corpus text is
not copied into release archives. This is not a broad, untouched LM benchmark.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, time
import torch
from .probe import load_energy_model, state_sketch, checkpoint_hash
from .experiment import atomic_save

@torch.no_grad()
def extract(teacher, documents, history=16, stride=4, max_bytes=384, batch=12, deadline=None):
    record=[]; captured={}; device=next(teacher.parameters()).device
    hook=teacher.norm.register_forward_hook(lambda m,a,o:captured.update(h=o.detach()))
    try:
        for start in range(0,len(documents),batch):
            subset=documents[start:start+batch]; lengths=[min(len(d['text']),max_bytes) for d in subset]
            tokens=torch.zeros(len(subset),max(lengths),dtype=torch.long)
            for j,d in enumerate(subset): tokens[j,:lengths[j]]=torch.tensor(list(d['text'][:lengths[j]]))
            tokens=tokens.to(device)
            state=None; sketches=[]; hidden=[]
            for t in range(tokens.shape[1]):
                if deadline is not None and t%32==0 and time.perf_counter()>=deadline:
                    raise TimeoutError('Feature extraction budget exhausted; no completed dataset is published')
                _,state=teacher(tokens[:,t:t+1],state)
                sketches.append(state_sketch(state)); hidden.append(captured['h'][:,0].clone())
            z=torch.stack(sketches,1).cpu(); h=torch.stack(hidden,1).cpu(); tokens=tokens.cpu()
            for j,d in enumerate(subset):
                anchors=torch.arange(history-1,lengths[j]-1,stride)
                if not len(anchors): continue
                past=anchors[:,None]+torch.arange(-history+1,1)[None]
                record.append({'split':d['split'],'h':h[j,anchors], 'x':z[j,past], 'y':tokens[j,anchors+1],
                               'group':torch.full((len(anchors),),d['group']), 'anchor':anchors,
                               'document':{k:v for k,v in d.items() if k!='text'}})
            print(f'traced {min(start+batch,len(documents))}/{len(documents)}',flush=True)
    finally: hook.remove()
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',required=True);p.add_argument('--forecast-data',required=True)
    p.add_argument('--corpus',required=True);p.add_argument('--secondary',required=True);p.add_argument('--out',required=True)
    p.add_argument('--stride',type=int,default=4);p.add_argument('--max-bytes',type=int,default=384);a=p.parse_args()
    torch.set_num_threads(1)
    if Path(a.out).exists():raise FileExistsError(a.out)
    if a.stride<1 or a.max_bytes<18:raise ValueError('Invalid extraction budget')
    parent=torch.load(a.forecast_data,weights_only=True,map_location='cpu');meta=parent['metadata']
    if checkpoint_hash(a.teacher)!=meta['teacher_sha256']:raise ValueError('Teacher differs from forecast dataset')
    raw=Path(a.corpus).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['corpus_sha256']:raise ValueError('Corpus differs from source manifest')
    lookup={hashlib.sha256(s).hexdigest():s for s in raw.splitlines()}
    docs=[]
    for i,d in enumerate(meta['documents']):
        if d['sha256'] not in lookup:raise ValueError('Missing source document')
        docs.append({'text':lookup[d['sha256']],'sha256':d['sha256'],'split':d['split'],'group':i})
    known=set(lookup)
    for s in Path(a.secondary).read_bytes().splitlines():
        digest=hashlib.sha256(s).hexdigest()
        if len(s)<18:continue
        if digest in known:raise ValueError('Duplicate secondary document')
        docs.append({'text':s,'sha256':digest,'split':'secondary','group':len(docs)})
    started=time.perf_counter();teacher=load_energy_model(a.teacher)
    rows=extract(teacher,docs,16,a.stride,a.max_bytes)
    result={}
    for name in ('train','validation','test','secondary'):
        selected=[r for r in rows if r['split']==name]
        result[name]={key:torch.cat([r[key] for r in selected]) for key in ('h','x','y','group','anchor')}
        result[name]['x']=(result[name]['x']-meta['center'])/meta['scale']
    result['weight']=teacher.head.weight.detach().clone()
    result['metadata']={'format':'decoder_probe_data_v1','teacher_sha256':checkpoint_hash(a.teacher),
        'forecast_data_sha256':checkpoint_hash(a.forecast_data),'corpus_sha256':meta['corpus_sha256'],
        'secondary_sha256':checkpoint_hash(a.secondary),'normalization':{k:meta[k] for k in ('center','scale')},
        'history':meta['history'],'variables':meta['variables'],'horizons':meta['horizons'],
        'width':teacher.config.width,'stride':a.stride,'max_bytes':a.max_bytes,
        'documents':[r['document'] for r in rows], 'seconds':time.perf_counter()-started,
        'warning':'Sampled next-byte labels; reused news domains and forecast split. Not a whole-corpus bpb or fresh benchmark.'}
    atomic_save(result,a.out,overwrite=False)
    summary={name:{'origins':len(result[name]['y']),'documents':len(result[name]['group'].unique())} for name in result if name not in ('metadata','weight')}
    Path(a.out+'.json').write_text(json.dumps({'counts':summary,'seconds':result['metadata']['seconds']},indent=2))
    print(summary,flush=True)
if __name__=='__main__':main()
