"""Cache h_t and current compact P/S reads. Targets are ONLY byte t+1.

Reuse the declared document splits; no raw document text is redistributed.
The native data here are development news, not a fresh reasoning benchmark.
"""
from pathlib import Path
import argparse,hashlib,json,time,inspect
import torch
from .probe import load_energy_model,checkpoint_hash
from .refine_model import compact_energy_read
from .experiment import atomic_save

@torch.no_grad()
def extract_refinement(teacher,documents,*,stride=4,max_bytes=384,batch=12):
    if min(stride,max_bytes,batch)<1:raise ValueError('Positive extraction settings required')
    captures={};rows=[];device=next(teacher.parameters()).device
    hook=teacher.norm.register_forward_hook(lambda m,a,o:captures.update(h=o.detach()))
    try:
        for start in range(0,len(documents),batch):
            docs=documents[start:start+batch];lengths=[min(max_bytes,len(d['text'])) for d in docs]
            tokens=torch.zeros(len(docs),max(lengths),dtype=torch.long,device=device)
            for j,d in enumerate(docs):tokens[j,:lengths[j]]=torch.tensor(list(d['text'][:lengths[j]]),device=device)
            state=None;hh=[];mm=[]
            for t in range(tokens.shape[1]):
                _,state=teacher(tokens[:,t:t+1],state)
                if t>=15 and (t-15)%stride==0:
                    hh.append(captures['h'][:,0].cpu().clone())
                    mm.append(compact_energy_read(state).cpu())
            h=torch.stack(hh,1);memory=torch.stack(mm,1)
            for j,d in enumerate(docs):
                anchors=torch.arange(15,lengths[j]-1,stride);n=len(anchors)
                if n:
                    rows.append({'h':h[j,:n], 'm':memory[j,:n],
                        'y':tokens[j,anchors.to(device)+1].cpu(),'anchor':anchors,
                        'group':torch.full((n,),d['group']),
                        'document':{k:v for k,v in d.items() if k!='text'},'split':d['split']})
            print('extracted',min(start+batch,len(documents)),len(documents),flush=True)
    finally:hook.remove()
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',required=True);p.add_argument('--split-manifest',required=True)
    p.add_argument('--corpus',required=True);p.add_argument('--secondary',required=True);p.add_argument('--out',required=True)
    p.add_argument('--batch',type=int,default=12);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--max-bytes',type=int,default=384);args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);t0=time.perf_counter()
    meta=torch.load(args.split_manifest,weights_only=True,map_location='cpu')['metadata']
    if checkpoint_hash(args.teacher)!=meta['teacher_sha256']:raise ValueError('Teacher mismatch')
    raw=Path(args.corpus).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['corpus_sha256']:raise ValueError('Corpus mismatch')
    source={hashlib.sha256(x).hexdigest():x for x in raw.splitlines()}
    docs=[]
    for i,row in enumerate(meta['documents']):
        docs.append({'text':source[row['sha256']], 'sha256':row['sha256'],'group':i,'split':row['split']})
    for text in Path(args.secondary).read_bytes().splitlines():
        if len(text)<18:continue
        digest=hashlib.sha256(text).hexdigest()
        if digest in source:raise ValueError('Secondary exact duplicate')
        docs.append({'text':text,'sha256':digest,'group':len(docs),'split':'secondary'})
    teacher=load_energy_model(args.teacher)
    records=extract_refinement(teacher,docs,stride=args.stride,max_bytes=args.max_bytes,batch=args.batch)
    data={}
    for split in ('train','validation','test','secondary'):
        rs=[r for r in records if r['split']==split]
        data[split]={k:torch.cat([r[k] for r in rs]) for k in ('h','m','y','anchor','group')}
    data['weight']=teacher.head.weight.detach().cpu().clone()
    data['metadata']={'format':'refinement_cache_v1','teacher_sha256':checkpoint_hash(args.teacher),
        'split_manifest_sha256':checkpoint_hash(args.split_manifest),'corpus_sha256':meta['corpus_sha256'],
        'secondary_sha256':checkpoint_hash(args.secondary),'width':teacher.config.width,
        'memory_width':16*teacher.config.heads,'stride':args.stride,'max_bytes':args.max_bytes,
        'memory_center':data['train']['m'].mean(0),'memory_scale':data['train']['m'].std(0).clamp_min(.05),
        'documents':[r['document'] for r in records],
        'sources':{str(Path(q).name):checkpoint_hash(q) for q in [__file__,inspect.getfile(type(teacher)),inspect.getfile(compact_energy_read)]},
        'seconds':time.perf_counter()-t0,'warning':'Repeated project news splits; lossy fixed current-memory queries; no independent reasoning benchmark.'}
    atomic_save(data,out,overwrite=False)
    report={s:{'positions':len(data[s]['y']),'documents':len(data[s]['group'].unique())} for s in ('train','validation','test','secondary')}
    (out.parent/(out.name+'.json')).write_text(json.dumps({'counts':report,'sha256':checkpoint_hash(out),'metadata':{k:v for k,v in data['metadata'].items() if k not in ('documents','memory_center','memory_scale')}},indent=2))
    print(report,flush=True)
if __name__=='__main__':main()
