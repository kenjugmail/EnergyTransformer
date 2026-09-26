"""Create document-separated frozen-teacher forecast examples. No text redistributed.
Usage: python -m energy_forecast.prepare --checkpoint ... --corpus ... --out data.pt
Input corpus: one byte document per line (newline separates documents).
"""
from pathlib import Path
import argparse,hashlib,json,time
from dataclasses import asdict
import torch
from .probe import load_energy_model,trace_states,aligned_windows,checkpoint_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--corpus',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-bytes',type=int,default=384);p.add_argument('--history',type=int,default=16);p.add_argument('--horizons',type=int,nargs='+',default=[1,4,16,64]);p.add_argument('--batch',type=int,default=12)
    p.add_argument('--device',default='cpu');p.add_argument('--source-note',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    raw=Path(a.corpus).read_bytes();unique={hashlib.sha256(b).hexdigest():b for b in raw.splitlines() if len(b)>=a.history+max(a.horizons)}
    docs=sorted(unique.items());model=load_energy_model(a.checkpoint,a.device)
    sets={s:[] for s in ('train','validation','test')};manifest=[];t0=time.perf_counter()
    for offset in range(0,len(docs),a.batch):
        sub=docs[offset:offset+a.batch];lengths=[min(len(b),a.max_bytes) for _,b in sub];tokens=torch.zeros(len(sub),max(lengths),dtype=torch.long,device=a.device)
        for i,((digest,b),length) in enumerate(zip(sub,lengths)):
            tokens[i,:length]=torch.tensor(list(b[:length]),device=a.device)
        traces=trace_states(model,tokens).cpu()
        for i,((digest,b),length) in enumerate(zip(sub,lengths)):
            bucket=int(digest[:8],16)%10;split='validation' if bucket==0 else 'test' if bucket==1 else 'train'
            result=aligned_windows(traces[i,:length],a.history,tuple(a.horizons))
            if result is None:continue
            x,y,anchors=result;docid=len(manifest);sets[split].append((x,y,torch.full((len(x),),docid),anchors))
            manifest.append({'sha256':digest,'split':split,'bytes':length,'examples':len(x)})
        print(f'traced {min(offset+a.batch,len(docs))}/{len(docs)} docs',flush=True)
    payload={}
    for split,records in sets.items():
        if not records:raise ValueError('An empty split; provide more documents')
        payload[split]={name:torch.cat([r[j] for r in records]) for j,name in enumerate(('x','y','group','anchor'))}
    # Only training rows determine the center and scale.
    center=payload['train']['x'].reshape(-1,payload['train']['x'].shape[-1]).mean(0)
    scale=payload['train']['x'].reshape(-1,len(center)).std(0).clamp_min(.001)
    for part in payload.values():
        part['x']=(part['x']-center)/scale;part['y']=(part['y']-center)/scale
    payload['metadata']={'source_note':a.source_note,'corpus_sha256':hashlib.sha256(raw).hexdigest(),
        'teacher_sha256':checkpoint_hash(a.checkpoint),'model_config':asdict(model.config),'teacher_parameters':model.parameters_count(),
        'history':a.history,'horizons':a.horizons,'variables':len(center),'center':center,'scale':scale,
        'documents':manifest,'seconds':time.perf_counter()-t0,'format':'energy_forecast_data_v1',
        'warning':'Lossy sketches of fixed teacher states; no general language improvement or unseen-corpus claim.'}
    out.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,out)
    summary={k:{'examples':len(v['x']),'documents':len(v['group'].unique())} for k,v in payload.items() if k!='metadata'}
    Path(str(out)+'.json').write_text(json.dumps({**summary,'seconds':payload['metadata']['seconds'],'teacher_sha256':payload['metadata']['teacher_sha256']},indent=2))
    print(summary,flush=True)
if __name__=='__main__':main()
