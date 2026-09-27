"""Prepare a nontraining, teacher-bound forecast check from separate documents.

No raw text is exported. Reused data domains are not called untouched benchmarks.
An optional original corpus is hash checked and used to reject exact/shared-64-byte
passages. This does not rule out shorter, semantic, or historical-model overlap.
"""
from pathlib import Path
import argparse,hashlib,json,time
import torch
from .experiment import digest,atomic_save
from .probe import load_energy_model,trace_states,aligned_windows


def prepare(corpus,exclude_corpus,teacher,parent_data,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    base=torch.load(parent_data,weights_only=True,map_location='cpu');meta=base['metadata']
    if digest(teacher)!=meta['teacher_sha256']:raise ValueError('Teacher differs from parent traces')
    if digest(exclude_corpus)!=meta['corpus_sha256']:raise ValueError('Wrong original corpus for overlap audit')
    raw=Path(corpus).read_bytes();original=Path(exclude_corpus).read_bytes()
    docs={hashlib.sha256(b).hexdigest():b for b in raw.splitlines() if len(b)>=meta['history']+max(meta['horizons'])}
    included=[];excluded=[]
    for sha,b in sorted(docs.items()):
        if any(b[i:i+64] in original for i in range(max(0,len(b)-63))):excluded.append(sha)
        else:included.append((sha,b))
    if not included:raise ValueError('No independent-length documents survive overlap check')
    model=load_energy_model(teacher);records=[];manifest=[];start=time.perf_counter()
    for offset in range(0,len(included),10):
        sub=included[offset:offset+10];lengths=[min(len(b),384) for _,b in sub]
        tokens=torch.zeros(len(sub),max(lengths),dtype=torch.long)
        for i,((_,b),n) in enumerate(zip(sub,lengths)):tokens[i,:n]=torch.tensor(list(b[:n]))
        states=trace_states(model,tokens)
        for i,((sha,b),length) in enumerate(zip(sub,lengths)):
            windows=aligned_windows(states[i,:length],meta['history'],tuple(meta['horizons']))
            if windows is None:continue
            x,y,t=windows;gid=100000+len(manifest)
            records.append((x,y,torch.full((len(x),),gid),t))
            manifest.append({'sha256':sha,'bytes_traced':length,'examples':len(x),'group':gid})
        print('secondary traced',offset+len(sub),'of',len(included),flush=True)
    part={k:torch.cat([r[j] for r in records]) for j,k in enumerate(('x','y','group','anchor'))}
    for k in ('x','y'):part[k]=(part[k]-meta['center'])/meta['scale']
    details={**meta,'format':'secondary_forecast_check_v1','source_note':'Separate supplied corpus; no retraining or tuning on these targets',
             'corpus_sha256':hashlib.sha256(raw).hexdigest(),'exclusion_corpus_sha256':digest(exclude_corpus),
             'documents':manifest,'excluded_document_sha256':excluded,'seconds':time.perf_counter()-start,
             'warning':'No shared exact64-byte passage; shorter/semantic overlap and earlier project use remain possible.'}
    out.parent.mkdir(parents=True,exist_ok=True);atomic_save({'test':part,'metadata':details},out)
    report={k:v for k,v in details.items() if not isinstance(v,torch.Tensor)}
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SECONDARY_READY',len(manifest),len(part['x']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','exclude-corpus','teacher','parent-data','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.corpus,a.exclude_corpus,a.teacher,a.parent_data,a.out)
if __name__=='__main__':main()
