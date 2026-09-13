"""Token-only editing task and validated local-text ingestion.

No separate operation flags, key IDs, or parser outputs reach the model. SET,
DELETE, NOISE and ASK are ordinary token IDs. This is a synthetic symbolic language,
NOT human natural language or RULER. Query target labels enter the loss only.
"""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
import torch

PAD,BOS,SET,DELETE,NOISE,ASK,SEP,UNKNOWN,EOS=range(9)
KEY_START,KEYS,VALUE_START,VALUES,VOCAB=16,32,64,16,128


def batch(seed:int,index:int,batch_size:int,*,keys:int=6,edits:int=2,gap:int|None=None,max_gap:int=16,query_weight:float=8.,lm_weight:float=.05):
    if not 1<=keys<=KEYS or min(edits,max_gap)<0:raise ValueError('Invalid task dimensions')
    rng=random.Random(f'energy-v19:{seed}:{index}')
    seqs=[];questions=[];answers=[]
    for _ in range(batch_size):
        chosen=rng.sample(range(KEYS),keys);memory={};s=[BOS]
        for key in chosen:
            value=rng.randrange(VALUES);memory[key]=value
            s.extend([SET,KEY_START+key,VALUE_START+value,SEP])
        for _ in range(edits):
            key=rng.choice(chosen)
            if rng.random()<.25:
                memory[key]=None;s.extend([DELETE,KEY_START+key,SEP])
            else:
                value=rng.randrange(VALUES);memory[key]=value
                s.extend([SET,KEY_START+key,VALUE_START+value,SEP])
        ngap=rng.randrange(max_gap+1) if gap is None else gap
        for _ in range(ngap):
            s.extend([NOISE,KEY_START+rng.randrange(KEYS),VALUE_START+rng.randrange(VALUES),SEP])
        # Query one key only. The model cannot read an earlier answer in this sequence.
        key=rng.choice(chosen)
        s.extend([ASK,KEY_START+key]);q=len(s)-1
        answer=UNKNOWN if memory[key] is None else VALUE_START+memory[key]
        s.extend([answer,EOS])
        seqs.append(s);questions.append(q);answers.append(answer)
    t=max(len(s)-1 for s in seqs)
    x=torch.full((batch_size,t),PAD,dtype=torch.long);y=x.clone();weights=torch.zeros(batch_size,t)
    query=torch.zeros(batch_size,t,dtype=torch.bool);reset=query.clone()
    for i,(s,q) in enumerate(zip(seqs,questions)):
        n=len(s)-1;x[i,:n]=torch.tensor(s[:-1]);y[i,:n]=torch.tensor(s[1:]);weights[i,:n]=lm_weight
        query[i,q]=True;weights[i,q]=query_weight;reset[i,0]=True
    return {'x':x,'y':y,'weights':weights,'query':query,'reset':reset,'answers':torch.tensor(answers)}


def objective(logits,item):
    import torch.nn.functional as F
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),item['y'].reshape(-1),reduction='none').view_as(item['weights'])
    denominator=item['weights'].sum()
    if denominator<=0:raise ValueError('No scored targets')
    return (loss*item['weights']).sum()/denominator


def validate_jsonl(path:str|Path):
    """Hash and group documents, requiring nonempty provenance/license declarations.

    Input lines: {text, source, license, document_id}. This validates supplied
    declarations, not the legal correctness of a license or near-duplicate freedom.
    No network requests and no corpus redistribution.
    """
    records=[];seen=set();path=Path(path)
    with path.open(encoding='utf-8') as fh:
        for i,line in enumerate(fh,1):
            d=json.loads(line)
            required={'text','source','license','document_id'}
            if not required.issubset(d) or any(not isinstance(d[k],str) or not d[k].strip() for k in required):
                raise ValueError(f'Invalid record on line {i}')
            digest=hashlib.sha256(d['text'].encode('utf-8')).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            bucket=int(digest[:8],16)%100
            records.append({**d,'sha256':digest,'split':'train' if bucket<90 else 'validation' if bucket<95 else 'test'})
    if not records:raise ValueError('No documents')
    return records
