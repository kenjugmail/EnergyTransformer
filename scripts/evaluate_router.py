#!/usr/bin/env python3
"""Held-out v20 evaluation. Checkpoint selection happens before generation here."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,dataclasses,hashlib,random
import numpy as np
import pandas as pd
import torch
from energy_transformer.write_router import WriteRouterLM,RouterConfig,training_labels
from energy_transformer.scaling_data import batch,KEY_START,VALUE_START,SET,DELETE,NOISE,ASK,SEP,UNKNOWN,PAD,VALUES
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=['hard_aux','hard_noaux','soft_aux','attention_aux']
SEEDS=[11,22,33]

def load_model(path):
    d=torch.load(path,weights_only=True);m=WriteRouterLM(RouterConfig(**d['config']));m.load_state_dict(d['model']);m.eval();return m,d

def scalar_oracle(x):
    """Independent token interpreter, ignores optional FILL=9 tokens."""
    x=[int(a) for a in x if int(a)!=9];memory={};i=0
    while i<len(x):
        tok=x[i]
        if tok==SET and i+2<len(x):memory[x[i+1]]=x[i+2];i+=3
        elif tok==DELETE and i+1<len(x):memory.pop(x[i+1],None);i+=2
        elif tok==NOISE:i+=3
        elif tok==ASK and i+1<len(x):return memory.get(x[i+1],UNKNOWN)
        else:i+=1
    raise ValueError('No query')

def padded_record(item):
    # Alter the local grammar: optional FILL markers separate a value from key/op.
    seqs=[];queries=[]
    for row,q in zip(item['x'],item['query']):
        cut=int(torch.nonzero(q)[0]);s=[]
        for token in row[:cut+1].tolist():
            if VALUE_START<=token<VALUE_START+VALUES:s.extend([9]*4)
            s.append(token)
        queries.append(len(s)-1);seqs.append(s)
    return pack(seqs,queries,item['answers'])

def pack(seqs,queries,answers):
    t=max(map(len,seqs));b=len(seqs);x=torch.full((b,t),PAD,dtype=torch.long);q=torch.zeros(b,t,dtype=torch.bool);valid=q.clone();reset=q.clone()
    for i,(s,j) in enumerate(zip(seqs,queries)):x[i,:len(s)]=torch.tensor(s);q[i,j]=True;valid[i,:len(s)]=True;reset[i,0]=True
    return {'x':x,'query':q,'reset':reset,'answers':answers,'weights':valid.float()}

def trim(item):
    seqs=[];queries=[]
    for row,q in zip(item['x'],item['query']):
        cut=int(torch.nonzero(q)[0]);seqs.append(row[:cut+1].tolist());queries.append(cut)
    return pack(seqs,queries,item['answers'])

@torch.inference_mode()
def predict(m,item,intervention='none',chunk=64):
    x=item['x'];b,t=x.shape;pred=torch.empty(b,dtype=torch.long);nll=torch.empty(b);state=None;wrong=total=0
    truth=training_labels(x)
    for start in range(0,t,chunk):
        stop=min(t,start+chunk);logits,state,aux=m(x[:,start:stop],state,reset=item['reset'][:,start:stop],intervention=intervention)
        q=item['query'][:,start:stop];inds=torch.nonzero(q)
        if len(inds):
            rows,cols=inds[:,0],inds[:,1];lp=logits[rows,cols].log_softmax(-1);pred[rows]=lp.argmax(-1);nll[rows]=-lp[torch.arange(len(rows)),item['answers'][rows]]
        op=aux[0].argmax(-1);key=aux[1].argmax(-1);value=aux[2].argmax(-1)
        ref=[a[:,start:stop] for a in truth];valid=item['weights'][:,start:stop]>0
        edit=(op==1)|(op==2);same=(op==ref[0])&(key==ref[1])&((op!=1)|(value==ref[2]))
        wrong+=int((edit&~same&valid).sum());total+=int(valid.sum())
    return pred,nll,wrong,total


def main():
    torch.set_num_threads(1);out=ROOT/'results/v20';dpath=out/'test';dpath.mkdir(exist_ok=True)
    # Lock selected checkpoints before generating any new final episodes.
    selection={}
    for fam in FAMILIES:
        for seed in SEEDS:
            path=out/f'{fam}_seed{seed}/best.pt';assert (path.parent/'receipt.json').exists();_,d=load_model(path)
            selection[f'{fam}_seed{seed}']={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'step':d['step'],'validation':d['validation']}
    (out/'SELECTION_BEFORE_TEST.json').write_text(json.dumps(selection,indent=2))
    conditions=[('primary',{'keys':6,'edits':2,'gap':8},256),('long',{'keys':6,'edits':2,'gap':64},256),('capacity',{'keys':24,'edits':6,'gap':8},256),('many_edits',{'keys':6,'edits':12,'gap':8},256),('all32',{'keys':32,'edits':12,'gap':16},256),('layout_shift',{'keys':6,'edits':2,'gap':8},256),('long2048',{'keys':6,'edits':2,'gap':2048},64)]
    data={};manifest={}
    for ci,(name,options,count) in enumerate(conditions):
        items=[]
        for j in range(count//8):
            raw=batch(209000+ci,j,8,**options);item=padded_record(raw) if name=='layout_shift' else trim(raw)
            for row,answer in zip(item['x'],item['answers']):assert scalar_oracle(row)==int(answer)
            items.append(item)
        torch.save(items,dpath/f'{name}.pt');data[name]=items;manifest[name]={'episodes':count,'sha256':hashlib.sha256((dpath/f'{name}.pt').read_bytes()).hexdigest(),'options':options}
    (dpath/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[];aggregates=[]
    for fam in FAMILIES:
        for seed in SEEDS:
            m,d=load_model(out/f'{fam}_seed{seed}/best.pt')
            for name,items in data.items():
                # Long attention stress is deliberately bounded to the same 64 episodes.
                wrong=nt=0
                for j,item in enumerate(items):
                    pred,nll,w,n=predict(m,item);wrong+=w;nt+=n
                    for i in range(len(pred)):records.append({'family':fam,'seed':seed,'condition':name,'episode':j*8+i,'prediction':int(pred[i]),'answer':int(item['answers'][i]),'correct':bool(pred[i]==item['answers'][i]),'nll':float(nll[i]),'intervention':'none'})
                acc=np.mean([r['correct'] for r in records if r['family']==fam and r['seed']==seed and r['condition']==name]);aggregates.append({'family':fam,'seed':seed,'condition':name,'accuracy':acc,'wrong_predicted_edit_tokens':wrong,'tokens':nt});print(fam,seed,name,acc,wrong,flush=True)
            if fam=='hard_aux':
                for intervention in ['memory_reset','history_shuffle']:
                    for j,item in enumerate(data['primary']):
                        pred,nll,_,_=predict(m,item,intervention)
                        for i in range(len(pred)):records.append({'family':fam,'seed':seed,'condition':'primary','episode':j*8+i,'prediction':int(pred[i]),'answer':int(item['answers'][i]),'correct':bool(pred[i]==item['answers'][i]),'nll':float(nll[i]),'intervention':intervention})
    df=pd.DataFrame(records);df.to_csv(out/'raw.csv',index=False);pd.DataFrame(aggregates).to_csv(out/'edit_audit.csv',index=False)
    summary=[];rng=np.random.default_rng(209999)
    for (f,c,i),part in df.groupby(['family','condition','intervention']):
        per=part.groupby('episode').correct.mean().to_numpy();samples=per[rng.integers(len(per),size=(5000,len(per)))].mean(1)
        summary.append({'family':f,'condition':c,'intervention':i,'accuracy':per.mean(),'ci_low':np.quantile(samples,.025),'ci_high':np.quantile(samples,.975),'episodes':len(per),'seeds':part.seed.nunique(),'mean_nll':part.nll.mean()})
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    print('DONE',len(records),'model_episode_records',flush=True)
if __name__=='__main__':main()
