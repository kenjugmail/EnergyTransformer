#!/usr/bin/env python3
"""v17 evaluation. Reports errors, coverage, and simulated edit risk separately.

No score in this script authorizes real memory writes. The legacy wording test
has been observed in previous project versions; composition tests are newly
constructed but still synthetic and produced by the same research process.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json, hashlib
import numpy as np
import pandas as pd
import torch
from energy_transformer.bridge import TextCodec,TextParser,NAMES,VALUES,HOLD,SET,DELETE,ASK,words
from energy_transformer.binding import OperationParser,bind
from energy_transformer.binding_data import ALIASES,ALIAS_VALUES
ROOT=Path(__file__).resolve().parents[1]

def rows(name):return [json.loads(s) for s in (ROOT/f'results/v17/data/{name}.jsonl').read_text().splitlines()]

def load(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    old=d.get('format')=='energy_text_bridge_v14'
    kind='joint' if old else d['kind']
    codec=TextCodec(d['vocabulary'])
    m=OperationParser(len(codec.vocabulary)) if kind=='bound' else TextParser(len(codec.vocabulary))
    m.load_state_dict(d['parser'] if old else d['weights']);m.eval()
    return m,codec,kind

@torch.inference_mode()
def predict(model,codec,kind,rr,alias=False):
    ns,vs=(ALIASES,ALIAS_VALUES) if alias else (NAMES,VALUES)
    bindings=[bind(r['text'],ns,vs) for r in rr]
    text=[b.canonical if kind=='bound' else r['text'] for b,r in zip(bindings,rr)]
    prediction=[];confidence=[]
    for start in range(0,len(rr),256):
        out=model(codec.encode(text[start:start+256]))
        if kind=='bound':
            op=out.argmax(-1).numpy(); cf=out.softmax(-1).max(-1).values.numpy()
            pp=np.array([[int(o),b.key if b.key is not None else -1,b.value if b.value is not None else -1]
                         for o,b in zip(op,bindings[start:start+256])])
        else:
            pp=np.stack([o.argmax(-1).numpy() for o in out],-1)
            cc=np.stack([o.softmax(-1).max(-1).values.numpy() for o in out],-1)
            cf=np.minimum(cc[:,0],np.where(pp[:,0]!=HOLD,cc[:,1],1))
            cf=np.minimum(cf,np.where(pp[:,0]==SET,cc[:,2],1))
        prediction.append(pp);confidence.append(cf)
    p=np.concatenate(prediction);c=np.concatenate(confidence)
    y=np.array([[r['op'],r['key'],r['value']] for r in rr])
    valid=[];known=[]
    for pp,b,t in zip(p,bindings,text):
        op,k,v=pp
        ok=b.reason is None
        if op!=HOLD:ok=ok and b.key is not None and k==b.key
        if op==SET:ok=ok and b.value is not None and v==b.value
        # Same single-entity schema applied to both fresh architectures.
        if op in (ASK,DELETE):ok=ok and b.value is None
        valid.append(ok)
        known.append(all(w in codec.index for w in words(t)))
    exact=(p[:,0]==y[:,0])&((y[:,0]==HOLD)|(p[:,1]==y[:,1]))&((y[:,0]!=SET)|(p[:,2]==y[:,2]))
    return {'pred':p,'confidence':c,'target':y,'valid':np.array(valid),'known':np.array(known),'exact':exact}


def metric(a,strict=False):
    p,y,good=a['pred'],a['target'],a['exact']
    accepted=(a['confidence']>=.9)&a['valid']
    if strict:accepted &=a['known']
    proposed_edit=np.isin(p[:,0],[SET,DELETE]);true_hold=y[:,0]==HOLD;true_edit=np.isin(y[:,0],[SET,DELETE])
    false_write=accepted&proposed_edit&~good
    def ratio(x,mask):return float(x[mask].mean()) if mask.any() else None
    return {'n':len(y),'op_accuracy':float((p[:,0]==y[:,0]).mean()),'exact':float(good.mean()),
      'coverage':float(accepted.mean()),'accepted_error':ratio(~good,accepted),
      'accepted_wrong_edits':int(false_write.sum()),'hold_false_edit_rate':ratio(accepted&proposed_edit,true_hold),
      'raw_hold_false_edit_rate':ratio(proposed_edit,true_hold),
      'true_edit_coverage':ratio(accepted,true_edit),'unknown_fraction':float((~a['known']).mean())}


def bootstrap_difference(a,b,rr,seed=17199):
    groups={}
    for i,r in enumerate(rr):groups.setdefault(r['template_id'],[]).append(i)
    idx=list(groups.values());delta=a['exact'].astype(float)-b['exact'].astype(float)
    count=np.array([len(g) for g in idx]);sums=np.array([delta[g].sum() for g in idx]);rng=np.random.default_rng(seed)
    picks=rng.integers(len(idx),size=(5000,len(idx)))
    vals=sums[picks].sum(1)/count[picks].sum(1)
    return {'difference':float(delta.mean()),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),'template_groups':len(idx)}


def main():
    torch.set_num_threads(1)
    result=[];arrays={};datasets={n:rows(n) for n in ['iid','composition','pairs','legacy_wording','aliases']}
    paths=list(sorted((ROOT/'checkpoints/v17').glob('*.pt')))
    paths+=list(sorted((ROOT/'checkpoints/v14').glob('bridge_seed*.pt')))
    outdir=ROOT/'results/v17/predictions';outdir.mkdir(exist_ok=True)
    for path in paths:
        m,c,k=load(path)
        name=path.stem
        arrays[name]={}
        for dataset,rr in datasets.items():
            a=predict(m,c,k,rr,alias=dataset=='aliases');arrays[name][dataset]=a
            np.savez_compressed(outdir/f'{name}_{dataset}.npz',**a)
            for strict in (False,True):
                result.append({'model':name,'dataset':dataset,'policy':'strict_vocab' if strict else 'score_only',**metric(a,strict)})
        print(name,[(n,round(arrays[name][n]['exact'].mean(),4)) for n in datasets],flush=True)
    df=pd.DataFrame(result);df.to_csv(ROOT/'results/v17/parser_results.csv',index=False)
    df['family']=df.model.str.replace(r'_seed\d+$','',regex=True)
    df.groupby(['family','dataset','policy']).agg(
        exact=('exact','mean'),op_accuracy=('op_accuracy','mean'),coverage=('coverage','mean'),
        accepted_error=('accepted_error','mean'),hold_false_edit_rate=('hold_false_edit_rate','mean'),
        accepted_wrong_edits=('accepted_wrong_edits','sum'),true_edit_coverage=('true_edit_coverage','mean'),
        unknown_fraction=('unknown_fraction','mean')).reset_index().to_csv(ROOT/'results/v17/summary.csv',index=False)
    pairs=[]
    for dataset,rr in datasets.items():
        for other in ('joint_expanded','bound_basic','joint_basic','bridge'):
            # Average across paired training seeds, cluster by whole template.
            a={'exact':np.mean([arrays[f'bound_expanded_seed{s}'][dataset]['exact'] for s in (11,22,33)],0)}
            b={'exact':np.mean([arrays[f'{other}_seed{s}'][dataset]['exact'] for s in (11,22,33)],0)}
            pairs.append({'dataset':dataset,'a':'bound_expanded','b':other,**bootstrap_difference(a,b,rr)})
    pd.DataFrame(pairs).to_csv(ROOT/'results/v17/paired.csv',index=False)
    print('EVALUATION_COMPLETE',len(result),flush=True)
if __name__=='__main__':main()
