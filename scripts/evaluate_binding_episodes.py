#!/usr/bin/env python3
"""Diagnostic only: simulate automatic proposals to expose compounded errors.

Production BindingPredictor never mutates state. Initial facts here are supplied
structured records; final keys are queried directly. This isolates whether later
text proposals corrupt or update the correct associations, not general question
answering. A dictionary uses precisely the same predictions as the energy store.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,random
import numpy as np
import pandas as pd
import torch
from energy_transformer.bridge import NAMES,VALUES,HOLD,SET,DELETE,ASK,ProtectedState
from scripts.evaluate_binding import load,predict
ROOT=Path(__file__).resolve().parents[1]

def load_data(name):return [json.loads(s) for s in (ROOT/f'results/v17/data/{name}.jsonl').read_text().splitlines()]

def build():
    rng=random.Random(177001); episodes=[]
    for family in ('iid','composition','legacy_wording'):
        by_op={o:[r for r in load_data(family) if r['op']==o] for o in range(4)}
        for ei in range(8):
            keys=rng.sample(range(16),4); initial=[{'op':SET,'key':k,'value':rng.randrange(16)} for k in keys]
            # One fixed edit prefix and distractor stream per base episode; all
            # gap lengths are exact prefixes, not newly sampled sentence sets.
            actions=[]
            for op,key in [(SET,keys[0]),(DELETE,keys[1])]:
                source=dict(rng.choice(by_op[op]));source['key']=key;source['value']=rng.randrange(16)
                source['text']=source['template'].format(name=NAMES[key],value=VALUES[source['value']]);actions.append(source)
            for _ in range(1024):
                source=dict(rng.choice(by_op[HOLD]));source['key']=rng.choice(keys);source['value']=rng.randrange(16)
                source['text']=source['template'].format(name=NAMES[source['key']],value=VALUES[source['value']]);actions.append(source)
            for gap in (0,64,1024):
                episodes.append({'family':family,'episode':ei,'gap':gap,'initial':initial,'keys':keys,'actions':actions[:gap+2]})
    return episodes

def state_values(state,keys):
    return [int(state.information[k].argmax()) if float(state.precision[k])>.25 else -1 for k in keys]

def main():
    torch.set_num_threads(1);episodes=build();root=ROOT/'results/v17'
    (root/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    unique={r['text']:r for e in episodes for r in e['actions']};rr=list(unique.values());index={r['text']:i for i,r in enumerate(rr)}
    records=[]
    paths=list(sorted((ROOT/'checkpoints/v17').glob('*.pt')))+list(sorted((ROOT/'checkpoints/v14').glob('bridge_seed*.pt')))
    for path in paths:
        m,c,k=load(path);a=predict(m,c,k,rr)
        for policy in ('score_only','strict_vocab'):
            accepted=(a['confidence']>=.9)&a['valid']
            if policy=='strict_vocab':accepted &=a['known']
            for e in episodes:
                state=ProtectedState.initial();dictionary={};truth={};false_edits=0
                for r in e['initial']:
                    state=state.edit(r['op'],r['key'],r['value']);dictionary[r['key']]=r['value'];truth[r['key']]=r['value']
                for r in e['actions']:
                    if r['op']==SET:truth[r['key']]=r['value']
                    elif r['op']==DELETE:truth.pop(r['key'],None)
                    i=index[r['text']]
                    if not accepted[i]:continue
                    op,key,value=map(int,a['pred'][i])
                    if op not in (SET,DELETE):continue
                    false_edits+=int(not a['exact'][i])
                    state=state.edit(op,key,value if op==SET else 0)
                    if op==SET:dictionary[key]=value
                    else:dictionary.pop(key,None)
                actual=state_values(state,e['keys']);expected=[truth.get(key,-1) for key in e['keys']]
                assert actual==[dictionary.get(key,-1) for key in e['keys']]
                correct=np.array(actual)==expected
                records.append({'model':path.stem,'policy':policy,'family':e['family'],'episode':e['episode'],'gap':e['gap'],
                    'accuracy':float(correct.mean()),'all_correct':bool(correct.all()),'wrong_edits':false_edits,
                    'dictionary_identical':True})
        print(path.stem,'done',flush=True)
    df=pd.DataFrame(records);df.to_csv(root/'episode_results.csv',index=False)
    df['model_family']=df.model.str.replace(r'_seed\d+$','',regex=True)
    df.groupby(['model_family','policy','family','gap']).agg(accuracy=('accuracy','mean'),all_correct=('all_correct','mean'),
       wrong_edits=('wrong_edits','sum')).reset_index().to_csv(root/'episode_summary.csv',index=False)
    print('EPISODES_COMPLETE',len(episodes),len(unique),len(records),flush=True)
if __name__=='__main__':main()
