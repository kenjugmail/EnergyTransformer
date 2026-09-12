#!/usr/bin/env python3
"""Fixed held-out parser, episodic recall, confidence, and memory controls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,random,time,hashlib
import torch
from energy_transformer.bridge import *
from energy_transformer.data import dataset,record,HELDOUT_TEMPLATES,TRAIN_TEMPLATES
ROOT=Path(__file__).resolve().parents[1]

@torch.inference_mode()
def parse_records(session,records):
 out=[]
 for i in range(0,len(records),128):
  chunk=records[i:i+128]; logits=session.parser(session.codec.encode([x['text'] for x in chunk]))
  prob=[x.softmax(-1) for x in logits];ids=torch.stack([x.argmax(-1) for x in logits],-1).tolist()
  conf=torch.stack([x.amax(-1) for x in prob],-1).tolist()
  for rec,(o,k,v),(co,ck,cv) in zip(chunk,ids,conf):
   c=min([co]+([ck] if o!=HOLD else [])+([cv] if o==SET else []))
   if not lexical_consistent(rec['text'],o,k,v):c=-1.
   out.append((o,k,v,c))
 return out

def make_episodes(split,count=32):
 rng=random.Random(800001 if split=='iid' else 800002)
 tasks=[]
 for gap in [0,64,1024]:
  for i in range(count):
   keys=rng.sample(range(16),4);truth={};seq=[]
   for k in keys:
    rec=record(rng,'composition' if split=='iid' else split,op=SET,key=k);seq.append(rec);truth[k]=rec['value']
   # Repeated revisions, one deletion, then competing distractors.
   rec=record(rng,'composition' if split=='iid' else split,op=SET,key=keys[0]);seq.append(rec);truth[keys[0]]=rec['value']
   seq.append(record(rng,'composition' if split=='iid' else split,op=DELETE,key=keys[1]));truth[keys[1]]=None
   for _ in range(gap):seq.append(record(rng,'composition' if split=='iid' else split,op=HOLD))
   queries=[record(rng,'composition' if split=='iid' else split,op=ASK,key=k) for k in keys]
   tasks.append({'split':split,'gap':gap,'id':i,'records':seq,'queries':queries,
                 'targets':[VALUES[truth[k]] if truth[k] is not None else 'unknown' for k in keys]})
 return tasks

@torch.inference_mode()
def evaluate():
 torch.set_num_threads(1);rows=[];parser_rows=[];details=[]
 tasks=make_episodes('iid')+make_episodes('wording')
 # Keep actual test sentences and labels for reproducibility; no real-person corpus.
 (ROOT/'results/v14/episodes.json').write_text(json.dumps(tasks,separators=(',',':')))
 for seed in [11,22,33]:
  session=EnergyTextSession(ROOT/f'checkpoints/v14/bridge_seed{seed}.pt')
  for split in ['iid','wording']:
   records=dataset(2048,601 if split=='iid' else 602,split);preds=parse_records(session,records)
   for rec,(o,k,v,c) in zip(records,preds):
    exact=(o==rec['op'] and (o==HOLD or k==rec['key']) and (o!=SET or v==rec['value']))
    parser_rows.append({'seed':seed,'split':split,'op':rec['op'],'text':rec['text'],'predicted_op':o,
      'exact':int(exact),'accepted':int(c>=session.threshold),'confidence':c,
      'false_edit':int(rec['op'] in [HOLD,ASK] and o in [SET,DELETE] and c>=session.threshold)})
  for task in tasks:
   recs=task['records'];queries=task['queries'];preds=parse_records(session,recs);qpred=parse_records(session,queries)
   for method in ['energy_guarded','energy_unguarded','dictionary_learned','oracle_energy','last_event_only']:
    state=ProtectedState.initial();dic={};rejected=0;false_edits=0
    for rec,(o,k,v,c) in zip(recs,preds):
     if method=='oracle_energy':o,k,v,c=rec['op'],rec['key'],rec['value'],1.
     if method!='energy_unguarded' and c<session.threshold:rejected+=1;continue
     if o in (SET,DELETE) and rec['op'] in (HOLD,ASK):false_edits+=1
     if method=='last_event_only':state=ProtectedState.initial()
     if method=='dictionary_learned':
      if o==SET:dic[k]=VALUES[v]
      elif o==DELETE:dic.pop(k,None)
     else:state=state.edit(o,k,v)
    answers=[]
    for query,(o,k,v,c) in zip(queries,qpred):
     if method=='oracle_energy':o,k,c=ASK,query['key'],1.
     if (method!='energy_unguarded' and c<session.threshold) or o!=ASK:answer='<abstained>'
     elif method=='dictionary_learned':answer=dic.get(k,'unknown')
     else:answer=session.decoder.text(state.feature(k)[None])[0]
     answers.append(answer)
    score=[a==b for a,b in zip(answers,task['targets'])]
    row={'seed':seed,'split':task['split'],'gap':task['gap'],'task':task['id'],'method':method,
         'correct':sum(score),'queries':4,'episode_exact':int(all(score)),'rejected_records':rejected,'false_edits':false_edits}
    rows.append(row)
    if task['id']<2:details.append({**row,'answers':answers,'targets':task['targets']})
   print(seed,task['split'],task['gap'],task['id'],flush=True) if task['id']==0 else None
 for name,data in [('parser_records.csv',parser_rows),('episode_records.csv',rows)]:
  with open(ROOT/'results/v14'/name,'w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 # Summary uses seeds as replications, not new independent task definitions.
 summary=[]
 for split in ['iid','wording']:
  for gap in [0,64,1024]:
   for method in ['energy_guarded','energy_unguarded','dictionary_learned','oracle_energy','last_event_only']:
    rr=[r for r in rows if r['split']==split and r['gap']==gap and r['method']==method]
    summary.append({'split':split,'gap':gap,'method':method,'accuracy':sum(r['correct'] for r in rr)/sum(r['queries'] for r in rr),
     'episode_exact':sum(r['episode_exact'] for r in rr)/len(rr),'false_edits':sum(r['false_edits'] for r in rr),
     'method_episode_records':len(rr)})
 with open(ROOT/'results/v14/summary.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 stats=[]
 for seed in [11,22,33]:
  for split in ['iid','wording']:
   rr=[r for r in parser_rows if r['seed']==seed and r['split']==split];a=[r for r in rr if r['accepted']]
   stats.append({'seed':seed,'split':split,'exact':sum(r['exact'] for r in rr)/len(rr),
     'coverage':len(a)/len(rr),'selective_exact':sum(r['exact'] for r in a)/len(a),
     'false_edits':sum(r['false_edit'] for r in rr)})
 (ROOT/'results/v14/parser_summary.json').write_text(json.dumps(stats,indent=2))
 (ROOT/'results/v14/sample_episodes.json').write_text(json.dumps(details,indent=2))
 print(json.dumps(stats,indent=2),flush=True)
 print('DONE',len(rows),'method/episode records',len(tasks),'unique episodes')
if __name__=='__main__':evaluate()
