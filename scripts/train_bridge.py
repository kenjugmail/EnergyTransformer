#!/usr/bin/env python3
"""Train parser and bounded memory-conditioned answer decoder, on CPU by default."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, hashlib, json, math, random, time
import torch
from torch.nn import functional as F
from energy_transformer.bridge import TextCodec,TextParser,AnswerDecoder,VALUES,SET,HOLD,MAX_ANSWER
from energy_transformer.data import dataset,record,vocabulary_texts
ROOT=Path(__file__).resolve().parents[1]

def parse_loss(out,labels):
 op,key,value=labels.unbind(-1)
 loss=F.cross_entropy(out[0],op)
 relevant=op!=HOLD
 if relevant.any(): loss=loss+F.cross_entropy(out[1][relevant],key[relevant])
 relevant=op==SET
 if relevant.any():loss=loss+F.cross_entropy(out[2][relevant],value[relevant])
 return loss

@torch.no_grad()
def score(model,codec,records):
 losses=[];correct=[]
 for i in range(0,len(records),64):
  rr=records[i:i+64]; lab=torch.tensor([[r['op'],r['key'],r['value']] for r in rr])
  out=model(codec.encode([r['text'] for r in rr]));losses.append((float(parse_loss(out,lab)),len(rr)))
  pred=torch.stack([o.argmax(-1) for o in out],-1)
  good=(pred[:,0]==lab[:,0]) & ((lab[:,0]==HOLD)|(pred[:,1]==lab[:,1])) & ((lab[:,0]!=SET)|(pred[:,2]==lab[:,2]))
  correct.extend(good.tolist())
 return {'loss':sum(x*n for x,n in losses)/len(records),'exact':sum(correct)/len(correct)}

def answer_training(decoder,seed,steps):
 gen=torch.Generator().manual_seed(seed+10000)
 optimizer=torch.optim.AdamW(decoder.parameters(),lr=.005,weight_decay=0.)
 history=[]
 for step in range(steps):
  label=torch.randint(0,17,(64,),generator=gen)
  known=label<16; loc=F.one_hot(label.clamp_max(15),16).float()*known[:,None]
  # Vary positive precision/evidence so the head reads features, not one fixed scale.
  mass=.75+torch.rand(64,1,generator=gen)*.5
  feature=torch.cat([loc*mass/(mass+.25),mass*known[:,None]],-1)
  texts=[VALUES[v] if v<16 else 'unknown' for v in label.tolist()]
  target=torch.full((64,MAX_ANSWER),256,dtype=torch.long)
  for i,t in enumerate(texts): target[i,:len(t)]=torch.tensor(list(t.encode()))
  logits=decoder(feature)
  loss=F.cross_entropy(logits.reshape(-1,257),target.reshape(-1))
  optimizer.zero_grad();loss.backward();optimizer.step()
  if (step+1)%100==0:history.append({'step':step+1,'loss':float(loss.detach())})
 return history

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=900);p.add_argument('--answer-steps',type=int,default=500);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(a.seed);random.seed(a.seed)
 codec=TextCodec.fit(vocabulary_texts());model=TextParser(len(codec.vocabulary));decoder=AnswerDecoder()
 validation=dataset(768,20001,'validation')
 optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
 rng=random.Random(a.seed);logs=[];best=float('inf');selected=0;t0=time.monotonic();sha=hashlib.sha256()
 initial=score(model,codec,validation)
 for step in range(1,a.steps+1):
  rr=[record(rng,'train') for _ in range(64)]
  texts=[r['text'] for r in rr];sha.update(('\n'.join(texts)+'\n').encode())
  label=torch.tensor([[r['op'],r['key'],r['value']] for r in rr]);out=model(codec.encode(texts));loss=parse_loss(out,label)
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  for g in optimizer.param_groups:g['lr']=.002*(.2+.8*.5*(1+math.cos(math.pi*step/a.steps)))
  optimizer.step()
  if step%150==0 or step==a.steps:
   s=score(model,codec,validation);logs.append({'step':step,'train_loss':float(loss.detach()),**s})
   print(a.seed,logs[-1],flush=True)
   if s['loss']<best:best=s['loss'];selected=step;weights={k:v.detach().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(weights)
 answer_log=answer_training(decoder,a.seed,a.answer_steps)
 target=ROOT/'checkpoints/v14'/f'bridge_seed{a.seed}.pt';target.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'format':'energy_text_bridge_v14','vocabulary':codec.vocabulary,'parser':model.state_dict(),
             'decoder':decoder.state_dict(),'threshold':.90,'seed':a.seed,'selected_step':selected},target)
 meta={'seed':a.seed,'parser_steps':a.steps,'answer_steps':a.answer_steps,'selected_step':selected,'initial_validation':initial,
       'validation':logs,'answer_training':answer_log,'training_sha256':sha.hexdigest(),
       'parameters':sum(x.numel() for x in model.parameters())+sum(x.numel() for x in decoder.parameters()),
       'seconds':time.monotonic()-t0}
 (ROOT/'results/v14'/f'train_seed{a.seed}.json').write_text(json.dumps(meta,indent=2))
 print('SAVED',target,meta['seconds'],flush=True)
if __name__=='__main__':main()
