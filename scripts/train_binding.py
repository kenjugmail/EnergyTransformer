#!/usr/bin/env python3
"""Bounded v17 2x2 parser study. Never changes a production parser checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, hashlib, json, math, random, time
import torch
from torch.nn import functional as F
from energy_transformer.bridge import TextCodec,TextParser,HOLD,SET
from energy_transformer.binding import OperationParser, bind
ROOT=Path(__file__).resolve().parents[1]

def read(name):
    return [json.loads(s) for s in (ROOT/f'results/v17/data/{name}.jsonl').read_text().splitlines()]

def tensors(records,codec,kind):
    text=[bind(r['text']).canonical if kind=='bound' else r['text'] for r in records]
    return codec.encode(text),torch.tensor([[r['op'],r['key'],r['value']] for r in records])

def loss_for(out,y,kind):
    if kind=='bound':return F.cross_entropy(out,y[:,0])
    loss=F.cross_entropy(out[0],y[:,0])
    mask=y[:,0]!=HOLD
    if mask.any():loss=loss+F.cross_entropy(out[1][mask],y[mask,1])
    mask=y[:,0]==SET
    if mask.any():loss=loss+F.cross_entropy(out[2][mask],y[mask,2])
    return loss

@torch.no_grad()
def evaluate(model,x,y,kind):
    loss=0.;correct=0
    for start in range(0,len(x),128):
        xx,yy=x[start:start+128],y[start:start+128];out=model(xx)
        loss+=float(loss_for(out,yy,kind))*len(xx)
        if kind=='bound':good=out.argmax(-1)==yy[:,0]
        else:
            pred=torch.stack([a.argmax(-1) for a in out],-1)
            good=(pred[:,0]==yy[:,0])&((yy[:,0]==HOLD)|(pred[:,1]==yy[:,1]))&((yy[:,0]!=SET)|(pred[:,2]==yy[:,2]))
        correct+=int(good.sum())
    return {'loss':loss/len(x),'exact':correct/len(x)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['joint','bound'],required=True);p.add_argument('--curriculum',choices=['basic','expanded'],required=True)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=900);a=p.parse_args()
    (ROOT/'checkpoints/v17').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    rows=read('train_'+a.curriculum);texts=[bind(r['text']).canonical if a.kind=='bound' else r['text'] for r in rows]
    codec=TextCodec.fit(texts);model=OperationParser(len(codec.vocabulary)) if a.kind=='bound' else TextParser(len(codec.vocabulary))
    x,y=tensors(rows,codec,a.kind);vx,vy=tensors(read('validation'),codec,a.kind)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    gen=torch.Generator().manual_seed(a.seed+17000)
    sha=hashlib.sha256();history=[];best=(-1.,float('inf'));t0=time.perf_counter();selected=0
    initial=evaluate(model,vx,vy,a.kind)
    weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    for step in range(1,a.steps+1):
        idx=torch.randint(len(x),(64,),generator=gen);sha.update(idx.numpy().tobytes())
        out=model(x[idx]);loss=loss_for(out,y[idx],a.kind)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for g in optimizer.param_groups:g['lr']=.002*(.2+.8*.5*(1+math.cos(math.pi*step/a.steps)))
        optimizer.step()
        if step%150==0 or step==a.steps:
            score=evaluate(model,vx,vy,a.kind);history.append({'step':step,**score,'train_loss':float(loss.detach())})
            print(a.kind,a.curriculum,a.seed,history[-1],flush=True)
            if (score['exact'],-score['loss'])>(best[0],-best[1]):
                best=(score['exact'],score['loss']);selected=step
                weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights)
    name=f'{a.kind}_{a.curriculum}_seed{a.seed}'
    torch.save({'format':'binding_proposer_v17','kind':a.kind,'curriculum':a.curriculum,'seed':a.seed,
                'vocabulary':codec.vocabulary,'weights':model.state_dict(),'threshold':.90,'selected_step':selected},ROOT/f'checkpoints/v17/{name}.pt')
    meta={**vars(a),'selected_step':selected,'initial_validation':initial,'validation':history,'batch_index_sha256':sha.hexdigest(),
          'parameters':sum(p.numel() for p in model.parameters()),'presented_examples':a.steps*64,'seconds':time.perf_counter()-t0,
          'training_source_sha256':hashlib.sha256((ROOT/f'results/v17/data/train_{a.curriculum}.jsonl').read_bytes()).hexdigest()}
    (ROOT/f'results/v17/train_{name}.json').write_text(json.dumps(meta,indent=2))
    print('SAVED',name,meta['parameters'],meta['seconds'],flush=True)
if __name__=='__main__':main()
