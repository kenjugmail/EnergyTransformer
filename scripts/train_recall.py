#!/usr/bin/env python3
"""Train the tiny v18 memory modules. No production checkpoint is changed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,math,hashlib
import torch
from energy_transformer.geometric_memory import EditMemory,KINDS
from energy_transformer.recall_data import episodes
from energy_transformer.recall_attention import SoftmaxRecall
ROOT=Path(__file__).resolve().parents[1]

@torch.no_grad()
def evaluate(model,b):
    out,_,_=model(b.keys,b.values,b.operations,b.queries)
    return float((out-b.targets).square().mean())

def run(kind,seed,steps=600,objective='delayed',pilot=False):
    (ROOT/'checkpoints/v18').mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=SoftmaxRecall() if kind=='softmax' else EditMemory(kind)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
    val=episodes(18000001,batch=96)
    initial=evaluate(model,val);best=initial;selected=0;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    history=[{'step':0,'validation_mse':initial}];sha=hashlib.sha256();start=time.perf_counter()
    for step in range(1,steps+1):
        batch=episodes(18010000+10000*seed+step)
        sha.update(batch.keys.numpy().tobytes());sha.update(batch.targets.numpy().tobytes())
        out,_,instant=model(batch.keys,batch.values,batch.operations,batch.queries,immediate=objective=='immediate')
        if objective=='delayed':loss=(out-batch.targets).square().mean()
        else:
            mask=batch.operations!=2
            truth=batch.values*(batch.operations==0).unsqueeze(-1)
            loss=(instant-truth).square().mean(-1)[mask].mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups:group['lr']=.003*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.step()
        if step%100==0 or step==steps:
            score=evaluate(model,val);row={'step':step,'training_loss':float(loss.detach()),'validation_mse':score};history.append(row)
            print(kind,objective,seed,row,flush=True)
            if score<best:best=score;selected=step;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    folder=ROOT/'results/v18'/('pilot' if pilot else 'training');folder.mkdir(parents=True,exist_ok=True)
    name=f'{kind}_{objective}_seed{seed}'
    if pilot:name='pilot_'+name
    path=ROOT/f'checkpoints/v18/{name}.pt'
    torch.save({'format':'memory_edit_v18','kind':kind,'objective':objective,'seed':seed,'dimension':12,'value_dim':4,'prior':.01,'weights':best_state,'selected_step':selected},path)
    record={'name':name,'kind':kind,'objective':objective,'seed':seed,'steps':steps,'selected_step':selected,'parameters':sum(p.numel() for p in model.parameters()),'episodes_presented':steps*24,'input_hash':sha.hexdigest(),'seconds_including_validation':time.perf_counter()-start,'history':history}
    (folder/f'{name}.json').write_text(json.dumps(record,indent=2));return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=(*KINDS,'softmax'),default='precision');p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=600);p.add_argument('--objective',choices=['delayed','immediate'],default='delayed');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    print(json.dumps(run(**vars(a)),indent=2))
