#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,dataclasses,hashlib,time,math
import torch
from energy_transformer.write_router import RouterConfig,WriteRouterLM,router_loss,training_labels
from energy_transformer.scaling_data import batch,objective
ROOT=Path(__file__).resolve().parents[1]

@torch.no_grad()
def score(m,seed=201001):
    m.eval();correct=total=0;loss=0.;op=key=val=nt=kt=vt=0
    for i in range(8):
        z=batch(seed,i,16);logits,_,a=m(z['x'],reset=z['reset']);q=z['query'];correct+=int((logits.argmax(-1)[q]==z['y'][q]).sum());total+=int(q.sum());loss+=float(objective(logits,z))
        targets=training_labels(z['x']);valid=z['weights']>0
        op+=int((a[0].argmax(-1)[valid]==targets[0][valid]).sum());nt+=int(valid.sum())
        for j in (1,2):
            mask=(targets[j]!=-100)&valid;c=int((a[j].argmax(-1)[mask]==targets[j][mask]).sum());n=int(mask.sum())
            if j==1:key+=c;kt+=n
            else:val+=c;vt+=n
    m.train();return dict(accuracy=correct/total,loss=loss/8,operation=op/nt,key=key/kt,value=val/vt)

def run(kind,aux,seed,steps,out):
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=RouterConfig(kind=kind);m=WriteRouterLM(cfg);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    identity={'model':dataclasses.asdict(cfg),'seed':seed,'steps':steps,'aux_weight':aux,'batch':16,'training_seed_offset':202000,'torch':str(torch.__version__)}
    (out/'config.json').write_text(json.dumps(identity,indent=2));best=-1;best_loss=1e9;start=time.perf_counter();digest=hashlib.sha256();history=[]
    initial=score(m);(out/'initial.json').write_text(json.dumps(initial))
    for step in range(1,steps+1):
        # Identical sequence curriculum across all variants.
        task=({'keys':1,'edits':0,'max_gap':0} if step<=100 else {'keys':2,'edits':1,'max_gap':2} if step<=200 else {'keys':4,'edits':2,'max_gap':6} if step<=300 else {})
        z=batch(seed+202000,step-1,16,**task)
        digest.update(z['x'].numpy().tobytes());digest.update(z['y'].numpy().tobytes())
        rate=.002*min(1.,step/20)*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        for g in opt.param_groups:g['lr']=rate
        logits,_,a=m(z['x'],reset=z['reset']);loss=router_loss(logits,a,z,aux)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%100==0 or step==steps:
            v=score(m);r={'step':step,'train_loss':float(loss.detach()),**v,'elapsed':time.perf_counter()-start};history.append(r);print(kind,aux,seed,r,flush=True)
            if (v['accuracy'],-v['loss'])>(best,-best_loss):
                best=v['accuracy'];best_loss=v['loss'];torch.save({'config':dataclasses.asdict(cfg),'model':m.state_dict(),'step':step,'seed':seed,'aux_weight':aux,'validation':v},out/'best.pt')
            (out/'log.json').write_text(json.dumps(history,indent=2))
    torch.save({'config':identity,'model':m.state_dict(),'optimizer':opt.state_dict(),'step':steps,'rng':torch.get_rng_state()},out/'last.pt')
    (out/'receipt.json').write_text(json.dumps({'batch_sha256':digest.hexdigest(),'parameters':m.parameter_count(),'elapsed_seconds':time.perf_counter()-start,'presented_episodes':16*steps,'steps':steps},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['hard','soft','attention'],default='hard');p.add_argument('--aux',type=float,default=1.);p.add_argument('--seed',type=int,default=11);p.add_argument('--steps',type=int,default=1000);p.add_argument('--out',required=True);a=p.parse_args();run(a.kind,a.aux,a.seed,a.steps,a.out)
