"""Small, fixed-budget Qwen conversion feasibility test; not a capability benchmark."""
from pathlib import Path
import gc,json,time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from energy_real.runtime import atomic_json,atomic_checkpoint,sha256
from energy_real.device import set_memory_limit,available_host_bytes
from .data import Documents
from .qwen_hybrid import QwenHybrid
from .spec import source_identity

DONOR={'id':'Qwen/Qwen2.5-0.5B','revision':'060db6499f32faf8b98477b0a26969ef7d8b9987'}

def prepare(root):
    root=Path(root);tok=AutoTokenizer.from_pretrained(root/'donor',local_files_only=True)
    result={}
    for split,count in [('train',64),('valid',8),('test',8)]:
        data=Documents(root.parent/'fineweb',split);rng=np.random.default_rng(707 if split=='train' else 809);batches=[]
        order=rng.permutation(len(data.docs))
        for index in order:
            doc=data.docs[int(index)];raw=bytes(data.data[doc['offset']:doc['offset']+doc['bytes']]).decode('utf8')
            ids=tok(raw,add_special_tokens=False)['input_ids']
            if len(ids)<129:continue
            start=int(rng.integers(len(ids)-128));batches.append(torch.tensor(ids[start:start+129],dtype=torch.long))
            if len(batches)==count:break
        if len(batches)!=count:raise ValueError('Insufficient document samples')
        result[split]=torch.stack(batches)
    atomic_checkpoint(result,root/'tokens.pt')
    atomic_json({'donor':DONOR,'source':source_identity(),'corpus':sha256(root.parent/'fineweb/manifest.json'),'tokens':sha256(root/'tokens.pt'),'context':128,'seed':707,'transfer_updates':64,'recovery_updates':64,'heldout_targets_per_split':1024,'caveat':'Fixed small feasibility sample; no broad capability or scaling claim'},root/'protocol.json')

def ce(logits,y):return torch.nn.functional.cross_entropy(logits.float().flatten(0,1),y.flatten())

def run(root):
    root=Path(root);out=root/'pilot001';out.mkdir(exist_ok=False)
    if available_host_bytes()<20*2**30:raise RuntimeError('Host admission threshold')
    torch.set_num_threads(1);set_memory_limit('cuda',4);torch.manual_seed(707)
    batches=torch.load(root/'tokens.pt',weights_only=True);teacher=AutoModelForCausalLM.from_pretrained(root/'donor',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    for p in teacher.parameters():p.requires_grad_(False)
    def guard():
        if available_host_bytes()<16*2**30:raise RuntimeError('Host reserve')
    def donor_eval(split):
        losses=[]
        with torch.no_grad():
            for row in batches[split]:
                guard();x=row[:-1][None].cuda();y=row[1:][None].cuda();losses.append(float(ce(teacher(x,use_cache=False).logits,y)))
        return sum(losses)/len(losses)
    baseline={s:donor_eval(s) for s in ('valid','test')};atomic_json(baseline,out/'donor_losses.json')
    # Frozen teacher targets are extracted once. Recovery uses a coarse-grained
    # KL: the top32 tokens plus one aggregate complement bucket, explicitly not full KL.
    features={};handles=[];energy_layers=tuple(i for i in range(24) if i%3!=2)
    for i in energy_layers:
        def hook(module,args,kwargs,output,i=i):features[i]=(kwargs['hidden_states'].detach().cpu(),output[0].detach().cpu())
        handles.append(teacher.model.layers[i].self_attn.register_forward_hook(hook,with_kwargs=True))
    targetdir=out/'targets';targetdir.mkdir()
    with torch.no_grad():
        for j,row in enumerate(batches['train']):
            guard();logits=teacher(row[:-1][None].cuda(),use_cache=False).logits.float();lp=logits.log_softmax(-1);values,indices=lp.topk(32,dim=-1)
            atomic_checkpoint({'features':dict(features),'top_logp':values.cpu(),'top_ids':indices.cpu()},targetdir/f'{j:03}.pt')
    for h in handles:h.remove()
    del teacher,features,logits,lp,values,indices;gc.collect();torch.cuda.empty_cache()
    base=AutoModelForCausalLM.from_pretrained(root/'donor',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
    model=QwenHybrid(base,window=128,donor=DONOR).cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,betas=(.9,.95),weight_decay=.1)
    def evaluate(split):
        model.eval();losses=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for row in batches[split]:
                guard();z,_=model(row[:-1][None].cuda());losses.append(float(ce(z,row[1:][None].cuda())))
        return sum(losses)/len(losses)
    history=[];initial=evaluate('valid');atomic_json({'untrained_valid_ce':initial,'parameters':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)},out/'initial.json')
    best=None
    for phase in ('transfer','recovery'):
        for j,row in enumerate(batches['train']):
            guard();model.train();optimizer.zero_grad(set_to_none=True);target=torch.load(targetdir/f'{j:03}.pt',weights_only=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if phase=='transfer':
                    terms=[]
                    for i,(x,y) in target['features'].items():
                        x=x.cuda();y=y.cuda();pred,_=model.base.model.layers[i].self_attn(x)
                        terms.append((pred.float()-y.float()).square().mean()/y.float().square().mean().clamp_min(1e-6))
                    loss=torch.stack(terms).mean();metrics={}
                else:
                    z,_=model(row[:-1][None].cuda());language=ce(z,row[1:][None].cuda());lp=z.float().log_softmax(-1);sp=lp.gather(-1,target['top_ids'].cuda());tp=target['top_logp'].cuda();prob=tp.exp();rest=(1-prob.sum(-1)).clamp_min(1e-7);srest=(1-sp.exp().sum(-1)).clamp_min(1e-7)
                    kl=((prob*(tp-sp)).sum(-1)+rest*(rest.log()-srest.log())).mean();loss=language+kl;metrics={'ce':float(language.detach()),'coarse_kl':float(kl.detach())}
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite conversion loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            record={'phase':phase,'step':j+1,'loss':float(loss.detach()),**metrics};history.append(record);print(record,flush=True)
            if (j+1)%16==0:
                score=evaluate('valid');record['valid_ce']=score
                if best is None or score<best:best=score;model.save(out/'best.pt')
                atomic_json(history,out/'history.json')
                atomic_checkpoint({'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'phase':phase,'completed_step':j+1,'source':source_identity()},out/'last.pt')
        atomic_json({'phase':phase,'valid_ce':evaluate('valid')},out/f'{phase}.json')
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['weights']);del saved
    test=evaluate('test');result={'status':'completed','donor':DONOR,'donor_losses':baseline,'best_valid_ce':best,'hybrid_test_ce':test,'relative_test_change':test/baseline['test']-1,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'context':128,'train_presentations':16384,'test_targets':1024,'larger_model_superiority_established':False,'limits':'Single-seed short feasibility run. Only replacements trained; no LoRA control or broad capability benchmark. Test is not used for checkpoint selection.'}
    atomic_json(result,out/'result.json');print(result,flush=True)

if __name__=='__main__':
    import sys
    (prepare if sys.argv[1]=='prepare' else run)(sys.argv[2])
