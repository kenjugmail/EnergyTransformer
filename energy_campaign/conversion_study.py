"""Bounded attention transfer and teacher-guided recovery for the pinned donor."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from energy_real.runtime import atomic_checkpoint,atomic_json,sha256,sync
from .conversion import ConvertedLM,EnergyStaticCache,add_lora,distillation_loss,cache_bytes
from transformers.cache_utils import StaticCache
from .spec import MODEL,source_identity
from .data import Documents


class TokenDocuments:
    def __init__(self,data,donor,split):
        self.tokenizer=AutoTokenizer.from_pretrained(donor,local_files_only=True)
        corpus=Documents(data,split);self.identity=corpus.file_identity();self.docs=[]
        tokenizer_hash=hashlib.sha256(''.join(sha256(p) for p in sorted(Path(donor).glob('*token*')) if p.is_file()).encode()).hexdigest()
        cache=Path(data)/f'subword-{split}-{tokenizer_hash[:16]}.pt'
        identity={'corpus':self.identity,'tokenizer':tokenizer_hash,'split':split}
        if cache.exists():
            saved=torch.load(cache,weights_only=True,map_location='cpu')
            if saved['identity']!=identity:raise ValueError('Tokenized corpus identity mismatch')
            self.docs=saved['documents'];return
        for d in corpus.docs:
            raw=bytes(corpus.data[d['offset']:d['offset']+d['bytes']]);text=raw.decode('utf-8')
            ids=self.tokenizer(text,add_special_tokens=False)['input_ids']
            self.docs.append(ids)
        atomic_checkpoint({'identity':identity,'documents':self.docs},cache)
    def batch(self,rng,length,device):
        eligible=[i for i,d in enumerate(self.docs) if len(d)>length]
        if not eligible:raise ValueError('No donor-token document at requested context')
        doc=self.docs[int(rng.choice(eligible))];start=int(rng.integers(len(doc)-length))
        x=torch.tensor([doc[start:start+length+1]],device=device)
        return x[:,:-1],x[:,1:]


def load_donor(path,device):
    cfg=json.loads((Path(path)/'config.json').read_text())
    if (cfg.get('hidden_size'),cfg.get('num_hidden_layers'),cfg.get('num_attention_heads'),cfg.get('num_key_value_heads'))!=(576,30,9,3):raise ValueError('Wrong donor architecture')
    return AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to(device)


def train(job):
    began=time.monotonic()
    mode=job['mode'];device=job.get('device','cuda');out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    if mode not in ('transfer','recovery','lora'):raise ValueError('Unknown conversion phase')
    donor=job['donor'];teacher=load_donor(donor,device).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    torch.manual_seed(707)
    if mode=='recovery':
        model=ConvertedLM.load(job['initial'],device);add_lora(model.base,8);model.lora=True
    else:model=ConvertedLM(load_donor(donor,device),layers=() if mode=='lora' else tuple(MODEL['layers']),lora=mode=='lora')
    if mode=='transfer' and not (out/'untrained.pt').exists():model.save(out/'untrained.pt')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,betas=(.9,.95),weight_decay=.1)
    train_data=TokenDocuments(job['data'],donor,'train');valid=TokenDocuments(job['data'],donor,'valid');rng=np.random.default_rng(707)
    identity={'source':source_identity(),'donor':MODEL,'donor_files':{p.name:sha256(p) for p in sorted(Path(donor).glob('*')) if p.is_file()},
              'data':train_data.identity,'job':job};start=0;best=None;history=[]
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Conversion resume identity mismatch')
        model.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng'];start=saved['step'];best=saved['best'];history=saved['history']
        torch.set_rng_state(saved['torch_rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    captured={};handles=[]
    if mode=='transfer':
        for index in model.layers:
            def hook(module,args,kwargs,output,index=index):captured[index]=(kwargs['hidden_states'].detach(),output[0].detach())
            handles.append(teacher.model.layers[index].self_attn.register_forward_hook(hook,with_kwargs=True))
    def loss_on(x,y):
        with torch.no_grad():t=teacher(x,use_cache=False).logits
        if mode=='transfer':
            errors=[]
            for i,(features,target) in captured.items():
                pred=model.base.model.layers[i].self_attn(features)[0]
                errors.append((pred.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-6))
            return torch.stack(errors).mean(),{}
        return distillation_loss(model(x).logits,t,y)
    steps=job.get('steps',1000000);step=start
    try:
        while step<steps and time.monotonic()-began<job['seconds']-30:
            model.train();x,y=train_data.batch(rng,1024,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):loss,metrics=loss_on(x,y)
            if not torch.isfinite(loss):raise FloatingPointError('Conversion loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1
            history.append({'step':step,'loss':float(loss.detach()),**metrics})
            if step%10==0 or step==steps:
                model.eval();vrng=np.random.default_rng(901);scores=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    for _ in range(4):
                        x,y=valid.batch(vrng,1024,device);vl,vm=loss_on(x,y);scores.append(float(vl) if mode=='transfer' else vm['ce'])
                score=float(np.mean(scores))
                if best is None or score<best:best=score;model.save(out/'best.pt')
                model.save(out/'latest_model.pt')
                atomic_checkpoint({'identity':identity,'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,
                                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
                                   'step':step,'best':best,'history':history},out/'last.pt')
    finally:
        for h in handles:h.remove()
    result={'mode':mode,'steps':step,'status':'completed' if step==steps else 'budget_exhausted','selection_loss':best,'identity':identity,'history':history}
    atomic_json(result,out/'result.json');return result


def evaluate(job):
    device=job.get('device','cuda');data=TokenDocuments(job['data'],job['donor'],'test');rows=[]
    for name,path in [('teacher',None),('untrained',job['untrained']),('converted',job['converted']),('lora',job['lora'])]:
        model=load_donor(job['donor'],device) if path is None else ConvertedLM.load(path,device);model.eval()
        for length in (1024,2048,8192):
            rng=np.random.default_rng(903);losses=[];latencies=[];cache_sizes=[]
            for _ in range(4):
                x,y=data.batch(rng,length,device);sync(device);start=time.monotonic()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    cache=StaticCache(model.config,max_cache_len=length) if path is None else EnergyStaticCache(model.base.config,length)
                    r=model(x,use_cache=True,past_key_values=cache);sync(device);prefill=time.monotonic()-start
                    losses.append(float(torch.nn.functional.cross_entropy(r.logits.float().flatten(0,1),y.flatten())))
                    cache_sizes.append(cache_bytes(r.past_key_values));cache.reset();r=model(x[:,:-32],use_cache=True,past_key_values=cache);cache=r.past_key_values;sync(device);start=time.monotonic()
                    for pos in range(length-32,length):
                        r=model(x[:,pos:pos+1],past_key_values=cache,use_cache=True);cache=r.past_key_values
                    sync(device);latencies.append((time.monotonic()-start)/32)
            rows.append({'model':name,'context':length,'decode_initial_context':length-32,'nll':float(np.mean(losses)),'decode_seconds_per_token':float(np.median(latencies)),'cache_bytes':max(cache_sizes),'cache_policy':'preallocated'})
        del model
    converted=next(r for r in rows if r['model']=='converted' and r['context']==8192);control=next(r for r in rows if r['model']=='lora' and r['context']==8192)
    quality=converted['nll']/control['nll']<=1.02;cache=1-converted['cache_bytes']/control['cache_bytes'];speed=control['decode_seconds_per_token']/converted['decode_seconds_per_token']-1
    result={'rows':rows,'gate':{'quality':quality,'cache_reduction':cache,'throughput_gain':speed,'continue':quality and (cache>=.15 or speed>=.15)},
            'scope':'single-seed conversion; not native byte scaling; no automatic promotion'}
    atomic_json(result,job['out']);return result
