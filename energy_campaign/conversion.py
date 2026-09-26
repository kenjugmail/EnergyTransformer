"""Version-pinned Llama-family conversion; causal unpadded streams only in v1."""
from dataclasses import asdict
import math
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache,StaticCache
from energy_real.model import Config,FastFunctionMixer
from energy_real.runtime import atomic_checkpoint
from .spec import MODEL


class EnergyCache(DynamicCache):
    def __init__(self,config=None):
        super().__init__(config=config);self.energy={}
    def reorder_cache(self,beam_idx):
        super().reorder_cache(beam_idx)
        self.energy={i:tuple(t.index_select(0,beam_idx.to(t.device)) for t in s) for i,s in self.energy.items()}
    def reset(self):
        super().reset();self.energy.clear()
    def crop(self,*args,**kwargs):
        raise NotImplementedError('Recurrent state cannot roll back; rebuild from prefix')


class EnergyStaticCache(StaticCache):
    """Preallocated attention plus fixed-size recurrent state for fair serving."""
    def __init__(self,config,max_cache_len):
        super().__init__(config=config,max_cache_len=max_cache_len);self.energy={}
    def reset(self):
        super().reset();self.energy.clear()
    def reorder_cache(self,beam_idx):
        super().reorder_cache(beam_idx)
        self.energy={i:tuple(t.index_select(0,beam_idx.to(t.device)) for t in s) for i,s in self.energy.items()}
    def crop(self,*args,**kwargs):
        raise NotImplementedError('Recurrent state cannot roll back; rebuild from prefix')


class EnergyAttention(nn.Module):
    def __init__(self,config,index,slots=16):
        super().__init__();self.layer_idx=index
        self.mixer=FastFunctionMixer(Config(width=config.hidden_size,heads=config.num_attention_heads,
                                           slots=slots,layers=1,ff_width=config.intermediate_size,chunk=16))
    def forward(self,hidden_states,position_embeddings=None,attention_mask=None,past_key_values=None,cache_position=None,**kwargs):
        if past_key_values is not None and not isinstance(past_key_values,(EnergyCache,EnergyStaticCache)):raise TypeError('EnergyCache required')
        state=None if past_key_values is None else past_key_values.energy.get(self.layer_idx)
        y,final=self.mixer(hidden_states,state)
        if past_key_values is not None:past_key_values.energy[self.layer_idx]=final
        return y,None


class LowRankLinear(nn.Module):
    def __init__(self,linear,rank=8):
        super().__init__();self.base=linear;self.rank=rank
        self.a=nn.Parameter(torch.empty(rank,linear.in_features,device=linear.weight.device,dtype=linear.weight.dtype))
        self.b=nn.Parameter(torch.zeros(linear.out_features,rank,device=linear.weight.device,dtype=linear.weight.dtype))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+(x@self.a.T)@self.b.T


def add_lora(model,rank=8):
    names=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in ('q_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
            parent,_,leaf=name.rpartition('.');setattr(model.get_submodule(parent),leaf,LowRankLinear(module,rank));names.append(name)
    return names


class ConvertedLM(nn.Module):
    def __init__(self,base,layers=tuple(MODEL['layers']),slots=16,lora=False):
        super().__init__();self.base=base;self.layers=tuple(layers);self.slots=slots;self.lora=lora
        if base.config.model_type!='llama':raise ValueError('Llama architecture required')
        if len(set(layers))!=len(layers) or any(i<=0 or i>=base.config.num_hidden_layers for i in layers):raise ValueError('Keep first attention layer; invalid replacement indices')
        for p in base.parameters():p.requires_grad_(False)
        for i in layers:
            old=base.model.layers[i].self_attn;replacement=EnergyAttention(base.config,i,slots)
            replacement.to(device=old.q_proj.weight.device,dtype=old.q_proj.weight.dtype)
            base.model.layers[i].self_attn=replacement
        if lora:add_lora(base)
    def forward(self,input_ids,*,past_key_values=None,use_cache=False,attention_mask=None,**kwargs):
        if attention_mask is not None and not bool(attention_mask.all()):raise ValueError('Padding/packed masks unsupported; pass document-contained unpadded batches')
        if kwargs:raise ValueError(f'Unsupported conversion forward options: {sorted(kwargs)}')
        if past_key_values is not None and (not use_cache or not isinstance(past_key_values,(EnergyCache,EnergyStaticCache))):raise ValueError('Invalid mixed cache')
        if use_cache and past_key_values is None:past_key_values=EnergyCache(config=self.base.config)
        return self.base(input_ids=input_ids,past_key_values=past_key_values,use_cache=use_cache)
    def save(self,path):
        atomic_checkpoint({'format':'energy_conversion_v1','base_config':self.base.config.to_dict(),'layers':self.layers,
                           'slots':self.slots,'lora':self.lora,'donor':MODEL,'weights':self.state_dict()},path)
    @classmethod
    def load(cls,path,device='cpu'):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['format']!='energy_conversion_v1':raise ValueError('Unknown conversion checkpoint')
        c=dict(data['base_config']);kind=c.pop('model_type');cfg=AutoConfig.for_model(kind,**c)
        model=cls(AutoModelForCausalLM.from_config(cfg),data['layers'],data['slots'],data['lora'])
        model.load_state_dict(data['weights'],strict=True);return model.to(device)


def distillation_loss(student_logits,teacher_logits,targets):
    s=student_logits.float();t=teacher_logits.detach().float()
    ce=torch.nn.functional.cross_entropy(s.flatten(0,1),targets.flatten())
    kl=torch.nn.functional.kl_div(s.log_softmax(-1),t.softmax(-1),reduction='sum')/targets.numel()
    return ce+kl,{'ce':float(ce.detach()),'kl':float(kl.detach())}


def cache_bytes(cache):
    tensors=[]
    for layer in cache.layers:
        for name in ('keys','values'):
            x=getattr(layer,name,None)
            if isinstance(x,torch.Tensor):tensors.append(x)
    if isinstance(cache,(EnergyCache,EnergyStaticCache)):tensors.extend(t for state in cache.energy.values() for t in state)
    return sum(t.numel()*t.element_size() for t in tensors)
