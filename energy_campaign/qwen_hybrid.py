"""Qwen2 energy/local conversion. Explicit unpadded causal streams, no global KV cache."""
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb,repeat_kv
from energy_real.model import Config,FastFunctionMixer
from energy_real.runtime import atomic_checkpoint

class LocalQwenAttention(nn.Module):
    def __init__(self,attention,window):
        super().__init__();self.attention=attention;self.window=window
    def forward(self,x,position_embeddings,state=None):
        a=self.attention;b,t,_=x.shape
        q=a.q_proj(x).view(b,t,-1,a.head_dim).transpose(1,2)
        k=a.k_proj(x).view(b,t,-1,a.head_dim).transpose(1,2)
        v=a.v_proj(x).view(b,t,-1,a.head_dim).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        previous=0 if state is None else state[0].shape[2]
        if state is not None:k=torch.cat((state[0],k),2);v=torch.cat((state[1],v),2)
        qp=torch.arange(t,device=x.device)+previous;kp=torch.arange(k.shape[2],device=x.device)
        mask=(kp[None,:]<=qp[:,None]) & (kp[None,:]>qp[:,None]-self.window)
        y=torch.nn.functional.scaled_dot_product_attention(q,repeat_kv(k,a.num_key_value_groups),repeat_kv(v,a.num_key_value_groups),attn_mask=mask,dropout_p=0.,scale=a.scaling)
        return a.o_proj(y.transpose(1,2).reshape(b,t,-1)),(k[:,:,-self.window:].clone(),v[:,:,-self.window:].clone())

class QwenHybrid(nn.Module):
    def __init__(self,base,window=128,energy_layers=None,slots=16,donor=None):
        super().__init__()
        if base.config.model_type!='qwen2':raise ValueError('This adapter supports Qwen2/Qwen2.5 only')
        if type(window)is not int or window<1:raise ValueError('Positive window required')
        self.base=base;self.window=window;self.slots=slots;self.donor=donor
        self.energy_layers=tuple(i for i in range(base.config.num_hidden_layers) if i%3!=2) if energy_layers is None else tuple(energy_layers)
        if len(set(self.energy_layers))!=len(self.energy_layers) or any(i<0 or i>=base.config.num_hidden_layers for i in self.energy_layers):raise ValueError('Invalid energy layers')
        for p in base.parameters():p.requires_grad_(False)
        for i,layer in enumerate(base.model.layers):
            old=layer.self_attn
            if i in self.energy_layers:
                cfg=Config(width=base.config.hidden_size,heads=base.config.num_attention_heads,slots=slots,ff_width=base.config.intermediate_size,chunk=16,energy_projection_fp32=True)
                # Trainable energy weights stay FP32 even when the frozen donor is BF16.
                layer.self_attn=FastFunctionMixer(cfg).to(old.q_proj.weight.device)
            else:layer.self_attn=LocalQwenAttention(old,window)
    def forward(self,input_ids,state=None):
        if input_ids.ndim!=2 or input_ids.shape[1]<1:raise ValueError('Nonempty unpadded token batch required')
        offset=0 if state is None else state['offset']
        states=[None]*len(self.base.model.layers) if state is None else state['layers']
        if len(states)!=len(self.base.model.layers):raise ValueError('State layer mismatch')
        x=self.base.model.embed_tokens(input_ids)
        pos=torch.arange(offset,offset+input_ids.shape[1],device=x.device)[None].expand(input_ids.shape[0],-1)
        rotary=self.base.model.rotary_emb(x,pos);updated=[]
        for i,(layer,s) in enumerate(zip(self.base.model.layers,states)):
            z=layer.input_layernorm(x)
            if i in self.energy_layers:y,s=layer.self_attn(z,s)
            else:y,s=layer.self_attn(z,rotary,s)
            x=x+y.to(x.dtype);x=x+layer.mlp(layer.post_attention_layernorm(x));updated.append(s)
        return self.base.lm_head(self.base.model.norm(x)),{'offset':offset+input_ids.shape[1],'layers':updated}
    def save(self,path):
        atomic_checkpoint({'format':'qwen_energy_local_v1','config':self.base.config.to_dict(),'window':self.window,'energy_layers':self.energy_layers,'slots':self.slots,'donor':self.donor,'weights':self.state_dict()},path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d['format']!='qwen_energy_local_v1':raise ValueError('Unknown format')
        c=dict(d['config']);kind=c.pop('model_type');base=AutoModelForCausalLM.from_config(AutoConfig.for_model(kind,**c))
        m=cls(base,d['window'],d['energy_layers'],d['slots'],d['donor']);m.load_state_dict(d['weights']);return m
