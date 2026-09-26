"""Text-only Qwen3.5 energy/local conversion; requires isolated Transformers5.17 runtime."""
from pathlib import Path
import json,copy
import torch
from torch import nn
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM,apply_rotary_pos_emb,repeat_kv
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from energy_real.model import Config,FastFunctionMixer
from energy_real.runtime import atomic_checkpoint

DONOR={'id':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17'}

def load_text_donor(path):
    cfg=Qwen3_5TextConfig(**json.loads((Path(path)/'config.json').read_text())['text_config']);cfg._attn_implementation='sdpa'
    with torch.device('meta'):model=Qwen3_5ForCausalLM(cfg)
    weights={}
    for p in Path(path).glob('*.safetensors'):
        with safe_open(p,framework='pt') as f:
            for key in f.keys():
                if key.startswith('model.language_model.'):weights[key.replace('model.language_model.','model.',1)]=f.get_tensor(key)
                elif key=='lm_head.weight':weights[key]=f.get_tensor(key)
    if 'lm_head.weight' not in weights:weights['lm_head.weight']=weights['model.embed_tokens.weight']
    model.load_state_dict(weights,strict=True,assign=True);model.tie_weights()
    # Rotary buffers are nonpersistent and must be materialized independently.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(cfg)
    return model.eval()

class LocalAttention(nn.Module):
    def __init__(self,attention,window):super().__init__();self.attention=attention;self.window=window
    def forward(self,x,rotary,state=None):
        a=self.attention;b,t,_=x.shape
        q,gate=a.q_proj(x).view(b,t,-1,2*a.head_dim).chunk(2,dim=-1)
        q=a.q_norm(q).transpose(1,2);k=a.k_norm(a.k_proj(x).view(b,t,-1,a.head_dim)).transpose(1,2);v=a.v_proj(x).view(b,t,-1,a.head_dim).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*rotary);previous=0 if state is None else state[0].shape[2]
        if state is not None:k=torch.cat((state[0],k),2);v=torch.cat((state[1],v),2)
        qp=torch.arange(t,device=x.device)+previous;kp=torch.arange(k.shape[2],device=x.device)
        mask=(kp[None,:]<=qp[:,None])&(kp[None,:]>qp[:,None]-self.window)
        y=torch.nn.functional.scaled_dot_product_attention(q,repeat_kv(k,a.num_key_value_groups),repeat_kv(v,a.num_key_value_groups),attn_mask=mask,dropout_p=0.,scale=a.scaling)
        y=y.transpose(1,2).reshape(b,t,-1)*gate.reshape(b,t,-1).sigmoid()
        return a.o_proj(y),(k[:,:,-self.window:].clone(),v[:,:,-self.window:].clone())

class Qwen35Hybrid(nn.Module):
    def __init__(self,base,window=128,slots=16,donor=None,energy_layers=None,lora=False):
        super().__init__();self.base=base;self.window=window;self.slots=slots;self.donor=donor
        if window<1:raise ValueError('Positive local window required')
        self.energy_layers=tuple(i for i in range(len(base.model.layers)) if i%3!=2) if energy_layers is None else tuple(energy_layers)
        self.lora=lora
        if len(set(self.energy_layers))!=len(self.energy_layers) or any(i<0 or i>=len(base.model.layers) for i in self.energy_layers):raise ValueError('Invalid replacement layers')
        originals={i:layer.self_attn for i,layer in enumerate(base.model.layers) if hasattr(layer,'self_attn')}
        if not originals:raise ValueError('At least one donor attention layer required')
        for p in base.parameters():p.requires_grad_(False)
        for i,layer in enumerate(base.model.layers):
            device=next(layer.parameters()).device
            if i in self.energy_layers:
                mixer=FastFunctionMixer(Config(width=base.config.hidden_size,heads=base.config.num_attention_heads,slots=slots,ff_width=base.config.intermediate_size,chunk=16,energy_projection_fp32=True)).to(device)
            else:
                nearest=min(originals,key=lambda j:(abs(j-i),j))
                attention=copy.deepcopy(originals[nearest])
                if lora:
                    from .conversion import LowRankLinear
                    for name in ('q_proj','v_proj','o_proj'):setattr(attention,name,LowRankLinear(getattr(attention,name),8))
                mixer=LocalAttention(attention,window)
            if hasattr(layer,'linear_attn'):del layer.linear_attn
            if hasattr(layer,'self_attn'):del layer.self_attn
            layer.mixer=mixer
    def mix(self,index,x,rotary,state=None):
        layer=self.base.model.layers[index]
        return layer.mixer(x,state) if index in self.energy_layers else layer.mixer(x,rotary,state)
    def forward(self,input_ids,state=None):
        if input_ids.ndim!=2 or input_ids.shape[1]<1:raise ValueError('Unpadded causal token batches required')
        offset=0 if state is None else state['offset'];states=[None]*len(self.base.model.layers) if state is None else state['layers']
        if len(states)!=len(self.base.model.layers):raise ValueError('Invalid state')
        x=self.base.model.embed_tokens(input_ids);pos=torch.arange(offset,offset+input_ids.shape[1],device=x.device)[None].expand(input_ids.shape[0],-1);rotary=self.base.model.rotary_emb(x,pos[None].expand(3,-1,-1));updated=[]
        for i,(layer,s) in enumerate(zip(self.base.model.layers,states)):
            z=layer.input_layernorm(x)
            y,s=self.mix(i,z,rotary,s)
            x=x+y.to(x.dtype);x=x+layer.mlp(layer.post_attention_layernorm(x));updated.append(s)
        return self.base.lm_head(self.base.model.norm(x)),{'offset':offset+input_ids.shape[1],'layers':updated}
    def save(self,path):
        atomic_checkpoint({'format':'qwen35_energy_local_v1','config':self.base.config.to_dict(),'window':self.window,'slots':self.slots,'donor':self.donor,'energy_layers':self.energy_layers,'lora':self.lora,'weights':self.state_dict()},path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d['format']!='qwen35_energy_local_v1':raise ValueError('Wrong checkpoint')
        m=cls(Qwen3_5ForCausalLM(Qwen3_5TextConfig(**d['config'])),d['window'],d['slots'],d['donor'],d['energy_layers'],d['lora']);m.load_state_dict(d['weights'],assign=True);m.base.tie_weights();return m
