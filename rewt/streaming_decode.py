"""Owned inference caches for matched streaming and full-attention decoding.

Full attention writes preallocated KV buffers without concatenating or cloning
its growing history. Window attention and delta memory carry bounded storage.
Sessions reject weight mutation and consume inputs only in causal order.
"""
import torch
from torch.nn import functional as F
from .streaming import Attention, rotate


class StreamingSession:
    def __init__(self, model, *, batch=1, capacity=8192, rounds=None):
        if model.training:
            raise ValueError('call eval before constructing an inference session')
        if type(batch) is not int or batch<1 or type(capacity) is not int or capacity<1:
            raise ValueError('positive batch and capacity required')
        self.model=model;self.batch=batch;self.capacity=capacity;self.position=0
        self.rounds=model.config.rounds if rounds is None else rounds
        if type(self.rounds) is not int or self.rounds<0:raise ValueError('invalid rounds')
        self.signature=model.signature();self.layers=[None]*len(model.blocks)
        self.energy_constants=None
        self.prepared_energy=None
        if model.energy_head is not None and self.rounds:
            if hasattr(model.energy_head,'prepare_inference'):
                self.prepared_energy=model.energy_head.prepare_inference(self.rounds)
            else:
                self._prepare_energy_constants()
        p=model.embedding.weight;c=model.config;h=c.heads;dh=c.width//h
        self.buffers=[]
        for block in model.blocks:
            if isinstance(block.mixer,Attention):
                length=c.window-1 if c.window else capacity
                self.buffers.append((p.new_empty(batch,h,length,dh),p.new_empty(batch,h,length,dh)))
            else:self.buffers.append(None)

    def _prepare_energy_constants(self):
        with torch.no_grad():
            head=self.model.energy_head;raw=head.raw_frequency
            norm=raw.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            frequency=2*raw*norm.tanh()/norm
            precision=.15+1.85*head.raw_precision.sigmoid()
            self.energy_constants=(frequency,precision,frequency.square().sum(-1),precision.max())

    @torch.inference_mode()
    def consume(self,tokens):
        model=self.model;c=model.config
        if model.training or model.signature()!=self.signature:
            raise ValueError('session weights or evaluation mode changed')
        if tokens.ndim!=2 or tokens.shape[0]!=self.batch or not tokens.shape[1] or tokens.dtype!=torch.long:
            raise ValueError('expected nonempty [session_batch,time] int64')
        if tokens.device!=model.embedding.weight.device or bool(((tokens<0)|(tokens>255)).any()):
            raise ValueError('token device or vocabulary mismatch')
        t=tokens.shape[1]
        if self.position+t>self.capacity:raise ValueError('session capacity exceeded')
        x=model.embedding(tokens);b=tokens.shape[0];d=c.width;h=c.heads
        positions=torch.arange(self.position,self.position+t,device=x.device)
        for i,block in enumerate(model.blocks):
            normalized=block.norm1(x)
            if isinstance(block.mixer,Attention):
                mixer=block.mixer;q,k,v=mixer.qkv(normalized).view(b,t,3,h,d//h).unbind(2)
                q,k,v=(a.transpose(1,2) for a in (q,k,v));q,k=rotate(q,positions),rotate(k,positions)
                kb,vb=self.buffers[i]
                if not c.window:
                    kb[:,:,self.position:self.position+t].copy_(k);vb[:,:,self.position:self.position+t].copy_(v)
                    k,v=kb[:,:,:self.position+t],vb[:,:,:self.position+t]
                    past=self.position
                else:
                    past=min(self.position,c.window-1)
                    if past:k,v=torch.cat((kb[:,:,:past],k),-2),torch.cat((vb[:,:,:past],v),-2)
                if t==1:
                    output=F.scaled_dot_product_attention(q,k,v)
                elif not past and not c.window:
                    output=F.scaled_dot_product_attention(q,k,v,is_causal=True)
                else:
                    qi=torch.arange(past,past+t,device=x.device)[:,None];ki=torch.arange(past+t,device=x.device)[None]
                    mask=ki<=qi
                    if c.window:mask=mask & (ki>qi-c.window)
                    output=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                if c.window:
                    keep=min(c.window-1,k.shape[-2])
                    if keep:kb[:,:,:keep].copy_(k[:,:,-keep:]);vb[:,:,:keep].copy_(v[:,:,-keep:])
                y=mixer.out(output.transpose(1,2).reshape(b,t,d))
            else:
                y,self.layers[i]=block.mixer(normalized,self.layers[i],self.position,return_state=True)
            x=x+y;a,g=block.up(block.norm2(x)).chunk(2,-1);x=x+block.down(F.silu(a)*g)
        x=model.norm(x)
        if self.prepared_energy is not None:
            x=self.prepared_energy._apply_validated(x)
        elif self.energy_constants is not None:
            frequency,precision,norm2,maximum=self.energy_constants
            head=model.energy_head
            amplitude=head.amplitude(x).tanh()/c.modes**.5
            eta=.95/(maximum+amplitude.abs()@norm2)[...,None]
            delta=torch.zeros_like(x)
            for _ in range(self.rounds):
                gradient=precision*delta-(amplitude*F.linear(delta,frequency,head.phase).sin())@frequency
                delta=delta-eta*gradient
            x=x+delta
        self.position+=t
        return F.linear(x,model.embedding.weight)

    def allocated_state_bytes(self):
        tensors=[]
        for pair in self.buffers+self.layers:
            if pair is not None:tensors.extend(pair)
        storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
        return sum(storages.values())
