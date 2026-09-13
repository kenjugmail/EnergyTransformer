"""V19 scaling candidate and matched small controls (not a published GDN reproduction).

The energy variant carries only diagonal quadratic P/S coefficients. Dense
projected v18 updates remain a separate experiment. This implementation supports
cold/chunked/streamed execution and segment resets, with FP32 recurrent arithmetic.
The parallel scan is a reference, NOT a fused GPU kernel or constant-training-memory
algorithm. Softmax and delta variants are explicit non-energy controls.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ScaleConfig:
    kind: str = 'energy'
    vocab: int = 128
    width: int = 96
    layers: int = 2
    heads: int = 4
    slots: int = 16
    ff_width: int = 192
    prior: float = .25
    chunk: int = 128
    activation_checkpointing: bool = False
    cpu_scan: bool = False

    def validate(self):
        if self.kind not in ('energy', 'delta', 'softmax'):
            raise ValueError('Unknown mixer')
        if any(type(v) is not int for v in (self.vocab,self.width,self.layers,self.heads,self.slots,self.ff_width,self.chunk)):
            raise ValueError('Dimensions must be integers')
        if not math.isfinite(self.prior):raise ValueError('Finite prior required')
        if min(self.vocab,self.width,self.layers,self.heads,self.slots,self.ff_width,self.chunk)<1:
            raise ValueError('Positive dimensions required')
        if self.width % self.heads or (self.width // self.heads) % 2 or self.prior <= 0:
            raise ValueError('Even head width and positive prior required')


def diagonal_scan(f: Tensor, b: Tensor, initial: Tensor) -> Tensor:
    """Inclusive affine scan, time axis 1. No division by cumulative products."""
    a, x = f, b
    distance = 1
    while distance < x.shape[1]:
        x = torch.cat((x[:, :distance], x[:, distance:] + a[:, distance:, ..., None] * x[:, :-distance]), 1)
        a = torch.cat((a[:, :distance], a[:, distance:] * a[:, :-distance]), 1)
        distance *= 2
    return x + a[..., None] * initial[:, None]


def transported_scan(f: Tensor, b: Tensor, initial: Tensor, strides: Tensor) -> Tensor:
    """Reduce a fixed cyclic permutation/diagonal recurrence to co-moving coordinates."""
    batch,t,h,k = f.shape
    time=torch.arange(1,t+1,device=f.device)
    slots=torch.arange(k,device=f.device)
    co=(slots[None,None,:]+time[:,None,None]*strides[None,:,None]) % k
    back=(slots[None,None,:]-time[:,None,None]*strides[None,:,None]) % k
    ico=co[None].expand(batch,-1,-1,-1)
    fc=f.gather(3,ico)
    bc=b.gather(3,ico[...,None].expand(-1,-1,-1,-1,b.shape[-1]))
    yc=diagonal_scan(fc,bc,initial)
    return yc.gather(3,back[None,...,None].expand(batch,-1,-1,-1,b.shape[-1]))


class FunctionMixer(nn.Module):
    def __init__(self,c:ScaleConfig):
        super().__init__(); self.c=c; self.v=c.width//c.heads
        self.projection=nn.Linear(c.width,c.heads*(3*c.slots+2*self.v+1))
        self.second=nn.Linear(2*c.width,c.heads*c.slots)
        self.output=nn.Linear(c.width,c.width)
        self.register_buffer('strides',torch.tensor(([0,1,3,7]*((c.heads+3)//4))[:c.heads])%c.slots)
        # A range of initialized retention times, NOT a guarantee of useful recall.
        half_life=torch.logspace(math.log10(4),math.log10(256),c.heads)
        f=torch.exp(-math.log(2)/half_life).clamp(.001,.999)
        with torch.no_grad():
            bias=self.projection.bias.view(c.heads,-1)
            bias[:,2*c.slots+self.v:3*c.slots+self.v]=torch.logit(f)[:,None]
        self.second_mix=nn.Parameter(torch.zeros(c.heads))

    def initial(self,batch:int,device):
        c=self.c
        if c.kind=='delta':return (torch.zeros(batch,c.heads,c.slots,self.v,device=device),)
        return (torch.full((batch,c.heads,c.slots),c.prior,device=device),
                torch.zeros(batch,c.heads,c.slots,self.v,device=device))

    def forward(self,x:Tensor,state=None,reset:Tensor|None=None):
        c=self.c;b,t,_=x.shape
        if state is None:state=self.initial(b,x.device)
        p=self.projection(x).view(b,t,c.heads,-1)
        # State algebra stays float32 even when outer projections use autocast.
        with torch.autocast(device_type=x.device.type,enabled=False):
            k,q,v,f,w,g=torch.split(p.float(),[c.slots,c.slots,self.v,c.slots,1,self.v],-1)
            k=F.normalize(k,dim=-1,eps=1e-6);q=F.normalize(q,dim=-1,eps=1e-6)
            f=torch.sigmoid(f).clamp(.0001,.99999)
            if reset is not None:f=f*(~reset[:,:,None,None]).float()
            w=F.softplus(w).clamp(max=20)+1e-4
            if c.kind=='energy':
                pp,ss=(a.float() for a in state)
                initial=torch.cat(((pp-c.prior)[...,None],ss),-1)
                factor=torch.cat(((w*k.square())[...,None],w[...,None]*k[...,None]*v[...,None,:]),-1)
                if c.cpu_scan and x.device.type == 'cpu':
                    from .scan_backend import available, cpu_scan
                    packed = cpu_scan(f,factor,initial,self.strides) if available(f) else transported_scan(f,factor,initial,self.strides)
                else:
                    packed=transported_scan(f,factor,initial,self.strides)
                precision=packed[...,0]+c.prior;info=packed[...,1:]
                mode=info/precision[...,None]
                final=(precision[:,-1].contiguous().clone(),info[:,-1].contiguous().clone())
            else:
                # Serial gated-delta reference. No claim of optimized/published GDN.
                m=state[0].float();history=[]
                beta=(w/(1+w)).squeeze(-1)
                for i in range(t):
                    old=m*f[:,i].mean(-1)[...,None,None]
                    residual=v[:,i]-torch.einsum('bhk,bhkv->bhv',k[:,i],old)
                    m=old+beta[:,i,...,None,None]*k[:,i,...,None]*residual[...,None,:]
                    history.append(m)
                mode=torch.stack(history,1);final=(m.contiguous().clone(),)
            r=torch.einsum('bthk,bthkv->bthv',q,mode)
            r=r*torch.rsqrt(r.square().mean(-1,keepdim=True)+1e-5)
            e=r.reshape(b,t,c.width)
        q2=self.second(torch.cat((x,e.to(x.dtype)),dim=-1))
        with torch.autocast(device_type=x.device.type,enabled=False):
            q2=F.normalize(q2.float().view(b,t,c.heads,c.slots),dim=-1,eps=1e-6)
            r2=torch.einsum('bthk,bthkv->bthv',q2,mode)
            r2=r2*torch.rsqrt(r2.square().mean(-1,keepdim=True)+1e-5)
            r=r+torch.sigmoid(self.second_mix)[None,None,:,None]*r2
            r=(r*torch.sigmoid(g)).reshape(b,t,c.width)
        return self.output(r.to(x.dtype)),final


class SoftmaxMixer(nn.Module):
    def __init__(self,c:ScaleConfig):
        super().__init__();self.c=c;self.v=c.width//c.heads
        self.qkv=nn.Linear(c.width,3*c.width);self.output=nn.Linear(c.width,c.width)
        self.register_buffer('freq',10000**(-torch.arange(0,self.v,2).float()/self.v))
    def rope(self,x,pos):
        angle=pos.float()[...,None]*self.freq
        co,si=angle.cos()[:,None],angle.sin()[:,None]
        a,b=x[...,0::2],x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),-1).flatten(-2).to(x.dtype)
    def forward(self,x,state=None,reset=None):
        b,t,d=x.shape;h=self.c.heads
        offset=0 if state is None else state[0].shape[2]
        pos=torch.arange(offset,offset+t,device=x.device)[None].expand(b,-1)
        q,k,v=self.qkv(x).view(b,t,3,h,self.v).permute(2,0,3,1,4).unbind(0)
        q,k=self.rope(q,pos),self.rope(k,pos)
        if state is None and reset is None:
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
            seg=torch.zeros(b,t,device=x.device,dtype=torch.long)
            return self.output(y.transpose(1,2).reshape(b,t,d)),(k.contiguous(),v.contiguous(),seg)
        if reset is None:reset=torch.zeros(b,t,device=x.device,dtype=torch.bool)
        seg=reset.long().cumsum(1)
        if state is not None:
            ko,vo,sego=state;seg=seg+sego[:,-1:]
            k,v=torch.cat((ko,k),2),torch.cat((vo,v),2)
            allseg=torch.cat((sego,seg),1)
        else:allseg=seg
        allowed=(torch.arange(k.shape[2],device=x.device)[None,None,:]<=pos[:,:,None]) & (seg[:,:,None]==allseg[:,None,:])
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[:,None],dropout_p=0.)
        result=y.transpose(1,2).reshape(b,t,d)
        return self.output(result),(k.contiguous(),v.contiguous(),allseg.contiguous())


class ScaleBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm1=nn.LayerNorm(c.width);self.norm2=nn.LayerNorm(c.width)
        self.mixer=SoftmaxMixer(c) if c.kind=='softmax' else FunctionMixer(c)
        self.ff=nn.Sequential(nn.Linear(c.width,c.ff_width),nn.GELU(),nn.Linear(c.ff_width,c.width))
    def forward(self,x,state,reset):
        y,s=self.mixer(self.norm1(x),state,reset)
        x=x+y
        return x+self.ff(self.norm2(x)),s


class ScaleLM(nn.Module):
    def __init__(self,config:ScaleConfig):
        super().__init__();config.validate();self.config=config
        self.embedding=nn.Embedding(config.vocab,config.width)
        self.blocks=nn.ModuleList([ScaleBlock(config) for _ in range(config.layers)])
        self.norm=nn.LayerNorm(config.width)
        self.head=nn.Linear(config.width,config.vocab,bias=False)
        self.head.weight=self.embedding.weight
        nn.init.normal_(self.embedding.weight,std=.02)
    def _chunk(self,tokens,state,reset):
        x=self.embedding(tokens)
        new=[]
        for i,block in enumerate(self.blocks):
            if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Non-reentrant checkpoint supports nested tuple outputs/state.
                x,s=checkpoint(block,x,state[i],reset,use_reentrant=False)
            else:x,s=block(x,state[i],reset)
            new.append(s)
        return self.head(self.norm(x)),tuple(new)
    def forward(self,tokens:Tensor,state=None,reset:Tensor|None=None):
        if tokens.ndim!=2 or tokens.shape[1]<1:raise ValueError('tokens must be nonempty [batch,time]')
        if tokens.dtype!=torch.long:raise TypeError('integer token IDs required')
        if reset is not None and (reset.shape!=tokens.shape or reset.dtype!=torch.bool):raise ValueError('Boolean reset mask must match tokens')
        if state is None:state=tuple(None for _ in self.blocks)
        if len(state)!=len(self.blocks):raise ValueError('Wrong layer count in state')
        logits=[]
        # Chunking bounds *inference* temporary memory; autograd still spans chunks.
        for start in range(0,tokens.shape[1],self.config.chunk):
            y,state=self._chunk(tokens[:,start:start+self.config.chunk],state,None if reset is None else reset[:,start:start+self.config.chunk])
            logits.append(y)
        return torch.cat(logits,1),state
    def parameters_count(self):return sum(p.numel() for p in self.parameters())


def state_bytes(state):
    return sum(t.numel()*t.element_size() for layer in state for t in layer)


def detached_state(state):
    return tuple(tuple(x.detach().clone() for x in layer) for layer in state)
