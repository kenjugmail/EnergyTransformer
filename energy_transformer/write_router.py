"""V20 token-derived, protected energy memory; a task-specialized research model.

Inference accepts token IDs and boundary resets only. Local operation/key/value
labels are TRAINING targets, never an inference argument. Memory addresses and
value channels have supplied cardinalities (32 and 17). This is not an unrestricted
language-model backbone. A three-embedding local shift register is retained and
explicitly counted: representing it as a quadratic's linear coefficients does not
make that local activation buffer cease to exist.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn, Tensor
from torch.nn import functional as F

@dataclass(frozen=True)
class RouterConfig:
    kind: str = 'hard'
    vocab: int = 128
    width: int = 64
    hidden: int = 128
    slots: int = 32
    values: int = 17
    prior: float = .25
    def validate(self):
        if self.kind not in ('hard','soft','attention'): raise ValueError('unknown kind')
        if min(self.vocab,self.width,self.hidden,self.slots,self.values)<1 or self.prior<=0: raise ValueError('invalid dimensions')
        if self.width%4: raise ValueError('width must be divisible by four')


def affine_prefix(f:Tensor,b:Tensor,initial:Tensor)->Tensor:
    """Pure Torch associative scan; inclusive time dimension one."""
    a,x=f,b;lag=1
    while lag<x.shape[1]:
        x=torch.cat((x[:,:lag],x[:,lag:]+a[:,lag:, :,None]*x[:,:-lag]),1)
        a=torch.cat((a[:,:lag],a[:,lag:]*a[:,:-lag]),1)
        lag*=2
    return x+a[...,None]*initial[:,None]


def choice(logits:Tensor,hard:bool)->Tensor:
    p=logits.softmax(-1)
    if not hard:return p
    z=F.one_hot(p.argmax(-1),p.shape[-1]).to(p)
    return z+(p-p.detach()) # straight-through, biased surrogate gradient


class WriteRouterLM(nn.Module):
    def __init__(self,config:RouterConfig):
        super().__init__();config.validate();self.config=config;c=config
        self.embedding=nn.Embedding(c.vocab,c.width)
        self.local=nn.Sequential(nn.Linear(4*c.width,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.width),nn.LayerNorm(c.width))
        self.operations=nn.Linear(c.width,4) # HOLD, SET, DELETE, QUERY
        self.address=nn.Linear(c.width,c.slots)
        self.value=nn.Linear(c.width,c.values)
        self.read=nn.Linear(c.values+1,c.width)
        self.ff=nn.Sequential(nn.LayerNorm(c.width),nn.Linear(c.width,2*c.width),nn.GELU(),nn.Linear(2*c.width,c.width))
        self.head=nn.Linear(c.width,c.vocab)
        if c.kind=='attention':
            self.attn_norm=nn.ModuleList([nn.LayerNorm(c.width) for _ in range(2)])
            self.qkv=nn.ModuleList([nn.Linear(c.width,3*c.width) for _ in range(2)])
            self.attn_out=nn.ModuleList([nn.Linear(c.width,c.width) for _ in range(2)])
        self.register_buffer('freq',10000**(-torch.arange(0,c.width//4,2).float()/(c.width//4)))

    def initial(self,batch:int,device=None):
        c=self.config;device=device or self.embedding.weight.device
        trace=torch.zeros(batch,3,c.width,device=device)
        if c.kind=='attention':return (trace,None)
        return (trace,torch.full((batch,c.slots),c.prior,device=device),torch.zeros(batch,c.slots,c.values,device=device))

    def _local(self,x,state,reset):
        b,t=x.shape;c=self.config;e=self.embedding(x);history=torch.cat((state[0].to(e),e),1)
        # Reset prevents old local context crossing a document boundary.
        seg=reset.long().cumsum(1);extseg=F.pad(seg,(3,0))
        feats=[]
        for lag in range(4):
            f=history[:,3-lag:3-lag+t]
            if lag:f=f*(seg==extseg[:,3-lag:3-lag+t])[...,None]
            feats.append(f)
        final=history[:,-3:].clone()
        # Ensure returned shift-register does not include pre-reset elements.
        valid=F.pad(seg,(3,0))[:,-3:]==seg[:,-1:]
        final=final*valid[...,None]
        return self.local(torch.cat(feats,-1)),final

    def _rope(self,x,pos):
        angle=pos.float()[...,None]*self.freq;co=angle.cos()[:,None];si=angle.sin()[:,None]
        a,b=x[...,0::2],x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),-1).flatten(-2)

    def forward(self,tokens:Tensor,state=None,reset:Tensor|None=None,*,intervention:str='none'):
        if tokens.ndim!=2 or tokens.shape[1]<1 or tokens.dtype!=torch.long:raise ValueError('nonempty long [B,T] required')
        if intervention not in ('none','memory_reset','history_shuffle'):raise ValueError('invalid intervention')
        c=self.config;b,t=tokens.shape
        reset=torch.zeros_like(tokens,dtype=torch.bool) if reset is None else reset
        if reset.shape!=tokens.shape or reset.dtype!=torch.bool:raise ValueError('reset mismatch')
        state=self.initial(b,tokens.device) if state is None else state
        u,trace=self._local(tokens,state,reset)
        op_logits,key_logits,val_logits=self.operations(u),self.address(u),self.value(u)
        aux=(op_logits,key_logits,val_logits)
        if c.kind=='attention':
            caches=[];hidden=u
            for li in range(2):
                old=None if state[1] is None else state[1][li]
                offset=0 if old is None else old[0].shape[2]
                pos=torch.arange(offset,offset+t,device=tokens.device)[None].expand(b,-1)
                q,k,v=self.qkv[li](self.attn_norm[li](hidden)).view(b,t,3,4,c.width//4).permute(2,0,3,1,4).unbind(0)
                q,k=self._rope(q,pos),self._rope(k,pos)
                seg=reset.long().cumsum(1)
                if old is not None:
                    seg=seg+old[2][:,-1:];k=torch.cat((old[0],k),2);v=torch.cat((old[1],v),2);allseg=torch.cat((old[2],seg),1)
                else:allseg=seg
                mask=(torch.arange(k.shape[2],device=tokens.device)[None,None]<=pos[:,:,None])&(seg[:,:,None]==allseg[:,None])
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None],dropout_p=0.).transpose(1,2).reshape(b,t,c.width)
                hidden=hidden+self.attn_out[li](y);caches.append((k.clone(),v.clone(),allseg.clone()))
            return self.head(hidden+self.ff(hidden)),(trace,tuple(caches)),aux
        hard=c.kind=='hard'
        op=choice(op_logits,hard);address=choice(key_logits,hard);value=choice(val_logits,hard)
        # All decisions predicted from token context, not token-ID comparisons.
        erase=(op[...,1]+op[...,2])[...,None]*address
        write=op[...,1,None]*address
        f=1-erase; f=f*(~reset)[...,None]
        if intervention=='memory_reset':f=f*0
        initial=torch.cat(((state[1].float()-c.prior)[...,None],state[2].float()),-1)
        update=torch.cat((write[...,None],write[...,None]*value[...,None,:]),-1)
        packed=affine_prefix(f.float(),update.float(),initial)
        p=packed[...,0]+c.prior;s=packed[...,1:]
        read_p,read_s=(p,s) if intervention!='history_shuffle' else (p.roll(1,0),s.roll(1,0))
        mode=read_s/read_p[...,None]
        r=torch.einsum('btk,btkv->btv',address,mode)
        mass=(address*(read_p-c.prior)).sum(-1,keepdim=True)
        hidden=u+self.read(torch.cat((r,mass),-1).to(u))
        logits=self.head(hidden+self.ff(hidden))
        return logits,(trace,p[:,-1].clone(),s[:,-1].clone()),aux

    def parameter_count(self):return sum(p.numel() for p in self.parameters())


def training_labels(tokens:Tensor):
    """Training-only semantic labels for the supplied synthetic grammar.

    No returned tensor is an inference input. Final answers are not used to label
    earlier positions: labels use current and three earlier tokens only.
    """
    from .scaling_data import SET,DELETE,NOISE,ASK,SEP,KEY_START,KEYS,VALUE_START,VALUES
    x=tokens;b,t=x.shape;h=F.pad(x,(3,0),value=0)
    cur=h[:,3:];p1=h[:,2:-1];p2=h[:,1:-2];p3=h[:,:-3]
    op=torch.zeros_like(x);key=torch.full_like(x,-100);val=torch.full_like(x,-100)
    setmask=(cur==SEP)&(p3==SET)&(p2>=KEY_START)&(p2<KEY_START+KEYS)&(p1>=VALUE_START)&(p1<VALUE_START+VALUES)
    delmask=(cur==SEP)&(p2==DELETE)&(p1>=KEY_START)&(p1<KEY_START+KEYS)
    query=(p1==ASK)&(cur>=KEY_START)&(cur<KEY_START+KEYS)
    op[setmask]=1;key[setmask]=p2[setmask]-KEY_START;val[setmask]=p1[setmask]-VALUE_START
    op[delmask]=2;key[delmask]=p1[delmask]-KEY_START
    op[query]=3;key[query]=cur[query]-KEY_START
    return op,key,val


def router_loss(logits,aux,item,weight:float):
    from .scaling_data import objective
    loss=objective(logits,item)
    if not weight:return loss
    targets=training_labels(item['x']);valid=item['weights']>0
    # Equal operation-class contribution avoids rewarding HOLD-only collapse.
    terms=[]
    for op in range(4):
        mask=valid&(targets[0]==op)
        if mask.any():terms.append(F.cross_entropy(aux[0][mask],targets[0][mask]))
    labelloss=torch.stack(terms).mean()
    for out,target in zip(aux[1:],targets[1:]):
        mask=valid&(target!=-100)
        if mask.any():labelloss=labelloss+F.cross_entropy(out[mask],target[mask])
    return loss+weight*labelloss
