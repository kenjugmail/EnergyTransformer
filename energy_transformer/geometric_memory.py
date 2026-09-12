"""Experimental v18 correlated-key memory; not a pretrained LM replacement.

Only quadratic coefficients cross steps for energy models. The delta comparator
carries a fast-weight matrix. HOLD is an explicitly supplied operation and is an
exact no-op for every model; these experiments do not test language extraction.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

KINDS = ('diagonal', 'euclidean', 'precision', 'delta', 'preconditioned_delta', 'projected')

@dataclass(frozen=True)
class MemoryState:
    precision: Tensor  # [batch,d,d], [batch,d], or empty for delta
    information: Tensor  # [batch,d,v]


def congruence_edit(p:Tensor,s:Tensor,k:Tensor,v:Tensor,erase:Tensor,write:Tensor,
                    *,prior:float=0.01,geometry:bool=False,diagonal:bool=False):
    """A=I-erase*d*k^T. Transform non-prior energy, then add a rank-one factor.

    geometry chooses d=P^-1 k/(k^T P^-1 k); otherwise d=k for unit k.
    Rank-two arithmetic computes the congruence without materializing A.
    Positive definiteness is guaranteed in real arithmetic, not interval arithmetic.
    """
    width=k.shape[-1]
    if diagonal:
        r=p-prior; direction=k; u=r*direction
        scalar=(direction*u).sum(-1)
        pnew=prior+r-2*erase[:,None]*k*u+(erase.square()*scalar)[:,None]*k.square()+write[:,None]*k.square()
    else:
        eye=torch.eye(width,device=p.device,dtype=p.dtype)
        r=p-prior*eye
        if geometry:
            d=torch.linalg.solve(p,k.unsqueeze(-1)).squeeze(-1)
            direction=d/(k*d).sum(-1,keepdim=True).clamp_min(torch.finfo(p.dtype).eps)
        else: direction=k
        u=(r@direction.unsqueeze(-1)).squeeze(-1)
        scalar=(direction*u).sum(-1)
        kk=k.unsqueeze(-1)*k.unsqueeze(-2)
        pnew=p-erase[:,None,None]*(k.unsqueeze(-1)*u.unsqueeze(-2)+u.unsqueeze(-1)*k.unsqueeze(-2))
        pnew=pnew+(erase.square()*scalar+write)[:,None,None]*kk
        pnew=.5*(pnew+pnew.transpose(-1,-2))
    old=(direction.unsqueeze(-2)@s).squeeze(-2)
    snew=s-erase[:,None,None]*k.unsqueeze(-1)*old.unsqueeze(-2)+write[:,None,None]*k.unsqueeze(-1)*v.unsqueeze(-2)
    return pnew,snew


class EditMemory(nn.Module):
    def __init__(self,kind:str='precision',dimension:int=12,value_dim:int=4,prior:float=.01):
        super().__init__()
        if kind not in KINDS or dimension<2 or value_dim<1 or prior<=0:raise ValueError('Invalid model configuration')
        self.kind=kind;self.dimension=dimension;self.value_dim=value_dim;self.prior=prior
        self.key=nn.Linear(dimension,dimension,bias=False)
        with torch.no_grad(): self.key.weight.copy_(torch.eye(dimension)+.01*torch.randn(dimension,dimension))
        self.controller=nn.Sequential(nn.Linear(dimension+3,32),nn.SiLU(),nn.Linear(32,3))
        with torch.no_grad(): self.controller[-1].bias.copy_(torch.tensor([2.,0.,5.]))
        self.log_gain=nn.Parameter(torch.zeros(()))

    def initial(self,batch:int,*,device=None,dtype=None):
        ref=next(self.parameters());device=device or ref.device;dtype=dtype or ref.dtype
        s=torch.zeros(batch,self.dimension,self.value_dim,device=device,dtype=dtype)
        if self.kind=='delta':p=torch.empty(batch,0,device=device,dtype=dtype)
        elif self.kind=='diagonal':p=torch.full((batch,self.dimension),self.prior,device=device,dtype=dtype)
        else:p=self.prior*torch.eye(self.dimension,device=device,dtype=dtype).expand(batch,-1,-1).clone()
        return MemoryState(p,s)

    def encode_key(self,x:Tensor):return F.normalize(self.key(x),dim=-1,eps=1e-8)

    def step(self,x:Tensor,value:Tensor,operation:Tensor,state:MemoryState):
        # 0=SET, 1=zero-valued RESET (not unknown), 2=HOLD. Fields are supplied.
        k=self.encode_key(x)
        logits=self.controller(torch.cat([x,F.one_hot(operation,3).to(x.dtype)],-1))
        active=(operation!=2).to(x.dtype)
        alpha=torch.sigmoid(logits[:,0])*active
        beta=(.05+3.95*torch.sigmoid(logits[:,1]))*active
        value=value*(operation==0).to(x.dtype)[:,None]
        if self.kind=='delta':
            # Scalar-decay gated delta reference, not complete GDN-2 or EDA.
            decay=1-active*(1-torch.sigmoid(logits[:,2]))
            m=state.information*decay[:,None,None]
            residual=value-(k.unsqueeze(-2)@m).squeeze(-2)
            update=torch.sigmoid(logits[:,1])*active
            result=MemoryState(state.precision,m+update[:,None,None]*k.unsqueeze(-1)*residual.unsqueeze(-2))
        elif self.kind=='projected':
            # Algebraic reparameterization of preconditioned_delta: only P/S persist.
            # Compute the old query response without materializing/carrying M=P^-1 S.
            u=torch.linalg.solve(state.precision,k.unsqueeze(-1)).squeeze(-1)
            leverage=(k*u).sum(-1).clamp_min(torch.finfo(k.dtype).eps)
            old=(u.unsqueeze(-2)@state.information).squeeze(-2)
            coefficient=beta[:,None]*old+(alpha*(1/leverage+beta))[:,None]*(value-old)
            p=state.precision+beta[:,None,None]*k.unsqueeze(-1)*k.unsqueeze(-2)
            s=state.information+k.unsqueeze(-1)*coefficient.unsqueeze(-2)
            result=MemoryState(p,s)
        elif self.kind=='preconditioned_delta':
            # Classical covariance-preconditioned, normalized delta correction.
            # P is a metric accumulator; information here is the carried mode M.
            p=state.precision+beta[:,None,None]*k.unsqueeze(-1)*k.unsqueeze(-2)
            d=torch.linalg.solve(p,k.unsqueeze(-1)).squeeze(-1)
            d=d/(k*d).sum(-1,keepdim=True).clamp_min(torch.finfo(p.dtype).eps)
            residual=value-(k.unsqueeze(-2)@state.information).squeeze(-2)
            s=state.information+alpha[:,None,None]*d.unsqueeze(-1)*residual.unsqueeze(-2)
            result=MemoryState(p,s)
        else:
            p,s=congruence_edit(state.precision,state.information,k,value,alpha,beta,
                               prior=self.prior,geometry=self.kind=='precision',diagonal=self.kind=='diagonal')
            result=MemoryState(p,s)
        # Exact byte-level hold contract, including floating arithmetic grouping.
        hold=operation==2
        p=torch.where(hold.reshape(-1,*([1]*(result.precision.ndim-1))),state.precision,result.precision)
        s=torch.where(hold[:,None,None],state.information,result.information)
        return MemoryState(p,s)

    def read(self,queries:Tensor,state:MemoryState):
        k=self.encode_key(queries)
        if self.kind in ('delta','preconditioned_delta'):mode=state.information
        elif self.kind=='diagonal':mode=state.information/state.precision.unsqueeze(-1)
        else:mode=torch.linalg.solve(state.precision,state.information)
        return torch.exp(self.log_gain.clamp(-3,3))*(k@mode)

    def forward(self,keys:Tensor,values:Tensor,operations:Tensor,queries:Tensor,*,immediate:bool=False):
        state=self.initial(keys.shape[0],device=keys.device,dtype=keys.dtype)
        history=[]
        for t in range(keys.shape[1]):
            state=self.step(keys[:,t],values[:,t],operations[:,t],state)
            if immediate: history.append(self.read(keys[:,t:t+1],state).squeeze(1))
        output=self.read(queries,state)
        return output,state,torch.stack(history,1) if immediate else None
