"""Experimental energy-native separate erase/write directions (not language-trained).

Stores a full quadratic, not the deployed diagonal fast memory. The update is a
congruence transformation of retained information followed by one positive factor.
This is NOT a reproduction of Erase-then-Delta Attention or Gated DeltaNet-2.
It keeps an explicit energy interpretation; full precision has quadratic storage.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class QuadraticState:
    precision: Tensor   # [d,d]
    information: Tensor # [d,v]
    prior: float = .25

    @classmethod
    def initial(cls,keys:int,values:int,*,prior:float=.25,dtype=torch.float64):
        if keys<1 or values<1 or prior<=0:raise ValueError('Positive dimensions/prior required')
        return cls(prior*torch.eye(keys,dtype=dtype),torch.zeros(keys,values,dtype=dtype),prior)

    def energy(self,w:Tensor)->Tensor:
        return .5*(w*(self.precision@w)).sum()-(self.information*w).sum()

    def mode(self)->Tensor:
        return torch.linalg.solve(self.precision,self.information)

    def edit(self,erase:Tensor,key:Tensor,value:Tensor,*,erase_strength:float=0.,weight:float=1.):
        d,v=self.information.shape
        if erase.shape!=(d,) or key.shape!=(d,) or value.shape!=(v,):raise ValueError('Invalid factor shapes')
        if not 0<=erase_strength<=1 or weight<0:raise ValueError('Invalid gate')
        if not all(torch.isfinite(t).all() for t in (erase,key,value)):raise ValueError('Nonfinite factor')
        norm=erase.norm()
        if norm<1e-12 and erase_strength!=0:raise ValueError('Nonzero erase direction required')
        e=erase/norm.clamp_min(1e-12)
        eye=torch.eye(d,dtype=self.precision.dtype,device=self.precision.device)
        a=eye-erase_strength*torch.outer(e,e)
        p=self.prior*eye+a.T@(self.precision-self.prior*eye)@a+weight*torch.outer(key,key)
        s=a.T@self.information+weight*torch.outer(key,value)
        return QuadraticState(.5*(p+p.T),s,self.prior)
