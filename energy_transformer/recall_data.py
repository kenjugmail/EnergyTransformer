"""Fresh continuous-valued edit episodes. No names, templates, or language corpus.
Episode keys are randomly sampled; the persistent model does not see key IDs.
SET and zero-valued RESET semantics are explicitly provided, not extracted.
"""
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F

@dataclass
class EpisodeBatch:
    keys:Tensor
    values:Tensor
    operations:Tensor
    queries:Tensor
    targets:Tensor
    initial_values:Tensor
    changed:Tensor
    key_ids:Tensor # evaluator metadata only, NEVER provided to the learned model


def episodes(seed:int,batch:int=24,nkeys:int=6,dimension:int=12,value_dim:int=4,
             edits:int=8,gap:int=4,correlation:float=.4):
    if not 0<=correlation<1 or min(batch,nkeys,dimension,value_dim)<1:raise ValueError('Invalid task')
    g=torch.Generator().manual_seed(seed)
    bank=torch.randn(batch,nkeys,dimension,generator=g)
    common=torch.randn(batch,1,dimension,generator=g)
    bank=F.normalize((1-correlation)**.5*bank+correlation**.5*common,dim=-1)
    target=torch.randn(batch,nkeys,value_dim,generator=g);initial=target.clone()
    keys=[bank[:,i] for i in range(nkeys)]; vals=[target[:,i].clone() for i in range(nkeys)]
    ops=[torch.zeros(batch,dtype=torch.long) for _ in range(nkeys)]
    ids=[torch.full((batch,),i,dtype=torch.long) for i in range(nkeys)]
    changed=torch.zeros(batch,nkeys,dtype=torch.bool)
    # Only the first half is edited. The second half tests collateral damage.
    b=torch.arange(batch)
    for t in range(edits):
        i=torch.randint(max(1,nkeys//2),(batch,),generator=g)
        op=(torch.rand(batch,generator=g)<.2).long()
        value=torch.randn(batch,value_dim,generator=g)
        target[b,i]=value*(op==0)[:,None];changed[b,i]=True
        keys.append(bank[b,i]);vals.append(value);ops.append(op);ids.append(i)
        # Interleave conflicting but explicitly ignorable observations.
        if t%2==0:
            i=torch.randint(nkeys,(batch,),generator=g)
            keys.append(bank[b,i]);vals.append(torch.randn(batch,value_dim,generator=g));ops.append(torch.full((batch,),2));ids.append(i)
    for _ in range(gap):
        i=torch.randint(nkeys,(batch,),generator=g)
        keys.append(bank[b,i]);vals.append(torch.randn(batch,value_dim,generator=g));ops.append(torch.full((batch,),2));ids.append(i)
    return EpisodeBatch(torch.stack(keys,1),torch.stack(vals,1),torch.stack(ops,1),bank,target,initial,changed,torch.stack(ids,1))
