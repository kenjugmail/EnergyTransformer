"""Small softmax associative-retrieval control, NOT a full Transformer backbone.
Stores every active key/value. Holds are excluded by the supplied operation flag.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .geometric_memory import MemoryState

class SoftmaxRecall(nn.Module):
    kind='softmax'
    def __init__(self,dimension=12,value_dim=4,prior=.01):
        super().__init__();self.dimension=dimension;self.value_dim=value_dim;self.prior=prior
        self.key=nn.Linear(dimension,dimension,bias=False)
        with torch.no_grad():self.key.weight.copy_(torch.eye(dimension)+.01*torch.randn(dimension,dimension))
        self.log_temperature=nn.Parameter(torch.tensor(3.))
        self.recency=nn.Parameter(torch.tensor(0.))
        self.log_gain=nn.Parameter(torch.zeros(()))
    def encode_key(self,x):return F.normalize(self.key(x),dim=-1,eps=1e-8)
    def forward(self,keys,values,operations,queries,*,immediate=False):
        if immediate:raise ValueError('No immediate-only experiment for this comparator')
        k=self.encode_key(keys);q=self.encode_key(queries);active=operations!=2
        values=values*(operations==0).to(values.dtype).unsqueeze(-1)
        # Order counts edits, not exact holds. Same order supplied to all methods.
        age=active.cumsum(1).to(keys.dtype)
        age=age-age[:,-1:]
        score=torch.exp(self.log_temperature.clamp(0,7))*(q@k.transpose(-1,-2))+F.softplus(self.recency)*age[:,None,:]
        score=score.masked_fill(~active[:,None,:],-torch.inf)
        output=torch.exp(self.log_gain.clamp(-3,3))*(score.softmax(-1)@values)
        return output,MemoryState(k,values),None
