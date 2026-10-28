"""Learned observed-token positions inside the same causal joint energy."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .memory import MemoryConfig,MemoryEnergyLM,load_rewt_checkpoint


@dataclass(frozen=True)
class PositionMemoryConfig(MemoryConfig):
    position_period: int=128
    position_initial_scale: float=.02

    def validate(self):
        super().validate()
        if type(self.position_period)is not int or self.position_period<self.history:
            raise ValueError('position_period must cover the observed attention history')
        if not math.isfinite(self.position_initial_scale) or self.position_initial_scale<0:
            raise ValueError('position_initial_scale must be finite and nonnegative')


class PositionMemoryEnergyLM(MemoryEnergyLM):
    def __init__(self,config):
        super().__init__(config)
        self.position=nn.Embedding(config.position_period,config.width)
        with torch.no_grad():self.position.weight.mul_(config.position_initial_scale)

    def encode_tokens(self,tokens,*,start=0):
        indices=torch.arange(start,start+tokens.shape[1],device=tokens.device)
        observed=self.embedding(tokens)+self.position(indices%self.config.position_period)[None]
        return self.encode_observed(observed)

    def observed_bank(self, tokens, state):
        observed = torch.cat((state.observed, self.embedding(tokens)), 1)
        indices = torch.arange(state.count - self.observed_capacity,
                               state.count + tokens.shape[1], device=tokens.device)
        positions = self.position(indices % self.config.position_period)
        observed = observed + positions[None] * (indices >= 0)[None, :, None]
        return self.encode_observed(observed)

    def encode_current(self, token, state):
        tail = self.config.context_kernel - 1
        observed = torch.cat((state.observed[:, -tail:] if tail else state.observed[:, :0],
                              self.embedding(token)[:, None]), 1)
        indices = torch.arange(state.count - tail, state.count + 1, device=token.device)
        positions = self.position(indices % self.config.position_period)
        observed = observed + positions[None] * (indices >= 0)[None, :, None]
        return self.encode_observed(observed)[:, -1]

    def condition(self,token,state,*,disabled=(),constants=None,attention_context=None,encoded_current=None,projections=None):
        if attention_context is None:
            observed=torch.cat((state.observed[:,1:],self.embedding(token)[:,None]),1)
            indices=torch.arange(state.count+1-observed.shape[1],state.count+1,device=token.device)
            positions=self.position(indices%self.config.position_period)
            # Prefix padding is empty memory, not a fictitious observed position.
            observed=observed+positions[None]*(indices>=0)[None,:,None]
            encoded=self.encode_observed(observed);current=encoded[:,-1]
            available=min(state.count+1,self.config.history);memory=encoded[:,-available:]
            h=self.config.attention_heads;d=self.config.width//h
            query_input=current+self.recurrent_condition(state) if self.config.context_kernel>1 else current
            query=self.query(query_input).reshape(len(token),h,d)
            keys=self.key(memory).reshape(len(token),available,h,d)
            values=self.value(memory).reshape(len(token),available,h,d)
            probability=((keys*query[:,None]).sum(-1)/math.sqrt(d)).softmax(1)
            attention_context=(probability[...,None]*values).sum(1).flatten(1)
            encoded_current=current
        return super().condition(token,state,disabled=disabled,constants=constants,
            attention_context=attention_context,encoded_current=encoded_current,projections=projections)


def load_learning_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['arm']=='architecture_energy':
        from .architecture import load_architecture_checkpoint
        return load_architecture_checkpoint(path)
    if saved['arm']!='position_memory_energy':return load_rewt_checkpoint(path)
    torch.manual_seed(saved['seed'])
    model=PositionMemoryEnergyLM(PositionMemoryConfig(**saved['config']))
    model.load_state_dict(saved['weights'],strict=True)
    return model,saved
