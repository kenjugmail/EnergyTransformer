"""Multiple causal retrieval heads inside the unchanged joint-energy workspace."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .model import EnergyConfig, JointEnergyLM


@dataclass(frozen=True)
class MemoryConfig(EnergyConfig):
    attention_heads: int = 2
    cache_history: bool = False

    def validate(self):
        super().validate()
        if (type(self.attention_heads) is not int or self.attention_heads < 1
                or self.width % self.attention_heads):
            raise ValueError('attention_heads must divide width')
        if self.cache_history:
            raise ValueError('The multi-head memory currently requires cache_history=False')


class MemoryEnergyLM(JointEnergyLM):
    """Split the existing Q/K/V channels; add no inference parameters.

    Each head attends only supplied prefix bytes. The concatenated retrieval is
    the ordinary attention proposal in the same complete conditional energy.
    """

    def condition(self, token, state, *, disabled=(), constants=None,
                  attention_context=None, encoded_current=None):
        if attention_context is None:
            observed=torch.cat((state.observed[:,1:],self.embedding(token)[:,None]),1)
            encoded=self.encode_observed(observed)
            current=encoded[:,-1]
            available=min(state.count+1,self.config.history)
            memory=encoded[:,-available:]
            h=self.config.attention_heads
            d=self.config.width//h
            query_input=current+self.recurrent_condition(state) if self.config.context_kernel>1 else current
            query=self.query(query_input).reshape(len(token),h,d)
            keys=self.key(memory).reshape(len(token),available,h,d)
            values=self.value(memory).reshape(len(token),available,h,d)
            scores=(keys*query[:,None]).sum(-1)/math.sqrt(d)
            probability=scores.softmax(1)
            attention_context=(probability[...,None]*values).sum(1).flatten(1)
            encoded_current=current
        return super().condition(token,state,disabled=disabled,constants=constants,
            attention_context=attention_context,encoded_current=encoded_current)


def load_rewt_checkpoint(path):
    """Load ordinary pilot checkpoints or a multi-head memory checkpoint."""
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['arm']!='memory_energy':
        from .experiment import load_model
        return load_model(path)
    torch.manual_seed(saved['seed'])
    model=MemoryEnergyLM(MemoryConfig(**saved['config']))
    model.load_state_dict(saved['weights'],strict=True)
    return model,saved
