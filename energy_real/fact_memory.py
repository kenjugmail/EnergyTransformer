"""Snapshot memory and gated readers for the native language model.

Facts are stored exactly outside the neural state. Fixed features only expose
that state to a learned reader; they do not confer language understanding.
One snapshot must be valid before every token in the supplied window. Split
windows at edits and invalidate recurrent state when the snapshot changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FactMemory:
    keys: Tensor
    values: Tensor
    mask: Tensor

    def validate(self, batch, dimension, capacity, device):
        if self.keys.ndim != 3 or self.keys.shape[0] != batch or self.keys.shape[2] != dimension:
            raise ValueError('Fact keys must be [batch,facts,memory_dim]')
        if not 1 <= self.keys.shape[1] <= capacity or self.values.shape != self.keys.shape:
            raise ValueError('Invalid fact capacity or value shape')
        if self.mask.shape != self.keys.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError('Fact mask must be Boolean [batch,facts]')
        if any(t.device != device for t in (self.keys, self.values, self.mask)):
            raise ValueError('Fact memory must be on the tokens device')
        if any(not t.is_floating_point() or not torch.isfinite(t).all() for t in (self.keys, self.values)):
            raise ValueError('Fact features must be finite floating-point tensors')

    def to(self, device):
        return FactMemory(self.keys.to(device), self.values.to(device), self.mask.to(device))

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values, self.mask))


def fixed_features(text, dimension):
    """Stable signed features, with no learned entity vocabulary."""
    if type(dimension) is not int or dimension < 1:
        raise ValueError('Positive feature dimension required')
    raw = hashlib.shake_256(text.encode('utf-8')).digest(dimension)
    return torch.tensor([1. if b & 1 else -1. for b in raw]) / math.sqrt(dimension)


def snapshot_memory(facts, dimension, capacity=256):
    if type(dimension) is not int or dimension < 1 or type(capacity) is not int or capacity < 1:
        raise ValueError('Positive feature dimension and capacity required')
    if len(facts) > capacity:
        raise ValueError('Fact snapshot exceeds memory capacity; no silent truncation')
    count = max(1, len(facts))
    keys, values = torch.zeros(1, count, dimension), torch.zeros(1, count, dimension)
    mask = torch.zeros(1, count, dtype=torch.bool)
    for i, fact in enumerate(facts):
        if fact['operation'] != 'set' or not isinstance(fact['object'], str):
            raise ValueError('Snapshot must contain current SET facts only')
        # Prefix/length encoding avoids ambiguous concatenations.
        address = f"{len(fact['subject'])}:{fact['subject']}{fact['relation']}"
        keys[0, i] = fixed_features('address:' + address, dimension)
        values[0, i] = fixed_features('entity:' + fact['object'], dimension)
        mask[0, i] = True
    return FactMemory(keys, values, mask)


class FactReader(nn.Module):
    def __init__(self, width, dimension):
        super().__init__()
        self.query = nn.Linear(width, dimension, bias=False)
        self.value = nn.Linear(dimension, width, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.null_value = nn.Parameter(torch.zeros(dimension))

    def forward(self, x, memory):
        if memory is None:
            return torch.zeros_like(x)
        # An explicit null candidate permits learned abstention when no address
        # matches. Its zero initial value also handles an empty snapshot safely.
        scores = torch.einsum('btd,bfd->btf', self.query(x).float(), memory.keys.float())
        scores = scores.masked_fill(~memory.mask[:, None], -1e9)
        weights = torch.cat((scores, torch.zeros_like(scores[..., :1])), -1).softmax(-1)
        read = torch.einsum('btf,bfd->btd', weights[..., :-1], memory.values.float())
        read = read + weights[..., -1:] * self.null_value.float()
        return self.gate.tanh() * self.value(read.to(x.dtype))
