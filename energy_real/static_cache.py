"""Opt-in preallocated full-attention cache for fair inference measurements.

The returned model owns one mutable stream. Always pass its latest state;
pass None to start over. Capacity is fixed, and gradients are disabled. This
does not change weights or training, and avoids concatenating growing KV
tensors on every decoded token. Cache buffers are not checkpoint contents.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from .bench import tensor_bytes
from .model import LM, SoftmaxMixer


class StaticSoftmaxMixer(SoftmaxMixer):
    def __init__(self, original: SoftmaxMixer, capacity: int):
        nn.Module.__init__(self)
        if type(capacity) is not int or capacity < 1 or original.c.attention_window:
            raise ValueError('Positive capacity and full attention required')
        self.c, self.v = original.c, original.v
        self.qkv, self.output = original.qkv, original.output
        self.register_buffer('freq', original.freq)
        self.capacity = capacity
        self.keys = self.values = self.segments = None
        self.length = 0
        self.segmented = False

    @property
    def allocated_bytes(self):
        return tensor_bytes((self.keys, self.values, self.segments))

    def forward(self, x, state=None, reset=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Static cache requires eval mode and disabled gradients')
        b, t, d = x.shape
        offset = 0 if state is None else state[0].shape[2]
        end = offset + t
        if end > self.capacity:
            raise ValueError('Static cache capacity exceeded')
        if state is not None and (offset != self.length or self.keys is None or
                any(a.data_ptr() != owned.data_ptr() for a, owned in
                    zip(state, (self.keys, self.values, self.segments)))):
            raise ValueError('Static cache received a different or stale stream')
        pos = (torch.arange(t, device=x.device)[None] + offset).expand(b, -1)
        q, k, v = self.qkv(x).view(b, t, 3, self.c.heads, self.v).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q, pos), self.rope(k, pos)
        if state is None:
            shape = (b, self.c.heads, self.capacity, self.v)
            if self.keys is None or self.keys.shape != shape or self.keys.device != k.device or self.keys.dtype != k.dtype:
                self.keys, self.values = k.new_empty(shape), v.new_empty(shape)
                self.segments = torch.empty(b, self.capacity, dtype=torch.long, device=x.device)
            self.length, self.segmented = 0, False
            previous = torch.zeros(b, 1, dtype=torch.long, device=x.device)
        else:
            previous = self.segments[:, offset-1:offset]
        segments = previous.expand(b, t)
        if reset is not None:
            segments = segments + reset.long().cumsum(1)
        self.keys[:, :, offset:end].copy_(k)
        self.values[:, :, offset:end].copy_(v)
        self.segments[:, offset:end].copy_(segments)
        keys, values, key_segments = self.keys[:, :, :end], self.values[:, :, :end], self.segments[:, :end]
        if offset == 0 and reset is None:
            y = F.scaled_dot_product_attention(q, keys, values, is_causal=True)
        elif t == 1 and reset is None and not self.segmented:
            y = F.scaled_dot_product_attention(q, keys, values)
        else:
            allowed = torch.arange(end, device=x.device)[None, None] <= pos[:, :, None]
            allowed = allowed & (segments[:, :, None] == key_segments[:, None])
            y = F.scaled_dot_product_attention(q, keys, values, attn_mask=allowed[:, None])
        self.length = end
        self.segmented = self.segmented or reset is not None
        return self.output(y.transpose(1, 2).reshape(b, t, d)), (keys, values, key_segments)


def with_static_cache(model: LM, capacity: int) -> LM:
    """Copy a model and replace full-attention mixers with preallocated caches."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Positive integer cache capacity required')
    result = copy.deepcopy(model)
    for block in result.blocks:
        if isinstance(block.mixer, SoftmaxMixer) and not block.mixer.c.attention_window:
            block.mixer = StaticSoftmaxMixer(block.mixer, capacity)
    return result.eval()


def allocated_state_bytes(model: LM, state) -> int:
    return sum(block.mixer.allocated_bytes if isinstance(block.mixer, StaticSoftmaxMixer)
               else tensor_bytes(layer) for block, layer in zip(model.blocks, state))
