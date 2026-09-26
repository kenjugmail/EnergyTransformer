"""Energy / softmax language model with explicit energy kernel selection.

`FastFunctionMixer` is a drop-in replacement for the reference `FunctionMixer`
(spark_kit/energy_spark/model.py) and loads its state_dict unchanged. It
computes the same function via `energy_real.kernel` -- see that module for the
derivation. Parameter shapes, initialisation and the read/gate structure are
preserved so that comparisons against the reference are like-for-like.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .kernel import chunk_gla, read_again, comoving_index


@dataclass(frozen=True)
class Config:
    kind: str = 'energy'
    vocab: int = 256
    width: int = 384
    layers: int = 6
    heads: int = 4
    slots: int = 16
    ff_width: int = 1536
    prior: float = .25
    chunk: int = 64
    activation_checkpointing: bool = False
    backend: str = 'dmat'
    memory_dim: int = 0
    memory_capacity: int = 256
    attention_window: int = 0
    attention_every: int = 3
    energy_projection_fp32: bool = False

    def mixer_kind(self, index: int) -> str:
        if self.kind == 'hybrid':
            return 'softmax' if (index + 1) % self.attention_every == 0 else 'energy'
        return self.kind

    def validate(self):
        if type(self.energy_projection_fp32) is not bool:
            raise ValueError('energy_projection_fp32 must be boolean')
        if self.kind not in ('energy', 'softmax', 'hybrid'):
            raise ValueError('Unknown mixer')
        if any(type(x) is not int or x < 1 for x in
               (self.vocab, self.width, self.layers, self.heads, self.slots, self.ff_width, self.chunk)):
            raise ValueError('Positive integer dimensions required')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('Head count must divide width with an even head dimension')
        if self.prior <= 0 or not math.isfinite(self.prior):
            raise ValueError('Positive finite prior required')
        if self.backend not in ('dmat', 'matmul'):
            raise ValueError('Choose the fixed dmat or matmul backend')
        if type(self.memory_dim) is not int or self.memory_dim < 0 or type(self.memory_capacity) is not int or self.memory_capacity < 1:
            raise ValueError('Nonnegative memory dimension and positive capacity required')
        if type(self.attention_window) is not int or self.attention_window < 0:
            raise ValueError('Attention window must be a nonnegative integer')
        if type(self.attention_every) is not int or self.attention_every < 2:
            raise ValueError('Attention interval must be at least two')
        if self.kind == 'hybrid' and (not self.attention_window or self.attention_every > self.layers):
            raise ValueError('Hybrid requires a positive window and at least one attention layer')
        if self.kind == 'energy' and self.attention_window:
            raise ValueError('An energy-only model has no attention window')


class FastFunctionMixer(nn.Module):
    """Energy mixer. Same parameters and function as the reference; chunkwise kernel."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.v = c.width // c.heads
        self.projection = nn.Linear(c.width, c.heads * (3 * c.slots + 2 * self.v + 1))
        self.second = nn.Linear(2 * c.width, c.heads * c.slots)
        self.output = nn.Linear(c.width, c.width)
        self.register_buffer('strides', torch.tensor(([0, 1, 3, 7] * ((c.heads + 3) // 4))[:c.heads]) % c.slots)
        half_life = torch.logspace(math.log10(4), math.log10(256), c.heads)
        f = torch.exp(-math.log(2) / half_life).clamp(.001, .999)
        with torch.no_grad():
            bias = self.projection.bias.view(c.heads, -1)
            bias[:, 2 * c.slots + self.v:3 * c.slots + self.v] = torch.logit(f)[:, None]
        self.second_mix = nn.Parameter(torch.zeros(c.heads))

    def initial(self, batch: int, device):
        c = self.c
        return (torch.full((batch, c.heads, c.slots), c.prior, device=device),
                torch.zeros(batch, c.heads, c.slots, self.v, device=device))

    def forward(self, x: Tensor, state=None, reset: Tensor | None = None):
        c = self.c
        b, t, _ = x.shape
        if state is None:
            state = self.initial(b, x.device)
        # Protect the joint key/query/value/forget/write projection from BF16
        # rounding before the normalized recurrent reads. Preserve the historical
        # policy unless explicitly selected and recorded in the checkpoint config.
        if c.energy_projection_fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                p = self.projection(x.float()).view(b, t, c.heads, -1)
        else:
            p = self.projection(x).view(b, t, c.heads, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            k, q, v, f, w, g = torch.split(p.float(), [c.slots, c.slots, self.v, c.slots, 1, self.v], -1)
            k = F.normalize(k, dim=-1, eps=1e-6)
            q = F.normalize(q, dim=-1, eps=1e-6)
            f = torch.sigmoid(f).clamp(.0001, .99999)
            if reset is not None:
                f = f * (~reset[:, :, None, None]).float()
            w = F.softplus(w).clamp(max=20) + 1e-4

            co = comoving_index(t, c.slots, self.strides, x.device)          # [T,H,K]
            co_b = co[None].expand(b, -1, -1, -1)
            kc = k.gather(3, co_b)
            fc = f.gather(3, co_b)
            qc = q.gather(3, co_b)
            wk = w * kc

            pp, ss = (a.float() for a in state)
            # precision scan: [B,T,H,K] only -- no value axis
            a_prev = pp - c.prior
            lf = fc.clamp_min(1e-20).log()
            acc = _diag_scan(fc, (w * kc.square())[..., None], a_prev[..., None])[..., 0]
            precision = acc + c.prior
            u = qc / precision

            r, s_final, ctx = chunk_gla(u, wk, v, lf, ss, c.chunk, force=c.backend)
            r = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + 1e-5)
            e = r.reshape(b, t, c.width)
            # The scan runs in co-moving slot coordinates; the carried state is
            # returned in absolute coordinates, matching the reference, which
            # gathers the whole history back before slicing the last step.
            slots = torch.arange(c.slots, device=x.device)
            back = (slots[None, :] - t * self.strides[:, None]) % c.slots      # [H,K]
            back_b = back[None].expand(b, -1, -1)
            final = (precision[:, -1].gather(2, back_b).contiguous(),
                     s_final.gather(2, back_b[..., None].expand(-1, -1, -1, self.v)).contiguous())

        q2 = self.second(torch.cat((x, e.to(x.dtype)), dim=-1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            q2 = F.normalize(q2.float().view(b, t, c.heads, c.slots), dim=-1, eps=1e-6)
            q2c = q2.gather(3, co_b)
            r2 = read_again(ctx, q2c / precision)
            r2 = r2 * torch.rsqrt(r2.square().mean(-1, keepdim=True) + 1e-5)
            r = r + torch.sigmoid(self.second_mix)[None, None, :, None] * r2
            r = (r * torch.sigmoid(g)).reshape(b, t, c.width)
        return self.output(r.to(x.dtype)), final


def _diag_scan(f: Tensor, b: Tensor, initial: Tensor) -> Tensor:
    """Inclusive affine scan; identical to the reference diagonal_scan."""
    a, x = f, b
    distance = 1
    while distance < x.shape[1]:
        x = torch.cat((x[:, :distance], x[:, distance:] + a[:, distance:, ..., None] * x[:, :-distance]), 1)
        a = torch.cat((a[:, :distance], a[:, distance:] * a[:, :-distance]), 1)
        distance *= 2
    return x + a[..., None] * initial[:, None]


class SoftmaxMixer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.v = c.width // c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.output = nn.Linear(c.width, c.width)
        self.register_buffer('freq', 10000 ** (-torch.arange(0, self.v, 2).float() / self.v))

    def rope(self, x, pos):
        angle = pos.float()[..., None] * self.freq
        co, si = angle.cos()[:, None], angle.sin()[:, None]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * co - b * si, a * si + b * co), -1).flatten(-2).to(x.dtype)

    def forward(self, x, state=None, reset=None):
        b, t, d = x.shape
        h = self.c.heads
        offset = 0 if state is None else (state[3] if len(state) == 4 else state[0].shape[2])
        pos = (torch.arange(t, device=x.device)[None] + offset).expand(b, -1)
        q, k, v = self.qkv(x).view(b, t, 3, h, self.v).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q, pos), self.rope(k, pos)
        previous_segment = (torch.zeros(b, 1, dtype=torch.long, device=x.device) if state is None
                            else state[2][:, -1:])
        segments = previous_segment.expand(b, t)
        if reset is not None:
            segments = segments + reset.long().cumsum(1)
        if state is None and reset is None and not self.c.attention_window:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.output(y.transpose(1, 2).reshape(b, t, d)), (k.contiguous(), v.contiguous(), segments)
        key_segments = segments
        if state is not None:
            ko, vo, old_segments = state[:3]
            k, v = torch.cat((ko, k), 2), torch.cat((vo, v), 2)
            key_segments = torch.cat((old_segments, segments), 1)
        old_length = 0 if state is None else state[0].shape[2]
        if self.c.attention_window:
            # Window-sized query blocks avoid allocating a T-by-T mask or
            # attention matrix during long prefills, including on CPU.
            window = self.c.attention_window
            chunks = []
            for begin in range(0, t, window):
                end = min(t, begin + window)
                lo, hi = max(0, old_length + begin - window + 1), old_length + end
                qi = torch.arange(old_length + begin, old_length + end, device=x.device)[:, None]
                ki = torch.arange(lo, hi, device=x.device)[None, :]
                allowed = ((ki <= qi) & (ki > qi - window))[None]
                allowed = allowed & (segments[:, begin:end, None] == key_segments[:, None, lo:hi])
                chunks.append(F.scaled_dot_product_attention(q[:, :, begin:end], k[:, :, lo:hi],
                              v[:, :, lo:hi], attn_mask=allowed[:, None]))
            y = torch.cat(chunks, 2)
        else:
            key_pos = torch.arange(-old_length, t, device=x.device)[None, :] + offset
            allowed = key_pos[:, None, :] <= pos[:, :, None]
            allowed = allowed & (segments[:, :, None] == key_segments[:, None, :])
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None])
        if self.c.attention_window:
            window = self.c.attention_window
            # clone releases the complete prefill allocation even for B=H=1.
            new_state = (k[:, :, -window:].clone().contiguous(), v[:, :, -window:].clone().contiguous(),
                         key_segments[:, -window:].clone(), (pos[:, -1:] + 1).clone())
        else:
            new_state = (k.contiguous(), v.contiguous(), key_segments)
        return self.output(y.transpose(1, 2).reshape(b, t, d)), new_state


class Block(nn.Module):
    def __init__(self, c, index=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.width)
        self.norm2 = nn.LayerNorm(c.width)
        self.mixer = SoftmaxMixer(c) if c.mixer_kind(index) == 'softmax' else FastFunctionMixer(c)
        self.ff = nn.Sequential(nn.Linear(c.width, c.ff_width), nn.GELU(), nn.Linear(c.ff_width, c.width))
        self.fact_reader = None
        if c.memory_dim:
            from .fact_memory import FactReader
            # Keep shared parameter initialization identical across the ablation.
            with torch.random.fork_rng(devices=[]):
                self.fact_reader = FactReader(c.width, c.memory_dim)

    def forward(self, x, state, reset, memory=None):
        y, s = self.mixer(self.norm1(x), state, reset)
        x = x + y
        if self.fact_reader is not None:
            x = x + self.fact_reader(self.norm2(x), memory)
        return x + self.ff(self.norm2(x)), s


class LM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab, config.width)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab, bias=False)
        self.head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=.02)

    def forward(self, tokens: Tensor, state=None, reset: Tensor | None = None, *, memory=None):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError('tokens must be nonempty [batch,time]')
        if tokens.dtype != torch.long:
            raise TypeError('integer token IDs required')
        if tokens.shape[0] < 1:
            raise ValueError('nonempty batch required')
        if reset is not None and (reset.shape != tokens.shape or reset.dtype != torch.bool or reset.device != tokens.device):
            raise ValueError('reset must be Boolean [batch,time] on the tokens device')
        if memory is not None:
            from .fact_memory import FactMemory
            if not self.config.memory_dim or not isinstance(memory, FactMemory):
                raise ValueError('Fact memory requires a configured native memory reader')
            memory.validate(tokens.shape[0], self.config.memory_dim, self.config.memory_capacity, tokens.device)
        if state is None:
            state = tuple(None for _ in self.blocks)
        if not isinstance(state, (tuple, list)) or len(state) != len(self.blocks):
            raise ValueError('One state per layer required')
        normalized = []
        for index, layer in enumerate(state):
            if layer is None:
                normalized.append(None)
                continue
            if not isinstance(layer, (tuple, list)) or not all(isinstance(a, Tensor) for a in layer):
                raise ValueError('Invalid layer state')
            c = self.config
            b = tokens.shape[0]
            if c.mixer_kind(index) == 'energy':
                shapes = [(b, c.heads, c.slots), (b, c.heads, c.slots, c.width // c.heads)]
                if len(layer) != 2 or [tuple(a.shape) for a in layer] != shapes:
                    raise ValueError('Invalid energy state shape')
            else:
                if len(layer) not in ((4,) if c.attention_window else (2, 3)) or layer[0].ndim != 4:
                    raise ValueError('Invalid softmax state')
                shape = (b, c.heads, layer[0].shape[2], c.width // c.heads)
                if shape[2] < 1 or any(tuple(a.shape) != shape for a in layer[:2]):
                    raise ValueError('Invalid KV cache shape')
                if len(layer) == 2:  # Legacy caches contained a single segment.
                    layer = (*layer, torch.zeros(b, shape[2], dtype=torch.long, device=tokens.device))
                if layer[2].shape != (b, shape[2]) or layer[2].dtype != torch.long:
                    raise ValueError('Invalid cached segment IDs')
                if c.attention_window:
                    if (shape[2] > c.attention_window or layer[3].shape != (b, 1)
                            or layer[3].dtype != torch.long or bool((layer[3] < shape[2]).any())):
                        raise ValueError('Invalid bounded attention position/cache')
            if any(a.device != tokens.device for a in layer) or any(not a.is_floating_point() for a in layer[:2]):
                raise ValueError('State dtype/device mismatch')
            normalized.append(layer)
        state = tuple(normalized)
        x = self.embedding(tokens)
        new = []
        for i, block in enumerate(self.blocks):
            if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                x, s = checkpoint(block, x, state[i], reset, memory, use_reentrant=False)
            else:
                x, s = block(x, state[i], reset, memory)
            new.append(s)
        return self.head(self.norm(x)), tuple(new)

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())
