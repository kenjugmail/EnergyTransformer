"""Opt-in streaming language models and matched attention controls.

All arms share RMS normalization, RoPE attention, SwiGLU and tied embeddings.
The hybrid replaces its first attention block with gated delta memory. Training
uses exact chunked triangular solves; inference carries fixed matrix/conv state.
An optional latent Fourier energy uses a per-token curvature upper bound.
This is a new architecture, not a checkpoint-preserving change to JointEnergyLM.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StreamingConfig:
    width: int = 64
    heads: int = 4
    hidden: int = 144
    layers: int = 2
    hybrid: bool = True
    window: int = 64
    chunk: int = 32
    modes: int = 16
    rounds: int = 2
    vocab_size: int = 256

    def validate(self):
        if type(self.vocab_size) is not int or self.vocab_size < 2:
            raise ValueError('vocab_size must be an integer >= 2')
        for name in ('width', 'heads', 'hidden', 'layers', 'chunk', 'modes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('window', 'rounds'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('even head dimensions required')
        if type(self.hybrid) is not bool:
            raise ValueError('hybrid must be boolean')


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def rotate(x, positions):
    """RoPE at absolute positions, including incremental decoding offsets."""
    half = x.shape[-1] // 2
    frequency = torch.exp(-math.log(10000.) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
    angle = positions.to(x.dtype)[:, None] * frequency
    a, b = x.chunk(2, -1)
    return torch.cat((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)


def gated_delta_chunk(keys, values, beta, decay, queries, memory):
    """Exact S_t = g_t S_(t-1) + beta_t k_t (v_t-g_t k_t S_(t-1)).

    Inputs are [batch*heads,time,head_dim]; gates have one last coordinate.
    Log products avoid dividing by tiny cumulative products. A chunk is bounded
    by configuration and retention is in [.9, 1], so its exponentials are safe.
    """
    length = keys.shape[1]
    logp = decay.log().cumsum(1)
    ratio = (logp - logp.transpose(1, 2)).clamp_max(0).exp()
    prefix = logp.exp()
    cross = keys @ keys.transpose(1, 2)
    lower = torch.tril(beta * ratio * cross, diagonal=-1)
    lower = lower + torch.eye(length, device=keys.device, dtype=keys.dtype)
    writes = torch.linalg.solve_triangular(lower, beta * (values - prefix * (keys @ memory)),
                                           upper=False, unitriangular=True)
    reads = prefix * (queries @ memory) + torch.tril(ratio * (queries @ keys.transpose(1, 2))) @ writes
    tail = (logp[:, -1:] - logp).exp()
    final = prefix[:, -1:, :] * memory + keys.transpose(1, 2) @ (tail * writes)
    return reads, final


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, state, position, *, return_state):
        b, t, d = x.shape
        h = self.config.heads
        q, k, v = self.qkv(x).view(b, t, 3, h, d // h).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        positions = torch.arange(position, position + t, device=x.device)
        q, k = rotate(q, positions), rotate(k, positions)
        past = 0
        if state is not None:
            oldk, oldv = state
            past = oldk.shape[-2]
            k, v = torch.cat((oldk, k), -2), torch.cat((oldv, v), -2)
        if not past and not self.config.window:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            qi = torch.arange(past, past + t, device=x.device)[:, None]
            ki = torch.arange(past + t, device=x.device)[None]
            mask = ki <= qi
            if self.config.window:
                mask = mask & (ki > qi - self.config.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        saved = None
        if return_state:
            keep = self.config.window - 1 if self.config.window else k.shape[-2]
            saved = (k[..., -keep:, :].contiguous().clone(), v[..., -keep:, :].contiguous().clone()) if keep else (k[..., :0, :].clone(), v[..., :0, :].clone())
        return self.out(y.transpose(1, 2).reshape(b, t, d)), saved


class DeltaMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.conv = nn.Conv1d(d, d, 3, groups=d, bias=True)
        self.qkvg = nn.Linear(d, 4*d, bias=False)
        self.gates = nn.Linear(d, 2*config.heads)
        self.norm = RMSNorm(d // config.heads)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, state, position, *, return_state):
        b, t, d = x.shape
        h, dh = self.config.heads, d // self.config.heads
        past, memory = (x.new_zeros(b, 2, d), x.new_zeros(b*h, dh, dh)) if state is None else state
        history = torch.cat((past, x), 1)
        features = F.silu(self.conv(history.transpose(1, 2)).transpose(1, 2))
        q, k, v, gate = self.qkvg(features).view(b, t, 4, h, dh).unbind(2)
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        q, k, v = (a.transpose(1, 2).reshape(b*h, t, dh) for a in (q, k, v))
        beta, decay = self.gates(features).view(b, t, 2, h).unbind(2)
        beta = beta.sigmoid().transpose(1, 2).reshape(b*h, t, 1)
        decay = (.9 + .1*decay.sigmoid()).transpose(1, 2).reshape(b*h, t, 1)
        reads = []
        for start in range(0, t, self.config.chunk):
            end = start + self.config.chunk
            if t == 1:
                memory = decay[:, 0, :, None] * memory
                error = v[:, 0] - (k[:, 0, :, None] * memory).sum(-2)
                memory = memory + beta[:, 0, :, None] * k[:, 0, :, None] * error[:, None]
                read = (q[:, 0, :, None] * memory).sum(-2)[:, None]
            else:
                read, memory = gated_delta_chunk(k[:, start:end], v[:, start:end], beta[:, start:end],
                                                 decay[:, start:end], q[:, start:end], memory)
            reads.append(read)
        y = torch.cat(reads, 1).view(b, h, t, dh).transpose(1, 2)
        y = self.norm(y) * F.silu(gate)
        saved = (history[:, -2:].contiguous().clone(), memory) if return_state else None
        return self.out(y.reshape(b, t, d)), saved


class Block(nn.Module):
    def __init__(self, config, delta):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.width), RMSNorm(config.width)
        self.mixer = DeltaMemory(config) if delta else Attention(config)
        self.up = nn.Linear(config.width, 2*config.hidden, bias=False)
        self.down = nn.Linear(config.hidden, config.width, bias=False)

    def forward(self, x, state, position, return_state):
        y, state = self.mixer(self.norm1(x), state, position, return_state=return_state)
        x = x + y
        a, b = self.up(self.norm2(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b), state


class LatentEnergy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.amplitude = nn.Linear(config.width, config.modes, bias=False)
        self.raw_frequency = nn.Parameter(torch.randn(config.modes, config.width) / math.sqrt(config.width))
        self.phase = nn.Parameter(torch.rand(config.modes)*2*math.pi-math.pi)
        self.raw_precision = nn.Parameter(torch.zeros(config.width))

    @staticmethod
    def energy(delta, amplitude, frequency, phase, precision):
        return .5*(precision*delta.square()).sum(-1)+(amplitude*F.linear(delta,frequency,phase).cos()).sum(-1)

    def forward(self, anchor, rounds, trace=False, return_path=False):
        if type(rounds) is not int or rounds < 0:
            raise ValueError('rounds must be nonnegative')
        amplitude = self.amplitude(anchor).tanh() / math.sqrt(self.config.modes)
        raw = self.raw_frequency
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        frequency = 2*raw*norm.tanh()/norm
        precision = .15+1.85*self.raw_precision.sigmoid()
        # ||H|| <= max(q) + sum_i |a_i| ||w_i||^2. Step 1/L yields descent.
        upper = precision.max() + amplitude.abs() @ frequency.square().sum(-1)
        eta = .95/upper[..., None]
        delta = torch.zeros_like(anchor)
        energies = []
        path = [anchor] if return_path else None
        for _ in range(rounds):
            if trace:
                energies.append(self.energy(delta,amplitude,frequency,self.phase,precision).detach())
            gradient = precision*delta - (amplitude*F.linear(delta,frequency,self.phase).sin()) @ frequency
            delta = delta - eta*gradient
            if return_path:
                path.append(anchor + delta)
        if trace:
            energies.append(self.energy(delta,amplitude,frequency,self.phase,precision).detach())
        return (path if return_path else anchor+delta), torch.stack(energies,-1) if trace else None


@dataclass
class StreamingState:
    layers: list
    position: int
    signature: tuple

    def tensor_bytes(self):
        return sum(t.numel()*t.element_size() for layer in self.layers if layer is not None for t in layer)


class StreamingLM(nn.Module):
    def __init__(self, config=StreamingConfig()):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config, config.hybrid and i%2 == 0) for i in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.energy_head = LatentEnergy(config) if config.modes and config.hybrid else None
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.mixer.out.weight, std=.02/math.sqrt(2*config.layers))
            nn.init.normal_(block.down.weight, std=.02/math.sqrt(2*config.layers))
            if isinstance(block.mixer, DeltaMemory):
                with torch.no_grad():
                    block.mixer.gates.bias[:config.heads].fill_(0.)
                    block.mixer.gates.bias[config.heads:].fill_(3.)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module,'bias',None) is not None:
                nn.init.zeros_(module.bias)

    def signature(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.parameters())

    def encode(self, tokens, state=None, *, return_state=False):
        """Causal backbone state before the optional prediction refinement."""
        if tokens.ndim != 2 or tokens.dtype != torch.long or not all(tokens.shape):
            raise ValueError('nonempty [batch,time] int64 tokens required')
        if bool(((tokens<0)|(tokens>=self.config.vocab_size)).any()):
            raise ValueError('token IDs outside the configured vocabulary')
        if state is not None and (state.signature != self.signature() or len(state.layers) != len(self.blocks)):
            raise ValueError('state belongs to different or changed weights')
        position = state.position if state is not None else 0
        x = self.embedding(tokens)
        saved = []
        for i, block in enumerate(self.blocks):
            x, layer = block(x, state.layers[i] if state is not None else None, position, return_state)
            saved.append(layer)
        x = self.norm(x)
        next_state = StreamingState(saved, position+tokens.shape[1], self.signature()) if return_state else None
        return x, next_state

    def refinement_states(self, tokens, rounds):
        """Differentiable states 0..rounds from one causal backbone pass.

        Targets are deliberately absent: training may supervise these states,
        while inference chooses its fixed depth without access to future words.
        """
        if self.energy_head is None:
            raise ValueError('A latent energy head is required')
        anchor, _ = self.encode(tokens)
        states, _ = self.energy_head(anchor, rounds, return_path=True)
        return states

    def forward(self, tokens, state=None, *, return_state=False, rounds=None, details=False):
        rounds = self.config.rounds if rounds is None else rounds
        if type(rounds) is not int or rounds < 0:
            raise ValueError('rounds must be nonnegative')
        x, next_state = self.encode(tokens, state, return_state=return_state)
        energies = None
        if self.energy_head is not None and (rounds or details):
            x, energies = self.energy_head(x, rounds, trace=details)
        logits = F.linear(x, self.embedding.weight)
        return logits, next_state, energies


def load_streaming(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('arm') != 'streaming_lm':
        raise ValueError('not a streaming checkpoint')
    model = StreamingLM(StreamingConfig(**saved['config']))
    model.load_state_dict(saved['weights'], strict=True)
    return model, saved
