"""Standalone streaming REWT student initialized from an LFM3 checkpoint.

The student contains no LFM3 convolution or global-attention operators. Its
13 donor convolution slots become gated delta memory; all attention is bounded.
Compatible token, feedforward, normalization and attention weights transfer.
Convolution-to-delta projection transfer is an initialization, not equivalence.
The original tokenizer is retained; the energy parameters are newly learned.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .streaming import gated_delta_chunk


@dataclass(frozen=True)
class ConversionConfig:
    vocab_size: int = 128000
    width: int = 2048
    hidden: int = 7680
    heads: int = 16
    kv_heads: int = 4
    head_dim: int = 128
    donor_layers: tuple = ('conv', 'conv', 'full_attention', 'conv', 'sliding_attention',
                           'conv', 'full_attention', 'conv', 'sliding_attention', 'conv',
                           'full_attention', 'conv', 'sliding_attention', 'conv',
                           'full_attention', 'conv', 'sliding_attention', 'conv',
                           'full_attention', 'conv', 'sliding_attention', 'conv',
                           'full_attention', 'conv')
    window: int = 256
    conv_width: int = 7
    chunk: int = 32
    modes: int = 64
    rounds: int = 2
    norm_eps: float = 1e-5
    full_theta: float = 1e6
    full_rotary_fraction: float = .5
    window_theta: float = 1e4
    pad_token_id: int = 124893
    eos_token_id: int = 124900
    teacher_repo: str = 'LiquidAI/LFM3-1.7B'
    teacher_revision: str = '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'

    def validate(self):
        for name in ('vocab_size', 'width', 'hidden', 'heads', 'kv_heads', 'head_dim',
                     'window', 'conv_width', 'chunk', 'modes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.width != self.heads*self.head_dim or self.head_dim % 2 or self.heads % self.kv_heads:
            raise ValueError('incompatible head dimensions')
        if type(self.rounds) is not int or self.rounds < 0:
            raise ValueError('invalid refinement rounds')
        if not self.donor_layers or any(t not in ('conv', 'full_attention', 'sliding_attention') for t in self.donor_layers):
            raise ValueError('unsupported donor layout')
        if not 0 < self.full_rotary_fraction <= 1 or self.norm_eps <= 0:
            raise ValueError('invalid normalization or rotary configuration')

    @classmethod
    def from_teacher(cls, config, **overrides):
        if config.get('model_type') != 'lfm3' or not config.get('use_sandwich_norm', False):
            raise ValueError('expected LFM3 with sandwich normalization')
        if config.get('conv_bias', False) or not config.get('tie_word_embeddings', False):
            raise ValueError('unsupported biased convolution or untied embedding')
        rotary = config['rope_parameters']
        if any(v.get('rope_type', 'default') != 'default' for v in rotary.values()):
            raise ValueError('only default RoPE is supported')
        if len(config['layer_types']) != config['num_hidden_layers']:
            raise ValueError('invalid donor layer count')
        values = dict(vocab_size=config['vocab_size'], width=config['hidden_size'],
                      hidden=config['intermediate_size'], heads=config['num_attention_heads'],
                      kv_heads=config['num_key_value_heads'], head_dim=config['head_dim'],
                      donor_layers=tuple(config['layer_types']), window=config['sliding_window'],
                      conv_width=config['conv_L_cache'], norm_eps=config['norm_eps'],
                      full_theta=rotary['full_attention']['rope_theta'],
                      full_rotary_fraction=rotary['full_attention'].get('partial_rotary_factor', 1.),
                      window_theta=rotary['sliding_attention']['rope_theta'],
                      pad_token_id=config['pad_token_id'], eos_token_id=config['eos_token_id'])
        values.update(overrides)
        result = cls(**values); result.validate()
        return result


class Norm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        working = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = working*torch.rsqrt(working.square().mean(-1, keepdim=True)+self.eps)
        return (normalized.to(x.dtype)*self.weight).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.width, c.hidden, bias=False)
        self.w3 = nn.Linear(c.width, c.hidden, bias=False)
        self.w2 = nn.Linear(c.hidden, c.width, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x))*self.w3(x))


class WindowAttention(nn.Module):
    """Donor-compatible GQA projections with a bounded, causal cache."""
    def __init__(self, c, donor_type):
        super().__init__()
        self.config = c
        self.q_proj = nn.Linear(c.width, c.heads*c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.width, c.kv_heads*c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.width, c.kv_heads*c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.heads*c.head_dim, c.width, bias=False)
        self.q_norm = Norm(c.head_dim, c.norm_eps)
        self.k_norm = Norm(c.head_dim, c.norm_eps)
        self.gate_proj = nn.Linear(c.width, c.heads, bias=False)
        self.theta = c.full_theta if donor_type == 'full_attention' else c.window_theta
        self.rotary_fraction = c.full_rotary_fraction if donor_type == 'full_attention' else 1.

    def _rotate(self, x, position):
        dim = x.shape[-1]; angles = int(self.rotary_fraction*dim)//2
        frequency = self.theta**(-torch.arange(0, 2*angles, 2, device=x.device, dtype=torch.float32)/dim)
        frequency = F.pad(frequency, (0, dim//2-angles))
        phase = torch.arange(position, position+x.shape[-2], device=x.device).float()[:, None]*frequency
        cosine, sine = torch.cat((phase, phase), -1).cos(), torch.cat((phase, phase), -1).sin()
        # The teacher rounds cos/sin to activation dtype before its float32 rotation.
        cosine, sine = cosine.to(x.dtype).float(), sine.to(x.dtype).float()
        first, second = x.float().chunk(2, -1)
        rotated = x.float()*cosine+torch.cat((-second, first), -1)*sine
        return rotated.to(x.dtype)

    def forward(self, x, state=None, position=0, *, return_state=False):
        c = self.config; b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, c.heads, c.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, c.kv_heads, c.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = self._rotate(q, position), self._rotate(k, position)
        past = 0
        if state is not None:
            past = state[0].shape[-2]
            k, v = torch.cat((state[0], k), -2), torch.cat((state[1], v), -2)
        saved = None
        if return_state:
            keep = min(c.window-1, k.shape[-2])
            saved = (k[:, :, -keep:].contiguous().clone(), v[:, :, -keep:].contiguous().clone()) if keep else (k[:, :, :0].clone(), v[:, :, :0].clone())
        groups = c.heads//c.kv_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        if not past and t <= c.window:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            qi = torch.arange(past, past+t, device=x.device)[:, None]
            ki = torch.arange(past+t, device=x.device)[None]
            allowed = (ki <= qi) & (ki > qi-c.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        y = y.transpose(1, 2)*self.gate_proj(x).sigmoid()[..., None]
        return self.o_proj(y.reshape(b, t, -1)), saved


class ConvertedDeltaMemory(nn.Module):
    """Streaming REWT delta recurrence, with float32 recurrent accumulation."""
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.conv = nn.Conv1d(c.width, c.width, c.conv_width, groups=c.width, bias=False)
        self.q_proj = nn.Linear(c.width, c.width, bias=False)
        self.k_proj = nn.Linear(c.width, c.width, bias=False)
        self.v_proj = nn.Linear(c.width, c.width, bias=False)
        self.gate_proj = nn.Linear(c.width, c.width, bias=False)
        self.gates = nn.Linear(c.width, 2*c.heads, bias=True)
        self.norm = Norm(c.head_dim, c.norm_eps)
        self.o_proj = nn.Linear(c.width, c.width, bias=False)
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[c.heads:].fill_(3.)

    def forward(self, x, state=None, position=0, *, return_state=False):
        c = self.config; b, t, _ = x.shape; h, d = c.heads, c.head_dim
        memory_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if state is None:
            past = x.new_zeros(b, c.conv_width-1, c.width)
            memory = torch.zeros(b*h, d, d, device=x.device, dtype=memory_dtype)
        else:
            past, memory = state
        history = torch.cat((past, x), 1)
        features = F.silu(self.conv(history.transpose(1, 2)).transpose(1, 2))
        q, k, v = [projection(features).view(b, t, h, d).transpose(1, 2).reshape(b*h, t, d)
                   for projection in (self.q_proj, self.k_proj, self.v_proj)]
        # Autocast must not lower the precision of the triangular solve or state.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k = F.normalize(q.to(memory_dtype), dim=-1, eps=1e-6), F.normalize(k.to(memory_dtype), dim=-1, eps=1e-6)
            v = v.to(memory_dtype)
            gate_values = F.linear(features.to(memory_dtype), self.gates.weight.to(memory_dtype), self.gates.bias.to(memory_dtype))
            beta, decay = gate_values.view(b, t, 2, h).unbind(2)
            beta = beta.sigmoid().transpose(1, 2).reshape(b*h, t, 1)
            decay = (.9+.1*decay.sigmoid()).transpose(1, 2).reshape(b*h, t, 1)
            reads = []
            for start in range(0, t, c.chunk):
                end = min(start+c.chunk, t)
                if end-start == 1:
                    memory = decay[:, start, :, None]*memory
                    error = v[:, start]-(k[:, start, :, None]*memory).sum(-2)
                    memory = memory+beta[:, start, :, None]*k[:, start, :, None]*error[:, None]
                    read = (q[:, start, :, None]*memory).sum(-2)[:, None]
                else:
                    read, memory = gated_delta_chunk(k[:, start:end], v[:, start:end], beta[:, start:end],
                                                     decay[:, start:end], q[:, start:end], memory)
                reads.append(read)
            y = torch.cat(reads, 1).view(b, h, t, d).transpose(1, 2).to(features.dtype)
        y = self.norm(y).reshape(b, t, c.width)*F.silu(self.gate_proj(features))
        saved = None
        if return_state:
            keep = c.conv_width-1
            saved = (history[:, -keep:].contiguous().clone() if keep else history[:, :0].clone(), memory)
        return self.o_proj(y), saved


class ConversionEnergy(nn.Module):
    """Same quadratic-plus-Fourier refinement as streaming REWT, identity at init."""
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.amplitude = nn.Linear(c.width, c.modes, bias=False)
        self.raw_frequency = nn.Parameter(torch.randn(c.modes, c.width)/math.sqrt(c.width))
        self.phase = nn.Parameter(torch.rand(c.modes)*2*math.pi-math.pi)
        self.raw_precision = nn.Parameter(torch.zeros(c.width))
        nn.init.zeros_(self.amplitude.weight)

    def forward(self, anchor, rounds=None):
        rounds = self.config.rounds if rounds is None else rounds
        if type(rounds) is not int or rounds < 0:
            raise ValueError('invalid energy rounds')
        if not rounds:
            return anchor
        dtype = torch.float64 if anchor.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            x = anchor.to(dtype)
            amplitude = F.linear(x, self.amplitude.weight.to(dtype)).tanh()/math.sqrt(self.config.modes)
            raw = self.raw_frequency.to(dtype); norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            frequency = 2*raw*norm.tanh()/norm
            precision = .15+1.85*self.raw_precision.to(dtype).sigmoid()
            phase = self.phase.to(dtype)
            eta = .95/(precision.max()+amplitude.abs()@frequency.square().sum(-1))[..., None]
            delta = torch.zeros_like(x)
            for _ in range(rounds):
                gradient = precision*delta-(amplitude*F.linear(delta, frequency, phase).sin())@frequency
                delta = delta-eta*gradient
            return (x+delta).to(anchor.dtype)


class ConvertedBlock(nn.Module):
    def __init__(self, c, donor_type):
        super().__init__()
        self.mixer = ConvertedDeltaMemory(c) if donor_type == 'conv' else WindowAttention(c, donor_type)
        self.feed_forward = FeedForward(c)
        for name in ('operator_norm', 'post_operator_norm', 'ffn_norm', 'post_ffn_norm'):
            setattr(self, name, Norm(c.width, c.norm_eps))

    def operator(self, x, state=None, position=0, *, return_state=False):
        output, state = self.mixer(self.operator_norm(x), state, position, return_state=return_state)
        return self.post_operator_norm(output), state

    def forward(self, x, state=None, position=0, *, return_state=False):
        output, state = self.operator(x, state, position, return_state=return_state)
        x = x+output
        return x+self.post_ffn_norm(self.feed_forward(self.ffn_norm(x))), state


@dataclass
class ConversionState:
    layers: list
    position: int
    signature: tuple

    def allocated_bytes(self):
        storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                   for pair in self.layers if pair is not None for t in pair}
        return sum(storage.values())


class ConvertedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate(); self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(ConvertedBlock(config, t) for t in config.donor_layers)
        self.norm = Norm(config.width, config.norm_eps)
        self.energy = ConversionEnergy(config)
        self.gradient_checkpointing = False

    def signature(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.parameters())

    def hidden(self, tokens, state=None, *, return_state=False, rounds=None):
        c = self.config
        if tokens.ndim != 2 or tokens.dtype != torch.long or not all(tokens.shape):
            raise ValueError('expected nonempty [batch,time] int64 tokens')
        if tokens.device != self.embedding.weight.device or bool(((tokens < 0)|(tokens >= c.vocab_size)).any()):
            raise ValueError('token device or vocabulary mismatch')
        if state is not None and (state.signature != self.signature() or len(state.layers) != len(self.blocks)):
            raise ValueError('cached state weights changed or belong to another model')
        if self.gradient_checkpointing and self.training and (return_state or state is not None):
            raise ValueError('checkpointed training requires an empty, uncached sequence')
        position = state.position if state is not None else 0
        x = self.embedding(tokens); layers = []
        for i, block in enumerate(self.blocks):
            previous = state.layers[i] if state is not None else None
            if self.gradient_checkpointing and self.training:
                x = checkpoint(lambda z, current=block: current(z)[0], x, use_reentrant=False)
                new_state = None
            else:
                x, new_state = block(x, previous, position, return_state=return_state)
            layers.append(new_state)
        x = self.energy(self.norm(x), rounds)
        new_state = ConversionState(layers, position+tokens.shape[1], self.signature()) if return_state else None
        return x, new_state

    def forward(self, tokens, state=None, *, return_state=False, rounds=None, last_only=False):
        x, new_state = self.hidden(tokens, state, return_state=return_state, rounds=rounds)
        if last_only:
            x = x[:, -1:]
        return F.linear(x, self.embedding.weight), new_state

    def trainable_conversion_parameters(self):
        """Keep copied lexical/feedforward weights fixed during recovery."""
        self.requires_grad_(False)
        for block in self.blocks:
            block.mixer.requires_grad_(True)
            block.operator_norm.requires_grad_(True)
            block.post_operator_norm.requires_grad_(True)
        self.energy.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def transfer(self, donor):
        """Read a tensor mapping or safetensors reader; return a complete map."""
        records = []; source_keys = set(donor.keys())
        source = donor.get_tensor if hasattr(donor, 'get_tensor') else donor.__getitem__
        def put(target, name, *, part=None, transformation='copy'):
            if name not in source_keys:
                raise ValueError(f'missing donor weight: {name}')
            value = source(name)
            if part is not None:
                value = value.chunk(3, 0)[part]
            if value.shape != target.shape:
                raise ValueError(f'weight shape mismatch: {name}: {value.shape} vs {target.shape}')
            target.copy_(value)
            records.append(dict(source=name, source_part=part, shape=list(target.shape), transformation=transformation))
        put(self.embedding.weight, 'model.embed_tokens.weight')
        put(self.norm.weight, 'model.embedding_norm.weight')
        for index, block in enumerate(self.blocks):
            prefix = f'model.layers.{index}.'
            for name in ('operator_norm', 'post_operator_norm', 'ffn_norm', 'post_ffn_norm'):
                put(getattr(block, name).weight, prefix+name+'.weight')
            for name in ('w1', 'w2', 'w3'):
                put(getattr(block.feed_forward, name).weight, prefix+'feed_forward.'+name+'.weight')
            mixer = block.mixer
            if isinstance(mixer, WindowAttention):
                for name, parameter in mixer.named_parameters():
                    put(parameter, prefix+'self_attn.'+name)
            else:
                put(mixer.conv.weight, prefix+'conv.conv.weight', transformation='reuse kernel before projections; operation differs')
                put(mixer.q_proj.weight, prefix+'conv.in_proj.weight', part=1, transformation='C projection initializes query')
                put(mixer.k_proj.weight, prefix+'conv.in_proj.weight', part=0, transformation='B projection initializes key')
                put(mixer.v_proj.weight, prefix+'conv.in_proj.weight', part=2, transformation='x projection initializes value')
                put(mixer.gate_proj.weight, prefix+'conv.in_proj.weight', part=1, transformation='C projection also initializes output gate')
                put(mixer.o_proj.weight, prefix+'conv.out_proj.weight')
        return dict(teacher_repo=self.config.teacher_repo, teacher_revision=self.config.teacher_revision,
                    delta_layers=sum(t == 'conv' for t in self.config.donor_layers),
                    bounded_attention_layers=sum(t != 'conv' for t in self.config.donor_layers),
                    original_convolution_layers=0, global_attention_layers=0,
                    transferred=records, newly_initialized=['delta recurrence gates', 'delta read normalization', 'Fourier energy parameters'],
                    note='Convolution-to-delta transfer is heuristic initialization. Capability recovery requires training.')

    def save(self, path, *, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(dict(format='rewt_lfm_conversion_v1', config=asdict(self.config),
                        state_dict=self.state_dict(), metadata=metadata or {}), temporary)
        temporary.replace(path)


def load_converted(path, *, device='cpu', dtype=None):
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('format') != 'rewt_lfm_conversion_v1':
        raise ValueError('not a REWT converted checkpoint')
    config = ConversionConfig(**saved['config'])
    with torch.device('meta'):
        model = ConvertedLM(config)
    model.load_state_dict(saved['state_dict'], assign=True)
    model.to(device=device, dtype=dtype)
    return model, saved['metadata']
