"""Fixed-shape one-token decoding and optional CUDA graph replay.

External inputs are checked once. Generated tokens stay on the device and are
known to be in vocabulary by construction. Caches, positions and graph outputs
belong to one session; model weights are never changed.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .fast_energy import _prepared_two, _prepared_general
from .streaming import Attention
from .streaming_decode import StreamingSession


class _TokenStep(nn.Module):
    def __init__(self, model, rounds, capacity):
        super().__init__()
        self.model = model
        self.rounds = rounds
        self.capacity = capacity
        parameter = model.embedding.weight
        half = model.config.width // model.config.heads // 2
        self.register_buffer('frequency', torch.exp(-math.log(10000.) * torch.arange(half, device=parameter.device, dtype=parameter.dtype) / half))
        size = model.config.window or capacity
        self.register_buffer('slots', torch.arange(size, device=parameter.device))
        if model.energy_head is not None and rounds:
            head = model.energy_head
            with torch.no_grad():
                raw = head.raw_frequency
                norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                w = 2 * raw * norm.tanh() / norm
                q = .15 + 1.85 * head.raw_precision.sigmoid()
                values = dict(w=w, q=q, norm2=w.square().sum(-1), maximum=q.max(),
                              phase_sin=head.phase.sin(), gram=w @ w.T,
                              basis=torch.cat((w, w*q), 0))
                for name, value in values.items():
                    self.register_buffer(name, value)

    def forward(self, tokens, position, states):
        model = self.model; c = model.config
        batch = tokens.shape[0]; heads = c.heads; dim = c.width // heads
        x = model.embedding(tokens[:, 0])
        angle = position.to(x.dtype) * self.frequency
        cosine, sine = angle.cos(), angle.sin()
        for index, block in enumerate(model.blocks):
            z = block.norm1(x); mixer = block.mixer
            first, second = states[index]
            if isinstance(mixer, Attention):
                query, key, value = mixer.qkv(z).view(batch, 3, heads, dim).unbind(1)
                qa, qb = query.chunk(2, -1); ka, kb = key.chunk(2, -1)
                query = torch.cat((qa*cosine-qb*sine, qa*sine+qb*cosine), -1)
                key = torch.cat((ka*cosine-kb*sine, ka*sine+kb*cosine), -1)
                slot = position.remainder(c.window) if c.window else position
                first.index_copy_(2, slot, key[:, :, None])
                second.index_copy_(2, slot, value[:, :, None])
                # A ring is a permutation of the same rotated keys and values.
                valid = self.slots <= position
                output = F.scaled_dot_product_attention(query[:, :, None], first, second, attn_mask=valid[None, :])
                y = mixer.out(output.reshape(batch, c.width))
            else:
                history = torch.cat((first, z[:, None]), 1)
                features = F.silu((history*mixer.conv.weight[:, 0].T).sum(1)+mixer.conv.bias)
                query, key, value, gate = mixer.qkvg(features).view(batch, 4, heads, dim).unbind(1)
                query = F.normalize(query, dim=-1, eps=1e-6).reshape(batch*heads, dim)
                key = F.normalize(key, dim=-1, eps=1e-6).reshape(batch*heads, dim)
                value = value.reshape(batch*heads, dim)
                beta, decay = mixer.gates(features).view(batch, 2, heads).unbind(1)
                beta = beta.sigmoid().reshape(batch*heads, 1, 1)
                decay = (.9+.1*decay.sigmoid()).reshape(batch*heads, 1, 1)
                decayed = decay*second
                error = value-(key[:, :, None]*decayed).sum(-2)
                memory = decayed+beta*key[:, :, None]*error[:, None]
                read = (query[:, :, None]*memory).sum(-2).view(batch, heads, dim)
                y = mixer.out((mixer.norm(read)*F.silu(gate)).reshape(batch, c.width))
                first.copy_(history[:, -2:]); second.copy_(memory)
            x = x+y
            a, b = block.up(block.norm2(x)).chunk(2, -1)
            x = x+block.down(F.silu(a)*b)
        x = model.norm(x)
        head = model.energy_head
        if head is not None and self.rounds:
            phase = head.phase if head.context_phase is None else head.phase + head.context_phase(x)
            phase_sin = self.phase_sin if head.context_phase is None else phase.sin()
            if self.rounds == 2:
                x = _prepared_two(x, head.amplitude.weight, phase, phase_sin,
                                  self.norm2, self.maximum, self.gram, self.basis)
            else:
                x = _prepared_general(x, head.amplitude.weight, self.w, phase, phase_sin,
                                      self.q, self.norm2, self.maximum, self.rounds)
        position.add_(1)
        return F.linear(x, model.embedding.weight)[:, None]


class FastDecodeSession:
    """Reference prefill, then eager/compiled/graph token decoding.

    `generate` returns [batch,steps] device token IDs. It does not synchronize
    every token to the CPU. `consume` still validates arbitrary external IDs.
    First token-mode use includes compilation/capture when requested.
    """

    def __init__(self, model, *, batch=1, capacity=8192, rounds=None, backend='compile'):
        if backend not in ('eager', 'compile', 'graph'):
            raise ValueError('backend must be eager, compile, or graph')
        if backend == 'graph' and model.embedding.weight.device.type != 'cuda':
            raise ValueError('graph decoding requires CUDA')
        self.prefill = StreamingSession(model, batch=batch, capacity=capacity, rounds=rounds)
        self.model = model; self.batch = batch; self.capacity = capacity
        self.rounds = self.prefill.rounds; self.signature = model.signature()
        self.backend = backend; self.position = 0; self.states = None
        self._last_logits = None; self.graph = None

    def _check_model(self):
        if self.model.training or self.model.signature() != self.signature:
            raise ValueError('session weights or evaluation mode changed')

    @torch.inference_mode()
    def prepare(self):
        self._check_model()
        if self.states is not None:
            return
        if not self.position:
            raise ValueError('consume a prompt before preparing decoding')
        c = self.model.config; p = self.model.embedding.weight
        self.states = []
        for index, block in enumerate(self.model.blocks):
            if isinstance(block.mixer, Attention):
                length = c.window or self.capacity
                shape = (self.batch, c.heads, length, c.width//c.heads)
                key, value = p.new_zeros(shape), p.new_zeros(shape)
                oldk, oldv = self.prefill.buffers[index]
                count = min(self.position, c.window-1) if c.window else self.position
                indices = torch.arange(self.position-count, self.position, device=p.device)
                if c.window:
                    indices = indices.remainder(c.window)
                key.index_copy_(2, indices, oldk[:, :, :count])
                value.index_copy_(2, indices, oldv[:, :, :count])
                self.states.append((key, value))
            else:
                self.states.append(tuple(t.clone() for t in self.prefill.layers[index]))
        self.token = torch.zeros(self.batch, 1, device=p.device, dtype=torch.long)
        self.device_position = torch.tensor([self.position], device=p.device, dtype=torch.long)
        step = _TokenStep(self.model, self.rounds, self.capacity).eval()
        self.step = step if self.backend == 'eager' else torch.compile(step, fullgraph=True)
        if self.backend != 'eager':
            saved = [(a.clone(), b.clone()) for a, b in self.states]
            for _ in range(3):
                self.device_position.zero_()
                self.step(self.token, self.device_position, self.states)
            if self.backend == 'graph':
                torch.cuda.synchronize(p.device)
                stream = torch.cuda.Stream(device=p.device)
                stream.wait_stream(torch.cuda.current_stream(p.device))
                self.device_position.zero_()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(self.graph, stream=stream):
                        self.graph_logits = self.step(self.token, self.device_position, self.states)
                torch.cuda.current_stream(p.device).wait_stream(stream)
            for (a, b), (old_a, old_b) in zip(self.states, saved):
                a.copy_(old_a); b.copy_(old_b)
            self.device_position.fill_(self.position)
        self.prefill = None

    def _advance(self, tokens):
        if self.graph is not None:
            self.token.copy_(tokens)
            self.graph.replay()
            logits = self.graph_logits
        else:
            logits = self.step(tokens, self.device_position, self.states)
        self.position += 1
        self._last_logits = logits
        return logits

    @torch.inference_mode()
    def consume(self, tokens):
        self._check_model()
        if tokens.ndim != 2 or tokens.shape[0] != self.batch or not tokens.shape[1] or tokens.dtype != torch.long:
            raise ValueError('expected nonempty [session_batch,time] int64')
        if tokens.device != self.model.embedding.weight.device or bool(((tokens < 0)|(tokens >= self.model.config.vocab_size)).any()):
            raise ValueError('token device or vocabulary mismatch')
        if self.position+tokens.shape[1] > self.capacity:
            raise ValueError('session capacity exceeded')
        if self.states is None:
            logits = self.prefill.consume(tokens)
            self.position += tokens.shape[1]; self._last_logits = logits[:, -1:]
            return logits
        rows = [self._advance(token[:, None]).clone() for token in tokens.unbind(1)]
        return torch.cat(rows, 1)

    @torch.inference_mode()
    def generate(self, steps, *, temperature=0., top_k=40, generator=None):
        self._check_model()
        if type(steps) is not int or steps < 1 or temperature < 0 or not 1 <= top_k <= self.model.config.vocab_size:
            raise ValueError('invalid generation options')
        if self._last_logits is None:
            raise ValueError('consume a prompt before generation')
        if self.position+steps > self.capacity:
            raise ValueError('session capacity exceeded')
        self.prepare()
        output = []
        for _ in range(steps):
            scores = self._last_logits[:, -1]
            if temperature:
                values, indices = (scores/temperature).topk(top_k, -1)
                token = indices.gather(-1, torch.multinomial(values.softmax(-1), 1, generator=generator))
            else:
                token = scores.argmax(-1, keepdim=True)
            output.append(token)
            self._advance(token)
        return torch.cat(output, 1)

    def allocated_state_bytes(self):
        if self.states is None:
            return self.prefill.allocated_state_bytes()
        tensors = [t for pair in self.states for t in pair]+[self.device_position, self.token]
        storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
        return sum(storage.values())
