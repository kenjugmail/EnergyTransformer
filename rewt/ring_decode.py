"""Fixed-shape, fixed-weight core decode with an optional whole-step CUDA graph.

KV entries are overwritten in circular slots. A device scalar controls position,
validity and rollover; no growing tensor or host scalar enters the captured step.
Returned logits own their storage, so graph replay cannot overwrite old outputs.
"""
import math

import torch
from torch.nn import functional as F

from .advanced_kernels import gru_from_input, packed_condition_weights, prepare_solver, reduced_refine
from .decode import _weight_signature
from .kernels import krylov2_refine, prepare_krylov2
from .model import Conditioning
from .position_memory import PositionMemoryEnergyLM


class RingDecodeSession:
    def __init__(self, model, *, cuda_graph=False, triton=False, triton_tile=0):
        c = model.config
        if (not isinstance(model, PositionMemoryEnergyLM) or c.support or c.recurrent_type != 'gru'
                or c.compatibility_geometry != 'anchor_relative' or c.wave_geometry != 'anchor_relative'
                or c.initial_carry_scale or c.activation != 'fourier' or c.schedule != 'all'
                or c.halting_tolerance or getattr(c, 'backbone', 'gru') != 'gru'
                or getattr(c, 'delta_size', 0) or getattr(c, 'patch_stride', 1) != 1):
            raise ValueError('ring session requires the anchor-relative position-memory GRU core')
        if model.training:
            raise ValueError('call eval before creating a session')
        if (cuda_graph or triton) and next(model.parameters()).device.type != 'cuda':
            raise ValueError('CUDA graph and Triton require a CUDA model')
        self.model, self.cuda_graph, self.triton = model, cuda_graph, triton
        if triton_tile not in (0, 64, 128, 256) or (triton_tile and not triton):
            raise ValueError('triton_tile requires Triton and must be 0/64/128/256')
        self.triton_tile = triton_tile
        self.solver = getattr(c, 'solver', c.refinement_backend)
        if triton and self.solver != 'preconditioned':
            raise ValueError('Triton kernel requires the preconditioned solver')
        self.signature = _weight_signature(model)
        self.ready = False

    def _validate(self):
        if self.model.training or _weight_signature(self.model) != self.signature:
            raise RuntimeError('ring session invalidated by changed weights or configuration')

    @torch.inference_mode()
    def prefill(self, tokens):
        self._validate()
        model, c = self.model, self.model.config
        output = model(tokens)
        initial = model.initial_state(len(tokens), device=tokens.device)
        count = tokens.shape[1]
        bank = model.observed_bank(tokens, initial)[:, -min(count, c.history):]
        p = model.embedding.weight
        self.keys = p.new_zeros(len(tokens), c.history, c.width)
        self.values = torch.zeros_like(self.keys)
        positions = torch.arange(count - bank.shape[1], count, device=tokens.device) % c.history
        self.keys.index_copy_(1, positions, model.key(bank))
        self.values.index_copy_(1, positions, model.value(bank))
        self.joint = output.state.joint.clone()
        tail = c.context_kernel - 1
        self.tail = output.state.observed[:, -tail:].clone() if tail else output.state.observed[:, :0].clone()
        self.count = torch.tensor(count, device=tokens.device, dtype=torch.long)
        self.input = torch.zeros(len(tokens), 1, device=tokens.device, dtype=torch.long)
        self.slots = torch.arange(c.history, device=tokens.device)
        self.tail_offsets = torch.arange(-tail, 1, device=tokens.device)
        self.constants = model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid()
        self.packed = packed_condition_weights(model)
        self.prepared = None
        if self.solver == 'krylov2' and c.rounds == 2:
            self.prepared = prepare_krylov2(model, *self.constants)
        elif self.solver in ('preconditioned', 'certified'):
            self.prepared = prepare_solver(model, *self.constants, self.solver)
            if self.triton:
                self.prepared = {k: v.contiguous() for k, v in self.prepared.items()}
                self.prepared['output_mean'] = self.prepared['lift'][:, model.blocks['output'].start:].mean(-1).contiguous()
        if self.cuda_graph:
            # Warm and capture on a side stream, then restore the exact prefill
            # state. Graph construction must not consume extra user bytes.
            buffers = (self.keys, self.values, self.joint, self.tail, self.count)
            snapshots = tuple(t.clone() for t in buffers)
            stream = torch.cuda.Stream(device=tokens.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize(tokens.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.graph_logits = self._step()
            for buffer, snapshot in zip(buffers, snapshots):
                buffer.copy_(snapshot)
        self.ready = True
        return output

    def _step(self):
        model, c = self.model, self.model.config
        raw = torch.cat((self.tail, model.embedding(self.input)), 1)
        positions = self.count + self.tail_offsets
        positioned = raw + model.position(positions % c.position_period)[None] * (positions >= 0)[None, :, None]
        current = model.encode_observed(positioned)[:, -1]
        slot = (self.count % c.history).reshape(1)
        self.keys.index_copy_(1, slot, model.key(current)[:, None])
        self.values.index_copy_(1, slot, model.value(current)[:, None])
        carry = self.joint[:, model.blocks['recurrent']].clamp(-1, 1)
        query = model.query(current + carry if c.context_kernel > 1 else current)
        h, d = c.attention_heads, c.width // c.attention_heads
        q = query.reshape(-1, h, d)
        k = self.keys.reshape(-1, c.history, h, d)
        v = self.values.reshape(-1, c.history, h, d)
        scores = (k * q[:, None]).sum(-1) / math.sqrt(d)
        valid = self.slots < (self.count + 1).clamp_max(c.history)
        attention = (scores.masked_fill(~valid[None, :, None], -torch.inf).softmax(1)[..., None] * v).sum(1).flatten(1)
        features = torch.cat((current, carry, attention), -1)
        hidden, output, amplitude = F.linear(features, *self.packed).split((c.hidden, c.vocab_size, c.modes), -1)
        feedforward = model.prior[2](model.prior[1](hidden))
        cell = model.recurrent_proposal
        recurrent = gru_from_input(F.linear(current, cell.weight_ih, cell.bias_ih), carry, cell)
        output = output + model.decoder((recurrent + attention + feedforward) / 3)
        anchor = model.canonical_state(torch.cat((recurrent, attention, feedforward, output), -1))
        amplitudes = .5 * amplitude.tanh() / math.sqrt(c.modes)
        if self.solver == 'krylov2' and c.rounds == 2:
            refined, _, _ = krylov2_refine(anchor, amplitudes, model.phase, self.constants[0], *self.prepared,
                output_start=model.blocks['output'].start, centered=c.output_geometry == 'centered')
        elif self.solver in ('preconditioned', 'certified'):
            if self.triton:
                from .triton_refine import triton_preconditioned
                refined = triton_preconditioned(anchor, amplitudes, model.phase, self.prepared,
                    rounds=c.rounds, output_start=model.blocks['output'].start, centered=c.output_geometry == 'centered',
                    tile_size=self.triton_tile)
            else:
                refined, _ = reduced_refine(model, anchor, amplitudes, *self.constants, self.prepared,
                                            self.solver, c.rounds)
        else:
            cond = Conditioning(anchor, amplitudes, None, features, {}, *self.constants)
            refined, _, _ = model.refine(anchor, cond)
        self.joint.copy_(refined)
        self.tail.copy_(raw[:, 1:])
        self.count.add_(1)
        return refined[:, None, model.blocks['output']]

    @torch.inference_mode()
    def step(self, tokens, *, validate_values=True):
        self._validate()
        if not self.ready:
            raise RuntimeError('prefill first')
        if tokens.shape != self.input.shape or tokens.dtype != torch.long or tokens.device != self.input.device:
            raise ValueError('one int64 byte per row required on the session device')
        # Greedy argmax callers may bypass the host range check, since argmax
        # over the configured logits guarantees a valid token. Check user input.
        if validate_values and bool(((tokens < 0) | (tokens >= self.model.config.vocab_size)).any()):
            raise ValueError('token IDs outside the configured vocabulary')
        self.input.copy_(tokens)
        if self.cuda_graph:
            self.graph.replay()
            return self.graph_logits.clone()
        return self._step().clone()
