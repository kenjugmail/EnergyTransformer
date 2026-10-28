"""Opt-in kernel and architecture study; legacy checkpoint defaults are intact.

The observation scan and delta memory depend on observed bytes only. Refined
states can feed back at explicit global chunk boundaries. Fixed byte patches
reuse the last correction between updates; only actual updates have a descent
guarantee for the current conditional energy.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .advanced_kernels import (affine_scan, delta_chunk, gru_from_input,
    packed_condition_weights, prepare_solver, reduced_refine)
from .kernels import krylov2_refine, prepare_krylov2
from .model import Conditioning, EnergyState, Output
from .position_memory import PositionMemoryConfig, PositionMemoryEnergyLM


@dataclass(frozen=True)
class ArchitectureConfig(PositionMemoryConfig):
    solver: str = 'krylov2'
    packed: bool = True
    backbone: str = 'gru'
    feedback_stride: int = 0
    delta_size: int = 0
    delta_chunk_size: int = 32
    patch_stride: int = 1
    mode_layout: str = 'shared'
    output_rank: int = 0
    preconditioner_factorization: str = 'dense'

    def validate(self):
        super().validate()
        if (self.support or self.activation != 'fourier' or self.schedule != 'all'
                or self.compatibility_geometry != 'anchor_relative'
                or self.wave_geometry != 'anchor_relative' or self.initial_carry_scale != 0
                or self.recurrent_type != 'gru' or self.halting_tolerance):
            raise ValueError('architecture variants require the anchor-relative Fourier GRU core')
        if self.solver not in ('reference', 'krylov2', 'certified', 'preconditioned'):
            raise ValueError('unknown solver')
        if self.preconditioner_factorization not in ('dense', 'structured'):
            raise ValueError('unknown preconditioner factorization')
        if self.backbone not in ('gru', 'scan') or self.mode_layout not in ('shared', 'hybrid'):
            raise ValueError('unknown backbone or mode layout')
        for name in ('feedback_stride', 'delta_size', 'output_rank'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('patch_stride', 'delta_chunk_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.feedback_stride and self.backbone != 'scan':
            raise ValueError('chunk feedback requires a scan backbone')
        if self.output_rank > min(self.width, self.vocab_size):
            raise ValueError('output rank exceeds latent width')


@dataclass
class ArchitectureState(EnergyState):
    scan: torch.Tensor
    memory: torch.Tensor
    correction: torch.Tensor

    def detach(self):
        return ArchitectureState(self.joint.detach(), self.observed.detach(), self.count, self.identity,
                                 self.scan.detach(), self.memory.detach(), self.correction.detach())


class FactorizedLinear(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.down = nn.Parameter(torch.empty(rank, inputs))
        self.up = nn.Parameter(torch.empty(outputs, rank))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up, a=math.sqrt(5))

    @property
    def weight(self):
        return self.up @ self.down

    def forward(self, x):
        return F.linear(F.linear(x, self.down), self.up, self.bias)


class ArchitectureEnergyLM(PositionMemoryEnergyLM):
    def __init__(self, config):
        super().__init__(config)
        d = config.width
        if config.backbone == 'scan':
            del self.recurrent_proposal
            self.scan_input = nn.Linear(d, 2 * d)
            if config.feedback_stride:
                self.feedback = nn.Linear(d, d, bias=False)
                self.feedback_gate = nn.Parameter(torch.full((d,), -3.))
        if config.delta_size:
            self.delta_key = nn.Linear(d, config.delta_size, bias=False)
            self.delta_query = nn.Linear(d, config.delta_size, bias=False)
            self.delta_value = nn.Linear(d, d, bias=False)
            self.delta_beta = nn.Linear(d, 1)
            self.memory_mix = nn.Linear(3 * d, d)
        if config.mode_layout == 'hybrid':
            with torch.no_grad():
                shared = config.modes // 2
                self.mode_mask[shared:] = 0
                blocks = tuple(self.blocks.values())
                for i in range(shared, config.modes):
                    self.mode_mask[i, blocks[(i - shared) % len(blocks)]] = 1
        if config.output_rank:
            self.decoder = FactorizedLinear(d, config.vocab_size, config.output_rank)
            self.output_prior = FactorizedLinear(3 * d, config.vocab_size, config.output_rank)

    def initial_state(self, batch, *, device=None, dtype=None):
        state = super().initial_state(batch, device=device, dtype=dtype)
        return ArchitectureState(state.joint, state.observed, state.count, state.identity,
            state.joint.new_zeros(batch, self.config.width),
            state.joint.new_zeros(batch, self.config.delta_size, self.config.width),
            torch.zeros_like(state.joint))

    def diversity_loss(self):
        frequency = F.normalize(self.frequencies(), dim=-1, eps=1e-6)
        gram = frequency @ frequency.T
        off_diagonal = gram - torch.diag_embed(gram.diagonal())
        return off_diagonal.square().sum() / max(1, self.config.modes * (self.config.modes - 1))

    @torch.inference_mode()
    def prepare_inference(self):
        """Explicit fixed-weight cache; version-checked on every inference call."""
        if self.training:
            raise ValueError('call eval before preparing inference')
        from .decode import _weight_signature
        constants = self.frequencies(), .15 + 1.85 * self.raw_precision.sigmoid()
        prepared = None
        if self.config.solver == 'krylov2':
            prepared = prepare_krylov2(self, *constants)
        elif self.config.solver in ('certified', 'preconditioned'):
            prepared = prepare_solver(self, *constants, self.config.solver)
        self._inference_cache = (_weight_signature(self), constants,
            packed_condition_weights(self) if self.config.packed else None, prepared)
        return self

    def _condition_batch(self, current, carry, recurrent, attention, constants, packed):
        features = torch.cat((current, carry, attention), -1)
        if packed is not None:
            hidden, output, amplitude = F.linear(features, *packed).split(
                (self.config.hidden, self.config.vocab_size, self.config.modes), -1)
            feedforward = self.prior[2](self.prior[1](hidden))
        else:
            feedforward, output, amplitude = self.prior(features), self.output_prior(features), self.amplitude(features)
        output = output + self.decoder((recurrent + attention + feedforward) / 3)
        anchor = self.canonical_state(torch.cat((recurrent, attention, feedforward, output), -1))
        amplitudes = .5 * amplitude.tanh() / math.sqrt(self.config.modes)
        return Conditioning(anchor, amplitudes, None, features, {}, *constants)

    def _refine_batch(self, cond, rounds, prepared, trace):
        c = self.config
        shape = cond.anchor.shape
        flat = Conditioning(cond.anchor.reshape(-1, self.dimensions), cond.amplitudes.reshape(-1, c.modes),
                            None, None, {}, cond.frequency, cond.precision)
        if c.solver == 'krylov2' and rounds == 2:
            result, energies, accepted = krylov2_refine(flat.anchor, flat.amplitudes, self.phase, cond.frequency,
                *prepared, output_start=self.blocks['output'].start, centered=c.output_geometry == 'centered')
            energies = list(energies.detach().unbind(-1)) if trace else []
            cost = dict(gradient_calls=2, energy_calls=3 + 2 * c.backtracks,
                        attempts=2 * c.backtracks * len(flat.anchor),
                        accepted=int(accepted.sum()) if trace else 0)
        elif c.solver in ('certified', 'preconditioned'):
            result, energies = reduced_refine(self, flat.anchor, flat.amplitudes, cond.frequency,
                cond.precision, prepared, c.solver, rounds, trace=trace)
            cost = dict(gradient_calls=rounds, energy_calls=rounds + 1 if trace else 0,
                        attempts=rounds * len(flat.anchor),
                        accepted=sum(int((b < a).sum()) for a, b in zip(energies, energies[1:])) if trace else 0)
        else:
            result, cost, energies = super().refine(flat.anchor, flat, rounds=rounds, trace=trace)
        return result.reshape(shape), (torch.stack(energies, -1).reshape(*shape[:-1], rounds + 1) if trace else None), cost

    def forward(self, tokens, state=None, *, rounds=None, details=False, trace_energy=None,
                state_gradient_interval=0, noise=None):
        c = self.config
        if noise is not None:
            raise ValueError('core architecture has no diffusion noise input')
        if tokens.ndim != 2 or tokens.dtype != torch.long or not tokens.shape[1] or not tokens.shape[0]:
            raise ValueError('tokens must be nonempty [batch,time] int64')
        if bool(((tokens < 0) | (tokens >= self.config.vocab_size)).any()):
            raise ValueError('token IDs outside the configured vocabulary')
        if type(state_gradient_interval) is not int or state_gradient_interval < 0:
            raise ValueError('invalid gradient interval')
        rounds = c.rounds if rounds is None else rounds
        if type(rounds) is not int or rounds < 0:
            raise ValueError('rounds must be nonnegative')
        trace_energy = details if trace_energy is None else trace_energy
        if trace_energy and not details:
            raise ValueError('energy traces require details')
        state = self.initial_state(len(tokens), device=tokens.device) if state is None else state
        expected = (len(tokens), self.dimensions)
        if (not isinstance(state, ArchitectureState) or state.identity != self.identity
                or type(state.count) is not int or state.count < 0
                or state.joint.shape != expected or state.correction.shape != expected
                or state.scan.shape != (len(tokens), c.width)
                or state.memory.shape != (len(tokens), c.delta_size, c.width)
                or state.observed.shape != (len(tokens), self.observed_capacity, c.width)
                or state.joint.device != tokens.device or state.joint.dtype != self.embedding.weight.dtype):
            raise ValueError('incompatible architecture state')
        cache = getattr(self, '_inference_cache', None)
        if cache is not None and not self.training and not torch.is_grad_enabled():
            from .decode import _weight_signature
            if cache[0] != _weight_signature(self):
                raise RuntimeError('fixed-weight cache invalidated; call prepare_inference again')
            _, constants, packed, prepared = cache
        else:
            constants = self.frequencies(), .15 + 1.85 * self.raw_precision.sigmoid()
            packed = packed_condition_weights(self) if c.packed else None
            prepared = None
            if c.solver == 'krylov2' and rounds == 2:
                prepared = prepare_krylov2(self, *constants)
            elif c.solver in ('certified', 'preconditioned'):
                prepared = prepare_solver(self, *constants, c.solver)
        bank = self.observed_bank(tokens, state)
        current = bank[:, self.observed_capacity:]
        keys, values = self.key(bank), self.value(bank)
        current_queries = self.query(current)
        if c.backbone == 'gru':
            input_affine = F.linear(current, self.recurrent_proposal.weight_ih, self.recurrent_proposal.bias_ih)
        else:
            gates, candidates = self.scan_input(current).chunk(2, -1)
            retention = gates.sigmoid()
            candidates = (1 - retention) * candidates.tanh()
        raw = torch.cat((state.observed, self.embedding(tokens)), 1)
        original_count, at = state.count, 0
        outputs, anchors, features, amplitudes, trajectories, updates = [], [], [], [], [], []
        total_cost = dict(gradient_calls=0, energy_calls=0, attempts=0, accepted=0)
        while at < tokens.shape[1]:
            count = original_count + at
            if self.training and state_gradient_interval and count and count % state_gradient_interval == 0:
                state = ArchitectureState(state.joint.detach(), state.observed, state.count, state.identity,
                                          state.scan.detach(), state.memory.detach(), state.correction.detach())
            length = 1 if c.backbone == 'gru' else tokens.shape[1] - at
            for interval in (c.feedback_stride, c.delta_chunk_size if c.delta_size else 0,
                             state_gradient_interval if self.training else 0):
                if interval:
                    length = min(length, interval - count % interval)
            chunk = current[:, at:at + length]
            if c.backbone == 'gru':
                carry = self.recurrent_condition(state)[:, None]
                recurrent = gru_from_input(input_affine[:, at], carry[:, 0], self.recurrent_proposal)[:, None]
            else:
                seed = state.scan
                if c.feedback_stride and count and count % c.feedback_stride == 0:
                    seed = seed + self.feedback_gate.sigmoid() * self.feedback(state.joint[:, self.blocks['recurrent']]).tanh()
                recurrent = affine_scan(retention[:, at:at + length], candidates[:, at:at + length], seed)
                carry = torch.cat((seed[:, None], recurrent[:, :-1]), 1).clamp(-1, 1)
            queries = current_queries[:, at:at + length]
            if c.context_kernel > 1:
                queries = queries + self.query(carry)
            # Batched local softmax: all queries in a scan chunk are independent
            # of refinement; the mask excludes future and prefix-padding keys.
            begin = max(self.observed_capacity - original_count, self.observed_capacity + at - c.history + 1)
            end = self.observed_capacity + at + length
            k, v = keys[:, begin:end], values[:, begin:end]
            h, d = c.attention_heads, c.width // c.attention_heads
            q = queries.reshape(len(tokens), length, h, d).transpose(1, 2)
            k = k.reshape(len(tokens), -1, h, d).transpose(1, 2)
            v = v.reshape(len(tokens), -1, h, d).transpose(1, 2)
            positions = torch.arange(begin, end, device=tokens.device)
            ends = torch.arange(self.observed_capacity + at, end, device=tokens.device)
            allowed = (positions[None] <= ends[:, None]) & (positions[None] > ends[:, None] - c.history)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            attention = (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v).transpose(1, 2).flatten(2)
            memory = state.memory
            if c.delta_size:
                dk = F.normalize(self.delta_key(chunk), dim=-1, eps=1e-6)
                dq = F.normalize(self.delta_query(chunk + carry), dim=-1, eps=1e-6)
                long, memory = delta_chunk(dk, self.delta_value(chunk), self.delta_beta(chunk).sigmoid(), dq, memory)
                mix = self.memory_mix(torch.cat((chunk, carry, attention), -1)).sigmoid()
                attention = attention + mix * long
            cond = self._condition_batch(chunk, carry, recurrent, attention, constants, packed)
            active = (torch.arange(count, count + length, device=tokens.device) + 1) % c.patch_stride == 0
            # Always use the global byte count, so call boundaries cannot create
            # extra latent updates. Between patches retain only the correction.
            indices = active.nonzero().flatten()
            traces = None
            cost = None
            if c.patch_stride == 1:
                joint, traces, cost = self._refine_batch(cond, rounds, prepared, trace_energy)
                correction = joint[:, -1] - cond.anchor[:, -1]
            elif len(indices):
                selected = Conditioning(cond.anchor[:, indices], cond.amplitudes[:, indices], None, None, {}, *constants)
                refined, traces, cost = self._refine_batch(selected, rounds, prepared, trace_energy)
                corrections = torch.cat((state.correction[:, None], refined - selected.anchor), 1)
                choice = active.long().cumsum(0)
                all_corrections = corrections[:, choice]
                joint = cond.anchor + all_corrections
                correction = all_corrections[:, -1]
            else:
                joint = cond.anchor + state.correction[:, None]
                correction = state.correction
            outputs.append(joint)
            if cost is not None:
                for key in total_cost:
                    total_cost[key] += cost[key]
            if details:
                anchors.append(cond.anchor); features.append(cond.features); amplitudes.append(cond.amplitudes)
                updates.append(active)
                if trace_energy and traces is not None:
                    trajectories.append(traces)
            at += length
            state = ArchitectureState(joint[:, -1], raw[:, at:at + self.observed_capacity],
                original_count + at, self.identity, recurrent[:, -1], memory, correction)
        joint = torch.cat(outputs, 1)
        info = dict(refinement_backend='reference' if c.solver == 'krylov2' and rounds != 2 else c.solver,
                    refinement_positions=sum(
                    (original_count + t + 1) % c.patch_stride == 0 for t in range(tokens.shape[1])))
        info['cost'] = total_cost
        info['cost_scope'] = 'batched refinement operations; geometry preparation excluded; acceptance counted only in traces'
        if prepared is not None and isinstance(prepared, dict):
            info['certified_step'] = prepared['step'].detach()
            info['curvature_upper'] = prepared['upper'].detach()
        if details:
            anchor, feature, amplitude = torch.cat(anchors, 1), torch.cat(features, 1), torch.cat(amplitudes, 1)
            info.update(joint=joint, initial_joint=anchor, initial_logits=anchor[..., self.blocks['output']],
                        features=feature, update_mask=torch.cat(updates),
                        conditioning=Conditioning(anchor, amplitude, None, feature, {}, *constants))
            if trace_energy:
                info['energy_trace'] = torch.cat(trajectories, 1) if trajectories else joint.new_empty(len(tokens), 0, rounds + 1)
        return Output(joint[..., self.blocks['output']], state, info)


def load_architecture_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ArchitectureEnergyLM(ArchitectureConfig(**saved['config']))
    model.load_state_dict(saved['weights'], strict=True)
    return model, saved
