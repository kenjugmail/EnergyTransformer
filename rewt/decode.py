"""Explicit, fixed-weight inference session with persistent observation caches.

The ordinary model forward retains its training and chunk-continuation behavior.
This session owns projected observations and mode matrices across decoding calls.
Changing parameter versions, buffers, dtype, device or training mode invalidates
the session. Parameter writes via .data bypass PyTorch's version tracking and
are unsupported, just as for autograd saved tensors.
"""
from __future__ import annotations

import torch

from .kernels import krylov2_refine, prepare_krylov2
from .model import EnergyState, JointEnergyLM, Output


def _weight_signature(model):
    return (id(model.config), model.identity,
            tuple((name, id(t), t._version, t.device, t.dtype)
                  for name, t in (*model.named_parameters(), *model.named_buffers())))


class EnergyDecodeSession:
    """Prefill once, then consume one observed byte per row at each step.

    No sampling, symbolic execution or reference answers occur here. A session
    is single-owner: start a separate session to branch a generation.
    """

    def __init__(self, model: JointEnergyLM, *, rounds: int | None = None):
        if not isinstance(model, JointEnergyLM) or model.config.support:
            raise ValueError('decode sessions currently support core energy models only')
        if hasattr(model.config, 'solver'):
            raise ValueError('architecture models use prepare_inference; GRU variants can use RingDecodeSession')
        if model.training:
            raise ValueError('call model.eval() before creating a decode session')
        if rounds is not None and (type(rounds) is not int or rounds < 0):
            raise ValueError('rounds must be nonnegative')
        self.model = model
        self.rounds = model.config.rounds if rounds is None else rounds
        self.signature = _weight_signature(model)
        self.state = None

    def _validate(self):
        if self.model.training or _weight_signature(self.model) != self.signature:
            raise RuntimeError('decode session invalidated by model changes; create a new session')

    @torch.inference_mode()
    def prefill(self, tokens):
        self._validate()
        model = self.model
        output = model(tokens, rounds=self.rounds, trace_energy=False)
        # This extra prefill-only projection builds a persistent cache. Decode
        # steps encode just the convolution tail and the newly supplied byte.
        initial = model.initial_state(len(tokens), device=tokens.device)
        bank = model.observed_bank(tokens, initial)[:, -min(tokens.shape[1], model.config.history):]
        self.keys, self.values = model.key(bank), model.value(bank)
        self.constants = (model.frequencies(), .15 + 1.85 * model.raw_precision.sigmoid())
        self.use_krylov2 = model.config.refinement_backend == 'krylov2' and self.rounds == 2
        if self.use_krylov2:
            self.krylov_context = prepare_krylov2(model, *self.constants)
        self.state = output.state
        return output

    @torch.inference_mode()
    def step(self, tokens):
        self._validate()
        if self.state is None:
            raise RuntimeError('prefill must be called before step')
        if (tokens.dtype != torch.long or tokens.shape != (self.state.joint.shape[0], 1)
                or tokens.device != self.state.joint.device):
            raise ValueError('step requires one int64 byte per row on the model device')
        if bool((tokens < 0).any()) or bool((tokens > 255).any()):
            raise ValueError('Bytes must be in 0..255')
        model, state = self.model, self.state
        c = model.config
        current = model.encode_current(tokens[:, 0], state)
        self.keys = torch.cat((self.keys[:, -(c.history - 1):] if c.history > 1 else self.keys[:, :0],
                               model.key(current)[:, None]), 1)
        self.values = torch.cat((self.values[:, -(c.history - 1):] if c.history > 1 else self.values[:, :0],
                                 model.value(current)[:, None]), 1)
        query_input = current + model.recurrent_condition(state) if c.context_kernel > 1 else current
        attention = model.retrieve(model.query(query_input), self.keys, self.values)
        cond, observed = model.condition(tokens[:, 0], state, constants=self.constants,
                                         encoded_current=current, attention_context=attention)
        if self.use_krylov2:
            kernel = getattr(model, '_compiled_refinement', krylov2_refine)
            refined, _, _ = kernel(cond.anchor, cond.amplitudes, model.phase, self.constants[0],
                                    *self.krylov_context, output_start=model.blocks['output'].start,
                                    centered=c.output_geometry == 'centered')
            cost = dict(gradient_calls=2, energy_calls=3 + 2 * c.backtracks,
                        accepted=0, attempts=2 * c.backtracks * len(tokens))
        else:
            initial = model.canonical_state(cond.anchor + c.initial_carry_scale * state.joint.tanh())
            refined, cost, _ = model.refine(initial, cond, rounds=self.rounds)
        self.state = EnergyState(refined, observed, state.count + 1, model.identity)
        return Output(refined[:, None, model.blocks['output']], self.state,
                      dict(cost=cost, refinement_backend='krylov2' if self.use_krylov2 else
                           'directional' if c.refinement_backend == 'directional' else 'reference',
                           persistent_decode_cache=True))
