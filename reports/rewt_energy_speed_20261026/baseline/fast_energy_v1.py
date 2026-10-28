"""Checkpoint-compatible execution of the streaming latent energy.

The first update starts at zero, so its phase is already known. Prepared
two-round inference moves a projection into a small, constant Gram matrix.
Optional compilation fuses pointwise work; it is lazy and has a startup cost.
No parameters, iteration counts, or curvature bounds are approximated.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .streaming import LatentEnergy


def _refine(anchor, weight, raw, phase, raw_precision, rounds):
    amplitude = F.linear(anchor, weight).tanh() / math.sqrt(weight.shape[0])
    norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    frequency = 2 * raw * norm.tanh() / norm
    precision = .15 + 1.85 * raw_precision.sigmoid()
    eta = .95 / (precision.max() + amplitude.abs() @ frequency.square().sum(-1))[..., None]
    # delta_0 = 0: avoid a zero allocation, zero projection and zero multiply.
    delta = eta * ((amplitude * phase.sin()) @ frequency)
    for _ in range(1, rounds):
        gradient = precision * delta - (amplitude * F.linear(delta, frequency, phase).sin()) @ frequency
        delta = delta - eta * gradient
    return anchor + delta


def _prepared_two(anchor, weight, phase, phase_sin, norm2, maximum, gram, basis):
    amplitude = F.linear(anchor, weight).tanh() / math.sqrt(weight.shape[0])
    eta = .95 / (maximum + amplitude.abs() @ norm2)[..., None]
    first = eta * amplitude * phase_sin
    second = eta * amplitude * (first @ gram + phase).sin()
    # delta_2 = (first + second) W - eta * first (W diag(q)).
    coefficients = torch.cat((first + second, -eta * first), dim=-1)
    return anchor + coefficients @ basis


def _prepared_general(anchor, weight, frequency, phase, phase_sin, precision, norm2, maximum, rounds):
    amplitude = F.linear(anchor, weight).tanh() / math.sqrt(weight.shape[0])
    eta = .95 / (maximum + amplitude.abs() @ norm2)[..., None]
    delta = eta * ((amplitude * phase_sin) @ frequency)
    for _ in range(1, rounds):
        gradient = precision * delta - (amplitude * F.linear(delta, frequency, phase).sin()) @ frequency
        delta = delta - eta * gradient
    return anchor + delta


class PreparedEnergy:
    """Inference-only constants owned by a weight-validated decoding session."""

    def __init__(self, head, rounds):
        self.head, self.rounds = head, rounds
        with torch.no_grad():
            raw = head.raw_frequency
            norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.frequency = 2 * raw * norm.tanh() / norm
            self.precision = .15 + 1.85 * head.raw_precision.sigmoid()
            self.norm2, self.maximum = self.frequency.square().sum(-1), self.precision.max()
            self.phase_sin = head.phase.sin()
            self.gram = self.frequency @ self.frequency.T
            self.basis = torch.cat((self.frequency, self.frequency * self.precision), dim=0)
        self.signature = self._signature()
        function = _prepared_two if rounds == 2 else _prepared_general
        self.run = torch.compile(function, fullgraph=True) if head.compiled else function

    def _signature(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.head.parameters())

    def __call__(self, anchor):
        if torch.is_grad_enabled():
            raise ValueError('prepared energy requires no_grad or inference_mode')
        if self._signature() != self.signature:
            raise ValueError('prepared energy weights changed')
        if self.rounds == 0:
            return anchor
        h = self.head
        if self.rounds == 2:
            return self.run(anchor, h.amplitude.weight, h.phase, self.phase_sin,
                            self.norm2, self.maximum, self.gram, self.basis)
        return self.run(anchor, h.amplitude.weight, self.frequency, h.phase,
                        self.phase_sin, self.precision, self.norm2, self.maximum, self.rounds)


class FastLatentEnergy(LatentEnergy):
    """Shares the original Parameter objects and keeps its state_dict keys."""

    def __init__(self, head, *, compiled=False):
        nn.Module.__init__(self)
        self.config = head.config
        self.amplitude = head.amplitude
        self.raw_frequency = head.raw_frequency
        self.phase = head.phase
        self.raw_precision = head.raw_precision
        self.compiled = compiled
        self.run = torch.compile(_refine, fullgraph=True) if compiled else _refine

    def forward(self, anchor, rounds, trace=False):
        if type(rounds) is not int or rounds < 0:
            raise ValueError('rounds must be nonnegative')
        if trace:
            # Diagnostics retain the original per-iteration energy trace.
            return super().forward(anchor, rounds, trace=True)
        if not rounds:
            return anchor, None
        return self.run(anchor, self.amplitude.weight, self.raw_frequency,
                        self.phase, self.raw_precision, rounds), None

    def prepare_inference(self, rounds):
        return PreparedEnergy(self, rounds)


def optimize_energy(model, *, compiled=False):
    """Enable exact fast energy in place; existing optimizers remain valid.

    Install before constructing inference sessions. Checkpoints still load in
    the original model; enable this execution option again after loading.
    Compilation is opt-in because short jobs may not recover its startup cost.
    """
    if model.energy_head is not None:
        model.energy_head = FastLatentEnergy(model.energy_head, compiled=compiled)
    return model
