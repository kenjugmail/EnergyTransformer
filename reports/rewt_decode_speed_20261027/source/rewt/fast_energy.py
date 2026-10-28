"""Checkpoint-compatible execution of the streaming latent energy.

The first update starts at zero, so its phase is already known. Prepared
two-round inference moves a projection into a small, constant Gram matrix.
Optional compilation fuses pointwise work; it is lazy and has a startup cost.
No parameters, iteration counts, or curvature bounds are approximated.
"""
import math

import numpy as np
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


class _SmallCPU:
    """Reduce framework dispatch for <=16 float32 CPU rows; inference only."""

    def __init__(self, prepared):
        self.weight = prepared.head.amplitude.weight.detach().numpy().T
        self.phase = prepared.head.phase.detach().numpy()
        for name in ('phase_sin', 'norm2', 'maximum', 'gram', 'basis'):
            setattr(self, name, getattr(prepared, name).numpy())
        self.scale = np.float32(self.weight.shape[1] ** -.5)

    def __call__(self, anchor):
        x = anchor.detach().numpy().reshape(-1, anchor.shape[-1])
        amplitude = np.tanh(x @ self.weight) * self.scale
        eta = np.float32(.95) / (self.maximum + np.abs(amplitude) @ self.norm2)[:, None]
        first = eta * amplitude * self.phase_sin
        second = eta * amplitude * np.sin(first @ self.gram + self.phase)
        output = x + np.concatenate((first + second, -eta * first), axis=-1) @ self.basis
        return torch.from_numpy(output.reshape(anchor.shape))


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
        self.cpu_small = (_SmallCPU(self) if rounds == 2 and not head.compiled and
                          head.raw_frequency.device.type == 'cpu' and head.raw_frequency.dtype == torch.float32 else None)
        if head.fused:
            if rounds != 2 or head.raw_frequency.device.type != 'cuda' or head.raw_frequency.dtype != torch.float32:
                raise ValueError('fused energy requires two rounds and CUDA float32')
            from .fast_energy_triton import prepared_two_triton
            self.run = prepared_two_triton

    def _signature(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.head.parameters())

    def __call__(self, anchor):
        if torch.is_grad_enabled():
            raise ValueError('prepared energy requires no_grad or inference_mode')
        if self._signature() != self.signature:
            raise ValueError('prepared energy weights changed')
        return self._apply_validated(anchor)

    def _apply_validated(self, anchor):
        # StreamingSession already validates every model parameter and disables
        # gradients once per consume. Avoid checking the same head twice there.
        if self.rounds == 0:
            return anchor
        if self.cpu_small is not None and anchor.dtype == torch.float32 and anchor.device.type == 'cpu' and anchor.numel() <= 16 * anchor.shape[-1]:
            return self.cpu_small(anchor)
        h = self.head
        if self.rounds == 2:
            return self.run(anchor, h.amplitude.weight, h.phase, self.phase_sin,
                            self.norm2, self.maximum, self.gram, self.basis)
        return self.run(anchor, h.amplitude.weight, self.frequency, h.phase,
                        self.phase_sin, self.precision, self.norm2, self.maximum, self.rounds)


class FastLatentEnergy(LatentEnergy):
    """Shares the original Parameter objects and keeps its state_dict keys."""

    def __init__(self, head, *, compiled=False, fused=False):
        nn.Module.__init__(self)
        self.training = head.training
        self.config = head.config
        self.amplitude = head.amplitude
        self.raw_frequency = head.raw_frequency
        self.phase = head.phase
        self.raw_precision = head.raw_precision
        self.compiled = compiled
        self.fused = fused
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


def optimize_energy(model, *, compiled=False, fused=False):
    """Enable exact fast energy in place; existing optimizers remain valid.

    Install before constructing inference sessions. Checkpoints still load in
    the original model; enable this execution option again after loading.
    Compilation is opt-in because short jobs may not recover its startup cost.
    Small CPU float32 inference uses NumPy views; larger batches use PyTorch.
    Fused CUDA inference is opt-in and supports two rounds, float32, width <=128,
    and <=32 modes. Training always uses differentiable PyTorch operations.
    """
    if model.energy_head is not None:
        model.energy_head = FastLatentEnergy(model.energy_head, compiled=compiled, fused=fused)
    return model
