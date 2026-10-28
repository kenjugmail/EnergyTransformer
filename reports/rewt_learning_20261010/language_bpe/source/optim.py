"""Causal gradient-history filters and comparable stability diagnostics."""
from __future__ import annotations

from collections import deque
import math

import torch


class GradientFilter:
    def __init__(self, parameters, method="clip", *, history=8, cutoff=0.5,
                 ema_decay=0.9, max_norm=1.0, spike_threshold=5.0, residual_gain=0.25):
        if method not in ("none", "clip", "ema", "fourier", "ema_residual", "fourier_residual"):
            raise ValueError("Unknown gradient filter")
        if type(history) is not int or history < 2 or not 0 < cutoff <= 1:
            raise ValueError("Require 2+ history steps and cutoff in (0,1]")
        if not 0 <= ema_decay < 1 or max_norm <= 0 or spike_threshold <= 0:
            raise ValueError("Invalid gradient-filter settings")
        if not math.isfinite(residual_gain) or residual_gain < 0:
            raise ValueError("Residual gain must be finite and nonnegative")
        self.parameters = [p for p in parameters if p.requires_grad]
        self.method, self.cutoff = method, cutoff
        self.ema_decay, self.max_norm = ema_decay, max_norm
        self.spike_threshold = spike_threshold
        self.residual_gain = residual_gain
        self.history = deque(maxlen=history)
        self.ema = None

    def state_dict(self):
        return dict(history=list(self.history), ema=self.ema)

    def load_state_dict(self, state):
        self.history.clear()
        self.history.extend(state["history"])
        self.ema = state["ema"]

    def apply(self):
        if any(p.grad is None for p in self.parameters):
            # Zero placeholders preserve fixed coordinates for parameters that
            # lack supervision in a particular masked minibatch.
            current = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                 for p in self.parameters])
        else:
            current = torch.cat([p.grad.flatten() for p in self.parameters])
        if not bool(torch.isfinite(current).all()):
            raise FloatingPointError("Non-finite gradient component")
        # Finite float32 components can have a norm whose intermediate squares
        # overflow float32. Clip the actual vector using a wide reduction; an
        # overflow in a diagnostic must not masquerade as an invalid gradient.
        wide = current.double()
        raw_norm = float(wide.norm())
        self.history.append(current.detach().clone())
        filtered = wide
        high_fraction = None
        if len(self.history) >= 2:
            # A fixed strided coordinate sample is a cheap diagnostic for EVERY
            # arm; filtering itself uses all coordinates and its cost is timed.
            stride = max(1, current.numel() // 1024)
            diagnostic = torch.stack([g[::stride] for g in self.history]).double()
            spectrum = torch.fft.rfft(diagnostic, dim=0)
            power = spectrum.abs().square().sum(1)
            cutoff_index = max(1, math.ceil(power.numel() * self.cutoff))
            high_fraction = float(power[cutoff_index:].sum() / power.sum().clamp_min(1e-12))
        if self.method in ("ema", "ema_residual"):
            self.ema = wide.detach().clone() if self.ema is None else (
                self.ema_decay * self.ema.double() + (1 - self.ema_decay) * wide)
            filtered = self.ema if self.method == "ema" else wide + self.residual_gain * self.ema
        elif self.method in ("fourier", "fourier_residual") and len(self.history) == self.history.maxlen:
            sequence = torch.stack(tuple(self.history)).double()
            spectrum = torch.fft.rfft(sequence, dim=0)
            cutoff_index = max(1, math.ceil(spectrum.shape[0] * self.cutoff))
            spectrum[cutoff_index:] = 0
            slow = torch.fft.irfft(spectrum, n=len(sequence), dim=0)[-1]
            filtered = slow if self.method == "fourier" else current.double() + self.residual_gain * slow
        if not bool(torch.isfinite(filtered).all()):
            raise FloatingPointError("Non-finite filtered gradient component")
        filtered_norm = float(filtered.double().norm())
        if self.method != "none":
            filtered = filtered * min(1.0, self.max_norm / max(filtered_norm, 1e-12))
        filtered = filtered.to(current.dtype)
        if not bool(torch.isfinite(filtered).all()):
            raise FloatingPointError("Filtered gradient cannot be represented in parameter dtype")
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters:
                size = parameter.numel()
                if parameter.grad is not None:
                    parameter.grad.copy_(filtered[offset:offset+size].reshape_as(parameter))
                offset += size
        return dict(raw_gradient_norm=raw_norm, filtered_gradient_norm=filtered_norm,
                    applied_gradient_norm=float(filtered.double().norm()), high_frequency_fraction=high_fraction,
                    spike=raw_norm > self.spike_threshold, filtered_spike=filtered_norm > self.spike_threshold,
                    clipped=self.method != "none" and filtered_norm > self.max_norm)
