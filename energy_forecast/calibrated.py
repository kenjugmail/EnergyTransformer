"""Calibrated function-valued forecasts, without modifying observed LM state.

Point mean and density shape have separate fitted roles. Recentring and positive
scaling preserve normalization and cross-horizon mixture membership. Calibration
is empirical; it is not a coverage guarantee or permission to skip tokens.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn, Tensor
from .model import ForecastConfig, ForecastEnergy, EnergyLandscape


class HorizonAnchor(nn.Module):
    """A bounded-history linear predictor with a fixed gain at each horizon.

The weights, constant and gains must be fitted without final test targets. A zero
horizon gain intentionally returns the training target mean, not persistence.
"""
    def __init__(self, history: int, variables: int, horizons: tuple[int, ...]):
        super().__init__()
        if min(history, variables) < 1 or not horizons:
            raise ValueError('Positive dimensions required')
        self.history, self.variables, self.horizons = history, variables, tuple(horizons)
        d = len(horizons) * variables
        self.register_buffer('weight', torch.zeros(history * variables + 1, d))
        self.register_buffer('constant', torch.zeros(d))
        self.register_buffer('gain', torch.zeros(len(horizons)))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.history, self.variables):
            raise ValueError('History shape differs from anchor')
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Finite floating point history required')
        features = torch.cat((x.flatten(1), x.new_ones((len(x), 1))), -1)
        raw = features @ self.weight
        return self.constant + self.gain.repeat_interleave(self.variables) * (raw - self.constant)


def affine_landscape(landscape: EnergyLandscape, mean: Tensor,
                     temperature: Tensor) -> EnergyLandscape:
    """Apply Y_new = mean + diag(temperature) (Y_old - E[Y_old]).

Both component offsets and within-component scales change. Scaling only the
within-component variance would be a different calibration model.
"""
    if mean.shape != landscape.mean.shape or not torch.isfinite(mean).all():
        raise ValueError('Invalid requested mean')
    if temperature.ndim != 1 or len(temperature) != mean.shape[-1]:
        raise ValueError('Temperature must have one entry per future coordinate')
    if not torch.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError('Positive finite temperature required')
    offsets = landscape.means - landscape.mean[:, None, :]
    return EnergyLandscape(landscape.logits, mean[:, None, :] + offsets * temperature,
                           landscape.scales * temperature, landscape.correlation_cholesky)


class AnchoredForecast(ForecastEnergy):
    """Train a joint residual density while fixing its conditional mean.

Overrides used by training are out-of-fold mean predictions. Ordinary forward()
accepts only the observed history; it never accepts ground-truth future values.
"""
    def __init__(self, config: ForecastConfig):
        super().__init__(config)
        self.anchor = HorizonAnchor(config.history, config.variables, config.horizons)
        self.register_buffer('temperature', torch.ones(len(config.horizons)))

    def landscape(self, history: Tensor, crossfit_mean: Tensor | None = None):
        base = super().forward(history)
        center = self.anchor(history) if crossfit_mean is None else crossfit_mean
        return affine_landscape(base, center, self.temperature.repeat_interleave(self.config.variables))

    def forward(self, history: Tensor):
        return self.landscape(history)


class RecenteredForecast(nn.Module):
    """No-neural-retraining repair of a previously fitted density head."""
    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config, self.d = config, len(config.horizons) * config.variables
        self.base = ForecastEnergy(config)
        self.anchor = HorizonAnchor(config.history, config.variables, config.horizons)
        self.register_buffer('temperature', torch.ones(len(config.horizons)))

    def forward(self, history: Tensor):
        return affine_landscape(self.base(history), self.anchor(history),
                                self.temperature.repeat_interleave(self.config.variables))


def load_calibrated(path: str | Path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get('format') != 'calibrated_trajectory_v2':
        raise ValueError('Not a calibrated forecasting checkpoint')
    c = dict(saved['config']); c['horizons'] = tuple(c['horizons'])
    config = ForecastConfig(**c); config.validate()
    if saved['variant'] not in ('anchored', 'recentered'):
        raise ValueError('Unknown calibrated variant')
    model = (AnchoredForecast if saved['variant']=='anchored' else RecenteredForecast)(config).to(device)
    model.load_state_dict(saved['weights'], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
        raise ValueError('Nonfinite checkpoint coefficients')
    if (model.temperature <= 0).any() or (model.anchor.gain < 0).any() or (model.anchor.gain > 1).any():
        raise ValueError('Invalid calibration parameters')
    model.eval()
    return model, saved
