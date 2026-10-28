"""Experimental PC-ALM at native-LM block boundaries.

This cuts credit propagation between blocks, not inside a block's causal
mixer. Activity/dual iterations are detached; each block uses ordinary local
autograd. Inference stays feedforward, and BP remains the default trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass
class LocalLoss:
    loss: torch.Tensor
    residual_rms: float
    activity_steps: int


def pcalm_loss(model, tokens, targets, *, memory=None, reset=None, steps=8,
               state_lr=.02, dual_lr=.1, rho=1., check_budget=None):
    if type(steps) is not int or not 1 <= steps <= 256:
        raise ValueError('PC-ALM requires 1–256 activity steps')
    if any(not math.isfinite(v) or v <= 0 for v in (state_lr, dual_lr, rho)):
        raise ValueError('Positive finite PC-ALM rates and penalty required')
    if tokens.ndim != 2 or not tokens.numel() or tokens.dtype != torch.long:
        raise ValueError('Integer [batch,time] tokens required')
    if targets.shape != tokens.shape or targets.dtype != torch.long or targets.device != tokens.device:
        raise ValueError('Targets must match token shape and device')
    if reset is not None and (reset.shape != tokens.shape or reset.dtype != torch.bool or reset.device != tokens.device):
        raise ValueError('Invalid reset mask')
    if memory is not None:
        if not model.config.memory_dim:
            raise ValueError('Model has no fact readers')
        memory.validate(tokens.shape[0], model.config.memory_dim, model.config.memory_capacity, tokens.device)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Experimental PC-ALM requires float32 model parameters')

    def check():
        if check_budget is not None:
            check_budget()

    def predictions(states):
        return [model.embedding(tokens), *(
            block(states[i], None, reset, memory)[0] for i, block in enumerate(model.blocks))]

    with torch.autocast(device_type=tokens.device.type, enabled=False):
        check()
        with torch.no_grad():
            states = [model.embedding(tokens)]
            for block in model.blocks:
                states.append(block(states[-1], None, reset, memory)[0])
        duals = [torch.zeros_like(s) for s in states]
        for step in range(steps):
            check()
            states = [s.detach().requires_grad_() for s in states]
            residuals = [s - p for s, p in zip(states, predictions(states))]
            logits = model.head(model.norm(states[-1]))
            energy = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='sum')
            energy = energy + sum((dual * r + .5 * rho * r.square()).sum() for dual, r in zip(duals, residuals))
            gradients = torch.autograd.grad(energy, states)
            with torch.no_grad():
                states = [s - state_lr * g for s, g in zip(states, gradients)]
                if not all(torch.isfinite(s).all() for s in states):
                    raise FloatingPointError('PC-ALM activities diverged')
                if step < steps - 1:
                    residuals = [s - p for s, p in zip(states, predictions(states))]
                    duals = [d + dual_lr * r for d, r in zip(duals, residuals)]
        check()
        states = [s.detach() for s in states]
        residuals = [s - p for s, p in zip(states, predictions(states))]
        task_loss = F.cross_entropy(model.head(model.norm(states[-1])).flatten(0, 1), targets.flatten())
        surrogate = task_loss + sum((d * r + .5 * rho * r.square()).sum()
                                    for d, r in zip(duals, residuals)) / targets.numel()
        if not torch.isfinite(surrogate):
            raise FloatingPointError('PC-ALM parameter objective diverged')
        # Report task CE while differentiating the local augmented objective.
        loss = surrogate + (task_loss.detach() - surrogate.detach())
        rms = float(torch.stack([r.detach().square().mean() for r in residuals]).mean().sqrt())
        return LocalLoss(loss, rms, steps)
