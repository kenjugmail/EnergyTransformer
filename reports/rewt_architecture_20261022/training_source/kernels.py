"""Mode-space execution of the two-round, anchor-relative core energy.

For displacement z, E(z) = z H z^T / 2 + a cos(z F^T + phase).
Starting at z=0, two gradient steps lie in span(F, F H), even when
backtracking rejects a step. H and F are fixed throughout a forward call.
The small Gram matrices below retain autograd; no learned tensor is detached.
This is an algebraic execution specialization, not a new learned architecture.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def core_quadratic_action(model, direction: Tensor, precision: Tensor) -> Tensor:
    """Apply the centered core Hessian without materializing a D by D matrix."""
    c = model.config
    direction = model.canonical_state(direction)
    recurrent, attention, feedforward, output = (
        direction[..., model.blocks[name]]
        for name in ('recurrent', 'attention', 'feedforward', 'output'))
    residual = output - model.output_coordinates(F.linear(
        (recurrent + attention + feedforward) / 3, model.decoder.weight))
    latent = c.decoder_weight * (residual @ model.decoder.weight) / 3
    agreement = c.attention_weight * (recurrent - attention)
    parts = (
        precision[model.blocks['recurrent']] * recurrent + agreement - latent,
        precision[model.blocks['attention']] * attention - agreement - latent,
        precision[model.blocks['feedforward']] * feedforward - latent,
        precision[model.blocks['output']] * output + c.decoder_weight * residual,
    )
    return model.canonical_state(torch.cat(parts, -1))


def prepare_krylov2(model, frequency: Tensor, precision: Tensor):
    """Once-per-forward preparation, including gradients through H and F."""
    hf = core_quadratic_action(model, frequency, precision)
    hhf = core_quadratic_action(model, hf, precision)
    grams = torch.stack((frequency @ frequency.T, frequency @ hf.T,
                         hf @ hf.T, hf @ hhf.T))
    steps = frequency.new_tensor([model.config.step_size * 0.5 ** k
                                  for k in range(model.config.backtracks)])
    return hf, grams, steps


def _choose_step(energies: Tensor, before: Tensor, steps: Tensor):
    # Steps decrease strictly. The largest passing step is the first passing
    # attempt in the reference loop. Zero implements KEEP when all fail.
    valid = torch.isfinite(energies) & (energies < before[..., None])
    return torch.where(valid, steps, 0.).amax(-1, keepdim=True)


def krylov2_refine(anchor: Tensor, amplitudes: Tensor, phase: Tensor,
                   frequency: Tensor, hf: Tensor, grams: Tensor, steps: Tensor,
                   *, output_start: int, centered: bool):
    """Two exact-arithmetic gradient/backtracking rounds in M-mode coordinates.

    All comparisons use the complete energy. Candidate step energies are
    evaluated together, without host synchronization or data-dependent loops.
    Reassociation changes floating-point rounding; this is not bitwise parity.
    Returns the joint state, three energies and two acceptance flags per row.
    """
    g0, g1, g2, g3 = grams.unbind(0)
    first_gradient = -amplitudes * phase.sin()
    projected = first_gradient @ g0
    curvature = ((first_gradient @ g1) * first_gradient).sum(-1, keepdim=True)
    before = (amplitudes * phase.cos()).sum(-1)
    candidates = (0.5 * steps.square() * curvature
                  + (amplitudes[..., None, :] *
                     (phase - steps[:, None] * projected[..., None, :]).cos()).sum(-1))
    alpha = _choose_step(candidates, before, steps)
    c = torch.where(alpha > 0, -alpha * first_gradient, torch.zeros_like(first_gradient))

    phase1 = c @ g0 + phase
    cg1, cg2 = c @ g1, c @ g2
    quadratic = 0.5 * (cg1 * c).sum(-1, keepdim=True)
    after_first = quadratic.squeeze(-1) + (amplitudes * phase1.cos()).sum(-1)
    wave_gradient = -amplitudes * phase1.sin()
    wg1 = wave_gradient @ g1
    projection = wave_gradient @ g0 + cg1
    slope = (cg1 * wave_gradient + cg2 * c).sum(-1, keepdim=True)
    curvature = (wg1 * wave_gradient + 2 * cg2 * wave_gradient
                 + (c @ g3) * c).sum(-1, keepdim=True)
    candidates = (quadratic - steps * slope + 0.5 * steps.square() * curvature
                  + (amplitudes[..., None, :] *
                     (phase1[..., None, :] - steps[:, None] * projection[..., None, :]).cos()).sum(-1))
    beta = _choose_step(candidates, after_first, steps)
    u, v = c - beta * wave_gradient, -beta * c
    result = anchor + u @ frequency + v @ hf
    # Match KEEP even if a rejected trial overflowed or had a nonfinite
    # gradient. In particular, multiplying a zero coefficient by NaN is NaN.
    result = torch.where((alpha > 0) | (beta > 0), result, anchor)
    if centered:
        logits = result[..., output_start:]
        result = torch.cat((result[..., :output_start], logits - logits.mean(-1, keepdim=True)), -1)

    # Energy at the selected beta; including beta=0 avoids an invalid gather.
    after_second = (quadratic - beta * slope + 0.5 * beta.square() * curvature).squeeze(-1)
    after_second = after_second + (amplitudes * (phase1 - beta * projection).cos()).sum(-1)
    energies = torch.stack((before, after_first, after_second), -1)
    accepted = torch.cat((alpha > 0, beta > 0), -1)
    return result, energies, accepted
