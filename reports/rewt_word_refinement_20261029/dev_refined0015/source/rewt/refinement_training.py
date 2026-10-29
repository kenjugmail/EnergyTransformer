"""Prediction supervision across a causal energy-minimization trajectory."""
import math

import torch
from torch.nn import functional as F


def refinement_loss(model, tokens, targets, *, steps=(0, 2, 4, 8), monotonic_weight=.1):
    """Train shared refinement weights using next-token labels on train only.

    Mean CE supervises every requested depth. A one-sided per-token penalty
    discourages later states from worsening CE relative to the detached earlier
    state. Detachment prevents reducing the penalty by making an earlier state
    worse. This is a training incentive, not a held-out monotonicity guarantee.
    Inference needs only the causal tokens and a validation-selected step count.
    """
    steps = tuple(steps)
    if (not steps or any(type(s) is not int or s < 0 for s in steps)
            or list(steps) != sorted(set(steps))):
        raise ValueError('Refinement steps must be distinct increasing nonnegative integers')
    if not math.isfinite(monotonic_weight) or monotonic_weight < 0:
        raise ValueError('monotonic_weight must be finite and nonnegative')
    if tokens.shape != targets.shape or targets.dtype != torch.long:
        raise ValueError('Aligned int64 next-token targets required')
    states = model.refinement_states(tokens, steps[-1])
    losses = [F.cross_entropy(F.linear(states[s], model.embedding.weight).flatten(0, 1),
                              targets.flatten(), reduction='none') for s in steps]
    supervised = torch.stack([loss.mean() for loss in losses]).mean()
    penalty = (torch.stack([(new - old.detach()).relu().mean()
                            for old, new in zip(losses, losses[1:])]).mean()
               if len(losses) > 1 else supervised.new_zeros(()))
    objective = supervised + monotonic_weight * penalty
    return objective, dict(cross_entropy=supervised.detach(), monotonic_penalty=penalty.detach(),
                          depth_nll={str(s): loss.mean().detach() for s, loss in zip(steps, losses)})
