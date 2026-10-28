"""Exact block elimination for the core preconditioner.

The output block is diagonal on the raw domain, or projected diagonal on the
centered domain. Eliminate it analytically and factor only 3*width coordinates.
This is equivalent to the dense gauge-fixed solve, including parameter gradients.
"""
import math
import torch
from .kernels import core_quadratic_action
from .advanced_kernels import spectral_upper


def structured_lift(model, frequency, precision):
    if model.config.support:
        raise ValueError('structured elimination supports the core quadratic only')
    cut = model.blocks['output'].start
    identity = torch.eye(cut, dtype=frequency.dtype, device=frequency.device)
    directions = torch.cat((identity, identity.new_zeros(cut, 256)), -1)
    rows = core_quadratic_action(model, directions, precision)
    hll, hlo = rows[:, :cut], rows[:, cut:]
    diagonal = precision[cut:] + model.config.decoder_weight
    inverse = diagonal.reciprocal()

    def output_inverse(rhs):
        result = rhs * inverse
        if model.config.output_geometry == 'centered':
            result = result - inverse * (result.sum(-1, keepdim=True) / inverse.sum())
        return result

    schur = hll - output_inverse(hlo) @ hlo.T
    schur = (schur + schur.T) / 2
    rhs = frequency[:, :cut] - output_inverse(frequency[:, cut:]) @ hlo.T
    latent = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(schur)).T
    output = output_inverse(frequency[:, cut:] - latent @ hlo)
    return model.canonical_state(torch.cat((latent, output), -1))


def prepare_structured(model, frequency, precision):
    lift = structured_lift(model, frequency, precision)
    gram = frequency @ lift.T
    gram = (gram + gram.T) / 2
    upper = 1 + .5 / math.sqrt(model.config.modes) * spectral_upper(gram)
    step = torch.minimum(upper.new_tensor(model.config.step_size), 1.9 / upper)
    return dict(lift=lift, gram=gram, step=step, upper=upper)
