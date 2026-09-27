"""Covariance-based diagnostic reference; NOT the deployed conditioning path.

Retained so historical benchmark commands do not accidentally compare the new
fast path with itself after reordered prefixes gained their own shortcut.
Ill-conditioned inputs are expected to expose covariance-rounding failures here.
"""
import math
import torch
from .conditioning import GaussianMixture, ConditioningResult


def dense_condition(function, given, observed):
    remaining = tuple(i for i in range(function.dimension) if i not in given)
    cov = function.component_covariance
    oo = cov[..., given, :][..., given]
    ro = cov[..., remaining, :][..., given]
    rr = cov[..., remaining, :][..., remaining]
    lo = torch.linalg.cholesky(oo)
    delta = observed[:, None] - function.means[..., given]
    solved = torch.cholesky_solve(delta[..., None], lo)
    mean = function.means[..., remaining] + (ro @ solved)[..., 0]
    cross = torch.linalg.solve_triangular(lo, ro.transpose(-1, -2), upper=False)
    conditional_cov = rr - cross.transpose(-1, -2) @ cross
    factor = torch.linalg.cholesky((conditional_cov + conditional_cov.transpose(-1, -2)) * .5)
    eps = torch.linalg.solve_triangular(lo, delta[..., None], upper=False)[..., 0]
    logp = -.5 * (eps.square().sum(-1) + len(given) * math.log(2 * math.pi))
    logp -= lo.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    weight = function.logits.log_softmax(-1) + logp
    evidence = torch.logsumexp(weight, -1)
    posterior = GaussianMixture(weight - evidence[:, None], mean, factor)
    return ConditioningResult(posterior, remaining, evidence)


def dense_marginal(function, indices):
    cov = function.component_covariance[..., indices, :][..., indices]
    return GaussianMixture(function.logits, function.means[..., indices], torch.linalg.cholesky(cov))
