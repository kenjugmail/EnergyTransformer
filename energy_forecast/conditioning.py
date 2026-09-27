"""Exact observed-coordinate conditioning of a frozen Gaussian-mixture energy.

This is classical Gaussian conditioning, not new learning or future information.
Use only measurements available at the current time. Conditional draws are
hypotheses, never observations. Dense batch-dependent Cholesky factors cost O(D^2).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import torch
from torch import Tensor
from .model import EnergyLandscape


def _indices(values: Sequence[int], dimension: int, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError('Coordinate indices must be a list or tuple of integers')
    idx = tuple(values)
    if (not idx and not allow_empty) or any(type(i) is not int or i < 0 or i >= dimension for i in idx):
        raise ValueError('Invalid coordinate indices')
    if len(set(idx)) != len(idx):
        raise ValueError('Repeated coordinates would double-count evidence')
    return idx


def _row_factor(rows: Tensor) -> Tensor:
    """Lower factor of rows @ rows.T, without forming that product.

    With rows.T = Q R, rows @ rows.T = R.T R. Reduced QR keeps
    autograd available; its columns are independent for any subset of a
    nonsingular Cholesky factor's rows. Normalize R's diagonal signs.
    This avoids squaring the condition number but cannot cure arbitrary
    ill-conditioning. Fail explicitly on numerical rank loss; never add jitter.
    """
    _, upper = torch.linalg.qr(rows.transpose(-1, -2), mode='reduced')
    diagonal = upper.diagonal(dim1=-2, dim2=-1)
    if not torch.isfinite(upper).all() or (diagonal == 0).any():
        raise FloatingPointError('Numerically rank-deficient square-root factor; use higher precision')
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return (upper * signs[..., :, None]).transpose(-1, -2).contiguous()


@dataclass(frozen=True)
class GaussianMixture:
    """A normalized joint function; inputs share dtype and device.

    logits [B,K], means [B,K,D], scale_tril [B,K,D,D]. No diagonal jitter or
    independence approximation is silently introduced. For ill-conditioned
    inputs, use float64 and investigate failures rather than hide them.
    """
    logits: Tensor
    means: Tensor
    scale_tril: Tensor

    def __post_init__(self):
        if any(not isinstance(t, Tensor) for t in (self.logits, self.means, self.scale_tril)):
            raise TypeError('Mixture coefficients must be tensors')
        if self.means.ndim != 3 or min(self.means.shape) < 1:
            raise ValueError('Means must be nonempty [B,K,D]')
        b, k, d = self.means.shape
        if self.logits.shape != (b, k) or self.scale_tril.shape != (b, k, d, d):
            raise ValueError('Mixture shapes disagree')
        ts = (self.logits, self.means, self.scale_tril)
        if any(t.dtype != self.means.dtype or t.device != self.means.device for t in ts):
            raise ValueError('Mixture dtype/device mismatch')
        if self.means.dtype not in (torch.float32, torch.float64):
            raise TypeError('Mixture arithmetic requires float32 or float64')
        if any(not torch.isfinite(t).all() for t in ts):
            raise ValueError('Nonfinite mixture coefficients')
        if (self.scale_tril.diagonal(dim1=-2, dim2=-1) <= 0).any() or torch.count_nonzero(torch.triu(self.scale_tril, 1)):
            raise ValueError('Cholesky factors must be lower triangular with positive diagonal')

    @classmethod
    def from_landscape(cls, landscape: EnergyLandscape, *, dtype=torch.float64):
        # diag(scale) @ L; preserving full cross-horizon correlation.
        scale = landscape.scales.to(dtype)
        chol = scale[..., None] * landscape.correlation_cholesky.to(dtype)[None]
        return cls(landscape.logits.to(dtype), landscape.means.to(dtype), chol)

    @property
    def dimension(self) -> int:
        return self.means.shape[-1]

    @property
    def mean(self) -> Tensor:
        return (self.logits.softmax(-1)[..., None] * self.means).sum(1)

    @property
    def component_covariance(self) -> Tensor:
        return self.scale_tril @ self.scale_tril.transpose(-1, -2)

    @property
    def variance(self) -> Tensor:
        weights = self.logits.softmax(-1)[..., None]
        within = self.scale_tril.square().sum(-1)
        return (weights * (within + (self.means - self.mean[:, None]).square())).sum(1)

    def energy(self, candidate: Tensor) -> Tensor:
        if candidate.shape != self.means[:, 0].shape or candidate.device != self.means.device:
            raise ValueError('Candidate must be [B,D] on the function device')
        if not candidate.is_floating_point() or not torch.isfinite(candidate).all():
            raise ValueError('Candidate must be finite floating point')
        z = (candidate.to(self.means.dtype)[:, None] - self.means)[..., None]
        whitened = torch.linalg.solve_triangular(self.scale_tril, z, upper=False)[..., 0]
        logdet = self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        logp = -.5 * (whitened.square().sum(-1) + self.dimension * math.log(2 * math.pi)) - logdet
        return -torch.logsumexp(self.logits.log_softmax(-1) + logp, dim=-1)

    def marginal(self, indices: Sequence[int]):
        """Exact ordered marginal; leading rows reuse the triangular factor.

        General subsets use QR on selected factor rows, not covariance/Cholesky.
        Requested coordinate ordering is preserved. Returned factors own storage.
        """
        keep = _indices(indices, self.dimension)
        idx = torch.tensor(keep, device=self.means.device)
        if keep == tuple(range(len(keep))):
            factor = self.scale_tril[..., :len(keep), :len(keep)].clone()
        else:
            factor = _row_factor(self.scale_tril.index_select(-2, idx))
        return GaussianMixture(self.logits, self.means.index_select(-1, idx), factor)

    def condition(self, indices: Sequence[int], observed: Tensor):
        """Return p(Y_remaining | Y_indices=observed) and log p(observed).

        Remaining indices are in ascending ORIGINAL-coordinate order. Reordered
        leading evidence is canonicalized before the fast prefix path. Other
        subsets permute factor rows then use QR to obtain an exact square root
        in that order. No covariance subtraction, jitter, value-dependent path
        selection or independence approximation is used.
        """
        given = _indices(indices, self.dimension, allow_empty=True)
        b, _, d = self.means.shape
        if not isinstance(observed, Tensor) or observed.shape != (b, len(given)):
            raise ValueError('Observation must be [B,len(indices)]')
        if observed.device != self.means.device or not observed.is_floating_point():
            raise ValueError('Observation dtype/device mismatch')
        if not torch.isfinite(observed).all():
            raise ValueError('Finite observed values required')
        if len(given) == d:
            raise ValueError('All coordinates are observed; no forecast remains')
        remaining = tuple(i for i in range(d) if i not in given)
        if not given:
            return ConditioningResult(self, remaining, self.means.new_zeros(b))
        # Dispatch depends on coordinate structure, never on evidence values.
        if tuple(sorted(given)) == tuple(range(len(given))):
            if given != tuple(range(len(given))):
                order = torch.tensor([given.index(i) for i in range(len(given))], device=observed.device)
                observed = observed.index_select(-1, order)
            return self.condition_prefix(len(given), observed)
        order = torch.tensor(given + remaining, device=self.means.device)
        permuted = GaussianMixture(
            self.logits, self.means.index_select(-1, order),
            _row_factor(self.scale_tril.index_select(-2, order)))
        result = permuted.condition_prefix(len(given), observed)
        return ConditioningResult(result.distribution, remaining, result.log_evidence)

    def condition_prefix(self, count: int, observed: Tensor):
        """Exact leading-coordinate condition without covariance reconstruction.

        For L=[[L_oo,0],[L_ro,L_rr]], observing O fixes eps_o=L_oo^-1
        (y_o-mu_o). The remaining mean is mu_r+L_ro eps_o and its factor
        is exactly L_rr. This avoids the cancellation in a Schur subtraction.
        A non-leading or reordered subset must use condition(). No jitter,
        correlation approximation, or learned weights are introduced.
        """
        b, k, d = self.means.shape
        if type(count) is not int or not 0 <= count < d:
            raise ValueError('Prefix count must leave at least one coordinate')
        if not isinstance(observed, Tensor) or observed.shape != (b, count):
            raise ValueError('Observation must be [B,count]')
        if observed.device != self.means.device or not observed.is_floating_point():
            raise ValueError('Observation dtype/device mismatch')
        if not torch.isfinite(observed).all():
            raise ValueError('Nonfinite observation')
        if count == 0:
            return ConditioningResult(self, tuple(range(d)), self.means.new_zeros(b))
        lo = self.scale_tril[..., :count, :count]
        lr = self.scale_tril[..., count:, :count]
        observed = observed.to(self.means.dtype)
        if not torch.isfinite(observed).all():
            raise ValueError('Observation overflows the function dtype')
        delta = observed[:, None] - self.means[..., :count]
        eps = torch.linalg.solve_triangular(lo, delta[..., None], upper=False)
        mean = self.means[..., count:] + (lr @ eps)[..., 0]
        log_obs = -.5 * (eps[..., 0].square().sum(-1) + count * math.log(2 * math.pi))
        log_obs = log_obs - lo.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        weight = self.logits.log_softmax(-1) + log_obs
        evidence = torch.logsumexp(weight, -1)
        posterior = GaussianMixture(weight - evidence[:, None], mean,
                                    self.scale_tril[..., count:, count:].clone())
        return ConditioningResult(posterior, tuple(range(count, d)), evidence)

    def sample(self, count: int, *, generator=None) -> Tensor:
        if type(count) is not int or count < 1:
            raise ValueError('Positive integer sample count required')
        b, k, d = self.means.shape
        indices = torch.multinomial(self.logits.softmax(-1), count, replacement=True, generator=generator)
        eps = torch.randn(b, count, d, dtype=self.means.dtype, device=self.means.device, generator=generator)
        samples = torch.zeros_like(eps)
        # Avoid one D x D factor per draw. Components stay shared across horizons.
        for j in range(k):
            draw = torch.einsum('bnd,bed->bne', eps, self.scale_tril[:, j]) + self.means[:, j, None]
            samples = torch.where((indices == j)[..., None], draw, samples)
        return samples

    def quantile(self, q: float, *, iterations: int = 56) -> Tensor:
        if not 0 < q < 1 or type(iterations) is not int or iterations < 1:
            raise ValueError('Invalid marginal quantile request')
        sigma = self.scale_tril.square().sum(-1).sqrt()
        # Bracket using the exact component quantiles. Their min/max bracket the
        # mixture quantile at ANY representable probability, not only +/-12 SD.
        normal_q = torch.special.ndtri(torch.tensor(q, dtype=torch.float64, device=self.means.device)).to(self.means.dtype)
        lo = (self.means + sigma * normal_q).amin(1)
        hi = (self.means + sigma * normal_q).amax(1)
        weights = self.logits.softmax(-1)[..., None]
        for _ in range(iterations):
            mid = (lo + hi) * .5
            z = (mid[:, None] - self.means) / sigma
            cdf = (weights * (.5 * torch.erfc(-z / math.sqrt(2)))).sum(1)
            lo = torch.where(cdf < q, mid, lo)
            hi = torch.where(cdf >= q, mid, hi)
        return (lo + hi) * .5

    def cloned(self):
        return GaussianMixture(*(x.detach().clone() for x in (self.logits, self.means, self.scale_tril)))

    def payload(self):
        return {name: getattr(self, name).detach().clone() for name in ('logits', 'means', 'scale_tril')}

    @classmethod
    def restore(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {'logits', 'means', 'scale_tril'}:
            raise ValueError('Invalid mixture snapshot schema')
        if any(not isinstance(t, Tensor) for t in payload.values()):
            raise ValueError('Invalid mixture tensors')
        return cls(**payload).cloned()


@dataclass(frozen=True)
class ConditioningResult:
    distribution: GaussianMixture
    remaining_indices: tuple[int, ...]
    log_evidence: Tensor


def blend(prior: GaussianMixture, posterior: GaussianMixture, weight: float) -> GaussianMixture:
    """Normalized conservative forecast pool (1-weight)*prior + weight*posterior.

    This is NOT the exact Bayesian posterior except at weight=1. No covariance
    averaging approximation: both complete correlated component sets are retained.
    """
    if type(weight) not in (float, int) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Blend weight must be a finite number in [0,1]')
    if prior.means.shape[0] != posterior.means.shape[0] or prior.dimension != posterior.dimension:
        raise ValueError('Incompatible forecast pools')
    if prior.means.device != posterior.means.device or prior.means.dtype != posterior.means.dtype:
        raise ValueError('Forecast pools have different dtype/device')
    if weight == 0:
        return prior
    if weight == 1:
        return posterior
    logits = torch.cat((prior.logits.log_softmax(-1) + math.log1p(-weight),
                        posterior.logits.log_softmax(-1) + math.log(weight)), 1)
    return GaussianMixture(logits, torch.cat((prior.means, posterior.means), 1),
                           torch.cat((prior.scale_tril, posterior.scale_tril), 1))
