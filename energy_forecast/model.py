"""A normalized, differentiable joint energy over multivariate future trajectories.

This is conditional Gaussian-mixture density estimation, not a novel energy law.
The memory summarizer is separate; predicted futures are never committed as facts.
Spectral features rotate the PAST TIME axis without adding information or peeking.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ForecastConfig:
    history: int = 16
    variables: int = 8
    horizons: tuple[int, ...] = (1, 4, 16, 64)
    width: int = 96
    components: int = 3
    correlated: bool = True
    spectral: bool = False
    min_scale: float = .05
    max_scale: float = 3.0

    def validate(self):
        if any(type(x) is not int or x < 1 for x in (self.history,self.variables,self.width,self.components)):
            raise ValueError('Positive integer dimensions required')
        if not self.horizons or any(type(h) is not int or h < 1 for h in self.horizons):
            raise ValueError('Horizons must be positive integers')
        if tuple(sorted(set(self.horizons))) != tuple(self.horizons):
            raise ValueError('Horizons must be strictly increasing')
        if not 0 < self.min_scale < self.max_scale or not math.isfinite(self.max_scale):
            raise ValueError('Finite positive scale bounds required')


def real_fourier_basis(n: int, dtype=torch.float32) -> Tensor:
    """N orthonormal real Fourier coordinates, only along observed time."""
    t=torch.arange(n,dtype=dtype)
    rows=[torch.ones(n,dtype=dtype)/math.sqrt(n)]
    for k in range(1,(n+1)//2):
        rows += [math.sqrt(2/n)*torch.cos(2*math.pi*k*t/n), math.sqrt(2/n)*torch.sin(2*math.pi*k*t/n)]
    if n % 2 == 0:
        rows.append(torch.cos(math.pi*t)/math.sqrt(n))
    return torch.stack(rows)

@dataclass
class EnergyLandscape:
    """Frozen-for-this-read conditional function, not an updated memory state.

means [B,K,D], scales [B,K,D], correlation_cholesky [K,D,D].
The normalizer is analytic: energy is EXACT -log probability density.
"""
    logits: Tensor
    means: Tensor
    scales: Tensor
    correlation_cholesky: Tensor

    @property
    def mean(self):
        return (self.logits.softmax(-1)[...,None]*self.means).sum(1)

    def energy(self, candidate: Tensor) -> Tensor:
        if candidate.shape != self.means[:,0].shape:
            raise ValueError('Candidate must be [batch, horizon*variables]')
        z=(candidate[:,None,:]-self.means)/self.scales
        sol=torch.linalg.solve_triangular(self.correlation_cholesky[None], z[...,None], upper=False)[...,0]
        logdet=self.scales.log().sum(-1)+self.correlation_cholesky.diagonal(dim1=-2,dim2=-1).log().sum(-1)[None]
        component=-.5*(sol.square().sum(-1)+candidate.shape[-1]*math.log(2*math.pi))-logdet
        return -torch.logsumexp(self.logits.log_softmax(-1)+component,dim=-1)

    def sample(self, count: int, *, generator=None) -> Tensor:
        """[B,count,D] samples; every horizon shares the same component draw."""
        if type(count) is not int or count<1:raise ValueError('Positive sample count required')
        b,k,d=self.means.shape
        indices=torch.multinomial(self.logits.softmax(-1),count,replacement=True,generator=generator)
        batch=torch.arange(b,device=self.means.device)[:,None]
        eps=torch.randn(b,count,d,device=self.means.device,dtype=self.means.dtype,generator=generator)
        # Avoid allocating [B,count,D,D] matrices for every sampled trajectory.
        noise=torch.zeros_like(eps)
        for component in range(k):
            transformed=eps@self.correlation_cholesky[component].T
            noise=torch.where((indices==component)[...,None],transformed,noise)
        return self.means[batch,indices]+self.scales[batch,indices]*noise

    def detached(self):
        return EnergyLandscape(*(x.detach() for x in (self.logits,self.means,self.scales,self.correlation_cholesky)))

    def refine(self, initial: Tensor, *, steps: int=8, learning_rate: float=.2, backtracks: int=12):
        """Deterministic local MAP search with per-example original-energy acceptance.

Not sampling, not a global-minimum certificate, not permission to skip tokens.
The landscape is held fixed throughout; no parameter gradients are accumulated.
"""
        if steps<0 or backtracks<1 or not math.isfinite(learning_rate) or learning_rate<=0:
            raise ValueError('Invalid refinement budget')
        fixed=self.detached()
        z=initial.detach().clone()
        if not torch.isfinite(z).all():raise ValueError('Nonfinite initial candidate')
        with torch.no_grad():
            baseline=fixed.energy(z)
        energies=[baseline]
        for _ in range(steps):
            with torch.enable_grad():
                z=z.detach().requires_grad_(True)
                value=fixed.energy(z)
                grad,=torch.autograd.grad(value.sum(),z)
            with torch.no_grad():
                grad=grad/grad.norm(dim=-1,keepdim=True).clamp_min(1.)
                current=z.detach(); best=current.clone(); best_e=value.detach().clone()
                rate=learning_rate
                for _ in range(backtracks):
                    proposal=current-rate*grad
                    e=fixed.energy(proposal)
                    accept=torch.isfinite(e)&(e<best_e)
                    best=torch.where(accept[:,None],proposal,best)
                    best_e=torch.where(accept,e,best_e)
                    rate*=.5
                z=best
                energies.append(best_e)
        return z.detach(),torch.stack(energies,1)

class ForecastEnergy(nn.Module):
    """Predict parameters of a future energy from a bounded past-only history."""
    def __init__(self, config: ForecastConfig):
        super().__init__();config.validate();self.config=config
        c=config;self.d=len(c.horizons)*c.variables
        basis=real_fourier_basis(c.history) if c.spectral else torch.eye(c.history)
        self.register_buffer('time_basis',basis)
        self.encoder=nn.Sequential(nn.Linear(c.history*c.variables,c.width),nn.SiLU(),nn.Linear(c.width,c.width),nn.SiLU())
        self.output=nn.Linear(c.width,c.components*(2*self.d+1))
        self.register_buffer('eye',torch.eye(self.d))
        if c.correlated:self.raw_correlation=nn.Parameter(torch.zeros(c.components,self.d,self.d))
        else:self.register_buffer('raw_correlation',torch.zeros(c.components,self.d,self.d))
        nn.init.normal_(self.output.weight,std=.01)
        nn.init.normal_(self.output.bias,std=.025)
        # Small heterogeneous component offsets break exact mixture symmetry.
        with torch.no_grad():
            self.output.bias.view(c.components,-1)[:,self.d:2*self.d].fill_(-.8)

    def forward(self, history: Tensor) -> EnergyLandscape:
        c=self.config
        if history.ndim!=3 or history.shape[1:]!=(c.history,c.variables):
            raise ValueError('History must be [batch, configured past steps, variables]')
        if not history.is_floating_point() or not torch.isfinite(history).all():
            raise ValueError('History must be finite floating point')
        feature=torch.einsum('kt,btv->bkv',self.time_basis.to(history.dtype),history)
        raw=self.output(self.encoder(feature.flatten(1))).view(-1,c.components,2*self.d+1)
        # Persistence is a context-only initialization, not a future label.
        anchor=history[:,-1,:].repeat(1,len(c.horizons))[:,None,:]
        means=anchor+raw[:,:,:self.d]
        scales=c.min_scale+(c.max_scale-c.min_scale)*raw[:,:,self.d:2*self.d].sigmoid()
        # Correlated hypotheses couple future times AND variable channels.
        lower=torch.tril(self.raw_correlation,-1)+self.eye
        lower=lower/lower.square().sum(-1,keepdim=True).sqrt()
        return EnergyLandscape(raw[:,:,-1],means,scales,lower)

    def loss(self, history: Tensor, future: Tensor):
        if future.shape!=(history.shape[0],len(self.config.horizons),self.config.variables):
            raise ValueError('Target horizon/channel shape differs')
        if not torch.isfinite(future).all():raise ValueError('Nonfinite target')
        return self(history).energy(future.flatten(1)).mean()/self.d
