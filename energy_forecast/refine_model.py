"""Candidate-energy refinement on a frozen persistent-memory language model.

A small hypothesis test, NOT a reproduction of Gladstone et al.'s complete EBT.
Only candidate z is optimized during inference. The context and P/S are fixed.
Unrolled analytic gradients retain mixed/second-order parameter derivatives.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F

KINDS = ('feedforward', 'memory_ff', 'recurrent', 'energy_hidden', 'energy_memory', 'latent_ff', 'energy_full')

@dataclass(frozen=True)
class RefinementConfig:
    kind: str = 'energy_memory'
    width: int = 160
    memory_width: int = 64
    latent: int = 32
    steps: int = 4
    step_size: float = .25
    parameter_budget: int = 36128
    def validate(self):
        if self.kind not in KINDS: raise ValueError('Unknown refinement kind')
        for x in (self.width, self.memory_width, self.latent, self.steps, self.parameter_budget):
            if type(x) is not int or x < 1: raise ValueError('Positive integer dimensions required')
        if type(self.step_size) not in (float,int) or not math.isfinite(self.step_size) or not 0 < self.step_size <= 1:
            raise ValueError('Invalid step size')
    @property
    def uses_memory(self): return self.kind in ('memory_ff', 'recurrent', 'energy_memory', 'latent_ff', 'energy_full')
    @property
    def energy(self): return self.kind.startswith('energy_')


def compact_energy_read(state) -> Tensor:
    """Four fixed slot queries and three value projections per final-layer head.

Each head yields 12 projected reads plus four log query variances. This is a
lossy current-state read, not the full memory and not a predicted future.
The query/value bases are orthonormal DCT rows; no test-data fit is involved.
"""
    p, s = state[-1]
    if p.ndim != 3 or s.ndim != 4 or s.shape[:3] != p.shape:
        raise ValueError('Expected P[B,H,K] and S[B,H,K,V]')
    if p.shape[-1] < 4 or s.shape[-1] < 3:
        raise ValueError('At least four slots and three value coordinates required')
    if not torch.isfinite(p).all() or not torch.isfinite(s).all() or (p <= 0).any():
        raise ValueError('Finite information and positive precision required')
    def basis(n, rows):
        k = torch.arange(rows, dtype=p.dtype, device=p.device)[:, None]
        j = torch.arange(n, dtype=p.dtype, device=p.device)[None]
        b = torch.cos(math.pi * k * (j + .5) / n) * math.sqrt(2 / n)
        b[0] /= math.sqrt(2)
        return b
    q, v = basis(p.shape[-1], 4), basis(s.shape[-1], 3)
    mode = s / p[..., None]
    read = torch.einsum('rk,bhkv,uv->bhru', q, mode, v).flatten(-2)
    variance = torch.einsum('rk,bhk->bhr', q.square(), p.reciprocal())
    return torch.cat((read, variance.log()), -1).flatten(1)


class RefinementAdapter(nn.Module):
    def __init__(self, config: RefinementConfig):
        super().__init__(); config.validate(); self.config = config
        c = config; inputs = c.width + (c.memory_width if c.uses_memory else 0)
        self.register_buffer('memory_center', torch.zeros(c.memory_width))
        self.register_buffer('memory_scale', torch.ones(c.memory_width))
        if c.kind == 'energy_full':
            if c.latent != c.width: raise ValueError('Full energy requires latent=width')
            units=max(1,round((c.parameter_budget-1)/(2*(inputs+1)+c.latent)))
            self.context=nn.Linear(inputs,2*units)
            self.directions=nn.Parameter(torch.randn(units,c.latent)/math.sqrt(c.latent))
            self.amplitude=nn.Parameter(torch.tensor(0.))
            self.output=nn.Identity();self.units=units
        elif c.energy:
            units = max(1, round((c.parameter_budget-c.width*c.latent) /
                                (2*(inputs+1)+c.latent)))
            self.context = nn.Linear(inputs, 2*units)
            self.directions = nn.Parameter(torch.randn(units, c.latent)/math.sqrt(c.latent))
            self.output = nn.Linear(c.latent, c.width, bias=False)
            self.units = units
        elif c.kind == 'latent_ff':
            units=max(1,round((c.parameter_budget-c.width*c.latent-c.latent)/(inputs+c.latent+1)))
            self.first=nn.Linear(inputs,units)
            self.transition=nn.Linear(units,c.latent)
            self.output=nn.Linear(c.latent,c.width,bias=False)
        elif c.kind == 'recurrent':
            units = max(1, round((c.parameter_budget-c.width*c.latent-c.latent) /
                                (inputs+2*c.latent+1)))
            self.first = nn.Linear(inputs+c.latent, units)
            self.transition = nn.Linear(units, c.latent)
            self.output = nn.Linear(c.latent, c.width, bias=False)
        else:
            units = max(1, round((c.parameter_budget-c.width)/(inputs+c.width+1)))
            self.first = nn.Linear(inputs, units)
            self.output = nn.Linear(units, c.width)
        if isinstance(self.output, nn.Linear):
            nn.init.zeros_(self.output.weight)
            if self.output.bias is not None: nn.init.zeros_(self.output.bias)

    def context_input(self, h, memory):
        c = self.config
        if h.ndim != 2 or len(h)<1 or h.shape[-1] != c.width or memory.shape != (len(h), c.memory_width):
            raise ValueError('Hidden/memory shape mismatch')
        if not h.is_floating_point() or memory.dtype != h.dtype or memory.device != h.device:
            raise ValueError('Hidden/memory dtype/device mismatch')
        if not torch.isfinite(h).all() or not torch.isfinite(memory).all():
            raise ValueError('Nonfinite inputs')
        if not c.uses_memory: return h
        return torch.cat((h, (memory-self.memory_center)/self.memory_scale), -1)

    def landscape(self, context):
        if not self.config.energy: raise ValueError('No scalar energy in this control')
        a, b = self.context(context).chunk(2, -1)
        return 2*a.sigmoid(), b

    def energy(self, z, landscape):
        """Coercive nonquadratic energy, centered so E(0|context)=0.

E=.5||z||^2 - 4/sqrt(J) sum a_j[softplus(u_j.z+b_j)-softplus(b_j)].
Bounded a and finite weights imply at most linear negative growth; the quadratic
therefore dominates at infinity. The partition function is not evaluated.
"""
        a, b = landscape
        gain=self.amplitude if self.config.kind=='energy_full' else 1.
        return .5*z.square().sum(-1) - gain*(4/math.sqrt(self.units))* (
            a*(F.softplus(F.linear(z, self.directions)+b)-F.softplus(b))).sum(-1)

    def gradient(self, z, landscape):
        a, b = landscape
        gain=self.amplitude if self.config.kind=='energy_full' else 1.
        return z - gain*(4/math.sqrt(self.units))*F.linear(
            a*torch.sigmoid(F.linear(z, self.directions)+b), self.directions.T)

    def logits(self, h, z, weight):
        return F.linear(h, weight)+F.linear(self.output(z), weight)

    def trajectory(self, h, memory, weight, *, steps=None):
        c = self.config; n = c.steps if steps is None else steps
        if type(n) is not int or n < 0 or n > 64: raise ValueError('Steps must be in [0,64]')
        context = self.context_input(h, memory)
        if c.kind == 'latent_ff':
            return [self.logits(h,self.transition(F.silu(self.first(context))),weight)]
        if c.kind in ('feedforward', 'memory_ff'):
            return [F.linear(h, weight)+F.linear(self.output(F.silu(self.first(context))), weight)]
        z = h.new_zeros((len(h), c.latent)); outputs = []
        fixed = self.landscape(context) if c.energy else None
        for _ in range(n):
            if c.energy: z = z-c.step_size*self.gradient(z, fixed)
            else:
                drive = self.transition(F.silu(self.first(torch.cat((context,z), -1))))
                z = (1-c.step_size)*z+c.step_size*drive
            outputs.append(self.logits(h,z,weight))
        return outputs or [F.linear(h,weight)]

    def forward(self, h, memory, weight, *, steps=None):
        # Inference needs only the final decoder projection. Training explicitly
        # calls trajectory() for its per-iterate supervision.
        c=self.config; n=c.steps if steps is None else steps
        if type(n) is not int or not 0<=n<=64: raise ValueError('Steps must be in [0,64]')
        if c.kind in ('feedforward','memory_ff','latent_ff'):
            return self.trajectory(h,memory,weight,steps=n)[-1]
        context=self.context_input(h,memory); z=h.new_zeros((len(h),c.latent))
        fixed=self.landscape(context) if c.energy else None
        for _ in range(n):
            if c.energy: z=z-c.step_size*self.gradient(z,fixed)
            else:
                drive=self.transition(F.silu(self.first(torch.cat((context,z),-1))))
                z=(1-c.step_size)*z+c.step_size*drive
        return self.logits(h,z,weight) if n else F.linear(h,weight)

    @torch.no_grad()
    def solve(self, h, memory, weight, *, max_steps=16, tolerance=1e-3,
              backtracks=8):
        """Inference-only fixed-energy descent; no target or future is accepted.

Per-example halting counts do not imply proportional batched wall-time savings.
A rejected line search leaves the candidate unchanged and terminates that row.
Energy descent is not proof of correct next-token prediction.
"""
        if not self.config.energy: raise ValueError('Adaptive energy solve requires an energy model')
        if (type(max_steps) is not int or not 1<=max_steps<=64 or
            type(backtracks) is not int or not 1<=backtracks<=32 or
            not math.isfinite(tolerance) or tolerance<0): raise ValueError('Invalid solve settings')
        fixed = self.landscape(self.context_input(h,memory))
        z=h.new_zeros((len(h),self.config.latent)); active=torch.ones(len(h),dtype=torch.bool,device=h.device)
        count=torch.zeros(len(h),dtype=torch.long,device=h.device)
        energies=[self.energy(z,fixed)]; accepted_steps=[]
        for _ in range(max_steps):
            if not active.any(): break
            old=energies[-1]; grad=self.gradient(z,fixed); lr=h.new_full((len(h),),self.config.step_size)
            accepted=~active; proposal=z.clone(); pe=old.clone()
            for _ in range(backtracks):
                trial=z-lr[:,None]*grad; te=self.energy(trial,fixed)
                ok=active & ~accepted & torch.isfinite(te) & torch.isfinite(trial).all(-1) & (te<=old)
                proposal=torch.where(ok[:,None],trial,proposal); pe=torch.where(ok,te,pe)
                accepted |= ok
                if accepted.all(): break
                lr=torch.where(accepted,lr,lr*.5)
            moved=active & accepted; count+=moved.long()
            delta=(old-pe).abs()/(1+old.abs())
            z=proposal; energies.append(pe); accepted_steps.append(moved)
            active=active & moved & (delta>tolerance) & (self.gradient(z,fixed).square().mean(-1).sqrt()>tolerance)
        return {'logits':self.logits(h,z,weight), 'z':z,'steps':count,
                'energies':torch.stack(energies,1),
                'attempts':len(energies)-1}


def load_refinement(path, *, device='cpu'):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('format')!='energy_refinement_v1': raise ValueError('Unknown refinement checkpoint')
    model=RefinementAdapter(RefinementConfig(**d['config']))
    model.load_state_dict(d['weights'],strict=True); model.to(device).eval()
    if not all(torch.isfinite(x).all() for x in model.state_dict().values()):
        raise ValueError('Nonfinite refinement checkpoint')
    if (model.memory_scale<=0).any():raise ValueError('Positive normalization required')
    return model,d
