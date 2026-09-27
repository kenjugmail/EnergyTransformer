"""Optional causal next-byte adapters; never install a predicted future as memory.

All adapters change logits explicitly, not teacher P/S. The energy-read arm asks
learned queries of the frozen joint forecast density; it is not KV attention,
iterative energy minimization, or distribution-preserving speculative decoding.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .model import EnergyLandscape
from .experiment import atomic_save, load_head
from .probe import checkpoint_hash, load_energy_model, state_sketch

KINDS = ('hidden', 'history', 'mean', 'moments', 'energy_read')

@dataclass(frozen=True)
class DecoderConfig:
    kind: str
    width: int = 160
    history: int = 16
    variables: int = 8
    horizons: tuple[int, ...] = (1, 4, 16, 64)
    queries: int = 2
    read_bottleneck: int = 36

    def validate(self):
        if self.kind not in KINDS or any(type(v) is not int or v < 1 for v in
                (self.width, self.history, self.variables, self.queries, self.read_bottleneck)):
            raise ValueError('Invalid adapter configuration')
        if self.history != 16 or len(self.horizons) != 4 or self.queries != 2:
            raise ValueError('This bounded study expects 16-history, four horizons, two energy queries')
        if tuple(sorted(set(self.horizons))) != self.horizons or min(self.horizons) < 1:
            raise ValueError('Invalid horizons')

    @property
    def dimension(self): return self.variables * len(self.horizons)
    @property
    def side_width(self): return self.history * self.variables
    @property
    def bottleneck(self):
        if self.kind == 'energy_read': return self.read_bottleneck
        qparams = (self.width + 1) * self.queries * self.dimension
        target = qparams + (2*self.width+self.side_width+1)*self.read_bottleneck + self.width
        side = 0 if self.kind == 'hidden' else self.side_width
        return max(1, round((target-self.width)/(2*self.width+side+1)))


def distribution_moments(fun: EnergyLandscape):
    weights = fun.logits.softmax(-1)[..., None]
    mean = (weights*fun.means).sum(1)
    within = fun.scales.square()*fun.correlation_cholesky.square().sum(-1)[None]
    variance = (weights*(within+(fun.means-mean[:, None]).square())).sum(1)
    return mean, variance


def energy_query_read(fun: EnergyLandscape, candidates: Tensor) -> Tensor:
    """[B,Q,D] candidates -> [B,Q,2D] expected component means and bounded force.

    The force is exactly -grad E before its stated dimensionless tanh readout.
    It is differentiable through the learned candidate, using two triangular
    solves. This read is an architecture proposal, not a Bayesian observation.
    """
    if candidates.ndim != 3 or candidates.shape[0] != fun.means.shape[0] or candidates.shape[-1] != fun.means.shape[-1]:
        raise ValueError('Candidate query shape mismatch')
    delta = (candidates[:, :, None]-fun.means[:, None])/fun.scales[:, None]
    lower = fun.correlation_cholesky[None, None]
    whitened = torch.linalg.solve_triangular(lower, delta[..., None], upper=False)
    inverse = torch.linalg.solve_triangular(lower.transpose(-1, -2), whitened, upper=True)[..., 0]/fun.scales[:, None]
    logdet = fun.scales.log().sum(-1)+fun.correlation_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)[None]
    logp = -.5*(whitened[..., 0].square().sum(-1)+candidates.shape[-1]*math.log(2*math.pi))-logdet[:, None]
    responsibility = (logp+fun.logits.log_softmax(-1)[:, None]).softmax(-1)
    selected = (responsibility[..., None]*fun.means[:, None]).sum(2)
    force = -(responsibility[..., None]*inverse).sum(2)
    _, variance = distribution_moments(fun)
    dimensionless = force*variance.clamp_min(1e-10).sqrt()[:, None]
    return torch.cat((selected, torch.tanh(dimensionless)), -1)


class DecoderAdapter(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__(); config.validate(); self.config = config
        c = config
        side = 0 if c.kind == 'hidden' else c.side_width
        self.first = nn.Linear(c.width+side, c.bottleneck)
        self.last = nn.Linear(c.bottleneck, c.width)
        nn.init.zeros_(self.last.weight); nn.init.zeros_(self.last.bias)
        if c.kind == 'energy_read':
            self.query = nn.Linear(c.width, c.queries*c.dimension)
            nn.init.normal_(self.query.weight, std=.01); nn.init.zeros_(self.query.bias)
        self.register_buffer('feature_center', torch.zeros(side))
        self.register_buffer('feature_scale', torch.ones(side))

    def raw_features(self, h: Tensor, history: Tensor, fun: EnergyLandscape | None = None,
                     mean: Tensor | None = None):
        c = self.config
        if h.ndim != 2 or h.shape[-1] != c.width or history.shape != (len(h), c.history, c.variables):
            raise ValueError('Hidden/history dimensions disagree')
        if c.kind == 'hidden': return h.new_empty((len(h), 0))
        if c.kind == 'history': return history.flatten(1)
        if c.kind == 'mean':
            if mean is None: raise ValueError('Mean forecast required')
            return mean.repeat(1, 4)*.5
        if fun is None: raise ValueError('Joint energy forecast required')
        if c.kind == 'moments':
            mu, var = distribution_moments(fun)
            return torch.cat((mu, .5*var.clamp_min(1e-10).log()), -1).repeat(1, 2)/math.sqrt(2)
        candidate = fun.mean[:, None]+.1*self.query(h).reshape(len(h), c.queries, c.dimension)
        return energy_query_read(fun, candidate).flatten(1)

    def delta(self, h: Tensor, history: Tensor, fun: EnergyLandscape | None = None,
              mean: Tensor | None = None, *, erase_side: bool = False):
        raw = self.raw_features(h, history, fun, mean)
        side = (raw-self.feature_center)/self.feature_scale
        if erase_side: side = torch.zeros_like(side)
        return self.last(F.gelu(self.first(torch.cat((h, side), -1))))

    def forward(self, h: Tensor, history: Tensor, weight: Tensor, fun: EnergyLandscape | None = None,
                mean: Tensor | None = None, *, erase_side: bool = False):
        return F.linear(h, weight)+F.linear(self.delta(h, history, fun, mean, erase_side=erase_side), weight)


def load_adapter(path, device='cpu'):
    d = torch.load(path, weights_only=True, map_location=device)
    if d.get('format') != 'decoder_bridge_v1': raise ValueError('Unknown adapter format')
    cfg = dict(d['config']); cfg['horizons'] = tuple(cfg['horizons'])
    m = DecoderAdapter(DecoderConfig(**cfg)).to(device)
    m.load_state_dict(d['weights'], strict=True); m.eval()
    if any(not torch.isfinite(t).all() for t in m.state_dict().values()) or (m.feature_scale <= 0).any():
        raise ValueError('Invalid adapter parameters')
    for p in m.parameters(): p.requires_grad_(False)
    return m, d


class DecoderBridgeSession:
    """Explicitly enabled logit adapter. Teacher memory evolves only from input bytes.

    Before the configured real-history warmup, outputs equal the teacher. A hidden
    adapter requires no additional recurrent history but observes the same warmup
    for evaluation fairness. No online-conditioning gate is reused here.
    """
    def __init__(self, teacher, adapter, head=None, device='cpu'):
        self.device = torch.device(device)
        self.teacher_hash = checkpoint_hash(teacher); self.adapter_hash = checkpoint_hash(adapter)
        self.adapter, self.metadata = load_adapter(adapter, self.device)
        if self.teacher_hash != self.metadata['teacher_sha256']: raise ValueError('Teacher identity mismatch')
        self.teacher = load_energy_model(teacher, self.device)
        if self.teacher.config.width != self.adapter.config.width: raise ValueError('Teacher width mismatch')
        self.head = None; self.head_hash = None
        if self.adapter.config.kind in ('mean', 'moments', 'energy_read'):
            if head is None: raise ValueError('Matching forecast checkpoint required')
            self.head_hash = checkpoint_hash(head)
            if self.head_hash != self.metadata['head_sha256']: raise ValueError('Forecast identity mismatch')
            self.head, hm = load_head(head, self.device)
            if hm['teacher_sha256'] != self.teacher_hash: raise ValueError('Forecast teacher mismatch')
            for p in self.head.parameters(): p.requires_grad_(False)
        norm = self.metadata.get('normalization')
        if not isinstance(norm, dict) or set(norm) != {'center', 'scale'}:
            raise ValueError('Missing normalization')
        for name, value in norm.items():
            if (not isinstance(value, Tensor) or value.shape != (self.adapter.config.variables,)
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('Invalid normalization')
        if (norm['scale'] <= 0).any() or self.adapter.config.variables != 2*self.teacher.config.heads:
            raise ValueError('Normalization or sketch width mismatch')
        if self.head is not None:
            if (self.head.config.history != self.adapter.config.history
                    or self.head.config.variables != self.adapter.config.variables
                    or self.head.config.horizons != self.adapter.config.horizons):
                raise ValueError('Forecast architecture mismatch')
            if any(not torch.equal(norm[k].cpu(), hm['normalization'][k].cpu()) for k in norm):
                raise ValueError('Forecast normalization mismatch')
        self.center = norm['center'].to(self.device)
        self.scale = norm['scale'].to(self.device)
        self._captured = None
        self._hook = self.teacher.norm.register_forward_hook(self._capture)
        self.reset()

    def _capture(self, module, args, output): self._captured = output.detach()
    def reset(self): self.state = None; self.history = None; self.steps = 0

    @torch.no_grad()
    def observe(self, tokens: Tensor):
        if (not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1
                or tokens.dtype != torch.long or (tokens < 0).any() or (tokens >= self.teacher.config.vocab).any()):
            raise ValueError('Expected one nonempty valid integer byte stream')
        answer=[]; c=self.adapter.config
        for i in range(tokens.shape[1]):
            base, new = self.teacher(tokens[:, i:i+1].to(self.device), self.state)
            self.state = new; self.steps += 1
            if c.kind != 'hidden':
                sketch = state_sketch(new)[:, None]
                self.history = sketch if self.history is None else torch.cat((self.history, sketch), 1)
                self.history = self.history[:, -c.history:].clone()
            if self.steps < c.history: answer.append(base); continue
            h=self._captured[:, -1]
            history = h.new_zeros((1, c.history, c.variables)) if c.kind == 'hidden' else (self.history-self.center)/self.scale
            fun = self.head(history) if c.kind in ('moments', 'energy_read') else None
            mean = self.head.anchor(history) if c.kind == 'mean' else None
            delta=self.adapter.delta(h, history, fun, mean)
            answer.append(base+F.linear(delta, self.teacher.head.weight)[:, None])
        self._captured = None
        return torch.cat(answer, 1)

    def save(self, path):
        atomic_save({'format':'decoder_bridge_session_v1','teacher':self.teacher_hash,'adapter':self.adapter_hash,
                     'head':self.head_hash,'state':self.state,'history':self.history,'steps':self.steps}, path)

    def restore(self, path):
        d=torch.load(path, weights_only=True, map_location=self.device)
        expected={'format','teacher','adapter','head','state','history','steps'}
        if set(d)!=expected or d['format']!='decoder_bridge_session_v1': raise ValueError('Invalid snapshot')
        if (d['teacher'],d['adapter'],d['head'])!=(self.teacher_hash,self.adapter_hash,self.head_hash): raise ValueError('Snapshot identity mismatch')
        n=d['steps']; state=d['state']; history=d['history']; c=self.teacher.config
        if type(n) is not int or n<0: raise ValueError('Invalid step count')
        if n==0:
            if state is not None or history is not None: raise ValueError('Invalid empty snapshot')
        else:
            if not isinstance(state,(tuple,list)) or len(state)!=c.layers: raise ValueError('Invalid layer state')
            for layer in state:
                if not isinstance(layer,(tuple,list)) or len(layer)!=2: raise ValueError('Invalid energy layer')
                p,s=layer
                if (not all(isinstance(t,Tensor) and t.dtype==torch.float32 and torch.isfinite(t).all() for t in (p,s))
                        or p.shape!=(1,c.heads,c.slots) or s.shape!=(1,c.heads,c.slots,c.width//c.heads) or (p<c.prior).any()):
                    raise ValueError('Invalid energy coefficients')
            if self.adapter.config.kind=='hidden':
                if history is not None: raise ValueError('Hidden-only adapter stores no sketch history')
            else:
                shape=(1,min(n,self.adapter.config.history),self.adapter.config.variables)
                if (not isinstance(history,Tensor) or history.shape!=shape or history.dtype!=torch.float32
                        or not torch.isfinite(history).all() or not torch.equal(history[:,-1],state_sketch(state))):
                    raise ValueError('Invalid observed history')
        self.state=state; self.history=history; self.steps=n
