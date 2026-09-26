"""Local predictive coding and conditional energy networks.

PC-ALM follows Seely & Gould, arXiv:2605.31022 (Algorithm 1), with
simultaneous layer updates, fresh per-example duals, and detached activity
inference. This is an independent PyTorch implementation, not the reference
JAX implementation or a reproduction of its parameterization.

``energy`` is a different model: prediction enumerates candidate labels and
relaxes hidden states to minimize a nonnegative scalar energy. Its finite-solve
training gradient is an approximate envelope gradient, not backprop through
the solver. No target label is passed to its prediction API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LocalConfig:
    inputs: int
    outputs: int
    width: int = 32
    depth: int = 8  # hidden layers; an additional linear output layer
    activation: str = 'tanh'
    memory_keys: int = 0
    memory_values: int = 0

    def validate(self):
        if min(self.inputs, self.outputs, self.width, self.depth) < 1:
            raise ValueError('Positive network dimensions required')
        if self.activation not in ('linear', 'tanh', 'relu'):
            raise ValueError('Unknown activation')
        if min(self.memory_keys, self.memory_values) < 0 or bool(self.memory_keys) != bool(self.memory_values):
            raise ValueError('Both memory dimensions must be positive, or both zero')


@dataclass
class Relaxation:
    states: Tensor
    duals: Tensor
    energy: Tensor
    residual_rms: float
    stationarity_rms: float
    trace: list[dict]


class LocalEnergyNetwork(nn.Module):
    """A chain of independent predictors with an optional memory reader per block.

    Each hidden state is [batch,width]. The layer axis is packed to make all
    local updates simultaneous and efficient. The first state is predicted
    from the input; subsequent predictions are residual transformations.
    Memory is a fixed [batch,address,value] analytic readout for an episode.
    Separate learned readers per residual block use that same factual state.
    """

    def __init__(self, config: LocalConfig):
        super().__init__()
        config.validate()
        self.config = config
        c = config
        self.input = nn.Linear(c.inputs, c.width)
        self.weights = nn.Parameter(torch.randn(c.depth - 1, c.width, c.width) / math.sqrt(c.width))
        self.biases = nn.Parameter(torch.zeros(c.depth - 1, c.width))
        self.output = nn.Linear(c.width, c.outputs)
        self.scale = 1 / math.sqrt(c.depth)
        if c.memory_keys:
            self.queries = nn.Parameter(torch.randn(c.depth - 1, c.width, c.memory_keys) / math.sqrt(c.width))
            self.values = nn.Parameter(torch.randn(c.depth - 1, c.memory_values, c.width) / math.sqrt(c.memory_values))
        else:
            self.register_parameter('queries', None)
            self.register_parameter('values', None)

    def activate(self, x: Tensor):
        if self.config.activation == 'linear':
            return x
        return x.tanh() if self.config.activation == 'tanh' else x.relu()

    def derivative(self, x: Tensor):
        if self.config.activation == 'linear':
            return torch.ones_like(x)
        return 1 - x.tanh().square() if self.config.activation == 'tanh' else (x > 0).to(x.dtype)

    def _memory(self, previous: Tensor, memory: Tensor | None):
        if self.queries is None:
            return None, None
        if memory is None or memory.shape != (previous.shape[1], self.config.memory_keys, self.config.memory_values):
            raise ValueError('Memory has the wrong shape or is absent')
        attention = torch.einsum('lbw,lwk->lbk', previous, self.queries).softmax(-1)
        read = torch.einsum('lbk,bkv->lbv', attention, memory)
        return attention, read

    def predictions(self, x: Tensor, states: Tensor, memory: Tensor | None = None):
        first = self.activate(self.input(x))
        previous = states[:-1]
        pre = torch.einsum('lbw,lvw->lbv', previous, self.weights) + self.biases[:, None]
        rest = previous + self.scale * self.activate(pre)
        _, read = self._memory(previous, memory)
        if read is not None:
            rest = rest + self.scale * torch.einsum('lbv,lvw->lbw', read, self.values)
        return torch.cat((first[None], rest), dim=0), self.output(states[-1])

    def feedforward(self, x: Tensor, memory: Tensor | None = None, *, return_states=False):
        state = self.activate(self.input(x))
        states = [state]
        for i in range(self.config.depth - 1):
            delta = self.activate(F.linear(state, self.weights[i], self.biases[i]))
            if self.queries is not None:
                if memory is None:
                    raise ValueError('Memory is required')
                attention = (state @ self.queries[i]).softmax(-1)
                read = torch.einsum('bk,bkv->bv', attention, memory)
                delta = delta + read @ self.values[i]
            state = state + self.scale * delta
            states.append(state)
        output = self.output(state)
        return (output, torch.stack(states)) if return_states else output

    def energy(self, x: Tensor, target: Tensor, states: Tensor,
               memory: Tensor | None = None, *, rho: float = 1., duals: Tensor | None = None):
        predicted, output = self.predictions(x, states, memory)
        residual = states - predicted
        value = .5 * (output - target).square().sum(-1) + .5 * rho * residual.square().sum((0, 2))
        if duals is not None:
            value = value + (duals * residual).sum((0, 2))
        return value

    def state_gradient(self, x: Tensor, target: Tensor, states: Tensor, memory: Tensor | None = None,
                       *, rho: float = 1., duals: Tensor | None = None):
        """Analytic nearest-neighbor VJPs; validated against autograd in tests."""
        predicted, output = self.predictions(x, states, memory)
        residual = states - predicted
        signal = rho * residual if duals is None else rho * residual + duals
        grad = signal.clone()
        previous = states[:-1]
        pre = torch.einsum('lbw,lvw->lbv', previous, self.weights) + self.biases[:, None]
        grad[:-1] -= signal[1:] + self.scale * torch.einsum(
            'lbv,lvw->lbw', signal[1:] * self.derivative(pre), self.weights)
        attention, _ = self._memory(previous, memory)
        if attention is not None:
            read_adjoint = torch.einsum('lbw,lvw->lbv', signal[1:], self.values)
            attention_adjoint = torch.einsum('lbv,bkv->lbk', read_adjoint, memory)
            score_adjoint = attention * (attention_adjoint - (attention * attention_adjoint).sum(-1, keepdim=True))
            grad[:-1] -= self.scale * torch.einsum('lbk,lwk->lbw', score_adjoint, self.queries)
        grad[-1] += (output - target) @ self.output.weight
        return grad, residual

    @torch.no_grad()
    def relax(self, x: Tensor, target: Tensor, memory: Tensor | None = None, *, steps: int = 16,
              state_lr: float = .2, dual_lr: float = 0., rho: float = 1.,
              initial: Tensor | None = None, trace: bool = False,
              check_budget: Callable | None = None):
        if steps < 1 or not all(math.isfinite(v) for v in (state_lr, dual_lr, rho)) or min(state_lr, rho) <= 0 or dual_lr < 0:
            raise ValueError('Invalid relaxation settings')
        states = self.feedforward(x, memory, return_states=True)[1].detach() if initial is None else initial.detach().clone()
        duals = torch.zeros_like(states)
        history = []
        for t in range(steps):
            if check_budget is not None and t % 16 == 0:
                check_budget()
            grad, residual = self.state_gradient(x, target, states, memory, rho=rho, duals=duals)
            if trace:
                history.append({'step': t, 'energy': float(self.energy(x, target, states, memory, rho=rho).mean()),
                                'residual_by_layer': residual.square().mean((1, 2)).sqrt().tolist(),
                                'credit_by_layer': (rho * residual + duals).square().mean((1, 2)).sqrt().tolist()})
            states = states - state_lr * grad
            # Algorithm 1 ends in a primal step, without a final dual update.
            if dual_lr and t != steps - 1:
                predicted, _ = self.predictions(x, states, memory)
                duals = duals + dual_lr * (states - predicted)
        grad, residual = self.state_gradient(x, target, states, memory, rho=rho, duals=duals)
        energy = self.energy(x, target, states, memory, rho=rho)
        if not torch.isfinite(states).all() or not torch.isfinite(duals).all() or not torch.isfinite(energy).all():
            raise FloatingPointError('Nonfinite activity dynamics; lower step size or inspect conditioning')
        return Relaxation(states, duals, energy, float(residual.square().mean().sqrt()),
                          float(grad.square().mean().sqrt()), history)

    def local_loss(self, x: Tensor, labels: Tensor, memory: Tensor | None = None, *, method='pcalm',
                   steps=16, state_lr=.2, dual_lr=1., rho=1., check_budget=None):
        target = F.one_hot(labels, self.config.outputs).to(x.dtype)
        if method == 'bp':
            output = self.feedforward(x, memory)
            return .5 * (output - target).square().sum(-1).mean(), None
        if method not in ('pc', 'pcalm'):
            raise ValueError('Local method must be bp, pc, or pcalm')
        settled = self.relax(x, target, memory, steps=steps, state_lr=state_lr,
                             dual_lr=dual_lr if method == 'pcalm' else 0., rho=rho,
                             check_budget=check_budget)
        # States and duals are detached: parameters receive only local credit.
        return self.energy(x, target, settled.states, memory, rho=rho, duals=settled.duals).mean(), settled

    def candidate_energies(self, x: Tensor, memory: Tensor | None = None, *, steps=16,
                           state_lr=.2, rho=1., candidate_chunk=16, check_budget=None,
                           return_diagnostics=False):
        """All candidate labels are treated identically. There is no gold-label input."""
        if candidate_chunk < 1:
            raise ValueError('Positive candidate chunk required')
        outputs = self.config.outputs
        batch = len(x)
        with torch.no_grad():
            initial = self.feedforward(x, memory, return_states=True)[1]
        energies, residuals, stationarity = [], [], []
        for start in range(0, outputs, candidate_chunk):
            end = min(start + candidate_chunk, outputs)
            count = end - start
            inputs = x[:, None].expand(-1, count, -1).reshape(batch * count, -1)
            targets = F.one_hot(torch.arange(start, end, device=x.device), outputs).to(x.dtype)
            targets = targets[None].expand(batch, -1, -1).reshape(batch * count, outputs)
            repeated = initial[:, :, None].expand(-1, -1, count, -1).reshape(self.config.depth, batch * count, -1)
            mem = None if memory is None else memory[:, None].expand(-1, count, -1, -1).reshape(batch * count, *memory.shape[1:])
            settled = self.relax(inputs, targets, mem, steps=steps, state_lr=state_lr,
                                 rho=rho, initial=repeated, check_budget=check_budget)
            energies.append(self.energy(inputs, targets, settled.states, mem, rho=rho).reshape(batch, count))
            residuals.append(settled.residual_rms)
            stationarity.append(settled.stationarity_rms)
        result = torch.cat(energies, dim=1)
        if return_diagnostics:
            return result, {'residual_rms': sum(residuals) / len(residuals),
                            'stationarity_rms': sum(stationarity) / len(stationarity)}
        return result

    def energy_loss(self, x: Tensor, labels: Tensor, memory: Tensor | None = None, *,
                    temperature=.1, positive_weight=.1, **kwargs):
        if temperature <= 0 or not math.isfinite(temperature) or positive_weight < 0:
            raise ValueError('Invalid energy objective settings')
        energies = self.candidate_energies(x, memory, **kwargs)
        positive = energies.gather(1, labels[:, None]).mean()
        return F.cross_entropy(-energies / temperature, labels) + positive_weight * positive

    @torch.no_grad()
    def predict(self, x: Tensor, memory: Tensor | None = None, *, method='bp', temperature=.1, **kwargs):
        if method == 'energy':
            return -self.candidate_energies(x, memory, **kwargs) / temperature
        if method not in ('bp', 'pc', 'pcalm'):
            raise ValueError('Unknown prediction method')
        return self.feedforward(x, memory)


def gradient_alignment(model: LocalEnergyNetwork, x: Tensor, labels: Tensor,
                       memory: Tensor | None = None, *, method='pcalm', **kwargs):
    """Compare all local parameter gradients to BP on the identical MSE objective."""
    params = list(model.parameters())
    bp, _ = model.local_loss(x, labels, memory, method='bp')
    reference = torch.autograd.grad(bp, params)
    loss, settled = model.local_loss(x, labels, memory, method=method, **kwargs)
    actual = torch.autograd.grad(loss, params)
    a, b = (torch.cat([g.detach().reshape(-1) for g in grads]) for grads in (actual, reference))
    return {'cosine': float(F.cosine_similarity(a, b, dim=0)),
            'relative_error': float((a - b).norm() / b.norm().clamp_min(1e-12)),
            'parameter_cosines': {name: float(F.cosine_similarity(g.reshape(-1), ref.reshape(-1), dim=0))
                                 for (name, _), g, ref in zip(model.named_parameters(), actual, reference)},
            'residual_rms': None if settled is None else settled.residual_rms,
            'stationarity_rms': None if settled is None else settled.stationarity_rms}


def model_payload(model: LocalEnergyNetwork):
    return {'config': asdict(model.config), 'state_dict': model.state_dict()}
