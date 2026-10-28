"""Differentiable solvers and execution primitives for REWT experiments.

Certificates are conservative floating-point spectral bounds, rebuilt from the
current weights. They are not convexity or prediction-correctness certificates.
No training graph or weight-dependent factor is cached across optimizer steps.
"""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from .kernels import core_quadratic_action


def packed_condition_weights(model):
    layers = (model.prior[0], model.output_prior, model.amplitude)
    return (torch.cat([x.weight for x in layers]), torch.cat([x.bias for x in layers]))


def gru_from_input(input_affine, hidden, cell):
    """PyTorch GRUCell ordering, including the reset gate on the hidden bias."""
    ir, iz, inn = input_affine.chunk(3, -1)
    hr, hz, hn = F.linear(hidden, cell.weight_hh, cell.bias_hh).chunk(3, -1)
    reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
    candidate = (inn + reset * hn).tanh()
    return candidate + update * (hidden - candidate)


def quadratic_action(model, direction, cond):
    """H_cond times a direction, including frozen routing and specialists."""
    if not model.config.support:
        return core_quadratic_action(model, direction, cond.precision)
    c = model.config
    x = model.canonical_state(direction)
    parts = {name: cond.precision[sl] * x[..., sl] for name, sl in model.blocks.items()}
    residual = x[..., model.blocks['recurrent']] - x[..., model.blocks['attention']]
    parts['recurrent'] = parts['recurrent'] + c.attention_weight * residual
    parts['attention'] = parts['attention'] - c.attention_weight * residual
    residual = x[..., model.blocks['output']] - model.output_coordinates(
        F.linear(model._decoder_latent(x), model.decoder.weight))
    parts['output'] = parts['output'] + c.decoder_weight * residual
    for name in ('recurrent', 'attention', 'feedforward', 'communication'):
        parts[name] = parts[name] - c.decoder_weight * (residual @ model.decoder.weight) / 4
    communication = x[..., model.blocks['communication']]
    message = model.bridge((cond.routing[..., None] * model._candidate_states(x)).sum(-2))
    residual = communication - message
    parts['communication'] = parts['communication'] + c.communication_weight * residual
    candidate = -c.communication_weight * (residual @ model.bridge.weight)
    for i, name in enumerate(model.candidate_names):
        contribution = cond.routing[..., i, None] * candidate
        if name == 'forecast':
            contribution = contribution @ model.forecast_bridge.weight
        parts[name] = parts[name] + contribution
    for i, link in enumerate(model.specialist_links):
        name = f'specialist_{i}'
        residual = x[..., model.blocks[name]] - link(communication)
        parts[name] = parts[name] + c.specialist_weight * residual
        parts['communication'] = parts['communication'] - c.specialist_weight * (residual @ link.weight)
    return model.canonical_state(torch.cat(tuple(parts.values()), -1))


def directional_refine(model, state, cond, *, rounds=None, update_states=True,
                       update_output=True, trace=False, active_mask=None):
    """All line-search energies from one quadratic directional expansion.

    Handles arbitrary initial states, absolute/relative geometry, Fourier/tanh,
    routing, masked proposals and coarse schedules. Comparisons always use the
    complete energy, irrespective of the proposal's mode mask.
    """
    c = model.config
    rounds = c.rounds if rounds is None else rounds
    if type(rounds) is not int or rounds < 0:
        raise ValueError('rounds must be nonnegative')
    if cond.frequency is None or cond.precision is None:
        cond = replace(cond, frequency=model.frequencies() if cond.frequency is None else cond.frequency,
                       precision=.15 + 1.85 * model.raw_precision.sigmoid() if cond.precision is None else cond.precision)
    state = model.canonical_state(state)
    mask = state.new_ones(model.dimensions)
    if not update_states:
        mask[:model.blocks['output'].start] = 0
    if not update_output:
        mask[model.blocks['output']] = 0
    if active_mask is not None:
        mask = mask * active_mask
    steps = state.new_tensor([c.step_size * .5**i for i in range(c.backtracks)])
    energies = []
    metrics = dict(gradient_calls=0, energy_calls=0, accepted=0, attempts=0)
    halted = torch.zeros(state.shape[:-1], dtype=torch.bool, device=state.device)
    for r in range(rounds):
        mode_mask = None
        if c.schedule == 'coarse_to_fine':
            order = cond.frequency.detach().square().sum(-1).argsort()
            n = max(1, math.ceil(c.modes * (r + 1) / rounds))
            mode_mask = state.new_zeros(c.modes).scatter(0, order[:n], 1.)
        phase = F.linear(state - cond.anchor if c.wave_geometry == 'anchor_relative' else state,
                         cond.frequency, model.phase)
        waves = phase.cos() if c.activation == 'fourier' else phase.tanh()
        derivative = -phase.sin() if c.activation == 'fourier' else 1 - waves.square()
        coefficients = cond.amplitudes * derivative
        # Recover Q's gradient from the full analytic gradient without numerical
        # differencing of energies or dense Hessian construction.
        full_gradient = model.gradient(state, cond)
        qgrad = full_gradient - model.canonical_state(coefficients @ cond.frequency)
        gradient = (qgrad + model.canonical_state(
            (coefficients if mode_mask is None else coefficients * mode_mask) @ cond.frequency)) * mask
        gradient = gradient * (~halted)[..., None]
        quadratic = model.quadratic(state, cond)
        before = quadratic + (cond.amplitudes * waves).sum(-1)
        slope = (qgrad * gradient).sum(-1)
        curvature = (gradient * quadratic_action(model, gradient, cond)).sum(-1)
        projected = model.canonical_state(gradient) @ cond.frequency.T
        trial_phase = phase[..., None, :] - steps[:, None] * projected[..., None, :]
        trial_waves = trial_phase.cos() if c.activation == 'fourier' else trial_phase.tanh()
        after = (quadratic[..., None] - steps * slope[..., None]
                 + .5 * steps.square() * curvature[..., None]
                 + (cond.amplitudes[..., None, :] * trial_waves).sum(-1))
        valid = torch.isfinite(after) & (after < before[..., None])
        alpha = torch.where(valid, steps, 0.).amax(-1)
        accepted = alpha > 0
        chosen = torch.where(accepted[..., None], state - alpha[..., None] * gradient, state)
        selected = torch.where(accepted, after.gather(-1, valid.to(torch.int32).argmax(-1, keepdim=True)).squeeze(-1), before)
        if trace:
            if not energies:
                energies.append(before.detach())
            # Report the actual selected-state energy, not just the expansion.
            energies.append(model.energy(chosen, cond).detach())
        if c.halting_tolerance and r + 1 >= c.minimum_refinement_rounds:
            drop = (before - selected) / before.abs().clamp_min(1.)
            sl = model.blocks['output']
            delta = (chosen[..., sl].softmax(-1) - state[..., sl].softmax(-1)).abs().sum(-1)
            halted = halted | ((drop <= c.halting_tolerance) & (delta <= c.halting_tolerance))
        state = chosen
        metrics['gradient_calls'] += 1
        metrics['energy_calls'] += 1 + c.backtracks + int(trace)
        metrics['attempts'] += len(state) * c.backtracks
        if trace:
            metrics['accepted'] += int(accepted.sum())
    if trace and not energies:
        energies.append(model.energy(state, cond).detach())
        metrics['energy_calls'] += 1
    return state, metrics, energies


def core_matrix(model, frequency, precision):
    identity = torch.eye(model.dimensions, device=frequency.device, dtype=frequency.dtype)
    h = core_quadratic_action(model, identity, precision)
    h = (h + h.T) / 2
    if model.config.output_geometry == 'centered':
        gauge = identity.new_zeros(model.dimensions)
        gauge[model.blocks['output']] = 1 / math.sqrt(256)
        h = h + gauge[:, None] * gauge[None]
    return h


def spectral_upper(matrix):
    """Detached double-precision eigenvalue plus an explicit rounding margin."""
    with torch.no_grad():
        a = matrix.detach().double()
        a = (a + a.T) / 2
        maximum = torch.linalg.eigvalsh(a)[-1]
        margin = 1e-6 * a.abs().sum(-1).amax().clamp_min(1.)
        return (maximum + margin).to(matrix.dtype)


def prepare_solver(model, frequency, precision, solver):
    if solver not in ('certified', 'preconditioned'):
        raise ValueError('unknown solver')
    if solver == 'preconditioned' and getattr(model.config, 'preconditioner_factorization', 'dense') == 'structured':
        from .structured_solver import prepare_structured
        return prepare_structured(model, frequency, precision)
    h = core_matrix(model, frequency, precision)
    amplitude_bound = .5 / math.sqrt(model.config.modes)
    if solver == 'preconditioned':
        # Gauge fixing makes the centered problem positive definite; F has no
        # gauge component, so solving this system gives its unique centered lift.
        lift = torch.cholesky_solve(frequency.T, torch.linalg.cholesky(h)).T
        gram = frequency @ lift.T
        gram = (gram + gram.T) / 2
        upper = 1 + amplitude_bound * spectral_upper(gram)
        step = torch.minimum(upper.new_tensor(model.config.step_size), 1.9 / upper)
        return dict(lift=lift, gram=gram, step=step, upper=upper)
    upper = spectral_upper(h + amplitude_bound * frequency.T @ frequency)
    step = torch.minimum(upper.new_tensor(model.config.step_size), 1.9 / upper)
    return dict(h=h, hf=core_quadratic_action(model, frequency, precision),
                gram=frequency @ frequency.T, step=step, upper=upper)


def reduced_refine(model, anchor, amplitudes, frequency, precision, prepared, solver,
                   rounds, *, trace=False):
    """Certified steps from the anchor, with arbitrary depth in M coordinates."""
    eta = prepared['step']
    if solver == 'preconditioned':
        coefficients = torch.zeros_like(amplitudes)
        values = []
        for r in range(rounds):
            projected = coefficients @ prepared['gram']
            if trace:
                values.append((.5 * (coefficients * projected).sum(-1)
                               + (amplitudes * (projected + model.phase).cos()).sum(-1)).detach())
            candidate = (1 - eta) * coefficients + eta * amplitudes * (projected + model.phase).sin()
            coefficients = torch.where(torch.isfinite(candidate).all(-1, keepdim=True), candidate, coefficients)
        state = model.canonical_state(anchor + coefficients @ prepared['lift'])
    elif rounds == 2 and not trace:
        # The two fixed steps require only two M-dimensional projections and
        # a final lift; no D-dimensional trial states or energy evaluations.
        hf = prepared['hf']
        first = -eta * (-amplitudes * model.phase.sin())
        wave = -amplitudes * (first @ prepared['gram'] + model.phase).sin()
        state = model.canonical_state(anchor + (first - eta * wave) @ frequency - eta * first @ hf)
        state = torch.where(torch.isfinite(state).all(-1, keepdim=True), state, anchor)
        values = []
    else:
        displacement = torch.zeros_like(anchor)
        values = []
        for r in range(rounds):
            hz = core_quadratic_action(model, displacement, precision)
            projected = displacement @ frequency.T + model.phase
            if trace:
                values.append((.5 * (displacement * hz).sum(-1) + (amplitudes * projected.cos()).sum(-1)).detach())
            gradient = hz - (amplitudes * projected.sin()) @ frequency
            candidate = displacement - eta * gradient
            displacement = torch.where(torch.isfinite(candidate).all(-1, keepdim=True), candidate, displacement)
        state = model.canonical_state(anchor + displacement)
    if trace:
        z = state - anchor
        values.append((.5 * (z * core_quadratic_action(model, z, precision)).sum(-1)
                       + (amplitudes * (z @ frequency.T + model.phase).cos()).sum(-1)).detach())
    return state, values


def affine_scan(a, b, initial):
    """Inclusive parallel scan of h[t] = a[t]*h[t-1] + b[t], no division.

    Hillis-Steele is a portable differentiable reference; unlike a cumprod ratio,
    it remains well defined when retention gates are zero or underflow.
    """
    offset = 1
    while offset < a.shape[1]:
        a, b = (torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), 1),
                torch.cat((b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]), 1))
        offset *= 2
    return a * initial[:, None] + b


def delta_memory(current, queries, memory, key_layer, value_layer, gate_layer):
    """Causal normalized delta writes, followed by a read at each position."""
    keys = F.normalize(key_layer(current), dim=-1, eps=1e-6)
    values = value_layer(current)
    beta = gate_layer(current).sigmoid()
    queries = F.normalize(queries, dim=-1, eps=1e-6)
    reads = []
    for t in range(current.shape[1]):
        key = keys[:, t]
        error = values[:, t] - (key[..., None] * memory).sum(-2)
        memory = memory + beta[:, t, :, None] * key[..., None] * error[:, None]
        reads.append((queries[:, t, :, None] * memory).sum(-2))
    return torch.stack(reads, 1), memory


def delta_chunk(keys, values, beta, queries, memory):
    """Exact delta recurrence via a unit lower-triangular solve within a chunk.

    u[t] + beta[t] sum(i<t) <k[t],k[i]> u[i]
        = beta[t] (v[t] - k[t] S_initial).
    No inverse is formed. Caller supplies normalized keys and queries.
    """
    length = keys.shape[1]
    cross = keys @ keys.transpose(-1, -2)
    lower = torch.tril(beta * cross, diagonal=-1)
    lower = lower + torch.eye(length, dtype=keys.dtype, device=keys.device)
    writes = torch.linalg.solve_triangular(lower, beta * (values - keys @ memory), upper=False, unitriangular=True)
    reads = queries @ memory + torch.tril(queries @ keys.transpose(-1, -2)) @ writes
    return reads, memory + keys.transpose(-1, -2) @ writes
