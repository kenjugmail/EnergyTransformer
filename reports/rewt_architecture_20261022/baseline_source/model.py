"""One conditional energy over every refinement workspace and output logit.

Neural networks propose guesses and parameterize the energy. Accepted refinement
updates minimize this *same* energy, including its quadratic compatibility terms.
Observed byte history is separate from forecasts and disposable candidates.
The analytic state gradient remains differentiable in the learned parameters;
training through refinement therefore needs no nested autograd invocation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EnergyConfig:
    width: int = 48
    hidden: int = 384
    modes: int = 16
    history: int = 16
    rounds: int = 2
    step_size: float = 0.25
    backtracks: int = 4
    frequency_bound: float = 2.0
    activation: str = "fourier"
    support: bool = False
    specialists: int = 3
    auxiliary_hidden: int = 32
    horizons: tuple[int, ...] = (1, 4, 16, 64)
    diffusion_samples: int = 2
    diffusion_levels: int = 3
    learned_frequencies: bool = True
    coupled_modes: bool = True
    schedule: str = "all"
    attention_weight: float = 0.04
    decoder_weight: float = 0.04
    communication_weight: float = 0.04
    specialist_weight: float = 0.02
    # Raw coordinates remain available to load the original pilot faithfully.
    output_geometry: str = "raw"
    normalize_frequency_blocks: bool = False
    cache_history: bool = False
    early_backtrack: bool = False
    halting_tolerance: float = 0.
    minimum_refinement_rounds: int = 1
    # Defaults preserve the archived architecture and checkpoint tensors.
    compatibility_geometry: str = "absolute"
    recurrent_type: str = "mlp"
    context_kernel: int = 1
    wave_geometry: str = "absolute"
    initial_carry_scale: float = 0.05
    refinement_backend: str = "reference"

    def validate(self):
        for name in ("width", "hidden", "modes", "history", "auxiliary_hidden",
                     "diffusion_samples", "diffusion_levels", "specialists", "context_kernel"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.rounds) is not int or self.rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if type(self.backtracks) is not int or self.backtracks < 1:
            raise ValueError("At least one backtracking attempt is required")
        for name in ("step_size", "frequency_bound"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("attention_weight", "decoder_weight", "communication_weight",
                     "specialist_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.activation not in ("fourier", "mlp"):
            raise ValueError("activation must be fourier or mlp")
        if self.schedule not in ("all", "coarse_to_fine"):
            raise ValueError("Unknown refinement schedule")
        if self.output_geometry not in ("raw", "centered"):
            raise ValueError("output_geometry must be raw or centered")
        if self.compatibility_geometry not in ("absolute", "anchor_relative"):
            raise ValueError("Unknown compatibility geometry")
        if self.recurrent_type not in ("mlp", "gru"):
            raise ValueError("recurrent_type must be mlp or gru")
        if self.wave_geometry not in ("absolute", "anchor_relative"):
            raise ValueError("Unknown wave geometry")
        if not math.isfinite(self.initial_carry_scale) or self.initial_carry_scale < 0:
            raise ValueError("initial_carry_scale must be finite and nonnegative")
        if self.refinement_backend not in ("reference", "krylov2"):
            raise ValueError("refinement_backend must be reference or krylov2")
        if self.refinement_backend == "krylov2" and (
                self.support or self.activation != "fourier" or self.schedule != "all"
                or self.compatibility_geometry != "anchor_relative"
                or self.wave_geometry != "anchor_relative"
                or self.initial_carry_scale != 0 or self.halting_tolerance != 0):
            raise ValueError("krylov2 requires the Fourier core, anchor-relative geometries, "
                             "zero initial carry, all-mode updates and no adaptive halting")
        if not math.isfinite(self.halting_tolerance) or self.halting_tolerance < 0:
            raise ValueError("halting_tolerance must be finite and nonnegative")
        if type(self.minimum_refinement_rounds) is not int or self.minimum_refinement_rounds < 1:
            raise ValueError("minimum_refinement_rounds must be a positive integer")
        if (not self.horizons or len(set(self.horizons)) != len(self.horizons)
                or any(type(h) is not int or h < 1 for h in self.horizons)):
            raise ValueError("Distinct positive forecast horizons required")


@dataclass(frozen=True)
class TransformerConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    feedforward: int = 384
    context: int = 128

    def validate(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Positive integer Transformer dimensions required")
        if self.width % self.heads:
            raise ValueError("Transformer width must be divisible by heads")


@dataclass
class EnergyState:
    joint: Tensor
    observed: Tensor
    count: int
    identity: str

    def detach(self):
        return EnergyState(self.joint.detach(), self.observed.detach(), self.count, self.identity)


@dataclass
class TransformerState:
    tokens: Tensor


@dataclass
class Output:
    logits: Tensor
    state: EnergyState | TransformerState
    details: dict


@dataclass
class Conditioning:
    anchor: Tensor
    amplitudes: Tensor
    routing: Tensor | None
    features: Tensor
    extras: dict
    frequency: Tensor | None = None
    precision: Tensor | None = None


def mlp(inputs: int, hidden: int, outputs: int):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class JointEnergyLM(nn.Module):
    """Full joint energy, or an exactly same-size non-Fourier activation control.

    Q is a positive quadratic: a diagonal anchor plus squared linear residuals
    for attention, decoder compatibility, communication and specialist agreement.
    Routing/critic coefficients are conditioned once on the observed prefix and
    remain fixed across candidate comparisons and every refinement round.
    """

    family = "energy"

    def __init__(self, config: EnergyConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.width
        self.blocks: dict[str, slice] = {}
        start = 0
        names = [("recurrent", d), ("attention", d), ("feedforward", d)]
        if config.support:
            names += [("forecast", d * len(config.horizons)), ("diffusion", d)]
            names += [(f"specialist_{i}", d) for i in range(config.specialists)]
            names += [("communication", d)]
        names += [("output", 256)]
        for name, size in names:
            self.blocks[name] = slice(start, start + size)
            start += size
        self.dimensions = start
        self.embedding = nn.Embedding(256, d)
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        # The core controls share the same causal context and prior networks.
        # The full model gets its recurrent proposal from a GRU. Avoid unused
        # feedforward outputs so the parameter match counts working parameters.
        self.prior = mlp(3 * d, config.hidden,
                         d if config.support or config.recurrent_type == "gru" else 2 * d)
        self.decoder = nn.Linear(d, 256)
        self.output_prior = nn.Linear(3 * d, 256)
        self.amplitude = nn.Linear(3 * d, config.modes)
        self.raw_frequency = nn.Parameter(torch.randn(config.modes, start) / math.sqrt(start),
                                          requires_grad=config.learned_frequencies)
        self.phase = nn.Parameter(torch.rand(config.modes) * 2 * math.pi - math.pi)
        self.raw_precision = nn.Parameter(torch.zeros(start))
        # Each row spans all blocks, or only one block in the coupling ablation.
        mode_mask = torch.ones(config.modes, start)
        if not config.coupled_modes:
            mode_mask.zero_()
            for k in range(config.modes):
                mode_mask[k, self.blocks[names[k % len(names)][0]]] = 1
        self.register_buffer("mode_mask", mode_mask)
        scale = torch.ones(start)
        if config.normalize_frequency_blocks:
            for sl in self.blocks.values():
                scale[sl] = math.sqrt(d / (sl.stop - sl.start))
        self.register_buffer("coordinate_scale", scale, persistent=False)
        if config.support or config.recurrent_type == "gru":
            self.recurrent_proposal = nn.GRUCell(d, d)
        if config.support:
            h = config.auxiliary_hidden
            self.forecast = mlp(3 * d, h, d * len(config.horizons))
            self.forecast_bridge = nn.Linear(d * len(config.horizons), d, bias=False)
            self.history_projection = nn.Linear(4 * d, d, bias=False)
            self.denoiser = mlp(4 * d + 1, h, d)
            self.specialists = nn.ModuleList([mlp(4 * d, h, d) for _ in range(config.specialists)])
            self.bridge = nn.Linear(d, d, bias=False)
            self.specialist_links = nn.ModuleList([nn.Linear(d, d, bias=False)
                                                  for _ in range(config.specialists)])
            self.candidate_names = ("feedforward", "forecast", "diffusion") + tuple(
                f"specialist_{i}" for i in range(config.specialists))
            self.router = nn.Linear(3 * d, len(self.candidate_names))
            self.critic = mlp(4 * d, h, 1)
        if config.context_kernel > 1:
            self.context_convolution = nn.Conv1d(d, d, config.context_kernel, bias=False)
            self.context_norm = nn.LayerNorm(d)
        self.identity = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compile_refinement(self):
        """Opt in to compilation of the complete specialized refinement region.

        Compilation is lazy and its first-call cost must be measured separately.
        The reference backend and all interventions remain available.
        """
        if self.config.refinement_backend != "krylov2":
            raise ValueError("compile_refinement requires refinement_backend='krylov2'")
        from .kernels import krylov2_refine
        self._compiled_refinement = torch.compile(krylov2_refine, fullgraph=True, dynamic=False)
        return self

    def initial_state(self, batch: int, *, device=None, dtype=None):
        p = self.embedding.weight
        return EnergyState(torch.zeros(batch, self.dimensions, device=device or p.device,
                                       dtype=dtype or p.dtype),
                           torch.zeros(batch, self.observed_capacity, self.config.width,
                                       device=device or p.device, dtype=dtype or p.dtype),
                           0, self.identity)

    @property
    def observed_capacity(self):
        # The extra raw embeddings provide the oldest attended key's causal
        # convolution context, including across incremental forward calls.
        return self.config.history + self.config.context_kernel - 1

    def encode_observed(self, observed: Tensor):
        if self.config.context_kernel == 1:
            return observed
        raw = F.pad(observed.transpose(1, 2), (self.config.context_kernel - 1, 0))
        return self.context_norm(F.silu(self.context_convolution(raw).transpose(1, 2)))

    def observed_bank(self, tokens: Tensor, state: EnergyState):
        """Encode supplied observations once, retaining their training graph.

        The caller slices each attention window at its current byte. The bank
        is local to this forward call and never persists across weight updates.
        """
        return self.encode_observed(torch.cat((state.observed, self.embedding(tokens)), 1))

    def encode_current(self, token: Tensor, state: EnergyState):
        """Encode one new byte using only its causal convolution tail."""
        tail = self.config.context_kernel - 1
        observed = torch.cat((state.observed[:, -tail:] if tail else state.observed[:, :0],
                              self.embedding(token)[:, None]), 1)
        return self.encode_observed(observed)[:, -1]

    def retrieve(self, query: Tensor, keys: Tensor, values: Tensor):
        scores = (keys * query[:, None]).sum(-1) / math.sqrt(self.config.width)
        return (scores.softmax(-1)[..., None] * values).sum(1)

    def recurrent_condition(self, state: EnergyState):
        previous = state.joint[:, self.blocks["recurrent"]]
        # GRU retention needs an identity path inside its bounded state domain.
        # This bounds conditioning only; accepted joint states are untouched.
        return previous.clamp(-1., 1.) if self.config.recurrent_type == "gru" else previous.tanh()

    def frequencies(self):
        # Bound each *vector's norm*, preserving mixed coordinate directions.
        raw = self.canonical_state(self.raw_frequency * self.mode_mask * self.coordinate_scale)
        norm = raw.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return raw * (self.config.frequency_bound * torch.tanh(norm) / norm)

    def output_coordinates(self, logits: Tensor):
        """Remove the softmax-invariant common offset when using the new geometry."""
        return logits if self.config.output_geometry == "raw" else logits - logits.mean(-1, keepdim=True)

    def canonical_state(self, s: Tensor):
        if self.config.output_geometry == "raw":
            return s
        cut = self.blocks["output"].start
        return torch.cat((s[..., :cut], self.output_coordinates(s[..., cut:])), -1)

    def _decoder_latent(self, s: Tensor):
        names = ("recurrent", "attention", "feedforward")
        if self.config.support:
            names += ("communication",)
        return torch.stack([s[..., self.blocks[name]] for name in names], -2).mean(-2)

    def _candidate_states(self, s: Tensor):
        values = []
        for name in self.candidate_names:
            value = s[..., self.blocks[name]]
            if name == "forecast":
                value = self.forecast_bridge(value)
            values.append(value)
        return torch.stack(values, -2)

    def condition(self, token: Tensor, state: EnergyState, *, disabled=(), constants=None,
                  attention_context=None, encoded_current=None) -> tuple[Conditioning, Tensor]:
        """Uses supplied input bytes only; no labels, utilities or future targets."""
        c, d = self.config, self.config.width
        observed = torch.cat((state.observed[:, 1:], self.embedding(token)[:, None]), 1)
        available = min(state.count + 1, c.history)
        if encoded_current is None or attention_context is None:
            encoded = self.encode_observed(observed)
            memory = encoded[:, -available:]
            current = encoded[:, -1]
        else:
            current = encoded_current
        carry = self.recurrent_condition(state)
        if attention_context is None:
            query = self.query(current + carry if c.context_kernel > 1 else current)
            attention_scores = (self.key(memory) * query[:, None]).sum(-1) / math.sqrt(d)
            attn = (attention_scores.softmax(-1)[..., None] * self.value(memory)).sum(1)
        else:
            attn = attention_context
        # Bounded *conditioning* prevents a feedforward proposal's learned
        # recurrent gain from growing exponentially across a 128-byte window.
        # This does not project or edit an accepted state after refinement.
        features = torch.cat((current, carry, attn), -1)
        if c.support or c.recurrent_type == "gru":
            feedforward = self.prior(features)
            recurrent = self.recurrent_proposal(current, carry) if not c.support else carry
        else:
            recurrent, feedforward = self.prior(features).chunk(2, -1)
        extras = {}
        routing = None
        pieces = [recurrent, attn, feedforward]
        if c.support:
            # Only observed input embeddings enter this FFT history. Candidate
            # forecasts/diffusion proposals never become observed history.
            spectrum = torch.fft.rfft(observed, dim=1, norm="ortho")[:, 1:3]
            if spectrum.shape[1] < 2:
                spectrum = F.pad(spectrum, (0, 0, 0, 2 - spectrum.shape[1]))
            spectral = self.history_projection(torch.cat((spectrum.real, spectrum.imag), 1).flatten(1))
            if "history" in disabled:
                spectral = spectral * 0
            recurrent = self.recurrent_proposal(current + spectral, carry)
            features = torch.cat((current + spectral, carry, attn), -1)
            forecast = (current.new_zeros(len(token), d * len(c.horizons))
                        if "forecast" in disabled else self.forecast(features))
            sigma = current.new_full((len(token), 1), 0.1)
            diffusion = (torch.zeros_like(feedforward) if "diffusion" in disabled else
                         feedforward - sigma * self.denoiser(torch.cat((features, feedforward, sigma.log()), -1)))
            message = (feedforward + self.forecast_bridge(forecast) + diffusion) / 3
            specialist_values = [(torch.zeros_like(feedforward) if "specialists" in disabled else
                                 net(torch.cat((features, self.bridge(message)), -1)))
                                 for net in self.specialists]
            candidates = torch.stack([feedforward, self.forecast_bridge(forecast), diffusion,
                                      *specialist_values], 1)
            expanded = features[:, None].expand(-1, len(self.candidate_names), -1)
            critic = (current.new_zeros(len(token), len(self.candidate_names)) if "critic" in disabled
                      else self.critic(torch.cat((expanded, candidates), -1)).squeeze(-1))
            router_logits = (current.new_zeros(len(token), len(self.candidate_names)) if "router" in disabled
                             else self.router(features))
            excluded = torch.tensor([name in disabled or ("specialists" in disabled and
                                     name.startswith("specialist_")) for name in self.candidate_names],
                                    dtype=torch.bool, device=current.device)
            routing = (router_logits - critic).masked_fill(excluded[None], -torch.inf).softmax(-1)
            communication = (torch.zeros_like(feedforward) if "communication" in disabled else
                             self.bridge((routing[..., None] * candidates).sum(1)))
            pieces = [recurrent, attn, feedforward, forecast, diffusion,
                      *specialist_values, communication]
            extras = dict(critic=critic, router_logits=router_logits,
                          candidate_logits=self.decoder(candidates), clean=feedforward)
        output = self.output_prior(features) + self.decoder(torch.stack(
            [pieces[0], pieces[1], pieces[2]], 1).mean(1))
        anchor = self.canonical_state(torch.cat((*pieces, output), -1))
        amplitudes = 0.5 * torch.tanh(self.amplitude(features)) / math.sqrt(c.modes)
        frequency, precision = constants if constants is not None else (
            self.frequencies(), 0.15 + 1.85 * self.raw_precision.sigmoid())
        return Conditioning(anchor, amplitudes, routing, features, extras, frequency, precision), observed

    def compatibility_state(self, s: Tensor, cond: Conditioning):
        return s - cond.anchor if self.config.compatibility_geometry == "anchor_relative" else s

    def compatibility_decoder(self, state: Tensor):
        bias = None if self.config.compatibility_geometry == "anchor_relative" else self.decoder.bias
        return self.output_coordinates(F.linear(self._decoder_latent(state), self.decoder.weight, bias))

    def quadratic(self, s: Tensor, cond: Conditioning) -> Tensor:
        c = self.config
        s = self.canonical_state(s)
        precision = cond.precision if cond.precision is not None else 0.15 + 1.85 * self.raw_precision.sigmoid()
        result = 0.5 * (precision * (s - cond.anchor).square()).sum(-1)
        compatible = self.compatibility_state(s, cond)
        residual = compatible[..., self.blocks["recurrent"]] - compatible[..., self.blocks["attention"]]
        result = result + 0.5 * c.attention_weight * residual.square().sum(-1)
        residual = compatible[..., self.blocks["output"]] - self.compatibility_decoder(compatible)
        result = result + 0.5 * c.decoder_weight * residual.square().sum(-1)
        if c.support:
            message = self.bridge((cond.routing[..., None] * self._candidate_states(compatible)).sum(-2))
            communication = compatible[..., self.blocks["communication"]]
            result = result + 0.5 * c.communication_weight * (communication - message).square().sum(-1)
            for i, link in enumerate(self.specialist_links):
                residual = compatible[..., self.blocks[f"specialist_{i}"]] - link(communication)
                result = result + 0.5 * c.specialist_weight * residual.square().sum(-1)
        return result

    def energy(self, s: Tensor, cond: Conditioning) -> Tensor:
        s = self.canonical_state(s)
        frequency = cond.frequency if cond.frequency is not None else self.frequencies()
        wave_state = s - cond.anchor if self.config.wave_geometry == "anchor_relative" else s
        projected = F.linear(wave_state, frequency, self.phase)
        waves = projected.cos() if self.config.activation == "fourier" else projected.tanh()
        return self.quadratic(s, cond) + (cond.amplitudes * waves).sum(-1)

    def gradient(self, s: Tensor, cond: Conditioning, *, mode_mask: Tensor | None = None) -> Tensor:
        """Exact gradient of energy; an optional mode mask affects proposals only."""
        c = self.config
        s = self.canonical_state(s)
        precision = cond.precision if cond.precision is not None else 0.15 + 1.85 * self.raw_precision.sigmoid()
        parts = {name: precision[sl]
                 * (s[..., sl] - cond.anchor[..., sl]) for name, sl in self.blocks.items()}
        compatible = self.compatibility_state(s, cond)
        residual = compatible[..., self.blocks["recurrent"]] - compatible[..., self.blocks["attention"]]
        parts["recurrent"] = parts["recurrent"] + c.attention_weight * residual
        parts["attention"] = parts["attention"] - c.attention_weight * residual
        residual = compatible[..., self.blocks["output"]] - self.compatibility_decoder(compatible)
        parts["output"] = parts["output"] + c.decoder_weight * residual
        latent_grad = c.decoder_weight * (residual @ self.decoder.weight)
        names = ("recurrent", "attention", "feedforward") + (("communication",) if c.support else ())
        for name in names:
            parts[name] = parts[name] - latent_grad / len(names)
        if c.support:
            message = self.bridge((cond.routing[..., None] * self._candidate_states(compatible)).sum(-2))
            communication = compatible[..., self.blocks["communication"]]
            residual = communication - message
            parts["communication"] = parts["communication"] + c.communication_weight * residual
            candidate_grad = -c.communication_weight * (residual @ self.bridge.weight)
            for j, name in enumerate(self.candidate_names):
                contribution = cond.routing[..., j, None] * candidate_grad
                if name == "forecast":
                    contribution = contribution @ self.forecast_bridge.weight
                parts[name] = parts[name] + contribution
            for i, link in enumerate(self.specialist_links):
                name = f"specialist_{i}"
                residual = compatible[..., self.blocks[name]] - link(communication)
                parts[name] = parts[name] + c.specialist_weight * residual
                parts["communication"] = parts["communication"] - c.specialist_weight * (residual @ link.weight)
        quadratic_grad = torch.cat(tuple(parts.values()), -1)
        frequency = cond.frequency if cond.frequency is not None else self.frequencies()
        wave_state = s - cond.anchor if c.wave_geometry == "anchor_relative" else s
        projected = F.linear(wave_state, frequency, self.phase)
        derivative = -projected.sin() if c.activation == "fourier" else 1 - projected.tanh().square()
        coefficients = cond.amplitudes * derivative
        if mode_mask is not None:
            coefficients = coefficients * mode_mask
        return self.canonical_state(quadratic_grad + coefficients @ frequency)

    def refine(self, s: Tensor, cond: Conditioning, *, rounds: int | None = None,
               update_states: bool = True, update_output: bool = True,
               trace: bool = False, active_mask: Tensor | None = None):
        c = self.config
        rounds = c.rounds if rounds is None else rounds
        if type(rounds) is not int or rounds < 0:
            raise ValueError("rounds must be nonnegative")
        # This is the optimization domain, before any energy comparison. There
        # is no post-acceptance projection under a different energy function.
        s = self.canonical_state(s)
        metrics = dict(gradient_calls=0, energy_calls=0, accepted=0, attempts=0)
        energies = []
        halted = torch.zeros(s.shape[:-1], dtype=torch.bool, device=s.device)
        # No in-place writes: differentiation through accepted updates is intact.
        update_mask = s.new_ones(self.dimensions)
        if not update_states:
            update_mask[:self.blocks["output"].start] = 0
        if not update_output:
            update_mask[self.blocks["output"]] = 0
        if active_mask is not None:
            update_mask = update_mask * active_mask
        for r in range(rounds):
            before = self.energy(s, cond)
            metrics["energy_calls"] += 1
            if trace and not energies:
                energies.append(before.detach())
            mode_mask = None
            if c.schedule == "coarse_to_fine":
                n = max(1, math.ceil(c.modes * (r + 1) / max(rounds, 1)))
                frequency = cond.frequency if cond.frequency is not None else self.frequencies()
                ordered = frequency.detach().square().sum(-1).argsort()
                mode_mask = s.new_zeros(c.modes)
                mode_mask[ordered[:n]] = 1
            gradient = self.gradient(s, cond, mode_mask=mode_mask) * update_mask
            if c.halting_tolerance:
                gradient = gradient * (~halted)[..., None]
            metrics["gradient_calls"] += 1
            chosen = s
            chosen_energy = before
            accepted = torch.zeros(s.shape[:-1], dtype=torch.bool, device=s.device)
            # All attempts use the *original* gradient, and all tests use the
            # complete energy, including modes excluded from a coarse proposal.
            for k in range(c.backtracks):
                candidate = s - (c.step_size * 0.5 ** k) * gradient
                after = self.energy(candidate, cond)
                ok = torch.isfinite(after) & (after < before) & ~accepted
                chosen = torch.where(ok[..., None], candidate, chosen)
                chosen_energy = torch.where(ok, after, chosen_energy)
                accepted = accepted | ok
                metrics["energy_calls"] += 1
                metrics["attempts"] += int(s.shape[0])
                if c.early_backtrack and s.device.type == "cpu" and bool((accepted | halted).all()):
                    break
            if c.halting_tolerance and r + 1 >= c.minimum_refinement_rounds:
                relative_drop = (before - chosen_energy) / before.abs().clamp_min(1.)
                probability_change = (chosen[..., self.blocks["output"]].softmax(-1)
                                      - s[..., self.blocks["output"]].softmax(-1)).abs().sum(-1)
                halted = halted | ((relative_drop <= c.halting_tolerance)
                                   & (probability_change <= c.halting_tolerance))
            s = chosen
            if trace:
                metrics["accepted"] += int(accepted.sum().item())
                energies.append(self.energy(s, cond).detach())
                metrics["energy_calls"] += 1
            if c.halting_tolerance and s.device.type == "cpu" and bool(halted.all()):
                # A flat state is a stopping feature, never a correctness claim.
                # Pad diagnostic trajectories with the retained state so their
                # declared maximum-round shape is unchanged across positions.
                if trace:
                    energies.extend([energies[-1]] * (rounds-r-1))
                break
        if trace and not energies:
            energies = [self.energy(s, cond).detach()]
            metrics["energy_calls"] += 1
        return s, metrics, energies

    def diffusion_proposals(self, s: Tensor, cond: Conditioning, noise: Tensor):
        """Annealed denoising initialization, selected by the same full energy.

        Reverse denoising is a candidate generator, not an accepted side-channel
        state update. Candidate selection includes KEEP and evaluates full E.
        """
        c = self.config
        energies = [self.energy(s, cond)]
        candidates = [s]
        sl = self.blocks["diffusion"]
        for i in range(c.diffusion_samples):
            z = cond.extras["clean"] + noise[:, i] * 0.3
            for sigma_value in torch.logspace(math.log10(0.3), math.log10(0.03), c.diffusion_levels,
                                               device=s.device, dtype=s.dtype):
                sigma = sigma_value.expand(len(s), 1)
                estimate = self.denoiser(torch.cat((cond.features, z, sigma.log()), -1))
                z = z - sigma * estimate
            candidate = torch.cat((s[:, :sl.start], z, s[:, sl.stop:]), -1)
            candidates.append(candidate)
            energies.append(self.energy(candidate, cond))
        bank = torch.stack(candidates, 1)
        scores = torch.stack(energies, 1)
        index = scores.detach().argmin(1)
        selected = bank[torch.arange(len(s), device=s.device), index]
        return selected, c.diffusion_samples + 1

    def forward(self, tokens: Tensor, state: EnergyState | None = None, *, rounds: int | None = None,
                noise: Tensor | None = None, details: bool = False,
                update_states: bool = True, update_output: bool = True,
                disabled: tuple[str, ...] = (), initial_noise: Tensor | None = None,
                trace_energy: bool | None = None, state_gradient_interval: int = 0) -> Output:
        if type(state_gradient_interval) is not int or state_gradient_interval < 0:
            raise ValueError("state_gradient_interval must be a nonnegative integer")
        if rounds is not None and (type(rounds) is not int or rounds < 0):
            raise ValueError("rounds must be nonnegative")
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 1:
            raise ValueError("tokens must be nonempty [batch,time] int64")
        if bool((tokens < 0).any()) or bool((tokens > 255).any()):
            raise ValueError("Bytes must be in 0..255")
        state = state or self.initial_state(len(tokens), device=tokens.device)
        if state.identity != self.identity or state.joint.shape != (len(tokens), self.dimensions):
            raise ValueError("Carried state belongs to a different configuration or batch")
        if state.observed.shape != (len(tokens), self.observed_capacity, self.config.width):
            raise ValueError("Invalid observed history shape")
        if noise is not None and noise.shape != (len(tokens), tokens.shape[1],
                                                 self.config.diffusion_samples, self.config.width):
            raise ValueError("Explicit diffusion noise must be [B,T,samples,width]")
        if initial_noise is not None and initial_noise.shape != (len(tokens), tokens.shape[1], self.dimensions):
            raise ValueError("Initialization noise must be [B,T,joint_dimensions]")
        unknown = set(disabled) - {"forecast", "diffusion", "specialists", "communication", "critic", "router", "history"}
        if unknown:
            raise ValueError(f"Unknown interventions: {unknown}")
        if disabled and not self.config.support:
            raise ValueError("Supporting-module interventions require the full model")
        trace_energy = details if trace_energy is None else trace_energy
        if trace_energy and not details:
            raise ValueError("Energy traces require details=True")
        logits, initial_logits, initial_values, joint_values, feature_values, conditions, extras, traces = [], [], [], [], [], [], [], []
        total = dict(gradient_calls=0, energy_calls=0, accepted=0, attempts=0)
        # Intervention masks hold a removed workspace at zero. They are used
        # only for explicit ablations, never silently in ordinary inference.
        disabled_blocks = tuple(name for name in self.blocks if name in disabled
                                or ("specialists" in disabled and name.startswith("specialist_")))
        mask = self.embedding.weight.new_ones(self.dimensions)
        for name in disabled_blocks:
            mask[self.blocks[name]] = 0
        # Model weights do not change inside this call. Cache only these
        # input-independent tensors, retaining their training graph. No cache
        # survives an optimizer step or an independent/chunked forward call.
        constants = (self.frequencies(), 0.15 + 1.85 * self.raw_precision.sigmoid())
        effective_rounds = self.config.rounds if rounds is None else rounds
        use_krylov2 = (self.config.refinement_backend == "krylov2" and effective_rounds == 2
                       and update_states and update_output and initial_noise is None and not disabled)
        if use_krylov2:
            from .kernels import krylov2_refine, prepare_krylov2
            krylov_context = prepare_krylov2(self, *constants)
            refinement_kernel = getattr(self, '_compiled_refinement', krylov2_refine)
        cached_keys = cached_values = cached_queries = cached_inputs = None
        if self.config.cache_history:
            bank = self.observed_bank(tokens, state)
            cached_keys, cached_values = self.key(bank), self.value(bank)
            cached_inputs = bank[:, self.observed_capacity:]
            if self.config.context_kernel == 1:
                cached_queries = self.query(cached_inputs)
        for t in range(tokens.shape[1]):
            if self.training and state_gradient_interval and t and t % state_gradient_interval == 0:
                # Truncate temporal credit only. Observed embeddings and every
                # carried state value are retained; no forward memory is reset.
                state = EnergyState(state.joint.detach(),state.observed,state.count,state.identity)
            attention_context = None
            if cached_keys is not None:
                end = self.observed_capacity + t + 1
                available = min(state.count + 1, self.config.history)
                # Slices end at the currently observed byte. Future keys and
                # values exist in the bank but never participate in attention.
                query = (cached_queries[:, t] if cached_queries is not None else
                         self.query(cached_inputs[:, t] + self.recurrent_condition(state)))
                attention_context = self.retrieve(query, cached_keys[:, end-available:end],
                                                  cached_values[:, end-available:end])
            cond, observed = self.condition(tokens[:, t], state, disabled=disabled, constants=constants,
                attention_context=attention_context,
                encoded_current=(cached_inputs[:, t] if cached_inputs is not None else None))
            # Ablated proposals/routing are removed while constructing cond.
            # That same conditioned energy is frozen for every round.
            initial = cond.anchor + self.config.initial_carry_scale * state.joint.tanh()
            if initial_noise is not None:
                initial = initial + initial_noise[:, t]
            initial = self.canonical_state(initial)
            initial = initial * mask
            if self.config.support and "diffusion" not in disabled:
                token_noise = (noise[:, t] if noise is not None else initial.new_zeros(
                    len(tokens), self.config.diffusion_samples, self.config.width))
                initial, calls = self.diffusion_proposals(initial, cond, token_noise)
                total["energy_calls"] += calls
            # Clamp absent blocks inside the optimization domain. Use a masked
            # gradient, not a post-hoc edit that could raise accepted energy.
            if use_krylov2:
                refined, reduced_trace, accepted = refinement_kernel(
                    cond.anchor, cond.amplitudes, self.phase, constants[0], *krylov_context,
                    output_start=self.blocks['output'].start,
                    centered=self.config.output_geometry == 'centered')
                # Include the selected final energy returned by the kernel,
                # even when the caller does not request a diagnostic trace.
                metrics = dict(gradient_calls=2, energy_calls=3 + 2 * self.config.backtracks,
                               attempts=2 * self.config.backtracks * len(tokens),
                               accepted=int(accepted.sum().item()) if trace_energy else 0)
                energy_trace = list(reduced_trace.detach().unbind(-1)) if trace_energy else []
            else:
                refined, metrics, energy_trace = self.refine(initial, cond, rounds=rounds,
                    update_states=update_states, update_output=update_output, trace=trace_energy, active_mask=mask)
            for key in total:
                total[key] += metrics[key]
            logits.append(refined[:, self.blocks["output"]])
            if details:
                initial_logits.append(initial[:, self.blocks["output"]])
                initial_values.append(initial)
                joint_values.append(refined)
                feature_values.append(cond.features)
                conditions.append(cond)
                extras.append(cond.extras)
                if trace_energy:
                    traces.append(torch.stack(energy_trace, -1))
            state = EnergyState(refined, observed, state.count + 1, self.identity)
        result_details = dict(cost=total, refinement_backend='krylov2' if use_krylov2 else 'reference')
        if details:
            result_details.update(joint=torch.stack(joint_values, 1),
                                  initial_joint=torch.stack(initial_values, 1),
                                  initial_logits=torch.stack(initial_logits, 1),
                                  features=torch.stack(feature_values, 1))
            if trace_energy:
                result_details["energy_trace"] = torch.stack(traces, 1)
            result_details["conditioning"] = Conditioning(
                torch.stack([ctx.anchor for ctx in conditions], 1),
                torch.stack([ctx.amplitudes for ctx in conditions], 1),
                (torch.stack([ctx.routing for ctx in conditions], 1) if self.config.support else None),
                result_details["features"], {}, *constants)
            if self.config.support:
                for key in extras[0]:
                    result_details[key] = torch.stack([e[key] for e in extras], 1)
        return Output(torch.stack(logits, 1), state, result_details)

    def auxiliary_losses(self, output: Output, targets: Tensor, *, valid: Tensor,
                         teacher_hidden: Tensor | None = None, noise: Tensor | None = None):
        """Labels/future teacher targets are accepted only by this training API."""
        if targets.shape != output.logits.shape[:2] or valid.shape != targets.shape or valid.dtype != torch.bool:
            raise ValueError("Targets and Boolean mask must match output positions")
        d = self.config.width
        result = {}
        if not self.config.support:
            return result
        if "joint" not in output.details:
            raise ValueError("Auxiliary training requires details=True")
        if teacher_hidden is not None:
            if teacher_hidden.requires_grad:
                raise ValueError("Frozen teacher targets must be detached")
            if teacher_hidden.shape[:2] != targets.shape or teacher_hidden.shape[-1] < d:
                raise ValueError("Teacher hidden targets need matching time/batch and width >= model width")
            teacher = F.layer_norm(teacher_hidden[..., :d], (d,))
            forecast = output.details["joint"][..., self.blocks["forecast"]].reshape(
                len(targets), targets.shape[1], len(self.config.horizons), d)
            losses = []
            for i, horizon in enumerate(self.config.horizons):
                if horizon >= targets.shape[1]:
                    continue
                usable = valid[:, :-horizon] & valid[:, horizon:]
                if usable.any():
                    losses.append((forecast[:, :-horizon, i][usable]
                                   - teacher[:, horizon:][usable]).square().mean())
            result["forecast"] = torch.stack(losses).mean() if losses else forecast.sum() * 0
        clean = output.details["clean"].detach()
        if noise is None:
            noise = torch.randn_like(clean)
        sigma = clean.new_full((*clean.shape[:-1], 1), 0.15)
        corrupted = clean + sigma * noise
        estimate = self.denoiser(torch.cat((output.details["features"], corrupted, sigma.log()), -1))
        result["denoising"] = (estimate[valid] - noise[valid]).square().mean() if valid.any() else estimate.sum() * 0
        candidate_logp = output.details["candidate_logits"].log_softmax(-1)
        labels = targets.clamp_min(0)[..., None, None].expand(*targets.shape, len(self.candidate_names), 1)
        utilities = candidate_logp.gather(-1, labels).squeeze(-1).detach()
        critic = output.details["critic"]
        differences = critic[..., :, None] - critic[..., None, :]
        ordered = (utilities[..., :, None] > utilities[..., None, :] + 1e-4) & valid[..., None, None]
        result["critic"] = F.softplus(differences[ordered]).mean() if ordered.any() else critic.sum() * 0
        if valid.any():
            result["routing"] = F.cross_entropy(output.details["router_logits"][valid], utilities[valid].argmax(-1))
        else:
            result["routing"] = output.details["router_logits"].sum() * 0
        return result

    def ranking_loss(self, tokens: Tensor, targets: Tensor, output: Output, valid: Tensor,
                     *, hard_negatives: bool = False, generator=None):
        """Training-only comparison of correct and sampled output candidates."""
        if targets.shape != tokens.shape or valid.shape != targets.shape or valid.dtype != torch.bool:
            raise ValueError("Ranking targets and mask must match observed input positions")
        if not valid.any():
            return output.logits.sum() * 0
        if "conditioning" not in output.details:
            raise ValueError("Ranking requires the original prefix conditioning (details=True)")
        cond = output.details["conditioning"]
        base = output.details["joint"].detach()
        true = targets.clamp_min(0)
        negative = (true + 1 + torch.arange(len(tokens), device=tokens.device)[:, None]) % 256
        if hard_negatives:
            draw = torch.randint(255, targets.shape, generator=generator, device=targets.device)
            negative = draw + (draw >= true).long()
        positive_logits = F.one_hot(true, 256).to(base.dtype) * 2
        negative_logits = F.one_hot(negative, 256).to(base.dtype) * 2
        positive = torch.cat((base[..., :self.blocks["output"].start], positive_logits), -1)
        negative = torch.cat((base[..., :self.blocks["output"].start], negative_logits), -1)
        differences = self.energy(positive, cond) - self.energy(negative, cond)
        loss = F.softplus(differences[valid]).mean()
        if hard_negatives:
            scores = output.logits.detach().clone()
            scores.scatter_(-1, true[..., None], -torch.inf)
            hard = scores.argmax(-1)
            hard_logits = F.one_hot(hard, 256).to(base.dtype) * 2
            candidate = torch.cat((base[..., :self.blocks["output"].start], hard_logits), -1)
            loss = 0.5 * (loss + F.softplus((self.energy(positive, cond) - self.energy(candidate, cond))[valid]).mean())
        return loss

    def repair_loss(self, output: Output, targets: Tensor, valid: Tensor):
        """Training-only task-loss feedback for each local refinement trajectory."""
        initial = F.cross_entropy(output.details["initial_logits"].flatten(0, 1), targets.flatten(),
                                  ignore_index=-100, reduction="none").reshape_as(targets)
        final = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten(),
                                ignore_index=-100, reduction="none").reshape_as(targets)
        return F.relu(final[valid] - initial[valid].detach()).mean() if valid.any() else final.sum() * 0

    def process_loss(self, output: Output, targets: Tensor, negative_targets: Tensor):
        """Rank the first incorrect trace byte using its unchanged gold prefix.

        Correctness and corrupted-trace labels enter this training-only loss;
        they are never inputs to condition(), forward(), or inference.
        """
        valid = negative_targets >= 0
        if not valid.any():
            return output.logits.sum() * 0
        base = output.details["joint"].detach()
        cut = self.blocks["output"].start
        positive = torch.cat((base[..., :cut], F.one_hot(targets.clamp_min(0), 256).to(base.dtype) * 2), -1)
        negative = torch.cat((base[..., :cut], F.one_hot(negative_targets.clamp_min(0), 256).to(base.dtype) * 2), -1)
        cond = output.details["conditioning"]
        return F.softplus((self.energy(positive, cond) - self.energy(negative, cond))[valid]).mean()


class TransformerBlock(nn.Module):
    def __init__(self, c: TransformerConfig):
        super().__init__()
        self.heads = c.heads
        self.norm_attention = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.projection = nn.Linear(c.width, c.width)
        self.norm_ff = nn.LayerNorm(c.width)
        self.ff = mlp(c.width, c.feedforward, c.width)

    def forward(self, x: Tensor, *, details=False):
        b, t, d = x.shape
        encoded = self.norm_attention(x)
        q, k, v = self.qkv(encoded).reshape(b, t, 3, self.heads, d // self.heads).unbind(2)
        attention = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                   v.transpose(1, 2), is_causal=True)
        x = x + self.projection(attention.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff(self.norm_ff(x))
        if details:
            return x, dict(attention_query=q, attention_key=k, attention_value=v,
                           attention_inputs=encoded)
        return x


class TransformerLM(nn.Module):
    family = "transformer"

    def __init__(self, config: TransformerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(256, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.decoder = nn.Linear(config.width, 256)

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens: Tensor, state: TransformerState | None = None, *, details=False, **kwargs):
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 1:
            raise ValueError("tokens must be nonempty [batch,time] int64")
        if bool((tokens < 0).any()) or bool((tokens > 255).any()):
            raise ValueError("Bytes must be in 0..255")
        offset = 0
        if state is not None:
            offset = state.tokens.shape[1]
            tokens = torch.cat((state.tokens, tokens), 1)
        if tokens.shape[1] > self.config.context:
            raise ValueError("Context exceeded; reset every model at the declared window boundary")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)[None]
        attention_details = None
        for layer in self.layers:
            if details:
                x, attention_details = layer(x, details=True)
            else:
                x = layer(x)
        hidden = self.norm(x)
        extras = dict(cost=dict(gradient_calls=0, energy_calls=0, accepted=0, attempts=0))
        if details:
            extras["hidden"] = hidden[:, offset:]
            extras.update({key: value[:, offset:] for key, value in attention_details.items()})
        return Output(self.decoder(hidden[:, offset:]), TransformerState(tokens), extras)


def matched_configurations(target=250_000, *, width=48, context=128, modes=16,
                           frequency_bound=2., output_geometry="raw", normalize_frequency_blocks=False,
                           cache_history=False, early_backtrack=False, step_size=0.25):
    """Match *active* parameters using useful feedforward width, not dummy pads."""
    result = {}
    for arm, support, activation in (("neural_energy", False, "mlp"),
                                     ("fourier_rewt", False, "fourier"),
                                     ("full_rewt", True, "fourier")):
        base = EnergyConfig(width=width, hidden=1, support=support, activation=activation, modes=modes,
                            frequency_bound=frequency_bound, output_geometry=output_geometry,
                            normalize_frequency_blocks=normalize_frequency_blocks,
                            cache_history=cache_history, early_backtrack=early_backtrack, step_size=step_size)
        one = JointEnergyLM(base).parameters_count()
        two = JointEnergyLM(replace(base, hidden=2)).parameters_count()
        hidden = max(8, round(1 + (target - one) / (two - one)))
        cfg = replace(base, hidden=hidden)
        count = JointEnergyLM(cfg).parameters_count()
        if count > target * 1.01:
            raise ValueError("Target too small for all supporting modules; increase parameter budget")
        result[arm] = cfg
    choices = []
    for d in (48, 64, 80, 96):
        for layers in (2, 3):
            base = TransformerConfig(width=d, layers=layers, feedforward=1, context=context)
            one = TransformerLM(base).parameters_count()
            two = TransformerLM(replace(base, feedforward=2)).parameters_count()
            ff = round(1 + (target - one) / (two - one))
            if ff < d:
                continue
            cfg = replace(base, feedforward=ff)
            count = TransformerLM(cfg).parameters_count()
            choices.append((abs(count - target), abs(ff / d - 4), cfg))
    if not choices:
        raise ValueError("No comparable Transformer fits the requested parameter budget")
    result["transformer"] = min(choices, key=lambda item: item[:2])[2]
    return result


def build_model(arm: str, config: EnergyConfig | TransformerConfig):
    if arm == "transformer":
        return TransformerLM(config)
    if arm in ("neural_energy", "fourier_rewt", "full_rewt"):
        return JointEnergyLM(config)
    raise ValueError(f"Unknown arm {arm}")
