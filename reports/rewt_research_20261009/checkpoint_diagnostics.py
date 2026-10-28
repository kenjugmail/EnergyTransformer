"""Read-only validation probes for the completed REWT pilot.

No parameter updates, checkpoint selection, or test-split evaluation occur here.
Every refinement sweep freezes each original two-round prefix's conditioning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rewt.data import Corpus
from rewt.evaluate import noise_for
from rewt.experiment import load_model
from rewt.model import Conditioning


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(value):
    value = value.detach().flatten().double()
    return dict(mean=float(value.mean()), median=float(value.median()),
                minimum=float(value.min()), maximum=float(value.max()))


def gradient_diagnostics(model, states, cond, targets):
    frequency = cond.frequency
    phases = F.linear(states, frequency, model.phase)
    wave_gradient = (cond.amplitudes * -phases.sin()) @ frequency
    total_gradient = model.gradient(states, cond)
    quadratic_gradient = total_gradient - wave_gradient
    output = model.blocks["output"]
    ce_gradient = states[:, output].softmax(-1) - F.one_hot(targets, 256)
    output_gradient = total_gradient[:, output]
    alignment = (ce_gradient * output_gradient).sum(-1)
    cosine = alignment / (ce_gradient.norm(dim=-1) * output_gradient.norm(dim=-1)).clamp_min(1e-12)
    common_mode_fraction = (256 * output_gradient.mean(-1).square()
                            / output_gradient.square().sum(-1).clamp_min(1e-12))
    blocks = {}
    for name, sl in model.blocks.items():
        q_norm = quadratic_gradient[:, sl].norm(dim=-1)
        w_norm = wave_gradient[:, sl].norm(dim=-1)
        blocks[name] = dict(quadratic_norm=stats(q_norm), wave_norm=stats(w_norm),
                           wave_to_quadratic_norm=stats(w_norm / q_norm.clamp_min(1e-12)))
    return dict(blocks=blocks, output_ce_gradient_cosine=stats(cosine),
                fraction_first_order_ce_descent=float((alignment > 0).float().mean()),
                output_common_shift_gradient_fraction=stats(common_mode_fraction))


@torch.no_grad()
def inspect(model, batch, seed):
    model.eval()
    generator = torch.Generator().manual_seed(581)
    noise = noise_for(model, batch.tokens, generator)
    result = model(batch.tokens, noise=noise, details=True)
    full = result.details["conditioning"]
    states = result.details["joint"].reshape(-1, model.dimensions)
    targets = batch.targets.flatten()
    cond = Conditioning(full.anchor.reshape_as(states),
                        full.amplitudes.reshape(-1, model.config.modes),
                        full.routing.reshape(-1, 6) if full.routing is not None else None,
                        full.features.reshape(-1, 3 * model.config.width), {},
                        full.frequency, full.precision)
    if model.config.support:
        cond.extras["clean"] = result.details["clean"].reshape(-1, model.config.width)
    previous = torch.cat((torch.zeros_like(result.details["joint"][:, :1]),
                          result.details["joint"][:, :-1]), 1).reshape_as(states)
    initial = cond.anchor + 0.05 * previous.tanh()
    if model.config.support:
        initial, _ = model.diffusion_proposals(initial, cond,
            noise.reshape(-1, model.config.diffusion_samples, model.config.width))
    output = model.blocks["output"]
    reconstructed, _, _ = model.refine(initial, cond, rounds=2)
    reconstruction_error = float((states - reconstructed).abs().max())
    if reconstruction_error > 1e-4:
        raise AssertionError(f"Frozen-prefix reconstruction error {reconstruction_error}")
    rounds = []
    for count in (0, 1, 2, 4, 8):
        refined, _, _ = model.refine(initial, cond, rounds=count)
        logp = refined[:, output].log_softmax(-1)
        loss = -logp.gather(-1, targets[:, None]).squeeze(-1)
        base_loss = F.cross_entropy(initial[:, output], targets, reduction="none")
        probability_delta = (refined[:, output].softmax(-1) - initial[:, output].softmax(-1)).abs().sum(-1)
        rounds.append(dict(rounds=count, bpb=float(loss.mean() / torch.log(torch.tensor(2.))),
            next_byte_accuracy=float((logp.argmax(-1) == targets).float().mean()),
            fraction_positions_nll_improved=float((loss < base_loss).float().mean()),
            mean_probability_l1_change=float(probability_delta.mean()),
            mean_energy=float(model.energy(refined, cond).mean())))
    logits = states[:, output]
    wrong_logits = logits.clone()
    wrong_logits.scatter_(1, targets[:, None], -torch.inf)
    hard_wrong = wrong_logits.argmax(-1)
    random_wrong = (targets + 1 + torch.arange(len(targets)) % 255) % 256

    def output_candidate_energy(labels):
        candidate = torch.cat((states[:, :output.start], F.one_hot(labels, 256).to(states.dtype) * 2), -1)
        return model.energy(candidate, cond)

    correct_energy = output_candidate_energy(targets)
    hard_margin = output_candidate_energy(hard_wrong) - correct_energy
    random_margin = output_candidate_energy(random_wrong) - correct_energy
    predicted_wrong = logits.argmax(-1) != targets
    shifted = states.clone()
    shifted[:, output] += 1
    gauge_probe = dict(logit_offset=1.,
        max_probability_change=float((shifted[:, output].softmax(-1)
                                      - logits.softmax(-1)).abs().max()),
        mean_energy_change=float((model.energy(shifted, cond) - model.energy(states, cond)).mean()))
    waves = (cond.amplitudes * F.linear(states, cond.frequency, model.phase).cos()).sum(-1)
    initial_waves = (cond.amplitudes * F.linear(initial, cond.frequency, model.phase).cos()).sum(-1)
    return dict(seed=seed, positions=len(targets), reconstruction_max_error=reconstruction_error,
        frozen_prefix_rounds=rounds,
        initial_gradient=gradient_diagnostics(model, initial, cond, targets),
        final_gradient=gradient_diagnostics(model, states, cond, targets),
        wave_energy=stats(waves), quadratic_energy=stats(model.quadratic(states, cond)),
        initial_wave_energy=stats(initial_waves), initial_quadratic_energy=stats(model.quadratic(initial, cond)),
        amplitudes=stats(cond.amplitudes.abs()), frequency_norms=stats(cond.frequency.norm(dim=-1)),
        correct_below_hard_wrong_fraction=float((hard_margin > 0).float().mean()),
        correct_below_hard_wrong_when_prediction_wrong=float((hard_margin[predicted_wrong] > 0).float().mean()),
        correct_below_random_wrong_fraction=float((random_margin > 0).float().mean()),
        common_logit_shift_probe=gauge_probe,
        hard_wrong_energy_margin=stats(hard_margin), random_wrong_energy_margin=stats(random_margin))


def main():
    torch.set_num_threads(1)
    pilot = ROOT / "reports/rewt_pilot_20261009_v2"
    plan = json.loads((pilot / "plan.json").read_text())
    corpus = Corpus(plan["corpus"]["path"])
    offsets = plan["validation_offsets"]
    batch = corpus.batch("valid", None, len(offsets), plan["settings"]["context"], starts=offsets)
    rows = []
    checkpoint_hashes = {}
    for seed in plan["settings"]["seeds"]:
        for arm in ("fourier_rewt", "full_rewt"):
            path = ROOT / "runs/rewt_pilot_20261009_v2" / f"seed_{seed}" / arm / "language/best.pt"
            checkpoint_hashes[str(path.relative_to(ROOT))] = digest(path)
            model, checkpoint = load_model(path)
            row = inspect(model, batch, seed)
            row.update(arm=arm, selected_training_step=checkpoint["step"])
            rows.append(row)
            print(json.dumps(dict(seed=seed, arm=arm, rounds=row["frozen_prefix_rounds"],
                hard_rank=row["correct_below_hard_wrong_fraction"], random_rank=row["correct_below_random_wrong_fraction"],
                wave_norm=row["initial_gradient"]["blocks"]["output"]["wave_norm"],
                quadratic_norm=row["initial_gradient"]["blocks"]["output"]["quadratic_norm"])), flush=True)
    for relative, expected in checkpoint_hashes.items():
        if digest(ROOT / relative) != expected:
            raise AssertionError("A checkpoint changed during the read-only probe")
    saved = dict(protocol="Exploratory validation-only read-only checkpoint diagnostics",
        validation_offsets=offsets, positions_per_model=len(offsets) * plan["settings"]["context"],
        no_weight_updates=True, test_data_used=False,
        round_sweep_scope="Conditioning and previous state fixed from each original two-round prefix; not a new full-model rollout",
        source_sha256=plan["source_sha256"], checkpoint_sha256=checkpoint_hashes,
        diagnostics_source_sha256=digest(__file__), rows=rows)
    (Path(__file__).parent / "checkpoint_diagnostics.json").write_text(
        json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
