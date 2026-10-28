"""Read-only postprocessing of the frozen pilot; never selects model weights."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from rewt.tasks import Task, score_response


ROOT = Path(__file__).resolve().parent
FIELDS = ("valid_format", "answer_correct", "steps_correct", "both_correct")
LABELS = dict(transformer="Transformer", neural_energy="Neural energy",
              fourier_rewt="Fourier REWT", full_rewt="Full REWT")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(rows):
    return dict(count=len(rows), **{field: sum(r[field] for r in rows) / max(1, len(rows)) for field in FIELDS})


def summarize(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values)>1 else 0.,
                values=values)


def analyze():
    plan = json.loads((ROOT / "plan.json").read_text())
    train = [Task(**row) for row in read_lines(ROOT / "tasks_train.jsonl")]
    test = [Task(**row) for row in read_lines(ROOT / "tasks_test.jsonl")]
    # These constants are chosen from TRAINING responses only. The family-aware
    # comparator receives an oracle family label and is a diagnostic upper prior.
    constant = Counter(t.response for t in train).most_common(1)[0][0]
    by_family = {family: Counter(t.response for t in train if t.family == family).most_common(1)[0][0]
                 for family in ("arithmetic", "rules", "recall")}
    prior_rows = {name: [dict(id=t.id, family=t.family, **score_response(t, predictor(t))) for t in test]
                  for name, predictor in (("unconditional_constant", lambda t: constant),
                      ("oracle_family_constant", lambda t: by_family[t.family]))}
    priors = {name: dict(all=stats(rows), by_family={f:stats([r for r in rows if r["family"]==f])
               for f in ("arithmetic", "rules", "recall")}) for name, rows in prior_rows.items()}
    priors["responses"] = dict(unconditional_constant=constant, oracle_family_constant=by_family)
    (ROOT / "constant_priors.json").write_text(json.dumps(priors, indent=2) + "\n")
    breakdown, rows, reasoning_only = {}, [], defaultdict(list)
    for seed in plan["settings"]["seeds"]:
        breakdown[str(seed)] = {}
        for arm in LABELS:
            metrics = json.loads((ROOT / f"seed_{seed}" / f"{arm}.json").read_text())
            records = read_lines(ROOT / f"seed_{seed}" / f"{arm}_reasoning.jsonl")
            family_steps = {family:{str(n):stats([r for r in records if r["family"]==family and r["steps"]==n])
                          for n in sorted({r["steps"] for r in records if r["family"]==family})}
                           for family in ("arithmetic", "rules", "recall")}
            breakdown[str(seed)][arm] = dict(by_family_and_steps=family_steps,
                metric_note="For recall, steps means distractor delay; arithmetic/rules steps means chain length.")
            reasoning_only[arm].append(dict(seed=seed, **stats([r for r in records if r["family"]!="recall"])))
            rows.append(dict(metrics))
    (ROOT / "reasoning_breakdown.json").write_text(json.dumps(breakdown, indent=2, sort_keys=True) + "\n")
    (ROOT / "reasoning_only_summary.json").write_text(json.dumps(dict(reasoning_only), indent=2, sort_keys=True) + "\n")
    aggregation = json.loads((ROOT / "aggregate.json").read_text())
    reference = {row["seed"]:row for row in rows if row["arm"] == "transformer"}
    paired = {arm: summarize([row["language"]["bpb"]-reference[row["seed"]]["language"]["bpb"]
                             for row in rows if row["arm"] == arm]) for arm in LABELS}
    (ROOT / "paired_bpb_differences.json").write_text(json.dumps(paired, indent=2) + "\n")
    inference = json.loads((ROOT / "ablations" / "inference_interventions.json").read_text())
    retraining = json.loads((ROOT / "ablations" / "retraining.json").read_text())
    # Masked direction coordinates do not affect the separate-block energy,
    # even though the dense storage tensor still has requires_grad=True.
    import torch
    effective = {}
    for name, row in retraining.items():
        checkpoint = torch.load(row["training"]["checkpoint"], map_location="cpu", weights_only=True)
        inactive = (checkpoint["weights"]["mode_mask"].numel()
                    - int(checkpoint["weights"]["mode_mask"].count_nonzero())
                    if checkpoint["config"]["learned_frequencies"] else 0)
        effective[name] = dict(allocated_trainable_parameters=row["training"]["parameters"],
            inactive_direction_coordinates=inactive,
            functional_parameter_count=row["training"]["parameters"]-inactive)
    (ROOT / "ablations" / "effective_parameter_counts.json").write_text(json.dumps(effective, indent=2)+"\n")
    lines = ["# Diagnostic evidence from the REWT pilot", "",
        "This analysis reads saved outputs and checkpoints only. It does not tune settings or change model selection.",
        "", "The raw experiment report contains the primary prediction/text/reasoning table. This companion "
        "separates paired-seed comparisons, constant-answer priors, and the shorter component protocols.", "",
        "| Energy model | Paired test-bpb difference vs Transformer ↓ |",
        "|---|---:|"]
    for arm in ("neural_energy", "fourier_rewt", "full_rewt"):
        v = paired[arm]
        lines.append(f"| {LABELS[arm]} | {v['mean']:+.4f} ± {v['std']:.4f} |")
    lines += ["", "Differences use identical held-out windows. Three-seed variation is descriptive, "
        "not a strong significance or scaling result.", "", "## Trivial reasoning references", "",
        "A constant response is selected only from training examples. The family-aware variant additionally "
        "receives the true task family, so it is an oracle-label diagnostic rather than a fair neural baseline.",
        "", "| Predictor | Exact answers % | Correct answers + complete steps % |", "|---|---:|---:|"]
    for name in ("unconditional_constant", "oracle_family_constant"):
        v = priors[name]["all"]
        lines.append(f"| {name} | {v['answer_correct']*100:.2f} | {v['both_correct']*100:.2f} |")
    lines += ["", "Accuracy by chain length is saved separately for arithmetic and rules. Recall is grouped by "
        "distractor delay rather than described as multi-step reasoning.", "",
        "## Arithmetic and rule reasoning, excluding recall", "",
        "| Model | Valid response format % | Exact answers % | Correct answers + complete steps % |",
        "|---|---:|---:|---:|"]
    for arm in LABELS:
        selected = reasoning_only[arm]
        def percent(field):
            v = summarize([r[field] for r in selected])
            return f"{v['mean']*100:.2f} ± {v['std']*100:.2f}"
        lines.append(f"| {LABELS[arm]} | {percent('valid_format')} | {percent('answer_correct')} | {percent('both_correct')} |")
    lines += ["", "A well-formed steps/answer string can still contain incorrect reasoning. This table isolates "
        "that distinction and keeps the easier digit-recall tasks out of the reasoning headline.", "",
        "## Inference refinement probes", "", "These use the same trained seed and "
        f"{plan['settings']['intervention_windows']} held-out windows. More rounds need not improve bpb.",
        "", "| Configuration | Test bpb ↓ | CPU ms/byte |", "|---|---:|---:|"]
    for name, metric in inference["metrics"].items():
        lines.append(f"| {name} | {metric['bpb']:.4f} | {metric['cpu_ms_per_byte']:.4f} |")
    lines += ["", "Removed proposals/workspaces are excluded from initialization/routing and held at zero, "
        "while the complete energy's compatibility constraints remain. Communication uses shared bridges, "
        "so this is a workspace intervention, not retraining a new model with all related parameters removed.",
        "", "## Short retraining and gradient filtering", "",
        "Each variant starts from the same seed and sees the same short batch sequence. Fixed directions "
        "have fewer trainable parameters; separate-block masking also leaves 4,800 direction coordinates "
        "inactive (245,129 functional parameters versus 249,929 for the dense core). Masking changes the "
        "initial frequency norms as well. These are structural diagnostics, not capacity/norm-matched "
        "proof that joint coupling itself helps. Spikes use the preset norm threshold 5; filtered spikes are "
        "measured before the common clipping step, which avoids crediting the Fourier filter for clipping.",
        "", "| Variant | Test bpb ↓ | Raw spike % | Filtered spike % | Maximum raw norm | Training CPU s |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, row in retraining.items():
        t, m = row["training"], row["language"]
        lines.append(f"| {name} | {m['bpb']:.4f} | {t['raw_spike_fraction']*100:.1f} | "
                     f"{t['filtered_spike_fraction']*100:.1f} | {t['maximum_raw_gradient_norm']:.3f} | {t['training_cpu_seconds']:.2f} |")
    lines += ["", "A low-spike pilot cannot demonstrate robust spike suppression. Filtering can remove useful "
        "gradient variation as well as noise; quality must be assessed with the saved bpb values.", "",
        "## Compute accounting", "",
        "Primary and equal-time training CPU counters include minibatch construction, forward/backward, "
        "auxiliary losses/teacher forwards, gradient diagnostics/filtering and AdamW updates. They exclude "
        "validation, initialization, checkpoint persistence and evaluation; those stages have separate "
        "recorded counters. The full model is charged for its shared primary teacher once. The equal-time "
        "full fit prepares a fresh short teacher inside its common budget. One complete update may overshoot.",
        "", "Native memory/timing profiles are isolated sequential processes, separate from concurrent training. "
        "The sampled RSS includes Python, Torch, parameters and data buffers; incremental RSS is also saved. "
        "Positive operator allocations are an allocation total, not peak resident memory.",
        "", "The initial failed attempt is preserved separately. Its data are excluded from the completed "
        "model comparison. A replay and a long-window high-gain regression test document the stability fix.",
        ""]
    profiles_path = ROOT / "profiles" / "all.json"
    if profiles_path.exists():
        profiles = json.loads(profiles_path.read_text())
        lines += ["## Sequential inference profiles", "",
            "| Model | Median CPU ms/byte | Sampled process peak RSS MiB | Incremental RSS MiB |",
            "|---|---:|---:|---:|"]
        for arm in LABELS:
            selected = [profiles[f"{arm}_{seed}"] for seed in plan["settings"]["seeds"]]
            lines.append(f"| {LABELS[arm]} | {np.mean([p['cpu_ms_per_byte_median'] for p in selected]):.4f} | "
                         f"{np.mean([p['sampled_inference_peak_rss_bytes'] for p in selected])/2**20:.2f} | "
                         f"{np.mean([p['sampled_incremental_inference_rss_bytes'] for p in selected])/2**20:.2f} |")
    (ROOT / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for arm in ("fourier_rewt", "full_rewt"):
        r = [(n, inference["metrics"][f"{arm}_rounds_{n}"]) for n in (0, 1, 2, 4, 8)]
        axes[0].plot([n for n,m in r], [m["bpb"] for n,m in r], marker="o", label=LABELS[arm])
    axes[0].set(xlabel="Refinement rounds", ylabel="Test bits per byte ↓", title="Inference-only round sweep")
    axes[0].legend()
    for method in ("none", "clip", "ema", "fourier"):
        path = Path(plan["runs"]) / "ablations" / f"gradient_{method}" / "trace.jsonl"
        trace = read_lines(path)
        axes[1].plot([r["step"] for r in trace], [r["filtered_gradient_norm"] for r in trace],
                     label=method, alpha=.8)
    axes[1].axhline(5, linestyle="--", color="gray", label="spike threshold")
    axes[1].set(xlabel="Training update", ylabel="Gradient norm before clipping", title="Separate filter study")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(ROOT / "diagnostics.png", dpi=170)
    plt.close(figure)
    # Fixed example IDs are chosen by original test order, never by success.
    example_text = ["# Fixed generated-output examples", "",
        "These are the first six test-task IDs and the first text prefix from the frozen plan, seed 101. "
        "They are shown regardless of whether they are correct. Full outputs for every case remain in the JSONL files.", ""]
    for task in test[:6]:
        example_text += [f"## {task.family}: {task.id[:12]}", "",
                         "Prompt:", "~~~text", task.prompt.rstrip(), "~~~",
                         "Correct response:", "~~~text", task.response.rstrip(), "~~~", ""]
        for arm in LABELS:
            record = next(r for r in read_lines(ROOT / "seed_101" / f"{arm}_reasoning.jsonl") if r["id"]==task.id)
            example_text += [f"{LABELS[arm]} (answer correct: {record['answer_correct']}; complete correct steps: {record['both_correct']}):",
                             "~~~text", record["response"].rstrip(), "~~~", ""]
    prompt_id = plan["text_prompts"][0]["id"]
    example_text += ["## First fixed text prefix", "", "~~~text", plan["text_prompts"][0]["prompt"], "~~~", ""]
    for arm in LABELS:
        record = next(r for r in read_lines(ROOT / "seed_101" / f"{arm}_text.jsonl") if r["prompt_id"]==prompt_id)
        example_text += [LABELS[arm] + ":", "~~~text", record["generation"], "~~~", ""]
    (ROOT / "OUTPUT_EXAMPLES.md").write_text("\n".join(example_text) + "\n")
    return paired


if __name__ == "__main__":
    analyze()
