"""Reproducible random-initialization REWT pilot, with resumable training.

Main controls share bytes, initialization seeds, AdamW settings and validation
selection. Equal-CPU-time fits, reasoning adaptation, and small component/filter
studies are separate protocols, labeled separately in the saved results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import time
import zlib

import numpy as np
import torch
from torch.nn import functional as F

from .data import Corpus, file_sha256, task_batch
from .evaluate import (language_metrics, model_output, noise_for, reasoning_evaluation,
                       refinement_audit, task_loss, text_evaluation)
from .model import EnergyConfig, JointEnergyLM, TransformerConfig, build_model, matched_configurations
from .optim import GradientFilter
from .tasks import generate_tasks, task_manifest


ARMS = ("transformer", "neural_energy", "fourier_rewt", "full_rewt")
AUXILIARY_WEIGHTS = dict(ranking=0.02, forecast=0.02, denoising=0.01, critic=0.01, routing=0.01)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in values))


def event(message, **fields):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), message=message, **fields)), flush=True)


def config_from(arm, payload):
    return TransformerConfig(**payload) if arm == "transformer" else EnergyConfig(**payload)


def new_model(arm, payload, seed):
    torch.manual_seed(seed)
    return build_model(arm, config_from(arm, payload))


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = new_model(checkpoint["arm"], checkpoint["config"], checkpoint["seed"])
    model.load_state_dict(checkpoint["weights"], strict=True)
    return model, checkpoint


def initialization_noise(model, tokens, *, seed, recipe):
    """Paired, label-free corruption of guesses; never corrupt observed bytes."""
    probability = recipe.get("corruption_probability", 0.)
    if not isinstance(model, JointEnergyLM) or probability == 0:
        return None, 0.
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    active = torch.rand(len(tokens), 1, 1, generator=generator, device=tokens.device) < probability
    noise = model.embedding.weight.new_zeros(*tokens.shape, model.dimensions)
    for name, sl in model.blocks.items():
        block_generator = torch.Generator(device=tokens.device).manual_seed(seed + zlib.crc32(name.encode()))
        scale = recipe.get("output_noise", .5) if name == "output" else recipe.get("state_noise", .05)
        noise[..., sl] = torch.randn(*tokens.shape, sl.stop-sl.start, generator=block_generator,
                                     device=tokens.device) * scale * active
    return noise, float(active.float().mean())


def curriculum_batch(tasks, rng, batch_size, *, stage, process_seed, process_supervision):
    pools = {name: [i for i, task in enumerate(tasks) if task.family == name
                   and task.steps <= (stage if name != "recall" else (4, 8, 16)[stage-1])]
             for name in ("arithmetic", "rules", "recall")}
    if any(not pool for pool in pools.values()):
        raise ValueError("Curriculum requires examples from every family at its current depth")
    first = int(rng.integers(3))
    names = tuple(pools)
    indices = [rng.choice(pools[names[(first+i) % 3]]) for i in range(batch_size)]
    return task_batch(tasks, indices, process_supervision=process_supervision, process_seed=process_seed)


def train(model, arm, seed, phase, plan, corpus, directory, *, teacher=None,
          tasks=None, initial_weights=None, method="clip", updates=None, cpu_budget=None,
          warm_updates=None, teacher_charge=None, recipe=None):
    """AdamW training with answer-only masks; all future targets stay here."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, best_path = directory / "training.json", directory / "best.pt"
    if summary_path.exists():
        model, checkpoint = load_model(best_path)
        return model, json.loads(summary_path.read_text())
    settings = plan["settings"]
    recipe = dict(plan.get("training_recipe", {}) if recipe is None else recipe)
    auxiliary_weights = {**AUXILIARY_WEIGHTS, **recipe.get("auxiliary_weights", {})}
    config = asdict(model.config)
    updates = updates if updates is not None else settings["lm_updates"]
    warm_updates = settings["warm_updates"] if warm_updates is None else warm_updates
    learning_rate = settings["learning_rate"]
    if tasks is not None:
        batch_size = settings["reason_batch_size"]
    else:
        batch_size = settings["batch_size"]
    if initial_weights is not None:
        model.load_state_dict(initial_weights, strict=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    filtering = GradientFilter(model.parameters(), method=method, history=8, cutoff=0.5,
                               max_norm=1., spike_threshold=5., residual_gain=recipe.get("residual_gain", .25))
    last_path, trace_path = directory / "last.pt", directory / "trace.jsonl"
    start_step, training_cpu, training_wall, validation_cpu = 0, 0., 0., 0.
    processed_bytes, best_score, best_step = 0, float("inf"), 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["phase"] != phase or state["config"] != config or state["seed"] != seed:
            raise ValueError("Resume checkpoint differs from the declared fit")
        if "training_recipe" in state and (state["training_recipe"] != recipe or state["method"] != method):
            raise ValueError("Resume recipe or gradient filter differs from the declared fit")
        model.load_state_dict(state["weights"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        filtering.load_state_dict(state["filter"])
        start_step, training_cpu, training_wall = state["step"], state["training_cpu"], state["training_wall"]
        validation_cpu, processed_bytes = state["validation_cpu"], state["processed_bytes"]
        best_score, best_step = state["best_score"], state["best_step"]
        if trace_path.exists():
            kept = [json.loads(line) for line in trace_path.read_text().splitlines()]
            save_lines(trace_path, [row for row in kept if row["step"] <= start_step])
    elif trace_path.exists():
        trace_path.unlink()
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)
    valid_offsets = plan["validation_offsets"]
    validation_tasks = generate_tasks("valid", settings["task_valid"],
                                     number_range=settings.get("task_number_range", 10),
                                     seed=settings.get("task_seed", 811))
    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
    last_validation = None
    event("fit_started", seed=seed, arm=arm, phase=phase, parameters=model.parameters_count(),
          updates=updates, cpu_budget=cpu_budget, resumed_step=start_step)
    for index in range(start_step, updates):
        if cpu_budget is not None and training_cpu >= cpu_budget:
            break
        # Per-step streams permit exact continuation without hidden RNG state
        # and present paired batches/noise to every comparison arm.
        rng = np.random.default_rng(seed * 1_000_000 + index + (77_000_000 if tasks else 0))
        generator = torch.Generator().manual_seed(seed * 1_000_000 + index + 19)
        step_wall, step_cpu = time.perf_counter(), time.process_time()
        if tasks is None:
            batch = corpus.batch("train", rng, batch_size, settings["context"])
        else:
            stage = 3
            if recipe.get("curriculum", False):
                progress = index / max(settings.get("reason_updates", updates), 1)
                stage = 1 if progress < .25 else 2 if progress < .55 else 3
                batch = curriculum_batch(tasks, rng, batch_size, stage=stage,
                    process_seed=seed * 1_000_000 + index,
                    process_supervision=recipe.get("process_supervision", False))
            else:
                batch = task_batch(tasks, rng.integers(len(tasks), size=batch_size),
                    process_supervision=recipe.get("process_supervision", False),
                    process_seed=seed * 1_000_000 + index)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rounds = 0 if index < warm_updates else 2
        if index >= warm_updates and recipe.get("variable_rounds", False):
            depth_rng = np.random.default_rng(seed * 1_000_000 + index + 90_000_000)
            rounds = int(depth_rng.integers(recipe.get("minimum_rounds", 1), recipe.get("maximum_rounds", 4) + 1))
        # The fixed warm-up LR rule is identical between comparison arms.
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * min(1., (index + 1) / 16)
        details = isinstance(model, JointEnergyLM)
        initial_noise, corruption_fraction = initialization_noise(model, batch.tokens,
            seed=seed * 1_000_000 + index + 91_000_000, recipe=recipe if rounds else {})
        output = model_output(model, batch.tokens, generator, details=details, rounds=rounds,
                              **({"initial_noise": initial_noise,
                                  "state_gradient_interval": recipe.get("state_gradient_interval_reasoning",
                                      recipe.get("state_gradient_interval",0)) if tasks is not None else
                                      recipe.get("state_gradient_interval",0),
                                  "trace_energy": recipe.get("trace_training_energy", True)} if details else {}))
        ce = F.cross_entropy(output.logits.reshape(-1, 256), batch.targets.flatten(), ignore_index=-100)
        extra = {}
        if details:
            extra["ranking"] = model.ranking_loss(batch.tokens, batch.targets, output, batch.valid,
                hard_negatives=recipe.get("hard_negatives", False), generator=generator)
            if recipe.get("repair_weight", 0.) > 0:
                extra["repair"] = model.repair_loss(output, batch.targets, batch.valid)
            if model.config.support:
                hidden = None
                if teacher is not None:
                    with torch.no_grad():
                        hidden = teacher(batch.tokens, details=True).details["hidden"].detach()
                extra.update(model.auxiliary_losses(output, batch.targets, valid=batch.valid,
                    teacher_hidden=hidden,
                    noise=torch.randn(*output.details["clean"].shape, generator=generator)))
        if batch.process_negatives is not None:
            if details:
                extra["process"] = model.process_loss(output, batch.targets, batch.process_negatives)
            else:
                usable = batch.process_negatives >= 0
                correct_logits = output.logits.gather(-1, batch.targets.clamp_min(0)[..., None]).squeeze(-1)
                wrong_logits = output.logits.gather(-1, batch.process_negatives.clamp_min(0)[..., None]).squeeze(-1)
                extra["process"] = F.softplus((wrong_logits - correct_logits)[usable]).mean()
        weights = {**auxiliary_weights, "repair": recipe.get("repair_weight", 0.),
                   "process": recipe.get("process_weight", .05)}
        loss = ce + sum(weights[key] * value for key, value in extra.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite objective at {arm}/{phase}/{index}")
        loss.backward()
        diagnostics = filtering.apply()
        optimizer.step()
        cpu, wall = time.process_time() - step_cpu, time.perf_counter() - step_wall
        training_cpu += cpu
        training_wall += wall
        processed_bytes += int(batch.valid.sum())
        row = dict(step=index + 1, ce=float(ce.detach()), total_loss=float(loss.detach()),
                   rounds=rounds, cpu_seconds=cpu, wall_seconds=wall,
                   initialization_corruption_fraction=corruption_fraction,
                   curriculum_stage=stage if tasks is not None else None,
                   auxiliary={key: float(value.detach()) for key, value in extra.items()},
                   **diagnostics)
        with trace_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        final = index + 1 == updates or (cpu_budget is not None and training_cpu >= cpu_budget)
        milestone = index + 1 in settings.get("budget_milestones", []) and tasks is None and cpu_budget is None
        should_validate = final or milestone or (cpu_budget is None and (index + 1) % settings["validate_every"] == 0)
        if should_validate:
            validation_start = time.process_time()
            if tasks is None:
                last_validation = language_metrics(model, corpus, valid_offsets,
                    context=settings["context"], batch_size=settings["batch_size"], seed=561)
                score = last_validation["bpb"]
            else:
                last_validation = task_loss(model, validation_tasks, batch_size=settings["reason_batch_size"])
                score = last_validation["answer_bpb"]
            validation_cpu += time.process_time() - validation_start
            if score < best_score:
                best_score, best_step = score, index + 1
                save_checkpoint(best_path, dict(arm=arm, config=config, seed=seed, phase=phase,
                    step=best_step, validation_score=best_score, weights=model.state_dict(),
                    source_sha256=plan["source_sha256"], plan_sha256=plan["plan_sha256"]))
            event("validation", seed=seed, arm=arm, phase=phase, step=index + 1,
                  validation_score=score, training_cpu_seconds=training_cpu, ce=row["ce"])
            if milestone:
                selected = torch.load(best_path, map_location="cpu", weights_only=True)
                selected.update(selected_at_budget=index + 1, processed_at_budget=processed_bytes,
                                training_cpu_until_budget=training_cpu)
                save_checkpoint(directory / f"budget_{index+1}.pt", selected)
        if should_validate or (index + 1) % 20 == 0:
            save_checkpoint(last_path, dict(arm=arm, config=config, seed=seed, phase=phase,
                weights=model.state_dict(), optimizer=optimizer.state_dict(), filter=filtering.state_dict(),
                step=index + 1, training_cpu=training_cpu, training_wall=training_wall,
                validation_cpu=validation_cpu, processed_bytes=processed_bytes,
                best_score=best_score, best_step=best_step,training_recipe=recipe,method=method))
        if (index + 1) % 20 == 0:
            event("training_progress", seed=seed, arm=arm, phase=phase, step=index + 1,
                  target_updates=updates, ce=row["ce"], training_cpu_seconds=training_cpu)
    if not best_path.exists():
        raise RuntimeError("Fit ended before producing a validation-selected checkpoint")
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    high = [row["high_frequency_fraction"] for row in rows if row["high_frequency_fraction"] is not None]
    teacher_cost = 0.
    if teacher is not None and teacher_charge is None:
        teacher_summary = json.loads((Path(plan["runs"]) / f"seed_{seed}" / "transformer" / "language" / "training.json").read_text())
        teacher_cost = teacher_summary["training_cpu_seconds"]
    elif teacher is not None:
        teacher_cost = teacher_charge
    summary = dict(seed=seed, arm=arm, phase=phase, method=method,
        updates=len(rows), best_step=best_step, best_validation_score=best_score,
        processed_target_bytes=processed_bytes, parameters=model.parameters_count(),
        all_parameters=sum(p.numel() for p in model.parameters()),
        training_cpu_seconds=training_cpu, training_wall_seconds=training_wall,
        validation_cpu_seconds=validation_cpu, remaining_phase_wall_seconds=time.perf_counter()-phase_wall,
        remaining_phase_cpu_seconds=time.process_time()-phase_cpu,
        shared_teacher_training_cpu_seconds=teacher_cost,
        training_plus_shared_teacher_cpu_seconds=training_cpu + teacher_cost,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        mean_raw_gradient_norm=float(np.mean([r["raw_gradient_norm"] for r in rows])),
        maximum_raw_gradient_norm=max(r["raw_gradient_norm"] for r in rows),
        raw_spike_fraction=sum(r["spike"] for r in rows) / len(rows),
        filtered_spike_fraction=sum(r["filtered_spike"] for r in rows) / len(rows),
        clip_fraction=sum(r["clipped"] for r in rows) / len(rows),
        mean_high_frequency_gradient_fraction=float(np.mean(high)) if high else None,
        checkpoint=str(best_path.resolve()),
        forecast_targets="frozen from-scratch Transformer hidden states; training only" if teacher else None,
        training_recipe=recipe,
        validation=last_validation, initialization="random" if initial_weights is None else "copy of this seed's course-trained LM")
    save_json(summary_path, summary)
    model, _ = load_model(best_path)
    return model, summary


def run_seed(plan, seed):
    torch.set_num_threads(plan["settings"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus = Corpus(plan["corpus"]["path"])
    base = Path(plan["runs"]) / f"seed_{seed}"
    results = Path(plan["out"]) / f"seed_{seed}"
    results.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks("train", plan["settings"]["task_train"])
    test_tasks = generate_tasks("test", plan["settings"]["task_test"])
    trained, summary = {}, {}
    for arm in ARMS:
        model = new_model(arm, plan["configurations"][arm], seed)
        teacher = trained.get("transformer") if arm == "full_rewt" else None
        model, fitting = train(model, arm, seed, "language", plan, corpus, base / arm / "language", teacher=teacher)
        trained[arm] = model
        row_path = results / f"{arm}.json"
        if row_path.exists():
            summary[arm] = json.loads(row_path.read_text())
            continue
        metrics = language_metrics(model, corpus, plan["test_offsets"],
            context=plan["settings"]["context"], batch_size=plan["settings"]["batch_size"], seed=562)
        text_summary, text_records = text_evaluation(model, plan["text_prompts"], seed=seed + 563)
        # Neither labels nor reference text are accepted by generation.
        save_lines(results / f"{arm}_text.jsonl", text_records)
        audit_batch = corpus.batch("test", None, 2, 16, starts=plan["test_offsets"][:2])
        zero_shot, zero_records = reasoning_evaluation(model, test_tasks, seed=seed + 564)
        save_lines(results / f"{arm}_reasoning_zero_shot.jsonl", zero_records)
        row = dict(arm=arm, seed=seed, training=fitting, language=metrics, text=text_summary,
                   refinement_audit=refinement_audit(model, audit_batch.tokens),
                   zero_shot_reasoning=zero_shot)
        save_json(row_path, row)
        summary[arm] = row
        event("language_evaluation", seed=seed, arm=arm, test_bpb=metrics["bpb"],
              printable=text_summary["printable_fraction"])
    for arm in ARMS:
        row_path = results / f"{arm}.json"
        row = json.loads(row_path.read_text())
        if "adapted_reasoning" not in row:
            base_model = trained[arm]
            copy = new_model(arm, plan["configurations"][arm], seed)
            copy, fitting = train(copy, arm, seed, "reasoning", plan, corpus, base / arm / "reasoning",
                teacher=trained["transformer"] if arm == "full_rewt" else None, tasks=tasks,
                initial_weights=base_model.state_dict(), updates=plan["settings"]["reason_updates"],
                warm_updates=0, teacher_charge=0.)
            reasoning, records = reasoning_evaluation(copy, test_tasks, seed=seed + 565)
            save_lines(results / f"{arm}_reasoning.jsonl", records)
            row.update(adapted_reasoning=reasoning, reasoning_training=fitting,
                       reasoning_teacher_forced=task_loss(copy, test_tasks))
            save_json(row_path, row)
            event("reasoning_evaluation", seed=seed, arm=arm,
                  exact_answer=reasoning["all"]["answer_correct"], valid_steps=reasoning["all"]["both_correct"])
        if "equal_cpu_time" not in row:
            copy = new_model(arm, plan["configurations"][arm], seed)
            teacher, teacher_cost, teacher_fitting = None, 0., None
            common_budget = plan["settings"]["equal_cpu_seconds"]
            if arm == "full_rewt":
                # A fresh small teacher is trained INSIDE the common budget,
                # rather than borrowing the longer primary-baseline fit.
                teacher = new_model("transformer", plan["configurations"]["transformer"], seed)
                teacher, teacher_fitting = train(teacher, "transformer", seed,
                    "equal_cpu_time_teacher", plan, corpus, base / arm / "equal_cpu_time_teacher",
                    updates=20_000, cpu_budget=common_budget * 0.25, warm_updates=4)
                teacher_cost = teacher_fitting["training_cpu_seconds"]
            # A fresh fit with a short common warm-up; no primary weights reused.
            copy, fitting = train(copy, arm, seed, "equal_cpu_time", plan, corpus,
                base / arm / "equal_cpu_time", teacher=teacher, teacher_charge=teacher_cost,
                updates=20_000, cpu_budget=max(0.1, common_budget - teacher_cost), warm_updates=4)
            row["equal_cpu_time"] = dict(training=fitting, language=language_metrics(copy, corpus,
                plan["test_offsets"], context=128, batch_size=plan["settings"]["batch_size"], seed=566),
                teacher_training=teacher_fitting, target_total_cpu_seconds=common_budget,
                teacher_charge_note="Fresh teacher gets 25% of the common CPU budget; full REWT gets the remainder. One complete update may overshoot the cap.")
            save_json(row_path, row)
            event("compute_evaluation", seed=seed, arm=arm,
                  updates=fitting["updates"], test_bpb=row["equal_cpu_time"]["language"]["bpb"])
        summary[arm] = row
    save_json(results / "complete.json", summary)
    return dict(seed=seed, completed=True)


def run_ablations(plan):
    """Single-seed diagnostic retraining and clearly labeled inference probes."""
    torch.set_num_threads(plan["settings"]["threads"])
    seed, corpus = plan["settings"]["seeds"][0], Corpus(plan["corpus"]["path"])
    output = Path(plan["out"]) / "ablations"
    base = Path(plan["runs"]) / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    cfg = config_from("fourier_rewt", plan["configurations"]["fourier_rewt"])
    variants = {f"gradient_{method}": (cfg, method) for method in ("none", "clip", "ema", "fourier")}
    variants.update(fixed_frequencies=(replace(cfg, learned_frequencies=False), "clip"),
                    separate_block_modes=(replace(cfg, coupled_modes=False), "clip"),
                    coarse_to_fine=(replace(cfg, schedule="coarse_to_fine"), "clip"))
    rows = {}
    for name, (config, method) in variants.items():
        saved = output / f"{name}.json"
        if saved.exists():
            rows[name] = json.loads(saved.read_text())
            continue
        model = new_model("fourier_rewt", asdict(config), seed)
        model, fitting = train(model, "fourier_rewt", seed, name, plan, corpus, base / name,
            method=method, updates=plan["settings"]["ablation_updates"], warm_updates=8)
        metrics = language_metrics(model, corpus, plan["test_offsets"], context=128,
                                   batch_size=plan["settings"]["batch_size"], seed=567)
        rows[name] = dict(training=fitting, language=metrics,
                         evidence_scope="Single-seed short retraining; not the primary three-seed comparison")
        save_json(saved, rows[name])
    inference_path = output / "inference_interventions.json"
    if not inference_path.exists():
        full, _ = load_model(Path(plan["runs"]) / f"seed_{seed}" / "full_rewt" / "language" / "best.pt")
        core, _ = load_model(Path(plan["runs"]) / f"seed_{seed}" / "fourier_rewt" / "language" / "best.pt")
        subset = plan["test_offsets"][:plan["settings"]["intervention_windows"]]
        sweeps = {}
        for arm, model in (("fourier_rewt", core), ("full_rewt", full)):
            for rounds in (0, 1, 2, 4, 8):
                sweeps[f"{arm}_rounds_{rounds}"] = language_metrics(model, corpus, subset,
                    context=128, batch_size=plan["settings"]["batch_size"], seed=568, rounds=rounds)
            for name, states, outputs in (("states_only", True, False), ("output_only", False, True)):
                sweeps[f"{arm}_{name}"] = language_metrics(model, corpus, subset,
                    context=128, batch_size=plan["settings"]["batch_size"], seed=568,
                    update_states=states, update_output=outputs)
        for name in ("forecast", "diffusion", "specialists", "communication", "critic", "router", "history"):
            sweeps[f"full_without_{name}"] = language_metrics(full, corpus, subset, context=128,
                batch_size=plan["settings"]["batch_size"], seed=568, disabled=(name,))
        # Full-band acceptance is preserved for this schedule intervention.
        coarse = build_model("fourier_rewt", replace(core.config, schedule="coarse_to_fine"))
        coarse.load_state_dict(core.state_dict())
        sweeps["fourier_inference_coarse_to_fine"] = language_metrics(coarse, corpus, subset,
            context=128, batch_size=plan["settings"]["batch_size"], seed=568)
        save_json(inference_path, dict(metrics=sweeps,
            evidence_scope="Single trained seed; same held-out windows; no retraining. Removal changes inference inputs/workspace constraints.",
            generation_settings="Fixed diffusion RNG stream and rounds; all accepted updates checked against full energy."))
    save_json(output / "retraining.json", rows)
    event("ablations_complete", variants=len(rows))


def freeze_plan(args):
    out, runs = Path(args.out).resolve(), Path(args.runs).resolve()
    manifest = out / "plan.json"
    if manifest.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume or choose a fresh directory")
        plan = json.loads(manifest.read_text())
        for name, digest in plan["source_sha256"].items():
            if file_sha256(Path(__file__).parent / name) != digest:
                raise ValueError(f"Training source changed since this run: {name}")
        return plan
    if out.exists() and any(out.iterdir()) or runs.exists() and any(runs.iterdir()):
        raise ValueError("Fresh run directories must be empty")
    corpus = Corpus(args.data)
    configurations = matched_configurations(target=args.parameters)
    configurations = {arm: replace(config, backtracks=2) if isinstance(config, EnergyConfig) else config
                      for arm, config in configurations.items()}
    counts = {arm: build_model(arm, config).parameters_count() for arm, config in configurations.items()}
    if max(counts.values()) / min(counts.values()) > 1.005:
        raise ValueError("Primary active parameter counts differ by more than 0.5%")
    package = Path(__file__).parent
    hashes = {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}
    settings = dict(seeds=args.seeds, context=128, batch_size=args.batch_size,
        reason_batch_size=args.reason_batch_size, lm_updates=args.lm_updates,
        reason_updates=args.reason_updates, warm_updates=min(24, args.lm_updates // 4),
        validate_every=args.validate_every, learning_rate=args.learning_rate,
        task_train=args.task_train, task_valid=args.task_valid, task_test=args.task_test,
        equal_cpu_seconds=args.equal_cpu_seconds, ablation_updates=args.ablation_updates,
        intervention_windows=args.intervention_windows, threads=args.threads)
    plan = dict(version=1, created=datetime.now(timezone.utc).isoformat(), out=str(out), runs=str(runs),
        settings=settings, configurations={arm: asdict(config) for arm, config in configurations.items()},
        counts=counts, corpus=corpus.identity(), source_sha256=hashes,
        validation_offsets=corpus.offsets("valid", args.validation_windows, 128, 701),
        test_offsets=corpus.offsets("test", args.evaluation_windows, 128, 702),
        text_prompts=corpus.text_prompts(count=30),
        task_manifests={split: task_manifest(generate_tasks(split, count))
                        for split, count in (("train", args.task_train), ("valid", args.task_valid), ("test", args.task_test))},
        auxiliary_weights=AUXILIARY_WEIGHTS, spike_threshold=5.,
        device="cpu", cuda_available=torch.cuda.is_available(), torch_version=str(torch.__version__),
        python_version=platform.python_version(), machine=platform.machine(),
        threads_per_worker=args.threads, protocol="bounded pilot; no external pretrained weights; no test tuning")
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    snapshot = out / "source"
    snapshot.mkdir()
    for path in package.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    save_json(manifest, plan)
    for split, count in (("train", args.task_train), ("valid", args.task_valid), ("test", args.task_test)):
        save_lines(out / f"tasks_{split}.jsonl", [task.to_dict() for task in generate_tasks(split, count)])
    save_json(out / "text_prompts.json", plan["text_prompts"])
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/enwik8")
    parser.add_argument("--out", default="reports/rewt_pilot_20261009_v2")
    parser.add_argument("--runs", default="runs/rewt_pilot_20261009_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parameters", type=int, default=250_000)
    parser.add_argument("--lm-updates", type=int, default=120)
    parser.add_argument("--reason-updates", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reason-batch-size", type=int, default=8)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--evaluation-windows", type=int, default=32)
    parser.add_argument("--validate-every", type=int, default=40)
    parser.add_argument("--task-train", type=int, default=4096)
    parser.add_argument("--task-valid", type=int, default=192)
    parser.add_argument("--task-test", type=int, default=150)
    parser.add_argument("--equal-cpu-seconds", type=float, default=30.)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--ablation-updates", type=int, default=40)
    parser.add_argument("--intervention-windows", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args(argv)
    if any(v < 1 for v in (args.workers, args.threads, args.lm_updates, args.reason_updates,
                          args.batch_size, args.reason_batch_size, args.validate_every,
                          args.task_train, args.task_valid, args.task_test, args.evaluation_windows,
                          args.validation_windows, args.ablation_updates, args.intervention_windows)):
        parser.error("Counts must be positive")
    if args.equal_cpu_seconds <= 0 or args.learning_rate <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive time/LR and distinct seeds required")
    torch.set_num_threads(args.threads)
    plan = freeze_plan(args)
    event("experiment_started", output=plan["out"], settings=plan["settings"], counts=plan["counts"])
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(plan["settings"]["seeds"])),
                             mp_context=context) as pool:
        futures = [pool.submit(run_seed, plan, seed) for seed in plan["settings"]["seeds"]]
        for future in as_completed(futures):
            event("seed_complete", **future.result())
    if not args.skip_ablations:
        run_ablations(plan)
    from .report import create_report
    create_report(plan)
    save_json(Path(plan["out"]) / "complete.json", dict(completed=datetime.now(timezone.utc).isoformat(),
        seeds=plan["settings"]["seeds"], all_primary_fits_complete=True, ablations=not args.skip_ablations))
    event("experiment_complete", report=str(Path(plan["out"]) / "REPORT.md"))


if __name__ == "__main__":
    main()
