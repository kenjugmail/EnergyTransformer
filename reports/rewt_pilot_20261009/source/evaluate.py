"""Measured held-out predictions, actual autoregressive output, and checkers."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import time
import tracemalloc

import numpy as np
import torch
from torch.nn import functional as F

from .data import task_batch
from .model import JointEnergyLM
from .tasks import score_response


def noise_for(model, tokens, generator):
    if not isinstance(model, JointEnergyLM) or not model.config.support:
        return None
    return torch.randn(len(tokens), tokens.shape[1], model.config.diffusion_samples,
                       model.config.width, generator=generator, device=tokens.device)


def model_output(model, tokens, generator, *, details=False, state=None, **kwargs):
    noise = noise_for(model, tokens, generator)
    return model(tokens, state=state, noise=noise, details=details, **kwargs)


@torch.inference_mode()
def language_metrics(model, corpus, offsets, *, context=128, batch_size=4, seed=530,
                     measure_memory=False, **kwargs):
    model.eval()
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    nll, brier, correct, count = 0., 0., 0, 0
    cost = dict(gradient_calls=0, energy_calls=0, attempts=0, accepted=0)
    if measure_memory:
        tracemalloc.start()
    wall, cpu = time.perf_counter(), time.process_time()
    for first in range(0, len(offsets), batch_size):
        selected = offsets[first:first + batch_size]
        batch = corpus.batch("test" if min(selected) >= 95_000_000 else "valid",
                             None, len(selected), context, starts=selected)
        output = model_output(model, batch.tokens, generator, **kwargs)
        logp = output.logits.log_softmax(-1)
        probability = logp.exp()
        true_probability = probability.gather(-1, batch.targets[..., None]).squeeze(-1)
        nll += float(-logp.gather(-1, batch.targets[..., None]).sum())
        brier += float((probability.square().sum(-1) - 2 * true_probability + 1).sum())
        correct += int((probability.argmax(-1) == batch.targets).sum())
        count += batch.targets.numel()
        for key in cost:
            cost[key] += output.details["cost"].get(key, 0)
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    python_peak = None
    if measure_memory:
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    # Calls are batched: divide by input windows, not by B*T. Also report
    # total batched calls so measured execution is auditable.
    return dict(bpb=nll / count / math.log(2), brier=brier / count,
                next_byte_accuracy=correct / count, evaluated_bytes=count,
                wall_seconds=wall, cpu_seconds=cpu, cpu_ms_per_byte=cpu * 1000 / count,
                gradient_calls_per_position=cost["gradient_calls"] / (math.ceil(len(offsets) / batch_size) * context),
                energy_calls_per_position=cost["energy_calls"] / (math.ceil(len(offsets) / batch_size) * context),
                cost_batched_calls=cost, python_peak_tracemalloc_bytes=python_peak)


@torch.inference_mode()
def task_loss(model, tasks, *, batch_size=8, seed=532):
    generator = torch.Generator().manual_seed(seed)
    nll, count, correct = 0., 0, 0
    model.eval()
    for first in range(0, len(tasks), batch_size):
        batch = task_batch(tasks, range(first, min(first + batch_size, len(tasks))))
        output = model_output(model, batch.tokens, generator)
        nll += float(F.cross_entropy(output.logits.reshape(-1, 256), batch.targets.flatten(),
                                    ignore_index=-100, reduction="sum"))
        valid = batch.valid
        correct += int(((output.logits.argmax(-1) == batch.targets) & valid).sum())
        count += int(valid.sum())
    return dict(answer_bpb=nll / count / math.log(2), teacher_forced_byte_accuracy=correct / count,
                answer_bytes=count)


@torch.inference_mode()
def generate_batch(model, prompts, *, max_new=96, sampling=False, seed=540,
                   temperature=0.8, stop_newline=False, **kwargs):
    """Equal-length prompts; no reference answer/continuation is accepted."""
    encoded = [prompt if isinstance(prompt, bytes) else prompt.encode("ascii") for prompt in prompts]
    if not encoded or not encoded[0] or len({len(p) for p in encoded}) != 1:
        raise ValueError("Use a nonempty group with equal nonzero prompt length")
    if max_new < 1 or len(encoded[0]) + max_new > 128:
        raise ValueError("Generation must fit the shared 128-byte window")
    if temperature <= 0:
        raise ValueError("Positive temperature required")
    generator = torch.Generator().manual_seed(seed)
    # Sampling uses its own stream so diffusion RNG consumption cannot change
    # the prescribed sampling uniforms between comparison arms.
    sample_generator = torch.Generator().manual_seed(seed + 1_000_000)
    model.eval()
    x = torch.tensor([list(raw) for raw in encoded], dtype=torch.long)
    output = model_output(model, x, generator, **kwargs)
    state, logits = output.state, output.logits[:, -1]
    generated = [[] for _ in encoded]
    done = torch.zeros(len(encoded), dtype=torch.bool)
    for step in range(max_new):
        if sampling:
            probability = (logits / temperature).softmax(-1)
            # Inverse-CDF sampling consumes exactly one uniform per row/step.
            uniform = torch.rand(len(encoded), 1, generator=sample_generator)
            next_token = (probability.cumsum(-1) < uniform).sum(-1).clamp_max(255)
        else:
            next_token = logits.argmax(-1)
        for row, token in enumerate(next_token.tolist()):
            if not bool(done[row]):
                generated[row].append(token)
                if stop_newline and token == 10:
                    done[row] = True
        if bool(done.all()) or step + 1 == max_new:
            break
        output = model_output(model, next_token[:, None], generator, state=state, **kwargs)
        state, logits = output.state, output.logits[:, -1]
    return [bytes(row) for row in generated]


def repetition_metrics(raw):
    grams = [raw[i:i+3] for i in range(max(0, len(raw) - 2))]
    word_text = raw.decode("ascii", errors="replace")
    words = word_text.split()
    word_grams = [tuple(words[i:i+3]) for i in range(max(0, len(words) - 2))]
    return dict(
        printable_fraction=sum(32 <= b < 127 or b in (9, 10, 13) for b in raw) / max(len(raw), 1),
        replacement_fraction=sum(b >= 128 for b in raw) / max(len(raw), 1),
        repeated_byte_trigram_fraction=(len(grams) - len(set(grams))) / max(len(grams), 1),
        repeated_word_trigram_fraction=(len(word_grams) - len(set(word_grams))) / max(len(word_grams), 1),
        distinct_byte_trigram_fraction=len(set(grams)) / max(len(grams), 1),
        bytes=len(raw))


def text_evaluation(model, prompts, *, seed, batch_size=10):
    records = []
    for first in range(0, len(prompts), batch_size):
        group = prompts[first:first + batch_size]
        generated = generate_batch(model, [bytes.fromhex(p["prompt_hex"]) for p in group],
                                   max_new=96, sampling=True, seed=seed + first)
        for prompt, raw in zip(group, generated):
            records.append(dict(prompt_id=prompt["id"], prompt=prompt["prompt"],
                                generation_hex=raw.hex(), generation=raw.decode("utf-8", errors="replace"),
                                reference_hex=prompt["reference_hex"], metrics=repetition_metrics(raw)))
    summary = {key: float(np.mean([r["metrics"][key] for r in records]))
               for key in records[0]["metrics"]}
    summary["count"] = len(records)
    summary["human_ratings"] = "pending; use the blinded review artifact"
    return summary, records


def reasoning_evaluation(model, tasks, *, seed=548, **kwargs):
    records, grouped = [], defaultdict(list)
    for task in tasks:
        grouped[len(task.prompt.encode("ascii"))].append(task)
    for length, group in sorted(grouped.items()):
        for first in range(0, len(group), 8):
            selected = group[first:first + 8]
            generated = generate_batch(model, [task.prompt for task in selected],
                max_new=min(40, 128 - length), stop_newline=True, seed=seed + length + first, **kwargs)
            for task, raw in zip(selected, generated):
                score = score_response(task, raw.decode("ascii", errors="replace"))
                records.append(dict(id=task.id, family=task.family, template=task.template,
                    steps=task.steps, difficulty=task.difficulty, prompt=task.prompt,
                    generated_hex=raw.hex(), **score))

    def summarize(rows):
        return dict(count=len(rows), **{key: sum(bool(r[key]) for r in rows) / max(len(rows), 1)
                for key in ("valid_format", "answer_correct", "steps_correct", "both_correct")})

    summary = dict(all=summarize(records))
    for field in ("family", "difficulty", "steps"):
        summary[f"by_{field}"] = {str(key): summarize([r for r in records if r[field] == key])
                                 for key in sorted({r[field] for r in records}, key=str)}
    return summary, records


@torch.inference_mode()
def refinement_audit(model, tokens, *, seed=549):
    if not isinstance(model, JointEnergyLM):
        return None
    generator = torch.Generator().manual_seed(seed)
    output = model_output(model, tokens, generator, details=True)
    trace = output.details["energy_trace"]
    differences = trace[..., 1:] - trace[..., :-1]
    return dict(positions=tokens.numel(), rounds=model.config.rounds,
                maximum_accepted_energy_increase=float(differences.max()) if differences.numel() else 0,
                fraction_rounds_decreasing=float((differences < -1e-7).float().mean()) if differences.numel() else 0,
                mean_energy_drop=float(trace[..., 0].mean() - trace[..., -1].mean()),
                all_accepted_states_nonincreasing=bool((differences <= 1e-6).all()))


def runtime_profile(model, tokens, *, repeats=3):
    """Sequential repeat timing and native PyTorch allocation attribution."""
    times = []
    for repeat in range(repeats + 1):
        generator = torch.Generator().manual_seed(550)
        start = time.process_time()
        with torch.inference_mode():
            model_output(model, tokens, generator)
        elapsed = time.process_time() - start
        if repeat:
            times.append(elapsed)
    # CPU profiler exposes allocations but not a reliable universal process
    # peak: report summed positive operator allocations with an explicit name.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True) as profiler:
        with torch.inference_mode():
            model_output(model, tokens, torch.Generator().manual_seed(550))
    allocations = sum(max(0, entry.self_cpu_memory_usage) for entry in profiler.key_averages())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return dict(cpu_seconds_median=float(np.median(times)), cpu_seconds_repeats=times,
                cpu_ms_per_byte_median=float(np.median(times)) * 1000 / tokens.numel(),
                positive_operator_allocation_bytes=allocations, parameter_bytes=parameter_bytes,
                memory_note="Positive allocation total is NOT peak resident memory; checkpoints contain parameters only.")
