"""Restart unstable full reasoning fits and re-evaluate on a declared fresh test.

Completed language/control weights are copied byte-for-byte with their original
training/source provenance. All three full reasoning adaptations restart from
their completed language weights, using sixteen-token temporal gradient bounds.
No partial evaluation from the stopped parent study is reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from .data import Corpus, file_sha256
from .experiment import event, load_model, save_json, save_lines
from .improve import LABELS, digest, fit_seed, nonoverlapping_offsets, worker_setup
from .tasks import generate_tasks, task_manifest


def _read(path):
    return json.loads(path.read_text())


def verify_reused_inference(parent, ledger):
    """Compare every reused checkpoint against its actual archived model code."""
    package = "_rewt_stopped_parent"
    spec = importlib.util.spec_from_file_location(package, parent / "source" / "__init__.py",
        submodule_search_locations=[str(parent / "source")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    old_model = sys.modules[package + ".model"]
    tokens = torch.randint(256, (1, 16), generator=torch.Generator().manual_seed(1911))
    records = []
    for record in ledger:
        checkpoint = torch.load(record["original_checkpoint"], map_location="cpu", weights_only=True)
        old_config = (old_model.TransformerConfig if checkpoint["arm"] == "transformer" else
                      old_model.EnergyConfig)(**checkpoint["config"])
        old = old_model.build_model(checkpoint["arm"], old_config).eval()
        old.load_state_dict(checkpoint["weights"], strict=True)
        current, _ = load_model(record["copied_checkpoint"])
        current.eval()
        kwargs = {}
        if checkpoint["arm"] == "full_rewt":
            kwargs["noise"] = torch.randn(1, tokens.shape[1], current.config.diffusion_samples,
                current.config.width, generator=torch.Generator().manual_seed(1912))
        with torch.inference_mode():
            a = old(tokens, **kwargs).logits
            b = current(tokens, **kwargs).logits
        if not torch.equal(a, b):
            raise ValueError("A reused checkpoint's inference changed: " + record["original_checkpoint"])
        records.append(dict(seed=record["seed"], variant=record["variant"], phase=record["phase"],
                            exact_logits_equal=True, positions=tokens.numel()))
    return records


def prepare(args):
    out, runs, parent = Path(args.out).resolve(), Path(args.runs).resolve(), Path(args.parent).resolve()
    if (out / "selected_protocol.json").exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume with the same source")
        plan = _read(out / "selected_protocol.json")
        for name, expected in plan["source_sha256"].items():
            if file_sha256(Path(__file__).parent / name) != expected:
                raise ValueError("Frozen source changed: " + name)
        for fit in plan["reused_fits"]:
            if file_sha256(Path(fit["copied_checkpoint"])) != fit["checkpoint_sha256"]:
                raise ValueError("A copied parent weight changed")
        plan["variants"] = {name:plan["variants"][name] for name in LABELS}
        return plan
    if (out.exists() and any(out.iterdir())) or (runs.exists() and any(runs.iterdir())):
        raise ValueError("Fresh output and checkpoint directories must be empty")
    prior = _read(parent / "selected_protocol.json")
    for name, expected in prior["source_sha256"].items():
        if file_sha256(parent / "source" / name) != expected:
            raise ValueError("Parent source snapshot changed: " + name)
    corpus = Corpus(prior["corpus"]["path"])
    if corpus.identity() != prior["corpus"]:
        raise ValueError("Parent corpus identity changed")
    exclusions = set(prior["previous_test_exclusions"])
    exclusions.update(prior["task_manifests"]["test"]["semantic_ids"])
    new_tasks = generate_tasks("test", prior["settings"]["task_test"], seed=1879,
        number_range=prior["settings"]["task_number_range"], exclude_ids=exclusions)
    task_groups = {}
    for split in ("train", "valid"):
        from .tasks import Task
        task_groups[split] = [Task(**json.loads(line)) for line in
                             (parent / f"tasks_{split}.jsonl").read_text().splitlines()]
        if task_manifest(task_groups[split]) != prior["task_manifests"][split]:
            raise ValueError("Parent task manifest changed: " + split)
    task_groups["test"] = new_tasks
    forbidden_offsets = prior["test_offsets"] + [p["offset"] for p in prior["text_prompts"]]
    original_pilot = parent.parent / "rewt_pilot_20261009_v2" / "plan.json"
    if original_pilot.exists():
        original = _read(original_pilot)
        forbidden_offsets += original["test_offsets"] + [p["offset"] for p in original["text_prompts"]]
    text_seed = 1880
    while True:
        prompts = corpus.text_prompts(count=len(prior["text_prompts"]), seed=text_seed)
        if all(abs(p["offset"] - old) > 128 for p in prompts for old in forbidden_offsets):
            break
        text_seed += 1
    test_offsets = nonoverlapping_offsets(corpus, "test", len(prior["test_offsets"]),
        128, 1881, forbidden_offsets + [p["offset"] for p in prompts])
    out.mkdir(parents=True)
    runs.mkdir(parents=True)
    ledger = []
    for seed in prior["settings"]["seeds"]:
        for variant in prior["variants"]:
            for phase in ("language", "reasoning"):
                if variant == "full_rewt" and phase == "reasoning":
                    continue
                old = Path(prior["runs"]) / f"seed_{seed}" / variant / phase
                summary = _read(old / "training.json")
                expected_updates = prior["settings"]["lm_updates" if phase == "language" else "reason_updates"]
                if summary["updates"] != expected_updates:
                    raise ValueError("Incomplete fit cannot be reused: " + str(old))
                target = runs / f"seed_{seed}" / variant / phase
                shutil.copytree(old, target)
                cp = torch.load(old / "best.pt", map_location="cpu", weights_only=True)
                ledger.append(dict(seed=seed, variant=variant, phase=phase,
                    original_directory=str(old), original_checkpoint=str(old / "best.pt"),
                    copied_checkpoint=str(target / "best.pt"), checkpoint_sha256=file_sha256(old / "best.pt"),
                    training_record_sha256=file_sha256(old / "training.json"),
                    source_sha256=cp["source_sha256"], original_plan_sha256=cp["plan_sha256"],
                    updates=summary["updates"], training_cpu_seconds=summary["training_cpu_seconds"]))
    parity = verify_reused_inference(parent, ledger)
    plan = dict(prior)
    plan.update(version=3, created=datetime.now(timezone.utc).isoformat(), out=str(out), runs=str(runs),
        task_files_authoritative=True, task_test_selection_seed=1879, text_selection_seed=text_seed,
        previous_test_exclusions=sorted(exclusions), test_offsets=test_offsets, text_prompts=prompts,
        forbidden_evaluation_offsets=forbidden_offsets,
        task_manifests={split:task_manifest(group) for split,group in task_groups.items()},
        parent_stopped_report=str(parent), parent_plan_sha256=prior["plan_sha256"],
        parent_source_sha256=prior["source_sha256"], reused_fits=ledger,
        reused_inference_parity=parity,
        restarted_fits=[dict(seed=seed,variant="full_rewt",phase="reasoning",reason="Training-only temporal-gradient boundary every 16 tokens")
                        for seed in prior["settings"]["seeds"]],
        protocol="27 completed parent fits reused with exact weights/source provenance; all three full reasoning fits restart from course-trained language weights; fresh test and text samples declared before restarted fits")
    # JSON persistence sorts keys; the teacher must be fitted/loaded before the
    # full student uses its reasoning targets and preparation-cost record.
    plan["variants"] = {name:dict(prior["variants"][name]) for name in LABELS}
    v = plan["variants"]["full_rewt"]
    v["recipe"] = {**v["recipe"], "state_gradient_interval_reasoning":16}
    plan["source_sha256"] = {p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    plan["plan_sha256"] = digest({k:v for k,v in plan.items() if k != "plan_sha256"})
    shutil.copytree(Path(__file__).parent, out / "source", ignore=shutil.ignore_patterns("__pycache__", "*.md"))
    if (parent / "screen").exists():
        shutil.copytree(parent / "screen", out / "screen")
    for name in ("screening_cost.json", "process_training_labels.jsonl"):
        if (parent / name).exists():
            shutil.copy2(parent / name, out / name)
    for split, group in task_groups.items():
        save_lines(out / f"tasks_{split}.jsonl", [t.to_dict() for t in group])
    save_json(out / "text_prompts.json", prompts)
    save_json(out / "reused_fits.json", ledger)
    save_json(out / "reused_inference_parity.json", parity)
    save_json(out / "plan.json", plan)
    save_json(out / "selected_protocol.json", plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="reports/rewt_improvements_20261009")
    parser.add_argument("--out", default="reports/rewt_stabilized_20261009")
    parser.add_argument("--runs", default="runs/rewt_stabilized_20261009")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("Positive worker count required")
    torch.set_num_threads(1)
    plan = prepare(args)
    event("stabilized_protocol_declared", reused_fits=len(plan["reused_fits"]),
          restarted_fits=len(plan["restarted_fits"]), out=plan["out"])
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=min(args.workers,len(plan["settings"]["seeds"])),
        mp_context=context, initializer=worker_setup, initargs=(plan["settings"]["threads"],))
    try:
        futures = [pool.submit(fit_seed, plan, seed) for seed in plan["settings"]["seeds"]]
        for future in as_completed(futures):
            event("stabilized_seed_complete", **future.result())
    except BaseException as error:
        save_json(Path(plan["out"]) / "failure.json", dict(error=repr(error),
            time=datetime.now(timezone.utc).isoformat()))
        for process in list(pool._processes.values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    subprocess.run([sys.executable, "-m", "rewt.profile", plan["out"]], check=True,
        env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    from .improve_report import create_report
    create_report(plan)
    save_json(Path(plan["out"]) / "complete.json", dict(completed=datetime.now(timezone.utc).isoformat(),
        seeds=plan["settings"]["seeds"],all_primary_fits_complete=True,
        reused_fits=len(plan["reused_fits"]), restarted_fits=len(plan["restarted_fits"])))
    event("stabilized_experiment_complete", report=str(Path(plan["out"]) / "REPORT.md"))


if __name__ == "__main__":
    main()
