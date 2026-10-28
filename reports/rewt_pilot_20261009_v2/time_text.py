"""Exploratory text check on ALREADY trained equal-time models; no new fitting.

Uses the first ten prefixes from the frozen prompt list, irrespective of model
outputs. The sampling stream exactly matches the primary first-ten group.
"""
import json
from pathlib import Path
import random

import torch

from rewt.evaluate import text_evaluation
from rewt.experiment import load_model, save_json, save_lines


ROOT = Path(__file__).resolve().parent
ARMS = ("transformer", "neural_energy", "fourier_rewt", "full_rewt")


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = json.loads((ROOT / "plan.json").read_text())
    collected, metrics = [], {}
    for seed in plan["settings"]["seeds"]:
        for arm in ARMS:
            checkpoint = Path(plan["runs"]) / f"seed_{seed}" / arm / "equal_cpu_time" / "best.pt"
            model, payload = load_model(checkpoint)
            summary, records = text_evaluation(model, plan["text_prompts"][:10], seed=seed+563)
            metrics[f"{arm}_{seed}"] = summary
            save_lines(ROOT / "time_text" / f"{arm}_{seed}.jsonl", records)
            collected.extend(dict(arm=arm, seed=seed, **record) for record in records)
    random.Random(609).shuffle(collected)
    cases, key = [], []
    for index, record in enumerate(collected):
        case_id = f"T{index:03d}"
        cases.append(dict(id=case_id, prompt=record["prompt"], text=record["generation"]))
        key.append(dict(id=case_id, arm=record["arm"], seed=record["seed"], prompt_id=record["prompt_id"]))
    save_json(ROOT / "time_text" / "cases.json", cases)
    save_json(ROOT / "time_text" / "key.json", key)
    save_json(ROOT / "time_text" / "metrics.json", metrics)
    save_json(ROOT / "time_text" / "protocol.json", dict(
        evidence_scope="Exploratory inference from the preplanned equal-time checkpoints, no new training or selection.",
        prompted_by="Primary 61,440-byte fits produced unusable sampled text; inspect whether a preplanned common-time comparison changes this.",
        prefix_ids=[p["id"] for p in plan["text_prompts"][:10]], sampling="Same primary seed+563, temperature 0.8, 96 bytes and paired uniforms.",
        count=len(cases), human_ratings="pending"))


if __name__ == "__main__":
    run()
