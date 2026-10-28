"""Isolated CPU timing and sampled native process memory for saved checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from dataclasses import replace

import torch

from .data import Corpus
from .evaluate import model_output, runtime_profile
from .experiment import load_model, save_json
from .model import JointEnergyLM,build_model


def resident_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmRSS unavailable")


def profile_checkpoint(checkpoint, data, offsets, batch=4, *, rounds=None, halting=0., disable_efficient=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    corpus = Corpus(data)
    model, payload = load_model(checkpoint)
    if isinstance(model,JointEnergyLM) and (rounds is not None or disable_efficient):
        config=replace(model.config,rounds=rounds if rounds is not None else model.config.rounds,
                       halting_tolerance=halting,
                       cache_history=False if disable_efficient else model.config.cache_history,
                       early_backtrack=False if disable_efficient else model.config.early_backtrack)
        adjusted=build_model(payload["arm"],config)
        adjusted.load_state_dict(model.state_dict(),strict=True)
        model=adjusted
    del payload
    model.eval()
    tokens = corpus.batch("test", None, batch, 128, starts=offsets[:batch]).tokens
    gc.collect()
    before, peak, stopped = resident_bytes(), [resident_bytes()], threading.Event()
    def sample():
        while not stopped.wait(.002):
            peak[0] = max(peak[0], resident_bytes())
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with torch.inference_mode():
            model_output(model, tokens, torch.Generator().manual_seed(550))
        peak[0] = max(peak[0], resident_bytes())
    finally:
        stopped.set()
        thread.join()
    process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    # Operator profiler allocations are gathered AFTER the RSS measurement.
    # Profiler bookkeeping can be large and must not inflate inference RSS.
    profile = runtime_profile(model, tokens, repeats=3)
    profile.update(isolated_process_id=os.getpid(), batch=batch, context=128,
        rounds=getattr(model.config,"rounds",None),halting_tolerance=getattr(model.config,"halting_tolerance",0.),
        cache_history=getattr(model.config,"cache_history",False),early_backtrack=getattr(model.config,"early_backtrack",False),
        rss_before_inference_bytes=before, sampled_inference_peak_rss_bytes=peak[0],
        sampled_incremental_inference_rss_bytes=max(0, peak[0] - before),
        process_peak_rss_before_profiler_bytes=process_peak,
        rss_note="Linux process RSS sampled every 2 ms in a fresh process; includes Python/Torch/model/data buffers, excludes training and operator-profiler bookkeeping.")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--rounds",type=int)
    parser.add_argument("--halting",type=float,default=0.)
    parser.add_argument("--disable-efficient",action="store_true")
    args = parser.parse_args()
    out = Path(args.directory).resolve()
    plan_path=out/"selected_protocol.json" if (out/"selected_protocol.json").exists() else out/"plan.json"
    plan = json.loads(plan_path.read_text())
    if args.checkpoint:
        profile = profile_checkpoint(args.checkpoint, plan["corpus"]["path"], plan["test_offsets"],
                                     batch=min(plan["settings"]["batch_size"], len(plan["test_offsets"])),
                                     rounds=args.rounds,halting=args.halting,disable_efficient=args.disable_efficient)
        save_json(args.output, profile)
        return
    profiles = {}
    for seed in plan["settings"]["seeds"]:
        for arm in (plan["variants"] if "variants" in plan else plan["configurations"]):
            checkpoint = Path(plan["runs"]) / f"seed_{seed}" / arm / "language" / "best.pt"
            target = out / "profiles" / f"{arm}_{seed}.json"
            if not target.exists():
                command=[sys.executable,"-m","rewt.profile",str(out),"--checkpoint",str(checkpoint),"--output",str(target)]
                record_path=out/f"seed_{seed}"/f"{arm}.json"
                if record_path.exists():
                    policy=json.loads(record_path.read_text()).get("language_policy",{})
                    if policy.get("rounds") is not None:
                        command += ["--rounds",str(policy["rounds"]),"--halting",str(policy["halting_tolerance"])]
                subprocess.run(command, check=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            profiles[f"{arm}_{seed}"] = json.loads(target.read_text())
    save_json(out / "profiles" / "all.json", profiles)


if __name__ == "__main__":
    main()
