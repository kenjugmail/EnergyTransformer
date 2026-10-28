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

import torch

from .data import Corpus
from .evaluate import model_output, runtime_profile
from .experiment import load_model, save_json


def resident_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmRSS unavailable")


def profile_checkpoint(checkpoint, data, offsets, batch=4):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    corpus = Corpus(data)
    model, payload = load_model(checkpoint)
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
    args = parser.parse_args()
    out = Path(args.directory).resolve()
    plan = json.loads((out / "plan.json").read_text())
    if args.checkpoint:
        profile = profile_checkpoint(args.checkpoint, plan["corpus"]["path"], plan["test_offsets"],
                                     batch=min(plan["settings"]["batch_size"], len(plan["test_offsets"])))
        save_json(args.output, profile)
        return
    profiles = {}
    for seed in plan["settings"]["seeds"]:
        for arm in plan["configurations"]:
            checkpoint = Path(plan["runs"]) / f"seed_{seed}" / arm / "language" / "best.pt"
            target = out / "profiles" / f"{arm}_{seed}.json"
            if not target.exists():
                subprocess.run([sys.executable, "-m", "rewt.profile", str(out),
                    "--checkpoint", str(checkpoint), "--output", str(target)], check=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            profiles[f"{arm}_{seed}"] = json.loads(target.read_text())
    save_json(out / "profiles" / "all.json", profiles)


if __name__ == "__main__":
    main()
