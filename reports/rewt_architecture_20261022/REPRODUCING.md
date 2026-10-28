# Reproducing the architecture study

Run from `/home/kt/Documents/EnergyTransformer` using `.venv/bin/python`.
The checked artifacts include all three seeds and every declared variant,
including failures. `plan.json` and the plans in `language`, `long_memory`,
`binding_memory` and `memory_ablation` define the data, update budgets and
selection rules. Checkpoint selection uses validation only.

## Verification

For CPU jobs on this shared machine, hide CUDA explicitly. This PyTorch build's
AdamW can initialize the accelerator while performing a CPU optimizer health
check; leaving CUDA visible caused one transient out-of-memory failure before
the first update. The failed log is retained, and only the missing fit was rerun.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest tests_rewt -q
.venv/bin/python scripts/verify_rewt_architecture.py
.venv/bin/python scripts/verify_rewt_training_replay.py
.venv/bin/python scripts/audit_rewt_architecture_models.py
.venv/bin/python scripts/audit_rewt_refinement_contribution.py
.venv/bin/python scripts/audit_rewt_language_documents.py
.venv/bin/python scripts/summarize_rewt_architecture.py
.venv/bin/python scripts/plot_rewt_memory_study.py
```

`baseline_source` preserves the package before this turn. `training_source` and
`training_script.py` preserve the initial architecture training implementation.
The final replay checks three optimizer updates for each of all 13 variants,
including every parameter gradient and post-update weight. Later opt-in
execution additions do not change the frozen training paths. The final source
archive and manifest identify the exact delivered implementation.

## Study runners

The scripts use the dated report/run paths declared at the top of each file.
Do not delete or silently overwrite the recorded experiments. For an independent
rerun, use a separate checkout/output directory and preserve the declared plans,
data and tokenizer. `--prepare` refuses to replace an existing plan. Completed
fit records are reused; interrupted partial training is not automatically
resumed. The examples below operate on these already declared plans.

```bash
.venv/bin/python scripts/study_rewt_architecture.py --train --workers 4
.venv/bin/python scripts/study_rewt_architecture.py --test --workers 4
.venv/bin/python scripts/study_rewt_architecture_language.py --train --workers 2
.venv/bin/python scripts/study_rewt_architecture_language.py --test --workers 2
.venv/bin/python scripts/study_rewt_long_memory.py --train --workers 2
.venv/bin/python scripts/study_rewt_long_memory.py --test --workers 2
.venv/bin/python scripts/study_rewt_binding_memory.py --train --workers 2
.venv/bin/python scripts/study_rewt_binding_memory.py --test --workers 2
.venv/bin/python scripts/study_rewt_memory_ablation.py --train
.venv/bin/python scripts/study_rewt_memory_ablation.py --test
.venv/bin/python scripts/study_rewt_equal_compute.py --train --workers 4
.venv/bin/python scripts/study_rewt_equal_compute.py --test --workers 4
.venv/bin/python scripts/package_rewt_language_checkpoints.py
```

The main and language runners require every fit to finish before opening the
test set. Main generation outputs, language samples and all memory predictions
are saved as JSONL. The original memory recipe, its failed long-gap results,
the binding-supervised follow-up, and the matched 2×2 ablation are all retained.
The final ablation reuses six frozen supervised checkpoints and trains six
response-only controls with the same seeds, batches and 1,536 updates. Its third
fresh test set avoids selecting a follow-up from repeated test queries.

`equal_compute/plan.json` defines a separate 15-fit comparison at 150 measured
training CPU seconds per fit. Four validation checkpoints and the learning-rate
schedule follow consumed CPU budget. Its 256 test documents exclude every
document in the earlier language test; actual update counts and one-update
budget overshoot are reported. This is a CPU time comparison, not equal FLOPs
or a GPU performance result.

Language training uses the existing official TinyStories training partition
and its training-only vocabulary. New holdout documents come from a separate
byte range of the pinned official validation file. `data/rewt_architecture_20261022`
contains the original range, HTTP/source metadata, normalized split indexes and
checksums. Partial documents at range boundaries and all old document IDs are
excluded. Evaluation divides token negative log-likelihood by actual decoded
target-byte counts.

## Performance

```bash
.venv/bin/python scripts/benchmark_rewt_architecture.py \
  --cpu 5 --repeats 7 --output benchmark_final.json
.venv/bin/python scripts/benchmark_rewt_directional.py
.venv/bin/python scripts/plot_rewt_architecture_results.py
.venv/bin/python scripts/write_rewt_architecture_findings.py
.venv/bin/python scripts/archive_rewt_architecture_study.py
```

Execution benchmarks alternate variants, warm each case, use a single CPU
thread, and report process and wall times. Training times come from concurrent
single-thread workers and are not isolated hardware speed comparisons. Benchmark
loss is a deterministic logit-square scalar; it measures forward/backward
execution, excluding the optimizer and task-specific auxiliary losses.

On CUDA, run from a shell where `CUDA_VISIBLE_DEVICES` exposes the intended GPU:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/check_rewt_architecture_cuda.py
OMP_NUM_THREADS=1 .venv/bin/python scripts/check_rewt_triton_tiles.py
OMP_NUM_THREADS=1 .venv/bin/python scripts/inspect_rewt_triton_resources.py
```

These checks disable TF32 and record float32 numerical error, wraparound,
prefill restoration and graph replay. The GPU was heavily occupied by unrelated
work during this study. Recorded GPU timings are diagnostic, not evidence of an
uncontended speedup. Compiler register/spill counts are resource reports, not
measured occupancy. The separate interpreter check validates index/formula logic:

```bash
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/check_rewt_triton_interpreter.py
```

Triton is optional for ordinary PyTorch training and CPU inference. The custom
kernel supports contiguous CUDA float32 inference tensors only and rejects
autograd. No model-quality conclusion follows from numerical kernel parity.
