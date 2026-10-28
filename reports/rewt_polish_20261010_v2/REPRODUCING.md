# Reproducing the broader-task REWT polish

Run commands from `/home/kt/Documents/EnergyTransformer` using the existing
`.venv`. The study uses PyTorch on CPU, one thread per worker. Its saved plans,
checkpoints, source snapshots and generated responses are the evidence for the
executed results. Training starts from random weights for the cold controls;
the two warm head-count variants continue the previous position-memory
checkpoint. Warm results carry that earlier training lineage.

## Use the existing one-step checkpoint

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/binding/aligned/best.pt \
  --prompt $'A=19;B=A+23;B=?\n' \
  --greedy --stop-newline --max-new 40 --cache-history
```

The recorded command generates `steps=42;ans=42`.
[Actual CLI output](legacy_cached_cli_sample.json) includes exact bytes and the
checkpoint hash. `--cache-history` changes execution by caching causal observed
encodings within a forward call, and copies the same learned weights. It adds
no parameters or labels to inference. The sampler also accepts `--prompt-file`
and `--json`; `--rounds 0` provides the same-weight refinement counterfactual.

The broader model and its exact selected path are recorded in
[the frozen test plan](test/plan.json). The one-step model remains available
because the broader curriculum does not solve extended arithmetic reliably.

## Inspect the executed study

The [main plan](curriculum/plan.json) fixes 2,048 updates, batch size 24,
128-byte windows, seeds 2101/2102/2103, and approximately 256K inference
parameters. There are three cold energy fits, six cold Transformer fits across
two learning rates, and two warm continuations. The main training script is
saved as `curriculum/training_script.py`, with the complete package source in
`curriculum/source/`.

The [termination plan](termination/plan.json) adds 1,024 updates to each
validation-selected main checkpoint, with a fresh AdamW optimizer and half the
preceding learning rate. It applies the same stronger syntax/final-newline and
remaining-step losses to both architectures, and adds two warm controls using
the existing absolute-wave setting. Its script and source are saved alongside
the plan. Checkpoints may select an earlier update; the cost ledger charges
the full executed budget.

All model selections precede fresh test generation. Transformer learning rate
is selected using the mean validation score across three seeds; warm case
selection uses validation complete accuracy. The test then evaluates all three
cold energy seeds and all three Transformer seeds at the selected rate, the
selected warm model with two and zero rounds, and the previous warm parent.
The checker runs after raw-byte generation. Auxiliary training heads and task
payloads are not passed to inference.

## Train an independent follow-up

Choose new report and run directories. The training programs reject a directory
containing a plan, preserving both completed and partial runs. The generator
also excludes every earlier declared test identity found under `reports/`;
an independent follow-up therefore has a new test set and is not expected to
reproduce the exact saved percentages.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/train_rewt_curriculum.py \
  --output-dir reports/my_curriculum --run-dir runs/my_curriculum \
  --updates 2048 --workers 6
.venv/bin/python scripts/refine_rewt_termination.py \
  --base-study reports/my_curriculum \
  --output-dir reports/my_termination --run-dir runs/my_termination \
  --updates 1024 --workers 6
.venv/bin/python scripts/evaluate_rewt_curriculum.py \
  --study-dir reports/my_termination \
  --output-dir reports/my_task_test --workers 4
.venv/bin/python scripts/summarize_rewt_curriculum.py \
  --study-dir reports/my_termination --test-dir reports/my_task_test \
  --output-dir reports/my_task_summary
```

Do not modify package source or training scripts while fits are running.
Checkpoint and source hashes are checked before final results are written.
There is no mid-run resume in these two standalone training scripts; the saved
`last.pt` contains optimizer state for inspection, while `best.pt` contains the
validation-selected inference weights and training-only heads.

`train_rewt_curriculum.py --smoke` checks a backward pass for both architectures
without evaluating fresh tests. `--updates 2` with fresh temporary directories
checks the full training/selection harness. The executed completion smoke
covered all 13 cases; a separate evaluation smoke used validation examples.

## Reproduce execution measurements

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/benchmark_rewt_memory.py \
  --checkpoint runs/rewt_learning_20261010/multistep_balanced/progress/best.pt \
  --output /tmp/rewt_memory_benchmark.json --batch 24 --context 64 --repeats 5
.venv/bin/python scripts/benchmark_rewt_generation.py \
  --checkpoint runs/rewt_learning_20261010/binding/aligned/best.pt \
  --prompt $'A=19;B=A+23;B=?\n' --max-new 40 \
  --output /tmp/rewt_generation_benchmark.json
```

The first benchmark measures forward and forward-plus-backward work without
optimizer updates. The second measures actual short-answer generation,
including prompt processing. Their speedups have different workloads and
should be reported separately. Both compare clones of the same weights.
[Recorded batch measurement](memory_benchmark.json),
[recorded generation measurement](generation_benchmark.json),
[exact source used for that generation measurement](generation_benchmark_script.py).

The separate analytic-gradient compilation probe is in
`scripts/benchmark_rewt_compile.py`; its [result](compile_benchmark.json)
reports compilation warmup separately and checks prediction and parameter
gradient tolerances. It did not change the training execution in this study.

The implementation checks run with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests_rewt -q
```

The saved generation records support auditing every complete-answer score.
`summarize_rewt_curriculum.py` rechecks the response bytes, prompts and checker
results, then produces the summary JSON and exportable cold-seed figure without
running model inference again.
