# Reproducible energy-memory and reasoning experiments

This package contains the repaired byte-LM implementation, full-state trainer,
fixed-backend kernels, small memory studies, and a separate local-model adapter.
It is an experimental implementation, not a competitive general-purpose model.

## Install and verify

From the repository root, using the existing virtual environment:

```bash
.venv/bin/python -m pip install -e '.[test]' -e './spark_kit[test]'
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests tests_real -q
(cd spark_kit && OMP_NUM_THREADS=2 ../.venv/bin/python -m pytest tests -q)
```

The original v21 binary is needed only for its separately marked integration
test and explicit original-checkpoint continuation. Numerical acceptance uses a
seeded model and reports that fact; it does not impersonate pretrained weights.

## Kernels and caches

`Config.kind` accepts `energy` and `softmax`. There is no language-model `delta`
alias. `Config.backend` accepts `dmat` (default) or `matmul`.

Both kernels factor the existing energy recurrence into a precision scan and
chunkwise state contraction. `dmat` forms causal within-chunk decay matrices and
supports strong forgetting. `matmul` is an explicitly selected optimization with
a checked log-span limit; crossing it stops the run. Data-dependent algorithm
switching was removed because it changed earlier rounding when future inputs
changed. The old throughput numbers must be remeasured for the chosen backend.

Energy state remains `(precision, information)` per layer. Softmax state is
`(keys, values, segment_ids)` so resets cannot read previous segments. An old
`(keys, values)` state is interpreted as a single segment. State is not mutated.

## Training and exact resume

The preserved local enwik8 file uses a 90M/5M/5M split. Its complete SHA-256 is
checked before training, and source/configuration/runtime identities are saved.
An alternative corpus fingerprint must be supplied explicitly with
`--data-sha256`; size checks still apply.

```bash
.venv/bin/python -m energy_real.train --kind energy --out runs/new_energy \
  --backend dmat --steps 1000 --max-seconds 1800 --max-tokens 8192000

# Same schedule, data, source, and runtime; restores optimizer and RNG as well:
.venv/bin/python -m energy_real.train --kind energy --out runs/new_energy \
  --backend dmat --steps 1000 --max-seconds 1800 --max-tokens 8192000 --resume

# A new optimizer trajectory, explicitly labelled as a warm start:
.venv/bin/python -m energy_real.train --kind energy --out runs/warm_energy \
  --warm-start runs/enwik8_energy_10m/best.pt --steps 1000 --max-seconds 1800
```

`last.pt` commits weights, optimizer, Torch/CUDA/NumPy RNG state, actual step and
token counts, best-checkpoint state, identities, elapsed time, and an exact log
prefix. Extra crash-tail bytes are preserved separately. Corrupt committed logs
or incompatible resume identities are rejected. `best.pt` is for evaluation;
historical weight-only files cannot recover the interrupted optimizer state.

CUDA commands refuse to start over an unrelated GPU workload. Allocation and
time limits are explicit. The trainer does not increase budgets on resume.

## Evaluation and profiling

```bash
.venv/bin/python -m energy_real.evaluation \
  runs/enwik8_energy_10m/best.pt runs/enwik8_softmax_10m/best.pt \
  --out runs/validation.json --freeze-selection

.venv/bin/python -m energy_real.evaluation \
  runs/enwik8_energy_10m/best.pt runs/enwik8_softmax_10m/best.pt \
  --split test --selection runs/validation.selection.json --out runs/test.json

.venv/bin/python -m energy_real.bench --out runs/cost.json \
  --include-reference --profile --max-seconds 1800
```

Evaluation resets state at each context window and counts every target once.
The frozen selection records checkpoint hashes, source identity, split boundaries,
byte limit, batch size, precision, device type and protocol. Test evaluation also
verifies the original validation report checksum; keep it beside its selection
manifest. Changed comparison settings require a fresh validation selection.
Each checkpoint report distinguishes planned targets from actually evaluated targets. Existing energy and
softmax checkpoints consumed different training budgets; their final scores are
descriptive, not a matched-budget winner test.

The benchmark records balanced per-round measurements for next-token training,
prefill, and decode at batch sizes 1/8 and contexts 512/2K/8K/16K. It records model
bytes, recurrent-state/KV bytes, peak allocations, parameter counts and dtype.
`--profile` saves an operator summary and Chrome trace. The memory ceiling stops
oversized shapes; CPU timing is not a substitute for the requested GPU results.

## Memory mechanism study

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m energy_real.memory_eval \
  --out runs/memory_baselines.json --per-family 10

OMP_NUM_THREADS=2 .venv/bin/python -m energy_real.memory_research \
  --out runs/controller_study --steps 60 --seeds 11 22 33 --max-seconds 900
```

This reuses `EditMemory` from the existing geometric-memory research. Operations
and addresses are supplied. The predictive objective concerns the next symbolic
value token **before** a write; it is not full-text language-model training.
The augmented objective adds delayed-query and edit-locality losses. Validation
selects weights before held-out testing, including the option of step zero.

Exact, diagonal, precision-aware, delta, preconditioned-delta and projected
controls cover updates, deletion, unknown facts, near addresses, overload,
distractors and multi-hop retrieval. The projected and preconditioned mechanisms
are treated as algebraic equivalence controls. Registry/provenance costs are
reported separately from raw state and cannot justify a misleading memory win.

## Separate practical adapter

```bash
.venv/bin/python -m energy_real.reasoning demo --out runs/memory_demo

# Uses the existing service; does not launch or modify LoopTest/llm-stack:
.venv/bin/python -m energy_real.reasoning evaluate --out runs/reasoning.json \
  --base-url http://127.0.0.1:8100 --score

# Separate diagnostic: predicted edits in disposable evaluation stores:
.venv/bin/python -m energy_real.reasoning evaluate --out runs/text_edits.json \
  --input-mode model_proposals --score
```

`FactStore.propose(text)` returns proposed edits and source references; it does
not write. `apply(reviewed_edits, expected_revision=..., request_id=...,
authorized=True)` is the caller's explicit reviewed action. `query(Query(...))`
returns evidence and either an answer or abstention. String queries have a narrow
local grammar (`Follow relation/relation from subject.`); broader interpretation
uses the configured service. This store is separate from the legacy protected
16-name/16-location ledger.

Approximate retrieval may return wrong or stale evidence. Those errors remain
visible in comparisons. Generation and multiple-choice scoring are separate.
The adapter records service identity, exact decoding settings, controller hash,
per-response model identity and supplied evidence. Arm order rotates across
episodes. Completed generation is saved before scoring, so a scoring outage
preserves the answer with scoring marked pending. Summaries count scored
episodes separately and retain generation accuracy, support accuracy, abstention
and latency even in partial runs. When
the service omits token usage, context tokens are `null`; bytes are reported
without calling them tokens. Unavailable service results are marked pending.

## Approved stage ceilings

```bash
.venv/bin/python -m energy_real.stages --out runs/device_stages \
  --stages smoke evaluation_profile

# Optional bounded wait; does not interrupt the current workload:
.venv/bin/python -m energy_real.stages --out runs/device_stages \
  --stages smoke evaluation_profile --wait-idle --max-wait-seconds 300
```

The persistent ceilings are 30 minutes for smoke checks, 90 minutes for checkpoint
evaluation/profiling, four hours for controllers, and two hours for reasoning:
eight GPU-hours maximum. The small controller study can run entirely on CPU.
Pending stages may be retried with the same ledger and unchanged source.
Failed, interrupted, incomplete or exhausted stages are not automatically rerun
or given extra time. SIGINT/SIGTERM stops the runner’s own child process group,
waits for cleanup, and persists elapsed time and interruption status. A successful
exit requires a valid completed report; missing, malformed or pre-existing output
cannot certify success. A source change requires a new stage directory; account
for time already used in earlier ledgers before assigning any remaining budget.
`--base-url` selects the service for both the reasoning health check and commands. No cloud spending, scale-up,
publication, or pretrained-model fine-tuning is performed by this runner.


## Shared recall and reasoning study (v2)

The audited v2.1 benchmark uses 384 training, 120 validation and 360 test episodes by
default. Entity namespaces and wording templates are disjoint. Each episode’s ID and
content remain stable when a caller requests a larger or smaller sample. Queries cover
one to three different relation types, corrections, missing intermediate links,
near-colliding addresses and capacity overload. Generated datasets and full
per-episode outputs remain under ignored `runs/` directories.

```bash
# Twelve seed-11 screens, then seeds 22/33 for each selected mechanism.
# CPU execution avoids occupying the GPU; the ledger still charges wall time.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m energy_real.stages --out runs/reasoning_v2_final_stages \
  --prior-ledger runs/reasoning_v2_stages/stages.json --stages controllers
```

The controller sweep compares precision energy and preconditioned delta at
widths 16/32/64 and priors 0.01/0.001, with 500 optimizer updates per fit. Training
minimizes delayed recall cross-entropy plus 0.1 times unrelated-read locality
loss. Every episode also queries an unseen address; deleted addresses use an
explicit null target. Initialization is a separately calibrated control.

Confidence calibration uses validation only: score thresholds 0.05/0.15/0.3/0.5/0.7
and top-two margins 0/0.02/0.1/0.2. Validation is checked at initialization and every
100 updates. Selection favors accuracy, fewer unsupported answers, then support
accuracy; tied configurations prefer smaller width, then the larger prior. Exact
ties retain initialization or the earlier training step. Calibration ties prefer
stricter score/margin thresholds. Test data is instantiated only after the final
selection manifest and controller checksums are frozen.

Per-hop traces show the top candidates, score, confidence margin, selected fact
and abstention. Evaluation distinguishes incorrectly ranked facts from correct
candidates rejected by the confidence threshold. Selected weights are evaluated
against their initialization and against the original 16-dimensional configuration
on the same fresh test episodes. These are structured-memory results; language
reasoning results are separate.

Cost reports include database bytes, serialized active facts/history and registry,
model parameters and recurrent tensors. Isolated Linux subprocesses measure full
resident peaks from `/proc/self/status` (`VmHWM`), including Python/native/SQLite
overhead and excluding inherited pre-exec high-water marks. Disk and resident bytes
are reported separately. Timings include writes, state build and repeated queries
with balanced arm order. CPU measurements do not establish a GPU efficiency win.

### Use the selected memory with the existing model service

After the study completes, its frozen selection is at
`runs/reasoning_v2_final_stages/controllers/attempt_0/study/selection.json`. The completed
report also provides a copy at `reports/reasoning_v2/selection.json`, with the small
dataset recipe/hash manifest beside it. Controller binaries remain in `runs/`.

```bash
.venv/bin/python -m energy_real.reasoning evaluate \
  --benchmark v2 --question-mode natural --gold-evidence --score \
  --memory-selection reports/reasoning_v2/selection.json \
  --out runs/reasoning_v2_natural.json
```

Use `--question-mode supplied_path` for a separately labelled traversal test.
Use `--input-mode model_proposals` to interpret statements into proposed edits in
disposable stores. Recent-context receives original source statements in that
mode, without oracle-parsed triples. The model determines the relation path from
the natural question once, shared by the exact and energy retrieval arms. An
invalid or wrong interpretation is recorded without substitution of the gold
path. The gold-evidence arm is explicitly diagnostic and receives the correct
supporting records independently of the retrieval arms.

`FactStore` additionally accepts `prior`, `score_threshold` and `margin_threshold`;
`query()` includes a `trace` array. Reviewed edit authorization and revision checks
retain their existing contract. Controller loading checks the mechanism, key
width, value capacity, prior and weights. Older 16-value controller checkpoints
require the matching explicit capacity; they cannot silently initialize a
256-record store. A frozen selection verifies its dataset and controller hashes.

The runner reports failures in edit interpretation, query interpretation, fact
retrieval, abstention and answer composition. Generation and multiple-choice
scoring retain separate counts, latencies and partial results. HTTP timeouts are
bounded by the remaining shared reasoning deadline.

### Remaining device and live stages

```bash
.venv/bin/python -m energy_real.stages --out runs/reasoning_v2_final_stages \
  --stages smoke evaluation_profile reasoning \
  --memory-selection reports/reasoning_v2/selection.json
```

This evaluates supplied-path and natural questions separately for structured and
model-proposed edits, within the same two-hour reasoning ceiling. The existing
service URL defaults to localhost:8100 and is configurable with `--base-url`.
Unavailable hardware/service remains pending; the runner does not stop another
workload or start the model server. `--prior-ledger` carries earlier usage into a
new source revision conservatively, without increasing the original ceilings.

The [v2 report](../reports/reasoning_v2/REPORT.md) records executed results and the
candidate decision. The [initial repair](../reports/repair/REPORT.md) and
[reliability pass](../reports/repair/POLISH.md) remain historical snapshots. Their
source-specific ledger commands predate this revision.

## Portable bundles and resumable evaluation

Export the completed selection and its verified controller files into an ignored
run directory. Historical selections and checkpoints remain unchanged:

```bash
.venv/bin/python -m energy_real.artifacts \
  --selection reports/reasoning_v2/selection.json --out runs/portable_v2

.venv/bin/python -m energy_real.reasoning evaluate \
  --benchmark v2 --question-mode natural --gold-evidence --score \
  --memory-selection runs/portable_v2/selection.json --out runs/portable_reasoning.json
```

The whole bundle may move to another directory or machine. Paths inside its
selection are relative, and every controller is verified against its frozen hash.
If metadata and binaries are stored separately, `--artifact-root /path/to/bundle`
resolves relative controller paths there. Keep `datasets.json` beside the selection.
Legacy absolute selections remain supported; export them before relocating.

Retry an interrupted evaluation with the identical command plus `--resume`.
Successful interpretations, edit proposals, generation and scoring calls are
persisted with episode/arm identity. Already completed answers are retained.
Source, dataset, checkpoint, declared model identity, decoding, scoring mode and
total time ceiling must match; elapsed time accumulates across attempts. A changed
model or configuration requires a new output. The stage runner retains completed
commands after service failures and resumes its unfinished reasoning command.

Build a new report from explicit inputs, without rewriting historical reports:

```bash
.venv/bin/python -m energy_real.reporting \
  --study runs/reasoning_v2_final_stages/controllers/attempt_0/study \
  --ledger runs/reasoning_v2_final_stages/stages.json \
  --test-log real=reports/reasoning_v2/tests_real.log \
  --test-log legacy=reports/reasoning_v2/tests_legacy.log \
  --test-log spark=reports/reasoning_v2/tests_spark.log \
  --out runs/rebuilt_v2_report
```

Report status and test totals come from those inputs. Supplying historical test
logs documents their evidence; it does not certify a different source revision.
Bundles, datasets, checkpoints and raw service traces stay outside source commits.
