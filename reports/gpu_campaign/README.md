# Integrated GPU campaign

This checkout combines the preserved local native/hybrid implementation with the
forecast branch pinned to `c5583551eb68b89c21337a0cc639e4c1bfad0b00`.
The original checkout and its checkpoints remain unchanged. The original-source
manifest is `native_manifest.json`; its archive is in the original checkout under
`runs/gpu_campaign_preservation/`.

## Run the campaign

Use this checkout's `.venv/bin/python`. Its isolated dependency environment imports
the existing CUDA-enabled Torch installation; it does not replace that installation.
The optional installation group is `.[test,forecast,campaign]`.

```bash
.venv/bin/python -m energy_campaign doctor
.venv/bin/python -m energy_campaign run --root runs/campaign
```

`run` returns `WAITING_FOR_EXCLUSIVE_GPU` if another compute process is present.
It never terminates a foreign process. CPU tests do not produce a GPU acceptance
receipt. A current CPU verification receipt, pinned corpus, donor files, native GPU
acceptance, and matching shape profiles are required before production training.

For a different output location, prepare the corpus and copy the pinned donor plus
its identity manifest before launching. The existing original 10.6M checkpoint is
read from `/home/kt/Documents/EnergyTransformer/runs/enwik8_energy_10m/best.pt`.

The campaign is staged: acceptance, profiling, validation-only LR selection,
matched-data grid, matched-time comparisons, frozen forecasting, joint training,
pretrained conversion, final evaluation and separately supervised recall curves.
The scheduler generates frozen job JSON files from real profiles; it does not invent
throughput or preselect the common data budget.

## Limits and restart behavior

The shared ledger caps acceptance at 2 hours, backbone work at 12 hours, forecasting
at 4 hours, conversion at 4 hours, and verification at 2 hours. Substage caps further
restrict calibration, matched-data/time, forecast heads/joint/evaluation, and
transfer/recovery/control/evaluation. Imports, compilation, failures, saves and
validation inside a supervised GPU process count. The supervisor reserves time
before starting the process; an unresolved reservation remains fully charged.

Native training saves optimizer, RNG, sampler coverage and committed log state.
Head, joint and conversion fits retain full resume checkpoints. Job identity binds
source, corpus, architecture and schedule. Changed identities require new runs.
A killed process cannot silently refund an unmeasured GPU interval. Review an
unsettled reservation before explicit recovery; do not edit it to invent free time.

A checkpoint from an interrupted matched-data fit is not a completed comparison.
The campaign stops on a failed subprocess. It also stops if a minimum replicated
calibration/grid cannot fit. This is a result, not authorization to increase budgets.
No checkpoint is promoted and no PR is merged automatically.

## Numerical and scientific boundaries

The differentiable summary trace is currently a serial correctness reference.
Its overhead is measured before the joint stage; no fused GPU trace is claimed.
Recurrent coefficients remain FP32. The default fixed `dmat` backend supports strong
forgetting; the restricted `matmul` backend retains its validity checks.

The forecast head selects the last ENERGY layer in a hybrid, never an attention
cache. Heads and normalization belong to an exact teacher checkpoint. Observed
state is not replaced with a predicted future. A zero residual initially preserves
language logits. Frozen teacher targets enter only the auxiliary objective.

The conversion adapter supports unpadded document-contained causal streams.
Arbitrary packed masks and recurrent-cache rollback are rejected. Remaining attention
layers still retain context-growing caches. The donor tokenizer is unchanged.

Native byte loss, summary-density NLL, symbolic recall, and pretrained subword loss
are different measurements. Profiling reports SDPA operators and partial profiler
FLOPs; it does not call these complete custom-kernel FLOPs. Scaling-law fitting
rejects correlated checkpoint samples and incomplete crossed grids. Pilot curves
are not evidence for extrapolation to billion-parameter models.

## Tests

Set `CUDA_VISIBLE_DEVICES=""` for all CPU-only checks. This Torch build may otherwise
initialize a CUDA context inside CPU optimizers/compiler code. Run suites separately because historical tests reuse module basenames and import
siblings directly:

```bash
.venv/bin/python -m pytest -q tests tests_real
.venv/bin/python -m pytest -q forecast_tests
.venv/bin/python -m pytest -q tests_campaign
PYTHONPATH=spark_kit .venv/bin/python -m pytest -q spark_kit/tests
```

GPU acceptance independently compares the current kernel against a serial recurrence,
checks streaming/reset/causality and FP32 state, BF16 engineering tolerances,
activation-checkpoint gradients, compiled/eager gradients, and fresh-process exact
optimizer/RNG/sample restart. Shape-specific profiles are additionally required.
Missing historical binary artifacts stay explicit skips in source-only tests.

## Scaling audit

See [scaling readiness](SCALING_READINESS.md) for corrected controls, current limits,
and the fully enumerated (unfunded) scaling/slot study proposal. The read-only
`readiness` command validates prepared source, corpus and donor identities.
