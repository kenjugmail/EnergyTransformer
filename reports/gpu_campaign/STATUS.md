# Latest update: hybrid precision repair verified

All18 corrected CUDA numerical cases and exact GPU restart passed. Installed
wheel:533 passed,51 skipped. Study002 completed one longer fit, then CUDA context creation
failed with out-of-memory before the next fit. Numerical repair uses explicit
first energy projection FP32. Original failed study001 remains retained. See
[results and status](hybrid_study/REPORT.md).

# Latest update: hybrid GPU numerical gate failed

17/18 actual-size CUDA cases passed. Hybrid 3.33M seed 606 failed BF16
gradient agreement while the identical saved fixture passes on CPU. Training
did not begin; the stop record is retained and an arithmetic/backend audit is
required. See [measurements](hybrid_study/REPORT.md).

# Latest update: replicated hybrid study prepared

The user selected hybrid energy/local as the next scaling axis. All 18 real-size
CPU numerical cases passed; GPU acceptance, deterministic restart and 18 longer
independent fits are waiting for 20 GiB available host memory. See
[hybrid study protocol and status](hybrid_study/REPORT.md). The existing hourly
follow-up now targets that bounded shared study, preserving the same ledger.

# Prior update: shared-device probe completed

Six preliminary hybrid/attention fits completed under the 4 GiB Torch cap; peak
reservation was 252 MiB. Pure energy remains blocked by a reproduced BF16 gradient
failure. See [measured results](shared_probe/REPORT.md). The exclusive campaign
has not run. The records below describe the earlier preparation baseline.

# Integrated campaign status

The implementation and CPU preparation are complete enough to begin GPU acceptance.
The 24-hour GPU study has **not run**. The idle-device guard returned
`WAITING_FOR_EXCLUSIVE_GPU`; existing VLLM and other compute workloads were left
running. No campaign GPU time has been consumed and no new architecture-quality
claim is established.

## Implementation

Isolated checkout: `/home/kt/Documents/EnergyTransformer-gpu-campaign`, branch
`codex/gpu-scaling-campaign`, forecasting base
`c5583551eb68b89c21337a0cc639e4c1bfad0b00`. The original checkout's 380 preserved
source/report files were audited unchanged. Historical checkpoints remain there.

Implemented the campaign specification, shared stage/substage budget ledger,
resumable subprocess runner, exact-source acceptance checks, profiling-driven
replicated native grid, forecast extraction and joint-training controls, pinned
SmolLM2 conversion with mixed dynamic/preallocated caches, evaluation/recall tools,
and guarded scaling analysis. The current verified recurrence is retained; no
fused-kernel speedup is claimed. See [runbook](README.md) for commands and limits.

## Prepared inputs

- FineWeb-Edu `sample-10BT`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`: 268,432,581 bytes in 49,953
  whole documents, just under the 256 MiB cap. Duplicate groups stay together
  across deterministic train/validation/test splits. The split has 44,855/2,561/2,537
  documents. MinHash/LSH near-duplicate detection is approximate, not a guarantee
  that all near duplicates were detected.
- SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` downloaded;
  tokenizer caches contain 53,107,881/2,991,255/2,963,062 train/validation/test tokens.
- Input files, attribution/source IDs, extraction order, hashes, pinned donor
  files and logs are retained in `runs/campaign/`. Large data and model files are
  excluded from Git.

## CPU evidence

Source suites passed: native 301 passed/32 skipped; forecasting 108/18;
campaign 27/0; independent Spark reference 84/1. Total **520 passed, 51 skipped**.
Skips are retained explicitly; they are not successful GPU checks.

CPU compiled/eager gradients differed by at most 4.02e-6 relative, with maximum
loss difference 4.77e-7. A deterministic six-update fresh-process restart reproduced
weights, optimizer state, RNG and sample hashes exactly. The actual 135M donor
adapter produced finite CPU logits, with whole/stream maximum absolute difference
8.78e-5. These are plumbing/numerical checks, not retained-language-quality evidence.

The Torch build can initialize CUDA from CPU optimizer/compiler code. CPU tests
therefore mask CUDA. Initial unmasked OOM failures and test-harness import failures
are preserved alongside successful reruns. Neither numerical tolerances nor GPU
acceptance requirements were relaxed.

The installed-wheel rerun also passed the native, forecasting and campaign suites;
the independent Spark source suite passed alongside them (520 passed, 51 skipped
in total). All 100 packaged Python files match the checkout byte for byte.
Installed-wheel results and source/file hashes are recorded separately in
`runs/campaign/installation_verification.json` and `cpu_verification.json`.
The independent Spark tests exercise its source package; it remains a separate
reference, not a GPU receipt for `energy_real`.

## Remote CI

The inspected GitHub job never executed tests. Its annotation identifies failed
account payments or a spending limit. The annotation is preserved in
`remote_ci_annotations.json`; remote CI is not green and account billing was not
changed.

## Pending measured deliverables

GPU acceptance, real-shape profiling, the native fits, frozen and joint forecast
results, conversion recovery, cache/latency measurements, final verification,
curves and measured follow-on costs remain pending. CPU throughput is not substituted
for GB10 measurements. The scheduler will choose the common target budget only
from verified GPU profiles, retaining seed replication before larger sizes.

The historical 10.6M GPU comparison favors full attention (1.6633 versus 1.7240
validation bits per byte at 40.96M training bytes). The later tiny CPU hybrid result
is separate evidence. Neither establishes that the new model beats a Transformer.

An hourly follow-up in this task checks device availability. It must use this
checkout and the existing `runs/campaign` ledger, avoid duplicate launches, remain
quiet on unchanged contention, and stop for audit after a failed experiment. No
larger campaign, checkpoint promotion or PR merge is authorized by a favorable
pilot. Keep the desktop app/host available for the follow-up to execute.

The subsequent [scaling readiness audit](SCALING_READINESS.md) corrected profiling,
resource monitoring, provenance and analysis controls. The previous unstarted
source baseline is preserved under `runs/campaign/revisions/`; no charged ledger
was reset. Current installed-wheel verification totals 520 passed and 51 skipped.
