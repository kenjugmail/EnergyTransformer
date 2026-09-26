# EnergyTransformer

Experimental energy-function memory, native-text prototypes, and explicitly reviewed
protected-memory edits. **Not a competitive general-purpose LLM or an instruction-following assistant.**

## Mixed energy and local attention

The native model now also supports alternating energy-recurrent blocks with
sliding-window attention. This combines a fixed recurrent state with a bounded
attention cache, and works with the existing optional fact readers and graph
workspace. Full-attention and windowed Transformer controls use the same codebase.

- [Architecture and training commands](reports/competitive/ARCHITECTURE.md)
- [Latest compute-controlled confirmation](reports/compute_confirmation/REPORT.md)
- [Kernel and preallocated-cache measurements](reports/stronger_comparison/REPORT.md)
- [Original token-matched comparison](reports/competitive/REPORT.md)

In the latest three-seed, approximately 145,000-parameter comparison, the mixed
model reached **2.512 test bits per byte versus 2.578 for full attention and
2.577 for windowed attention**, at equal steady-state CPU training budgets.
All arms use compiled training, the same tuning allowance, and one CPU core
with interleaved time blocks. Compilation and evaluation costs are reported
separately. All three hybrid seeds beat both controls at the trained context.

The CPU kernel is faster, and 32K-prefix decoding beats a preallocated full
Transformer cache. Full attention is faster at shorter prefixes; windowed
attention remains faster at every measured length. These small-model results
do not establish general superiority over Transformers or GPU efficiency.

Sample the saved checkpoint on CPU with the installed package:

```bash
.venv/bin/python -m energy_real.sample \
  --checkpoint runs/compute_confirmation_selected/hybrid.pt --prompt 'The ' \
  --tokens 128 --temperature 0.8 --top-k 40 --seed 42
```

Omit the sampling options for greedy output. Prompt processing uses bounded
chunks; `--output sample.bin` saves exact bytes to a new file. See the
[original usability follow-up](reports/native_polish/REPORT.md) and
[current installed-package verification](reports/compute_confirmation/installation_verification.json).

## Native model with explicit fact memory

The native model now supports a learned fact reader in every block, an exact
graph workspace with bounded programs and evidence traces, and an optional
PC-ALM training rule. Backpropagation remains the default. The hybrid session
invalidates recurrent state after committed fact edits. Existing checkpoints
can be augmented with zero-gated readers while preserving their original outputs.

- [Integrated architecture](reports/hybrid_integration/ARCHITECTURE.md)
- [Verification, reader adaptation, and current checkpoint](reports/hybrid_integration/REPORT.md)
- [Hybrid commands](energy_real/README.md#native-hybrid-architecture)

## Component studies: local learning and full energy models

The new architecture experiments compare backpropagation, predictive coding,
PC-ALM, full conditional energy-based predictors, and learned energy-memory
readers. A separate graph model learns scalar energy factors over explicit
entity IDs, without learned embeddings. Image, structured reasoning, and small
next-byte tasks use matched controls and three training seeds.

- [Results, trained models, and verification](reports/local_energy/REPORT.md)
- [Architecture and assessment of latent-state alternatives](reports/local_energy/DESIGN.md)
- [Runnable commands](energy_real/README.md#local-learning-and-full-energy-predictors)

PC-ALM improves deep predictive coding in the bounded image experiment. Energy
memory readers help the full EBM on graph tasks, but the matched backpropagation
control performs similarly. Discrete graph factors succeed with supplied rule
primitives; that is not evidence of learned language grounding or general AI.

## Reproducible memory retrieval and reasoning experiments

The local repair adds an explicit-backend energy kernel, correct attention resets,
full-state resumable training, source-only acceptance, controlled memory studies,
and an HTTP adapter for the existing local reasoning model.

- [Publication polish and focused follow-up](reports/publication/REPORT.md)
- [Repair findings and evidence](FINDINGS.md)
- [Commands and interfaces](energy_real/README.md)
- [Shared memory and reasoning study](reports/reasoning_v2/REPORT.md)
- [Reliability pass](reports/repair/POLISH.md)
- [Initial repair report and memory results](reports/repair/REPORT.md)

The new memory experiments favor exact structured storage; the tested energy
candidate has not earned a scale-up promotion over stronger classical controls.
Live reasoning evaluation and fresh GPU measurements have separate pending
statuses when the model service is unavailable or the device is occupied.

## Historical v23 target: a bounded DGX Spark test

The complete self-contained **v23 Spark source** is in [`spark_kit/`](spark_kit/README.md).
It includes the native energy/softmax implementations, resumable training, local
corpus preparation, device acceptance, configs, tests, and a read-only host doctor.
The earlier experimental modules and their authorization boundaries remain unchanged.

- [Start on the Spark](spark_kit/SPARK_START_HERE.md)
- [Architecture and math](spark_kit/ARCHITECTURE_SPARK.md)
- [Executed v23 changes and tests](spark_kit/reports/v23/REPORT.md)
- [Original model installation](spark_kit/checkpoints/README.md)

```bash
cd spark_kit
# The standalone download already includes the original checkpoint.
# To sample or continue original weights, install them from the local kit:
python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip
bash spark/run.sh doctor
bash spark/run.sh accept
```

The original bundled v23 report recorded 83 local CPU tests; current source-only
results and explicit artifact skips are recorded in the repair report. The new importer uses less
Python-tracked preparation memory on the measured fixture, and resume preserves
crash-tail logs. **No new model training, actual GB10/CUDA result, or serious general
pretraining approval is claimed by v23.** Run acceptance on the target device before
starting any corresponding capped pilot.

## Source and binary publication are different

This source commit does not upload the original pretrained binary or historical
experiment archives. The complete available collection is preserved in the supplied
v1-v23 download: 22 original experiment ZIPs and 96 original standalone artifacts,
plus the current kit. Their inventory/checksums are not a substitute for an upload.

From a networked machine holding that download and normal GitHub CLI authorization:

```bash
python spark_kit/publish_release.py \
  --archive /path/to/EnergyTransformer_All_Research_v1_v23.zip --publish
```

The uploader refuses a public target and conflicting assets, verifies downloaded
bytes, and publishes only after verification. It never force-pushes or changes
repository visibility. Do not paste tokens into chat or commit credentials.

## Earlier research and bounded memory interfaces

[Earlier README](README_LEGACY.md) preserves the previous entry point byte for byte.
See [architecture](ARCHITECTURE.md), [model card](MODEL_CARD.md),
[rights](RIGHTS.md), [storage](STORAGE.md), and [history](reports/HISTORY.md).
The existing private releases contain bounded parser/answer checkpoints, not the
latest native-text model or all original historical bytes. Their limitations remain.

No new open-source license is imposed. Old failed experiments and classical controls
retain their original qualifications. Energy coefficients remain numerical internal
state; terminology alone does not establish novel mathematics or superiority over
attention. The next useful evidence is target-device numerical behavior and paired
validation curves against measured training cost.

## Combined GPU research campaign

The isolated integration adds `energy_campaign`, combines the native hybrid work
with the forecast branch at `c5583551`, and provides a 24-GPU-hour budget supervisor,
document-separated public data, native acceptance/profiling, paired backbone studies,
frozen-teacher forecast training, and a SmolLM2 conversion pilot. See
[the campaign runbook](reports/gpu_campaign/README.md) and its recorded execution
status. Prepared code and CPU checks do not establish GPU quality or scaling results.
