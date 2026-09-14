# Run EnergyTransformer v23 on the DGX Spark

This is a bounded single-device acceptance and pilot kit. The original 972,896-
parameter native-text checkpoint is unchanged. No CUDA GPU or NVIDIA container
was executed during this final hardening pass. A device PASS permits only its
corresponding capped experiment, not serious general pretraining.

## 1. Enter the kit and check the host

From the standalone ZIP:

```bash
cd EnergyTransformer_Spark_v23
bash spark/run.sh doctor
```

From a Git checkout, enter `EnergyTransformer/spark_kit` instead. A source-only
clone can run numerical acceptance using a labelled, seeded random fixture. To
sample or continue the historical pretrained model, install its hash-verified binary:

```bash
python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip
```

`doctor` is read-only and needs only Python's standard library. It reports missing
Arm64, Docker and driver prerequisites without changing the machine. Missing
historical artifacts are reported separately; `doctor --require-original` also
requires the original checkpoint.

## 2. Run actual device acceptance

```bash
bash spark/run.sh accept
```

The launcher pulls the v22-documented NVIDIA image `nvcr.io/nvidia/pytorch:25.11-py3`,
resolves it to a digest, and preserves that digest in `runs/image.lock`. It does not
claim this image is the newest or update your host driver. Container execution has
networking disabled and mounts only this project plus designated data/output folders.
It never installs a replacement Torch over NVIDIA's supplied stack.

The command checks real GB10/Arm identity, FP32/BF16 behavior, float32 recurrent
coefficients, causality, gradients, state restart, optimizer/RNG restart, and bounded
fit/update measurements at approximately 973K, 3.29M, and 10.61M parameters. Those
shape checks are not trained large-model checkpoints or language-quality results.

Keep `runs/acceptance/acceptance.json` and `benchmark.json`.
`PASS_FOR_BOUNDED_DEVICE_PILOT` is required by CUDA training. A failure is a result
to inspect, not something to bypass by editing its receipt. `accept-small` checks
only the original size and writes to `runs/acceptance-small`.

Acceptance never overwrites a prior receipt. To rerun:

```bash
bash spark/run.sh accept --out /runs/acceptance-v23
```

For training with that receipt, prefix the command with
`ENERGY_ACCEPTANCE=/runs/acceptance-v23/acceptance.json`. Source, configuration, or
image changes invalidate earlier receipts.

## 3. Sample the unchanged model

```bash
bash spark/run.sh sample --prompt "The " --length 160
```

No corpus or Hugging Face account is needed. This tiny byte model still generates
inaccurate/incoherent text; deployment work does not turn it into an assistant.

## 4. Prepare permitted documents

Place `documents.jsonl` in `local_data/`. Each independent UTF-8 document must provide
nonempty `document_id`, `source`, `license`, and `text` strings. Use at least ten
unique documents of 1,025 bytes or more; that minimum only enables the software and
is not adequate evidence for large training.

```bash
bash spark/run.sh prepare \
  --jsonl /data/documents.jsonl --out /data/corpus
```

Exact content is deduplicated before splitting; training windows stay inside a
document. License fields are declarations, not legal clearance. Near-duplicate and
semantic contamination audits remain separate. The v23 importer spools raw text to
disk; it still retains metadata and one whole input record in memory, and needs
scratch disk while writing shards. No data are downloaded or uploaded.

## 5. Start the capped continuation

Continuation requires acceptance that explicitly verifies the installed original
checkpoint, unlike from-scratch numerical acceptance:

```bash
bash spark/run.sh accept --verify-original --out /runs/acceptance-original
```

Then use that receipt:

```bash
ENERGY_ACCEPTANCE=/runs/acceptance-original/acceptance.json \
bash spark/run.sh train \
  --config /workspace/configs/v22/continuation_973k.json \
  --data /data/corpus --out /runs/pilot-973k
```

The cap is 200 updates and 102,400 presented byte targets. Initial weights are
eligible for validation selection. Use the exact same command with `--resume` after
a requested stop. Optimizer/RNG state, source/data/runtime identity, and sample
hashes are restored; uncheckpointed log tails are preserved separately, not counted
twice. Incompatible resumes and existing non-resume runs fail.

Ctrl-C/SIGTERM requests a save after a complete update. An OOM/partial-update failure
preserves the previous verified checkpoint. Cooperative time/memory checks are not
hard isolation from hung kernels or other processes.

## 6. Larger pilots are separate decisions

```bash
bash spark/run.sh train \
  --config /workspace/configs/v22/pilot_3m.json \
  --data /data/corpus --out /runs/pilot-3m

bash spark/run.sh train \
  --config /workspace/configs/v22/pilot_3m_softmax.json \
  --data /data/corpus --out /runs/pilot-3m-softmax
```

The 3.29M and 10.61M models start from scratch and must have passed their exact
shape checks. Their caps remain 512,000 and 1,024,000 presented byte targets. These
are diagnostic budgets, not useful-model or adequate-data guarantees. Compare
validation curves against elapsed time and tokens, not equal updates alone.

BF16 is used for outer projections; P/S updates and storage remain FP32. Default
policies retain 16 GiB host memory, cap observed Torch CUDA reservation at 48 GiB,
and use a 60% per-process allocator fraction. They are not whole-system hard memory
isolation. The scripts do not kill processes, clear system caches, or change clocks.

## 7. Inspect the evidence

```bash
bash spark/run.sh profile \
  --config /workspace/configs/v22/continuation_973k.json \
  --out /runs/profile-973k
bash spark/run.sh report
```

Profiling may require CUDA/CUPTI support. The useful outputs are the acceptance and
benchmark JSON plus the pilot's `history.json` and `completed.json`. Review before
sharing: source paths, states, checkpoints, and corpus identifiers can be private.
No telemetry is sent automatically.

See `ARCHITECTURE_SPARK.md`, `reports/v23/REPORT.md`, and `RIGHTS_SPARK.md` for
mathematics, executed tests, prior-source provenance, and limitations.
