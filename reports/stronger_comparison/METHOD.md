# Stronger comparison protocol

This follow-up tests the native hybrid against the full-attention and windowed
Transformer controls. It does not alter the historical studies or checkpoints.
The models are small byte predictors, not instruction-tuned assistants.

## Implementation changes

The CPU energy kernel reduces the slot axis explicitly instead of using the
three-operand contraction. One-token inference uses the direct recurrence and
reuses its factors for both reads, avoiding padded square chunk work. Other
sequence lengths on accelerators retain the previous contraction because the
GPU is occupied and has not been benchmarked for this change.

The full Transformer has an optional preallocated KV cache. It copies incoming
keys and values into fixed storage and uses a mask-free attention call when a
single token can attend to its entire unsegmented prefix. Chunked prefills and
document resets retain their causal and segment masks. The cache owns one
mutable stream, requires evaluation with gradients disabled, and rejects
capacity overflow. Reserved capacity is counted in memory measurements.

## Training experiment

- Preserved enwik8 corpus and 90/5/5-million-byte train/validation/test split.
- Approximately 145,000 parameters per model; three layers, width 64.
- Batch eight, 128 bytes; AdamW, gradient clipping, float32 CPU arithmetic.
- Both PyTorch thread pools and OpenMP/OpenBLAS/MKL are set to one thread.
- Equal 20-process-CPU-second pilot fits for learning rates .001, .002, .004.
- Validation selects each architecture's learning rate, using pilot seed 17.
- Three fresh seeds (101, 202, 303), 90 process CPU seconds of updates per fit.
- A CPU-progress schedule gives each model 4% warmup and cosine decay.
- The faster models receive more updates and training bytes. Each seed uses
  the same minibatch-stream prefix across architectures.
- Checkpoints are selected by validation at quarter-budget intervals. All
  selections are frozen before test evaluation.
- Fresh validation/test window recipes, saved datasets, checkpoint hashes,
  source archive, full curves, and per-window losses are retained.

CPU timing includes forward, backward, clipping, optimizer and schedule work;
sampling data, validation and checkpoint I/O are excluded equally. A final
update may exceed the target by one update. CPU seconds are not FLOPs or GPU
compute. Wall time and training bytes are also reported. CPU frequency and
shared-host effects can still influence time-budget experiments.

The quality gate requires hybrid losses below both controls at both 128 and
512 bytes, with conditional paired-window 95% upper bounds below zero. The
512-byte test extrapolates from 128-byte training. Bootstrap intervals average
paired seed differences before resampling windows; they do not capture all
training-seed uncertainty, and random corpus crops can overlap.

## Timing experiment

The original selected weights are identical across before/after kernel timing
arms. Training timing uses six balanced rounds with fresh copies and optimizer
states. Inference compares growing and preallocated full-attention caches,
old/new hybrid kernels, windowed attention, and energy-only recurrence.
All process prompts in 256-byte chunks and decode the same 16 supplied tokens
after 128-, 4,096-, or 32,768-byte prefixes. Six inference rounds rotate arm
order so each arm occupies every position. Training rounds also reverse the
three-arm order. This is native forward-pass timing, not end-to-end serving throughput.

The first pilot was interrupted before completion because process CPU time
revealed uncontrolled numerical-library threads. Its log and partial outputs
remain in `runs/compute_matched`; the corrected run is separate. No test data
were evaluated in that aborted pilot.

## Reproduce

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m energy_real.compute_study \
  --out runs/my_compute_comparison --max-seconds 3600

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python reports/stronger_comparison/benchmark.py
```

The benchmark script writes into this report directory; use a copy when
reproducing so completed evidence stays immutable. Training refuses existing
output paths. Both commands use the cumulative local research budget ledger.
