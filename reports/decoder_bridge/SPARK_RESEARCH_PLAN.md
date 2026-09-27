# Evidence-driven DGX Spark plan after the decoder study

Prepared September26,2026. This plan separates executed local evidence, shipped commands, and proposed architecture experiments. It does not authorize an unattended large run or claim that a10.6M experiment has already executed.

## Executive choice

Do the cheapest discriminating10.6M experiment first: freeze your actual teacher and train ordinary hidden/history adapters on causal cached features. The new decoder cache supports that without waiting for forecasting-head training. This directly tests whether the same decoding bottleneck found on the archived973K teacher exists on your Spark model.

The forecast-energy reader did improve relative to the frozen small teacher, but lost a similarly sized hidden adapter. Therefore, forecasting is an optional research arm, not a required first step for all future pretraining.

## A. Preserve the experimental basis

1. Use an isolated worktree at the published decoder-study commit on `research/forecast-energy-20260926`. The original additive patch was based on `c5583551eb68b89c21337a0cc639e4c1bfad0b00`; do not reapply it after pulling the decoder commit. Do not change source under a running training job.
2. Record teacher SHA, corpus SHA, Git revision/dirty diff, Torch/CUDA/driver versions, and precision settings. Use the already working GB10 software environment; do not upgrade the driver and change architecture together.
3. Resolve the previously reported remote CI failure or retain a clean, equivalent local receipt. This turn did not diagnose hosted CI. The new local study tests do not establish the entire repository is green.
4. Run source tests and `energy_real.device_smoke` before extraction. A small seeded numerical test is not a full-size throughput test.
5. Leave unrelated GPU jobs running. Do not bypass the idle guard, remove other processes, expose credentials, or relax bounds just to complete an experiment.

NVIDIA documents that Spark's GPU shares system DRAM and that `nvidia-smi` framebuffer reporting and `cudaMemGetInfo` alone do not fully describe availability. Record Torch allocated/reserved memory, host MemAvailable, swap pressure and wall time separately. The new cache's8GiB Torch policy is not a hard guarantee against all host/library allocations.

## B. Current executable experiment: actual10.6M hidden/history bridge

The full commands are in `FORECAST_DECODER_START_HERE.md`. Start with64 train,16 validation,16 test blocks of512 bytes, stride4, teacher batch2. This yields7,936/1,984/1,984 sampled positions after16-byte warmup, calculated from the alignment code. The teacher executes serially; it may be slower than pretraining prefill. Do not interpret an extraction timeout as a mathematical failure.

Each complete batch of blocks becomes an identity-bound atomic shard. Repeat the exact command with `--resume` after a partial result; completed shards are reused and unfinished current work is recomputed. The per-invocation deadline is not a total project-budget accountant. Assembly still materializes the cache, so measure host RAM before increasing block count. Shard checksums are integrity records, not cryptographic authentication against a malicious actor.

Train hidden and history with the same three seeds,600 updates,batch128. Only adapter parameters update; the teacher and its output embedding remain frozen. CPU head fitting is sufficient for this first bounded test. The adaptive hidden widths are recomputed for the teacher width. Do not claim the exact973K-model adapter counts on the10.6M model.

Use all agreed validation observations for checkpoint selection, include step zero, then freeze hashes. Open test only once after choices are fixed. On a fixed sampled dataset, report baseline and adapter loss over identical positions and document/byte-block grouped confidence intervals. Supplement with an every-byte evaluation on reserved contiguous blocks before declaring a language improvement. Byte blocks can cross articles and are not independent documents.

**Go:** Repeatable improvement over the frozen teacher with healthy target-device behavior.
**Stop/investigate:** Nonfinite values, source/checkpoint identity mismatch, resumed/uninterrupted discrepancies beyond the declared precision tolerance, systematic test degradation, or excessive extraction memory.

A hidden/history win still does not beat full teacher continuation. Add that control before calling it better pretraining.

## C. Forecast integration, only after the cheap control

Freeze the actual10.6M teacher for all trace extraction. A forecasting head trained on changing teacher weights has a moving target and invalidates simple checkpoint-bound comparisons.

Fit the same teacher's horizon-regularized means and calibrated density using TRAIN only for normalization/crossfit, validation selection and calibration for their specified roles, and reserved test for the final score. The old973K heads and gate must remain rejected for this model.

Build a new decoder cache using `--forecast-data` so all five adapter arms share its exact teacher, feature normalization and sampled blocks. Retrain hidden/history on that same cache; do not compare with the different headless cache from stageB.

Matrix:

| Arm | Question |
|---|---|
| Frozen teacher | Starting probability model |
| Hidden-only MLP | Does ordinary decoder capacity explain the gain? |
| History MLP | Does directly accessing more observed information help? |
| Ridge-mean adapter and folded version | Does the linear forecasting inductive bias help? |
| Mixture moments | Does uncertainty/correlation-derived spread help? |
| Two energy-query reads | Does candidate-dependent access add value? |

Keep added trainable parameter budgets close, but report total frozen parameters, forecast pretraining, runtime buffers, actual fitting time and decoding time too. The density components are extra resources, not free features. Compare the folded affine version when measuring a mean-only arm.

Zero and cross-example shuffling are diagnostic interventions. They do not replace a separately trained baseline. Report when zeroing features improves performance, as it did for the small teacher on part of this study.

## D. Proposed architecture experiments (NOT implemented or trained in this turn)

### D1. Richer reads of CURRENT energy state

Motivation: the frozen final hidden representation already contains information discarded by the eight-variable forecast sketch. Forecasting a lossy sketch may add less useful information than simply reading the current function more effectively.

Compare the hidden-only adapter with an equal-budget read of per-head log precision, one or more S/P query responses, and query variance. Keep teacher weights frozen. Fit dimensionality reduction on TRAIN only. Try one additional learned read before adding many projections. A permutation/matched-norm control can help locate what the read actually uses.

Success criterion: an incremental improvement beyond hidden-only under the same origin sampling and evaluation. If it fails, retain the simple adapter and do not add perpetual state merely because it sounds richer.

### D2. Multi-token auxiliary prediction

Add heads at t+1,t+2,t+4 to the current backbone, initially using the same representation. Compare ordinary next-byte loss against multi-token auxiliary training with matched main-model parameters and measured compute. True future bytes are training labels only, not inference input. Separate auxiliary-head parameters from backbone parameters.

This is motivated by published multi-token prediction, not a reproduction yet. It is a more direct way to train future-relevant features than optimizing eight summary coordinates. Start at10.6M; changing tokenizer, context and width at the same time would obscure the result.

### D3. Learned multi-horizon sketch

Only if fixed sketches clearly limit forecasting, learn a readout with both predictive and anti-collapse diagnostics. A small forecast MSE can be meaningless if the representation becomes constant. Track feature variance/rank, a constant baseline, and downstream language utility. Keep a frozen or explicitly EMA target and compare the same capacity without the forecast objective.

### D4. Low-rank distribution structure

Dense32-dimensional covariance is manageable here, but its cost grows quadratically with horizons/features. A diagonal-plus-low-rank covariance is a credible future model class. Test exact normalized likelihood and conditioning against a dense reference. It changes the distribution family and is not the same as lossless square-root numerical optimization.

### D5. Local attention/energy hybrids

The attached architecture note describes a historical hybrid; the current repository's energy backbone does not require local attention. Treat a hybrid as a separate arm with explicit per-layer mixer assignment. Compare energy-only, local-only and a limited hybrid. Count retained local KV entries; do not describe the hybrid as P/S-only.

### D6. Nonlinear basins and Fourier ideas

The current positive quadratic memory read has one minimum. The forecast mixture can be multimodal, but following its gradient does not imply factual correctness or better byte loss. Reserve multiscale search for a task with an actual diagnosed optimization bottleneck. Do not conflate a basis change with changing the energy landscape. No new fractal or saddle-escape experiment was executed in this turn.

## E. When to unfreeze the backbone

First demonstrate an adapter benefit beyond the strongest equally tuned control. Then compare:

- frozen teacher + trained adapter;
- ordinary full-teacher continuation;
- the same continuation plus a small auxiliary forecast/read loss.

Start with the existing10.6M weights as explicitly labeled warm starts. Historical weight-only checkpoints are not exactly resumable optimizer trajectories. Subsequent full checkpoints should capture optimizer, RNG, schedule, sampler, source and data identities. Test resume in a separate short job before a long allocation.

For an initial bounded continuation, the existing plan's512-byte context, microbatch4, accumulation4 and1,000 planned updates (8,192,000 presented byte targets) remain an example, not an optimized prescription. Inspect after200 updates without altering the planned schedule. Learning rate1e-4 is a conservative proposal, not validated as optimal here. Use new run directories, no silent overwrites.

Evaluate both equal data and equal measured GPU time. An adapter that learns more slowly or has costly dependencies can win at equal updates without being practically better.

## F. Promotion and stopping rules

Example PROJECT criteria, not universal statistical laws:

1. Three seeds show the same direction against the strongest comparator.
2. Improvement of at least0.01 bpb, with a paired block/document interval below zero, on a newly reserved check. Tiny p-values on reused data do not substitute for this.
3. At most25% extra fitting time and10% extra decoding latency for that quality gain, unless the intended workload explicitly values another trade-off. Measure target-device performance; do not reuse CPU ratios.
4. No causality, reset, state-restoration or numerical failures at the selected precision/context.
5. Real content-retention and edit-locality checks survive. Lower news loss alone is not a reasoning score.

A feature failing these criteria remains optional research. It need not block a plain energy or Transformer baseline training run. Avoid an endless process in which every proposed mechanism becomes a mandatory component.

## G. Scaling ladder

After successful10.6M replication, increase to the previously planned approximately24.6M model (width512,8 layers,4 heads,16 slots,FFN2048). Keep byte vocabulary/context fixed initially. Only then compare longer context and the approximately37.9M configuration. These are configurations, not newly trained checkpoints or verified GB10 fits.

Before each promotion, benchmark full forward/backward/optimizer time, effective batch, host and device peaks, prefill and decode separately. Repeat a small matched softmax baseline. More system DRAM does not make the dense chunkwise energy update cheap; profile its materialized intermediates.

A broader document-separated, provenance-tracked corpus and a reserved generalization set are required before calling this serious general pretraining. More repetitions of enwik8 add token presentations, not unique training information.

## H. Concrete experiment queue

| Priority | Executable now? | Evidence sought | Next decision |
|---|---|---|---|
| Actual10.6M hidden/history adapter | Yes, new cache/trainer | Replicate downstream gain cheaply | Keep/decline adapter |
| Actual10.6M five-arm forecast comparison | Yes, requires teacher-specific heads | Increment over strong control | Retain/disable forecast integration |
| Full teacher continuation control | Existing trainer | Compare total-time/data efficiency | Adapter vs ordinary training |
| Richer current P/S read | Proposed | Remove summary bottleneck | Which representation to expose |
| Auxiliary next2/4 prediction | Proposed | Train future-relevant backbone | Forecasting vs direct targets |
|24.6M scale probe | Existing architecture configuration | Size/generalization/device behavior | Bounded scale promotion |

## Research sources

Gloeckle et al., Better & Faster Large Language Models via Multi-token Prediction (2024): https://arxiv.org/abs/2404.19737

Li et al., EAGLE (2024/2025): https://arxiv.org/abs/2401.15077 . Their distribution-preserving verifier is not present in our logit adapter. Do not claim token skipping from state forecasts alone.

Wang, Shi and Fox, Test-time regression (2025): https://arxiv.org/abs/2501.12352 . Associative storage and retrieval are distinct mechanisms.

NVIDIA DGX Spark known issues / UMA guidance (checked September26,2026): https://docs.nvidia.com/dgx/dgx-spark/known-issues.html

No external paper's benchmark or speedup is reproduced by merely using its name in this plan.
