# Qwen3.5-0.8B full energy/local conversion

User-selected donor: Qwen/Qwen3.5-0.8B, revision2fc06364715b967f1860aea9cf38778875588b17. The strict-loaded text donor contains752,393,024 parameters. Original vision weights remain in the downloaded donor; this experiment is text-only.

## Implemented architecture

All24 text-layer mixers are converted to eight repetitions of Energy → Energy → Local attention:16energy layers,8local-attention layers, window128tokens. There are no DeltaNet mixers or global-attention mixers in the converted text path. Preserve Qwen's token vocabulary, tied embeddings/output, RMSNorm and SwiGLU feed-forward blocks. Energy blocks have8heads and16slots per head, two reads, FP32 state and the first-projection precision correction. New local positions initialize from the nearest pretrained full-attention layer and use rank8 trainable q/v/output adapters. This is a full mixer conversion retaining the pretrained non-mixer backbone.

## Verification and runtime

Separate .venv-qwen runtime uses Transformers5.17.0 with the existing Torch2.14+cu130; original runtime is unchanged. Tiny-model tests verify gated-attention/RoPE parity, bounded caches, chunked/full agreement, causality, save/load, the complete replacement pattern, and finite backward gradients. All3 Qwen-specific tests pass against the installed wheel. Existing installed suites passed535 tests with51skips before the final Qwen-only adjustments; these adjustments passed the dedicated installed tests. No actual-size converted GPU acceptance has run yet.

## Fixed feasibility protocol

One seed707. Document-contained128-token batches from the existing fixed train/validation/test corpus splits.64teacher-target batches;64normalized layer-output transfer updates, then64language-loss plus teacher-distillation recovery updates. Distillation uses32teacher tokens plus an aggregate complement bucket, explicitly a coarse KL rather than full-vocabulary KL. Validation selects checkpoints; final test is1,024targets from8documents. This small sample cannot establish broad capabilities or superiority over larger models. Only energy replacements and attention adapters are trained; a matched unchanged-model adaptation control remains necessary for a quality claim.

The unchanged0.8B donor provides the first baseline. Frozen2B and4B comparator revisions are recorded in runs/campaign/qwen08/larger_comparators.json; neither has been evaluated. Larger-model claims need common held-out language, reasoning/coding and recall evaluations with uncertainty, and separate memory/latency measurements.

## Current execution status

Attempt001 stopped at the worker's20GiB host-memory admission check before CUDA initialization, donor loading or training. Its log and4.918460seconds charge are preserved. The audited next attempt002 requires24GiB at parent admission for startup cushion, still uses a4GiB Torch cap, preserves16GiB host memory, and reserves at most1,800seconds from the same existing conversion-stage ledger. No numerical thresholds were relaxed and no budget was reset.

When headroom permits and no other reservation/process is running:

```bash
PYTHONPATH=. .venv-qwen/bin/python scripts/run_qwen35_pilot.py
```

This command refuses to overwrite an existing attempt. Inspect failures rather than automatically retrying. Raw protocol, CPU receipt, donor, tokens and retained first failure are under runs/campaign/qwen08. The converted model has not been trained or shown to beat its donor or larger models.
