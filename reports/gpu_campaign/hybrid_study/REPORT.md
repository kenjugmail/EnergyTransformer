# Precision repair verified; longer training hit resource failure

The explicit `energy_projection_fp32` policy fixes the saved CUDA BF16 failure. It protects the first energy projection and keeps the other outer operations in BF16. Historical configurations default to the old policy; corrected hybrid runs opt in and save the setting in checkpoint/configuration identity.

- Saved failing CUDA fixture: gradient error **0.470485 → 0.018502**, cosine **0.894821 → 0.999873**. Forced math SDPA alone did not fix it.
- Corrected GPU grid: **18/18 passed**, maximum gradient relative error **0.095558**, minimum cosine **0.999028**, with unchanged acceptance thresholds.
- GPU interrupted/resumed training: **exact pass** for weights, optimizer, all saved RNG states, coverage, step and target counts; committed log contains each update exactly once.
- Installed-wheel suites: **533 passed, 51 skipped**. Skips do not substitute for CUDA acceptance.

The first longer fit (full attention,1.05M,seed404) completed1,048,576 targets. The next worker (local attention) failed creating its CUDA primary context with `CUDA_ERROR_OUT_OF_MEMORY`, before training. Study002 is stopped and the resource failure is retained; it has not been automatically retried. The existing ledger and memory limits remain in force. Original study001 and failure are retained. Training and final held-out evaluation are not yet complete; no architecture superiority claim is established. This acceptance covers the stated eager shapes, sizes and seeds, not arbitrary scaling or compiled kernels. The revised precision has an additional cost that must be measured rather than hidden.

---

# GPU acceptance blocked: hybrid BF16 gradients

The available-memory gate cleared and GPU validation ran. **17 of 18 cases passed**. The 3.33M hybrid at seed 606 failed the unchanged FP32/BF16 gradient criteria:

| Metric | CUDA | Exact saved fixture on CPU | Required |
|---|---:|---:|---:|
| Relative gradient error | 0.470486 | 0.073198 | <0.30 |
| Gradient cosine | 0.894821 | 0.997535 | >0.95 |
| Relative logit error | 0.026054 | 0.021519 | <0.08 |

All three 1.05M hybrid cases passed; the 3.33M hybrid seeds 404 and 505 passed. Both attention controls passed at both sizes and all seeds. Seed-404 hybrid streaming, resets, causality and checkpointed-gradient checks passed at both sizes.

The same weights and tokens pass on CPU, so the previous CPU receipt does not establish CUDA mixed-precision correctness. The underlying cause remains unresolved; differing arithmetic or backend behavior needs a controlled audit. No training fits, restart checks or test-set evaluation ran in this attempt. Do not omit the failing seed or loosen tolerances.

Next investigation: isolate CUDA FP32 repeatability and SDPA backend effects, then energy projection/read precision on the saved fixture. Any diagnostic GPU execution must use the existing ledger and resource limits. Only an audited correction followed by all-seed acceptance can reopen training. The failed stage has not been retried and its stop record remains intact.

Validation charged 9.318616 seconds. Total cumulative campaign charge: 108.456719 seconds. Raw failure and replay checkpoint: `runs/campaign/hybrid_study_001/gpu_validation/`; stop record: `runs/campaign/hybrid_study_001/stopped.json`.

---

# Hybrid energy/local study

The selected architecture repeats two energy layers followed by one causal local-attention layer (128-byte window). The next experiment compares it with full and local attention at approximately 1.05M and 3.33M parameters, using seeds 404, 505 and 606.

## Verified so far

All 18 actual-size CPU FP32/BF16 checks passed at batch 1 and context 512. Hybrid streaming, single-token, reset, causality and checkpointed-gradient checks passed at both sizes for seed 404. Diagnostic reductions use FP64: an earlier audit produced impossible cosine values above one from FP32 reductions; that audit remains retained and is superseded by cpu_numerics.json.

The earlier six GPU fits were only eight updates each. Their favorable hybrid validation losses are preliminary and do not establish convergence, superiority, recall performance or scaling laws. The pure-energy BF16 gradient failure remains unresolved and the original full campaign remains stopped.

## Next GPU experiment

Run ` .venv/bin/python -m energy_campaign.hybrid_study runs/campaign runs/campaign/hybrid_study_001 ` from the integration checkout.

1. Repeat actual-size numerical validation on CUDA for all 18 configurations/seeds.
2. Verify deterministic interrupted/resumed GPU training against uninterrupted training, including weights, optimizer, RNG, data coverage and committed logs.
3. Train 18 independent endpoints from scratch: three families, two sizes, three seeds; 128 updates, 1,048,576 presented targets per fit, context 512, effective batch 8,192. Fixed peak LR 0.001, eight-update warmup; this is not a tuned LR comparison.
4. Select checkpoints on validation only, freeze their hashes, check trained hybrid numerical behavior, then evaluate 262,144 held-out targets per model.

This study uses the existing 24-hour ledger, with at most 11,280 seconds of additional reservations across acceptance, backbone and verification. Actual charged time includes failures. No ledger reset or new allocation is allowed. Every worker uses the shared-GPU 4 GiB Torch cap, needs 20 GiB available host memory at admission, and preserves a 16 GiB reserve. Foreign services remain untouched. A failed job requires audit before retry.

## Current limitation

GPU execution is waiting for host memory: approximately 9 GiB is available at preparation time. CPU checks do not establish GPU correctness. Installed-wheel verification and source receipts are recorded in runs/campaign. No speed claim is valid under shared contention; two sizes at one data budget cannot establish a scaling exponent. Recall, larger sizes, compiled/fused execution and pretrained conversion remain separate work. No model is automatically promoted.

Installed-wheel verification: **531 passed, 51 skipped** across native, forecasting, campaign and independent Spark suites. GPU-dependent skips are not acceptance. Current source: `c8b61bde270ef054e0d918969384b67b8137665f7f50187b3a32a63e6d771769`. The existing ledger remains unchanged at 99.13810311100679 charged seconds.

Installed-wheel verification: **533 passed, 51 skipped** across native, forecasting, campaign and independent Spark suites. GPU-dependent skips are not acceptance. Current source: `c088b45aa5a83654c9e20107c5552043c073cf1cc0c391f138ecbf9c625fb9fd`. The existing ledger remains unchanged at 113.92531052901177 charged seconds.
