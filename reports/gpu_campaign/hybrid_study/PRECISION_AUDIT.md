# Verification completed

The implemented opt-in first-projection FP32 policy passed all18 CUDA numerical cases and exact GPU restart. Installed wheel:533 passed,51 skipped. Study002 subsequently stopped on CUDA context out-of-memory after one longer fit; do not rerun the original diagnostic. See REPORT.md for the current state.

# CUDA-confirmed repair candidate implemented

The saved failing CUDA fixture reproduces gradient error0.470485 under original mixed precision and0.470484 with forced math SDPA. Protecting only the first energy projection reduces the error to0.018502 and raises cosine to0.999873. Both projections FP32 and full energy FP32 also pass this fixture, but are broader changes. The first-projection policy passed all18 CPU cases before implementation.

Implemented `Config.energy_projection_fp32` and native training flag `--energy-projection-fp32`. It defaults to false for historical compatibility. The revised hybrid study explicitly enables it for hybrid models, records it in the experiment identity and checkpoint configuration, and leaves attention controls unchanged. Projection parameters and shapes are unchanged; outer feed-forward and attention computations still use BF16. There is no adaptive precision fallback.

Installed-wheel verification is in progress. All-seed CUDA validation and deterministic restart must pass before training resumes in a new study directory. Original failed study001 is retained.

Raw CUDA comparison: `runs/campaign/hybrid_precision_audit_001/results.json`.

---

# Precision repair investigation

The user authorized diagnosing and correcting the hybrid CUDA BF16 failure. Original training and stop records remain unchanged. A GPU precision ablation was prepared but did not launch because available host memory fell below 20 GiB. No extra GPU time was charged.

Exact saved fixture CPU results:

| Precision variant | Relative gradient error | Cosine |
|---|---:|---:|
| Original mixed precision | 0.073198 | 0.997535 |
| First energy projection FP32 | 0.015624 | 0.999894 |
| Second energy projection FP32 only | 5.962943 | 0.777344 |
| Both projections FP32 | 0.085499 | 0.998334 |

Replacing the CPU contraction order with the CUDA einsum expression gave the same qualitative behavior. These measurements do not establish the CUDA cause. Selective precision changes can worsen this model, so merely adding FP32 is not a verified fix.

The first-projection candidate passed all 18 CPU family/size/seed checks, including streaming/reset/causality/checkpointing for hybrid seed404 at both sizes. Maximum hybrid gradient relative error: 0.031193634309235527. This candidate is applied by a diagnostic wrapper only; production source and acceptance receipts have not changed.

## Next executable diagnostic

When available memory is at least20 GiB, with no current GPU reservation or duplicate audit, run from the integration checkout:

```bash
PYTHONPATH=. .venv/bin/python runs/campaign/supervise_precision_audit.py runs/campaign/hybrid_precision_audit_001 runs/campaign/audit_hybrid_precision.py runs/campaign/hybrid_precision_audit_001/results.json
```

This charges the existing acceptance ledger, caps the worker at4 GiB, preserves16 GiB host memory, and limits the diagnostic to180 seconds. It compares original, math SDPA, first projection FP32, second projection FP32, both projections FP32, and entire energy mixer FP32 on the saved CUDA failure. A failed precision case is an audit measurement, not permission to train. Never overwrite existing audit output or retry a failed audit automatically.

After the audit, select a correction from actual evidence, implement an explicitly identified precision policy with tests, rebuild/reverify the wheel, and require all18 GPU numerical cases plus deterministic restart. Preserve hybrid_study_001 as the original failed run. Do not clear its stop marker, drop seed606, loosen thresholds, or treat CPU success as GPU acceptance.
