# Supplemental Transformer validation check

The five-arm primary protocol remains unchanged. Its Transformer was chosen by minimum parameter-count error, producing width 48 and feedforward width 1,029. This separate study screens hidden width and learning rate using the same language training/validation data, then evaluates the selected allocation across three seeds on the replacement study's already-declared fresh test samples. All weights start from scratch; language bytes, context resets, reasoning curriculum and adaptation budget match the revised primary models. This is a stronger baseline check, not an isolated estimate of a REWT repair effect.

| Width / learning rate | Validation bpb at 64 updates |
|---|---:|
| w80_lr0.004 | 4.0102 |
| w96_lr0.004 | 4.0291 |
| w48_lr0.004 | 4.0581 |
| w80_lr0.002 | 4.1614 |
| w96_lr0.002 | 4.1819 |
| w48_lr0.002 | 4.2373 |
| w96_lr0.001 | 4.3887 |
| w80_lr0.001 | 4.3985 |
| w48_lr0.001 | 4.5324 |

Selected **w80_lr0.004**; 250,122 active parameters. No held-out score entered this selection.

| Metric | Mean ± seed standard deviation |
|---|---:|
| test_bpb | 3.980 ± 0.019 |
| equal_time_test_bpb | 4.030 ± 0.037 |
| reasoning_answer % | 3.667 ± 0.577 |
| reasoning_complete % | 1.500 ± 0.500 |
| recall % | 58.000 ± 1.732 |

Nine additional single-seed screens are a supplemental tuning budget and are charged separately. The original primary baseline had no independent architecture/LR search; the primary REWT recipe screened fourteen cases. Short screening need not choose the best longer-run configuration. The full model's teacher remains the original primary Transformer; this supplemental baseline does not change its targets. All ninety supplemental continuations were actually read and scored in the [assistant review](/home/kt/Documents/EnergyTransformer/reports/rewt_stabilized_20261009/baseline_supplement/TEXT_QUALITY_REVIEW.md), with the model family known and scores locked before joining seed identities. Fluency and coherence both average 1/5. Human scores remain pending. [Actual text-generation timings](/home/kt/Documents/EnergyTransformer/reports/rewt_stabilized_20261009/baseline_supplement/GENERATION_RUNTIME.md) are recorded separately.
