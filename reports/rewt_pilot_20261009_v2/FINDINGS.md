# REWT: implementation works; the proposed advantages remain unproven

The complete proposed stack is implemented and has run from random weights.
This is a bounded CPU feasibility pilot: four models, three seeds, about 250K
working parameters, 128-byte windows, 61,440 LM training target bytes per fit,
and 240 separate reasoning-adaptation updates. It is smaller than the planned
twelve-GPU-hour course study.

| Model | Equal-byte test bpb ↓ | Equal-time test bpb ↓ |
|---|---:|---:|
| Transformer | 4.453 ± 0.021 | 3.850 ± 0.013 |
| Non-Fourier neural energy | 3.936 ± 0.036 | 4.343 ± 0.070 |
| Basic Fourier REWT | 3.937 ± 0.035 | 4.408 ± 0.019 |
| Full REWT | 3.997 ± 0.025 | 6.930 ± 0.045 |

Values are means ± sample standard deviations over seeds. Equal-time fits have
a common 30-CPU-second optimizer budget; the full model trains a fresh forecast
teacher inside that budget. Exact timings and any one-update overshoot are
recorded. Validation, setup, checkpoint persistence and evaluation are separate.

The energy models learn byte prediction more quickly per observed training byte
in this setup. The almost identical non-Fourier control gives no evidence that
cosine modes cause that improvement. The full stack is slightly worse than the
basic energy models and costs substantially more. At equal CPU time, the
Transformer performs best. These observations apply to this implementation,
brief budget and shared host, with no per-model hyperparameter search.

Isolated sequential inference measurements average 0.0105 ms/byte for the
Transformer, 0.2343 for basic Fourier REWT and 0.8885 for full REWT. This prototype
therefore has a material inference-cost problem as well as a training-cost gap.
Native process memory measurements are preserved with their library/data
overhead identified.

Text quality is poor. All 360 primary continuations and 120 exploratory
equal-time continuations scored 1/5 for fluency, coherence and relevance in the
explicitly labeled assistant review. They are mostly invented words or markup;
the short-budget full model also emits many nontext bytes. The rubric has a
floor effect and cannot rank the architectures here. Human ratings remain
pending in the separate anonymous review page.

| Model | Arithmetic/rule exact answers % | Correct answers + all valid steps % |
|---|---:|---:|
| Transformer | 1.67 ± 2.08 | 0.33 ± 0.58 |
| Non-Fourier neural energy | 3.67 ± 1.53 | 0.33 ± 0.58 |
| Basic Fourier REWT | 2.67 ± 1.53 | 0.33 ± 0.58 |
| Full REWT | 3.00 ± 2.65 | 0.00 ± 0.00 |

This table excludes digit recall. Full REWT learns the required response format
on every reasoning case but gives incorrect answers on 97% and has no fully
correct reasoning trace. The higher combined-task numbers in the raw report
are largely driven by recall. Performance is close to trivial answer priors;
this run does not demonstrate useful reasoning.

The mechanism works: all nine trained energy-model audits decrease the complete
energy in every checked round. The analytic derivatives match independent
autograd, and causality, chunk replay and fixed inference weights pass checks.
The initial unbounded recurrent-conditioning failure was reproduced, fixed
through bounded inputs to new proposals, and followed by fresh training for
every model. The failed run is preserved and excluded from these results.

Extra refinement has small benefits. On the fixed single-seed probe, basic REWT
improves from 3.9572 bpb at zero gradient rounds to 3.9531 at two, while eight
rounds reach 3.9528 at much higher runtime. Full REWT improves from 4.0170 to
4.0135 at two rounds, then worsens at eight. Energy reduction therefore should
not be described as a guarantee of better output quality.

The separate 40-update filter study finds 4.5933 bpb with Fourier filtering,
versus 4.5660 with clipping and 4.5196 without filtering. Those arms have no
raw norm spikes above the preset threshold 5, so they cannot establish spike
suppression. EMA has worse prediction and raw spikes despite a smoother filtered
gradient. Frequency/coupling studies are short structural diagnostics with
explicit capacity and initialization caveats.

The next useful study is a larger controlled core-model run with jointly
validated tuning and a fresh evaluation sample. It should test whether the
Fourier term contributes beyond the same neural priors and whether meaningful
text/reasoning gains justify refinement cost before expanding the support stack.

- [Full measurements and protocol](REPORT.md)
- [Component studies, reasoning breakdown and cost accounting](DIAGNOSTICS.md)
- [Fixed generated-text and reasoning examples](OUTPUT_EXAMPLES.md)
- [Assistant text-quality assessment](TEXT_QUALITY_REVIEW.md)
- [Anonymous human review page](text_review.html)
- [Implemented components](IMPLEMENTATION.md)
- [Implementation and commands](../../rewt/README.md)

All checkpoints, raw outputs, traces, data/source hashes, frozen task instances,
profiling measurements and the original source snapshot are saved locally.
