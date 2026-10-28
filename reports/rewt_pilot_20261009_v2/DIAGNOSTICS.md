# Diagnostic evidence from the REWT pilot

This analysis reads saved outputs and checkpoints only. It does not tune settings or change model selection.

The raw experiment report contains the primary prediction/text/reasoning table. This companion separates paired-seed comparisons, constant-answer priors, and the shorter component protocols.

| Energy model | Paired test-bpb difference vs Transformer ↓ |
|---|---:|
| Neural energy | -0.5167 ± 0.0268 |
| Fourier REWT | -0.5156 ± 0.0263 |
| Full REWT | -0.4556 ± 0.0046 |

Differences use identical held-out windows. Three-seed variation is descriptive, not a strong significance or scaling result.

## Trivial reasoning references

A constant response is selected only from training examples. The family-aware variant additionally receives the true task family, so it is an oracle-label diagnostic rather than a fair neural baseline.

| Predictor | Exact answers % | Correct answers + complete steps % |
|---|---:|---:|
| unconditional_constant | 3.33 | 3.33 |
| oracle_family_constant | 6.67 | 4.00 |

Accuracy by chain length is saved separately for arithmetic and rules. Recall is grouped by distractor delay rather than described as multi-step reasoning.

## Arithmetic and rule reasoning, excluding recall

| Model | Valid response format % | Exact answers % | Correct answers + complete steps % |
|---|---:|---:|---:|
| Transformer | 45.33 ± 18.15 | 1.67 ± 2.08 | 0.33 ± 0.58 |
| Neural energy | 96.67 ± 5.77 | 3.67 ± 1.53 | 0.33 ± 0.58 |
| Fourier REWT | 97.67 ± 4.04 | 2.67 ± 1.53 | 0.33 ± 0.58 |
| Full REWT | 100.00 ± 0.00 | 3.00 ± 2.65 | 0.00 ± 0.00 |

A well-formed steps/answer string can still contain incorrect reasoning. This table isolates that distinction and keeps the easier digit-recall tasks out of the reasoning headline.

## Inference refinement probes

These use the same trained seed and 8 held-out windows. More rounds need not improve bpb.

| Configuration | Test bpb ↓ | CPU ms/byte |
|---|---:|---:|
| fourier_inference_coarse_to_fine | 3.9532 | 0.2553 |
| fourier_rewt_output_only | 3.9575 | 0.2746 |
| fourier_rewt_rounds_0 | 3.9572 | 0.0807 |
| fourier_rewt_rounds_1 | 3.9542 | 0.1705 |
| fourier_rewt_rounds_2 | 3.9531 | 0.2893 |
| fourier_rewt_rounds_4 | 3.9527 | 0.4741 |
| fourier_rewt_rounds_8 | 3.9528 | 0.8532 |
| fourier_rewt_states_only | 3.9531 | 0.2910 |
| full_rewt_output_only | 4.0190 | 1.0772 |
| full_rewt_rounds_0 | 4.0170 | 0.4857 |
| full_rewt_rounds_1 | 4.0141 | 0.7727 |
| full_rewt_rounds_2 | 4.0135 | 1.0419 |
| full_rewt_rounds_4 | 4.0140 | 1.5620 |
| full_rewt_rounds_8 | 4.0151 | 2.4650 |
| full_rewt_states_only | 4.0116 | 1.0634 |
| full_without_communication | 4.0130 | 0.9504 |
| full_without_critic | 4.0132 | 0.8748 |
| full_without_diffusion | 4.0131 | 0.6487 |
| full_without_forecast | 4.0137 | 1.0036 |
| full_without_history | 4.0610 | 0.8981 |
| full_without_router | 4.0127 | 0.9382 |
| full_without_specialists | 4.0144 | 0.8724 |

Removed proposals/workspaces are excluded from initialization/routing and held at zero, while the complete energy's compatibility constraints remain. Communication uses shared bridges, so this is a workspace intervention, not retraining a new model with all related parameters removed.

## Short retraining and gradient filtering

Each variant starts from the same seed and sees the same short batch sequence. Fixed directions have fewer trainable parameters; separate-block masking also leaves 4,800 direction coordinates inactive (245,129 functional parameters versus 249,929 for the dense core). Masking changes the initial frequency norms as well. These are structural diagnostics, not capacity/norm-matched proof that joint coupling itself helps. Spikes use the preset norm threshold 5; filtered spikes are measured before the common clipping step, which avoids crediting the Fourier filter for clipping.

| Variant | Test bpb ↓ | Raw spike % | Filtered spike % | Maximum raw norm | Training CPU s |
|---|---:|---:|---:|---:|---:|
| coarse_to_fine | 4.5658 | 0.0 | 0.0 | 3.414 | 17.16 |
| fixed_frequencies | 4.5661 | 0.0 | 0.0 | 3.419 | 16.11 |
| gradient_clip | 4.5660 | 0.0 | 0.0 | 3.411 | 15.81 |
| gradient_ema | 5.3322 | 27.5 | 0.0 | 8.007 | 15.91 |
| gradient_fourier | 4.5933 | 0.0 | 0.0 | 3.715 | 16.55 |
| gradient_none | 4.5196 | 0.0 | 0.0 | 3.034 | 16.13 |
| separate_block_modes | 4.5652 | 0.0 | 0.0 | 3.405 | 16.66 |

A low-spike pilot cannot demonstrate robust spike suppression. Filtering can remove useful gradient variation as well as noise; quality must be assessed with the saved bpb values.

## Compute accounting

Primary and equal-time training CPU counters include minibatch construction, forward/backward, auxiliary losses/teacher forwards, gradient diagnostics/filtering and AdamW updates. They exclude validation, initialization, checkpoint persistence and evaluation; those stages have separate recorded counters. The full model is charged for its shared primary teacher once. The equal-time full fit prepares a fresh short teacher inside its common budget. One complete update may overshoot.

Native memory/timing profiles are isolated sequential processes, separate from concurrent training. The sampled RSS includes Python, Torch, parameters and data buffers; incremental RSS is also saved. Positive operator allocations are an allocation total, not peak resident memory.

The initial failed attempt is preserved separately. Its data are excluded from the completed model comparison. A replay and a long-window high-gain regression test document the stability fix.

## Sequential inference profiles

| Model | Median CPU ms/byte | Sampled process peak RSS MiB | Incremental RSS MiB |
|---|---:|---:|---:|
| Transformer | 0.0105 | 524.55 | 14.14 |
| Neural energy | 0.2393 | 522.60 | 13.12 |
| Fourier REWT | 0.2343 | 523.31 | 13.17 |
| Full REWT | 0.8885 | 526.36 | 17.67 |
