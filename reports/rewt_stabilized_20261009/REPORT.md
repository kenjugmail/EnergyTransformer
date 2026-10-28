# REWT fixes and controlled follow-up

This is the completed replacement evaluation for a stopped follow-up. Twenty-seven completed fits (all fifteen language fits and twelve control reasoning fits) reuse the parent study's exact checkpoint bytes and original source/training records. All three full-model reasoning fits restart from their completed language weights; their training gradients detach carried latent states every sixteen observed tokens. Forward state values, attention history and the full 128-byte memory window are retained. Float64 gradient-norm and spectral-power reductions prevent a finite-vector norm from overflowing; actual non-finite components are rejected.

The fresh test windows, text prefixes and synthetic test identities were declared before these restarts. They exclude both the original pilot and the stopped follow-up's evaluations. No partial scores from the stopped run enter this report. The archived and current implementations produced bitwise-equal logits for all twenty-seven reused checkpoints under the recorded parity probe. Reused-source checksums and the current source snapshot remain separately identified in the plan.

This report records models trained from scratch for this project and actual generated outputs. The revised implementation uses centered output coordinates throughout the joint energy and its analytic derivatives, noisy training guesses, hard ranking negatives, refinement-repair feedback, variable training depth, a reasoning curriculum with first-error supervision, configurable Fourier capacity, cached causal history, and optional early stopping. Residual gradient filtering is a separate validation-screened option.

Five variants × 3 seeds; about 250,000 parameters each. Every language fit starts from random weights and processes 1,048,576 target bytes in 256 updates with batch 32 and 128-byte resets. The raw control keeps raw logits, 16 modes, fixed two-round training, ordinary initialization and easy negatives. It receives the same longer byte budget and common selected learning rate. The four revised comparison arms share the selected recipe; the Transformer receives the same curriculum and first-error labels, with a logit ranking loss rather than an energy ranking loss.

Validation screening selected **repair_32_faster_lr**. All screening cases, validation scores, fixed recipe values, source snapshots and checksums are saved. Selection used no test outputs. Frequency-mode comparisons compensate with feedforward width to preserve the parameter budget; they test budget allocation and are not isolated frequency-only effects.

The follow-up uses new nonoverlapping language windows and text prefixes. Synthetic test identities exclude the previous pilot's test identities and remain disjoint from current training and validation by semantic hash. Arithmetic starting values expand to 0–19 and increments to 1–19. These changes mean that comparisons with the previous pilot's numerical test scores are not paired estimates of a repair effect.

Language and cost below are mean ± sample standard deviation across seeds. Lower bpb is better. Inference policies are selected on validation from 1/2/4/8 rounds and two bounded halting thresholds. The policy must remain within 0.002 bpb of the best validation alternative before selecting measured CPU cost. Energy or probability stability is a compute stopping feature, not a correctness certificate.

| Variant | Parameters | Fresh test bpb ↓ | Equal-time bpb ↓ | Training + teacher CPU s | Inference ms/byte |
|---|---:|---:|---:|---:|---:|
| Transformer | 249,994 | 4.051 ± 0.004 | 4.060 ± 0.005 | 48.9 ± 1.1 | 0.035 ± 0.002 |
| Raw REWT control | 249,929 | 3.271 ± 0.024 | 4.144 ± 0.066 | 230.2 ± 2.2 | 0.062 ± 0.025 |
| Revised neural energy | 249,989 | 3.267 ± 0.045 | 4.331 ± 0.026 | 329.5 ± 8.6 | 0.066 ± 0.014 |
| Revised Fourier REWT | 249,989 | 3.268 ± 0.044 | 4.316 ± 0.022 | 337.5 ± 3.2 | 0.046 ± 0.010 |
| Revised full REWT | 250,020 | 3.367 ± 0.010 | 5.488 ± 0.081 | 1101.3 ± 15.0 | 0.187 ± 0.057 |

Separate fresh-process profiles use each selected inference policy and measure RSS before operator-profiler bookkeeping. RSS includes Python, Torch, model and data buffers; incremental RSS is sampled every 2 ms and is not a precise activation-allocation peak.

| Variant | Sequential inference ms/byte | Process peak RSS MiB | Incremental inference RSS MiB |
|---|---:|---:|---:|
| Transformer | 0.0164 | 569.0 | 55.9 |
| Raw REWT control | 0.0357 | 537.7 | 23.1 |
| Revised neural energy | 0.0447 | 541.6 | 26.8 |
| Revised Fourier REWT | 0.0329 | 541.6 | 26.7 |
| Revised full REWT | 0.1647 | 557.8 | 40.6 |

The separate equal-time fits have 60 process-CPU seconds from new random weights. Full REWT spends 25% on a fresh forecast teacher and the remaining budget on its own fit. One final update may overshoot the cap. Main full-model cost charges its shared course-trained language teacher. Reasoning uses the baseline's separately adapted reasoning teacher, freezes it, and charges that adaptation once; its earlier language preparation is already included in the language pipeline. Common student bytes do not imply equal total pipeline compute.

Reasoning copies receive 800 updates, batch 16. The revised curriculum first presents one-step arithmetic/rules and short recall, then expands to two and three steps. A generated wrong trace is labeled through its first invalid step; the corresponding incorrect next byte is ranked under the unchanged correct prefix in the training loss. This is local process feedback, not an independently trained whole-solution verifier. No symbolic checker, expected answer or future label enters inference.

Generated reasoning is scored after greedy autoregressive generation. Arithmetic and rule chains are separated from easier recall. Complete correctness requires every intermediate step and the final answer. ID uses seen wording and at most three reasoning steps; OOD uses withheld wording or four-step chains. These tasks measure bounded skills, not general reasoning.

| Variant | Reasoning answers % | Complete reasoning % | ID complete % | OOD complete % | Recall complete % | Seen training complete % | Reasoning + teacher CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 3.0 ± 0.5 | 0.7 ± 0.6 | 1.3 ± 1.1 | 0.0 ± 0.0 | 52.0 ± 10.4 | 30.2 ± 5.5 | 68.1 |
| Raw REWT control | 2.8 ± 1.2 | 0.2 ± 0.3 | 0.3 ± 0.6 | 0.0 ± 0.0 | 6.7 ± 2.1 | 5.2 ± 3.8 | 510.2 |
| Revised neural energy | 3.3 ± 1.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.7 ± 1.5 | 5.2 ± 2.1 | 500.7 |
| Revised Fourier REWT | 4.2 ± 1.2 | 0.2 ± 0.3 | 0.3 ± 0.6 | 0.0 ± 0.0 | 10.3 ± 4.9 | 3.8 ± 2.4 | 499.6 |
| Revised full REWT | 4.3 ± 2.0 | 0.2 ± 0.3 | 0.3 ± 0.6 | 0.0 ± 0.0 | 8.7 ± 1.5 | 5.2 ± 1.8 | 1791.4 |

Training-only constant-response priors reach 0.0% complete reasoning without family information and 1.0% when the family is supplied. The latter is an oracle diagnostic, not a generative model.

Generated-text evidence includes all 450 96-byte sampled continuations after the same fresh 32-byte prefixes, with temperature 0.8 and paired sampling uniforms. Printable and repetition rates are diagnostic measurements; they do not establish fluency or coherence. The [anonymous review](text_review.html) uses the fixed 1–5 fluency/coherence/relevance rubric. Human ratings remain pending. Any assistant review is identified separately and must be locked before reading the identity key.

| Variant | Printable bytes % | Repeated byte trigrams % | Language-training raw gradient spikes % |
|---|---:|---:|---:|
| Transformer | 100.0 ± 0.0 | 7.8 ± 0.4 | 0.0 ± 0.0 |
| Raw REWT control | 99.9 ± 0.1 | 8.9 ± 1.7 | 0.0 ± 0.0 |
| Revised neural energy | 100.0 ± 0.0 | 9.7 ± 1.7 | 0.0 ± 0.0 |
| Revised Fourier REWT | 100.0 ± 0.0 | 9.5 ± 1.2 | 0.0 ± 0.0 |
| Revised full REWT | 99.9 ± 0.0 | 8.9 ± 1.1 | 0.0 ± 0.0 |

The spike column describes language-training updates under the declared raw-norm threshold of five. Full-model reasoning adaptation still exhibits severe finite gradient spikes; see the [reasoning stability audit](/home/kt/Documents/EnergyTransformer/reports/rewt_stabilized_20261009/STABILITY.md). All 450 anonymous primary continuations have now been read and rated in the [exploratory assistant review](/home/kt/Documents/EnergyTransformer/reports/rewt_stabilized_20261009/TEXT_QUALITY_REVIEW.md), with ratings locked before joining model identities. Every continuation received coherence 1/5. Human ratings remain pending. The [completed findings](/home/kt/Documents/EnergyTransformer/reports/rewt_stabilized_20261009/FINDINGS.md) bring the quality, stability and actual generation-cost results together.

Validation-only local diagnostics freeze the original prefix conditioning and compare rounds 0/1/2/4/8 from its actual initialization. They record energy, prediction loss, gradient alignment and the common-logit shift probe. These measurements isolate local refinement and are distinct from autoregressive rollouts that alter subsequent recurrent conditioning.

| Variant / seed | Common output gradient squared share % | Initial → two-round bpb | Initial → two-round energy |
|---|---:|---:|---:|
| Raw REWT control / 101 | 58.8015 | 3.2338 → 3.2314 | 72.612 → 51.568 |
| Revised neural energy / 101 | 0.0000 | 3.1817 → 3.1801 | 36.867 → 30.957 |
| Revised Fourier REWT / 101 | 0.0000 | 3.1646 → 3.1629 | 33.714 → 28.790 |
| Revised full REWT / 101 | 0.0000 | 3.3030 → 3.3008 | 78.365 → 60.790 |
| Raw REWT control / 202 | 59.0510 | 3.1943 → 3.1945 | 66.610 → 44.782 |
| Revised neural energy / 202 | 0.0000 | 3.1816 → 3.1812 | 32.091 → 25.959 |
| Revised Fourier REWT / 202 | 0.0000 | 3.1857 → 3.1855 | 30.783 → 25.334 |
| Revised full REWT / 202 | 0.0000 | 3.3588 → 3.3563 | 187.661 → 150.036 |
| Raw REWT control / 303 | 57.7772 | 3.1019 → 3.1027 | 66.747 → 45.849 |
| Revised neural energy / 303 | 0.0000 | 3.1433 → 3.1431 | 35.346 → 29.238 |
| Revised Fourier REWT / 303 | 0.0000 | 3.1330 → 3.1326 | 36.079 → 30.079 |
| Revised full REWT / 303 | 0.0000 | 3.2765 → 3.2746 | 32.121 → 27.839 |

Single-seed validation screening:

| Case | Validation bpb ↓ | Filter | Eligible recipe |
|---|---:|---|---|
| repair_32_faster_lr | 3.6292 | clip | True |
| repair_32_residual_filter | 3.8309 | fourier_residual | True |
| centered_16 | 3.8318 | clip | False |
| repair_16 | 3.8324 | clip | True |
| raw_16 | 3.8337 | clip | False |
| repair_32_ema_residual | 3.8431 | ema_residual | True |
| repair_32 | 3.8482 | clip | True |
| repair_32_small_step | 3.8491 | clip | True |
| repair_32_high_band | 3.8498 | clip | True |
| repair_32_no_clip | 3.8500 | none | False |
| repair_64 | 3.8505 | clip | True |
| repair_32_low_band | 3.8530 | clip | True |
| repair_32_replacement_filter | 3.8754 | fourier | False |
| repair_32_ema_replacement | 4.7638 | ema | False |

The study bundles several training repairs; the main revised-versus-raw comparison cannot assign a gain to one component. Screening is one seed and a short budget. The non-Fourier energy control is essential to assessing whether a benefit is specific to Fourier modes.

The full model retains forecasting, diffusion proposals, specialists, communication and critic/router conditioning under one fixed joint energy. At a fixed parameter budget, those modules reduce feedforward width; that capacity tradeoff remains explicit. Frozen teacher targets also inherit the teacher's training limits. The finite wave basis is not an exact representation of an arbitrary high-dimensional energy landscape.

CPU timings come from a shared host and should not be read as GPU speed claims. Cumulative worker RSS is not per-model activation memory. Source/data hashes, checkpoint paths, all recipe coefficients, task labels, raw outputs and optimizer histories are saved. The archived original pilot is unchanged. Resume with the same source and --resume; larger byte budgets require a newly declared run.
