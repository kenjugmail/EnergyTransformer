# Recurrent Energy Wave Transformer experiment

## Whole-word language models

Use the new word-based path for custom language training and prediction:

```bash
.venv/bin/python scripts/predict_rewt_words.py --prompt 'Once upon a' --top 5
.venv/bin/python scripts/predict_rewt_words.py --prompt 'The dog was very' --generate 24
.venv/bin/python scripts/train_rewt_words.py \
  --output runs/rewt_words_new --report reports/rewt_words_new
```

The trained default checkpoint has 1,399,096 parameters and an 8,192-entry
whole-word vocabulary. It achieves 28.36% next-word accuracy on the recorded
story test sample. Words and integers remain intact; punctuation is separate,
spaces are reconstructed, and unseen words become one `<unk>`. There is no byte
fallback. It predicts the next word after a complete prompt word, rather than
finishing a partially typed word.

`StreamingConfig`, `EnergyConfig` and `TransformerConfig` now expose
`vocab_size`. Joint-energy memory/architecture variants inherit it, including
their output-state and cached-decoding dimensions. Config defaults remain 256
for old checkpoint compatibility; the word trainer sets the actual vocabulary
size. Use `--architecture joint`, `transformer` or `window-transformer` to train
other custom models with the same word data. The word tokenizer is fitted on
training documents only and embedded in new checkpoints. LFM3 remains on its
pretrained tokenizer.

**305 REWT tests pass**, including whole-word, corpus isolation, checkpoint,
gradient and cache tests. [Results, limitations and reproducibility](../reports/rewt_words_20261028/FINDINGS.md).

## Previous conversion and byte/subword experiments

The [LFM3 conversion](../reports/lfm3_conversion_20261028/FINDINGS.md) adds a
standalone 1.784B-parameter model in `lfm_conversion.py`, initialized from the
pinned LFM3-1.7B donor. It replaces 13 original convolution operators with delta
memory, bounds all 11 attention layers, and adds final latent-energy refinement.
The initial three recovery stages improve matched test continuation loss from 8.452 to
2.347 versus the donor's 2.312. Instruction and factual errors remain; full
capability recovery is not established. That recovery version passed **279 tests**.

The [subsequent capability recovery attempt](../reports/lfm3_capability_recovery_20261028/FINDINGS.md) improves the small science score from 43/128 to 59/128 and exact responses from 9/50 to 13/50. The recommended bundle is the second-pass checkpoint; the broader final pass regressed on these checks. This is partial recovery with post-hoc model selection, not donor parity.

```bash
.venv-lfm3/bin/python runs/lfm3_capability_recovery_20261028/converted/sample.py \
  --prompt 'Reply with exactly the word hello.' --tokens 48
```

The inference bundle contains its own model code, tokenizer and donor license.
See the [conversion design](../reports/lfm3_conversion_20261028/DESIGN.md) for
transfer details, data separation, mixed precision and recovery objectives.

The [complete decoder update](../reports/rewt_decode_speed_20261027/FINDINGS.md)
adds `FastDecodeSession` in `fast_decode.py`: ring attention, a direct one-token
delta recurrence, compilation and optional CUDA graphs. At an 8K prefix,
32-token energy-model generation is 1.89× faster on CPU and 12.44× faster on
the shared GB10. These are warmed measurements excluding prefill/setup;
all controls receive the same optimizations. **263 tests pass**, with 126
checked continuations matching the original. No new accuracy gain is claimed.

Add `--decoder compile` on CPU, or `--device cuda --decoder graph` on CUDA,
to `scripts/sample_rewt_streaming.py`. For immediate startup use
`--decoder eager`. In Python, construct `FastDecodeSession(model.eval(),
capacity=..., backend='compile')`, `consume(prompt_tokens)`, then optionally
`prepare()` to pay compilation before `generate(steps)`. `generate` returns
device token IDs, shape `[batch, steps]`, and retains the state for continuation.
External input/mode/weight checks occur at API calls; do not mutate weights
concurrently with a generation call. Default sampling retains the reference path.

The [energy execution update](../reports/rewt_energy_speed_20261026/FINDINGS.md)
adds `fast_energy.py` and an optional fused CUDA kernel. Both retain the same
weights and two energy updates. Measured energy-head wall time is about 4.4×
faster on CPU and 4.5× on a busy GB10; whole-model CPU decoding improves by
about 6–7%. No reliable training-speed gain is claimed. That update passed 245 tests,
and 126 checked story/task continuations match the original across three seeds.

For an energy checkpoint, add `--fast-energy` to the streaming sampler on CPU,
or `--device cuda --fused-energy` for CUDA float32. In Python, call
`optimize_energy(model)` from `rewt.fast_energy` before making a
`StreamingSession`; use `fused=True` after moving the model to CUDA for the
fused inference path. Existing optimizers and checkpoint keys are preserved.
Optional `compiled=True` compiles the differentiable training path but pays
startup cost; it is not the recommended CPU decoding option.

The [streaming hybrid comparison](../reports/rewt_streaming_20261026/FINDINGS.md)
adds `streaming.py` and owned, preallocated inference sessions in
`streaming_decode.py`. It combines gated delta memory, 64-token rotary attention,
SwiGLU and optional latent energy refinement. In 36 three-seed fits, the hybrid
improves story prediction at equal updates and long recall versus matched
Transformers, while losing on language quality at equal CPU time, short-prefix
latency and eager GPU training speed. The complete family breakdown, no-energy
controls and timing scope are retained. This is a new opt-in architecture;
earlier checkpoints and the separate structured rule head remain available.
That comparison passed **226 REWT implementation tests** before the energy execution additions.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt_streaming.py \
  --checkpoint runs/rewt_streaming_20261026/language_hybrid_26301/best.pt \
  --prompt 'Once upon a time, ' --tokens 96
```

See the [architecture and math](../reports/rewt_streaming_20261026/DESIGN.md).
The language checkpoints use the existing training-only BPE tokenizer; task
checkpoints are independent raw-byte autoregressive models. Neither uses the
structured reasoner's annotations or serializer during inference.

The [October 26 reasoning extension](../reports/rewt_reasoning_20261026/FINDINGS.md)
adds an opt-in structured rule head in `reasoning.py`: learned byte extraction,
soft memory lookup, and a changed query at each internal read. All three seeds
complete 1,440 fresh bounded rule prompts. Zero-energy controls tie the result,
so this supports rereading rather than a wave-refinement advantage. Its different
withheld validation phrasings score only 43.15%; general wording robustness and
free-form generation remain unsolved. The language-model sampler is unchanged.
That extension passed 207 REWT tests before the streaming additions.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt_reasoner.py \
  --checkpoint runs/rewt_reasoning_20261026/reread_energy_26103/best.pt \
  --prompt $'B>C;A>B;C>D;X>Y;from A go 3=?\n'
```

This returns predicted trace nodes and a predicted length using a structured
serializer. It is not autoregressive text generation. Each read defines a new
conditional energy; only updates within one read have a descent contract.
See the [ten-minute presentation](../course_project/presentation_20261028/REWT_class_presentation.pptx)
and [speaker notes](../course_project/presentation_20261028/speaker_notes.txt).

The [completed architecture follow-up](../reports/rewt_architecture_20261022/FINDINGS.md)
adds general and preconditioned solvers, structured preparation, scan/delta-memory
variants, ring decoding and optional Triton/CUDA-graph execution. It includes
96 new training fits, matched memory controls, numerical audits and complete
positive and negative results. Usage is documented at the end of this file.

The [October 22 kernel investigation](../reports/rewt_kernels_20261022/FINDINGS.md)
adds an optional two-round mode-space solver and persistent inference caches.
The unchanged broader-task checkpoint runs about 2.11× faster for batched CPU
forward passes, about 1.5–1.9× faster for batched forward/backward workloads,
and 2.06–2.27× faster for the measured compiled generation workloads. All 648
validation generations match the frozen reference byte for byte; 144 tests pass.
The report derives the reduction and seven further kernel/architecture
directions, including certified steps, a 32-coordinate preconditioned solver,
and a parallel observation-memory backbone. GPU correctness was checked;
GPU throughput and new learned-quality gains are not claimed.

This package implements the course proposal as a self-contained PyTorch pilot.
Every language model starts from random weights. No older repository checkpoint
or external pretrained language model is used.

The [October 10 learning repair](/home/kt/Documents/EnergyTransformer/reports/rewt_learning_20261010/FINDINGS.md)
now gives a usable bounded one-step checkpoint: 125/132 complete arithmetic
responses, 80/96 rule responses and 96/96 recall responses on a fresh test in
familiar formats. It has 249,952 inference parameters, fixed learned weights,
and two rounds of joint energy refinement. Unseen wording remains weak. The
report includes actual generated responses, a zero-round counterfactual,
training costs and the current broader-learning limitations.

A separate position-aware continuation now completes 30/41 fresh two-rule traces
and 211/211 recall responses after 4, 8 or 16 distractors in familiar formats.
Two-addition accuracy is 16/72; three/four-step traces fail. The separate story
model reaches 1.448 test bits per raw byte, but its generated prose remains
incoherent. These are single-initialization results with fixed inference weights.

The [latest broader-task polish](../reports/rewt_polish_20261010_v2/FINDINGS.md)
adds a separate 256,096-parameter checkpoint with 432/432 complete fresh recall
responses, including new wording and a randomly placed key among 4/8/16
distractors. Three cold energy seeds reach 96.3–100% new-wording recall. Their
mean complete accuracy over all broader tasks is 36.3%, versus 23.9% for the
matched Transformer controls, at 6.82× the training CPU cost. Extended arithmetic
remains poor, so the older one-step checkpoint stays available. All 100 then-existing
REWT tests passed at that snapshot. The report includes every output, the fixed-weight audits,
remaining failures and runnable commands.

Generate with the narrower one-step checkpoint from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/binding/aligned/best.pt \
  --prompt $'A=19;B=A+23;B=?\n' --greedy --stop-newline --max-new 40 --cache-history
```

The core now supports optional GRU recurrent proposals, causal context encoding,
anchor-relative quadratic constraints and waves, and configurable initial carry.
`rewt.memory` splits learned retrieval channels across heads; `rewt.position_memory`
adds learned positions for observed tokens. Story loaders enforce document
boundaries, and optional 256-piece vocabularies are trained only on training text.
Defaults preserve archived checkpoint behavior. Training-only regression and
attention labels are not inference inputs or components.

`--cache-history` also supports the multi-head and position-memory checkpoints.
It reuses causal observed-token encodings within a forward call, with matching
predictions and parameter gradients in the implementation checks. The measured
batch-24, 64-byte CPU workload is about 2.03 times faster for forward passes and
1.77 times faster for forward plus backward. Short-answer generation on the
checked addition prompt is about 1.06 times faster. These are different
workloads; the cache does not persist inference keys/values across decode calls.
Exact measurements and scope are in the
[broader-task reproduction notes](../reports/rewt_polish_20261010_v2/REPRODUCING.md).

The new opt-in `--refinement-backend krylov2 --decode-cache` path also retains
observation projections and mode matrices across generated bytes. It requires
the anchor-relative Fourier core with zero initial carry and specializes two
rounds; interventions and other round counts retain the reference solver.
Add `--compile-refinement` for repeated workloads where its startup cost is
acceptable. [Commands and constraints](../reports/rewt_kernels_20261022/REPRODUCING.md).

The core joint workspace contains recurrent, attention, feedforward and output
states. The full model adds multi-horizon forecasts, denoising candidates, three
specialists and a communication state. A positive quadratic energy anchors
input-conditioned proposals and scores linear compatibility constraints;
context-weighted cosine modes span the entire joint workspace. The original pilot
uses sixteen modes, and the October 10 one-step checkpoint uses thirty-two. Their
directions, amplitudes and phases are learned. A conditional critic and router set coefficients
before the optimization rounds. Each accepted refinement step decreases the same
complete energy; a coarse proposal is still checked against its complete energy.

Neural proposals initialize guesses. Diffusion proposes candidate initializations
and compares them to a keep candidate by full energy. Forecasts never become
observed bytes. Frozen future targets, correct-answer candidates and critic
utilities appear exclusively in training losses. Inference adapts states and
logits, with all learned weights fixed.

Run from the repository root:

    .venv/bin/python -m pytest tests_rewt -q
    .venv/bin/python -m rewt.experiment

The revised experiment is available with:

    .venv/bin/python -m rewt.improve

It screens fourteen configurations using validation alone, then trains a Transformer,
raw REWT control, revised non-Fourier energy control, revised Fourier REWT and
full REWT across three seeds. The default language budget is 1,048,576 target
bytes per fit, about 17.1 times the original pilot. Reasoning copies receive 800
adaptation updates with a balanced one-to-three-step curriculum. New held-out
language windows, text prefixes and synthetic task identities are declared in
the plan before training. Larger studies can set --lm-updates and --reason-updates
in a fresh output directory.

The revised energy uses zero-mean output coordinates in its anchors, quadratic
compatibility terms, waves and exact analytic gradients. It is invariant to a
common logit shift; its quadratic is positive on the admissible centered domain.
Raw geometry stays available for faithful loading of the archived pilot.
Initial training guesses can be corrupted with label-free noise; all accepted
refinements still lower one fixed energy. Hard negatives, local prediction-repair
feedback and first-invalid-step labels enter only training losses. The latter
ranks the first incorrect trace byte under its unchanged correct prefix, rather
than claiming to verify whole solutions. The full model's forecasting targets
come from frozen, course-trained language and reasoning teachers, with their
preparation costs recorded.

Mode count, frequency bound, normalized frequency-block scales and positive
update size are configurable. Training can sample one to four refinement rounds.
Validation chooses an inference policy from bounded rounds and optional stopping
based on relative energy and probability change. Stable energy does not certify
a correct answer. Cached causal keys/values, CPU backtracking early exits and
optional trace-free training reduce redundant work. The Fourier and EMA residual
filters add a slow component while preserving the current raw gradient; ordinary
clipping and the original replacement filters remain experimental controls.

Revised results go to reports/rewt_improvements_20261009 and checkpoints to
runs/rewt_improvements_20261009. The runner saves source/data hashes, optimizer
histories, training-budget checkpoints, validation selection, actual generated
text and reasoning, energy audits, and isolated process timing/RSS profiles.
Resume only the same frozen source with --resume. Pending human text ratings are
never synthesized by the training program.

The first long follow-up was stopped after unstable full-model reasoning
gradients. Its source, replay diagnostics and partial records are preserved.
The repaired replacement evaluation runs with:

    .venv/bin/python -m rewt.stabilize

It reuses the twenty-seven completed parent fits with exact checkpoint and
source provenance, verifies that their inference logits remain bitwise equal,
and restarts all three full-model reasoning fits from their language weights.
Only training credit through carried latent states is bounded to sixteen-token
segments; forward memory and inference still retain the full 128-byte window.
Gradient norms and spectral powers accumulate in float64, with actual non-finite
components rejected. Fresh test identities and text/language windows exclude
the original pilot and the stopped follow-up, and are frozen before the restarts.
Results and checkpoints go to reports/rewt_stabilized_20261009 and
runs/rewt_stabilized_20261009. Use --resume only with the same frozen source.

The [completed follow-up findings](/home/kt/Documents/EnergyTransformer/reports/rewt_stabilized_20261009/FINDINGS.md)
confirm the output-geometry repair but find no clear Fourier-specific quality gain.
Revised Fourier test bpb is 3.268, versus 3.271 for the raw control and 3.267
for the revised neural energy control. Generated reasoning remains near zero,
sampled text is incoherent, and the full model still has severe raw gradient spikes.
All 50 implementation tests passed. The report includes locked exploratory
assistant text ratings, fresh evaluations, actual generation timings and a
separate validation-selected Transformer check; independent human ratings are pending.

The default is a bounded CPU pilot: Transformer, a same-state non-Fourier tanh
energy control, basic Fourier REWT and full REWT, at about 250K working parameters,
three seeds, 128-byte windows, 120 LM updates and 240 reasoning adaptation updates.
It also runs a separate 30-CPU-second fit for each arm, including teacher training
within the full model's budget; short single-seed retraining tests the frequency,
coupling, schedule and gradient-filter choices. Inference probes cover refinement
rounds, state/output updates and each supporting component.

Evaluation saves held-out enwik8 bpb/Brier, actual sampled text, exact generated
arithmetic and rule-chain answers with checked intermediate steps, distractor
recall, gradient spikes, accepted-energy traces, CPU time and cumulative worker
memory. The anonymous text-review page supplies the fixed fluency/coherence/
relevance rubric. Pending human scores are explicitly pending.

Results go to reports/rewt_pilot_20261009_v2; weights and exact optimizer continuation
checkpoints go to runs/rewt_pilot_20261009_v2. Use --resume to continue the frozen plan.
The plan records source/data hashes, data offsets, task identities, seeds and all
settings. Changed training source is rejected on resume. Choose fresh output/runs
directories when changing the experiment.

This small run tests functionality and early behavior. It cannot establish
competitive text generation, general reasoning, or a scaling advantage. The
three-page proposal's larger experiment remains a later study.

The original reports/rewt_pilot_20261009 run was stopped for a recurrent numerical
instability. Version 2 bounds the previous recurrent state only when using it as
conditioning for new guesses. Every accepted state still comes from the same
energy descent; it is never clipped after acceptance. The failed run and source
snapshot are retained, and all version-2 models restart from random weights.
# Architecture-specific execution and learning study (October 22, 2026)

The [architecture study](/home/kt/Documents/EnergyTransformer/reports/rewt_architecture_20261022/DESIGN.md)
implements packed conditioning, a general directional line search, certified
fixed steps, arbitrary-depth preconditioned refinement in Fourier-mode space,
and a structured Schur solve. Optional scan, chunk-feedback, delta-memory,
fixed-patch, mode-diversity and factorized-head variants have separate training
comparisons. These variants are explicit configurations; legacy checkpoints keep
their original defaults.

The execution paths have different contracts:

- `packed_projections=True` with `cache_history=True` preserves the original
  parameters while batching conditioning and GRU input projections.
- `refinement_backend='directional'` preserves the full conditional energy and
  evaluates line-search candidates through quadratic directional identities.
  It supports the full model and arbitrary rounds; a CPU first-trial early exit
  can still be faster on easy steps.
- `ArchitectureConfig(solver='preconditioned')` changes the finite-step update
  rule. It performs all rounds in M coordinates, with one final lift. Train and
  evaluate that policy explicitly. `preconditioner_factorization='structured'`
  is an exact preparation alternative that factors only the 3d latent block.
- `RingDecodeSession` owns a fixed-size cache for the position-memory GRU core.
  `cuda_graph=True` captures the entire step; `triton=True` uses the optional
  float32 inference kernel for a preconditioned model. `triton_tile=64/128/256`
  trades repeated mode work for lower lift register pressure. No custom Triton
  backward is implemented.

For an existing compatible checkpoint, the exact execution options are:

```bash
.venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt \
  --prompt 'your prompt' --greedy --cache-history --packed-projections \
  --refinement-backend krylov2 --ring-cache
```

For a separately trained preconditioned checkpoint:

```bash
.venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_architecture_20261022/precond4_22101/best.pt \
  --prompt 'your prompt' --greedy --structured-preconditioner \
  --device cuda --disable-tf32 --cuda-graph --triton-refinement --triton-tile 64
```

Scan, delta-memory and patch models use their normal carried `ArchitectureState`;
the GRU ring session rejects them. After moving a model to its final device, call
`model.eval().prepare_inference()` to cache fixed-weight factors. Training always
rebuilds differentiable factors. Changed weights, buffers, configuration or
training mode invalidate fixed-weight sessions; create/prepare a new one.
Mutation through `.data` is unsupported because it bypasses version tracking.

Language checkpoints under `runs/rewt_architecture_20261022/language` include
separate `inference.pt` files with the verified BPE vocabulary metadata. Use
those with `sample_rewt.py`; frozen `best.pt` files remain the original training
artifacts. The language screen is a small-budget study, not a fluency claim.

See the [results and limitations](/home/kt/Documents/EnergyTransformer/reports/rewt_architecture_20261022/FINDINGS.md)
and [reproduction instructions](/home/kt/Documents/EnergyTransformer/reports/rewt_architecture_20261022/REPRODUCING.md).

