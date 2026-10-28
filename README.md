# EnergyTransformer

Experimental energy-function memory, native-text prototypes, and explicitly reviewed
protected-memory edits. **Not a competitive general-purpose LLM or an instruction-following assistant.**

## Whole-word prediction — October 28

The current custom-model training path now predicts **whole words**, with
separate punctuation tokens, instead of byte or subword fragments. A new
**1.40M-parameter streaming REWT** is trained with an **8,192-entry vocabulary**.
On the held-out story sample it predicts **28.36%** of next-word positions
correctly; `Once upon a` gives `time` as the leading word at **95.28%**.

```bash
.venv/bin/python scripts/predict_rewt_words.py --prompt 'Once upon a' --top 5
.venv/bin/python scripts/predict_rewt_words.py --prompt 'The dog was very' --generate 24
```

The local checkpoint is `runs/rewt_words_20261028/best.pt`; its vocabulary is
embedded. Unknown words map to `<unk>`, with no byte fallback. The original
joint-energy model and attention controls also support word vocabularies through
`scripts/train_rewt_words.py`. Historical checkpoints retain their original
tokenization, and LFM3 retains its pretrained tokenizer.

[Word-model results, limitations and training commands](reports/rewt_words_20261028/FINDINGS.md).

## October 28 class presentation

The updated ten-minute talk covers compact-memory results, decoding speed, the
LFM3 conversion and partial capability recovery. It has 12 main slides, three
appendices and speaker notes for every slide.

- [Google Slides](https://docs.google.com/presentation/d/11p2zXD7gTftP4EZXtRqXL948wSRPqkao5pTgoINKgVQ)
- [Full presenter reference in Google Docs](https://docs.google.com/document/d/1lTWL-q_2IcA7wKsH_Di9v4lonHuqEd7sPTFNCMgSKFU)
- Offline [slides PDF](course_project/presentation_20261028/REWT_class_presentation.pdf), [editable PowerPoint](course_project/presentation_20261028/REWT_class_presentation.pptx), [reference PDF](course_project/presentation_20261028/presenter_reference.pdf) and [searchable HTML reference](course_project/presentation_20261028/presenter_reference.html).

The reference includes timed speaking notes, result tables, demo commands and
backups, likely questions, equations and source links. The completed repository
test run passed **580 tests with 32 skipped**; this verifies implementation,
not general model capabilities. Model checkpoints, downloaded corpora and
virtual environments remain local under the existing exclusions.

## LFM3 capability recovery — October 28

A further recovery attempt processes **1.66 million token exposures** across three
passes. The recommended second-pass checkpoint includes 1.29 million of those exposures
and improves a small fixed science
comparison from **43/128 to 59/128** and exact-response checks from **9/50 to 13/50**.
The original donor scores 94/128 and 25/50. Full capability recovery is **not achieved**;
long recall and structured tool-call JSON remain weak. The final full-FFN pass
regressed on these checks despite lower validation loss, so both versions are retained.

[Recovery results, raw outputs, selection caveat and runnable bundle](reports/lfm3_capability_recovery_20261028/FINDINGS.md).
The final recommendation uses this exploratory suite; it is not an independent test result.

## LFM3-1.7B architecture conversion

A standalone **1.784B-parameter streaming REWT** now inherits LFM3's token,
feedforward and compatible attention weights. Its 13 original convolution
slots become delta memory, all 11 attention layers are bounded to 256 tokens,
and a learned energy head refines the final latent state. The initial three recovery stages
complete **1,024 updates and 851,968 training-token exposures**.

On the same 8,192 held-out token predictions, loss falls from **8.452 to 2.347**;
the donor scores **2.312**. JSON extraction recovers in the fixed sample, but
format following and factual wording still show errors. This is a working
architecture conversion, **not established capability parity** with LFM3.
Carried state is 18.783 MiB per sequence, excluding weights and temporaries.
All **277 tests pass**, with actual full-width FP32 chunk verification.

- [Results, limitations, raw samples and runnable bundle](reports/lfm3_conversion_20261028/FINDINGS.md)
- [Architecture, weight mapping and recovery objective](reports/lfm3_conversion_20261028/DESIGN.md)

## October 27: faster complete decoding and proposal showcase

Fixed-shape decoding, ring attention caches, compilation and CUDA graph replay
now accelerate the complete streaming model. At an 8K prefix, generating 32
tokens with the same energy checkpoint takes **15.43 → 8.18 ms on CPU (1.89×)**
and **24.51 → 1.97 ms on the shared GB10 (12.44×)**. These are warmed decode
times; prefill and startup are separate. All four architecture controls receive
the same optimizations. **263 tests pass; 126 checked continuations match.**

Use `--decoder compile` on CPU or `--device cuda --decoder graph` with the
streaming sampler. [Full timings, controls and usage](reports/rewt_decode_speed_20261027/FINDINGS.md).
The [October 28 showcase brief](course_project/presentation_20261028/streaming_showcase/presenter_brief.txt)
includes a reproducible memory demo, presentation figures and possible use cases.
The current evidence supports compact synthetic recall; energy's added accuracy
benefit and real application performance remain open questions.

## October 26: faster energy inference

The streaming energy head now has checkpoint-compatible CPU and fused CUDA
inference paths that retain both refinement steps. Measured energy-head wall
time falls from **68.9 to 15.7 µs on CPU** and **177.0 to 39.0 µs on the shared
GB10**. Whole-model CPU decoding takes about **6–7% less time**; the head speedup
does not apply to the entire model. Training gains remain unestablished.
That update passed **245 REWT tests**, and **126/126** checked greedy continuations matched.

Use `--fast-energy` with the streaming sampler on CPU, or
`--device cuda --fused-energy` for two-round float32 CUDA decoding.
[Measurements, limits and commands](reports/rewt_energy_speed_20261026/FINDINGS.md).

## October 26: streaming hybrid versus matched Transformers

A new opt-in autoregressive model combines gated delta memory, local attention,
and optional latent energy refinement. **36 cold fits across three seeds** compare
it with full and windowed Transformers at almost identical parameter counts.
The hybrid without refinement achieves **1.5638 versus 1.6065 story bits/byte**
at equal updates, **37.62% versus 22.34%** complete synthetic-task accuracy, and
**86.11% versus 14.35%** recall with 32–128 distractors. The task gain is mainly
retrieval; arithmetic and multi-step rule generation remain weak.

At an 8,192-token prefix, its CPU decoding is **1.54× faster** than the
preallocated full Transformer and carries **228× less cache/state**. It loses
on short-prefix speed, GPU training speed, and language quality at equal CPU
training time. Windowed attention is faster at every measured decode length.
Energy refinement slightly changes language loss but adds cost; it does not
improve the frozen model's task accuracy. This does **not** establish superiority
in every setting. That version passed **226 REWT tests**, with GB10 numerical checks.

- [Complete results, controls, timings and runnable sampler](reports/rewt_streaming_20261026/FINDINGS.md)
- [Architecture, equations and comparison protocol](reports/rewt_streaming_20261026/DESIGN.md)
- [New model implementation](rewt/streaming.py)

## October 26: learned rereading and class presentation

An opt-in **structured rule-reasoning head** now uses each predicted node to
query memory again. All three seeds solve 1,440 fresh one-to-three-hop prompts,
including eight distractor edges. Fixed-query controls average 27.4% on the main
test. Zero-energy versions also solve every test prompt: the demonstrated gain
comes from learned extraction and repeated reads, with no measured accuracy
benefit from the waves. Different withheld validation phrasings remain difficult.

This is a specialized head with predicted trace nodes and read count, not an
upgrade to free-form language generation. Inference receives raw bytes and
fixed learned weights; symbolic annotations are training/scoring only.
That extension passed **207 REWT tests**; the subsequent streaming suite has 226.

- [Results, limits, ablations and runnable demo](reports/rewt_reasoning_20261026/FINDINGS.md)
- [Editable Google Slides for October 28](https://docs.google.com/presentation/d/11p2zXD7gTftP4EZXtRqXL948wSRPqkao5pTgoINKgVQ)
- [Ten-minute class presentation: editable PowerPoint](course_project/presentation_20261028/REWT_class_presentation.pptx)
- [Presentation PDF](course_project/presentation_20261028/REWT_class_presentation.pdf)
- [Timed speaker notes](course_project/presentation_20261028/speaker_notes.txt)
- [Earlier kernels, persistent memory and language experiments](reports/rewt_architecture_20261022/FINDINGS.md)

## Recurrent energy wave follow-up

The October 10 REWT polish adds causal memory caching and a broader retrieval
checkpoint. It completes **432/432 fresh digit-recall prompts**, including new
wording, random placement of the requested key and 4/8/16 distractors. Three
cold energy seeds reach 96.3–100% recall on new wording. The measured batch CPU
workload is 2.03× faster for forward passes and 1.77× for forward plus backward;
short-answer generation improves by about 6%. Inference weights remain fixed.

On 1,296 broader fresh prompts, cold energy models average **36.3% complete
accuracy versus 23.9% for the Transformer**, at the same examples and update
budget but **6.82× the training CPU cost**. Extended arithmetic and rule chains
remain weak. This is a bounded retrieval result, not general Transformer
superiority. All 100 REWT implementation tests pass.

The earlier one-step checkpoint remains useful for inputs 0–31 in familiar
formats: 94.7% arithmetic, 83.3% rule following and 100% recall. Its separate
position-aware continuation reaches 73.2% on two-rule traces in the earlier
familiar formats. The broader model does not supersede those narrower task
results. Story generation still produces incoherent prose.

- [Latest broader retrieval, three-seed comparison and execution measurements](reports/rewt_polish_20261010_v2/FINDINGS.md)
- [Current learning findings and runnable sampler](reports/rewt_learning_20261010/FINDINGS.md)
- [Earlier three-seed findings and measurements](reports/rewt_stabilized_20261009/FINDINGS.md)
- [Implementation and runnable commands](rewt/README.md)

## Mixed energy and local attention

The native model now also supports alternating energy-recurrent blocks with
sliding-window attention. This combines a fixed recurrent state with a bounded
attention cache, and works with the existing optional fact readers and graph
workspace. Full-attention and windowed Transformer controls use the same codebase.

- [Architecture and training commands](reports/competitive/ARCHITECTURE.md)
- [Latest compute-controlled confirmation](reports/compute_confirmation/REPORT.md)
- [Kernel and preallocated-cache measurements](reports/stronger_comparison/REPORT.md)
- [Original token-matched comparison](reports/competitive/REPORT.md)

In the latest three-seed, approximately 145,000-parameter comparison, the mixed
model reached **2.512 test bits per byte versus 2.578 for full attention and
2.577 for windowed attention**, at equal steady-state CPU training budgets.
All arms use compiled training, the same tuning allowance, and one CPU core
with interleaved time blocks. Compilation and evaluation costs are reported
separately. All three hybrid seeds beat both controls at the trained context.

The CPU kernel is faster, and 32K-prefix decoding beats a preallocated full
Transformer cache. Full attention is faster at shorter prefixes; windowed
attention remains faster at every measured length. These small-model results
do not establish general superiority over Transformers or GPU efficiency.

Sample the saved checkpoint on CPU with the installed package:

```bash
.venv/bin/python -m energy_real.sample \
  --checkpoint runs/compute_confirmation_selected/hybrid.pt --prompt 'The ' \
  --tokens 128 --temperature 0.8 --top-k 40 --seed 42
```

Omit the sampling options for greedy output. Prompt processing uses bounded
chunks; `--output sample.bin` saves exact bytes to a new file. See the
[original usability follow-up](reports/native_polish/REPORT.md) and
[current installed-package verification](reports/compute_confirmation/installation_verification.json).

## Native model with explicit fact memory

The native model now supports a learned fact reader in every block, an exact
graph workspace with bounded programs and evidence traces, and an optional
PC-ALM training rule. Backpropagation remains the default. The hybrid session
invalidates recurrent state after committed fact edits. Existing checkpoints
can be augmented with zero-gated readers while preserving their original outputs.

- [Integrated architecture](reports/hybrid_integration/ARCHITECTURE.md)
- [Verification, reader adaptation, and current checkpoint](reports/hybrid_integration/REPORT.md)
- [Hybrid commands](energy_real/README.md#native-hybrid-architecture)

## Component studies: local learning and full energy models

The new architecture experiments compare backpropagation, predictive coding,
PC-ALM, full conditional energy-based predictors, and learned energy-memory
readers. A separate graph model learns scalar energy factors over explicit
entity IDs, without learned embeddings. Image, structured reasoning, and small
next-byte tasks use matched controls and three training seeds.

- [Results, trained models, and verification](reports/local_energy/REPORT.md)
- [Architecture and assessment of latent-state alternatives](reports/local_energy/DESIGN.md)
- [Runnable commands](energy_real/README.md#local-learning-and-full-energy-predictors)

PC-ALM improves deep predictive coding in the bounded image experiment. Energy
memory readers help the full EBM on graph tasks, but the matched backpropagation
control performs similarly. Discrete graph factors succeed with supplied rule
primitives; that is not evidence of learned language grounding or general AI.

## Reproducible memory retrieval and reasoning experiments

The local repair adds an explicit-backend energy kernel, correct attention resets,
full-state resumable training, source-only acceptance, controlled memory studies,
and an HTTP adapter for the existing local reasoning model.

- [Publication polish and focused follow-up](reports/publication/REPORT.md)
- [Repair findings and evidence](FINDINGS.md)
- [Commands and interfaces](energy_real/README.md)
- [Shared memory and reasoning study](reports/reasoning_v2/REPORT.md)
- [Reliability pass](reports/repair/POLISH.md)
- [Initial repair report and memory results](reports/repair/REPORT.md)

The new memory experiments favor exact structured storage; the tested energy
candidate has not earned a scale-up promotion over stronger classical controls.
Live reasoning evaluation and fresh GPU measurements have separate pending
statuses when the model service is unavailable or the device is occupied.

## Historical v23 target: a bounded DGX Spark test

The complete self-contained **v23 Spark source** is in [`spark_kit/`](spark_kit/README.md).
It includes the native energy/softmax implementations, resumable training, local
corpus preparation, device acceptance, configs, tests, and a read-only host doctor.
The earlier experimental modules and their authorization boundaries remain unchanged.

- [Start on the Spark](spark_kit/SPARK_START_HERE.md)
- [Architecture and math](spark_kit/ARCHITECTURE_SPARK.md)
- [Executed v23 changes and tests](spark_kit/reports/v23/REPORT.md)
- [Original model installation](spark_kit/checkpoints/README.md)

```bash
cd spark_kit
# The standalone download already includes the original checkpoint.
# To sample or continue original weights, install them from the local kit:
python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip
bash spark/run.sh doctor
bash spark/run.sh accept
```

The original bundled v23 report recorded 83 local CPU tests; current source-only
results and explicit artifact skips are recorded in the repair report. The new importer uses less
Python-tracked preparation memory on the measured fixture, and resume preserves
crash-tail logs. **No new model training, actual GB10/CUDA result, or serious general
pretraining approval is claimed by v23.** Run acceptance on the target device before
starting any corresponding capped pilot.

## Source and binary publication are different

This source commit does not upload the original pretrained binary or historical
experiment archives. The complete available collection is preserved in the supplied
v1-v23 download: 22 original experiment ZIPs and 96 original standalone artifacts,
plus the current kit. Their inventory/checksums are not a substitute for an upload.

From a networked machine holding that download and normal GitHub CLI authorization:

```bash
python spark_kit/publish_release.py \
  --archive /path/to/EnergyTransformer_All_Research_v1_v23.zip --publish
```

The uploader refuses a public target and conflicting assets, verifies downloaded
bytes, and publishes only after verification. It never force-pushes or changes
repository visibility. Do not paste tokens into chat or commit credentials.

## Earlier research and bounded memory interfaces

[Earlier README](README_LEGACY.md) preserves the previous entry point byte for byte.
See [architecture](ARCHITECTURE.md), [model card](MODEL_CARD.md),
[rights](RIGHTS.md), [storage](STORAGE.md), and [history](reports/HISTORY.md).
The existing private releases contain bounded parser/answer checkpoints, not the
latest native-text model or all original historical bytes. Their limitations remain.

No new open-source license is imposed. Old failed experiments and classical controls
retain their original qualifications. Energy coefficients remain numerical internal
state; terminology alone does not establish novel mathematics or superiority over
attention. The next useful evidence is target-device numerical behavior and paired
validation curves against measured training cost.
