# EnergyTransformer: reconciled evidence and repair results

Updated 2026-09-14 on the NVIDIA GB10 host. This document separates inspected
artifacts, executed tests, historical reports, and conclusions supported by them.

## What the previous conversations establish

The ChatGPT conversation **Evolved Latent Space Reasoning** reported CPU training
and an original 972,896-parameter checkpoint supplied in a separate downloadable
kit. It explicitly said the original binaries and historical archives had not
been uploaded with the final GitHub source commit. That binary is absent from
this checkout and the local download locations inspected. Its local absence does
**not** establish that it was never produced, or that no earlier training occurred.
The original downloadable archive has not been independently recovered here.

Claude subsequently implemented `energy_real`, trained local enwik8 models, and
stopped the energy run following the user's concern about reasoning quality. The
original untracked source was preserved in `runs/source_before_repair.tar.gz`;
`reports/repair/baseline_artifacts.json` records its files and all discovered
local data/run/checkpoint hashes before edits.

Two real 10.6M-parameter checkpoints are present, about 41 MB each. Parameters
remain applicable to these learned models; weight-file size, recurrent-state
size, optimizer state, and total runtime memory are different measurements.

## Historical training results verified from local logs

| Quantity | Energy | Softmax |
|---|---:|---:|
| Parameters | 10,606,896 | 10,607,436 |
| Validation bpb at step 5,000 | 1.723972 | 1.663339 |
| Training targets at step 5,000 | 40,960,000 | 40,960,000 |
| Last logged training step | 5,832 | 12,000 |
| Saved best checkpoint step | 5,000 | 12,000 |
| Saved best validation bpb | 1.723972 | 1.497544 |

Lower bpb is better. The matched 5,000-step comparison favors softmax. Comparing
the two saved checkpoints directly is an **unequal-training-budget comparison**.
The energy checkpoint does not contain optimizer/RNG state, so its interrupted
run cannot be exactly resumed. Loading it starts a separately labelled warm run.

These scores evaluate 999,999 next-byte targets from validation, resetting state
at each 512-byte window. They are not held-out test results, reasoning scores, or
a directly interchangeable protocol with every published enwik8 result. The
90M/5M/5M split alone does not establish protocol equivalence. The local corpus
SHA-256 is recorded; its acquisition provenance was not independently audited.

Claude reported an approximately 15.4x kernel speedup over the reference, and a
remaining throughput/memory disadvantage versus softmax. Those are **historical
session measurements**, not newly reproduced benchmark results. The repaired
fixed-backend kernel needs fresh measurements. Another active GPU job prevents
an uncontended remeasurement at the time of this report.

## Verified defects and repairs

- `kind=delta` instantiated the energy mixer. It now raises instead of labelling
  that model as a distinct delta implementation. Existing v18 delta memory
  controls remain available in their own experimental module.
- Softmax ignored reset flags. Its cache now carries segment identities and
  masks previous segments, including mixed reset positions within a batch and
  resets after cached prefixes. Legacy two-tensor caches mean one segment.
- A future-dependent automatic energy backend switch changed prefix outputs by
  up to 2.86e-6 in a targeted fixture. The default is now fixed `dmat`; explicit
  `matmul` rejects unsupported decay spans instead of silently switching.
- The new trainer had only non-atomic weight snapshots. It now has atomic full
  checkpoints, source/data/runtime identities, exact CPU resume, explicit warm
  starts, output ownership, log-prefix recovery, and interruption/budget status.
- Random sampling excluded the final valid byte crop. Target-coverage tests now
  include the final crop, ragged windows, and split boundaries.
- From-scratch Spark numerical acceptance depended on a missing historical
  checkpoint. It now uses a deterministic random fixture. Explicit historical
  verification remains required to authorize continuation from that checkpoint.
- Source-only installer/receipt tests now use generated fixtures. An actual
  original-checkpoint integration test remains explicitly skipped when absent.
- The root test dependencies now include safetensors. The three test suites have
  separate source-only CI commands.

The earlier audit overstated the absence of independent tests. Several parity
tests compare closely related implementations, but `accept.scan_audit` already
compared the transported scan and its gradients with a serial recurrence. The
new tests strengthen that coverage; they did not create all independence from
nothing.

Historical v17/v18 table backing artifacts are incomplete locally. A condensed
report, an absent local raw file, and a separately reported archive must not be
conflated with proof that the underlying experiment never occurred.

## Initial bounded memory evidence

`reports/repair/memory_baselines_final.json` evaluates 100 deterministic held-out
relation episodes with **oracle structured edits**, ten each for insertion,
replacement, deletion, absence, similar addresses, distractors, overload,
two-hop queries, three-hop queries, and alternate wording.

| Memory backend | Answer accuracy | Stale facts retrieved |
|---|---:|---:|
| Exact dictionary | 100% | 0 |
| Diagonal edit memory | 66% | 5 |
| Precision-aware energy edits | 80% | 0 |
| Simple gated delta | 46% | 0 |
| Preconditioned delta | 80% | 0 |
| Projected energy reparameterization | 80% | 0 |

These are the existing small geometric-memory mechanisms, **not** a new language
model accuracy measurement. Zero stale facts does not imply correct retrieval:
missing answers and wrong associations also count as failures. Alternate wording
is not an interpretation test when structured edits are supplied.

The controller study ran 30 CPU fits: five mechanisms × two objectives × three
seeds, 60 updates per fit with equal optimization budgets. One objective predicts
the next **symbolic value token before writing it**, given oracle operation and
address fields. The other adds delayed-query and edit-locality losses. This is
not ordinary-text next-token pretraining. Validation may select initialization
(step zero); the selected step is retained in each record.

Precision-aware energy did not meet the required five-point advantage over the
stronger validation-selected classical control, under either objective. The
projected and preconditioned forms behaved as equivalence controls. Raw tensor
state and serialized dictionary records exclude different costs, so neither is
used to claim a system-memory efficiency win.

## Practical reasoning status and decisions

The separate relation store supports reviewed proposals, revision-checked edits,
idempotent retries, provenance, corrections, deletions, and one-to-three-hop
queries. Approximate retrieval errors are returned unchanged; an exact-store
lookup never silently substitutes the correct answer.

The HTTP adapter supports the existing local service's health, chat, and scoring
contracts. Mock-backed integration tests verify the three arms and their supplied
evidence. The offline demo demonstrates memory operations, not LLM reasoning.
Actual service evaluation is **pending** because localhost:8100 is unavailable.
The report retains that status rather than presenting fixture answers as model
results.

- **Engineering:** retain the repairs and reproducible harness. New GPU cost and
  numerical results remain pending an idle device.
- **Energy research:** retain the bounded results; no scale-up promotion. There
  is no demonstrated advantage over the stronger classical controls here.
- **Practical system:** continue with exact evidence-backed memory as the
  baseline; evaluate the energy arm through the same adapter when the existing
  service becomes available.

Poor samples from a tiny byte LM do not establish a universal parameter threshold
for reasoning, nor prove the entire family is a dead end. Equally, correct energy
identities and synthetic recall results do not establish useful general reasoning.
The next decision should follow the missing measurements, not either extreme.

See `energy_real/README.md` for commands, `reports/repair/REPORT.md` for executed
checks and artifact links, and the stage ledger for pending resource work.


## Shared recall and reasoning study, v2.1

The next controlled study uses 384 training, 120 validation and 360 fresh test
episodes. Entity namespaces and wording templates are split-disjoint; individual
episode IDs and contents remain stable across sample sizes. Two mechanisms are
screened at widths 16/32/64 and priors 0.01/0.001 with score/margin calibration.
Twelve seed-11 fits select configurations; seeds 22/33 replicate those selections.
Every fit receives 500 updates of delayed recall plus 0.1 times edit-locality loss.

On the same fresh test set, the original precision-energy configuration scores
83.61%. The validation-selected configuration/checkpoint averages 98.06% across
three seeds, versus 97.69% for its separately calibrated initialization. The
preconditioned classical control averages 99.26%; exact storage scores 100%.
The energy-minus-classical paired interval is -2.31 to -0.28 percentage points,
with a mean difference of -1.20 points. These results do not meet the promotion
gate. Most of the energy improvement comes from configuration/calibration; only
one seed has a positive learning interval excluding zero. Five unsupported
answers remain across the three 360-episode energy evaluations.

Natural-language query interpretation and a separate gold-evidence diagnostic
are implemented and mock-tested. They are distinct from supplied-path memory
tests. Live model reasoning and GPU measurements remain pending the unavailable
service and busy device. No new large language-model training was started.

The audited study completed 16 fits/8,000 updates in 3.95 CPU minutes. Including
the preserved initial diagnostic run, its stage ledger charges 7.94 CPU minutes
and zero actual GPU time. The first run was repeated after fixing sample-size-
dependent episode IDs and inherited process-memory high-water marks; its outputs
remain preserved and are not used for the final comparison.

See [the full v2.1 report](reports/reasoning_v2/REPORT.md), including validation
selection, per-seed results, full-process memory costs, test logs and artifact
hashes. The current verification total is 277 passed and 34 explicit skips.


## Publication follow-up: controller freezing and reliable evaluation

The fresh 12-fit, three-seed comparison completed 6,000 updates in 3.95 CPU minutes.
Validation selected controller-only training for both mechanisms. Precision energy
improved from 96.39% at initialization to 98.61% on the new held-out episodes; the
paired gain interval was +1.11 to +3.61 percentage points, with positive point gains
in all three seeds. The selected classical control reached 99.07%. Neither the
quality nor efficiency promotion gate passed, and freezing geometry did not reliably
beat all-parameter training. These results do not replace the historical v2 scores.

Portable artifact bundles, shared retrieval ranking and resumable model evaluation
are implemented. Verification totals 295 passing tests, 32 skips and one deselected
original-artifact integration test; isolated source installation and CPU CLI checks
also passed. GPU measurements and live reasoning remain pending, with no GPU work
started over the existing workloads. See [the complete report](reports/publication/REPORT.md).


## Local learning, energy modules, and discrete-state study

The new [72-model study](reports/local_energy/REPORT.md) trained local BP/PC/PC-ALM predictors, full conditional energy models, energy-memory modules, and discrete graph factors. At 32 layers, PC-ALM recovered 10.47 accuracy points over PC and remained close to BP. Memory added 5.31 points to the graph EBM, without beating BP with memory. On fresh byte data, EBM loss was 4.682 bits/byte versus BP-CE 4.472 and a fixed unigram reference 4.917. Explicit discrete graph models reached 100% with supplied semantics and path execution; this is not learned general reasoning. Verification: 320 passed, 33 skipped; CPU only. The [design assessment](reports/local_energy/DESIGN.md) explains why typed graphs, programs, and addressable memory are stronger primary candidates than topology alone. Earlier reports remain unchanged.

## Integration into the native architecture

The [native hybrid](reports/hybrid_integration/REPORT.md) adds learned fact readers in all six blocks, an exact graph-program interface, and optional native PC-ALM training. The existing 10,606,896 weights stayed unchanged while 147,654 reader parameters were adapted. Across three seeds, mean accuracy was 95.94% on the supplied-symbol fact task; the validation-selected checkpoint reached 99.33%. Exact graph programs answered all 360 fresh diagnostics correctly. Reader transfer is limited: it failed known-answer queries with only one stored fact, outside its three/four-fact training setup, while the exact executor remained correct. Existing checkpoints retain compatibility, native state resets on committed revisions, and verification passed 334 tests with 33 skips. The training deadline interrupted final evaluation; evaluation completed separately from frozen checkpoints without changing model selection. CPU only, with all earlier report bytes preserved.

## Mixed energy and local-attention language model

The [new native comparison](reports/competitive/REPORT.md) trains four parameter-
matched architectures at three seeds from scratch, each on 1.024 million bytes.
The approximately 145,000-parameter mixed model alternates two energy blocks
with one 64-token local-attention block. On held-out 128-byte windows, mean bits
per byte are 2.9979 mixed, 3.0369 energy-only, 3.1119 windowed Transformer, and
3.1394 full Transformer. At 512-byte context they are 2.9469, 2.9913, 3.0596, and
3.3681 respectively. All three mixed seeds outperform the full and windowed
controls. All checkpoints were selected on validation before test evaluation.

This is a data- and parameter-matched improvement, not a compute-matched win:
mixed CPU training took 3.09 times as long as full attention. The report gives
conditional paired-window intervals and measured cache/inference costs. It does
not establish general language or reasoning superiority, GPU efficiency, or a
better large pretrained checkpoint. Greedy output from the tiny model remains
repetitive. The original three large checkpoints retain bitwise outputs and
states; verification passed 348 tests with 34 optional skips, and a separate
non-editable installation matches all 50 package source files.

At a 32,768-byte prompt plus 16 decoded bytes, mixed carries 37,512 bytes of
state versus 51,143,040 for full attention; CPU median decode latency is 3.505
versus 20.620 ms per token. Full attention is faster at short prompts, and the
windowed Transformer is faster at every measured length. These measurements
use the native concatenating-cache implementation, not optimized static/paged
caches or GPU serving. The 32K profile measures inference cost, not text quality.

## Stronger compute and inference controls

The [kernel/static-cache follow-up](reports/stronger_comparison/REPORT.md)
preserves the recurrence and checkpoint format while reducing measured CPU
hybrid update time by 1.75× at the same chunk size, or 2.05× with chunk size eight.
Against a preallocated full Transformer cache, 32K-prefix decoding is 0.805 vs
2.534 ms per byte, with 37,512 vs 51,143,040 bytes of allocated state. The full
Transformer is faster at 4K, and windowed attention remains faster throughout.
The first equal-CPU-budget language experiment was effectively tied at 128 bytes.

The [confirmation](reports/compute_confirmation/REPORT.md) pins all models to
one performance core, rotates half-second CPU blocks, compiles every training
objective and uses foreach AdamW throughout. Three new seeds and equal
validation-only tuning budgets yield 2.5121 test bits per byte for the hybrid,
2.5782 for full attention and 2.5770 for windowed attention at 128 bytes. All
three hybrid seeds beat both controls; the conditional paired-window intervals
are below zero. The 512-byte test also favors the hybrid, but extrapolates
beyond the training context and does not test distant-fact recall.

This is a steady-state CPU update-budget result, excluding compiler setup,
validation and I/O. It does not establish an end-to-end training-cost win,
large-model superiority or GPU performance. The selected seed-606 checkpoint
is `runs/compute_confirmation_selected/hybrid.pt`; earlier artifacts remain
preserved. Current verification: 385 passing tests, 33 skips, compiled/eager
gradient parity on real crop layouts and trained weights, and an isolated
non-editable installation matching all 53 package source files.
