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
