# Energy memory and reasoning: executed v2.1 results

The shared memory benchmark, recall objective, confidence calibration and natural-
language reasoning diagnostics are implemented. The CPU study completed with
**16 fits and 8,000 optimizer updates** in
**3.95 minutes**. Live model reasoning and GPU measurements
remain separate pending work.

## What improved

All figures below use the same 360 fresh test episodes, with 30 episodes in each
of 12 families. The original configuration uses seed 11, width 16, prior 0.01, score
threshold 0.5 and no confidence-margin requirement. Other columns average seeds
11/22/33. Width 64 and prior 0.001 were selected for both mechanisms on validation.

| Mechanism | Original configuration | Chosen configuration before training | Validation-selected checkpoint |
|---|---:|---:|---:|
| Exact store | 100.00% | 100.00% | 100.00% |
| precision | 83.61% | 97.69% | 98.06% |
| preconditioned_delta | 83.33% | 99.26% | 99.26% |

Configuration/calibration gains must not be attributed entirely to learning.
The initialization column uses the selected architecture and separately calibrated
initial weights. A validation-selected checkpoint can be update zero:

| Mechanism | Seed | Selected update | Initial test accuracy | Selected test accuracy |
|---|---:|---:|---:|---:|
| precision | 11 | 100 | 97.78% | 98.33% |
| preconditioned_delta | 11 | 0 | 99.17% | 99.17% |
| precision | 22 | 100 | 98.06% | 96.94% |
| preconditioned_delta | 22 | 0 | 99.44% | 99.44% |
| precision | 33 | 100 | 97.22% | 98.89% |
| preconditioned_delta | 33 | 0 | 99.17% | 99.17% |

The energy-minus-classical difference is **-1.20 percentage
points**, with a paired 95% episode-bootstrap interval of
**[-2.31, -0.28]**. It does not meet the five-point
promotion gate. Per-seed learning intervals and family scores are in
[the machine-readable results](results.json); a single improving seed does not
establish a replicated training benefit.

Energy produced **5 unsupported answers** across
the three 360-episode evaluations. These remain visible. The comparison includes
deletions, absent facts and missing intermediate links, so aggregate accuracy
alone should not determine whether approximate memory is used for an answer.

## Experimental controls and costs

Training uses 384 structured episodes; validation uses 120. Split namespaces and
wording templates are disjoint. Twelve seed-11 configurations cover two mechanisms,
three widths and two priors. Validation chooses one configuration per mechanism;
seeds 22/33 replicate those choices. Each fit receives 500 updates of delayed-recall
cross-entropy plus 0.1 times edit-locality loss. Unseen/deleted addresses train an
explicit null target. No future random-value prediction objective is used.

Score and top-two-margin thresholds are calibrated on validation, including
unavailable facts. The [selection](selection.json) was frozen before test generation
and evaluation. [Dataset recipes and hashes](datasets.json) accompany it; complete
episodes and per-case traces remain in the local run directory. Query traces
distinguish wrong top candidates from correct candidates rejected by confidence.

| Mechanism | Median query (ms) | Recurrent state (KiB) | Process peak RSS (MiB) |
|---|---:|---:|---:|
| exact | 0.130 | 0.0 | 501.7 |
| precision | 1.138 | 80.0 | 525.2 |
| preconditioned_delta | 0.629 | 80.0 | 524.8 |

The cost sample uses 24 fixed test episodes and balanced arm order. Reports include
database bytes, resident evidence/registry serialization, parameters, recurrent
state, writes, state construction and repeated queries. Full-process resident
peaks use isolated Linux processes and `/proc/self/status` VmHWM, including runtime
and native allocations. Serialized components are accounting measures, not peak
RAM, and are kept separate from the measured process peak.

Energy’s state reduction versus the classical control is
0.00%; its measured CPU query-latency reduction is
-80.88% (negative means slower).
No GPU efficiency promotion is claimed.

## Reasoning adapter and verification

The adapter compares recent context, exact retrieval and energy retrieval with
identical questions/decoding. Natural questions require a model-produced relation
path; supplied paths remain a separately labelled test. Wrong paths are recorded
without oracle substitution. A fourth, explicitly diagnostic gold-evidence arm
isolates answer composition from retrieval and interpretation failures.

With model-proposed edits, recent context contains original statements rather
than oracle-parsed triples. Persistent writes still require caller authorization
and revision checks. Per-answer reports retain evidence, provenance, confidence
traces, model/configuration identity, interpretation errors, abstention and separate
generation/scoring results. A shared deadline bounds all service calls; a scoring
outage preserves already generated answers.

**277 tests passed, 34 skipped, zero failures.** Logs:
[repaired suite](tests_real.log), [legacy suite](tests_legacy.log),
[Spark suite](tests_spark.log). The [CPU numerical smoke test](cpu_smoke.json) passed.
The [selected-checkpoint smoke check](selected_memory_smoke.json) also matches the
frozen evaluation through the practical FactStore interface for all 12 families.
Mock tests verify natural-path interpretation, strict arm separation, missing
links, source-only recent context, malformed inputs, confidence rejection,
checkpoint/config compatibility, frozen selection, outage recovery and deadlines.
These fixture results are not live language-model reasoning scores.

## Resources, preserved evidence and decisions

The [stage ledger](device_stages.json) carries earlier usage into this revision.
Controller-stage time totals **7.94 CPU minutes**, including the preserved
initial diagnostic run. Actual GPU time for this phase is zero. GPU stages remain
pending while unrelated workloads are active; localhost:8100 remains unavailable.

Before finalizing, an audit found that the initial generator could reuse IDs for
different content when sample size changed, and the first RSS counter could inherit
a pre-exec high-water mark. Version 2.1 fixes both and repeats the unchanged settings.
The original run and its source snapshot remain preserved; its scores are not used
for the final comparison above. All earlier repair reports, data and checkpoints
remain unchanged. [Artifact hashes](artifacts.json) identify this revision.

- **Memory quality:** retain the improved configuration as an experimental candidate.
  It has not demonstrated the required advantage over the stronger classical control.
- **Learning:** retain measured seed-specific results; no blanket claim that training
  improves every selected mechanism or seed.
- **Practical reasoning:** continue with exact memory as the reliable baseline and
  compare the selected energy memory against the existing live service when available.
- **Scaling:** do not start larger pretraining or additional model fine-tuning from
  these results.

Resume the pending stages without restarting completed controller work:

```bash
.venv/bin/python -m energy_real.stages --out runs/reasoning_v2_final_stages \
  --stages smoke evaluation_profile reasoning \
  --memory-selection reports/reasoning_v2/selection.json
```

See [interfaces and reproducible commands](../../energy_real/README.md).
Rebuild this compact report with `python3 reports/reasoning_v2/build_report.py`.
