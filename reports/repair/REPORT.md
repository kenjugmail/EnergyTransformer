# Repair implementation and executed results

The implementation covers the engineering repairs, bounded CPU memory research,
and practical local-model adapter. **Device measurements and live model quality
remain pending external availability.** No large pretraining run was started.

## Verification

| Suite | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Repaired kernel/trainer/adapter/stage suites | 69 | 0 | 0 |
| Earlier research suite | 94 | 32 | 0 |
| Spark suite | 83 | 2 | 0 |
| **Total** | **246** | **34** | **0** |

The earlier suite retains its optional artifact skips. Spark skips concern an
unavailable original artifact and a CPU-host-specific test on this CUDA host.
The tests do not run GPU workloads merely because CUDA is present.

Logs: [new suite](tests_real.log), [legacy suite](tests_legacy.log),
[Spark suite](tests_spark.log). The [CPU numerical smoke report](cpu_smoke_final.json)
also passed. Both editable packages installed, and all three packages imported
successfully from outside the repository. `git diff --check` passed.

Meaningful new checks include future-induced backend switching, reset/cache
segmentation, decay derivatives and both reads, independent finite differences,
fresh-process optimizer/RNG resume, SIGTERM, atomic failure, uncommitted best-file
recovery, crash-tail logs, exact byte coverage, frozen test selection, source-only
acceptance, retry/revision semantics, provenance, multi-hop answers, local HTTP
contracts, and stage time ceilings.

## Memory results

The [100-episode memory comparison](memory_baselines_final.json) uses oracle
structured edits. Exact memory scores 100%; diagonal 66%; precision-aware 80%;
simple delta 46%; preconditioned delta and its projected equivalent both 80%.
The diagonal arm retrieves five stale facts. All errors remain visible.

The [controller study](controller_study.json) completed 30 CPU fits: five
mechanisms, two objectives, and three seeds (11/22/33), 60 updates each. The
next-value-token objective predicts the value before it is written, conditional
on supplied operation/address fields. It is a small symbolic-memory study,
**not natural-text pretraining**. The second objective adds delayed-query and
edit-locality losses. Validation can retain initialization; selected steps and
all final per-episode test accuracies are recorded.

| Mechanism | Next value token only | + delayed recall/locality |
|---|---:|---:|
| diagonal | 53.49% | 53.67% |
| precision | 92.01% | 92.01% |
| delta | 21.22% | 45.05% |
| preconditioned_delta | 91.54% | 92.52% |
| projected | 91.54% | 92.52% |

These accuracies average episode-level query accuracy across ordinary, similar-
address and overloaded test families. The classical comparator is chosen by
validation. Bootstrap intervals retain matched episodes and report per-seed
differences separately. Neither objective meets the five-point energy promotion
criterion. The equivalence control behaves as expected. No efficiency promotion
is claimed from incomparable raw-state and provenance accounting.

The study is intentionally bounded and has no hyperparameter search. Its result
supports retaining these mechanisms as research controls, not scaling the byte LM.
Checkpoints and complete run records remain under `runs/repair_memory_study_final`.

## Practical adapter

The [offline demo](offline_demo.json) executes reviewed facts, a correction,
a two-hop query and a deletion against exact and energy stores. It is explicitly
labelled as an offline memory demonstration, not an LLM reasoning benchmark.

The HTTP fixture tests exercise `/health`, `/v1/chat/completions`, and `/v1/score`.
They verify three separate evidence arms and ensure that the energy arm cannot
receive hidden corrections from the exact store. Models propose edits; only the
caller can authorize the revision-checked application.

[Live evaluation](live_reasoning.json) is **pending_service**: localhost:8100
refused the connection. Generated answers and multiple-choice scores will remain
separate. Missing service token usage is reported as unavailable, not estimated
and labelled as an actual token count.

## Pending resource work

The [stage ledger](device_stages.json) records zero GPU time spent by this repair:
GPU smoke checks and checkpoint evaluation/profiling are **pending_gpu** because
other workloads are active. The reasoning stage is **pending_service**. Existing
workloads were not stopped or modified.

Run these commands from the repository root once the dependencies are available:

```bash
.venv/bin/python -m energy_real.stages --out runs/repair_device_stages \
  --stages smoke evaluation_profile reasoning
```

Optional bounded wait: add `--wait-idle --max-wait-seconds 300`. This command does
not install or start the other project's model server. The controller work was
already completed on CPU; there is no reason to spend its four-hour GPU ceiling
repeating it merely to consume the allowance.

Limits remain 30 minutes for smoke, 90 minutes for checkpoint evaluation/profiling,
four hours for controllers, and two hours for reasoning. The ledger preserves
used time; completed, failed, and exhausted stages are not silently rerun. A live
service may own its GPU; unrelated training jobs still block inference stages.

## Preserved evidence and outcome

[Baseline hashes](baseline_artifacts.json) identify the original untracked source,
data, logs and both approximately 41 MB model checkpoints. The original untracked
source is archived locally. Original checkpoints and data remain byte-identical.
Large data, runs, and checkpoint directories are excluded from source commits.

- **Engineering:** repairs implemented and CPU-verified; actual device acceptance
  and new cost measurements remain pending.
- **Energy research:** retain as a bounded result; no scale-up promotion.
- **Practical reasoning:** use exact evidence-backed storage as the baseline and
  compare live model answers when the existing service becomes available.

See [commands and interface details](../../energy_real/README.md) and
[reconciled findings](../../FINDINGS.md). The work remains local and is not pushed.
