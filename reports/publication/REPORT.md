# EnergyTransformer follow-up results

Energy memory shows a replicated learning gain on this fresh structured benchmark,
but does not meet the quality or efficiency promotion gates. Portability and
resumable-evaluation improvements are verified and ready to publish.

Study: **completed**. Verification: **passed**.
Fits: 12; optimizer updates: 6000; measured study time: 236.81 seconds.

| Mechanism / training | Initial accuracy | Selected accuracy | Unsupported answers |
|---|---:|---:|---:|
| precision/all | 96.39% | 98.89% | 9 |
| precision/controller_only | 96.39% | 98.61% | 8 |
| preconditioned_delta/all | 98.89% | 98.61% | 11 |
| preconditioned_delta/controller_only | 98.89% | 99.07% | 8 |

These are structured-memory results. Live reasoning and device measurements have separate stage statuses.

## Stage status

- smoke: pending_gpu; charged time 0.00 seconds.
- evaluation_profile: pending_gpu; charged time 0.00 seconds.
- reasoning: pending_service; charged time 0.00 seconds.
- controllers: completed; charged time 716.24 seconds.

## Test evidence

- root: 211 passed, 32 skipped, 0 failed, 0 errors.
- spark: 84 passed, 0 skipped, 0 failed, 0 errors.

[Machine-readable results, paired intervals, costs, and decisions](results.json).

## Interpretation and controls

Seed-11 validation selected **controller-only** training for both mechanisms.
Energy accuracy increased from **96.39% to 98.61%** against separately calibrated
initialization. The paired gain was **2.22 percentage points**, with a 95% episode
bootstrap interval of **[1.11, 3.61] points**. All three tested seeds improved.
All-parameter energy training reached 98.89%, but it was not the validation-selected
mode; test results did not change the frozen choice.

The selected classical control reached **99.07%**. Energy minus classical was
**-0.46 points**, with interval **[-1.11, 0.09]**. Both selected mechanisms produced
8 unsupported answers across 1,080 evaluations. Exact storage scored 100%.
Controller-only versus all-parameter intervals include zero for both mechanisms:
freezing geometry and gain has not established a reliable advantage.

These fresh episodes use seeds 271101/271102/271103 and 384/120/360 train/validation/
test cases. Their absolute scores must not be treated as a direct improvement over
the earlier v2 test set. Training still uses delayed-recall cross-entropy plus 0.1
locality loss, with the unchanged optimizer, confidence grid and checkpoint schedule.
The 12 fits completed all 6,000 updates. [Fit configurations and selections](fits.json)
retain selected steps and byte-exact frozen-parameter checks.

Selected CPU median query times were **1.250 ms energy** and **1.005 ms classical**.
Both used 80 KiB recurrent state; complete component accounting was also equal.
Energy was approximately 24.4% slower on this sample. Per-component disk/resident
accounting and isolated process peaks remain separate in the results. These are
CPU measurements, not GPU efficiency evidence.

**Decision:** retain both training modes as experimental results, keep exact memory
as the reliable practical baseline, and do not begin larger pretraining or additional
model fine-tuning. Continue live reasoning evaluation when the existing service is
available. The actual device check found unrelated GPU workloads; the service health
request was refused. GPU time consumed by this phase is zero. The cumulative controller
ledger charges 716.24 CPU seconds, including 476.31 seconds inherited from earlier work.

## Portability and reliability

- Portable bundles resolve relative checkpoint paths, verify hashes, and support an
  explicit artifact root. Legacy absolute-path manifests remain usable and unchanged.
- Study and practical retrieval share ranking and confidence logic, with stable
  event-order tie breaking.
- Resumable evaluation durably caches successful proposals, query interpretations,
  answers and scoring by episode/arm. Retries preserve completed rows and accumulate
  elapsed time. Changed source, dataset, checkpoint, declared model identity, decoding,
  scoring mode or total budget is rejected.
- Stage retries retain completed comparisons. Fixture outages verify that successful
  work is reused without duplicate rows or reset budgets.
- Fresh non-editable wheel installation passed with CPU-only PyTorch. Isolated imports
  from outside the checkout, the offline demo, CPU smoke test, and a tiny 12-fit CLI
  study passed without historical data/checkpoints. All 52 installed package source
  files match the tested workspace. [Installation evidence](installation.json).

The local suites total **295 passed and 32 skipped**; one original-artifact integration
test was explicitly deselected in the Spark source-only run. Numerical checks and
fixtures do not constitute live language-model reasoning scores.

## Reproduce and relocate

The [portable selection](selection.json), [dataset recipe](datasets.json) and
[bundle manifest](bundle.json) describe the selected artifacts. Binaries remain in
ignored `runs/publication_followup_bundle`; source publication does not upload them.
Move that whole bundle, or point `--artifact-root` at its new directory. A source-only
clone can regenerate a fresh study using the documented command; exact original
checkpoint reuse requires the preserved, checksum-verified bundle.

```bash
.venv/bin/python -m energy_real.reasoning evaluate \
  --benchmark v2 --question-mode natural --gold-evidence --score \
  --memory-selection reports/publication/selection.json \
  --artifact-root runs/publication_followup_bundle \
  --out runs/publication_live_reasoning.json
# Retry the same command with --resume after a service interruption.

.venv/bin/python -m energy_real.reporting \
  --study runs/publication_final_stages/controllers/attempt_0/study \
  --ledger runs/publication_final_stages/stages.json \
  --test-log root=reports/publication/tests_root.log \
  --test-log spark=reports/publication/tests_spark.log \
  --out runs/publication_report_rebuilt
```

The report command regenerates the evidence tables and machine-readable results;
the interpretation and release notes above explain this completed run. Historical
repair and v2 reports are unchanged. Full datasets, individual predictions, controller
files and the final source archive are identified by [artifact hashes](artifacts.json).
