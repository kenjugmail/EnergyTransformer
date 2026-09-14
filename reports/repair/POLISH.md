# Reliability polish — 2026-09-14

This pass hardens experiment completion, frozen comparisons, and partial local-model
results. The earlier memory study remains a bounded historical result; no new
accuracy improvement or device performance result is claimed.

## Changes

- Stage completion now requires a readable JSON object with a status. Missing or
  malformed reports cannot turn a successful process exit into a completed stage.
  Pre-existing command output is rejected before launch.
- SIGINT/SIGTERM terminates and reaps the runner’s own child process, records the
  interruption and elapsed time, and restores the caller’s signal handlers.
  Launch failures also leave a charged, failed ledger. Interrupted and incomplete
  stages require review rather than silently receiving another run.
- Ledger reads and source checks happen after taking the ownership lock. Pending
  stages remain retryable, and the configurable service URL reaches both the
  health check and reasoning subprocesses.
- Frozen evaluation includes byte limits, batch size, split boundaries, source,
  device type, Torch version and precision settings. Test evaluation verifies the
  original validation report checksum. Reports distinguish planned targets from
  consumed targets; changed settings require a new validation selection.
- Reasoning saves each generated answer before multiple-choice scoring. A scoring
  outage preserves that answer, marks scoring pending, and keeps usable summaries.
  Reports include decoding settings, controller hash, response model, seed and
  memory configuration. Arm order rotates across episodes. Generation and scoring
  counts, accuracy and latency remain separate.
- Invalid limits fail before service calls or experiment work. Documentation
  explains overlapping input boundaries and exact target coverage.

## Verification

| Suite | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Root research and repaired implementation | 177 | 32 | 0 |
| Spark | 83 | 2 | 0 |
| **Total** | **260** | **34** | **0** |

The pass adds 14 regression cases, including an actual subprocess SIGTERM test,
missing and malformed reports, stale output, launch failure, URL forwarding,
changed evaluation limits, validation-report tampering, invalid input limits,
and a scoring outage after successful generation. Existing resume, kernel,
provenance and memory tests also pass.

Logs: [root](polish_tests_root.log), [Spark](polish_tests_spark.log).
The [CPU numerical smoke test](polish_cpu_smoke.json) passed for both mixers.
`git diff --check` passed. The 34 skips retain the prior artifact and host-specific
conditions. No skipped test is counted as a pass.

## Resource status and next run

The [current stage ledger](polish_device_stages.json) records zero device-stage
time used. GPU smoke and checkpoint evaluation/profiling remain `pending_gpu`
while the existing llm-stack job runs. Live reasoning remains `pending_service`:
localhost port 8100 refused the connection.

The earlier zero-use ledger remains preserved. Source changes require a new
ledger, so use the current one for this revision:

```bash
.venv/bin/python -m energy_real.stages --out runs/polish_device_stages \
  --stages smoke evaluation_profile reasoning
```

For a different existing service, add `--base-url http://host:port`. No service is
started by this command. The original stage ceilings still apply.

## Evidence and decisions

Original data and historical run artifacts were rechecked against the baseline
hash manifest and remain unchanged. [This pass’s manifest](polish_artifacts.json)
records current source, tests, documentation and reports, plus a local source
archive. [The initial report](REPORT.md) and its source hashes remain historical
snapshots; its stage-resume command predates this revision.

Engineering: continue with the pending device checks. Energy memory: retain the
existing bounded result; the scale-up criterion remains unmet. Practical reasoning:
continue the live exact-memory versus energy-memory comparison when the existing
service becomes available. No additional training or cloud work was started.
