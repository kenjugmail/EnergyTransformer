# Scaling readiness audit

## Decision

Proceed to the bounded single-GB10 acceptance/profile pilot when the device is
exclusive. Do not enlarge the training allocation or claim an architecture win
from this audit. GPU evidence remains absent. The occupied device is an operational
blocker; CPU preparation can still advance.

The experiment should determine whether quality per byte survives conversion to
quality per elapsed time. The original 10.6M GPU result favors full attention;
the later tiny CPU hybrid result motivates investigating the hybrid, not promoting it.

## Defects found and corrected

| Finding | Consequence before correction | Current behavior |
|---|---|---|
| Profile stepped AdamW every 512 targets; trainer accumulates 8,192 | Timing model described a different update | Profile accumulates 16 matching microbatches, clips once, steps once; records batch, context, accumulation and effective targets |
| Only tiny FP32 compiled fixtures were checked | A production BF16 shape could compile differently without numerical comparison | Each profile compares eager/selected execution losses and gradients on the actual configuration, stride and precision before optimizer updates |
| Supervisor used a CUDA API to identify the GPU | Could initialize a competing parent CUDA context before worker admission | Supervisor queries GPU name, UUID and driver through nvidia-smi without initializing CUDA |
| Acceptance did not bind driver and device UUID | A changed runtime could reuse an old GPU receipt | Acceptance and profile receipts bind name, UUID and driver; launch validation rejects changes |
| Host reserve checked only at startup outside native training | Forecast/conversion stages could overrun available unified memory | Supervisor checks at five-second intervals, allows only its process group, and stops only that group on memory pressure or foreign contention |
| Preparation had manifests but no complete launch audit | Changed donor files or inconsistent document metadata could survive until a later error | Read-only preflight checks pinned identities, shard/document hashes, contiguous offsets, split-group isolation, byte cap, donor files/configuration and source/CPU receipt |
| Final native evaluation omitted matched-time checkpoints | Equal-time quality conclusions would lack corresponding held-out rows | Both matched-data and matched-time checkpoints are evaluated with comparison and original fit status retained |
| A reduced grid could omit 3.33M teachers | Joint track would fail after backbone spend | Schedule rejects that allocation before matched-data training |
| Forecast target cache depended on teacher/tokens but not normalization artifact | Changed normalization could reuse stale frozen targets | Extraction hash binds target-cache namespace and joint resume identity; source/corpus mismatches are rejected |
| Scaling fitter accepted mixed families or repeated trajectories | Spurious apparent scaling evidence | One architecture per fit, unique model/data/seed endpoints, and no repeated declared trajectory IDs |

Resource monitoring is periodic, not an OS-wide exclusive GPU lock. There remains
an interval between checks. The Torch allocation cap remains 48 GiB and the host
reserve remains 16 GiB. A resource interruption is retained and charged, then
requires audit; it is not an automatic retry or a budget refund.

Production parity keeps the existing compilation limits: loss difference <0.005
and relative gradient error <0.02. These are acceptance thresholds, not measured
GPU errors. No tolerance was relaxed to obtain a pass.

## Reproducible preparation checks

From this checkout, using its existing environment:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m energy_campaign readiness --root runs/campaign
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m energy_campaign design
.venv/bin/python -m energy_campaign doctor
.venv/bin/python -m energy_campaign run --root runs/campaign
```

`readiness` is a CPU provenance audit, not GPU authorization. `design` only emits a
proposal. Existing frozen source receipts are archived before rebaselining this
unstarted campaign. A charged ledger may not be reset to obtain a new allocation.

The final installed-wheel test evidence is in `cpu_verification.json`. The package
is verified outside the checkout; the independent Spark suite remains a separate
source reference. Raw failed and successful checks remain under `runs/campaign`.

## What the GPU pilot still must prove

1. Current recurrence, both reads, slot transport, incoming state, resets and
   chunk/stream execution satisfy the independent numerical checks.
2. FP32 recurrent state is preserved with BF16 projections, including gradients.
3. Eager/compiled execution agrees at each selected real shape, and exact restart
   fixtures reproduce optimizer, RNG and sample state.
4. Native candidates finish replicated matched-data fits and equal-time fits.
   Failed or incomplete runs remain visible. Throughput cannot stand in for loss.
5. Recall learning curves distinguish lost memory capacity from ordinary LM loss.
6. Joint forecasting improves LM loss against both continuation and residual
   controls at acceptable cost. Forecast NLL alone is insufficient.
7. Conversion recovers language quality while demonstrating a measured serving
   benefit. Remaining attention caches are context-growing.

Profiling includes optimizer accumulation, cold numerical/compile checks and
steady updates. Its timing model still uses an overhead allowance for later I/O
and evaluation; it is not a guarantee that every fit completes. The shared budget
supervisor and retained failure records remain authoritative. A timeout is not a
completed scaling observation.

## Follow-on scaling design: 96 independent fits

`followon_design.json` enumerates every proposed row:

- Full attention plus one energy family selected on validation evidence.
- Four model sizes: approximately 1.05M, 3.33M, 10.61M and 27.64M parameters.
- Four independent data endpoints: D0/4, D0/2, D0 and 2D0.
- Seeds 404, 505 and 606 at every cell.
- An endpoint-specific training schedule for each fit; checkpoints from one
  trajectory do not substitute for separate runs.

D0, the selected energy family and GPU cost are deliberately unresolved until
actual pilot measurements exist. The manifest has `execution_authorized: false`.
After supplying D0 and the selected family, `design --d0 N --selected-family hybrid`
(or `energy`) resolves the rows without executing them.

Fit the separable model/data loss law only after these measurements support
identifiable parameters. Keep architecture fits separate. Check predictions by
holding out sizes and budgets, bootstrap paired seed/document measurements, and
validate at a newly measured larger size before extending extrapolation. The
current fitter's seed intervals alone do not supply document uncertainty.

## Slot capacity design: 27 proposed fits

A separate controlled capacity study holds width 208, six layers and four heads:

- Energy and hybrid at 8, 16, 32 and 64 slots/head, each with three seeds: 24 fits.
- Three full-attention control fits: 27 fits total.
- Feed-forward width is adjusted so total parameters remain within 0.5%.
- The manifest records actual parameters, FP32 recurrent-state bytes and BF16
  local-KV bytes per sequence for every configuration.

This tests allocation of parameters to memory rather than FF capacity. Increasing
slots while shrinking the FF is a tradeoff, not a pure slot intervention. A later
fixed-FF control sweep would separate those effects but is not included or funded
here. Use matched data, equal tuning opportunity, LM loss, controlled recall,
training memory, and measured update/decode cost together.

Do not change the pilot's fixed 16-slot architecture after inspecting test results.

## Optimization and larger training gates

The first kernel target remains the decay intermediate with two within-chunk
position axes. A fused implementation should tile the causal contractions, retain
FP32 recurrence, preserve both reads, store boundary states and recompute internal
intermediates in backward. It must independently pass forward/state/gradient
parity before entering comparisons. No fused kernel was added in this audit.

The serial differentiable summary trace is another potential cost center. Measure
it before implementing a fast trace. The current three joint arms expose that
cost; an attractive forecasting metric cannot excuse a language-loss regression.

Before multi-GPU work, separately implement and validate gradient synchronization,
valid-target-weighted reduction, deterministic rank sharding, rank-aware RNG and
sampler checkpoints, collective failure handling and distributed resume. Compare
one update against a single-device accumulated reference before a quality study.
Those capabilities are not implemented by the single-device campaign runner.

## Sentinel use

Sentinel supported investigating controls/resource gates before speculative kernel
or architecture changes. After the fixes, one ranking strongly preferred GPU work
despite an occupied device. That recommendation was rejected: device evidence and
the user's no-interruption instruction govern execution. Rankings are recorded as
triage assistance, never correctness evidence or permission to exceed budgets.
