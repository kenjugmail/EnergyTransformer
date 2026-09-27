# Decoder training integrity and configurable evaluation

September 27, 2026. Baseline: 64c1f8c8b27252092bf7193612e31b6bb44a27df.
This revision changes experiment control, not the teacher, energy equations,
adapter architecture, data split, optimization objective, or deployed weights.

## Defects reproduced before editing

1. After a completed one-step study, changing the cached data and rerunning with
   --resume returned success. FROZEN.json adopted the new data hash while retaining
   exactly the old selected-checkpoint hashes. The shortcut checked only steps.
2. --stop-after 0 returned success and printed COMPLETED, but wrote no last.pt.
3. Resuming a step-three checkpoint with --stop-after 2 rewrote the training record
   to claim two completed steps while last.pt still contained step three.
4. A study trained with seed 7 could not be evaluated: the evaluator attempted to
   open hidden_seed11/best.pt instead of the actual selected seed.

These were tiny generated-feature fixtures, not new language experiments. The
edited original train/evaluation source blobs were verified against the live
baseline before making the patch.

## Repairs

- New immutable STUDY.json binds every requested run to data, supplied head,
  teacher, model config, seed, schedule, source hashes and Torch version.
- Resume checks every existing arm BEFORE any arm continues. Completed runs must
  also match their saved selection, counters and best/last artifact hashes.
- Repeating an unchanged completed study is byte-preserving for its checkpoints,
  reports and selection. Corrupted completed artifacts fail rather than being
  relabelled under a new dataset or silently accepted.
- Step-zero full checkpoints are written; backwards stops are rejected before
  output mutation. Actual saved progress determines partial/completed status.
- Partial CLI training exits 75, completion exits 0. No FROZEN.json for partial fits.
- Evaluation uses the explicit Cartesian kind/seed selection. Custom seeds work;
  malformed paths and missing combinations are rejected. Legacy complete manifests
  without an explicit seeds field can infer it from their validated filenames.
- Forecast-head normalization, horizon, history, and variable metadata must match
  cached features, not merely share a teacher hash.
- Unique atomic JSON replacement protects individual metadata files against tested
  replace failures. Nonfinite validation values are rejected, not written as NaN.
- Local advisory study/run locks prevent cooperating processes from concurrently
  mutating the same output. Process-exit release and lock-file reuse are tested.

No GPU program is launched by these changes. The cached-feature trainer remains
CPU-only. Existing checkpoint inference formats are unchanged. New-source resumes
of historical training folders are intentionally rejected; preserve old pinned
source or create a separate study instead of bypassing identity validation.

## Executed checks

- 34 new regression tests passed.
- Complete decoder-focused suite: 97 passed, exit 0.
- Independent source-only copy: 84 passed, 13 missing-artifact skips, exit 0.
- Four old-versus-new eight-update checks (hidden/history, seeds 7/19): exact equal
  weight tensors, sample hashes and validation histories; maximum weight change 0.
- All 53 inherited .pt artifacts in the supplied study bundle retained their hashes.
  This count includes data/experiment files, not 53 trained neural models.

Environment: Linux x86_64, Python 3.13.5, PyTorch 2.10.0+cpu, one Torch thread.
Tests used the supplied standalone functional energy_real reference; this is not
execution of the entire remote repository, hosted Actions or GB10/CUDA tests.
Only tiny generated-feature optimization checks were run. No new teacher fit,
forecast quality result, language accuracy improvement or scaling promotion.

Two early synchronous test launches hit tool deadlines; the final suite ran to a
recorded successful process exit. An initial parity-audit maximum over an empty
feature buffer was corrected in the audit only, then all four comparisons passed.

## Limits

Hashes are consistency checks, not signatures or protection from an actor rewriting
all records. Locks require cooperating processes on a local POSIX filesystem.
Atomicity is per file, not an entire multi-file transaction. A crash before a run's
initial checkpoint is published fails closed on resume. Test labels remain excluded
from fitting; the serialized input hash still identifies the entire cache artifact.
The prior trained adapters and the more complex Fourier/nonlinear research tracks
are not modified or newly validated by this engineering revision.

See DECODER_RESUME_START_HERE.md for executable commands.
