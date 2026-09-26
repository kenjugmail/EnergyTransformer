# Decoder study publication polish

September 26, 2026. This pass publishes the previously packaged decoder study on
`research/forecast-energy-20260926`, based on `c5583551eb68b89c21337a0cc639e4c1bfad0b00`.
The historical experiment report and results are retained; no new language or
forecast weights were trained, selected, or replaced in this publication pass.

## Changes made before publication

- A partial `decoder_cache` extraction returns process exit code **75**, rather
  than 0. It still prints its partial status and preserves completed shards.
  A completed cache returns 0. Consequently `extract && train` cannot proceed
  through a partial extraction as though a usable dataset existed.
- Unknown completion statuses raise rather than being silently treated as success.
- Batch/stride values must be positive integers (not booleans or fractions);
  time budgets must be positive finite numbers. Invalid block records, unknown
  partitions, and invalid corpus sizes are rejected explicitly.
- An existing completed output is rejected before loading the teacher or creating
  a shard directory. Completed data remains unchanged in the regression test.
- The runbook documents the exit statuses and the Git branch route. Existing
  extraction identities include source hashes: do not change source during an
  extraction and expect exact-resume identity checks to accept it.

## Tests executed in this pass

The uploaded experiment was first verified against all **266** original manifest
entries. Its unmodified decoder suite passed **42 tests**. After the polish,
**63 tests passed**, including 21 additional parametrized regression cases. A
separate copy without private study checkpoints passed **50 tests**, with **13
explicit missing-artifact skips**. The trained hidden-adapter demo completed
successfully with the original teacher and adapter bytes.

These are focused decoder tests, not the entire remote repository's historical
suite or a claim of successful hosted CI. The local test environment uses the
bundle's documented functional `reference/energy_real` snapshot; its comments and
formatting differ from remote source blob identities. Existing forecasting
module dependencies were checked against the branch blob identities. No CUDA,
DGX Spark, remote training, or fresh language evaluation was executed.

See `publication_tests.log` for the new test summaries. The older 15-fit study's
results remain development evidence on reused news domains. Hidden-only remained
its validation-selected adapter; this publication does not promote the energy
reader over that control or merge the draft into main.

## Artifact scope

Source, tests, the original experiment summaries, and the expanded Spark plan are
part of the commit. Original checkpoint binaries and full evaluation arrays
remain in `EnergyTransformer_Decoder_Study_Experiment.zip`, supplied separately.
This source commit is not a GitHub release-asset upload.
