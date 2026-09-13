# V23 — final Spark hardening and source publication

Prepared September 13, 2026 for Kenju Tomita. Engineering continuation of the
self-contained v22 kit. No scientific model training, new corpus training, GPU
execution, NVIDIA-container execution, remote Spark access, or language-quality
improvement was performed or claimed. The original native-text weights are unchanged.

## Actual changes

1. The JSONL importer no longer retains all document bodies in Python memory. It
   indexes document identities in temporary SQLite and spools text to a disk file.
   Stable exact-content deduplication, split ordering, shard bytes, offsets and
   manifest identity remain unchanged on the tested inputs. The first duplicate's
   provenance is retained, as before. Scratch disk can require another corpus copy;
   manifest metadata still scales with document count and one complete input record
   is decoded at a time. This is not a complete near-duplicate or rights audit.
2. Resume reconciles `steps.jsonl` with the authoritative saved checkpoint. Bytes
   representing an uncommitted crash tail are preserved separately rather than
   counted a second time. Corruption in the committed portion causes a failure.
   The existing full-trainer optimizer/RNG/data-stream restart test still passes.
3. Acceptance receipts reject nonfinite, future-dated, missing, and stale times.
   Shape caps are checked before allocation. Source identity now also binds launcher
   and configuration files. Prior acceptance receipts must be regenerated. These
   hashes are not signatures or protection against a deliberately modified program.
4. A stdlib-only read-only host doctor checks the checkpoint, architecture, Docker
   and driver commands without starting containers, training, or changing the host.
5. The launcher supports an explicit `ENERGY_ACCEPTANCE` path for a fresh receipt.
   Existing result directories and checkpoints are not silently overwritten.
6. A local checkpoint installer copies only the expected SHA-256-verified original
   model from a local `.pt` or standalone kit. It never contacts a service or extracts
   arbitrary ZIP paths, and atomically refuses an existing conflicting destination.
7. Acceptance refuses Python optimized mode (`-O`) because its numerical checks use
   assertions. No numerical thresholds, pilot budgets, model equations, or learned
   parameters were changed.

## Importer measurement

A generated JSONL corpus contains 128 documents and 67,123,986 input bytes. The old
and new importers were run in separate Python processes with `tracemalloc` tracking
allocations during preparation. The raw timing and executed script are retained.

| Importer | Python-tracked peak bytes | Elapsed seconds in this one trial |
|---|---:|---:|
| Original v22 | 137,008,532 | 0.713958 |
| Disk-spooled v23 | 4,313,113 | 0.393711 |

The peak allocation ratio is about 31.8x. This is NOT total process RSS, whole-system
RAM, a GPU-memory reduction, a language-model improvement, or a repeated/statistical
performance study. Both output manifests have identity
`03fae8110c284dc13f34fdb52ee0ac8bc01bab5c15f92f6bc214fd3ff64a8b6d`.
The old and new shards and offsets match byte for byte.

## Tests and limits

The final focused suite passes **83 tests**: 45 inherited kit tests, 25 hardening
cases, and four checkpoint-installation tests, plus nine inherited publisher tests. It is not the full historical
research repository's test suite. Tests cover inherited checkpoint parity, causal
resets, forward/backward comparisons, BF16 boundaries, accumulation, source/runtime
receipt rejection, fresh-process training restart, importer identity, crash-log
recovery, and safe local checkpoint installation. Bash syntax and a sample from the
actual unchanged checkpoint also pass. The extracted final package is checked
separately, with its results recorded in the final verification receipt.

A longer CPU acceptance invocation was interrupted by the container command timeout.
Seven numerical/restart checks and five of six requested shape measurements had
completed. That incomplete run is preserved and is NOT a passing device receipt.
The completed seven checks included zero future-input influence, positive precision,
trained fresh-process inference identity, and tiny-model optimizer/RNG restart.
A 24-update fixed synthetic batch is only an optimization-plumbing check.

The local doctor reports a non-Spark host and unavailable Docker/GPU prerequisites.
No actual GB10, CUDA/BF16, target-device memory, or container result was obtained.
The user's next measurement is `doctor`, then `accept` on the Spark, followed by a
separately selected bounded corpus continuation.

## Unchanged model and pilot scope

The native model has 972,896 learned parameters and stores explicit diagonal
precision/information coefficients. Its checkpoint hash is
`78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`.
The supplied continuation and approximately 3.29M/10.61M from-scratch pilots keep
the same token/time limits, single-device restrictions, and cooperative memory
reserves as v22. The directory name `configs/v22` intentionally preserves those
unchanged configurations. A source change invalidates earlier device receipts.

This is ready to try the bounded acceptance test, not a claim of serious general
pretraining readiness, fractal reasoning, or compute superiority over attention.

## Publication contract

The source update is additive under `spark_kit/`, preserving the existing repository
code. The remote publication receipt must identify the actual committed tree and
head; merely staging a Git tree is not a pushed update. Original historical ZIPs
and standalone artifacts are preserved in the comprehensive downloadable bundle.
Source-tool calls accept inline strings, not a mounted local release-asset file;
direct Git networking in this environment still fails DNS. Binary artifacts are
not labeled as uploaded unless an asset receipt confirms their verified transfer.

No repository visibility, license, old production checkpoint, reviewed-edit policy,
or ledger authorization was changed. The provided private-release uploader allows
the owner to finish bulk transfer from a normally authenticated machine; its
validation-only execution is not a live upload.

External command documentation: GitHub CLI release upload and verification use
https://cli.github.com/manual/gh_release_upload and the release create/download/edit
commands. GitHub release assets must be smaller than 2 GiB, per
https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases .
These references justify the packaging route, not any local empirical result.
