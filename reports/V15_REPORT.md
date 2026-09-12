# V15: reviewed memory edits and publication completion work

September 12, 2026. This is an engineering continuation of the supplied v14 package. **No new model was trained locally, and no parser-generalization improvement is claimed.**

## Scope

V14 correctly retains already parsed facts but fails badly on held-out phrasing. The source report records mean held-out exact parsing of 57.8% and 1.04% guarded recall after 1,024 unfamiliar distractor records. Those results are retained, not reclassified as solved.

`ReviewedEnergySession` now exposes a two-stage interface. `propose(text)` uses the same parser, but can only return an answer, a hold, an abstention, or an edit needing review. It never executes parser-predicted writes. `approve` requires the caller to supply the exact reviewed structured meaning, explicit authorization, and the current revision. Proposals are bound to the existing coefficient hash and invalidated after any intervening edit. Approval is one-use. `apply_edits` provides atomic explicit edits without depending on linguistic parsing.

This prevents the dangerous *automatic application* of a misparse when applications use the new interface. It does not make a proposed meaning correct. A reviewer can approve the wrong edit. The hosting application must implement real authentication and authorization; a boolean flag is not a security boundary against a hostile caller controlling the process.

## State and lifecycle

The protected memory still stores float32 precision and information coefficients (1,088 raw bytes). Administrative metadata adds a revision, model identity, hashes, and at most 128 process-local pending proposals. This metadata is extra memory and is not hidden in the coefficient count. Pending proposals are not included in snapshots and expire on restore. Queries remain temporary. The inherited v14 API is preserved for reproduction and must not be confused with the reviewed API.

Snapshots bind to model bytes and include an integrity hash of the coefficient tensors. Hashes detect accidental mismatch; they are not digital signatures or proof of factual truth. Restoration and approval are serialized by an in-process lock, not a distributed transaction service. Restoring an earlier valid snapshot is an explicit rollback, not prevented by revision numbers.

## Executed checks

The full current suite passes **43 tests** with the supplied seed-11 checkpoint. Twenty are inherited v14 checks. New checks exercise nonmutating proposals, authorization defaults, exact semantic approval, replay rejection, stale-proposal invalidation, atomic rollback on invalid batches, defensive state copies, bounded proposals, snapshot integrity, and a fresh-process restore. Publication tests include a simulated create/upload/verify/publish cycle, resume with matching existing assets, mismatch refusal, and failure-closed handling of permission errors. Simulated publication tests are not a live upload claim.

The executed demo returns unknown before approval, garden after an explicitly reviewed set, garden after an unapproved new sentence, and unknown after an explicitly authorized deletion. No new memory-capacity or language-accuracy benchmark is inferred from these tests.

## Artifact recovery

All 65 artifacts named in the previous full-payload inventory were recovered from their mounted originals and their SHA-256 hashes matched. The original v14 ZIP and its three standalone exports were added. The resulting inventory contains **14 experiment ZIPs and 55 standalone artifacts** (69 originals total, 303,637,647 bytes). All 14 original ZIPs pass CRC checks. Their 1,945 entries were screened for unsafe paths, selected sensitive filenames, and limited credential patterns; no flags were found. This is not a complete privacy or licensing audit.

The original binary artifacts are preserved byte-for-byte, not regenerated or quantized. The existing private repository remained at commit `0898a528756144959c09f116036855abfe8be384` when this continuation started. The final source-publication receipt separately records the actual new commit and remote verification.

## Transfer limitation and reproducible builds

Direct Git networking from this runtime still fails DNS resolution. The current connected GitHub actions accept inline source/blob contents but do not offer a local-file or release-asset upload parameter. No source-only commit should be described as uploading the entire binary collection.

The complete bundle is packaged for direct download. `scripts/publish_release.py` validates a specified local ZIP, uses existing `gh` authentication, refuses to target a public repository, and creates a draft prerelease only on explicit `--publish`. It never force-pushes, overwrites assets, or changes visibility. Existing assets must match by re-download and SHA-256 or the script stops. Final publication follows verification of the archive and checksum sidecar. The script has offline simulated tests; live bulk execution remains unverified unless a publication receipt states otherwise.

A GitHub-hosted checkpoint rebuild, if accepted by the repository's workflow permissions, is a distinct reproduction route. Rebuilt checkpoints are new training outputs tied to the source commit. They must not be represented as exact transfers of the historical checkpoints or archives. Any workflow outcome is recorded separately; merely committing its YAML is not a successful build.

## What remains

Original bulk artifact upload still requires a working transfer path. Semantic robustness across unfamiliar wording and joint memory-conditioned general language training remain unsolved by this engineering change. No new LLM, GPU benchmark, attention comparison, or fractal-reasoning result is claimed.

## Primary command references

- GitHub CLI release upload: https://cli.github.com/manual/gh_release_upload
- GitHub CLI release create: https://cli.github.com/manual/gh_release_create
- GitHub CLI release download: https://cli.github.com/manual/gh_release_download
- GitHub CLI release edit: https://cli.github.com/manual/gh_release_edit

Architectural motivation: Kenju Tomita. Implementation and execution: this conversation with ChatGPT. No independent replication or peer review is claimed.
