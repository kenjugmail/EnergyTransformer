# V16: durable reviewed energy edits and portable distribution

September 12, 2026. Engineering continuation; **no new training, language-quality gain,
Transformer comparison, or parser-generalization improvement is claimed**.

## Implemented changes

`EnergyLedger` stores the protected P/S coefficients in local SQLite, with an explicit
revision, model fingerprint, request identifiers, and an append-only edit journal.
All mutations require externally reviewed `Edit` values and `authorized=True`.
Parser predictions are never committed automatically. The hosting application must
supply actual authentication and authorization; the flag is not a security boundary.

`BEGIN IMMEDIATE` serializes write transactions across independent connections and
processes. A request must match the current revision. An exact repeat with the same
request identifier returns its original receipt without another edit. Reusing an ID
for different request content fails. Multi-edit operations, state replacement, and
the journal insertion commit together. A process exiting before commit does not
leave a partially applied state in the tested crash scenario.

The journal is replayable and hash-linked. `audit()` detects inconsistencies between
records and live coefficients. **Hashes are not signatures; the mechanism does not
protect against a malicious complete rewrite or rollback of the database.** This is
one local SQLite database, not distributed consensus. Database and journal storage
grow beyond the 1,088-byte numerical memory. DELETE clears live coefficients but
retains historical values in the journal; it is not privacy erasure.

`portable.py` exports the existing bounded v14 parser and answer decoder as
`model.safetensors` plus JSON vocabulary/configuration. All tensor values are kept,
without quantization, pruning, retraining, or changing inference architecture.
A dedicated loader checks the weight checksum and exact tensor keys. The format
avoids pickle for the deployed tensor file, but does not certify that learned
predictions are correct. It is not Hugging Face AutoModel integration or hosted
inference. Original .pt artifacts remain preserved separately.

## Executed checks

**67 tests pass locally**: 43 inherited and 24 additional test cases. New tests cover
transaction atomicity, explicit authorization, stale revisions, exact request retries,
request-ID collisions, defensive copies, restart, intentional journal and state
corruption, and rollback after forced process exit. Eight independent thread
connections and four separate Python processes each compete for revision zero;
exactly one commit succeeds in each scenario and the others receive stale-revision
errors. These are deterministic mechanism tests, not a comprehensive crash/power-loss
or adversarial-storage certification.

For all three existing checkpoints, every exported tensor is bitwise equal to the
original. Parser logits on three specified sentences and answer logits for all 17
bounded outcomes match exactly. This proves the checked serialization preserves the
stored model, not broader capability or equivalence across arbitrary devices.

The durable demo reopens a stored association and returns `garden`; repeating the
same request leaves revision 1 unchanged. The journal audit reports one valid event.
No parser proposal is automatically approved in the demo.

## Publication work and limits

All 69 originals indexed by v15 were located in the current runtime and matched
recorded SHA-256 values: **303,637,647 original bytes**. V15's own small package and
standalone report/receipt/checksum are additionally retained in the new full bundle.
The previous full-archive ZIP itself is not a mounted input; the new full archive is
assembled from verified originals, not represented as byte-identical to that ZIP.

The current connector exposes inline source/blob writes but no mounted-file release
upload argument. Direct DNS lookup for GitHub and Hugging Face fails in this runtime.
The Hugging Face app is not connected; its directory description is inspection,
not proof of binary-upload capability. Connecting an app alone is not represented
as a solution to local artifact transport.

GitHub Releases can hold this approximately 300 MB bundle: individual release assets
must be below 2 GiB. **Hugging Face is optional, not required by the archive size.**
The new `publish_hf.py` is a local script for an explicitly named EXISTING PRIVATE
model repository. It defaults to validation only. On `--publish`, it uses existing
local Hugging Face authentication, checks privacy, uses content-addressed paths,
rejects conflicting existing bytes, commits with an expected parent, then downloads
both archive and checksum at the returned immutable commit and compares hashes.
Its five fake-backend tests are not a live Hugging Face upload claim.

The GitHub packaging workflow only reuses the already published, hash-pinned bounded
checkpoint release, exports portable tensors, runs tests, and packages the current
source. **It cannot recover the original v1-v15 archive from local chat storage.**
A workflow definition is not a completed upload; the separate publication receipt
records the observed run and release outcome.

## Run

```
pip install -r requirements.txt
python -m pytest -q tests
python scripts/demo_durable.py
python scripts/export_portable.py  # only when portable/ does not already exist
```

For the optional HF route, create/select a private repository in your own account
and authenticate on your machine with `hf auth login`. Never paste a token into chat.
Then run `python scripts/publish_hf.py --repo-id ACCOUNT/EnergyTransformer --archive
/PATH/TO/ARCHIVE.zip --publish`. Without `--publish` no external call is made.
The existing `publish_release.py` supplies the GitHub route. A complete original
archive upload is not claimed unless a live, verified receipt records it.

## Primary documentation

- GitHub release limits: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- HF uploads: https://huggingface.co/docs/huggingface_hub/guides/upload
- HF authentication: https://huggingface.co/docs/hub/security-tokens
- Safetensors: https://huggingface.co/docs/safetensors/index
- SQLite transaction semantics: https://www.sqlite.org/lang_transaction.html

The model still has 16 names and 16 locations. Its previously reported failure on
unfamiliar phrasing remains unresolved. This release improves persistent execution
and packaging; it must not be described as a more capable language model.
