# EnergyTransformer — final Spark hardening (v23)

The existing 972,896-parameter language checkpoint is unchanged. V23 is an
engineering revision, not another scientific training run or GPU-validation claim.

## Use on the DGX Spark

From the standalone download, enter its root. From GitHub, enter `spark_kit/`.
The Git checkout supplies source; the standalone download also supplies the original
`checkpoints/v21_energy_973k.pt`. `doctor` explicitly reports a missing checkpoint. Install the unchanged binary from
the downloaded kit with `python -m energy_spark.install_checkpoint /path/to/EnergyTransformer_v23_Spark_Kit.zip`.

```bash
bash spark/run.sh doctor
bash spark/run.sh accept
```

Doctor is read-only: no image pull, driver change, account access or training.
Acceptance runs the existing bounded tests on the machine; it does not launch a
pretraining campaign. No device acceptance was obtained while preparing v23.

After acceptance succeeds, prepare your permitted JSONL documents and run the capped
973K continuation exactly as documented in `SPARK_START_HERE.md`. Larger pilots
remain separate explicit commands. No thresholds were relaxed in this revision.

`accept` preserves old receipts and refuses to overwrite one. For a new test:

```bash
bash spark/run.sh accept --out /runs/acceptance-v23
ENERGY_ACCEPTANCE=/runs/acceptance-v23/acceptance.json bash spark/run.sh train \
  --config /workspace/configs/v22/continuation_973k.json \
  --data /data/corpus --out /runs/pilot-v23
```

The config directory remains `v22` because sizes, schedules and budgets did not change.
Changing source/launcher/configuration invalidates old device acceptance receipts.

## Actual changes and tests

* The corpus importer now spools text to disk with a SQLite index, retaining only
  per-document metadata and the current input record in Python memory. Its scratch
  disk use can approach another copy of the corpus while shards are being written.
  The complete manifest still scales with document count. This is not an unlimited
  streaming service and license declarations are not independent legal clearance.
* In a 64 MiB generated corpus test, Python-tracked peak preparation allocations
  dropped from 137,008,532 to 4,313,113 bytes (about 31.8x). Exact shard bytes and
  manifest identity matched v22. This is one synthetic importer measurement,
  not total process RSS or a language-model speedup.
* Resume now preserves uncheckpointed log tails in `recovered_steps.*.jsonl` and
  reconciles the main log to the authoritative saved optimizer step. Committed-log
  corruption fails closed instead of silently rewriting history.
* Future-dated/nonfinite/stale acceptance timestamps and oversized per-operation
  configurations are rejected. Symlinked dataset shards are rejected.
* A read-only host doctor makes missing model/Docker/driver prerequisites explicit.

The final focused suite passed 83 tests before packaging. See `reports/v23/` for its
actual result, the importer measurement, and the interrupted full CPU acceptance
log. That longer acceptance attempt completed seven numerical/restart checks and
five shape measurements before the container command timed out; it is not a
completed device receipt. No GPU result, Docker execution, new weights, new quality
benchmark or serious-pretraining approval is claimed.

## Publication boundary

GitHub receives the complete current Spark source, configs, tests, and small reports.
The connected tool accepts inline text but cannot stream the local binary archives;
direct Git networking from this runtime fails DNS. The original checkpoint and
original research ZIPs remain byte-for-byte in the supplied downloads unless an
explicit publication receipt records a successful asset upload. Do not call a
source commit a transfer of those binary artifacts.

No repository visibility, license, existing production checkpoint, reviewed edit
policy or ledger authorization is changed.

## Finish original binary publication from a networked machine

After downloading the full historical bundle, use existing GitHub CLI authorization:

```bash
python publish_release.py \
  --archive /path/to/EnergyTransformer_All_Research_v1_v23.zip \
  --publish
```

Omit `--publish` for local validation only. The publisher refuses a public target
and conflicting assets, creates a draft prerelease, verifies downloaded bytes,
and only then publishes. It does not change repository visibility or force-push.
Its simulated tests are not a claim that the original archive has been uploaded.
