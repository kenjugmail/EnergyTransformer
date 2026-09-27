# Decoder studies: safe stop, resume, and evaluation

Updated September 27, 2026. Use the existing forecasting research branch. The
language backbone, energy equations, trained weights, and draft-PR status are not
changed by this update. No new accuracy or GPU-performance gain is claimed.

## Update and test

Do not change source beneath a running job. From a clean, idle checkout:

```bash
git switch research/forecast-energy-20260926
git pull --ff-only
python -m pip install -e '.[test,forecast]'
python -m pytest -q forecast_tests/test_decoder_integrity.py
```

## Start a new bounded adapter study

Use a COMPLETED feature cache from the current decoder_cache workflow. The
hidden/history controls need no separate forecasting heads. Adapter fitting uses
CPU cached features; this does not launch teacher training on the Spark GPU.

```bash
python -m energy_forecast.train_decoder \
  --data runs/decoder10m/cache.pt \
  --out runs/decoder10m/fits-integrity \
  --kinds hidden history --seeds 11 22 33 \
  --steps 600 --batch 128 --stop-after 200
```

Exit **75** means the planned study is incomplete and can be resumed. Exit **0**
means every requested run completed. Other exceptions are errors, not a request
to blindly retry. FROZEN.json is written only after completion.

Resume with identical data, heads, kinds, seeds, schedule, source and Torch version:

```bash
python -m energy_forecast.train_decoder \
  --data runs/decoder10m/cache.pt \
  --out runs/decoder10m/fits-integrity \
  --kinds hidden history --seeds 11 22 33 \
  --steps 600 --batch 128 --resume
```

A stop is an absolute step, not an additional step budget. A stop before the saved
step is rejected. Stopping at zero now writes a resumable initial checkpoint.
A completed resume verifies checkpoint identities and does not rewrite them.

## Evaluate only the frozen selection

```bash
python -m energy_forecast.evaluate_decoder \
  --data runs/decoder10m/cache.pt \
  --fits runs/decoder10m/fits-integrity \
  --out runs/decoder10m/eval-integrity
```

Evaluation derives its seeds from the selected checkpoint list, rather than
assuming 11/22/33. New manifests also bind STUDY.json. Older complete selection
manifests remain readable when their filenames and hashes are valid.

## Compatibility and limits

Existing adapter/teacher weights and inference formats remain usable. Do not
resume an old training directory under this new source: its missing/new study
identity must fail closed. Complete it with the pinned old source or start a
separately named study. No automatic migration or invented training provenance.

STUDY.json binds the cache file, teacher and supplied head hashes, source files,
model configuration, seeds, batch size, and planned updates. Supplied heads must
also use the same normalization and forecast coordinates as the cache.

The .writer.lock files must remain in place. Locks are local POSIX advisory locks
for cooperating writers, not distributed coordination or authentication. Atomic
JSON files are individual publications, not an all-files transaction. A crash
before the first full checkpoint can leave a directory that must be inspected;
the trainer does not silently start over inside it.
