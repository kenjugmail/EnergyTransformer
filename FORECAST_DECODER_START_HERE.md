# Run the decoder-adapter study and move it to DGX Spark

September26,2026. Research source baseline is forecasting branch `c5583551eb68b89c21337a0cc639e4c1bfad0b00`. New code is additive. This package does not replace main or any production checkpoint.

## Included demonstration

The bundle has the archived973K teacher, all15 new adapter fits, the three existing matching forecast heads, and three folded exports. It does NOT have your actual10.6M teacher or raw news/enwik8 text.

From the extracted standalone `EnergyTransformer_Decoder_Study` directory:

```bash
export PYTHONPATH="$PWD/reference:$PWD"
python -m pip install -r requirements-forecast.txt
python -m energy_forecast.demo_decoder --kind hidden --prompt 'The ' --length 80
python -m pytest -q forecast_tests
```

These are unreliable tiny byte-model samples, not factual answers. The original experiment included 42 decoder tests. The publication pass adds regression tests for incomplete-extraction exit status, configuration validation, and early output-conflict rejection; see `reports/decoder_bridge/PUBLICATION_POLISH.md` for the newly executed test count. The demo defaults to seed11, not cherry-picked generated text.

For energy read, use `--kind energy_read`. The mean-only converted model can run without a forecast head:

```bash
python -m energy_forecast.demo_decoder --kind history \
  --adapter results/decoder_bridge/folded_mean_seed11.pt
```

## First Spark experiment: no forecast-head dependency

Use the already working Torch/CUDA environment. Do not change drivers or packages under an active training job. Use the `research/forecast-energy-20260926` branch containing the decoder-study commit; do not add standalone `reference/` to PYTHONPATH inside the actual repository.

Set real paths on YOUR machine:

```bash
export TEACHER=/absolute/path/to/runs/enwik8_energy_10m/best.pt
export DATA=/absolute/path/to/data/enwik8
export DATA_SHA=2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8
```

These are the project's previously recorded corpus identity and an example model location. Verify your actual file hashes. An old973K adapter will be rejected for the10.6M teacher; retraining is required.

Run repository tests and `energy_real.device_smoke` before GPU extraction. Then create a bounded local cache:

```bash
python -m energy_forecast.decoder_cache \
  --teacher "$TEACHER" --corpus "$DATA" --data-sha256 "$DATA_SHA" \
  --shards runs/decoder10m/shards --out runs/decoder10m/cache.pt \
  --device cuda --train-blocks 64 --eval-blocks 16 \
  --block-bytes 512 --batch 2 --stride 4 --max-seconds 900
```

No GPU extraction was executed here. The command refuses other active GPU compute workloads; never kill a job merely to make it pass. It prints `status: partial` when the invocation budget is exhausted. No completed dataset is then available. Repeat the IDENTICAL command with `--resume` to reuse completed verified shards. The CLI returns exit code **75** for a partial extraction and **0** only for a completed cache. This makes `extract && train` stop before training on partial data. Resume with the same source/configuration; do not change the checkpoint or arguments mid-extraction. Require `cache.pt` plus its completed receipt before training.

A complete default collection has7,936 training positions and1,984 each for validation and test. Nearby origins overlap;64/16/16 byte blocks, not independent articles, are the grouping units. Exact corpus boundaries are preserved. CPU fixtures verify the shard resume logic; CUDA reproducibility remains a target-device test.

Train the first two inexpensive controls on cached features:

```bash
python -m energy_forecast.train_decoder \
  --data runs/decoder10m/cache.pt --out runs/decoder10m/fits \
  --kinds hidden history --steps 600 --batch 128
```

Defaults are seeds11,22,33. The default training code is CPU cached-feature training; it does not run backpropagation through the10.6M teacher or need a forecast head. The adapter hidden widths are recalculated for the actual teacher width, not assumed to remain the973K counts. A run can be stopped at an absolute update with `--stop-after 200` then resumed using the same command, `--resume`, and no `--stop-after`. Configuration/data/source identities must match. CUDA head training and distributed resume are not provided.

Inspect validation and `FROZEN.json` before opening test results:

```bash
python -m energy_forecast.evaluate_decoder \
  --data runs/decoder10m/cache.pt --fits runs/decoder10m/fits \
  --out runs/decoder10m/evaluation
```

The evaluated three-seed default is deliberate; custom seed lists need a corresponding evaluator extension. This step scores cached sampled positions, not a full independent corpus benchmark.

## Only then add forecast arms

Prepare and fit the forecasting heads on the SAME frozen10.6M teacher. Follow the existing `FORECAST_REPAIR_START_HERE.md`, preserving train/selection/calibration/test roles. Use a new cache with `--forecast-data runs/forecast_10m.pt` so its byte blocks and normalization agree. Never silently combine heads and data from different teachers or normalization.

```bash
python -m energy_forecast.decoder_cache \
  --teacher "$TEACHER" --corpus "$DATA" --data-sha256 "$DATA_SHA" \
  --forecast-data runs/forecast_10m.pt \
  --shards runs/decoder10m-forecast/shards --out runs/decoder10m-forecast/cache.pt \
  --device cuda --batch 2 --stride 4 --max-seconds 900

python -m energy_forecast.train_decoder \
  --data runs/decoder10m-forecast/cache.pt \
  --heads runs/forecast-repaired/fits \
  --out runs/decoder10m-forecast/fits --steps 600 --batch 128
```

Ensure the heads directory actually contains `anchored_seed11.pt`, `anchored_seed22.pt`, and `anchored_seed33.pt`. Run all five arms on this new identical cache, not a comparison between models trained on different origin samples. Evaluate with `--heads` pointing to the same directory after selection is frozen.

## Local experiment reproduction

The fixture source files are not bundled. The following requires installed Gensim fixtures and the inherited numeric `results/state_data.pt` (included):

```bash
python -m energy_forecast.prepare_decoder \
  --teacher checkpoints/teacher_973k.pt --forecast-data results/state_data.pt \
  --corpus /path/to/gensim/test/test_data/lee_background.cor \
  --secondary /path/to/gensim/test/test_data/lee.cor \
  --out runs/reproduced-cache.pt
python -m energy_forecast.train_decoder \
  --data runs/reproduced-cache.pt --heads results/repair/fits \
  --out runs/reproduced-fits --steps 600 --batch 128
python -m energy_forecast.evaluate_decoder \
  --data runs/reproduced-cache.pt --heads results/repair/fits \
  --fits runs/reproduced-fits --out runs/reproduced-eval
```

Recreated Torch files can have different serialized hashes even when tensors match. The new fits are bound to their newly generated data artifact. Do not feed a regenerated file into the old frozen-evaluation identity without reconciliation. Exact source used for the original fits is preserved in `executed_source`.
