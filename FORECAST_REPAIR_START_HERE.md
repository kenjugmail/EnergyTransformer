# Calibrated joint energy forecasting: run the repair

Continue the existing `research/forecast-energy-20260926` draft, not the working
language-model baseline. No model weights or permissions in main are changed.

## Run the included checkpoint (no corpus required)

The standalone archive contains original and repaired heads, the exact teacher,
numeric summary tensors, and all results. Do not add `reference/` to PYTHONPATH
when running inside the actual repository: use its `energy_real` implementation.
For the standalone directory only:

```bash
export PYTHONPATH="$PWD/reference:$PWD"
pip install -r requirements-forecast.txt
python -m energy_forecast.demo \
  --teacher checkpoints/teacher_973k.pt \
  --head results/repair/fits/anchored_seed11.pt
python -m pytest -q forecast_tests
```

The demo labels +64's point mean as `training_target_mean`. That is the measured
fallback, not a successful 64-byte prediction. The density can still represent
uncertainty and alternative future summaries. The language decoder is unchanged.

## Reproduce the head repair

```bash
python -m energy_forecast.repair \
  --data results/state_data.pt \
  --parent-runs results/native_runs \
  --out runs/repair-new

python -m energy_forecast.repair_evaluate \
  --data results/state_data.pt \
  --parent-runs results/native_runs \
  --fits runs/repair-new \
  --out runs/repair-new-test
```

Both commands refuse existing output directories. They default to a bounded CPU
study. Fit runs three 600-update heads and six four-parameter calibrations. Frozen
hashes must match before evaluation. The fitter cannot receive test tensors through
its internal fit routines; the legacy serialized dataset contains all splits, but
its test view is removed immediately on loading.

## On the actual 10.6M Spark teacher

First prepare traces using the existing `prepare_enwik8` command and train the
three parent spectral heads for THAT checkpoint, all on the exact same data file:

```bash
for seed in 11 22 33; do
  python -m energy_forecast.experiment --data runs/forecast_data.pt \
    --out runs/forecast-parent/spectral_seed$seed --kind spectral --seed $seed --steps 600
done
python -m energy_forecast.repair --data runs/forecast_data.pt \
  --parent-runs runs/forecast-parent --out runs/forecast-repaired
python -m energy_forecast.repair_evaluate --data runs/forecast_data.pt \
  --parent-runs runs/forecast-parent --fits runs/forecast-repaired \
  --out runs/forecast-repaired-test
```

The current 973K-teacher head must NOT be reused for the 10.6M checkpoint. CUDA trace
extraction uses the repository's idle-device guard. No Spark extraction or fit was
executed as part of this CPU repair; do not start over another user's GPU workload.

## What to inspect

- `mean_selection.json`: train crossfit penalties, gains and validation evidence.
- `FROZEN.json`: selected checkpoint hashes before test scoring.
- `summary.csv` and `paired.csv`: MSE, exact density NLL, energy score and intervals.
- `training.json`: all density fits and calibration losses, including worse controls.
- `point_forecast_status()`: which point horizons fall back; not confidence of truth.

Use the conditional mean for MSE-oriented predictions. Optional MAP refinement is
not a replacement for that mean, not a global optimizer and not token verification.
Raw summary normalization, teacher binding, positive covariance and atomic loading
are tested. The larger language-training and skip-ahead hypotheses remain untested.
