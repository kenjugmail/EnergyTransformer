# Robust multivariate conditioning and verified forecast packaging

The existing calibrated forecast weights, mean anchors and online gate are retained.
No teacher is retrained, no observation is replaced with a prediction, and no
language-decoding speed or +64-byte forecasting improvement is claimed.

## Repository installation

From a checkout of `research/forecast-energy-20260926`, using the existing Torch
installation (in particular, keep NVIDIA's Torch build on DGX Spark):

```bash
python -m pip install -e '.[test,forecast]'
python -m pytest -q forecast_tests
python -m energy_forecast.numerical_audit --out runs/square_root_check.json
```

The source-only tests explicitly skip binary-dependent integration tests when
historical artifacts are absent. Forecast modules now participate in the installed
package, and the research workflow runs `forecast_tests` on pushes to this branch
and on pull requests. A configured workflow is not a completed remote CI run.

## Standalone artifact bundle

The download includes the original 973K teacher, calibrated heads and online gate:

```bash
cd EnergyTransformer_Forecast
export PYTHONPATH="$PWD/reference:$PWD"
pip install -r requirements-forecast.txt
python -m energy_forecast.online_demo
python -m pytest -q forecast_tests
```

Only the standalone copy needs the `reference` path. A repository checkout uses
its own `energy_real`; do not replace that code with the archived reference.
The included head/gate belong to the 973K teacher, not your 10.6M checkpoint.

## Reproduce numerical checks on the retained trained head

```bash
python -m energy_forecast.numerical_audit \
  --head results/repair/fits/anchored_seed11.pt \
  --data results/state_data.pt \
  --out runs/new_square_root_audit.json
```

This performs no model fitting. It compares exact distributions on the existing
634 test origins and benchmarks component operations on CPU. It is not a fresh
quality benchmark. Use a new output path; previous reports are not overwritten.

## What changed

Ordered marginals take an owned slice of the stored square root. Reordered
leading observations are canonicalized, and arbitrary subsets use reduced QR
on factor rows. No covariance subtraction, jitter or diagonal approximation is
introduced. General QR can be slower; the change prioritizes avoiding cancellation.
The exact-posterior online policy no longer computes an unused prior marginal.

Snapshot writes use unique temporary files. Dataset export uses create-only atomic
publication so an interrupted or competing writer does not replace a completed
dataset. These guarantees assume local POSIX filesystem semantics, not a distributed
transaction. Nothing automatically launches a training or GPU job.
