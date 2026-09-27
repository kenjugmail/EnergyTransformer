# Observed-state forecasting: run and reproduce

This is an optional continuation of draft PR #1. No LM weights are changed.
The standalone bundle includes the archived 973K teacher, inherited forecasting
heads, numeric traces, and the validation-selected gate. The Git branch contains
source and small result records, not those binaries.

## Standalone demo

From the extracted `EnergyTransformer_Forecast` folder:

```bash
export PYTHONPATH="$PWD/reference:$PWD"
pip install -r requirements-forecast.txt
python -m energy_forecast.online_demo
python -m pytest -q forecast_tests
```

The demo issues a forecast, then runs 64 real continuation bytes through the
teacher. At elapsed +1 and +4 it updates the remaining distributions; at +16 the
validation gate keeps the original +64 distribution. After +64 all original
targets are observed. It neither guesses input bytes nor skips the LM.

## Minimal API

```python
import torch
from energy_forecast.online import ObservedForecastSession
s = ObservedForecastSession(
    "checkpoints/teacher_973k.pt",
    "results/repair/fits/anchored_seed11.pt",
    gate_path="results/online_gate/gate.json",
)
s.observe(torch.tensor([list(b"A sufficiently long real prefix.")], dtype=torch.long))
s.issue()
s.observe(torch.tensor([list(b" and")], dtype=torch.long))
print(s.status())                 # Original target timestamps; only matured evidence
forecast = s.distribution()      # Normalized, validation-gated forecast
print(forecast.mean)
s.save("observed_forecast.pt")
```

`distribution(policy="exact_conditional")` exposes the exact statistical
conditioning diagnostic, which was worse in several tests. The default `guarded`
policy requires a matching validation gate to use it; without a gate it keeps the
original distribution. Gate weights are not guarantees. No API admits guessed
summaries as actual evidence. Session methods are single-threaded.

## Reproduce the frozen-head audit

Use new output directories. No new neural training is needed.

```bash
python -m energy_forecast.conditioning_audit \
  --data results/state_data.pt --fits results/repair/fits \
  --out runs/online_raw
python -m energy_forecast.conditioning_gate \
  --data results/state_data.pt --fits results/repair/fits \
  --out runs/online_gate
```

The scripts explicitly target eight variables and +1/+4/+16/+64 horizons. The
underlying `GaussianMixture` conditioning library is dimension-general, but its
cost is dense. The study artifacts are tied to the 973K teacher. To use the 10.6M
Spark model, follow `FORECAST_REPAIR_START_HERE.md` to extract its own traces and
fit its own anchored heads FIRST; then run this audit with the new data and heads.
Do not reuse the old teacher, normalization, or gate. CUDA was not tested here.

MSE/NLL comparisons at an update time score only still-unobserved targets. A
forecast at t+16 of the original t+64 target has 48 bytes left, not 64. Receiving
new observations is not better foresight at time t.
