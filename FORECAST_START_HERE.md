# Joint energy forecasting — experimental, 2026-09-26

Based on `kenjugmail/EnergyTransformer` at `397dfae647f9521875344b99227a51a3974c0dfd`.
This additive module leaves the current language model, protected-memory permissions,
and ledger untouched. It predicts **distributions of future energy-memory summaries**,
not future words or complete replacement P/S states. It does not skip decoding.

## What is implemented

* A past-only 16-step history of eight summary quantities (two per final-layer head).
* Joint targets at +1, +4, +16, +64 **bytes** for the current byte model.
* Independent Gaussian, correlated Gaussian, and three-component correlated mixture
  controls; a same-size Fourier-history mixture is an ablation.
* Exact normalized density and explicit candidate energy, differentiable candidate
  gradients, joint sampling, and fixed-energy local MAP refinement.
* A separate ForecastSession which preserves the original LM's logits and P/S and
  adds a 512-byte float32 history of summaries for this four-head configuration.
* Document-separated prototype data and a fixed-split enwik8 adapter for the actual
  repository's 10.6M checkpoints. Training-only normalization and checkpoint binding.

## Local result, not a language improvement

The executed study used the archived 972,896-parameter teacher, not the 10.6M GB10
weights unavailable in this container. On 634 forecast origins from 32 held-out
Lee Background documents, the spectral mixture had 0.897245 standardized MSE and
1.214732 nats/coordinate negative log likelihood. Ridge regression had better MSE,
0.880202. At +64 bytes, spectral MSE 1.109560 was worse than the constant training
mean's 1.052856. Do not promote this module to the default decoder from these results.
All existing language predictions are unchanged by the sidecar.

## Standalone download

The standalone bundle includes a functionally copied, comment-condensed snapshot of
`energy_real/model.py` and `kernel.py` under `reference/`, an archived teacher, all
18 head fits, projected numeric data (not the corpus text), and results.

```bash
export PYTHONPATH="$PWD/reference:$PWD"
OMP_NUM_THREADS=1 python -m pytest -q forecast_tests
python -m energy_forecast.demo \
  --teacher checkpoints/teacher_973k.pt \
  --head results/native_runs/spectral_seed11/best.pt \
  --text 'The new research report describes '
```

In the actual repository, **do not add the standalone `reference/` to PYTHONPATH**:
use its current `energy_real` implementation. A source checkout does not include
these binary checkpoints; train a head for your own checkpoint as below.

## Bounded run on the current Spark checkpoint

Keep the existing PyTorch/CUDA environment. `requirements-forecast.txt` lists only
extra Python dependencies. This code was CPU-tested; CUDA throughput is unmeasured.
The corpus path below is an example: point it to your preserved local enwik8 file.
The corpus fingerprint and split boundaries match the inspected repository.
The CUDA extraction refuses to start over another GPU workload.

```bash
python -m energy_forecast.prepare_enwik8 \
  --checkpoint runs/enwik8_energy_10m/best.pt \
  --corpus /absolute/path/to/enwik8 \
  --data-sha256 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 \
  --out runs/forecast_data.pt --device cuda \
  --train-blocks 128 --eval-blocks 32 --max-seconds 900

python -m energy_forecast.experiment \
  --data runs/forecast_data.pt --out runs/forecast/spectral_seed11 \
  --kind spectral --seed 11 --steps 600
```

Repeat the head command for `diagonal`, `joint`, `mixture`, `spectral` and seeds
11/22/33. Head training defaults to CPU. With all training complete:

```bash
python -m energy_forecast.evaluate \
  --data runs/forecast_data.pt --runs runs/forecast --out runs/forecast_evaluation
```

A learned head is bound to its teacher checkpoint. Do not use the old 973K head on
the 10.6M model: rerun extraction and training. The original language checkpoint is
never optimized, overwritten, or automatically replaced by these commands.

## Boundaries

This is a tractable Gaussian-mixture energy model, not a free-form neural EBM, a
new mathematical paradigm, fractal attention, or a lossless state compression.
Only summaries are forecast, and Gaussian tails can include unrealizable summary
combinations. No reconstruction or committing of future P/S is exposed.
Fourier features operate on observed TIME, not arbitrary latent-coordinate order;
the full transform is invertible and adds no information. Correlations and mixtures
change the forecast density; the spectral variant rotates the input representation.
A lower candidate energy need not mean lower error or a more correct answer.
See `reports/FORECAST_20260926.md` for evidence, controls and next steps.
