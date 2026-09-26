# Faster exact conditioning; do not replace the calibrated heads

The new ordered square-root implementation is enabled automatically when actual
observed coordinates are a leading horizon prefix. It does not change forecast
parameters, teacher logits or guarded policy. All nine learning-update candidates
selected their original weights, so the current heads and gate are retained.

## Run the bundled model

```bash
export PYTHONPATH="$PWD/reference:$PWD"  # standalone archive only
pip install -r requirements-forecast.txt
python -m energy_forecast.online_demo
python -m pytest -q forecast_tests
```

The existing model artifacts in this archive use the original973K teacher, not the
10.6M checkpoint on the Spark. For a repository checkout, use its installed
`energy_real` rather than the standalone `reference` folder.

## Reproduce one bounded head-training arm

```bash
python -m energy_forecast.conditional_training \
  --data results/state_data.pt --parents results/repair/fits \
  --out runs/new_conditional_seed11 --kind conditional --seed 11 --steps 350
```

Other choices are `joint` and `conditional_mse`. Run seeds11/22/33 with all three
choices to reproduce the nine fits. The script is CPU-only; no teacher or language
weights are updated. Existing output directories are not overwritten. A stopped
run resumes using the same arguments plus `--resume`; its budget cannot change.
`--stop-at N` is available for restart testing and does not add checkpoint-selection
opportunities. Whole-model future targets remain training annotations only.

The standalone `run_conditional_study.py` records the protocol, runs all fits, and
freezes selected hashes. Its results directory must be new; the shipped original
record is not overwritten. The documented individual command is simpler for a
fresh run. Initial weights are eligible for validation selection.

## Numerical benchmark (no corpus or Spark access required)

```bash
python -m energy_forecast.prefix_benchmark \
  --head results/repair/fits/anchored_seed11.pt --data results/state_data.pt \
  --out runs/new_prefix_benchmark.json
```

Make the parent output directory first. Times exclude teacher execution and head
construction. These measurements are not language decoding speedups.

## Separate documents (optional; raw corpus not bundled)

Use `python -m energy_forecast.prepare_secondary --help`. Supply the actual teacher,
parent numeric dataset, new one-document-per-line corpus, and original corpus for
the exact64-byte overlap check. It creates numeric traces and a hash manifest only.
An unknown corpus is not silently downloaded. A different teacher requires a newly
prepared dataset and separately fitted anchors/heads/gates using the earlier
`FORECAST_START_HERE.md` and `FORECAST_REPAIR_START_HERE.md` protocols.

Read `reports/FORECAST_CONDITIONAL_20260926.md` for failures, timing scope, and the
separate-set result. No new model or gate should be promoted from a test-only score.
