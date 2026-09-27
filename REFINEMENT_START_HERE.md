# Try the persistent-memory + candidate-energy experiment

This adds an optional scalar-energy refinement head. It does not modify the
working language model, forecast gate, memory permissions, or main branch.
Read `reports/ENERGY_REFINEMENT_20260927.md` before scaling: ordinary adapters won,
and more energy descent often made next-byte prediction worse.

## Run the supplied experiment bundle

The standalone bundle includes the original973K teacher, all21 newly trained heads,
derived cache features, numeric predictions, primary/follow-up protocols, and tests.
No full corpus text is included. From the standalone directory only:

```bash
export PYTHONPATH="$PWD/reference:$PWD"
python -m pip install -r requirements-refinement.txt
python -m pytest -q forecast_tests/test_refinement_core.py forecast_tests/test_refinement_integration.py

# The scalar-energy candidate, TWO validation-selected steps.
python -m energy_forecast.refine_session \
  --adapter results/refinement/fits/energy_memory_seed11/best.pt \
  --steps 2 --prompt 'The ' --length 80

# Stronger ordinary adapter with the current energy-memory read.
python -m energy_forecast.refine_session \
  --adapter results/refinement/fits/memory_ff_seed11/best.pt \
  --prompt 'The ' --length 80
```

These are unreliable tiny byte models, not instruction-following assistants. The
trained artifact is teacher-specific and must not be attached to the10.6M checkpoint.
The adaptive tolerance is available via `--tolerance`, but was NOT selected over
fixed two-step inference. Do not mistake energy decrease for a truth check.

## Source-only checkout

Inside the actual repository, do NOT add the standalone reference to PYTHONPATH:

```bash
python -m pip install -e '.[test,forecast]'
python -m pytest -q forecast_tests/test_refinement_core.py forecast_tests/test_refinement_integration.py
```

Seven integration tests require the supplied artifacts and skip explicitly when
absent. Model, derivative, and generated-feature optimizer tests need no corpus.

## Reproduce the primary15 fits

Use a new output directory; the study launcher refuses an existing directory.

```bash
python -m energy_forecast.refine_train \
  --data results/refinement/data.pt --out runs/refine-primary \
  --kinds feedforward memory_ff recurrent energy_hidden energy_memory \
  --seeds 11 22 33 --steps 600 --batch 128

python -m energy_forecast.refine_evaluate \
  --data results/refinement/data.pt --fits runs/refine-primary \
  --out runs/refine-primary-evaluation
```

The evaluator validates selected checkpoint hashes, chooses inference depth on
validation first, saves that decision, and then evaluates the declared final sets.
Re-running this known dataset is development work, not a new independent test.

Post-hoc diagnostic reproduction uses `--kinds latent_ff energy_full` in another
fresh fit directory. Their original results must remain labeled post-hoc.

## Regenerate causal features

The original feature extraction uses the installed Gensim `lee_background.cor`
and `lee.cor` fixtures plus the inherited forecast split manifest. It requires
exact corpus/teacher hashes. `refine_data --help` shows these explicit paths.
The bundle already contains the resulting cache, so it is unnecessary for the demo.
Current-memory reads are64-dimensional, different from the prior decoder cache's
16x8 forecasting-history features. Do not rename an old cache to make it compatible.

## On the DGX Spark

This pass did not run CUDA or the10.6M teacher. Existing8/10M cache files containing
only historical8-variable sketches do not contain the new64-dimensional P/S read.
They cannot be used as though they did. For a proper replicate, extract fresh
causal h and compact_energy_read(P,S) under the actual pinned teacher, retain its
hash and fresh split provenance, and fit controls together. The helper
`extract_refinement` accepts a loaded frozen teacher and explicit document records;
a bounded resumable Spark-specific adapter for these new features is still needed.
The existing production `energy_real` trainer and reviewed permissions are unchanged.
Do not launch a larger pretraining job based on these adapter results.
