# Reproduce the word-refinement comparison

Run commands from the repository root with the existing PyTorch environment.
The original word corpus is `data/rewt_words_20261028`; its vocabulary was fitted
only on the pinned training stories. See the [original word report](../rewt_words_20261028/FINDINGS.md)
for corpus preparation and tokenization. Downloaded data and model checkpoints
remain under the repository's existing Git exclusions.

## Training

The following commands reproduce one confirmation seed. Use new output/report
directories: the trainer refuses to overwrite an existing experiment. Repeat
with seeds **291201, 291202, 291203** to reproduce the full comparison.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/study_rewt_word_refinement.py train \
  --profile transformer --hidden 288 --lr .003 \
  --steps 12000 --eval-every 2000 --seed 291201 \
  --output runs/word_reproduce/transformer_291201 \
  --report reports/word_reproduce/transformer_291201

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/study_rewt_word_refinement.py train \
  --profile trajectory --conditioned-phase --supervise-steps 2 4 8 \
  --lr .003 --steps 12000 --eval-every 2000 --seed 291201 \
  --output runs/word_reproduce/refined_291201 \
  --report reports/word_reproduce/refined_291201

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/study_rewt_word_refinement.py train \
  --profile hybrid --lr .003 --steps 12000 --eval-every 2000 --seed 291201 \
  --output runs/word_reproduce/hybrid_291201 \
  --report reports/word_reproduce/hybrid_291201
```

The `hybrid` profile is a separately trained REWT backbone without an energy
head. The `final` profile supports experiments with supervision only at the
last refinement step. `trajectory` supervises the requested intermediate
steps. Inference depths 0, 1, 2, 4, 8, 16, and 32 are evaluated on validation
at each checkpoint. The saved checkpoint records the selected depth and
contains its complete word vocabulary.

The defaults used here are batch 16, context 64, width 128, two layers, four
heads, 16 Fourier modes, AdamW with weight decay 0.01, gradient norm clipping at
1.0, 100-step warmup, and cosine learning-rate decay to 15% of the peak rate.
The monotonicity penalty weight is 0.1. Each run sees 12,288,000 target-token
exposures. Changing the context, vocabulary, seed, or data invalidates a direct
paired comparison with the recorded results.

Training evaluates **validation only**. Each report includes its configuration,
exact data/vocabulary identities, training-source hashes and source snapshot,
per-update metrics, selected checkpoint, and checkpoint hash. CUDA elapsed
training time excludes evaluation; the machine is shared, so recorded times
should not be treated as isolated hardware benchmarks.

## Fresh holdout and frozen evaluation

The fresh holdout preparation is a separate command:

```bash
.venv/bin/python scripts/prepare_rewt_word_holdout.py
```

It fetches an exact byte range of the pinned official validation file, checks
the HTTP range and size, drops incomplete boundary documents, excludes known
document IDs, and reuses the existing vocabulary. It refuses to overwrite an
existing holdout directory. The source metadata, identities, word masks,
uint32 tokens and fixed evaluation offsets are retained locally. Training and
validation data paths remain the same as in the original word corpus.

Before evaluating, freeze a protocol containing the final model hashes,
validation-selected depths, the fresh corpus manifest hash, context, and test
offsets. The recorded protocol is `final_test_protocol.json`. The evaluator
checks that the model, corpus and context belong to that protocol:

```bash
.venv/bin/python scripts/study_rewt_word_refinement.py \
  --data data/rewt_word_holdout_20261029 evaluate \
  --checkpoint runs/rewt_word_refinement_20261029/confirm_refined_291201/best.pt \
  --protocol reports/rewt_word_refinement_20261029/final_test_protocol.json \
  --output /tmp/word_refined_291201_test.json
```

Reproduced checkpoints will generally have different file hashes; give a new
reproduction its own frozen protocol. Do not edit the recorded protocol to make
new weights appear to be the original experiment.

The test file contains per-document metrics at every diagnostic depth. Its
`selected_rounds` is the validation choice. Do not choose a different depth
after reading test results. Primary comparison averages all confirmation seeds;
the default checkpoint is chosen by validation rather than test performance.

## Analysis and inference

```bash
.venv/bin/python scripts/analyze_rewt_word_refinement.py
MPLCONFIGDIR=/tmp/rewt-word-mpl .venv/bin/python scripts/plot_rewt_word_refinement.py

.venv/bin/python scripts/predict_rewt_words.py \
  --checkpoint runs/rewt_word_refinement_20261029/recommended.pt \
  --prompt 'Once upon a' --top 5 --generate 24
```

The prediction CLI's optional `--rounds` overrides the saved validation choice.
More steps are not guaranteed to help a new prompt. No training data, tokenizer
file, or labels are needed to load a complete word checkpoint and run inference.

To run the REWT regression suite:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m pytest -q tests_rewt
```

The [design](DESIGN.md) explains the energy, supervision, controls, data isolation,
and the distinction between energy descent and prediction-loss improvement.
