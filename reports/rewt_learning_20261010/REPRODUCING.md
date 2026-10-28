# Reproduce the learning study

Use the repository's existing environment. The data, saved checkpoints and
study artifacts are local; no pretrained model is required. Synthetic-task and
story models are separate checkpoints.

## Generate an answer

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/binding/aligned/best.pt \
  --prompt $'A=19;B=A+23;B=?\n' --greedy --stop-newline --max-new 40
```

This command generated `steps=42;ans=42`. Use `--json` for the exact bytes,
checkpoint hash and energy-call counts. The sampler accepts a raw prompt or
`--prompt-file`; it receives no answer, task payload or training-only head.
The saved [sampler checks](sampler_checks.json) contain actual executions.

The separate longer-chain checkpoint can generate two-rule traces:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/multistep_balanced/progress/best.pt \
  --prompt $'walk E 2:L>U;T>D;E>L;S>T=?\n' --greedy --stop-newline --max-new 40
```

This command generated `steps=L,U;ans=U`. The fresh two-rule score is 30/41,
not perfect accuracy; deeper chains and multi-step arithmetic remain unreliable.

The final language checkpoint accepts ordinary ASCII text through its frozen
training vocabulary:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/language_bpe_extended/fourier80/best.pt \
  --prompt 'Once upon a time, there was a little girl named Lily.' \
  --max-new 96 --temperature 0.5 --seed 1991
```

Its generated prose remains unreliable. The command demonstrates actual model
inference, not a claim of coherent storytelling. Tokenized checkpoints verify
the saved vocabulary's SHA-256 before use.

## Check the implementation

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests_rewt -q
```

The current suite has 88 passing tests. [Verification metadata](FINAL_VERIFICATION.json)
records library versions, model-source hashes, test hashes and the sampler hash.
Archived default-checkpoint parity is recorded in
[legacy_parity_relative_waves.json](legacy_parity_relative_waves.json).

## Training lineage and controls

The helper scripts are runnable experiment programs, and their saved `plan.json`
files describe the exact data identities, seeds, configurations and source hashes.
Run training in a fresh copy of the repository: the helpers use fixed experiment
directories and would overwrite their own study outputs in this checkout.
Preserve the original runs and reports when reproducing them.

The selected one-step checkpoint follows these stages, in order:

| Script | Parent stage | Declared fit updates | Selected checkpoint update |
|---|---|---:|---:|
| `foundation.py` | Random initialization, seed 941 | 1,024 | Combined arm, 512 |
| `grounding.py` | Combined foundation | 1,024 | Grounded arm, 1,024 |
| `grounding_v2.py` | Grounded | 1,024 | Operand-copy arm, 1,024 |
| `grounding_heads.py` | Operand-copy | 1,024 | One-head arm, 1,024 |
| `precision.py` | One-head | 2,048 | One-head arm, 2,048 |
| `binding.py` | Precision one-head | 1,024 | Aligned arm, 896 |

This is 6,528 updates along one selected initialization lineage. The scripts
also train comparison arms. Auxiliary heads are saved for continuing training
but are excluded from inference. Ground-truth arithmetic and graph labels occur
only in task generation, training annotations and post-generation checking.

The story experiments use `prepare_stories.py` to obtain a capped, pinned
TinyStories subset, followed by `prepare_bpe.py` to learn a 256-piece vocabulary
from training text. `language_bpe.py` trains the matched 250K-parameter models
for 2,048 updates from random weights; `language_bpe_extended.py` continues all
four arms for another 2,048 updates with their optimizer moments retained.
The Fourier model is selected by validation loss before fresh testing.

Longer-chain helpers retain the frozen one-step parent as a separate artifact.
`multistep.py`, `multistep_position.py`, `multistep_huber.py` and
`multistep_balanced.py` share declared validation and fresh-test tasks. The final
helper balances one- and two-step examples and all recall delays. Its
[metadata clarification](multistep_balanced/METADATA_CLARIFICATION.json)
corrects inherited prose in the original plan without modifying that plan or
the frozen execution source.

The [fit ledger](completed_fit_ledger.json) reports completed alternatives and
measured costs. Equal updates or target tokens do not imply equal CPU time.
Continuation seeds are not independent random-initialization replications.

## Locked evaluation

`test_onehop.py`, `test_language.py` and `test_multistep.py` save their plans
before generating answers. Model calls receive supplied prefixes only. Expected
responses and references are used by separate scoring code. Checkpoints and
model state dictionaries are hashed before and after evaluation.

The story test and longer-chain test programs refuse to overwrite a completed
test result. For a repeat, preserve their original output directories and use
an isolated copy. The longer-chain test excludes every previously opened
one-step test identity as well as the preceding studies' test identities.
Fresh test results do not feed further model selection.

`uncertainty.py` measures conditional test-sample uncertainty by resampling whole
semantic number pairs or story documents. It does not estimate training-seed
variation. `plot_results.py` exports the measured results as PNG, PDF and SVG.
The assistant's anonymous text ratings are exploratory; human ratings remain
pending.
