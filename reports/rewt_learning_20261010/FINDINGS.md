# REWT learning repair, October 10, 2026

The native Fourier joint-energy core now generates useful answers on bounded,
unseen one-step tasks. This is a learned autoregressive model: its inference
inputs are raw byte prompts, and its weights remain fixed while recurrent,
attention, feedforward and output states undergo joint energy refinement.

## Locked fresh one-step test

The checkpoint was selected using validation data, before generating test
answers. Test semantic identities are disjoint from training and validation,
and exclude identities from the three previous REWT studies. Correct means
both the entire intermediate trace and the final answer match the checker.

| Task, familiar wording | Correct / prompts | Complete accuracy | Same weights, zero refinement rounds |
|---|---:|---:|---:|
| One addition, inputs 0–31 | 125 / 132 | 94.7% | 67.4% |
| One rule application with distractors | 80 / 96 | 83.3% | 58.3% |
| Recall after 4 distractors | 96 / 96 | 100% | 87.5% |

The 132 arithmetic prompts contain 66 distinct number pairs, each in two
familiar wordings. They are not 132 independent semantic problems. The primary
policy uses two refinement rounds; the zero-round column disables refinement
in the same trained model and is not an independently trained competitor.
The generator uses starting values 0–31 and positive addends 1–31; this
particular fresh subset contains starting values 0–30.

A paired bootstrap over the 66 number pairs gives a conditional 95% interval
of 89.4–98.5% for two-round arithmetic accuracy. Its 27.3 percentage-point
gain over zero rounds has an interval of 18.2–36.4 points. These intervals
describe test-sample uncertainty for this frozen checkpoint; they do not cover
training-seed or validation-selection variability.
[Method and measurements](test_uncertainty.json), [exportable figure](results_figure.pdf).

The fresh recall test contains four-distractor cases only. One-distractor cases
are present in training and validation, but the prior-test identity exclusions
leave none in this fresh test. A [payload audit](onehop_test/recall_scope_audit.json)
corrects earlier prose that described the training delay range as test coverage.

Transfer to wording withheld from training remains weak: arithmetic 6/66
(9.1%), rules 21/80 (26.3%), recall 85/96 (88.5%). The result supports a bounded
reasoning prototype, not general arithmetic or natural-language instruction
following. It is one random-initialization lineage after validation exploration;
independent seed replication is still needed.

The exact checkpoint has 249,952 inference parameters:

    runs/rewt_learning_20261010/binding/aligned/best.pt

Checkpoint SHA-256:

    96d648b633c27ee4440f41556e94f63cea760b1839dff9ef8515808df9d2b012

[Locked plan](onehop_test/plan.json), [measurements](onehop_test/results.json),
[all primary generated responses](onehop_test/rounds_2_outputs.jsonl),
[zero-round responses](onehop_test/rounds_0_outputs.jsonl).

All 23 failures in the familiar formats were syntactically valid: seven wrong
arithmetic responses across five number pairs, and sixteen wrong rule responses.
Two rule responses had the correct trace but a different final answer. Recall
had no failures in this test. The checker requires the complete trace and final
answer, rather than counting correctly formatted guesses.
[Complete failure analysis](onehop_test/error_analysis.json).

## Run the actual model

From the repository root, with the existing Python environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/binding/aligned/best.pt \
  --prompt $'A=19;B=A+23;B=?\n' --greedy --stop-newline --max-new 40
```

The checked command generated:

```text
steps=42;ans=42
```

The sampler also accepts `--prompt-file`, `--rounds`, `--temperature`, `--seed`,
and `--json`. It supports original byte checkpoints and the training-only learned
story vocabularies. It never receives a task payload, expected answer, symbolic
solver, numeric regression head or reference continuation.

[Reproduction instructions](REPRODUCING.md) include actual sampler executions,
the selected training lineage and the locked evaluation programs.

## What changed

The earlier model learned response formatting much more readily than answer
content. A causal three-token encoder gives memory keys a local role, and a GRU
proposal retains query information in the recurrent energy state. Compatibility
constraints can now operate relative to the input-conditioned anchor. Fourier
waves can likewise use displacement from that anchor. These are optional
settings; archived default checkpoints retain their original predictions.

Learning uses weighted response tokens and training-only supervision for fact
attention, observed-number representations, running sums and source symbols.
The final continuation adds query/key alignment for observed rule symbols and
more training graphs. Ground-truth numbers and rules occur only in dataset
construction, training annotations and post-generation checking. The selected
checkpoint's auxiliary heads add 12,738 training parameters and are absent
from the inference model.

The selected lineage uses 512 foundation updates, then 1,024 grounding updates,
1,024 operand-copy updates, 1,024 linear-head updates, 2,048 precision updates and
896 binding updates. These are sequential continuations, not six independent
random-seed replications. Failed and alternative validation trials remain
archived, with their budgets and hashes; this is not a compute-matched advantage
claim over a Transformer.

The [completed-fit ledger](completed_fit_ledger.json) records selected
checkpoints, active parameter counts, seeds, fit budgets and measured CPU costs
for the completed alternatives. Continuation stages are not independent seeds.
The ledger separates training from language validation where those timers are
available; it excludes data preparation, evaluation and interrupted harness
runs, so its sum is not total project wall time.

## Energy and stability evidence

The analytic energy gradient matches autograd. Causality, chunked continuation,
memory rollover and centered output geometry are tested. Six archived core,
neural-energy and full checkpoints retained bitwise-equal predictions under
their original defaults.

All 88 current REWT implementation tests pass. The
[archived-checkpoint parity check](legacy_parity_relative_waves.json) records
the exact hashes and zero prediction differences for the six old checkpoints.

On an audited 152 input positions, all accepted updates decreased the complete
energy. The mean two-round energy drop was 0.1654. State-dictionary hashes before
and after fresh generation and profiling were identical. A two-row, 32-position
forward profile measured 0.585 CPU milliseconds per input byte; this is a local
single-thread CPU measurement. Operator allocation totals in the JSON are not
peak resident memory.

The separate validation refinement sweep found 0/1/2/4/8-round complete
accuracies of 62.7/81.6/88.3/81.9/73.1% on the preceding precision checkpoint.
More energy descent is not a guarantee of a better answer. Two rounds were
chosen before the fresh test.

Early high-rate memory fits suffered severe gradient growth and lost their
learned behavior. Paired lower-rate continuations stabilized both old and new
wave settings, so the repair cannot be attributed to wave coordinates alone.
The full support-module model has not been revalidated by this learning study.

## Fresh language test

The best Fourier language configuration was selected on validation data: an
80-dimensional recurrent state and a training-learned 256-piece byte-pair
vocabulary, with 250,126 inference parameters. It starts from random weights,
uses ordinary next-token cross entropy, and has a separate checkpoint from
the synthetic-task model. The comparison arms share the vocabulary, batches,
128-token context and 16,777,216 target-token budget. Each saw 32,362,917 target
raw bytes over 4,096 updates.

| Model | Fresh test bits / raw byte | Validation bits / raw byte | Total training CPU seconds |
|---|---:|---:|---:|
| Fourier energy, width 80 | 1.4476 | 1.4062 | 3,953.6 |
| Neural energy, width 80 | 1.4491 | 1.4054 | 3,865.6 |
| Transformer, two layers | 1.6911 | 1.6255 | 504.1 |

The test comprises 64 document-bounded windows: 8,192 target tokens or 15,616
raw bytes. Token accuracy is reported as token accuracy, not byte accuracy.
This is one initialization per arm and equal token budgets with unequal CPU
costs; it does not establish a general architecture advantage. Fourier and
neural energy are effectively tied here, and the Transformer comparison is
limited to this small model and common training recipe.

The paired difference between Fourier and neural energy is −0.0015 bits per
raw byte, with a document-bootstrap interval of −0.0028 to −0.0002 for this
frozen pair of checkpoints. The effect is small, validation favors neural
energy, and training-seed uncertainty is unmeasured. It does not establish a
general Fourier advantage. Individual document losses and the bootstrap method
are saved in the [uncertainty artifact](test_uncertainty.json).

All three models generated 96-token continuations for the same 30 unseen story
prefixes at temperature 0.5, using identical sampling uniforms. I read and
rated all 90 anonymous continuations before opening the model key. They contain
recognizable clauses, but broken grammar, drifting characters and disconnected
events. None qualified as a useful whole continuation under the saved rubric.
These are exploratory ratings by the implementing assistant; independent human
ratings remain pending. Better predictive loss has not delivered coherent text.
All 30 saved prefixes also decode and re-encode exactly using only the displayed
known text and the frozen vocabulary, as recorded in the
[prefix audit](language_test/prefix_audit.json).

[Locked language plan](language_test/plan.json),
[measurements and fixed-weight checks](language_test/results.json),
[all anonymous text](language_test/anonymous_texts.jsonl),
[rating rubric](language_test/rating_protocol.json),
[all ratings](language_test/assistant_ratings.jsonl),
[rating summary](language_test/rating_summary.json).

Earlier byte, whole-word and smaller-recurrent-state experiments remain
archived. Their validation decoding sweeps also produced broken prose. The
byte-pair tokenizer improves compression and handles uncommon words more
flexibly, but the study changes vocabulary, raw context volume and recurrent
capacity together; those changes are not isolated architectural ablations.

## Longer chains

Longer-chain continuations are separate from the tested one-step checkpoint.
The position-aware curriculum adds 6,144 active parameters (2.46%) and
training-only supervision for remaining trace steps. Squared numeric error
initially damaged the existing one-step behavior; bounded Huber losses reduced
that interference. Validation inspection then localized another failure:
models often generated the correct first rule result but stopped the trace
before applying the next rule.

The final continuation balances one- and two-step tasks, retains observed
symbol query/key alignment, and increases delimiter and remaining-step
supervision. It was selected at update 2,048 by the same known-wording
validation macro used for all four longer-chain studies. The new checkpoint has
256,096 inference parameters and 13,318 training-only auxiliary parameters.
It continues the selected one-step initialization lineage; it is not another
independent random-initialization replication. The added fit measured 3,260.8
CPU seconds, including its validation work.

The fresh test excludes every previously opened one-step semantic identity
and all earlier studies' test identities. It contains 960 new tasks, including
held-out wording and three/four-step transfer. All prompts and correct responses
fit the shared generation window, so truncation does not explain these failures.

| Fresh task, familiar wording | Complete correct / prompts | Accuracy | Same weights, zero rounds |
|---|---:|---:|---:|
| Two additions | 16 / 72 | 22.2% | 0.0% |
| One rule application | 46 / 53 | 86.8% | 64.2% |
| Two rule applications | 30 / 41 | 73.2% | 26.8% |
| Arithmetic, three/four additions | 0 / 140 | 0.0% | 0.0% |
| Rules, three/four applications | 0 / 111 | 0.0% | 0.0% |
| Recall, four distractors | 83 / 83 | 100% | 85.5% |
| Recall, eight distractors | 62 / 62 | 100% | 91.9% |
| Recall, sixteen distractors | 66 / 66 | 100% | 87.9% |

There are no fresh one-addition or one-distractor cases left after the identity
exclusions. The separate earlier checkpoint remains the tested choice for
one-addition arithmetic. The recall format places `K` first, followed by `X`
distractors; this is not a test of arbitrary memory positions or unlimited delay.
The new result establishes a useful bounded two-rule capability and longer
recall, while multi-step arithmetic remains poor. Three/four-step generalization
fails. Withheld wording gives 2/29 complete two-rule traces, 0/108 arithmetic
traces and 64/109 recall responses: instruction transfer also remains weak.

The model generated both successful and failed traces. For example:

```text
Prompt:    walk E 2:L>U;T>D;E>L;S>T=?
Generated: steps=L,U;ans=U
Expected:  steps=L,U;ans=U

Prompt:    A=7;B=A+20;C=B+8;C=?
Generated: steps=27,30;ans=30
Expected:  steps=27,35;ans=35
```

Every audited accepted energy update decreased the same complete energy over
352 input positions; mean two-round drop was 0.2571. Before/after model hashes
and the checkpoint file hash are identical. This energy check validates the
inference procedure; it does not certify the answer's correctness.

Checkpoint:

    runs/rewt_learning_20261010/multistep_balanced/progress/best.pt

SHA-256:

    917f367430ca4dcf4f9b8c8a7ced5f9d0e38cd62d20fe953b430cbfeed446737

[Locked longer-chain selection and test plan](multistep_test/plan.json),
[all measurements and fixed-weight checks](multistep_test/results.json),
[all primary generations](multistep_test/rounds_2_outputs.jsonl),
[zero-round generations](multistep_test/rounds_0_outputs.jsonl),
[execution-plan clarification](multistep_balanced/METADATA_CLARIFICATION.json).

TinyStories text is pinned to publisher revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Its source card and license are retained
with the local data. Training, validation and test use disjoint document
identities; batches stay within individual stories. The 256-piece word and
byte-pair vocabularies are learned only from the training split and exactly
reverse ASCII text. They change token volume and context measured in raw bytes,
so their results must not be presented as a paired architectural improvement
over the previous byte-only enwik8 study.

[Publisher dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
[dataset paper](https://arxiv.org/abs/2305.07759).
