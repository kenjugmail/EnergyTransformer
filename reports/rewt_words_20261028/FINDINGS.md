# Whole-word REWT — October 28, 2026

The custom language-model path now supports **whole-word prediction**. The first
trained streaming model has **1,399,096 parameters** and an **8,192-entry**
vocabulary: 8,173 word/number entries, 16 punctuation entries, and three special
tokens. A known word is one output decision; it is not emitted byte by byte or
assembled from subword pieces.

The original joint-energy model, its memory/architecture variants and Transformer
controls also accept configurable vocabulary sizes. Their embedding, output,
energy-state and cached-decoding dimensions follow `vocab_size`. Old checkpoints
without that setting retain their 256-entry vocabulary. This change does not
replace the pretrained LFM3 tokenizer or its embeddings.

## Use the trained word model

From the repository root:

```bash
.venv/bin/python scripts/predict_rewt_words.py --prompt 'Once upon a' --top 5
.venv/bin/python scripts/predict_rewt_words.py \
  --prompt 'The little girl went to the' --top 5 --generate 24
```

The default checkpoint is `runs/rewt_words_20261028/best.pt`. Its complete
vocabulary is embedded in the checkpoint; inference does not need the training
corpus or a tokenizer file at its original path. `--generate` counts word and
punctuation tokens; `--top` lists complete words at the immediate next position.
Candidate probabilities use the full output distribution, including punctuation
and special tokens, so the displayed word probabilities need not sum to one.

The existing streaming sampler also recognizes this checkpoint:

```bash
.venv/bin/python scripts/sample_rewt_streaming.py \
  --checkpoint runs/rewt_words_20261028/best.pt \
  --prompt 'Once upon a' --tokens 24 --decoder eager
```

## Held-out story results

The selected checkpoint is update 1,200, chosen by the lowest validation token
NLL among six checkpoints. Training starts from random weights, uses two energy
refinement rounds, and processes **1,228,800 token exposures**. Evaluation uses
96 document-contained crops of 64 target tokens: 6,144 token positions, including
5,218 word positions on the test sample.

| Measure | Result |
|---|---:|
| Next-word accuracy | 28.36% |
| Next-word top-5 accuracy | 53.97% |
| Known-word accuracy | 28.52% |
| Vocabulary coverage of evaluated word positions | 99.46% |
| Token NLL, natural-log units | 3.3824 |
| Token perplexity | 29.44 |
| Train-only unigram token perplexity | 384.15 |

Word accuracy tests the model's **unfiltered full-vocabulary prediction at word
positions**: a punctuation prediction there is an error, and every unknown word
is an error even if the model correctly predicts the `<unk>` label. The CLI's
candidate display removes punctuation and special tokens. Token NLL/perplexity
includes punctuation, EOS and `<unk>` targets. These metrics are not directly
comparable to previous bits-per-byte results.

This is a single exploratory fit on an existing TinyStories subset, whose
holdouts were inspected in earlier work. It does not establish general-purpose
assistant quality or superiority over a matched word-based Transformer.

Disabling refinement in the same checkpoint yields **28.48%** next-word accuracy
and **29.97** token perplexity. Refinement lowers the token loss slightly here but
does not improve top-1 word accuracy. This is an inference ablation, not a
separately trained no-energy control.

## Example predictions

These prompts were fixed in the run plan before training. Full outputs,
probabilities and token IDs are retained in [results.json](results.json).

| Prompt | Highest-ranked word | Probability |
|---|---|---:|
| `Once upon a` | `time` | 95.28% |
| `The little girl went to the` | `park` | 29.40% |
| `The dog was very` | `sad` | 20.39% |
| `She opened the door and` | `saw` | 45.19% |
| `They lived happily ever` | `away` | 4.81% |

The final example is a failure: `after` receives only about 2.4%, and greedy
generation selects punctuation. Longer continuations can repeat or become
incoherent. The format change does not make every prediction correct.

One unedited 24-token continuation is:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and make a lot of fun.

## Data and tokenization

- Vocabulary is fitted **only on training documents**, with a frequency cap and
  deterministic tie-breaking. Validation and test text cannot extend it.
- Words retain case; contractions and hyphenated words remain single tokens.
  Punctuation has separate tokens. Whitespace is normalized when decoding.
- An unseen word becomes one `<unk>` token. There is no byte/subword fallback,
  so this closed-vocabulary model cannot spell an arbitrary new word. Prompt
  words absent from the vocabulary are reported by the prediction CLI.
- A prompt should end at a word boundary. This predicts the next whole word;
  it is not a partial-word spelling completion interface.
- Source: the pinned local `roneneldan/TinyStories` subset, revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, under CDLA-Sharing-1.0. Existing
  document identities and train/validation/test partitions are preserved.
- The vocabulary covers 99.73% of all training words, 99.60% of validation words,
  and 99.59% of test words. Exact rates and counts are in the corpus manifest.
- IDs are stored as little-endian uint32, not bytes. Every document has BOS/EOS
  markers; training crops never cross document boundaries.

## Train another word model

```bash
.venv/bin/python scripts/train_rewt_words.py \
  --output runs/rewt_words_new --report reports/rewt_words_new
```

Use `--architecture joint` for the original joint-energy model or
`--architecture transformer` / `--architecture window-transformer` for attention
controls. `--vocab-size` and `--min-count` define the training-only vocabulary;
changing them requires a new `--data` directory and fresh model weights. The
trainer refuses to overwrite existing run/report directories. Defaults use the
current 8,192-entry corpus. Models trained with different vocabularies cannot
reuse each other's embedding or output weights directly.

The original joint-energy model includes vocabulary logits in its refined
state, so larger word vocabularies increase its state and solver cost. The
streaming model refines a latent state before projecting to the vocabulary.
The optional tiled legacy Triton solver remains explicitly limited to a
256-dimensional output; use the general PyTorch path for word-sized joint models.

## Verification and provenance

- **305 REWT tests pass**; [test output](tests.txt) and [verification record](completed_verification.json).
- The actual 8,192-entry trained model's CUDA-graph decoder reproduces all 12
  checked reference continuation token IDs exactly: [cache audit](cached_decode_audit.json).
- [Run plan and exact training source hashes](plan.json).
- [Training/validation log](metrics.jsonl) and [complete results](results.json).
- [Vocabulary](vocabulary.json) and [corpus identities and coverage](corpus_manifest.json).
- Exact training sources are retained under `source/`. Later loader validation
  and documentation edits are separate from that archived training snapshot.
- The new tests cover intact words/OOV behavior, IDs above 255, uint32 corpus
  storage, train-only vocabulary fitting, document isolation, old checkpoint
  compatibility, new checkpoint loading, model gradients, cache parity and
  word-sized joint-energy solvers. The three training/prediction CLIs were
  exercised for streaming, original joint-energy and Transformer models.

The trained checkpoint stays in the existing ignored `runs/` directory. Source,
vocabulary, metrics and report are versioned; downloaded corpus data stays local.
