# Architecture: function-state language research and the v14 text bridge

This document describes the historical v14 prototype. The current native model,
fact readers, graph programs, and local-training option are documented in
[the integrated architecture](reports/hybrid_integration/ARCHITECTURE.md).
The mixed energy/local-attention layer schedule and bounded cache are documented
in [the native mixer extension](reports/competitive/ARCHITECTURE.md).
The recurrence-preserving CPU kernel changes and optional preallocated full-
attention cache are measured in [the execution follow-up](reports/stronger_comparison/REPORT.md).
The [compute-controlled confirmation](reports/compute_confirmation/REPORT.md)
compares the same architectures with compiled training and matched CPU budgets.

## Persistent protected state

For a key i and categorical value vector W_i, store

`E_i(W_i) = 0.5 * P_i * ||W_i||^2 - S_i^T W_i`, with `P_i >= 0.25`.

The read-time mode is `W_i = S_i/P_i`. Only P and S persist; the mode and neural features are temporary. In v14 the protected store has 16 keys x 16 values: 1,088 float32 bytes. These arrays are still numerical hidden state broadly understood.

A replacement clears the addressed non-prior energy and writes a new one-hot target. A deletion resets the row to its prior. A hold is exact identity, rather than a forgetting coefficient slightly below one. All other rows remain untouched. The state retains no observation provenance or unlimited key namespace.

## New text bridge

A training-vocabulary word tokenizer feeds a stateless sentence encoder: 40-dimensional embeddings, convolution widths 3/5/7 with 40 channels each, masked mean/max pooling, then a 96-unit SiLU layer. Three classifiers predict one of four operations, one of 16 entity IDs, and one of 16 value IDs. These labels are supervised, not discovered without labels.

A supplied lexical guard validates that the predicted entity and value explicitly occur, and rejects unsupported multi-entity edits. It does not determine the operation or solve open-world entity linking. Confidence threshold 0.90 is a heuristic. On held-out wording the model makes high-confidence false edits; the threshold is not a correctness guarantee.

For a question, a predicted key selects the current energy read. A separate trained answer head consumes the 16 mode coordinates, evidence mass, and a 16-dimensional answer-position embedding. A 64-unit MLP predicts up to 12 byte positions and EOS. It learns the 16 location words and `unknown`; it is not unrestricted generation. No answer lookup table or recurrent answer state is used at inference.

The parser and answer head total 73,741 static parameters. They are separately trained. This is a supervised pipeline, not a single end-to-end instruction-tuned LLM.

## Relationship to v13 and the inherited language runtime

The preceding v13 component required caller-supplied operation/key/value records. V14 accepts sentences and produces bounded learned memory-conditioned answers. The previous two-layer byte language model is archived unchanged. It is not automatically converted into a conversational model by this bridge, and its free-form output logits are not trained to use protected facts in this release.

The original fast language memory uses positive-prior diagonal energies, slot transport, input-dependent decay, two learned queries, and residual feed-forward blocks. That path remains useful for studying sequence prediction but can forget. The protected path uses exact no-change actions to retain supported associations.

## Invariants and limits

- Holds and queries do not modify protected coefficients in the supported grammar.
- Writes are local to the predicted row; a wrong parsed row can still damage memory.
- State snapshots bind to the checkpoint hash and contain P/S only.
- Reads depend on stored coefficients, not a saved answer or old prompt.
- New entities, coreference, unrestricted paraphrases, provenance reasoning, and arbitrary relations are not supported.
- The 16-slot capacity is supplied. Nothing makes it lossless for unlimited facts.
- The deployed energy is convex; there are no read-time saddles to escape here.

## Prior art and research positioning

Regression-based associative memory: Wang, Shi, Fox, *Test-time regression* (arXiv:2501.12352), https://arxiv.org/abs/2501.12352.

Learned memory access and answers: Sukhbaatar et al., *End-To-End Memory Networks* (arXiv:1503.08895), https://arxiv.org/abs/1503.08895.

Neither a new name for internal state nor combining these ingredients establishes novelty or superiority. V14's contribution is an executed bounded integration, with explicit negative wording-generalization results and classical controls.
