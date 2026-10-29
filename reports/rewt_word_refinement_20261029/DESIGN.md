# Word prediction with supervised energy refinement

This experiment develops and tests the custom whole-word streaming REWT. It
does not change the pretrained LFM3 conversion. The primary question is whether
REWT predicts held-out words better than a standard causal Transformer at
approximately equal parameter count and equal training-token exposure.

## Architecture and refinement

Both models use an 8,192-entry train-only word vocabulary, tied embeddings,
width 128, four attention heads, two residual blocks, RMS normalization, RoPE
attention, and SwiGLU feedforward layers. REWT uses delta memory in its first
block and causal attention in its second. The Transformer uses attention in
both blocks. The 64-token training/evaluation context fits entirely within
REWT's 64-token attention window.

The feedforward widths are 256 for REWT and 288 for the Transformer. This gives
**1,401,144 parameters for the refined REWT and 1,401,472 for the Transformer**:
the control has 328 more parameters. The separately trained hybrid ablation
uses the same REWT backbone without an energy head.

Let `h` be the normalized causal backbone state and `d` its refinement offset.
The optional context-conditioned energy is

```text
A(h) = tanh(Wa h) / sqrt(number of modes)
phi(h) = phi0 + Wp h
E(d | h) = 0.5 sum_j q_j d_j² + sum_i A_i(h) cos(w_i · d + phi_i(h))
d_0 = 0
d_(k+1) = d_k - eta * gradient_d E(d_k | h)
logits_k = embedding_matrix * (h + d_k)
```

There are 16 Fourier modes. Frequency norms and diagonal precisions retain the
existing bounds. The new phase projection lets context change where the waves
are centered. It is fixed throughout each prediction's refinement trajectory.
The curvature bound is therefore unchanged:

```text
L = max(q) + sum_i abs(A_i(h)) * ||w_i||²
eta = 0.95 / L
```

In exact arithmetic this step decreases the conditional energy. **It does not
guarantee that prediction cross-entropy decreases**, either for a particular
word or on unseen data. A learned scalar energy is not a known ground-truth
prediction loss.

## Training later steps to improve predictions

The new helper runs the causal backbone once and retains differentiable latent
states along the energy trajectory. The selected recipe supervises steps 2, 4,
and 8 with the same next-token training targets:

```text
objective = mean(CE_2, CE_4, CE_8)
          + 0.1 * mean(relu(CE_later - stop_gradient(CE_earlier)))
```

The penalty is computed per token for transitions 2→4 and 4→8. Detaching the
earlier loss prevents reducing the penalty by making that earlier prediction
worse. Targets are used only during training. The model's inference API takes
causal input tokens and a fixed number of refinement steps; it cannot inspect
the next word when deciding how to refine.

Validation considers 0, 1, 2, 4, 8, 16, and 32 inference steps. Extra inference
steps reuse the same learned energy and backbone state. They do not add model
parameters or re-run the backbone. No depth is chosen from test labels.

The work is related to the broader idea of training predictions through energy
minimization in [Energy-Based Transformers are Scalable Learners and Thinkers](https://arxiv.org/abs/2507.02092).
This repository's latent Fourier head, hybrid backbone, and training objective
are a local architecture experiment, not a reproduction of that paper.

## Matched comparison and its limits

Development uses one seed and a fixed 512-document validation panel. It checks
the previous two-step model, extra final-only steps, supervision across several
steps, two Fourier mode counts, context-conditioned phases, and learning rates.
All attempted runs are retained. The longer comparison checks learning rates
0.0015, 0.003, and 0.006 for both final architectures; 0.003 wins for both.

The confirmation protocol fixes three new seeds and 12,000 updates. Every
paired run sees the same sequence of training crops: batch 16, context 64,
12,288,000 target-token exposures. Optimizer, clipping, decay, warmup and cosine
schedule are shared. Checkpoint and depth selection use only validation NLL.
The proposed default checkpoint is selected among REWT seeds by validation.

This matches parameters and data exposure, **not training FLOPs**. Trajectory
training evaluates three vocabulary projections/losses and differentiates
through eight refinement steps. The ordinary Transformer evaluates one loss.
Elapsed times are recorded on a shared GB10 and are not an isolated speed
benchmark. A loss advantage at this budget does not establish superiority at
equal compute, at larger scale, or on other tasks.

## Fresh evaluation data

The final test comes from bytes 16,777,216 through 19,447,281 of the pinned
official TinyStories validation text, revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Earlier local experiments used earlier
ranges. Incomplete edge documents are discarded. The preparation script uses
the existing normalization and excludes every document identity already indexed
in the local data directories. It reuses the exact word vocabulary without
fitting anything on test text.

The new range yields 2,967 documents. A fixed sample of 1,024 distinct documents
provides 65,536 target positions. Training and validation remain unchanged.
The corpus manifest, source range/hash, selected offsets, checkpoint hashes,
and validation-selected depths are frozen before test evaluation.

Primary NLL includes punctuation, EOS, and the unknown-word label. Known-word
NLL and whole-word accuracy are also reported. Whole-word accuracy uses the
full-vocabulary argmax at word positions; punctuation predictions and every
unknown word count as errors. Every model uses the same vocabulary and positions.

The analysis retains every confirmation seed. It compares paired document NLLs
and bootstraps both seed and document dimensions. Three seeds provide limited
information about training variability; the confidence interval is evidence
within this experiment, not a universal performance guarantee.

## Compatibility and verification

Both new configuration flags default to the historical behavior. Existing
checkpoints load without adding parameters. Context-conditioned phases are
supported by reference, prepared, and cached inference. The old specialized
Triton two-step kernel explicitly rejects context-conditioned phases; the
general eager/compiled paths support them.

Tests cover trajectory/inference agreement, causal masking, finite gradients,
energy descent, wide vocabulary IDs, checkpoint restoration, cached predictions,
and agreement between the new and existing word evaluators. The mathematical
energy-descent test is kept separate from the empirical prediction-loss test.
