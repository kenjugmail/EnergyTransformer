# Research priorities for EnergyTransformer after v16
Prepared September 12, 2026. These recommendations distinguish measured local results, primary-source claims, and proposed future experiments.

## 1. Fix the information interface before adding a harder optimizer
The protected P/S store can retain correct categorical edits. Its earlier failure was often deciding what a sentence authorizes, not forgetting a correctly parsed fact. A confidence threshold is not evidence of semantic correctness on new wording. V17 therefore separates lexical binding from operation interpretation and tests broader compositions and negative-scope examples. The parser is stateless across records; only an explicitly approved edit can reach the existing ledger.

This is informed by COGS, which distinguishes random-split performance from new word/structure combinations, and by semantic-tagging and compositional-parser research. These works motivate making semantic structure explicit. V17 is a much narrower synthetic task, not a COGS reproduction or a new state-of-the-art parser.

Primary sources:
- Kim and Linzen, COGS (EMNLP 2020): https://aclanthology.org/2020.emnlp-main.731/
- Zheng and Lapata, Compositional Generalization via Semantic Tagging (Findings EMNLP 2021): https://arxiv.org/abs/2010.11818
- Compositional generalization with a broad-coverage semantic parser (2022): https://aclanthology.org/2022.starsem-1.4/

## 2. Treat writing, retention, and reading as different design choices
Test-time regression formalizes associative sequence memory as a regression problem. MIRAS separately identifies the memory architecture, inner objective, retention mechanism, and learning algorithm. This argues against bundling another feature map, another cache, and another optimizer into one experiment and crediting the complete gain to 'energy.'

For this project, the next LM-scale matrix should isolate:
1. Current diagonal P/S memory with the same two reads.
2. A block-quadratic reader with correlated keys.
3. Content-addressed erasure with separate write strength.
4. A matched local-attention and modern gated-delta baseline.

Keep the tokenizer, training examples, width/parameter accounting, and tuning budget comparable. Then show accuracy against total training FLOPs, wall time, recurrent bytes, and decoding latency, not just an update-count win. No such large-model comparison is executed in v17.

Primary sources:
- Wang, Shi, Fox, Test-time regression: https://arxiv.org/abs/2501.12352
- Behrouz et al., MIRAS: https://arxiv.org/abs/2504.13173
- Google Research overview: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/

## 3. Separate WHERE to erase from WHERE to write
Gated DeltaNet-2 (May 2026) separates channel-wise erase and write gates. Erase-then-Delta Attention (June 2026) additionally separates the erasure address from the write address. Their reported language-model results are external findings, not results reproduced here.

This distinction matters when a new association should replace stale information at a different location, rather than simply reduce all historical precision. V17 adds a separate experimental quadratic primitive to preserve an explicit energy interpretation:

Let P >= lambda I, S be information coefficients, e a unit erasure direction, and A = I - alpha e e^T, with 0 <= alpha <= 1. A new factor uses key k, value v and beta >= 0:

    P_new = lambda I + A^T (P - lambda I) A + beta k k^T
    S_new = A^T S + beta k v^T

The erasure direction e and the write key k are independent. The prior remains positive. With alpha=beta=0, the update is the identity. This is a congruence transformation of the non-prior part of the old quadratic, plus a new squared-error factor. It is not the EDA recurrence and is not a claim of novel mathematics.

The code passes derivative and energy-identity tests and maintains its prior floor through 100 random edits. It has NOT been trained as a language layer, connected to the deployed fast recurrence, or benchmarked against EDA/Gated DeltaNet-2. It stores a dense key-key precision matrix, so storage and solve cost grow. General collateral-memory preservation is NOT guaranteed: erasing a coupled direction can alter other inferred modes. The test asserting an unaffected query is restricted to an orthogonal diagonal example.

Primary sources:
- Gated DeltaNet-2: https://arxiv.org/abs/2605.22791
- Erase-then-Delta Attention: https://arxiv.org/abs/2606.26560

## 4. Train on delayed usefulness, not merely immediate token loss
A proposed next objective is

    L = L_next_token + lambda L_delayed_query + mu L_edit_locality.

L_delayed_query should supervise a held-out query after many distractors, revisions and deletions. L_edit_locality should test untouched associations under controlled edits. These objectives are proposals; no end-to-end memory-conditioned LM was trained this turn.

Use unfamiliar entities and value strings, explicit aliases, multiple mentions, negation, corrections, conflicting sources, and refusal to act on quoted instructions. Do not grade only a supplied record interface: evaluate extraction, memory editing, and answering separately, then end-to-end. The language parser must not receive the correct operation during inference.

RULER extends simple needle retrieval to more demanding multi-key, multi-hop and aggregation tests. ATLAS separately argues for improving memory capacity and optimization using more than the last token. These are useful comparison points, not evidence that this project's tiny model already has those capabilities.

Primary sources:
- RULER: https://arxiv.org/abs/2404.06654
- ATLAS: https://arxiv.org/abs/2505.23735

## 5. Measure the edit's effect, not just its confidence
Use paired histories: hold all text fixed and change one source fact. Inspect the resulting P/S differences, query responses, and output probabilities. Then replace one write while holding its norm fixed. Such interventions help distinguish stored information from correlations in the current input.

WriteSAE studies sparse decompositions of recurrent matrix writes and interventions at that site. Its claimed successes are specific to its models/tasks. No SAE has been trained here. A future small probe should begin with known synthetic edits and matched-norm controls, rather than attach an interpretability label to arbitrary memory directions.

Primary source: https://arxiv.org/abs/2605.12770

## 6. Reserve basin escapes for an actual optimization bottleneck
The deployed fast read P_i > 0 has a unique quadratic minimum. It does not need a saddle-escape routine. Finite Fourier modes can define nonlinear read energies, but the previous local results did not show that their multiscale optimization improved language prediction. A coordinate transform of an unchanged function is not a mechanism for removing its minima.

Energy Transformer (Hoover et al., NeurIPS 2023) explicitly designs updates around an energy; it is separate prior art, not this repository. Retain that distinction in titles and comparisons. A future nonlinear intervention must show task accuracy as well as objective reduction and include a no-intervention and ordinary-restart control.

Primary source: https://arxiv.org/abs/2302.07253

## 7. Risk and coverage must be reported together
Selective classification trades acceptance coverage against error among accepted predictions. A model that refuses every difficult sentence is not a better semantic parser. V17 therefore retains raw exact interpretation accuracy, acceptance coverage, wrong-edit proposals, and true-edit acceptance separately. Its threshold is a heuristic, not a calibrated OOD risk guarantee.

Even a small per-record error can matter over long use. Under independent error probability p, at least one error in n trials has probability 1-(1-p)^n; with only an upper bound epsilon per event, the union bound is n*epsilon, capped at one. Neither independence nor a uniform error bound is established by the synthetic test. Review and external authorization remain necessary.

Primary source: https://arxiv.org/abs/1705.08500

## Priority order
For the bounded text interface, next test an actual span/role model and compositional negation scope on independent human-written examples, not more versions of the same templates. For the original language backbone, next train the delayed-query objective with a gated-delta baseline and count resources. The energy-native congruence update remains an isolated mathematical component until a matched task shows a benefit. Do not replace current checkpoints merely because a new mechanism sounds more advanced.
