# Research synthesis and next test priorities — v18

September 12, 2026. Primary sources checked through their official papers. External reported results are not local reproductions.

## What changed in our interpretation

The v17 separate erase/write congruence was only unit-tested. V18 trains that family and investigates correlated keys. It finds that geometry matters far more than renaming an update as energy. A conventional normalized covariance-preconditioned correction is substantially stronger than the initial new eraser. We therefore derive an algebraically equivalent P/S state representation, retaining explicit energy semantics without pretending it improves the classical estimator.

Test-time regression [1] is the central organizing reference: memory reads can be understood as regression. MIRAS [2] argues for separating the memory objective, function class, retention, and learning rule. Generalized forgetting RLS [3] provides established analysis for forgetting mechanisms and stability. The v18 constrained metric projection is classical linear algebra, not a new theorem.

## Selective edits versus global decay

Gated DeltaNet-2 [4] separates erase and write strength across channels. Erase-then-Delta Attention [5] also introduces an independently chosen erasure address. Those papers report large language-model experiments; our scalar-gated delta control is not a reproduction of their complete networks, channel gates, data, kernels or results.

Our metric direction d=P^-1 k/(k^T P^-1 k) is another distinction: it changes how an edit propagates through a correlated function representation. It minimizes d^T P d subject to k^T d=1. It does not discover a semantic entity, infer which fact is stale, or certify an unrelated fact is preserved. Full precision incurs O(d^2) storage and solves.

## Capacity must be measured rather than assumed

With six random associations in 12 dimensions, metric-aware updates work well. With twenty associations in 12 dimensions, this fixed linear read cannot express arbitrary current values. The softmax comparator, which retains the history, then outperforms the fixed-size proposal on the evaluated condition. A conventional exact-key table has zero error throughout and is also smaller than the full matrix for six associations, excluding container overhead.

Memory Caching [6] explicitly explores growing recurrent memory by retaining state checkpoints. This is a relevant next comparison, but it would reintroduce growing historical state, not maintain a fixed-size-memory claim. An honest capacity-adaptive proposal should compare sparse overflow entries, block precision, and state checkpoints under byte budgets, and include unsuccessful retrieval and interference rates. None of those capacity mechanisms is implemented in v18.

## Delayed supervision and locality

The new actual training objective is final recalled-value MSE over both edited and untouched keys. A same-architecture control is trained only on the immediately addressed value. The final-query objective provides a small benefit for the precision-congruence model; it is not a dramatic new generalization result. HOLD is a supplied operation with an exact no-op for all contestants. The long-hold test therefore checks the contract, not learned relevance detection.

RULER [7] motivates evaluating multi-key, multi-hop and aggregation rather than one planted fact. A full downstream study should train a memory-conditioned LM with next-token, delayed-query and edit-locality losses, include unknown facts and authorized deletion, and measure total training cost. V18 does not add the next-token loss or integrate these modules into the old language model.

## The text interface still needs structural tests

SLOG [8] distinguishes structural generalization from lexical substitutions. Our v17 canonical binding test is not enough to establish negation scope, roles in multi-entity sentences, or unknown-language robustness. A next parser experiment should use independently authored labels, span/role targets, and ambiguity-aware abstention. No independent human annotation set was collected in this turn, and no new parser accuracy is claimed. Approval and ledger protections remain unchanged.

## No reason to add saddle escape to these reads

All current reads use positive quadratic P. There is a unique minimizer. The nonlinear spectral experiments remain separate; no new fractal or saddle claim is justified here. Fixing an inadequate representation, selecting an edit, and solving a nonconvex optimization problem are different tasks.

## Practical experiment order

1. Test block precision and optimized Cholesky/rank-update implementations on the existing continuous recall suite, retaining the full preconditioned control. Measure prefix/step latency separately and include factorization cost.
2. Add an explicit capacity-overflow mechanism with a strict memory budget and identical training streams, accepting that history storage can grow.
3. Train a jointly memory-conditioned language model on genuinely new multi-domain data, with exact operation supervision progressively removed. Compare full attention and strong modern gated-delta implementations, not just a toy softmax read.
4. Test parser role/scope generalization with independently written sentences and report coverage, incorrect proposals, and end-to-end errors together.

These are next experiments, not completed claims. V18 completes a core module study, a conventional-control-driven redesign, state-only restart tests, and a reproducible benchmark.

## Primary references

[1] Wang, Shi, Fox. Test-time regression: a unifying framework for designing sequence models with associative memory. https://arxiv.org/abs/2501.12352

[2] Behrouz et al. It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization. https://arxiv.org/abs/2504.13173

[3] Lai and Bernstein. Generalized Forgetting Recursive Least Squares: Stability and Robustness Guarantees. https://arxiv.org/abs/2308.04259

[4] Hatamizadeh, Choi, Kautz. Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention. May 21, 2026. https://arxiv.org/abs/2605.22791

[5] Li et al. Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention. June 25, 2026. https://arxiv.org/abs/2606.26560

[6] Behrouz et al. Memory Caching: RNNs with Growing Memory. February 27, 2026. https://arxiv.org/abs/2602.24281

[7] Hsieh et al. RULER: What's the Real Context Size of Your Long-Context Language Models? https://arxiv.org/abs/2404.06654

[8] Li et al. SLOG: A Structural Generalization Benchmark for Semantic Parsing. https://arxiv.org/abs/2310.15040
