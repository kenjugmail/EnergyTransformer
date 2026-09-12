# V18: precision-aware energy-memory editing

Executed September 12, 2026. This is a numerical associative-memory study, not a new LLM or parser. The existing language checkpoint, reviewed approvals, and ledger are unchanged. The experiment download contains the longer V18_FULL_REPORT.md, all inputs, checkpoints and raw predictions.

## What was implemented

The v17 congruence erase/write operation is now batched, differentiable, and trained on fresh continuous-key episodes. Euclidean erasure uses the raw key direction. A new metric-aware eraser uses d=P^-1 k/(k^T P^-1 k), minimizing d^T P d subject to k^T d=1. It is much better in the tested correlated-key regime, but a classical covariance-preconditioned delta correction is stronger.

The delivered `projected` variant re-expresses that stronger classical model using only P/S:

    u = solve(P,k); ell = k^T u; y = u^T S
    P_new = P + beta k k^T
    S_new = S + k [beta y + alpha (beta + 1/ell)(v-y)]^T

Its temporary mode satisfies

    M_new = M + alpha [P^-1 k/(k^T P^-1 k)](v-k^T M)^T.

This is exact real-arithmetic equivalence, not superiority over the classical estimator or a newly discovered theorem. The writer computes the current response without storing the mode. P remains positive in real arithmetic. It is a metric, not calibrated factual confidence. A fresh process can resume from P/S alone.

## Training and scope

21 final fits: five recurrent families plus one softmax-retrieval family at 3 seeds, and three immediate-loss ablations. Each receives 600 updates of 24 episodes (14,400 episodes per fit). All variants at a seed see identical streams. Four 60-update pilots are separate. Three projected checkpoints are no-training conversions, not additional fits. Each recurrent model stores 756 learned scalars; the softmax comparator stores 147. Recurrent state and computation are not matched.

An episode has six newly sampled 12-dimensional keys and four-dimensional values, eight edits, and explicit HOLD distractors. Only half the keys are edited, so collateral damage is measured separately. RESET means assigning a zero vector, not UNKNOWN or semantic deletion. No integer key IDs reach the learned models. The operation flags and final query keys are supplied; no language interpretation occurs. Every method has an exact HOLD contract.

## Results

MSE, averaged across three seeds; lower is better:

| Method | Primary | Nearly collinear | 32 edits | 20 keys in 12 dimensions |
|---|---:|---:|---:|---:|
| Diagonal energy | 0.405145 | 0.934957 | 0.494232 | 0.718342 |
| Euclidean energy | 0.246108 | 0.378239 | 0.301721 | 0.515914 |
| Precision-guided eraser | 0.000732 | 0.041968 | 0.004017 | 0.639233 |
| Scalar-gated delta | 0.441476 | 1.027249 | 0.484567 | 0.785915 |
| Classical preconditioned delta | 0.000073 | 0.005291 | 0.000054 | 0.647401 |
| Projected energy | 0.000073 | 0.005291 | 0.000054 | 0.647401 |
| Small softmax retrieval | 0.053389 | 0.782929 | 0.633794 | 0.241521 |
| Exact-key table | 0 | 0 | 0 | 0 |

The numerical test has 1,344 independent episodes across seven conditions. There are 32,256 learned/converted model-episode records, not that many independent problems. Full inputs and per-query errors are retained in the download. A post-hoc fixed, no-training preconditioned control achieves 0.000824 primary MSE. Most of the advance comes from classical geometry, although learning improves this specific control.

The new projected model's primary changed/untouched MSEs are 0.000036/0.000109. The precision-eraser delayed loss improves 0.000765 immediate-trained MSE to 0.000732; delayed supervision is a small effect, not the main gain. At capacity overload, the softmax comparator wins. An exact-key table is perfect and is also smaller in raw key/value scalars at six entries. This is not universal attention superiority.

## Cost and audits

Median CPU time for a batch of 16 short episodes: projected 4.116 ms, classical preconditioned 3.834 ms, scalar delta 2.779 ms, softmax 0.137 ms. The parallel attention comparator is much faster here. No GPU, LM-token latency or FLOP-matched claim is made.

Projected P/S payload is 768 float32 bytes; diagonal 240 and scalar delta 192. Softmax returns 1,408 prefix bytes in this test and grows with sequence length. Weights, temporary workspaces and metadata are excluded.

110 full-repository tests pass. Three fresh-process projected-state restores exactly reproduce their serial continuations. Projected versus classical output disagreement is at most 9.9421e-5 in the inspected float32 test predictions. The same-prefix HOLD audit leaves four states unchanged after 4,096 extra supplied HOLD operations. This is not learned relevance detection.

## Integrity and limitations

The internal protocol followed pilots, before final tests; it was not public preregistration. An execution timeout interrupted the launcher after three completed fits; partial attempts were retained and unfinished fits restarted. Softmax controls were added before final scoring. The projected conversion and fixed control were explicitly post-hoc additions after the classical baseline proved stronger. No test labels, exclusions or model weights were changed to obtain conversion parity. The original training core is preserved in the download.

The delta reference is a scalar-gated comparator, not a complete Gated DeltaNet-2 or EDA reproduction. This suite has no natural-language decoder. Parser role/scope robustness, jointly memory-conditioned language training, block precision, adaptive overflow capacity and resource-matched modern attention remain uncompleted experiments. See RESEARCH_v18.md for primary sources and the next protocol priorities.

## Run

    pip install -r requirements-research.txt
    python scripts/demo_recall.py
    python -m pytest -q tests/test_geometric_memory.py

The demo needs the checkpoint from the download. To regenerate all final small fits and evaluations without a corpus:

    python scripts/run_v18.py

Only source and reports are committed by this change. Binary artifacts in the conversation download are not implicitly a GitHub release. No repository visibility or permission rules are changed. Architectural motivation: Kenju Tomita; implementation and experiment: this conversation. Not independently replicated or peer reviewed.
