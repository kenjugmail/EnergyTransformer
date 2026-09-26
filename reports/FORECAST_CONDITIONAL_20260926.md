# Ordered square-root forecast conditioning and a rejected learning update

Prepared September 26, 2026. Base branch commit: `3f6384a69454b1c4a0b90441b936a173596b881d`.

## Decision

Ship the exact ordered-coordinate conditioning improvement and retain the existing
calibrated forecasting heads and conservative online gate. Nine new warm-training
runs all selected their starting weights on validation. No new head is promoted.
This is a numerical/runtime improvement plus a controlled negative learning result,
not a new language-quality, long-horizon, or pretraining-readiness claim.

The teacher remains the original 972,896-parameter checkpoint (SHA256
`78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`).
No 10.6M checkpoint, CUDA device, DGX Spark job, paid service or new LM fit was used.
The working language model, permissioned memory, and main branch are not changed.

## 1. A better exact conditioning algorithm

The forecast is the existing normalized conditional Gaussian mixture

    p(Y | C) = sum_j pi_j(C) Normal(Y; mu_j(C), L_j(C)L_j(C)^T),
    E(Y | C) = -log p(Y | C).

The observed variables arrive in prefix order: the eight +1 coordinates, then the
eight +4, then the eight +16. Partition each lower triangular factor as

    L = [[L_oo, 0], [L_ro, L_rr]].

Writing Y = mu + L epsilon with standard-normal epsilon gives directly

    epsilon_o = solve_triangular(L_oo, y_o - mu_o),
    mu_r_given_o = mu_r + L_ro epsilon_o,
    L_r_given_o = L_rr,
    posterior_weight_j proportional to pi_j Normal(y_o; mu_o,j, L_oo,j L_oo,j^T).

There is no full covariance reconstruction, Schur-complement subtraction or second
Cholesky decomposition. This is standard Gaussian square-root algebra, not a new
energy law, covariance approximation, added jitter, or a changed density family.
Arbitrary/reordered observations retain the general dense path. Dispatch depends
on the requested index structure, never on observed numeric values or future data.
The density identity and gradients are checked against the general path and
PyTorch distributions. Sequential and combined observations agree numerically.

### Cancellation regression

For float32 L=[[1,0],[1,1e-4]], forming its covariance rounds the lower diagonal
from 1+1e-8 to 1. A Schur subtraction therefore returns zero variance. Reading the
conditional factor directly retains 1e-4, hence variance 1e-8. The test checks this
constructed case; it is not evidence of a previously observed failure on every
trained forecast. Other operations such as arbitrary marginals may still require
factorization and can be ill-conditioned. No universal numerical guarantee follows.

### Timings

One-thread CPU, float64, the same frozen seed11 head and observations. Eleven
randomized interleaved repetitions, 20 calls per measurement, after other workers
finished. The general path receives reversed observation order so it cannot use
the prefix specialization; this preserves exactly the same information set.

| Batch | Observed coordinates | General dense, ms/call | Ordered, ms/call | Ratio |
|---|---:|---:|---:|---:|
| 1 | 8 | 0.6697 | 0.3237 | 2.07x |
| 1 | 16 | 0.6557 | 0.3202 | 2.05x |
| 1 | 24 | 0.6511 | 0.3375 | 1.93x |
| 64 | 8 | 5.5632 | 1.5971 | 3.48x |
| 64 | 16 | 4.2221 | 1.2748 | 3.31x |
| 64 | 24 | 4.2424 | 0.9724 | 4.36x |

The maximum mean difference across these checks is 1.11e-15, covariance difference
1.78e-15, and log-evidence difference 5.69e-14. These are float64 numerical checks,
not bitwise equivalence across algorithms/devices. Times exclude the teacher,
forecast head, prior marginal, pool, quantiles and sampling. This is NOT a doubling
of language-generation speed. Persistent state and model parameter counts do not
change. Temporary views do not imply new lossless memory compression.

## 2. Test the training objective instead of adding another gate

Let N_0 be joint NLL per coordinate and N_s the remaining-trajectory conditional
NLL after the first s complete horizons become observed. Let M_s be the corresponding
conditional-mean squared error. Three arms start from the identical per-seed
anchored/calibrated parent:

- joint: N_0 (ordinary continued likelihood training).
- conditional: 0.5 N_0 + 0.5 mean_s N_s.
- conditional_mse: the conditional objective + 0.25 mean_s M_s.

All include the same small correlation regularizer. The original mean anchor and
calibration buffers stay fixed. Training uses the inherited document-crossfit mean;
inference uses the refitted anchor. This is a warm-start study, not retraining an
entire model or inventing a new conditional-density estimator.

Future coordinates here are supervised targets. A conditional loss reveals only
its declared prefix. The online API still receives evidence by executing actual
input bytes; no guessed future is committed as an observation.

Three seeds and three arms give nine completed fits, each 350 updates of batch64:
3,150 updates and 201,600 presented training windows, repeatedly sampled from 4,617
windows in 231 documents. All arms at each seed have identical cumulative sampled
batch hashes. AdamW peak LR 2e-4, cosine decay, clipping1. No tuning grid was run.

All arms share checkpoint selection: 0.5 N_0 + 0.5 mean_s N_s on the same original
15 validation-selection documents, with step zero eligible. Validation is performed
only every50 steps and at the planned final step. The other15 validation documents
were used for the parent's calibration, not for fitting new weights here. This
inherits previous calibration/selection exposure and is not an unbiased new study.

**Every one of the nine runs selected step zero.** Actual parameter equality to
its parent is checked, not inferred from equal displayed scores. No candidate
neural update is installed. The serialized initial FROZEN file uses a first-entry
tie-break to name `joint`; DECISION.json explicitly records that all three families
tie and no trained family wins. Full final/optimizer checkpoints and all validation
histories are retained rather than discarding the failed updates.

The trainer saves optimizer, generator/global RNG, sampled batch hash, best model,
selection state, and logs. A focused fresh-process stop/resume test matches weights,
optimizer, RNG and selection. CPU behavior is tested; the script is explicitly
CPU-only and is not a validated GPU training harness.

## 3. A separate document check

The old final set has 634 origins from32 documents and has been inspected repeatedly.
An additional check uses the separately installed Gensim `lee.cor` file:50 documents,
975 forecast origins. All selected hashes were frozen BEFORE extracting those traces.
Normalization, anchors, temperatures and gate values are unchanged. No weights or
gates are fit on the separate set.

An exact64-byte passage audit against the ENTIRE original `lee_background.cor`
rejects zero of these50 documents. Source-file identities, document hashes and
numeric traces are retained. No raw text is redistributed. This is the same broad
news domain and this separate corpus appeared in older language-model experiments;
it is new to this forecasting evaluation, NOT a pristine cross-domain benchmark.
Shorter passages, event overlap and semantic overlap are not ruled out.

### Initial predictions

| Dataset | Retained parent MSE | Parent NLL | Conditional-final NLL | Conditional+MSE-final NLL |
|---|---:|---:|---:|---:|
| Reused development | 0.871523 | 1.201045 | 1.206008 | 1.206384 |
| Separate news | 0.820232 | 1.185642 | 1.189439 | 1.189865 |

All means remain mathematically identical because the mean anchor is fixed.
Different losses here change mixture shapes/covariance, not the initial conditional
mean. Tiny numerical MSE differences do not establish a new prediction gain.

### After actual +1/+4 observations on separate news

| Available evidence | Method | Remaining MSE | NLL/coordinate |
|---|---|---:|---:|
| +1 | Existing raw condition | 0.916632 | 1.294501 |
| +1 | Existing conservative gate | 0.911611 | 1.288111 |
| +1 | Final conditional+MSE weights, NOT selected | 0.914920 | 1.297324 |
| +1 | Classical correlated Gaussian | 0.904694 | 1.303148 |
| +1,+4 | Existing raw condition | 0.966533 | 1.355462 |
| +1,+4 | Existing conservative gate | 0.951941 | 1.335827 |
| +1,+4 | Final conditional+MSE weights, NOT selected | 0.965438 | 1.358416 |
| +1,+4 | Classical correlated Gaussian | 0.951271 | 1.351909 |

The final unselected weights can modestly improve a conditional point estimate,
but worsen likelihood and coverage. They are diagnostic endpoints, not a model
selected after inspecting these results. The Gaussian still provides a strong
point baseline; the mixture remains useful as a probabilistic model, not a universal
estimator winner. The original +64-only gate remains zero; no longer-horizon skill
has been repaired or newly claimed.

Evaluation retains 200 aggregate model/stage/dataset records and per-origin values.
They reuse1,609 underlying origins from82 document groups. Model seeds, selected
checkpoints identical to their parents, and multiple stages are not independent
new data. Paired intervals cluster by complete document after averaging seeds.
They do not quantify historical selection bias or future distribution shift.

## 4. Checks and preserved amendments

The complete focused suite passes89 tests:71 inherited plus18 new. It checks
ordered/dense density agreement, independent likelihood, gradient flow, cancellation,
sequential conditioning, no mutation, invalid prefixes, objective target accounting,
paired sampling, non-overwrite and exact fresh-process training resume. Inherited
trained tests still verify original LM outputs and bounded online-session restoration.

An initial tool timeout interrupted one fit at a saved step100; it resumed from the
full checkpoint without extra validation opportunities. A first test invocation was
also interrupted; a later full run completed. These logs are preserved. A failed
process-launch command ran no training. No final test drove additional fitting.
The original protocol and initial deterministic tie label are retained with an
explicit decision/amendment file. There is no concealed winner selection.

## 5. Deployment and research context

Keep the old calibrated heads and gate. `GaussianMixture.condition()` uses the new
ordered path automatically only when the given coordinates are a leading prefix.
Existing raw or guarded policies preserve their mathematical definition. A new
head artifact still cannot use the old checkpoint-bound gate, even when its numeric
weights happen to match: no silent identity bypass was added.

Next worthwhile modeling evidence should use the actual10.6M checkpoint and fresh
permitted observations. Further fitting to the same small summaries is not supported
by this study. Ordinary autoregressive density models (e.g. RNADE) and modern joint
forecasting flows are important controls if the distribution family is changed;
this work reproduces neither their architectures nor their external benchmark gains.

Primary sources:
- Uria, Murray, Larochelle (2013), RNADE: https://arxiv.org/abs/1306.0186
- Yalavarthi et al. (AAAI2025), ProFITi: https://ojs.aaai.org/index.php/AAAI/article/view/35494
- Gensim corpus fixture documentation: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html

No new teacher weights, token acceleration, fractal dynamics, exact long-range
forecasting, general reasoning improvement or scaling promotion is claimed.
