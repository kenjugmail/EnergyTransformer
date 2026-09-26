# Forecast repair: separate point skill from future-density shape

2026-09-26. Continuation of draft PR #1 at branch commit
`8576347b66fad388f253991501f38ee9fe463c19`. No backbone, runtime kernel,
protected-memory permission, production checkpoint or main-branch code is changed.

## Executive result

Three newly trained, mean-anchored spectral mixture heads improve held-out MSE
from 0.897245 to 0.871523 (2.867% reduction) and joint NLL from 1.214732 to
1.201045 nats per standardized forecast coordinate. Stronger, horizon-specific
ridge regularization accounts for the point improvement. The identical means in
a Gaussian regression control achieve the same MSE. Learned mixture shape improves
NLL over that Gaussian (1.234638), but their Monte Carlo energy scores do not differ
conclusively. This is development evidence, not a language-model improvement.

The +64-byte mean no longer extrapolates an unsupported signal: it returns the
training-target mean. This fixes the earlier loss relative to that baseline by
falling back, NOT by demonstrating useful +64 context-dependent point prediction.
Near-horizon (+1) MSE is slightly worse than the original spectral head; +4/+16
improve. No decoder skips, generated facts, or forecasted P/S updates are authorized.

## Diagnosis and changes

1. The original model initialized every component at persistence and asked NLL
   optimization to learn mean, scale, mixture shape and correlation together.
   NLL-selected means were worse than ridge under squared error.
2. The old ridge grid (0.1,1,10,100) used a shared penalty. Longer horizons need
   substantially stronger shrinkage in the present training traces.
3. The new mean uses five-fold document-group crossfit within TRAIN only. Separate
   penalties are selected per horizon from 0.1 through 1,000,000. The selected
   penalties are 100,1,000,10,000,1,000,000 for +1,+4,+16,+64. Out-of-fold gains toward
   the training-target mean are 0.991119,1,1,0. Gains are enabled only if their
   fixed-predictor advantage over the constant has a positive lower bound in a
   document-bootstrap check on the validation-selection half. The +64 gain is zero.
4. A mean-centered residual mixture is trained by exact NLL. The mean cannot drift
   away from the linear anchor when the distribution-shape parameters change.
5. Four positive temperatures are fitted on the other validation half, never the
   final test. Temperatures rescale both between-component offsets and within-component
   scales. Prediction intervals widen about 4.775% on average, improving empirical
   marginal 90% coverage from 88.60% to 90.34%. This is not joint-trajectory coverage,
   a conformal guarantee, factual confidence, or a robust OOD guarantee.
6. Existing checkpoint loading supports the new format; ForecastSession reports
   per-horizon point-mean support and rejects inconsistent snapshots. It still
   preserves the original LM logits and the observed P/S.

## Exact energy construction

Let m(C) be the gated linear mean, logits define mixture probabilities pi_j, and
raw component offsets r_j(C) be centered as

    delta_j = r_j - sum_k pi_k r_k.

Thus sum_j pi_j delta_j = 0. Component means are m + delta_j, with the existing
learned correlated Gaussian factors. The complete function is

    E(Y|C) = -log sum_j pi_j Normal(Y; m(C)+delta_j(C), Sigma_j(C)).

A positive diagonal temperature D gives

    mu'_j = m + D delta_j,
    Sigma'_j = D Sigma_j D.

The distribution remains normalized and E[Y|C]=m. For a source density with mean
m0, the exact change-of-variables identity is

    E_new(y) = E_old(m0 + D^-1(y-m)) + log det D.

Tests compare energies, samples and gradients under this identity. This is ordinary
conditional-mixture and affine-density mathematics, not a new energy law. A mixture
can remain multi-basin even with a fixed mean; a two-basin numerical test verifies
that property. No fractal-dynamics result is claimed. MAP and conditional-mean
predictions remain different decisions; the default point estimate is the mean.

## Protocol and fairness

The underlying archived teacher is unchanged: 972,896 parameters, SHA256
`78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`.
The actual GB10 10.6M checkpoint is not present locally and was not evaluated.
There are sixteen past sketches, eight variables and four joint horizons. They
remain lossy summaries, not complete states or semantic plans.

This reuses the parent study's 4,617 training windows (231 document groups),
600 validation windows (30 groups), and 634 final windows (32 groups). Validation
is deterministically partitioned into 15 selection and 15 calibration documents.
The parent heads historically selected checkpoints using all 30 validation groups;
postcalibration of those reused heads is therefore not wholly independent of their
original selection. The three NEW fits use the selection half only for selection.
Normalization remains the unchanged parent TRAIN-history normalization.

Crossfit produces group-held-out linear predictions, not nested cross-validation:
penalty and gain selection share the training folds. Final linear weights are fitted
on all training documents. Training residual densities see out-of-fold anchor means,
whereas inference uses the refitted linear mean. These are choices, not claims of
an unbiased generalization estimate. The bootstrap gate is an empirical heuristic,
not simultaneous risk control over all selected horizons.

Before the new final evaluation, all fit/calibration checkpoint hashes were frozen.
No new head was fitted or selected after seeing these final outcomes. The final
set WAS already examined in the parent project, so it is not a pristine confirmation.

Executed new work: 3 neural fits x600 AdamW updates x64 windows =115,200 presented
windows, paired across seeds with the old head streams (verified by batch hashes).
That is 38,400 repeated presentations per fit, not 115,200 independent documents.
Six four-temperature calibrations (150 steps each) cover the three recentered
old heads and three new heads. Existing 18 parent fits are retained, not counted
as new training. All new best/final optimizer snapshots and calibration traces
are retained. The new fit script starts fresh; no new exact-resume claim is made.

## Held-out results

Means over three seeds for neural variants; all share the same test documents.
Lower is better for MSE, NLL and the energy score. The energy score is a proper
multivariate distribution score, NOT the internal function E or a reasoning score.
It is estimated using 128 pairs of independent draws, normalized by sqrt(32).

| Model | MSE | NLL/coordinate | Energy score | Marginal 90% coverage |
|---|---:|---:|---:|---:|
| Original spectral mixture | 0.897245 | 1.214732 | 0.661181 | 88.60% |
| Original ridge | 0.880202 | not scored | not scored | not scored |
| Horizon/gated ridge | 0.871523 | not scored | not scored | not scored |
| Horizon ridge + correlated Gaussian | 0.871523 | 1.234638 | 0.650891 | 90.72% |
| Recentered old mixture, calibrated, no neural retraining | 0.871523 | 1.219629 | 0.650927 | 89.95% |
| New anchored mixture, before scale calibration | 0.871523 | 1.204646 | 0.651334 | 88.97% |
| New anchored mixture, calibrated | 0.871523 | 1.201045 | 0.650993 | 90.34% |

Simple recentering improves point accuracy but worsens NLL relative to the original
model. Retraining the density shape is the component that improves likelihood.
The correlated Gaussian uses out-of-fold residual covariance, with shrinkage selected
from 0,.1,.5,1 using selection documents. It is a strong classical distribution control,
not a deliberately diagonal weak comparator.

| Horizon | Original spectral MSE | New mean MSE | Constant MSE |
|---|---:|---:|---:|
| +1 | 0.557219 | 0.563106 | 1.027827 |
| +4 | 0.867025 | 0.852347 | 0.988719 |
| +16 | 1.055178 | 1.017778 | 1.049338 |
| +64 | 1.109558 | 1.052860 | 1.052860 |

Paired document-cluster intervals, averaging the three seeds:
- New minus old spectral MSE: -0.025722, 95% interval [-0.029952,-0.021687].
- NLL change: -0.013687 nats/coordinate, interval [-0.020766,-0.006786].
- Energy-score change: -0.010189, interval [-0.011885,-0.008586].
- New minus Gaussian energy score: +0.000102, interval [-0.000221,+0.000451].
Tiny differences (~1e-9) between mathematically equal means are rounding, NOT evidence
of an accuracy advantage. Monte Carlo score intervals do not separately quantify
all sampling error. The 20 evaluated configurations produce 12,680 context-model
records; shared windows and neural seeds are not independent new observations.

## Engineering checks and scope

43 focused tests pass, including 27 inherited checks. They check normalized affine
density transformations, candidate/temperature gradients, shared-component sampling,
retained multimodality, invariant conditional means, zero-gain behavior, disjoint
validation groups, Gaussian quantiles, checkpoint identities, matched batch streams,
nonmutation and unchanged teacher outputs. A fresh process restores the calibrated
session and reproduces continuation logits and forecast energy bit for bit in the
recorded CPU environment. Malformed empty/nonempty snapshots are rejected before
mutating live state.

The new head has 43,683 neural parameters, identical to the original spectral head,
plus 4,168 stored fitted mean/calibration scalars. It adds no recurrent history beyond
the existing 512-byte summary buffer. Dense covariance matrices and temporary
workspaces still cost memory. No CUDA, Spark job, new language fit, token speedup,
10.6M generalization, or pretraining-readiness improvement was measured.

## Primary sources and interpretation

- Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation (2007):
  https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  Motivates separate probabilistic scores, including energy score. This implementation
  uses exact mixture NLL and Monte Carlo energy-score evaluation.
- Kuleshov et al., Accurate Uncertainties for Deep Learning Using Calibrated Regression (2018):
  https://proceedings.mlr.press/v80/kuleshov18a.html
  Calibration is established research. Our four-temperature affine calibration is
  not their algorithm and does not inherit its asymptotic guarantees.
- Zeng et al., Are Transformers Effective for Time Series Forecasting? (2022):
  https://arxiv.org/abs/2205.13504
  Motivates keeping strong simple linear controls. No external benchmark was reproduced.

The fix is deliberately statistical: preserve useful short-horizon predictions,
stop forcing unsupported long-horizon point extrapolation, and train an uncertainty
model whose mean agrees with the chosen point forecaster. It does not make future
observations available or turn forecasts into factual evidence.
