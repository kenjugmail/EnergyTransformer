# Observation-driven forecast updates and conservative conditioning

September 26, 2026. Base research branch: `research/forecast-energy-20260926`, commit `4d099b9b1345581c5600c20c00e0a334097af16f`. The branch remains separate from the working language model. No neural weights, language logits, observed energy coefficients, production memory permissions or kernel implementations are changed.

## Outcome

The model can now maintain one fixed-origin forecast and revise its unobserved portions when actual nearer-horizon teacher summaries arrive. Conditioning is analytic and differentiable. A conservative, validation-selected mixture pool reduces aggregate remaining-forecast error by 0.46% after observing +1 and 0.22% after observing +1/+4. It refuses the unsupported +64-only update. Raw conditioning often worsens forecasts, and a correlated Gaussian control still has better point accuracy. This is not a language-quality, token-skipping or scaling-readiness improvement.

No new neural training occurred. Three existing anchored/calibrated forecast heads and the existing Gaussian regression control are frozen. The teacher is the original 972,896-parameter checkpoint, SHA-256 `78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`. The actual GB10 10.6M weights were not available locally and no GPU or paid job ran.

## Meaning of time and evidence

At origin t, the original model predicts a joint distribution of eight summary variables at t+1, t+4, t+16 and t+64. At t+1, only the first eight coordinates have become observations. At t+4 the first sixteen have arrived. At t+16 only the t+64 target remains, which is then 48 bytes away, NOT another 64 bytes into the future.

`ObservedForecastSession.observe(tokens)` obtains evidence by executing the frozen language model on each actual byte. There is no API for passing guessed future summaries as observed evidence. It remembers the issued function, original timestamps, and at most four matured observations. The base language state is never overwritten by a forecast. The session also exposes an ordinary fresh forecast from the current prefix; that is different from revising the original targets.

A tensor-only analytic conditioning function can be used for hypothetical analysis. Such an assumed value remains a hypothesis, not permission to commit memory or skip bytes. A caller-supplied byte stream is not authenticated ground truth about the external world; this interface only enforces its local timing/provenance contract.

## Exact energy update

For each component j, partition Y into observed O and remaining R. Its joint Gaussian has mean (mu_O,mu_R) and covariance blocks Sigma_OO, Sigma_OR, Sigma_RO and Sigma_RR. Given y_O:

    mu_R|O = mu_R + Sigma_RO solve(Sigma_OO, y_O - mu_O)
    Sigma_R|O = Sigma_RR - Sigma_RO solve(Sigma_OO, Sigma_OR)
    pi_j|O proportional to pi_j * Normal(y_O; mu_O,j, Sigma_OO,j)

The new conditional energy is minus the log of the normalized posterior mixture. The implementation uses Cholesky solves, not explicit covariance inversion, with float64 conditioning by default. It does not silently insert jitter or drop correlations. A failed factorization must be investigated.

The density identity E(R|O)=E(O,R)-E(O) is tested against independent PyTorch distributions. Sequential and joint conditioning agree numerically. Within-component covariance decreases, but TOTAL mixture variance can increase after informative evidence changes the mixture weights. A regression test prevents the false general claim that every observation must reduce uncertainty.

## Why raw conditioning is not the default

Correct Bayesian algebra does not make an imperfect learned joint model correct. The first unmodified replay found that raw conditioning worsened aggregate MSE from 0.974328 to 0.974341 after +1; from 1.035319 to 1.046476 after +4; and from 1.052860 to 1.071712 at the +64-only stage. Likelihood also worsened. Those results are retained under `results/online_raw`.

The follow-up is explicitly post-hoc engineering, after inspecting that diagnostic. It uses the normalized pool

    p_safe(R | observed) = (1-g) p_original(R) + g p_conditional(R | observed).

This is a mixture of complete joint distributions, not averaging covariance matrices. Except when g=1 it is NOT the exact Bayesian posterior of the original model, and the series of gated forecasts need not be a single temporally coherent Bayesian model.

Gate weights are chosen ONLY on the original 15-document validation-selection half. The grid is 0,0.05,0.1,0.25,0.5,0.75,1. A positive candidate is admissible when both its MSE and NLL improvements have negative upper 95% document-bootstrap bounds; the admissible lowest mean NLL is chosen. Otherwise g=0. All three seeds are averaged in gate selection. Selected weights: +1 stage 0.75, +4 stage 0.25, +16 stage 0.

The other 15 validation documents provide a descriptive check and do not alter the weights. They were used for temperature calibration in the previous repair, so they are not independent of that inherited model. The bootstrap is an empirical heuristic, not simultaneous multiple-comparison risk control, a conditional-coverage guarantee, or robustness under shift.

The gate is serialized and checksum-bound to its allowed forecasting-head bytes and teacher. New teachers require their own fitted/evaluated gate. Without a matching gate, the online session defaults to the original distribution; exact raw conditioning is an explicit diagnostic policy.

## Results on identical remaining targets

All test results reuse the previous 634 origins from 32 documents. Nearer observations are supplied ONLY at the time they become available, and observed coordinates are excluded from forecast loss. This evaluates a new information set, not magically improved time-t foresight. This domain/test set was inspected in prior iterations and is not an untouched confirmation benchmark.

| Available actual summaries | Method | Remaining-target MSE | NLL/coordinate | Marginal 90% coverage |
|---|---|---:|---:|---:|
| +1 | Original mixture, unchanged | 0.974328 | 1.313901 | 90.184% |
| +1 | Exact raw mixture condition | 0.974341 | 1.316264 | 89.439% |
| +1 | Guarded mixture, g=0.75 | 0.969848 | 1.309075 | 89.840% |
| +1 | Conditioned Gaussian control | 0.964090 | 1.326775 | 90.017% |
| +1,+4 | Original mixture, unchanged | 1.035319 | 1.367651 | 89.886% |
| +1,+4 | Exact raw mixture condition | 1.046476 | 1.383821 | 88.933% |
| +1,+4 | Guarded mixture, g=0.25 | 1.033029 | 1.365956 | 89.905% |
| +1,+4 | Conditioned Gaussian control | 1.030773 | 1.383882 | 89.550% |
| +1,+4,+16 | Original / guarded mixture | 1.052860 | 1.387145 | 90.011% |
| +1,+4,+16 | Exact raw mixture condition | 1.071712 | 1.416490 | 89.347% |

At t+1, the original t+4 forecast improves from 0.852347 to 0.826453 MSE (about 3.04%), while the t+16/t+64 components become slightly worse. At t+4 the remaining t+16 MSE improves from 1.017778 to 1.012714; t+64 is slightly worse at 1.053343 versus 1.052860. Aggregate gains must not be represented as every horizon improving.

Guarded-minus-prior paired MSE changes: after +1, -0.004480 with 95% document interval [-0.008650,-0.000572]; after +4, -0.002290 with interval [-0.003992,-0.000633]. NLL differences are -0.004826 and -0.001696, respectively, with negative paired intervals. At the final stage the gate is zero and outputs are exactly the prior.

The Gaussian control has lower MSE at both updating stages, while the guarded mixture has lower NLL. There is no universal estimator win. Nominal marginal coverage is near, not guaranteed to equal, 90%; joint-trajectory coverage is not established. A shuffled-observation diagnostic degrades performance and demonstrates that using incorrectly associated evidence can be harmful.

There are 40 raw-condition aggregate records and nine guarded records on each of the calibration-check and test subsets. The raw replay has 25,360 context-policy evaluations; the test gate replay adds 5,706. They reuse 634 underlying test origins and must not be counted as independent examples.

## Engineering fixes and checks

- Validate the ENTIRE token batch before applying its first byte. Previously a valid prefix followed by an invalid token could partially advance the state before raising.
- Check finite, positive normalization with the correct variable shape.
- Verify that the final saved sketch agrees with the saved P/S, not merely that shapes match.
- Store an owned 16-step history clone. The old contiguous tail could retain a 17-step backing allocation; the new physical allocation is exactly 512 bytes.
- Track absolute target times; capture each matured horizon once; reject a second live forecast unless replacement is explicit.
- Check snapshot head, teacher, gate, function shapes, finite Cholesky factors, timestamps, and exact sets of matured evidence before mutating live state. Integrity hashes are not signatures or protection against a malicious process that rewrites everything.
- No recurrent snapshot grows with token count. One fixed-origin function remains until cleared or explicitly replaced.

The complete focused suite passes 71 tests: 43 inherited and 28 new. Checks include independent density agreement, conditional joint/marginal identity, sequential equivalence, candidate/observation/parameter gradients, sampling moments, quantiles, a counterexample where mixture variance rises, no mutation of LM outputs, exact bounded history allocation, malformed restoration, and fresh-process restoration of original targets, observed evidence and outputs.

The initial inherited test invocation omitted the documented standalone PYTHONPATH and failed one subprocess import. It was rerun with the correct environment; this is retained as a setup log, not a model defect or hidden failed scientific result. Minor evaluator dimension checks and extreme-quantile numerical hardening were added after scoring; they change neither selected gates nor reported normal-range metrics. All final source/tests and original logs are retained.

## Cost and limits

One frozen forecast carries 25,368 raw bytes in float64 (three 32-dimensional component means and Cholesky factors plus weights), and at most 256 bytes of observed summary evidence. The existing 973K teacher uses 41,984 coefficient bytes plus a 512-byte sketch history. Total after all horizons mature is 68,120 raw bytes. Weights, Python metadata, fresh posterior workspaces, sampling arrays and logits are excluded. There is no hidden full-token KV cache, but the added function storage is NOT free and dense covariance scales quadratically.

A single-thread CPU microbenchmark measured about 0.517 ms for one marginal, one eight-coordinate condition, and a prior/posterior pool at batch1. It excludes the teacher, head evaluation, sampling and quantiles. This is additional overhead, not a token-generation speedup. No CUDA, Spark timing, new language model, unseen-corpus gain, token skipping or pretraining readiness is claimed.

## Research context

Gaussian-mixture conditioning and density normalization are standard probability. Estimating conditional distributions for missing values has established literature; this revision does not reproduce its neural architecture or external results:
- Przewiezlikowski, Smieja, Struski (2020), https://arxiv.org/abs/2010.02183
- PyTorch MultivariateNormal / MixtureSameFamily documentation, https://docs.pytorch.org/docs/stable/distributions.html

The next useful model research is to evaluate conditional-forecast training on a fresh dataset and the current 10.6M teacher, with the strong Gaussian baseline and equal observed information. Do not attach this gate to that model: checkpoint identity checks deliberately reject it. Decoder conditioning, hierarchical token infilling, and distribution-preserving skip-ahead remain unimplemented.
