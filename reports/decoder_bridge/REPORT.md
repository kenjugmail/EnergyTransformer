# Forecast-to-decoder architecture study

**Executed September 26, 2026.** Repository baseline inspected: main `397dfae647f9521875344b99227a51a3974c0dfd`, draft forecasting branch `c5583551eb68b89c21337a0cc639e4c1bfad0b00`.

## Decision

The strongest measured language improvement comes from an ordinary hidden-state adapter, not from forecast features or the nonlinear energy reader. Keep it as a research candidate and reproduce it on the actual 10.6M Spark teacher. Do not promote the forecast integration to the working model on these results.

This is the first downstream adapter comparison in this sequence. Fifteen completed fits compare five small architectures at three seeds. Three additional exports fold a linear forecast into an ordinary history adapter without training. No original teacher, forecasting head, kernel, memory authorization, or production checkpoint was changed. No CUDA, Spark, paid service, or remote training ran.

## 1. What was actually executed

The frozen teacher is the archived 972,896-parameter byte language model, SHA-256:

`78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`.

It is loaded against the preserved `reference/energy_real` functional snapshot in the preceding artifact bundle. That snapshot has condensed source/comments and does not have the same source-blob identity as the full remote module. Its behavior is audited locally; the entire remote repository and actual GB10 stack were NOT rebuilt or tested this turn. The new modules use the inspected current `energy_real` interface.

PyTorch is `2.10.0+cpu`. The teacher is four layers, width160, four heads, sixteen slots, feed-forward width320. Its raw float32 recurrent P/S payload is41,984 bytes per sequence. It is not an instruction-following model.

Three previously trained mean-anchored calibrated mixture heads supply forecasts for the matching seeds. Each has43,683 frozen neural parameters, plus4,168 fitted anchor/calibration scalars. Their prior training and fixed buffers are additional resources. Matching only TRAINABLE adapter parameter counts is not total-parameter or compute matching.

## 2. Causal decoder integration

At byte t, the frozen teacher supplies its final normalized representation h_t. Its original logits are W_out h_t. Every adapter changes them explicitly to

    logits_t = W_out h_t + W_out Delta_phi(h_t, features_t).

W_out and all teacher/forecast weights remain frozen. Only phi is trained. The adapter is after the final normalization and never writes into teacher P/S. For identical input bytes, the underlying teacher states remain identical. In free-running generation, changed probabilities can change sampled bytes and consequently later histories; we do NOT promise identical generated sequences.

Each adapter's last linear projection starts at zero. Initial logits therefore equal the teacher exactly. The new state reader can be enabled or disabled explicitly; it is not distribution-preserving speculative decoding.

The input C_t contains the previous16 eight-variable sketches of observed P/S, including time t. There is no true future sketch in the inputs. Labels are byte t+1, which is not consumed until after prediction. Warmup is16 actual bytes, including for the hidden-only arm, to keep evaluation positions identical.

### Five arms

| Arm | Features beyond h_t | Bottleneck | Trainable parameters |
|---|---|---:|---:|
| hidden | None |82|26,482|
| history | Observed16x8 summary history |59|26,651|
| mean | Frozen32-coordinate joint-forecast mean, repeated/rescaled to128 |59|26,651|
| moments | Forecast mean and log marginal standard deviation, repeated/rescaled |59|26,651|
| energy_read | Two learned32-dimensional queries into the three-component energy |36|26,628|

The maximum adapter parameter difference is169. This small budget matching does not match all architectural properties: the hidden-only arm can spend its budget on a wider hidden transformation, and forecast arms incur extra frozen components. That is a meaningful strong comparator, not an isolated one-variable experiment.

For energy_read, each query is

    q_a = m(C_t) + 0.1 Q_a h_t.

The normalized mixture energy is

    E(y|C) = -log sum_j pi_j(C) N(y; mu_j(C), Sigma_j(C)).

Each query reads responsibilities r_aj proportional to pi_j N(q_a;mu_j,Sigma_j), their weighted component means, and the negative candidate gradient:

    -grad_q E(q_a|C) = sum_j r_aj Sigma_j^-1 (mu_j - q_a).

A dimensionless tanh transformation bounds the gradient features before residual fusion. Two triangular solves implement the exact pre-transformation gradient. Tests compare it with independent autograd. The queries are hypotheses, never observed evidence. There is no iterative descent or claim that smaller internal energy means a correct next byte.

The mixture describes a genuine nonlinear energy over predicted summaries. It is conventional normalized mixture mathematics, not a new physical energy law. The forecast bottleneck is still only eight summary variables; it does not contain all teacher state information.

## 3. Data and protocol

The study uses the separately installed Gensim Lee Background and Lee news fixtures. No raw corpus text is included in the deliverable. These domains and documents have appeared repeatedly in prior project work. This is development evidence, not an untouched generalization benchmark.

The adapter preserves the forecasting dataset's document assignment. It extracts causal serial teacher h_t and sketch histories, then scores every fourth eligible origin, up to384 source bytes per document.

| Partition | Documents | Scored/training positions |
|---|---:|---:|
| Train |231|21,245|
| Validation |30|2,760|
| Test |32|2,921|
| Secondary news |50|4,520|

Summary normalization is the unchanged training-history normalization from the preceding forecasting study. Adapter side-feature normalization is fit on adapter TRAIN only. The pretrained forecast heads historically used validation and calibration splits from these same domains; their outputs do not become independent evidence simply because the adapter uses a new loss.

The preregistered-in-this-session protocol was written before new extraction, fitting, and final scoring. It names energy_read as the research candidate and ordinary hidden/history adapters as controls. Fifteen fits each receive600 AdamW updates of batch128, peakLR.001, weight decay.01, cosine decay to.2 times peak, and gradient clipping1. Each fit sees76,800 presented positions; total1,152,000 presentations are repeated sampling, NOT independent documents. All arms at a seed have identical cumulative batch-index hashes.

Validation is evaluated at step0 and each scheduled100 updates. The best next-byte validation loss selects individual checkpoints; the cross-seed architecture winner is hidden, before final test scoring. All15 selected a trained checkpoint. No additional neural fitting occurred after viewing test outcomes. Checkpoint hashes are frozen in `fits/FROZEN.json`.

## 4. Primary results

Bits per byte on the same selected positions; lower is better. Means across three seeds:

| Model | Test,2,921 positions | Secondary,4,520 positions |
|---|---:|---:|
| Frozen original teacher |2.486589|2.521613|
| **hidden** |**2.412866**|**2.484073**|
| history |2.428626|2.501614|
| mean |2.419956|2.493949|
| moments |2.421958|2.490600|
| energy_read |2.425448|2.495052|

The hidden adapter reduces byte perplexity by4.982% and2.568% relative to the frozen teacher. It adds26,482 learned parameters and additional optimization, so this is not evidence that it beats a fully continued teacher or a newly trained Transformer at equal compute.

The primary hidden-minus-base difference is-0.073723 bpb,95% paired document-bootstrap interval[-0.096185,-0.051452]. On secondary it is-0.037540, interval[-0.055969,-0.019395]. Energy_read is worse than hidden by+0.012582 bpb on test, interval[+0.005849,+0.019816]. Its improvement over the original teacher cannot be attributed exclusively to forecast information.

Intervals use5,000 whole-document bootstrap draws after averaging the three seeds. They are conditional on these datasets and fits, do not fully represent seed uncertainty, and do not correct historical adaptive test reuse or simultaneous comparisons.

### Feature interventions

For energy_read, test loss changes from2.425448 to2.437615 with standardized side features set to zero, and to2.447046 with histories/forecasts swapped across documents. That suggests useful aligned input in this fit. However, on secondary, zeroing the side features improves loss from2.495052 to2.489126. The same family still loses the ordinary hidden adapter.

These interventions are not separately trained controls. Shuffling demonstrates sensitivity to mismatched input, not a unique need for the forecasting architecture. All normal, zero, and shuffled records remain in the results.

## 5. Denser and longer-prefix audit (post hoc)

After checkpoint choices and primary outcomes were fixed, every valid next byte after warmup was evaluated through the first1,024 bytes of the same82 documents. No new training or checkpoint selection followed. There are50,436 targets:26,627 test and23,809 secondary. They include the original sampled positions, so these counts are not independent additional datasets.

| Model | Dense test bpb | Dense secondary bpb |
|---|---:|---:|
| Base |2.551356|2.515154|
| **hidden** |**2.476703**|**2.483322**|
| history |2.499334|2.502192|
| mean |2.486705|2.492956|
| moments |2.489334|2.493073|
| energy_read |2.492793|2.495290|

The hidden advantage also persists in the later t>=383 region not represented in adapter training crops. This is a within-domain diagnostic, not arbitrary long-context understanding. Dense energy_read-minus-hidden differences are+0.016091(test) and+0.011968(secondary), with positive paired intervals. The ordering did not reverse.

## 6. A simplification: remove the mean-only forecasting dependency

For these retained heads, the conditional mean is a fixed affine map of the normalized history:

    m(C)=A^T vec(C)+b.

The mean arm concatenates a repeated, scaled version of that mean before a linear adapter projection. The two affine maps can be composed exactly in real arithmetic. A converted history adapter therefore represents the same computation without separately loading or running the forecast model. Feature centering and scaling are folded into the bias and weight.

All three mean adapters were converted without training. Across all three seeds and the7,441 sampled evaluation positions, the maximum inspected logit difference was1.90735e-6 in float32 and there were zero next-byte argmax changes. Tiny likelihood differences are rounding, not an improvement. Float64 forward and gradient equivalence are tested.

This also limits the scientific interpretation: for this linear mean, the mean-adapter function lies inside the corresponding history-adapter family. Any observed advantage can result from feature conditioning or training bias, not extra future information or strictly greater representational power.

## 7. Runtime and storage

Single-thread CPU, five randomized/interleaved repeats, identical forced inputs,64 timed continuation bytes. Prefix states are computed once and cloned outside timing. These are end-to-end serial session measurements at batch1, not CUDA results, training speed, or model-independent performance.

After a512-byte prefix:

| Model | ms/byte | Raw recurrent bytes |
|---|---:|---:|
| Base |5.2686|41,984|
| hidden |5.2922|41,984|
| history |5.3868|42,496|
| mean |5.6887|42,496|
| moments |6.1157|42,496|
| energy_read |6.2871|42,496|
| Folded mean |5.6994|42,496|

Energy_read adds about19.3% in this measurement. The tiny hidden overhead varies by prefix and is noisy; it is not a guaranteed0.45% overhead. Folding removes a dependency but does not demonstrate a runtime speedup in this implementation.

Storage excludes weights, framework objects, temporary covariance/query tensors and logits. Only non-hidden arms retain the extra16x8 float32 summary history (512 bytes). No predicted future is carried as teacher P/S. Cached extraction, unlike streaming inference, stores many h_t and labels on disk and can consume substantial host memory when assembled.

## 8. Engineering improvements

- New `DecoderBridgeSession` checks teacher/head/checkpoint identities, normalization, token validity, warmup, snapshot shapes, final-sketch/P/S agreement and causality. The hidden candidate requires no forecast head.
- The new trainer saves optimizer, RNG, sampler hash, best-selection state and full latest checkpoint. Same-environment fresh-process resume is tested. Changing source/runtime/schedule/data is rejected; original scientific runs are not promised to resume after this source hardening.
- `decoder_cache` publishes complete derived-feature shards atomically. A timeout retains completed shards; resume verifies identities, source/runtime signatures and content hashes before reuse. CPU fixtures compare resumed and uninterrupted outputs bit for bit. Altered shards, changed batch and overlapping ranges are rejected.
- The Spark cache supports hidden/history studies without any pretrained forecast head, and can alternatively reuse the current teacher's forecast manifest/normalization for all five arms.
- CUDA cache execution is opt-in, guarded against another active GPU job, and capped using the repository8GiB allocator policy. It was NOT executed here. The host-memory reserve is checked at startup, not continuously guaranteed; assembling a large cache still costs memory. Local POSIX atomic-file assumptions apply.

The combined local suite printed **168 passed in33.64s** (126 inherited and42 new tests). Two earlier tool invocations timed out after test progress; their incomplete logs are retained. The completed run used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. This is not a remote CI result or the full historical repository suite.

Scientific source snapshots are retained in `executed_source/`. Dense auditing, folding, session hardening, and restartable extraction were added after scoring. Their validation does not change the original fits or magically turn a post-hoc analysis into a preregistered one.

The released focused package includes the original teacher, retained heads, all15 adapter checkpoints and optimizer endpoints, three conversions, predictions and logs. Raw corpus text and the large cached feature/next-byte-label dataset are not included; regenerate them from the documented local corpus. Existing older forecasting artifacts are not all duplicated. Thus this is a new study package, not the entire project history.

## 9. Research context and interpretation

- Multi-token prediction (Gloeckle et al.,2024): https://arxiv.org/abs/2404.19737 . Training multiple future-token outputs on a shared trunk is an important simpler competing objective. It was not trained here; published gains are not transferred to this project.
- EAGLE (Li et al.,2024/2025): https://arxiv.org/abs/2401.15077 . Feature forecasting plus verification is different from our adapter, which intentionally changes the model distribution and does not accelerate verified decoding.
- Test-time regression: https://arxiv.org/abs/2501.12352 . Separating stored associative information from its access and update rules remains relevant. This does not establish novelty for regression or mixture-energy mathematics.
- NVIDIA Spark memory guidance: https://docs.nvidia.com/dgx/dgx-spark/known-issues.html . Use multiple memory signals on UMA hardware; the nominal system-memory size is not a safe standalone allocator budget.

The next useful experiment is now simpler: test hidden/history adapters on the actual10.6M teacher using the restartable cache, then introduce richer current-state or forecast features only against that strong control. A larger research plan is supplied separately. No large pretraining readiness, replacement of Transformer attention, new fractal dynamics, or general reasoning gain follows from this study.
