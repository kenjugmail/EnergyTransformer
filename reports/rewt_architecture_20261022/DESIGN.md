# REWT kernel and architecture implementation

This study implements the follow-ups in the preceding kernel investigation. The
legacy model and checkpoints remain the reference. Architecture choices are
explicit in `ArchitectureConfig`; execution-only transformations are tested for
both forward values and parameter gradients.

## Implemented paths

| Component | What changes | Parameter/checkpoint consequence |
|---|---|---|
| Packed conditioning | One affine map for prior input, output prior and amplitudes; GRU input affine precomputed over observed positions | Original parameter names and tensors retained |
| General directional solver | One quadratic expansion per round evaluates all step candidates | Same conditional energy and line search, subject to floating-point reassociation |
| Certified solver | Conservative curvature bound removes candidate comparisons; two rounds use a reduced lift | Same step recurrence when the certified step equals the original accepted step; larger steps are a different inference/training policy |
| Preconditioned solver | Arbitrary rounds in M coordinates using the quadratic metric | Changes optimizer geometry, retains the learned energy parameterization |
| Structured factorization | Eliminate the 256-dimensional output block analytically; factor 3d latent coordinates | Exact alternative preparation for the same preconditioner |
| Observation scan | Input-dependent affine recurrence independent of refined state | Changes recurrent parameterization and enables batched conditioning/refinement across time |
| Chunk feedback | Refined recurrent state feeds the observation scan only at global chunk boundaries | New learned feedback map and gate |
| Delta memory | Persistent matrix with learned normalized keys, values, reads and write gates, added to local softmax retrieval | Additional learned projections and state; no symbolic lookup |
| Byte patches | Refine at fixed global byte intervals and reuse the latest correction between boundaries | Fewer energy updates; skipped positions have no current-energy descent guarantee |
| Mode diversity | Training penalty on normalized frequency correlations | No extra inference parameters |
| Shared/block modes | Half of Fourier modes shared; remaining modes cycle through individual blocks | Changed frequency mask, same parameter tensor shapes |
| Factorized output heads | Low-rank output-prior and decoder weights | Fewer learned parameters; the joint energy still includes all 256 logits |
| Ring decode | Fixed-capacity projected K/V and device-resident positions | Single-owner, fixed-weight inference session |
| Whole-step CUDA graph | Capture conditioning, local retrieval, state refinement and cache mutation together | Requires fixed batch/device/dtype and supported GRU core |
| Triton mode kernel | All preconditioned iterations and final lift in one GPU launch | Optional inference-only path; PyTorch solver supplies training gradients |

## Conditional energy and the directional identity

For the anchor-relative core, let z = s - anchor. Its energy is

    E(z) = 1/2 z H zᵀ + Σ a_i cos((z Fᵀ)_i + phase_i).

H is constant during refinement and across tokens while the weights are fixed.
Amplitudes depend on the observed prefix. F includes every workspace, including
the output logits. Centered output coordinates remove only the common logit
shift. The full supporting-module model has a prefix-dependent H because routing
changes its quadratic residuals, but H is still fixed during each line search.

For any direction g and scalar eta, a quadratic obeys

    Q(s - eta g) = Q(s) - eta <grad Q(s), g> + eta²/2 <g, H g>.

The wave phases change by `-eta (g Fᵀ)`. Thus all candidate energies need one
quadratic value, slope, curvature and mode projection per round. This remains
valid for absolute coordinates, frozen routing, specialists, initial noise,
coordinate masks and coarse proposals. A coarse proposal still compares the
complete energy. The implementation preserves KEEP when no finite trial improves
energy. Tanh activation is also supported by the directional identity.

## Certified steps

The amplitude parameterization implies `|a_i| <= 0.5 / sqrt(M)`. A global smoothness
upper bound for this conditional energy is

    L <= lambda_max(H + a_max Fᵀ F).

Any `0 < eta < 2/L` is a descent step in exact arithmetic. The implementation uses
`eta = min(requested_step, 1.9/L_upper)`, where the detached double-precision
spectral estimate has a conservative rounding margin. This is a numerical
curvature certificate, not an interval-arithmetic proof. It does not certify
convexity, convergence to a global minimum, or correctness of a prediction.

The certificate is rebuilt from current weights during training. Explicit
inference caches carry parameter/buffer version signatures and reject stale
weights. Step-size selection is treated as control, with no derivative through
the spectral certificate. Learned energy parameters and the state updates remain
differentiable. Nonfinite proposals retain the previous state.

For two fixed steps, the displacement stays in the span of F and FH, and no
full-dimensional intermediate candidate is needed. Trace-enabled execution
computes actual full-state energies for audit; trace-free execution uses the
reduced two-step lift.

## Arbitrary-depth mode-space solver

Let B = F H⁻¹ and G = F H⁻¹ Fᵀ. Starting at the anchor, write `z = c B`. A
preconditioned gradient step gives the closed recurrence

    c_next = (1 - eta) c + eta a sin(c G + phase).

Only M coefficients are carried across arbitrarily many rounds. The corresponding
metric smoothness bound is `1 + a_max lambda_max(G)`. The final state is lifted
once. The 256-coordinate output correction therefore already lies in at most an
M-dimensional learned subspace, without removing output participation from the
Fourier energy. The separate low-rank-head experiment is a parameter-saving
architecture change, not a claim that output waves can be dropped for free.

The dense reference solve adds a unit quadratic in the otherwise-null centered
output gauge. F has no gauge component, so this fixes the solve without changing
the centered displacement. The structured implementation instead eliminates the
output block. For `D = diag(precision_output + decoder_weight)`, its inverse on
the centered subspace is

    R = D⁻¹ - D⁻¹ 1 (1ᵀ D⁻¹ 1)⁻¹ 1ᵀ D⁻¹.

The latent Schur system is `H_ll - H_lo R H_ol`. Only `3*width` coordinates need
Cholesky factorization (144 rather than 400 for the study model). Values and
parameter gradients are checked against the dense gauge-fixed solve, including
nonuniform output precision and zero decoder coupling.

## Parallel observation path and explicit feedback

The observation backbone uses `h_t = a_t h_(t-1) + b_t`, with gates and candidates
computed from observed input. A division-free associative scan composes the
pairs `(a,b)`; it remains defined for zero or underflowing retention gates.
This follows the scan-compatible recurrence idea studied in
[Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201), rather than claiming
its results transfer to this model.

Independent observation states make local attention queries, conditioning and
energy refinement parallel across a chunk. Feedback is an explicit alternative:
a gated projection of the previous refined recurrent block enters the scan only
at global byte multiples of the feedback stride. Global positions, rather than
Python call boundaries, control feedback and patch updates. Tests split prompts
at awkward boundaries and compare all outputs and carried memory.

Truncated temporal credit during training detaches carried learned state at the
specified global interval. It does not erase observed bytes or reset forward
memory. Independent scan conditioning still has its ordinary within-chunk graph.

## Long-term memory

The local attention window is supplemented by a matrix S. A write uses

    S_t = S_(t-1) + beta_t k_t (v_t - k_tᵀ S_(t-1))ᵀ,

with normalized keys and a sigmoid write gate. Reads use normalized queries and
include the current write. A learned gate combines the read with local softmax.
The reference implementation is tested against an equivalent unit-lower-
triangular chunk solve; it never forms a matrix inverse. This uses the delta-rule
approach studied in [DeltaNet](https://arxiv.org/abs/2406.06484). The repository
implementation is a small PyTorch chunk kernel, not an unverified claim to use
Flash Linear Attention or match DeltaNet's large-scale results.

Tests cover overwriting an identical key while retaining an orthogonal key,
causality, chunk continuation, gradients and optimizer changes. A separate
learned overwrite-recall study uses eight possible keys and ten values, with
far queries outside the 16-byte local window and test prompts up to 259 bytes.
Algebraic overwrite behavior by itself does not establish learned retrieval.

## GPU execution contract

The ring session replaces growing K/V tensors with circular overwrites. Causal
convolution retains only its short raw tail; learned positions and the ring index
are device scalars. All captured operations have fixed shapes and addresses.
Warmup/capture state is restored before consuming the first real decode byte.
Returned logits are cloned so later graph replays cannot overwrite earlier
outputs. Parameter versions, buffer versions, dtype, device and training mode
invalidate a session. Writes through `.data` bypass PyTorch version tracking and
are unsupported.

The optional Triton kernel keeps one row's coefficients and its small Gram
matrix within a program, runs the requested number of iterations, then lifts to
the joint state. It is restricted to contiguous CUDA float32 tensors and rejects
autograd use. It is an inference kernel, not a replacement backward kernel.
References: [Triton fused reductions](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
and [PyTorch CUDA graph constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

Full-sequence and one-byte convolutions can take different cuDNN arithmetic
paths. Numerical audits explicitly disable TF32 for both paths. This precision
choice, capture/compile cost, output ownership cost and shared-device load must
be reported with GPU results.

## Study boundaries

The main experiment is a cold-start comparison of 13 variants across three
seeds, 1,024 updates each. All share batches, supervision and original REWT
initialization by seed. Newly added modules follow their variant's construction
order. The training set has 5,400 semantic tasks; validation and test each include
familiar and withheld wordings. Fresh test semantic IDs exclude all earlier
recorded test sets. Auxiliary labels and heads are used in training only.

The separate language screen compares eight variants across the same three
seeds for 512 updates, or 1,572,864 BPE tokens per run. It reuses the original
training partition and training-only vocabulary; new validation/test documents
come from bytes 8..12 MiB of the pinned official validation source, excluding all
old normalized document hashes. This is a screening budget, not a demonstration
of fluent prose or a replacement for the earlier longer language study.

Changed architectures have different parameter counts. Equal updates are not
equal compute. Training CPU process times include concurrent-worker contention;
execution benchmarks use alternating arms and report both CPU and wall time.
No test result is used to pick a checkpoint. Final findings must include failures
and seed variation as well as wins.

## What refinement depth can and cannot add

In this anchor-relative core, the refinement displacement is a function of the
amplitude vector and fixed learned geometry: `state = anchor + R(amplitudes)`.
The anchor itself does not change the residual landscape once coordinates are
shifted. At any centered stationary point,

    H z = Fᵀ [a sin(F z + phase)],

so the displacement lies in the M-dimensional preconditioned mode span. This is
why an arbitrary-depth M-coordinate solver is possible. Ordinary finite-step
gradient descent can have additional transient directions, so switching a
checkpoint to preconditioning is still a change of its finite-step prediction
function. The structured factorization, by contrast, preserves that function.

More rounds optimize this same conditional landscape; they do not introduce new
observations or guarantee better task reasoning. The sequential dependency in
the original architecture comes from feeding the refined recurrent block into
the next token's conditioning. Separating the observation scan removes that
dependency and exposes entire chunks to batched kernels. Chunk feedback restores
an explicit, less frequent dependency when the learned task needs it.

This distinction is central to interpreting results: execution parity, energy
descent, prediction quality and memory capacity are four different claims. Each
needs its own evidence.

## Data and follow-up audits

The new reasoning dataset is a fresh seeded draw, not wholly unseen relative to
all past training/validation files: 502 of its 5,400 training identities and 3 of
its 81 validation identities appeared in earlier corresponding partitions.
All 162 test identities are new relative to the 2,792 previously recorded test
identities. Every model in the new comparison starts cold, and train/validation/
test identities remain mutually disjoint within this study. See
`split_overlap_audit.json` for exact counts.

The first memory experiment used response loss alone for 768 updates and failed
to learn robust distant recall. A separate follow-up supplied training-only
key/query alignment, value copying and delta write-gate targets for 1,536 updates.
Both local and delta arms received the common binding/copy targets; only delta
has a write gate. To distinguish supervision from training duration, a third
holdout was declared and response-only local/delta controls were trained with
exactly the follow-up's seeds, batches, optimizer, selection and 1,536 updates.
The six supervised checkpoints are reused unchanged for that matched ablation.
No auxiliary labels, target values or symbolic checkers enter model inference.

A separate equal-compute screen completes the original plan's time-budget
comparison. It uses five solver paths and three new seeds, with 150 measured
training process seconds per fit. Validation and checkpoint I/O are excluded;
the learning-rate and validation schedules follow consumed CPU time. The final
update may overshoot by one batch, and every trace records cumulative CPU time.
Another 256 language documents exclude the entire previous test-document set.
This screen was declared after the original equal-update results and uses a
fresh test partition; it is not a preregistered part of the original 81 fits.
