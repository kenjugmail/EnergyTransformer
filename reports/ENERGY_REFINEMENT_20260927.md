# Persistent energy memory plus candidate-energy computation

Executed September 27, 2026. Additive research against the forecasting branch at
`69f16a3e01c7954e04aadfc951785033dbcb73c5`.

## Decision

The attachment's proposed distinction is now implemented and tested: the teacher
keeps its persistent P/S memory, while a separate learned scalar energy drives
multiple gradient steps over a temporary candidate before next-byte decoding.
A tolerance-based, backtracking inference solver is included as an optional
adaptive-computation policy. No candidate is committed to the teacher memory.

**Keep the iterative-energy variants experimental.** They improve the frozen
teacher but do not beat stronger ordinary adapters. More energy descent is often
worse for prediction. The default working model, existing forecast heads/gates,
protected-memory permissions, kernels, and main branch are unchanged.

Fifteen primary fits and six explicitly post-hoc fits were executed. This is a
frozen-teacher next-byte adapter study, not full end-to-end EBT pretraining, a
reasoning benchmark, or evidence that the full method in the attached paper fails.

## 1. Basis and what was actually used

The user's supplied note proposes persistent energy memory + latent energy
optimization + adaptive compute, with candidate updates `z <- z - eta grad E`
and energy-change or learned stopping. That motivates this controlled test rather
than further Gaussian-forecast calibration.

Gladstone et al., *Energy-Based Transformers are Scalable Learners and Thinkers*,
is confirmed in the ICLR 2026 proceedings. It frames prediction through learned
input/candidate compatibility and optimization. This small adapter does NOT copy
its complete Transformer energy network, full training recipe, model scale,
random candidate initialization, candidate space, datasets, or System-2 procedure.
External reported improvements are not results of this repository.

Primary sources:
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/e19a65fd53b6f9a88b354da98813465d-Abstract-Conference.html
- https://energy-based-transformers.github.io/
- Hoover et al., Energy Transformer (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html

The teacher is the original 972,896-parameter byte model, SHA-256
`78b4662a3c30d2e1a8905340e1c465eb1f83c34e98bcfc28177b2dbfb93ea06b`.
All teacher weights remain frozen and byte-identical. The actual 10.6M GB10
checkpoint was not available locally; no CUDA, paid service, or remote training
job ran. The supplied standalone `reference/energy_real` functional snapshot was
used. It is not byte-identical source to the complete remote implementation, so
these local counts must not be represented as a full checkout/GB10 acceptance.

## 2. Two different energies, with different jobs

The existing persistent memory represents a positive diagonal quadratic:

    E_memory(W) = sum_i [.5 P_i ||W_i||^2 - S_i^T W_i],
    W_i* = S_i / P_i.

Its read needs no saddle escape. It carries information across actual input bytes.

The new candidate energy is conditioned on the present teacher representation h
and an optional current-memory read m. During a single refinement, those inputs
and the energy function remain FIXED. A new byte can produce a different energy.
The candidate z is temporary and starts at zero residual; this is a deliberately
restricted initializer, not a learned or random answer proposal as in full EBT.

The extra memory read has 64 coordinates for this four-head teacher. Each head
uses four fixed orthonormal slot queries and three fixed value projections of S/P,
plus four log query variances from P. No projection is fit on test data. This is
lossy; it is not the complete P/S and is not a future forecast. The read is
normalized using TRAIN positions only. Even the "energy_hidden" control has
indirect access to history through h; it lacks only the EXTRA explicit read.

For J conditional potential units:

    a(c) = 2 sigmoid(Ac+a0),  b(c)=Bc+b0,
    E_phi(z | c) = .5 ||z||^2
        - 4/sqrt(J) sum_j a_j(c) [softplus(u_j^T z+b_j(c))-softplus(b_j(c))].

The subtraction fixes E(0|c)=0. With finite weights and bounded a, softplus's
1-Lipschitz property gives `E(z) >= .5 ||z||^2 - C||z||`. The energy is therefore
coercive, and exp(-E) is integrable. Its normalization constant is NOT computed;
we do not claim an exact continuous likelihood or calibrated verifier score.

The derivative used in the code is exactly

    grad_z E = z - 4/sqrt(J) U^T [a * sigmoid(Uz+b)].

The Hessian is `I - 4/sqrt(J) U^T diag(a*sigma*(1-sigma)) U`, which can have negative
eigenvalues. This is genuinely nonlinear scalar-energy computation, not an
arbitrary recurrent map renamed as energy. Its mathematics is conventional.
Tests compare the derivative, candidate gradients, and unrolled parameter
derivatives to independent autograd. No stop-gradient approximation is used.

For the primary models, z has 32 coordinates and performs four shared-weight steps
with eta=.25. The logits are `W_out h + W_out B z`, where B is trainable and starts
at zero. Consequently every fresh model initially matches the teacher. All
iterates are decoded through the same frozen W_out. The candidate is not retained
between bytes. The optional session preserves only the teacher's P/S plus counters.

## 3. Primary protocol and controls

The five arms are:

| Arm | Meaning | Trainable parameters |
|---|---|---:|
| feedforward | Ordinary h-only residual MLP | 36,112 |
| memory_ff | Same-budget MLP with the current memory read | 35,965 |
| recurrent | Unconstrained shared recurrent update with that read | 36,075 |
| energy_hidden | Scalar-energy descent using h only | 36,272 |
| energy_memory | Scalar-energy descent using h and m | 35,968 |

Parameter differences are small, but computation is NOT matched. The iterative
arms use a 32-dimensional output bottleneck; the ordinary MLPs can allocate more
capacity to their wider output representation. A post-hoc control examines this.
The recurrent control has the same candidate dimension and number of steps but
is not constrained to be an energy gradient.

Each arm has seeds 11,22,33 and 600 AdamW updates of batch128. The same seed gets
identical sampled-index hashes across all arms. CE is averaged across the four
iterates for iterative models, versus one output for feedforward controls.
The peak LR is .001, weight decay .01, cosine decay to20% of the peak, and global
gradient clipping1. No energy target labels, manual correctness rewards, future
summary inputs, or teacher updates are supplied.

The protocol was saved before fitting; extraction was already in progress.
Step zero and each100 updates are eligible for selection using the FOURTH-iterate
validation bpb. After these hashes are frozen, inference depth is selected on the
same validation documents from 0,1,2,4,8,16 and three energy-stop tolerances.
Final test labels do not choose depth. Both validation sets and model-family
selection are historically reused; this is not a preregistered independent study.

The corpus is the locally installed Gensim Lee Background fixture with inherited
document assignment, plus its separate Lee fixture. The teacher runs serially;
the label at origin t is byte t+1, never consumed before its prediction. Positions
are every fourth byte after15, through at most384 bytes per document.

| Partition | Documents | Positions |
|---|---:|---:|
| Train | 231 | 21,245 |
| Validation | 30 | 2,760 |
| Primary test | 32 | 2,921 |
| Secondary news | 50 | 4,520 |

These domains have been inspected repeatedly in prior iterations. Both test sets
are development evidence. They are not general reasoning or untouched natural-
language benchmarks. The exported cache has derived numerical features and sparse
next-byte labels, not full document text. Its split/teacher/corpus hashes are saved.

## 4. Results: useful adapters, not a win for iterative energy

Mean bpb over three seeds, identical positions; lower is better. Iterative rows
use the primary prespecified four steps.

| Model | Primary | Secondary |
|---|---:|---:|
| Frozen teacher | 2.486589 | 2.521613 |
| Ordinary h adapter | 2.413779 | **2.482872** |
| **Memory-read MLP** | **2.410568** | 2.484301 |
| Recurrent refinement | 2.445360 | 2.507568 |
| Energy, h only | 2.452396 | 2.504293 |
| Energy, h plus current memory | 2.452212 | 2.505569 |

The memory-read MLP won the mean validation comparison before final scoring.
Its primary reduction versus the teacher is .076020 bpb (about5.13% lower byte
perplexity); the paired document-bootstrap interval for its signed loss change
is [-.103582,-.049849]. Its small advantage over the h-only adapter is inconclusive:
-.003211 bpb, interval[-.011328,.004995]. Extra capacity/training cannot be ignored;
no fully continued-teacher or new Transformer comparison was run here.

Energy+memory improves the frozen teacher by .034377 bpb at four steps, but is
WORSE than the memory-read MLP by .041643 bpb, interval[.025473,.058406]. Its
comparison with the ordinary recurrent update is inconclusive (.006852 bpb,
interval[-.000853,.015466]). Adding the explicit memory read to the scalar energy
has no established advantage over h-only energy (difference-.000185,
interval[-.006130,.005903]). h itself already depends on persistent memory.

All paired intervals resample complete documents after averaging seeds. They do
not fully describe seed uncertainty, correct multiple comparisons or historical
test reuse, or prove robustness to distribution shift. All per-position arrays
and worse conditions are retained in the archive.

## 5. Does more test-time energy descent help?

| Energy+memory iterations | Primary bpb | Secondary bpb |
|---|---:|---:|
| 0: teacher | 2.486589 | 2.521613 |
| **2: validation-selected depth** | **2.446436** | **2.493470** |
| 4: trained depth | 2.452212 | 2.505569 |
| 16 | 2.527625 | 2.588029 |

All three primary iterative families selected TWO steps on validation, not an
adaptive energy tolerance. Two steps help relative to four, but still lose the
ordinary adapters. The unconstrained recurrent control also deteriorates with
more steps; this is not an energy-specific universal failure.

The optional solver backtracks until the candidate has no higher fixed energy,
and halts by relative energy improvement or gradient RMS, with a maximum16 steps.
A rejected search leaves that row unchanged. The normalized energy is not
recomputed from a new context at each internal step. No labels enter this solver.
Batched active-row counts must not be interpreted as proportional wall-time savings.

At tolerance .01, energy+memory averages9.95 accepted updates and achieves2.503073
primary bpb, worse than the teacher. Tighter thresholds generally run longer and
perform worse. Therefore stopping at a small energy change is NOT promoted as a
prediction-quality policy. The safe experimental comparison remains fixed depth.

On 24,576 inspected primary energy+memory update transitions (two data sets,
three seeds,256 positions each,16 steps), no unguarded update increased energy
beyond the specified numerical tolerance, yet14,090 transitions (57.33%) lowered
energy while increasing the correct byte's loss. These are repeated transitions,
not24,576 independent examples. This is the central negative result: a learned
scalar objective and its task performance are not automatically aligned outside
the optimized trajectory range.

The curvature diagnostic inspects32 contexts per seed at0/4/16 updates. Two of
288 energy+memory query-state cases have negative curvature; zero also meet the
small-gradient criterion. No near-stationary saddle trap is diagnosed in this
sample. No fractal dynamics or global-optimum guarantee is claimed.

## 6. Post-hoc bottleneck controls

After viewing the primary results, two fixed-schedule follow-ups were registered:

| Follow-up | Parameters | Primary bpb | Secondary bpb |
|---|---:|---:|---:|
| MLP through a32-dimensional candidate | 36,249 | **2.426009** | **2.491348** |
| Direct160-dimensional energy residual | 35,991 | 2.459703 | 2.502125 |

The rank-matched feedforward control still improves over the primary iterative
models. Increasing candidate width alone also does not solve the problem. These
are post-hoc diagnostics, not independent confirmation or a fair exhaustive search
of all energy parameterizations. No subsequent model tuning was performed.

The full-width energy has no learned output B. Instead z itself adjusts h; one
learned scalar amplitude on the nonlinear potential begins at zero to preserve
the initial teacher. Its two-step policy is again selected by validation. It does
not improve the main candidate or justify another architecture promotion.

Total:21 fits,12,600 updates,1,612,800 presented next-byte targets. These are repeated
presentations of21,245 training positions. Original and final checkpoints, optimizer
states, selected hashes, and the primary/follow-up executed source are retained.

## 7. Cost and numerical checks

Inference now projects only the final candidate into byte logits. The training
trajectory still projects all four supervised iterates. A post-scoring parity
audit confirms exactly equal final logits for all21 selected checkpoints; no new
quality gain is attributed to this algebraic simplification.

One-thread CPU, seven randomized interleaved repeats, identical cached inputs:

| Adapter | Batch1 inference ms | Batch128 training update ms |
|---|---:|---:|
| h-only MLP | .0541 | 1.1867 |
| memory-read MLP | .0630 | 1.2544 |
| recurrent,4 steps | .1545 | 4.0445 |
| energy+memory,4 steps | .1600 | 3.3369 |
| full-width energy,4 steps | .1646 | 3.9507 |

These exclude the teacher and extraction of the compact memory read. Training
includes all four losses and AdamW at zero learning rate on a model copy. Timing
is not a GPU benchmark, equal-compute learning comparison, or language-generation
speedup. The teacher's persistent payload remains41,984 bytes per stream. Candidate
states and memory reads are temporary; weights and workspaces are separate costs.

The final focused suite passes39 tests, including exact analytic/autograd force
agreement, mixed parameter derivatives through all iterations, possible negative
curvature, bounded-descent rejection, causality, no teacher-state mutation, fresh-
process inference restoration, and full optimizer/RNG restart on a small fixture.
No new source is represented as the entire historical repository's test suite.
Actual CUDA/container/GB10 execution was not performed.

## 8. Implementation limits and next experiment

The narrow test freezes the teacher, uses a limited current-memory summary and
zero candidate initialization, uses one small energy family, and trains four
updates with one shared step size. It therefore does not establish that the larger
EBT approach or joint energy-memory training cannot work. It establishes that this
specific cheaply testable combination has not earned replacing the simple control.

Before adding Fourier/fractal jumps or a long default loop, the next controlled
study should vary TRAINING depth (not only test depth), include a learned initializer
with its own zero-iteration control, and directly evaluate whether an energy ranks
better candidate predictions lower. Compare a learned progress/halting estimate to
the strong fixed-two-step baseline. Candidate normalization and step-size changes
should be isolated rather than bundled. Fresh data and the actual10.6M teacher
are needed; none of these proposed follow-ups ran here.

The source supports optional live decoding but does not install it by default.
Under the same forced byte stream, P/S is unchanged. In free-running generation,
changed logits can produce a different next byte and therefore a different future
history. This is not distribution-preserving speculative decoding or token skipping.

Training and extraction are CPU-only in the executed study. The new per-run fit
function has a tested exact restart, but this separate study launcher is intentionally
create-only, does not reuse old completion reports, and has no multi-process orchestration
or CUDA acceptance. Reproduce from a fresh directory or use the pinned per-run source;
do not bypass the existing repository's stronger production study integrity checks.
