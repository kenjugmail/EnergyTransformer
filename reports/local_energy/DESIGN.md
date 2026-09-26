# Local learning and alternatives to dense hidden states

This study implements the user's requested local-learning, energy-module, and
full energy-model comparisons. It also evaluates a discrete graph-energy
control motivated by the proposed alternatives to continuous latent spaces.

## What each model actually does

| Model | State | Training | Prediction |
|---|---|---|---|
| BP | Dense hidden activations | Backpropagation of squared prediction error | Feedforward output |
| PC | Dense hidden activations and local prediction errors | Activity descent, then detached local weight gradients | Feedforward output |
| PC-ALM | Dense hidden activations, errors, and temporary dual variables | Activity descent and dual ascent, then detached local weight gradients | Feedforward output |
| Full conditional EBM | Dense hidden activations for each candidate answer | Approximate envelope gradient of discriminative energy loss | Relax all candidate answers; choose minimum energy |
| Energy memory module | Explicit precision/information memory plus learned per-block readers | Fixed causal writer; reader parameters train with the surrounding model | Analytic memory mode read inside each residual predictor |
| Discrete graph energy | Entity IDs, typed edges, candidate path assignments | Supplied rules; no training | Exact min-sum dynamic programming |
| Learned discrete energy | The same entity IDs and explicit feature factors | Seven scalar weights trained by exact path marginalization | Candidate free energies from sum-product |

The PC-ALM implementation follows the primal/dual order of Algorithm 1 in
[Seely and Gould](https://arxiv.org/html/2605.31022). Updates are simultaneous
across layers. Duals reset for each example. The final activity step has no
following dual update. State inference is detached; local parameter derivatives
do not differentiate through the iteration history. The residual architecture,
initialization, optimizer, and bounded subsets here differ from Sakana's setup.
The [official reference](https://github.com/SakanaAI/pc-alm) is pinned locally at
`660747f61a8a7e547c0ecd2c48c8883380a7d1f6` for algorithm inspection.

Our independent full EBM uses

\[
E_\theta(x,y,h)=\tfrac12\|f_{out}(h_L)-e_y\|^2+
\tfrac{\rho}{2}\sum_i\|h_i-f_i(h_{i-1},M)\|^2.
\]

The input is fixed. Every possible answer receives the same initialization and
number of relaxation steps. Prediction has no gold-label argument. Training
uses cross-entropy of the negative candidate energies at temperature 0.1, plus
0.1 times the positive example's energy. Since the hidden solve is finite, the
detached derivative is only an approximate envelope gradient; stationarity is
reported rather than assumed. This full EBM is our experiment, not PC-ALM.

Memory has quadratic coefficients `P` and `S`, with mode `P^-1 S`. The new fixed
writer uses supplied orthogonal addresses; its edits are checked against the
existing full-precision congruence operator. SET replaces a row factor, DELETE
stores unknown, and HOLD changes no bytes. Each neural block has its own learned
query and value projection. Blocks share one factual memory backing store;
this experiment does not claim independently learned stores or learned writes.

## Registered comparisons

- Fashion-MNIST: hidden depths 8 and 32, width 32, BP/PC/PC-ALM, plus a full EBM
  at depth 8; seeds 11/22/33; 400 updates per fit. Fixed subsets contain
  6,000/1,000/2,000 training/validation/test images. This is 21 fits.
- Structured graph tasks: depth 4, width 48, BP/PC/PC-ALM/full EBM, each with and
  without learned energy-memory readers; three seeds; 1,000 updates per fit.
  There are 4,096/512/1,200 episodes. This is 24 fits. Every neural arm receives
  the same complete current fact table, entity, and relation query.
- Next-byte prediction: depth 3, width 32, context 8 bytes, all 256 byte labels;
  BP-MSE, BP-CE, PC-ALM, and full EBM; three seeds; 200 updates per fit.
  There are 4,096/128/256 nonoverlapping windows in disjoint corpus sections.
  This is 12 fits. It is a small next-byte model, not a competitive language model.
- A preliminary 19-fit validation-only pilot checks numerical stability and
  command execution. It never evaluates test examples and is reported separately.
- A supplementary discrete study trains seven scalar factor weights for 400
  updates with seeds 11/22/33 on the same graph data. Factors identify current,
  missing, stale, hypothetical, other-relation, unknown, and absorbing-unknown
  cases. The feature vocabulary and composition algorithm are supplied. It
  learns costs, not the feature predicates or graph execution semantics.
- A follow-up extends the same four byte architectures to 2,000 updates, using
  a disjoint enwik8 segment at offsets 300,000–600,000 and 4,096/512/1,000 windows.
  It was motivated by the short run's weak generation and limited training
  exposure. Its fresh test set is separate from the first byte comparison.

Graph training includes one-hop paths and the two-hop compositions 00, 01, 10.
Validation uses those same path types in fresh graphs. The final test has six
groups: seen paths, withheld 11 composition, unseen three-hop paths, 128 HOLD
events of delay, repeated conflicting updates, and missing information.
Because HOLD is an explicit no-op and the writer is fixed, long-delay success
tests retention by this memory engine, not learned temporal credit assignment.

Models share initialization and batch order within each architecture and seed.
BP/PC/PC-ALM share the MSE objective. The EBM has a different training objective
and more expensive inference. The memory addition increases parameter count;
its matched control is the same memory architecture trained by BP. Counts,
updates, examples, measured CPU time, and per-seed scores are retained.

Validation chooses checkpoints (including initialization) and probability
temperature. All choices freeze before test evaluation. Additional iteration
sweeps use frozen weights and remain diagnostic. Feature probes fit fixed-ridge
linear readouts on first-hop supervision with the network frozen; EBM features
use the model's own predicted answer, never the true answer.

The study keeps previous reports unchanged and uses a durable ledger carrying
forward the prior 716.24 seconds. Training uses a two-hour CPU allocation within
the existing eight-hour cumulative ceiling. The occupied GPU is not used.

## Assessment of the proposed state replacements

**The best fit is an explicit relational workspace with local energy factors
and executable rules.** Candidate states can be graphs, paths, bindings, or
programs. An energy function needs an ordering over candidate configurations;
it does not require continuous embeddings. In a discrete system, updates are
variable substitutions, edge edits, dynamic programming, or combinatorial
search rather than Euclidean gradient descent.

Discrete states do not force REINFORCE or prohibit differentiable learning.
For the tractable chain in this study, exact sum-product computes a partition
function over entity-ID paths. Its derivatives train seven scalar factor
weights by ordinary backpropagation, without dense hidden embeddings. The
challenge for richer graphs or programs is tractable inference and discovering
useful structures, not merely obtaining a gradient.

| Proposal | Fit for energy functions | Assessment for this project |
|---|---|---|
| Typed relational graphs | Strong: local factors penalize incompatible edges, bindings, or paths | Implemented as the discrete control. A plausible primary state for structured memory and reasoning |
| Executable programs / DSLs | Strong: score execution error, constraints, and program complexity | Promising next extension for learned reusable reasoning operations; program search and grounding remain hard |
| Addressable memory | Strong: energy terms can score content consistency and updates | Implemented as quadratic memory. Exact symbolic addresses are possible; continuous memory values still exist in the neural variant |
| Hypergraphs / simplicial complexes | Useful for explicit higher-order relationships | Hyperedges can express multi-object constraints without a learned embedding; homology alone is too lossy for full semantics |
| Persistent homology / topological invariants | Useful auxiliary descriptors or regularizers | Better suited to shape/connectivity diagnostics than replacing all representations |
| Continuous relaxation of discrete choices | Useful optimization tool | Does not remove continuous training states; hard inference can still be discrete |
| Neuromorphic or graph hardware | Potential execution target | Defer until profiling identifies a stable algorithm and actual bottleneck |

Weighted logical constraints already have a probabilistic interpretation in
[Markov logic networks](https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf).
This provides a relevant precedent for discrete energy factors, but inference
over general structures can be combinatorially expensive. Our functional chain
queries are deliberately tractable and do not establish general graph scaling.

[DreamCoder](https://arxiv.org/abs/2006.08381) is evidence that systems can learn
reusable program abstractions. It does not imply that any program representation
automatically reasons correctly or generalizes to all new tasks. A wrong program
can execute perfectly; the specification and evidence still matter.

The [DNC](https://www.nature.com/articles/nature20101) combines a neural
controller with an external memory matrix. It is a useful memory precedent,
but it retains continuous representations. Likewise,
[Gumbel-Softmax](https://arxiv.org/abs/1611.01144) uses a continuous relaxation
for categorical variables; it is not a complete removal of continuous state.

[Persistence stability](https://math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/Edelsbrunner%2C%20Harer%2C%20Stability.pdf)
concerns robustness of persistence diagrams to suitable perturbations. It does
not establish preservation of semantic content. Two differently labeled graphs
can have the same topology while asserting different facts. Labels, direction,
time, and provenance must remain represented somewhere.

Finally, "latent" means unobserved, not necessarily continuous. A hypothesized
symbolic binding is still a latent variable. The practical research objective
is therefore to replace opaque dense state where explicit discrete structure
works better, and to measure any remaining continuous components honestly.
Catastrophic forgetting and expensive memory are not inevitable consequences
of vectors; they depend on training, storage, capacity, and update mechanisms.

## Relationship to earlier repository results

The preceding 98.61% precision-memory and 99.07% classical-control results tested
retrieval with an explicit path-following loop. The new neural graph predictor
must learn the final answer from the supplied facts and query, including how to
compose relations. Its accuracy is not directly comparable to the old retrieval
accuracy. The discrete controls explicitly supply that composition algorithm.
They test the value of this structural assumption and cannot establish that a
neural model has learned to invent it.
