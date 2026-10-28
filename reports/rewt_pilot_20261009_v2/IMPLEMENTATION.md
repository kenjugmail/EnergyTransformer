# What the REWT pilot implements

The implementation is in the independent rewt package. It does not reuse an old
trained model. The four primary controls have 249,929–250,005 trainable parameters
and share 128-byte reset boundaries, batches, optimizer settings and three seeds.

| Proposal component | Working implementation |
|---|---|
| One joint conditional energy | model.py, energy(): a positive quadratic plus 16 learned cosine modes; full modes span every workspace and the 256 output logits. |
| Functional state blocks | Recurrent, attention, feedforward and output states; the full model adds four forecast horizons, diffusion, three specialists and communication. Decoder/communication compatibility and the joint modes connect their values to prediction. |
| Recurrent state | Accepted workspace carried between observed bytes, reset at the common window boundary. The previous recurrent state is bounded with tanh only when conditioning a new guess. |
| Attention energy | Observed-history attention initializes a guess; a squared recurrent/attention compatibility term participates in the same total energy and gradient. |
| Output refinement | Output logits are part of the joint state, refined alongside hidden states before softmax. |
| Fourier landscape | Context-dependent amplitudes, learned phases and vector-norm-bounded frequency directions. This is a finite energy family, not reconstruction of every possible landscape. |
| Past-history Fourier features | FFT of observed embeddings only; first two non-DC bins condition the full model. Forecasts and candidates never enter the observed history. |
| Future forecasts | Horizons 1/4/16/64, coupled to candidate communication and the energy. Future hidden-state supervision comes only from a frozen, course-trained Transformer inside the training-loss API. |
| Diffusion proposals | Two noisy candidates, each with three reverse denoising stages. The complete energy chooses among them and a keep candidate. A separate training-only denoising loss teaches the proposal network. |
| Specialists and communication | Three independent learned proposal networks and a coupled communication state. Routing is fixed while comparing candidates and refining the current token. |
| Learned critic/router | Critic trained by relative next-byte utility, router trained to prefer the best candidate. At inference they use observed context and proposals; labels/utilities are unavailable. |
| Energy-only accepted updates | Analytic partial derivatives of the same complete energy; two rounds with bounded backtracking. No accepted state is clipped or edited afterward. |
| Coarse-to-fine optimization | Low-norm modes can propose early updates; every candidate is accepted against the complete energy with all modes. |
| Fourier training-noise filter | Separate causal FFT filter over eight optimizer-gradient samples, tested against no filtering, clipping and EMA. It is distinct from the learned Fourier energy. |
| Fixed inference weights | Inference adapts states/logits only. AdamW is confined to training. |
| Cached coefficients | Input-independent frequencies and quadratic precisions are cached within each forward call, retaining the training graph. The cache never survives an optimizer step. |

The quadratic terms include diagonal anchors and squared linear residuals for
attention, decoding, communication and specialist agreement. Their fixed
coefficients are 0.04/0.04/0.04/0.02. Diagonal precision is learned in the positive
range 0.15–2.0; frequency-vector norms are bounded by 2.0.

Training uses next-byte cross-entropy plus ranking (0.02), forecast (0.02),
denoising (0.01), critic (0.01) and routing (0.01) losses where applicable.
The first 24 primary LM updates train proposals before two-round joint training.
Reasoning adaptation starts from copies of those course-trained models and uses
two refinement rounds throughout. No future target is passed to a model forward.

| Evaluation | Saved evidence |
|---|---|
| Prediction/confidence | Held-out bits per byte, next-byte accuracy and multiclass Brier score. |
| Text output | All 30 fixed test prompts × four models × three seeds, each with 96 actual sampled bytes; raw hex and readable outputs retained. |
| Text quality | Fixed anonymous 1–5 fluency/coherence/relevance rubric, plus printable/repetition diagnostics. Human scores remain pending; any assistant review is explicitly labeled. |
| Reasoning quality | Autoregressive arithmetic/rule-chain answers checked for exact final answers and every intermediate step; withheld instances/templates and four-step chains. |
| Recall | Distractor tasks checked separately, with delay recorded. |
| Cost | Process CPU/wall time, gradient/energy calls, shared forecast-teacher preparation and isolated inference memory/timing profiles. |
| Mechanism | Accepted-energy traces, causality/fixed-weight/chunk-replay checks and independent derivative checks. |
| Component studies | Single-seed short retraining for modes/frequencies/scheduling/filtering; separately labeled inference interventions and round/state/output sweeps. |

The first attempt exposed an unbounded recurrent-conditioning instability and
was stopped. Its source and partial traces remain in the neighboring failed-run
folder. A replay reproduced the failure at update 26; the bounded-conditioning
replay remained finite. All models in this completed-protocol version restart
from fresh random weights; failed-run results are excluded.

This CPU pilot has a smaller budget than the proposal's full course study. Its
results can support feasibility and implementation claims. They cannot establish
general reasoning or competitive text generation at scale.
