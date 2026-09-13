# Energy-state native backbone and the Spark experiment

## What is carried

Each head retains `P[batch, slots]` and `S[batch, slots, value_width]`:

\[
E_t(W)=\sum_i[\tfrac12P_{t,i}\|W_i\|^2-S_{t,i}^T W_i].
\]

Up to an irrelevant additive constant, those coefficients determine a positive diagonal quadratic. Temporary preferred values are `M=S/P`. A stateless writer derives key k, query q, value v, positive strength w, retention f and output gates from the current layer input. Updates remain the v21 equations:

\[
P_{t,i}=\lambda+f_{t,i}(P_{t-1,\pi(i)}-\lambda)+w_tk_{t,i}^2,
\quad
S_{t,i}=f_{t,i}S_{t-1,\pi(i)}+w_tk_{t,i}v_t.
\]

The first query combines rows of M. A second query depends on the current input and the first response and reads the same M again. Pre-normalized residual/MLP blocks and a tied byte embedding/output projection complete the network. There is no second carried answer vector, token KV cache, protected categorical address namespace, parser, or Fourier basin search in the energy candidate.

The coefficients remain numerical hidden state. This interface does not abolish vectors. The independently defined diagonal slot factors omit arbitrary cross-slot correlations; they are not lossless dense regression in disguise. The read is convex and closed-form, so a saddle escape cannot improve its exact minimizer. Learned prediction can still be wrong.

## What changed in this kit

The generic energy recurrence and the supplied v21 learned weights are unchanged. The model runs the portable Torch scan by default. Its optional old Numba CPU path is not treated as a CUDA kernel. Its `cpu_scan=False` path no longer imports Numba, avoiding an unnecessary compiled dependency on Arm.

The original softmax baseline constructed a quadratic boolean mask even for a cold, unsegmented document. The new baseline uses `scaled_dot_product_attention(..., is_causal=True)` for that precise case, permitting PyTorch to select its supported kernels. Explicit document resets or cached prefixes still use the original causal/segment mask. Cold full-window baselines use `chunk=sequence_length`; energy uses chunk128. This is the same attention semantics up to numerical grouping, not a claim that FlashAttention ran on untested hardware. Both outputs and gradients are checked against the original implementation.

Training is now configurable by microbatch, gradient accumulation, sequence length, precision, activation recomputation, and explicit budgets. Inference chunks bound temporary state but do not make training memory constant: autograd still crosses chunks. Non-reentrant checkpointing recomputes block work instead of storing all internal activations. It costs extra computation; actual peak memory must be measured.

## Mathematical storage accounting

For L layers, H heads, K slots, V value dimensions, batch B, float32 energy state requires

\[
4BLHK(V+1)\text{ bytes}.
\]

For the 973K model: L=4,H=4,K=16,V=40, or 41,984 bytes/sequence.

For the 3.29M model: L=4,H=4,K=16,V=64, or 66,560 bytes/sequence.

For the 10.61M model: L=6,H=4,K=16,V=96, or 148,992 bytes/sequence.

Those are INFERENCE STATE payloads. One full sequence of coefficient histories alone is approximately T times larger before scan/autograd intermediates and temporary features. Parameter+gradient+two-FP32-Adam-moment accounting alone is roughly 16 bytes per unique parameter; it excludes activations, allocations, kernels, extra copies, and checkpoint I/O. BF16 autocast does not imply the FP32 master parameters or optimizer moments disappear.

A finite fixed state cannot retain arbitrary unlimited independent observations. The prior retention task and language study remain the evidence base. A working 10M configuration and a seven-update fit benchmark are not a trained 10M model or a scaling law.

## Training objective and data

Every language step minimizes ordinary next-byte cross entropy. No gold event/write/address labels are passed in training or inference. Batches sample document-contained windows; states reset between those windows. Gradient accumulation averages equally sized microbatches, and the tests compare it with a full batch. This is not long-document state-carrying pretraining.

Document shards are local UTF-8 bytes. Exact content groups are separated before splitting; an indexed shard never crosses a document boundary. Dataset and source hashes are resume identity checks, not factual verification or a rights review. Checkpoint selection uses validation loss; the test partition is untouched by the pilot trainer. The sample test corpus included in numerical audits is synthetic and is never called a natural-language benchmark.

## Runtime acceptance is not model-quality acceptance

The acceptance thresholds test specified implementation tolerances, correct execution and limited shape feasibility. They do not measure scientific superiority or justify serious pretraining. The Spark has an Arm CPU and GB10 GPU; actual device probes and computation are required. Unified memory makes separate 'VRAM available' assumptions unreliable, so the report keeps `/proc/meminfo`, Torch allocated/reserved statistics and CUDA memory reports distinct. No cache-dropping or driver-modifying commands are run.

Use the kernel profile and paired validation/time curves to decide whether a custom CUDA/Triton scan is the next worthwhile implementation project. Do not claim a fused CUDA scan exists merely because the CPU adjoint is fast.
