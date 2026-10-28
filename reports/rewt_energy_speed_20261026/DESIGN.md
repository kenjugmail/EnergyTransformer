# Faster execution of the same latent energy

This changes execution of the October 26 streaming model. It does not change its
weights, energy, number of updates, or training objective, and does not retrain it.
Floating-point reduction order changes, so equivalence is checked numerically
and by greedy continuations, rather than asserted to be bitwise equality.

For an anchor `h`, the original head defines amplitude `a`, frequencies `W`,
phase `phi`, diagonal precision `q`, and the same conservative step size `eta`:

```text
a = tanh(A h) / sqrt(number_of_modes)
E(delta) = 1/2 sum(q * delta^2) + sum(a * cos(W delta + phi))
eta = .95 / (max(q) + sum(abs(a) * row_squared_norm(W)))
delta[0] = 0
delta[k+1] = delta[k] - eta * gradient(E(delta[k]))
output = h + delta[2]
```

The differentiable path skips the first projection of an all-zero displacement.
Its first update is simply `delta[1] = eta * (a * sin(phi)) @ W` in row-vector
notation. Subsequent updates and the gradient through the step size are retained.
The original diagnostic path still supplies the complete energy trace.

For two-round inference, prepare `G = W @ W.T` and `B = concat(W, W * q)` once
per session. Then the exact algebra is:

```text
c1 = eta * a * sin(phi)
c2 = eta * a * sin(c1 @ G + phi)
delta[2] = concat(c1 + c2, -eta * c1) @ B
```

This replaces a projection through the full latent width with a small Gram
matrix. The norm bound and `eta` are unchanged. General iteration counts retain
the direct recurrence after the simplified first step.

For at most 16 float32 CPU rows, the prepared path uses NumPy views and array
operations to reduce PyTorch dispatch overhead. Larger batches stay in PyTorch;
the small-array shortcut was slower on large batches. Returned tensors own their
result storage and cannot be overwritten by a later session call. NumPy is
already a project dependency. Training continues to use PyTorch autograd.

The optional CUDA path fuses amplitude, the curvature bound, both updates, and
the residual addition into one Triton kernel per batch. It currently supports
float32, two rounds, width up to 128, and at most 32 modes. Each program processes
one row, with masked reductions for non-power-of-two dimensions. It uses
libdevice sine/tanh and disables fused multiply-add contraction for this kernel.
It is an inference kernel; it does not substitute an unimplemented backward pass.

`optimize_energy(model)` retains the original Parameter objects and state-dict
keys, so an existing optimizer still updates them and ordinary checkpoints still
load. Enable the execution option after loading. Session consumption checks every
model parameter's identity, version, device, and dtype before using prepared
constants. The internal application method relies on that check; standalone
prepared-head calls perform their own check and reject enabled gradients.

Prepared constants add about 9 KiB versus the previous head at width 64 and
16 modes. This is fixed model-side preparation, separate from the carried token
state counted in the earlier streaming benchmark. Preparation and compilation
costs are excluded from warmed throughput and recorded separately. Existing
compiler caches mean later warmup measurements are not cold-install estimates.

Optional `torch.compile` keeps training differentiable, but compilation is not
automatically beneficial for these small workloads. The benchmark retains that
control and the energy-disabled timing lower bound. The disabled model is not
presented as an equivalent quality replacement.

Implementation references:

- [PyTorch compilation API and mode limitations](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
- [PyTorch compilation startup costs](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html).
- [Triton's fused-operation and reduction tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

The original model/training sources and all trained checkpoint bytes remain
unchanged. Earlier decoder/sampler implementations and intermediate optimization
sources are retained under `baseline/`; the earlier benchmark hashes refer to
those versions. Historical benchmark JSON files are retained alongside the final
`cpu_verified.json` and `cuda_verified.json` measurements.
