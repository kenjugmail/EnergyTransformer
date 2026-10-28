# Faster streaming generation — October 27, 2026

The complete frozen streaming decoder now has a fixed-shape token path, compiled CPU/CUDA execution and optional CUDA graph replay. At an 8,192-token prefix, the energy-enabled model generates 32 tokens in **8.18 ms instead of 15.43 ms on CPU (1.89×)** and **1.97 ms instead of 24.51 ms on the shared GB10 (12.44×)**. This is an execution improvement with the same trained weights and two refinement rounds; accuracy has not been retrained or improved.

## What changed

[FastDecodeSession](../../rewt/fast_decode.py) uses reference chunked prefill, then a fixed-shape one-token computation. Window attention writes a ring buffer instead of concatenating its history. Delta memory uses the one-token recurrence directly. Compilation combines small tensor operations, and CUDA graphs replay the GPU step. Generated IDs remain on the device until the complete output is returned, removing a host synchronization per token. External inputs and model weights are checked at API boundaries. Returned outputs have independent ownership.

The same decoder is available to full-attention, window-attention, hybrid and energy-hybrid controls. This prevents an optimization available to every model from being misreported as a unique architecture advantage. The old decoder and all checkpoints remain usable.

## Matched warmed generation measurements

Each cell is the median of nine alternating repetitions. One CPU thread pinned to core 5; NVIDIA GB10; float32; TF32 disabled. Generate 32 greedy tokens and return them to CPU, batch 1. All backends use identical prompts and weights within each model; generated IDs match. Prefixes are seeded random token IDs for timing, not recall questions. One language checkpoint per architecture, seed 26301.

| Model | CPU previous, 128 | CPU compiled, 128 | CPU previous, 8192 | CPU compiled, 8192 |
|---|---:|---:|---:|---:|
| Full Transformer | 11.59 ms | 7.24 ms | 21.12 ms | 16.24 ms |
| Window Transformer | 11.58 ms | 7.18 ms | 11.49 ms | 7.14 ms |
| Hybrid | 14.25 ms | 7.54 ms | 14.13 ms | 7.59 ms |
| Hybrid + energy | 15.36 ms | 8.09 ms | 15.43 ms | 8.18 ms |

| Model | GPU previous, 128 | GPU graph, 128 | GPU previous, 8192 | GPU graph, 8192 |
|---|---:|---:|---:|---:|
| Full Transformer | 21.91 ms | 2.54 ms | 52.60 ms | 41.86 ms |
| Window Transformer | 22.72 ms | 1.85 ms | 22.41 ms | 1.86 ms |
| Hybrid | 23.76 ms | 1.83 ms | 23.50 ms | 1.78 ms |
| Hybrid + energy | 25.08 ms | 2.01 ms | 24.51 ms | 1.97 ms |

The new hybrid without refinement is **2.14× faster than the equally compiled full Transformer** at the 8K prefix on CPU, and carries **225× less cache/state**: 37,392 versus 8,421,392 bytes. The window Transformer remains faster on CPU; its bounded cache is 65,552 bytes. These counts include recurrent/attention state and token/position buffers, but exclude parameters, logits, prepared constants, graph pools and compiler workspace. They are not total RAM/VRAM consumption.

The GPU result includes removal of Python dispatch and per-token CPU transfers. It is not a 12× arithmetic-kernel speedup or a training-speed claim. GPU utilization was 3% before and 44% after the pass; other machine activity can affect absolute timings. The full-attention compiled/graph path scans its allocated capacity, which equals prefix plus 32 here. Oversizing capacity can worsen its latency. Bounded hybrid state does not grow with prefix length.

The previous sampler already uses the preceding fast energy optimization. Compare the paired numbers here; earlier reports were measured under different GPU load. Eager and compiled intermediate controls, individual samples and checkpoint/source hashes are in [CPU data](cpu.json) and [GPU data](cuda.json).

## Startup and prompt cost

Prefill, preparation, compilation and capture are excluded from the generation tables and recorded separately. For the energy hybrid, the first compiled CPU preparation in this process took 1.23 seconds; first compiled CUDA preparation took 1.07 seconds. Disk compiler caches may already be populated, so these are not clean-install compilation guarantees. Graph preparation after compiler warmup took about 6–8 ms. The 8K prompt took about 122 ms on CPU and 107 ms on GPU in these setup samples. First-use costs can dominate a short request.

For an immediate short demo use `--decoder eager`; use `compile` on CPU or `graph` on CUDA for warmed sessions. The sampler's default remains the reference path. `FastDecodeSession.prepare()` explicitly pays preparation after a prompt is consumed, and `generate()` can then continue the same session repeatedly. This change optimizes decoding, not prefill or training.

## Correctness

- **263 REWT tests pass**: ring wrapping, one-token delta parity, zero/one/two refinement rounds, batch sizes, CPU compilation, CUDA graph replay, repeated generation, output ownership, input/capacity validation and stale-weight rejection. [Test log](tests.log).
- **126/126 checked continuations match** the original energy model: 63 CPU compiled and 63 CUDA graph comparisons across three trained seeds. These cover story prompts and the first validation prompt per task family/depth/wording category. [CPU continuations](continuations_cpu.json), [CUDA continuations](continuations_cuda.json).
- The benchmark separately verifies all four model variants and every execution backend against each model's reference continuation, before and during all repetitions.
- The presentation demo reruns the first saved long-recall example at each length, with all four models and every failure retained. All 12 outputs reproduce the published study responses. This is a replay of known test examples, not a new quality evaluation.

## Reproduce and use

From the repository root, generate with the energy checkpoint on CPU:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt_streaming.py \
  --checkpoint runs/rewt_streaming_20261026/language_hybrid_energy_26301/best.pt \
  --decoder compile --prompt 'Once upon a time, ' --tokens 96
```

For CUDA, add `--device cuda` and replace `--decoder compile` with `--decoder graph`. Fast decoding includes prepared energy math; an additional `--fast-energy` flag is unnecessary for its token step. The earlier energy flags remain available for reference execution/prefill.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 .venv/bin/python scripts/benchmark_rewt_fast_decode.py \
  --device cpu --output reports/rewt_decode_replication_cpu.json
# Repeat with --device cuda and a different output file for GPU.

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/demo_rewt_streaming.py --output reports/rewt_demo_replication
```

Use new output paths; existing measurements are never overwritten by these commands. The complete [showcase brief](../../course_project/presentation_20261028/streaming_showcase/presenter_brief.txt), [memory figure](../../course_project/presentation_20261028/streaming_showcase/memory_showcase.pdf), [speed figure](../../course_project/presentation_20261028/streaming_showcase/decode_speed.pdf) and [recorded demo](../../course_project/presentation_20261028/streaming_showcase/demo/transcript.txt) are ready for October 28.

The strongest measured behavior remains bounded associative recall. Real log understanding, arbitrary conversational memory, robust prose and multi-step autoregressive reasoning remain future work. The earlier [quality study](../rewt_streaming_20261026/FINDINGS.md) includes the weak third Transformer seed and equal-CPU language results; these execution changes do not remove those limitations.
