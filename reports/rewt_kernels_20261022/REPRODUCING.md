# Reproduce the October 22 kernel investigation

Run from `/home/kt/Documents/EnergyTransformer`. The virtual environment already
contains the measured PyTorch runtime. No dependencies were installed by this work.
Use a fresh output directory when repeating measurements to retain this record.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests_rewt -q

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/rewt_krylov_reproduction \
  .venv/bin/python scripts/benchmark_rewt_kernels.py \
  --checkpoint runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt \
  --output reports/rewt_kernels_reproduction/benchmark.json
```

The benchmark uses fixed synthetic input tokens and unchanged checkpoint
weights, rotates execution order, reports CPU and wall times, checks float32
predictions and gradients, and records cold calls separately. Training timing
uses a differentiable logit-squared loss, not an optimizer or the curriculum's
auxiliary losses. `--skip-compile` omits the compiler arm. Compilation caches
must start empty if comparing cold starts.

Use the existing narrow arithmetic model with the optional backend:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/sample_rewt.py \
  --checkpoint runs/rewt_learning_20261010/binding/aligned/best.pt \
  --prompt $'A=19;B=A+23;B=?\n' --greedy --stop-newline --max-new 40 \
  --cache-history --refinement-backend krylov2 --decode-cache
```

Checked output: `steps=42;ans=42` followed by a newline. Add
`--compile-refinement` when repeated execution justifies the cold-start cost.
The broader checkpoint's arithmetic remains weaker; changing execution does not
repair those learned responses.

Programmatic use:

```python
from dataclasses import replace
import torch
from rewt.position_memory import load_learning_checkpoint
from rewt.decode import EnergyDecodeSession

loaded, _ = load_learning_checkpoint(checkpoint_path)
model = type(loaded)(replace(loaded.config, cache_history=True,
                            refinement_backend='krylov2'))
model.load_state_dict(loaded.state_dict(), strict=True)
model.eval()
# Optional: model.compile_refinement()
session = EnergyDecodeSession(model)
output = session.prefill(torch.tensor([list(b'A=19;B=A+23;B=?\n')]))
next_byte = output.logits[:, -1].argmax(-1)[:, None]
output = session.step(next_byte)
```

For training, call the ordinary model forward and create no decode session.
Every forward rebuilds differentiable mode matrices. A decode session is an
explicit fixed-weight inference object; create a new one after model changes.
Sessions currently support the core model only. Ordinary defaults and the
reference `refine` method remain available.

Additional checks (create the destination directory first):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/verify_rewt_kernels.py \
  --checkpoint runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt \
  --reference-source reports/rewt_kernels_20261022/baseline_source \
  --validation reports/rewt_polish_20261010_v2/termination/tasks_valid.jsonl \
  --output reports/rewt_kernels_reproduction/verification.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/analyze_rewt_geometry.py \
  --checkpoint runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt \
  --output reports/rewt_kernels_reproduction/geometry.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/profile_rewt_kernels.py \
  --checkpoint runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt \
  --backend krylov2 --output reports/rewt_kernels_reproduction/profile.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python scripts/check_rewt_kernel_cuda.py \
  --checkpoint runs/rewt_polish_20261010_v2/termination/warm_heads2/best.pt \
  --output reports/rewt_kernels_reproduction/cuda_correctness.json
```

The CUDA command checks one four-row conditional forward/backward, not an
end-to-end GPU training or performance benchmark. The profile command warms
each workload, but profiler timings still include instrumentation overhead and
must not replace the benchmark measurements.

Artifacts in this directory:

- `baseline_source/`: the 18 REWT Python files before this task's edits.
- `implementation_source.tar.gz`: the final package, tests and execution scripts.
- `benchmark.json`: final CPU timings, raw repetitions, cold calls and numeric parity.
- `verification.json`: eight archived default-path checks and every validation generation pair.
- `cuda_correctness.json`: compiled CUDA state/gradient parity and allocation peak.
- `geometry.json`: curvature spectra, descent bounds, rank measurements and 12-round preconditioned identity check.
- `profile_reference.json`, `profile_krylov2.json`: warmed CPU operator/stage attribution.
- `tests.log`: the complete REWT suite result.
- `kernel_speedups.png`, `kernel_speedups.pdf`: plots of the measured CPU comparisons.
- `manifest.json`: final source, script, test and artifact hashes.

These files distinguish measurements, algebraic identities and proposed
architecture experiments. No saved training checkpoint or previous report was
overwritten. The existing report's source hashes remain a record of that older
implementation; they are not expected to equal the newly extended package.
