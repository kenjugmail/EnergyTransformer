"""Optional inference-only fused mode-space solver for CUDA float32.

One program owns a row: all small Gram iterations and the final full-state lift.
Training uses the differentiable PyTorch implementation. Triton is imported only
when this optional module is requested; ordinary CPU installations do not need it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _preconditioned(Anchor, Amp, Phase, Gram, Lift, Step, Output,
                    M: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                    ROUNDS: tl.constexpr, CUT: tl.constexpr, CENTERED: tl.constexpr):
    row = tl.program_id(0)
    m = tl.arange(0, BM)
    d = tl.arange(0, BD)
    amp = tl.load(Amp + row * M + m, m < M, 0)
    phase = tl.load(Phase + m, m < M, 0)
    gram = tl.load(Gram + m[:, None] * M + m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
    eta = tl.load(Step)
    coef = tl.full((BM,), 0, tl.float32)
    for _ in range(ROUNDS):
        projected = tl.sum(coef[:, None] * gram, axis=0)
        candidate = (1 - eta) * coef + eta * amp * tl.sin(projected + phase)
        finite = tl.sum(((candidate == candidate) & (tl.abs(candidate) < float('inf'))).to(tl.int32), axis=0) == BM
        coef = tl.where(finite, candidate, coef)
    lift = tl.load(Lift + m[:, None] * D + d[None, :], (m[:, None] < M) & (d[None, :] < D), 0)
    anchor = tl.load(Anchor + row * D + d, d < D, 0)
    result = anchor + tl.sum(coef[:, None] * lift, axis=0)
    if CENTERED:
        mean = tl.sum(tl.where((d >= CUT) & (d < D), result, 0), axis=0) / (D - CUT)
        result = result - tl.where(d >= CUT, mean, 0)
    tl.store(Output + row * D + d, result, d < D)


@triton.jit
def _preconditioned_tiled(Anchor, Amp, Phase, Gram, Lift, LiftMean, Step, Output,
                          M: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, TILE: tl.constexpr,
                          ROUNDS: tl.constexpr, CUT: tl.constexpr, CENTERED: tl.constexpr):
    # Recompute the small recurrence for each output tile. This trades repeated
    # M*M work for lower lift register pressure and more programs at batch one.
    row = tl.program_id(0)
    m = tl.arange(0, BM)
    d = tl.program_id(1) * TILE + tl.arange(0, TILE)
    amp = tl.load(Amp + row * M + m, m < M, 0)
    phase = tl.load(Phase + m, m < M, 0)
    gram = tl.load(Gram + m[:, None] * M + m[None, :], (m[:, None] < M) & (m[None, :] < M), 0)
    eta = tl.load(Step)
    coef = tl.full((BM,), 0, tl.float32)
    for _ in range(ROUNDS):
        projected = tl.sum(coef[:, None] * gram, axis=0)
        candidate = (1 - eta) * coef + eta * amp * tl.sin(projected + phase)
        finite = tl.sum(((candidate == candidate) & (tl.abs(candidate) < float('inf'))).to(tl.int32), axis=0) == BM
        coef = tl.where(finite, candidate, coef)
    lift = tl.load(Lift + m[:, None] * D + d[None, :], (m[:, None] < M) & (d[None, :] < D), 0)
    result = tl.load(Anchor + row * D + d, d < D, 0) + tl.sum(coef[:, None] * lift, axis=0)
    if CENTERED:
        out = CUT + tl.arange(0, 256)
        anchor_mean = tl.sum(tl.load(Anchor + row * D + out), axis=0) / 256
        lift_mean = tl.load(LiftMean + m, m < M, 0)
        mean = anchor_mean + tl.sum(coef * lift_mean, axis=0)
        result = result - tl.where(d >= CUT, mean, 0)
    tl.store(Output + row * D + d, result, d < D)


def triton_preconditioned(anchor, amplitudes, phase, prepared, *, rounds, output_start,
                          centered=True, tile_size=0):
    tensors = (anchor, amplitudes, phase, prepared['gram'], prepared['lift'], prepared['step'])
    if any(t.device.type != 'cuda' or t.device != anchor.device or t.dtype != torch.float32 or not t.is_contiguous() for t in tensors):
        raise ValueError('Triton refinement requires contiguous CUDA float32 tensors')
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise ValueError('inference kernel has no backward; use the PyTorch solver for training')
    if type(rounds) is not int or rounds < 0:
        raise ValueError('rounds must be nonnegative')
    if anchor.ndim != 2 or amplitudes.ndim != 2 or anchor.shape[0] != amplitudes.shape[0]:
        raise ValueError('anchor and amplitudes must be [batch,dimensions/modes]')
    b, d = anchor.shape; m = amplitudes.shape[-1]
    if b < 1 or m < 1 or type(output_start) is not int or not 0 <= output_start < d:
        raise ValueError('nonempty batch/modes and a valid output block start are required')
    if prepared['step'].numel() != 1 or type(tile_size) is not int:
        raise ValueError('step must be scalar and tile_size must be an integer')
    if prepared['gram'].shape != (m, m) or prepared['lift'].shape != (m, d) or phase.shape != (m,):
        raise ValueError('incompatible prepared mode geometry')
    output = torch.empty_like(anchor)
    if tile_size:
        if tile_size not in (64, 128, 256) or d - output_start != 256:
            raise ValueError('tiles must be 64/128/256 with a 256-dimensional output block')
        if 'output_mean' not in prepared:
            raise ValueError('precompute prepared[output_mean] = lift[:,output_start:].mean(-1).contiguous()')
        mean = prepared['output_mean']
        if mean.shape != (m,) or mean.device != anchor.device or mean.dtype != anchor.dtype or not mean.is_contiguous():
            raise ValueError('invalid prepared output mean')
        if torch.is_grad_enabled() and mean.requires_grad:
            raise ValueError('inference kernel has no backward for the prepared output mean')
        _preconditioned_tiled[(b, triton.cdiv(d, tile_size))](anchor, amplitudes, phase,
            prepared['gram'], prepared['lift'], mean, prepared['step'], output, m, d,
            triton.next_power_of_2(m), tile_size, rounds, output_start, centered,
            num_warps=4, enable_fp_fusion=False)
    else:
        _preconditioned[(b,)](anchor, amplitudes, phase, prepared['gram'], prepared['lift'],
            prepared['step'], output, m, d, triton.next_power_of_2(m), triton.next_power_of_2(d),
            rounds, output_start, centered, num_warps=4, enable_fp_fusion=False)
    return output
