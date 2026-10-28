"""Optional float32 CUDA inference kernel for the two-round latent energy."""
import torch
import triton
import triton.language as tl


@triton.jit
def _two_round(X, A, Phase, SinPhase, Norm2, Maximum, Gram, Basis, Out,
               D: tl.constexpr, M: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    m = tl.arange(0, BM)
    x = tl.load(X + row * D + d, d < D, other=0.)
    weight = tl.load(A + m[:, None] * D + d[None, :], (m[:, None] < M) & (d[None, :] < D), other=0.)
    projected = tl.sum(weight * x[None, :], axis=1)
    # libdevice tanh preserves float32 accuracy without an exp-based rewrite.
    amplitude = tl.extra.cuda.libdevice.tanh(projected) * (M ** -.5)
    norm2 = tl.load(Norm2 + m, m < M, other=0.)
    eta = .95 / (tl.load(Maximum) + tl.sum(tl.abs(amplitude) * norm2, axis=0))
    phase = tl.load(Phase + m, m < M, other=0.)
    phase_sin = tl.load(SinPhase + m, m < M, other=0.)
    first = eta * amplitude * phase_sin
    gram = tl.load(Gram + m[:, None] * M + m[None, :], (m[:, None] < M) & (m[None, :] < M), other=0.)
    angle = tl.sum(first[:, None] * gram, axis=0) + phase
    second = eta * amplitude * tl.extra.cuda.libdevice.sin(angle)
    w = tl.load(Basis + m[:, None] * D + d[None, :], (m[:, None] < M) & (d[None, :] < D), other=0.)
    wq = tl.load(Basis + (m[:, None] + M) * D + d[None, :], (m[:, None] < M) & (d[None, :] < D), other=0.)
    delta = tl.sum((first + second)[:, None] * w - (eta * first)[:, None] * wq, axis=0)
    tl.store(Out + row * D + d, x + delta, d < D)


def prepared_two_triton(anchor, weight, phase, phase_sin, norm2, maximum, gram, basis):
    if anchor.device.type != 'cuda' or anchor.dtype != torch.float32:
        raise ValueError('fused energy requires CUDA float32')
    width = anchor.shape[-1]; modes = weight.shape[0]
    if width != weight.shape[1] or not weight.is_contiguous():
        raise ValueError('anchor width must match contiguous energy weights')
    if not (1 <= width <= 128 and 1 <= modes <= 32):
        raise ValueError('fused energy supports width <= 128 and modes <= 32')
    x = anchor.contiguous()
    output = torch.empty_like(x)
    if not x.numel():
        return output
    _two_round[(x.numel() // width,)](x, weight, phase, phase_sin, norm2, maximum, gram, basis, output,
                                    width, modes, triton.next_power_of_2(width), triton.next_power_of_2(modes),
                                    num_warps=4, enable_fp_fusion=False)
    return output
