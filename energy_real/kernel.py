"""Exact chunkwise kernel for the energy mixer.

The reference implementation (spark_kit/energy_spark/model.py) runs a
Hillis-Steele scan over a packed [B,T,H,K,1+V] tensor. That scan performs
log2(T) rounds of `torch.cat`, so autograd retains ~log2(T) full copies of the
packed state history -- ~1.4 GiB per layer at width 384 / L6 / T512, which is
why the reference mixer is ~30x slower than softmax attention.

This module computes the identical function with a different factorisation.

Reference recurrence, per (batch, head), in co-moving slot coordinates (the
fixed cyclic slot transport is folded into the coordinates exactly as the
reference `transported_scan` does):

    A_t[k] = f_t[k] * A_{t-1}[k] + w_t * k_t[k]^2       (precision, minus prior)
    S_t[k] = f_t[k] * S_{t-1}[k] + w_t * k_t[k] * v_t   (information)
    P_t[k] = prior + A_t[k];  mode_t = S_t / P_t;  r_t = sum_k q_t[k] * mode_t[k]

The reference materialises `mode`, a [B,T,H,K,V] tensor, at every timestep.
The key observation is that P_t[k] is a *scalar per slot*, so

    r_t = sum_k (q_t[k] / P_t[k]) * S_t[k]  =  u_t . S_t ,   u_t := q_t / P_t

is **linear in S_t**. That lets us (1) scan only A, a [B,T,H,K] tensor with no
value axis, and (2) contract u against S with the standard chunkwise
linear-attention algorithm, which never materialises S_t for every t.

Two explicitly selected paths compute the same recurrence:

* ``dmat`` (default) materialises the [B,N,C,C,H,K] causal decay tensor. It
  supports strong forgetting and has no data-dependent algorithm switch.
* ``matmul`` folds decay into query/key factors, anchored at the first position
  of each chunk. It rejects log-decay spans above SAFE_LOG_SPAN. It never
  silently falls back: a future-driven backend switch would change rounding
  in earlier outputs. Use it only for a separately identified experiment.

Neither path divides by a cumulative product, so the underflow hazard that
motivated the reference's scan choice does not arise.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor

# float32 overflows near exp(88), but the masked entries of the intra-chunk
# product reach exp(span) * max|w| * max|u| * slots before they are discarded.
# With |w| <= 20, |u| <= 1/prior and 16 slots that is ~1e3 * exp(span), so a span
# of 80 sits within a factor of five of overflow. 20 leaves a wide margin and
# still covers any decay that is not already numerically complete forgetting
# (exp(-20) ~ 2e-9 over a single chunk).
SAFE_LOG_SPAN = 20.0


def comoving_index(t: int, k: int, strides: Tensor, device) -> Tensor:
    """co[t,h,k]: slot that head h's slot k occupies at time t (time is 1-indexed)."""
    time = torch.arange(1, t + 1, device=device)
    slots = torch.arange(k, device=device)
    return (slots[None, None, :] + time[:, None, None] * strides[None, :, None]) % k


def _pad_to_chunk(x: Tensor, chunk: int):
    t = x.shape[1]
    rem = (-t) % chunk
    if rem == 0:
        return x, 0
    pad = [0, 0] * (x.ndim - 2) + [0, rem]
    return torch.nn.functional.pad(x, pad), rem


def chunk_gla(u: Tensor, wk: Tensor, v: Tensor, lf: Tensor, s0: Tensor, chunk: int, force: str = 'dmat'):
    """Chunkwise gated linear attention with per-slot decay.

    u  [B,T,H,K]  query, already divided by precision
    wk [B,T,H,K]  w_t * k_t
    v  [B,T,H,V]
    lf [B,T,H,K]  log decay
    s0 [B,H,K,V]  incoming state

    Returns (r, s_final, ctx). ``ctx`` lets a second query read the same states
    via :func:`read_again` without redoing the recurrence.
    """
    if force not in ('matmul', 'dmat'):
        raise ValueError('Select a fixed backend: matmul or dmat; automatic switching is not causal')
    if type(chunk) is not int or chunk < 1 or u.ndim != 4 or min(u.shape) < 1:
        raise ValueError('Positive chunk and nonempty [B,T,H,K] queries required')
    b, t, h, k = u.shape
    if wk.shape != u.shape or lf.shape != u.shape or v.ndim != 4 or v.shape[:3] != (b, t, h):
        raise ValueError('Kernel input shapes differ')
    vdim = v.shape[-1]
    if s0.shape != (b, h, k, vdim):
        raise ValueError('Incoming state shape differs')
    if t == 1:
        if force == 'matmul' and not bool(torch.isfinite(lf).all()):
            raise ValueError('forced matmul path outside its validity domain: nonfinite log decay')
        decay = lf[:, 0].exp()
        ctx = _StepCtx(wk[:, 0], v[:, 0], decay, s0)
        final = decay[..., None] * s0 + wk[:, 0, ..., None] * v[:, 0, :, None, :]
        return _read_step(u, ctx), final, ctx
    up, rem = _pad_to_chunk(u, chunk)
    wkp, _ = _pad_to_chunk(wk, chunk)
    vp, _ = _pad_to_chunk(v, chunk)
    lfp, _ = _pad_to_chunk(lf, chunk)          # padded steps get log(1)=0 -> identity
    if rem:
        # zero the padded inputs so they contribute nothing to the state
        wkp = torch.cat((wkp[:, :t], wkp[:, t:] * 0), 1)
        vp = torch.cat((vp[:, :t], vp[:, t:] * 0), 1)
    n = up.shape[1] // chunk
    u_ = up.view(b, n, chunk, h, k)
    wk_ = wkp.view(b, n, chunk, h, k)
    v_ = vp.view(b, n, chunk, h, vdim)
    lf_ = lfp.view(b, n, chunk, h, k)

    lc = lf_.cumsum(2)                                  # [B,N,C,H,K], inclusive
    lc_last = lc[:, :, -1]                              # [B,N,H,K]
    causal = torch.ones(chunk, chunk, device=u.device, dtype=torch.bool).tril()

    use_matmul = force == 'matmul'
    if use_matmul:
        span = float((lc - lc[:, :, :1]).detach().abs().max())
        if not math.isfinite(span) or span > SAFE_LOG_SPAN:
            raise ValueError(
                f'forced matmul path outside its validity domain: log-decay span '
                f'{span:.1f} > {SAFE_LOG_SPAN}. Select dmat for a separate run.')
    if use_matmul:
        shifted = lc - lc[:, :, :1]                     # <= 0, depends only on the chunk start
        keys = wk_ * (-shifted).exp()
        dmat = None
    else:
        shifted = keys = None
        d = lc.unsqueeze(3) - lc.unsqueeze(2)           # [B,N,C,C,H,K]
        # Mask before exponentiating. Masked entries have a positive exponent and
        # overflow to inf; torch.where would then backpropagate 0 * inf = NaN.
        dmat = d.masked_fill(~causal[None, None, :, :, None, None], float('-inf')).exp()

    # each source position's decayed contribution to the end of its own chunk
    kv_local = torch.einsum('bnchk,bnchv->bnhkv', wk_ * (lc_last.unsqueeze(2) - lc).exp(), v_)
    decay_chunk = lc_last.exp()                         # [B,N,H,K]

    # sequential recurrence over chunks only -- N steps, not T
    starts, s = [], s0
    for i in range(n):
        starts.append(s)
        s = decay_chunk[:, i, ..., None] * s + kv_local[:, i]
    starts_t = torch.stack(starts, 1)                   # [B,N,H,K,V]

    ctx = _Ctx(wk_, keys, v_, lc, shifted, dmat, causal, starts_t, chunk, n, t, b, h, vdim)
    return _read(u_, ctx), s, ctx


class _Ctx:
    __slots__ = ('wk', 'keys', 'v', 'lc', 'shifted', 'dmat', 'causal', 'starts',
                 'chunk', 'n', 't', 'b', 'h', 'vdim')

    def __init__(self, wk, keys, v, lc, shifted, dmat, causal, starts, chunk, n, t, b, h, vdim):
        (self.wk, self.keys, self.v, self.lc, self.shifted, self.dmat, self.causal,
         self.starts, self.chunk, self.n, self.t, self.b, self.h, self.vdim) = (
            wk, keys, v, lc, shifted, dmat, causal, starts, chunk, n, t, b, h, vdim)


class _StepCtx:
    __slots__ = ('wk', 'v', 'decay', 'initial')

    def __init__(self, wk, v, decay, initial):
        self.wk, self.v, self.decay, self.initial = wk, v, decay, initial


def _read_step(u: Tensor, c: _StepCtx) -> Tensor:
    inter = torch.einsum('bhk,bhkv->bhv', u[:, 0] * c.decay, c.initial)
    intra = (u[:, 0] * c.wk).sum(-1)[..., None] * c.v
    return (inter + intra)[:, None]


def _read(u_: Tensor, c: _Ctx) -> Tensor:
    # contribution of the state entering this chunk
    inter = torch.einsum('bnchk,bnhkv->bnchv', u_ * c.lc.exp(), c.starts)
    # contribution of positions inside this chunk
    if c.dmat is None:
        att = torch.einsum('bnchk,bndhk->bnchd', u_ * c.shifted.exp(), c.keys)
        att = torch.where(c.causal[None, None, :, None, :], att, att.new_zeros(()))
    elif u_.device.type == 'cpu':
        # Explicit slot reduction avoids the expensive tiny batched matrix
        # products selected for this three-operand contraction on CPU.
        att = (u_[:, :, :, None] * c.wk[:, :, None] * c.dmat).sum(-1).permute(0, 1, 2, 4, 3)
    else:
        att = torch.einsum('bnchk,bndhk,bncdhk->bnchd', u_, c.wk, c.dmat)
    intra = torch.einsum('bnchd,bndhv->bnchv', att, c.v)
    return (inter + intra).reshape(c.b, c.n * c.chunk, c.h, c.vdim)[:, :c.t]


def read_again(ctx: _Ctx | _StepCtx, u2: Tensor) -> Tensor:
    """Second read against the same states, reusing the saved chunk context."""
    if isinstance(ctx, _StepCtx):
        return _read_step(u2, ctx)
    u2p, _ = _pad_to_chunk(u2, ctx.chunk)
    return _read(u2p.view(ctx.b, ctx.n, ctx.chunk, ctx.h, -1), ctx)
