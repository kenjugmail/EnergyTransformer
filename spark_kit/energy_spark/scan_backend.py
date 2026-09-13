"""Optional CPU-only O(T) transported affine scan with a first-order adjoint.

Not a CUDA/Triton implementation; no second backward derivatives. The pure Torch
scan remains the reference and the fallback. Compiles on first use; exclude that
cost only in explicitly steady-state benchmarks.
"""
from __future__ import annotations
import numpy as np
import torch
try:
    import numba
except ImportError:
    numba=None

if numba is not None:
    @numba.njit(cache=True,nogil=True)
    def _forward(f,b,initial,strides):
        bs,t,h,k,c=b.shape;y=np.empty_like(b)
        for bi in range(bs):
            for ti in range(t):
                for hi in range(h):
                    for ki in range(k):
                        prev=(ki-strides[hi])%k
                        for ci in range(c):
                            old=initial[bi,hi,prev,ci] if ti==0 else y[bi,ti-1,hi,prev,ci]
                            y[bi,ti,hi,ki,ci]=f[bi,ti,hi,ki]*old+b[bi,ti,hi,ki,ci]
        return y

    @numba.njit(cache=True,nogil=True)
    def _backward(f,y,initial,strides,grad):
        bs,t,h,k,c=y.shape
        df=np.empty_like(f);db=np.empty_like(y);di=np.zeros_like(initial)
        for bi in range(bs):
            carry=np.zeros((h,k,c),dtype=y.dtype)
            for ti in range(t-1,-1,-1):
                previous=np.zeros((h,k,c),dtype=y.dtype)
                for hi in range(h):
                    for ki in range(k):
                        pk=(ki-strides[hi])%k;total=0.
                        for ci in range(c):
                            g=grad[bi,ti,hi,ki,ci]+carry[hi,ki,ci]
                            old=initial[bi,hi,pk,ci] if ti==0 else y[bi,ti-1,hi,pk,ci]
                            total+=g*old;db[bi,ti,hi,ki,ci]=g
                            previous[hi,pk,ci]=g*f[bi,ti,hi,ki]
                        df[bi,ti,hi,ki]=total
                carry=previous
            di[bi]=carry
        return df,db,di

class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,f,b,initial,strides):
        f=f.contiguous();b=b.contiguous();initial=initial.contiguous();strides=strides.contiguous()
        y=torch.from_numpy(_forward(f.detach().numpy(),b.detach().numpy(),initial.detach().numpy(),strides.numpy()))
        ctx.save_for_backward(f,y,initial,strides)
        return y
    @staticmethod
    def backward(ctx,grad):
        if torch.is_grad_enabled():raise RuntimeError('CPU scan supports first-order backward only')
        f,y,initial,strides=ctx.saved_tensors
        grads=_backward(f.detach().numpy(),y.detach().numpy(),initial.detach().numpy(),strides.numpy(),grad.detach().contiguous().numpy())
        return *(torch.from_numpy(a) for a in grads),None


def available(f):
    return numba is not None and f.device.type=='cpu' and f.dtype in (torch.float32,torch.float64)


def cpu_scan(f,b,initial,strides):
    if not available(f):raise RuntimeError('CPU scan unavailable for this device/dtype')
    return _Scan.apply(f,b,initial,strides)
