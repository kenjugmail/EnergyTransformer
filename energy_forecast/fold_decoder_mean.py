"""Fold an affine mean forecaster into the history adapter's first projection.

No training. Identical real-arithmetic function, subject to floating-point grouping
changes. The conditional-mixture/force readers are nonlinear and cannot use this
conversion. This demonstrates an expressivity control, not new future information.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import argparse,json,copy
import torch
from .decoder_bridge import load_adapter,DecoderAdapter
from .experiment import load_head,atomic_save
from .probe import checkpoint_hash

@torch.no_grad()
def fold_mean(adapter, anchor):
    if adapter.config.kind!='mean':raise ValueError('Only affine-mean adapters can be folded')
    first=adapter.first.weight
    c=adapter.config
    out=DecoderAdapter(replace(c,kind='history')).to(dtype=first.dtype,device=first.device)
    if out.config.bottleneck!=c.bottleneck:raise ValueError('Bottleneck mismatch')
    # Compute conversion in float64, then store in the original parameter dtype.
    dtype=torch.float64;device=first.device
    gain=anchor.gain.to(device=device,dtype=dtype).repeat_interleave(c.variables)
    aw=anchor.weight.to(device=device,dtype=dtype)
    const=anchor.constant.to(device=device,dtype=dtype)
    matrix=(aw[:-1]*gain[None]).repeat(1,4)*.5
    intercept=(const+gain*(aw[-1]-const)).repeat(4)*.5
    scale=adapter.feature_scale.to(dtype);center=adapter.feature_center.to(dtype)
    normalized_matrix=matrix/scale[None]
    normalized_intercept=(intercept-center)/scale
    side=first[:,c.width:].to(dtype)
    out.first.weight[:,:c.width].copy_(first[:,:c.width])
    out.first.weight[:,c.width:].copy_(side@normalized_matrix.T)
    out.first.bias.copy_(adapter.first.bias.to(dtype)+side@normalized_intercept)
    out.last.load_state_dict(adapter.last.state_dict())
    out.feature_center.zero_();out.feature_scale.fill_(1)
    return out.eval()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--adapter',required=True);p.add_argument('--head',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    model,old=load_adapter(a.adapter);head,hm=load_head(a.head)
    if old['head_sha256']!=checkpoint_hash(a.head) or hm['teacher_sha256']!=old['teacher_sha256']:raise ValueError('Source model identity mismatch')
    result=fold_mean(model,head.anchor)
    meta=copy.deepcopy(old);meta.update(config=asdict(result.config),weights=result.state_dict(),head_sha256=None,
        conversion={'kind':'affine_mean_fold','source_adapter_sha256':checkpoint_hash(a.adapter),'source_head_sha256':checkpoint_hash(a.head),
                    'training_updates':0,'real_arithmetic_equivalence':True,'float32_bitwise_equivalence':False})
    atomic_save(meta,a.out,overwrite=False)
    print(json.dumps(meta['conversion'],indent=2))
if __name__=='__main__':main()
