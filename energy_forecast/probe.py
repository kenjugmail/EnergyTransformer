"""Causal, checkpoint-agnostic sketches of the current energy_real LM.

Uses the declared final-layer P/S. This is a lossy 2H-dimensional sketch, NOT a
forecast of every coefficient or a replacement state. Does not alter LM weights.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import Tensor

def state_sketch(state) -> Tensor:
    p,s=state[-1]
    if p.ndim!=3 or s.shape[:3]!=p.shape or (p<=0).any():
        raise ValueError('Expected positive final-layer P and compatible S')
    # A fixed, deterministic projection per head; no fit on test data.
    size=s.shape[-2]*s.shape[-1]
    i=torch.arange(size,device=s.device,dtype=s.dtype)
    basis=torch.cos((i+.5)*1.618033988749895)
    basis=basis/basis.norm()
    mode=(s/p[...,None]).flatten(-2)
    return torch.stack((p.log().mean(-1),(mode*basis).sum(-1)),dim=-1).flatten(1)

@torch.no_grad()
def trace_states(model, tokens: Tensor) -> Tensor:
    """Exact serial causal trace [B,T,2H]; no future-dependent backend selection."""
    if model.config.kind!='energy':raise ValueError('Energy coefficients required')
    model.eval();state=None;trace=[]
    for t in range(tokens.shape[1]):
        _,state=model(tokens[:,t:t+1],state)
        trace.append(state_sketch(state))
    return torch.stack(trace,1)


def aligned_windows(trace: Tensor, history: int, horizons: tuple[int,...], stride: int=16):
    """Targets are after the current byte: at t+h, never t+h-1. One doc only."""
    if trace.ndim!=2 or history<1 or stride<1 or not horizons or any(h<1 for h in horizons):raise ValueError('Invalid trace/window')
    if trace.shape[0] < history + max(horizons):return None
    anchors=torch.arange(history-1,trace.shape[0]-max(horizons),stride,device=trace.device)
    if len(anchors)==0:return None
    past=anchors[:,None]+torch.arange(-history+1,1,device=trace.device)[None]
    future=anchors[:,None]+torch.tensor(horizons,device=trace.device)[None]
    return trace[past],trace[future],anchors


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_energy_model(path, device='cpu'):
    """Load known original or repository checkpoint formats without pickle globals.

Unknown formats fail rather than guessing a config from weight dimensions.
"""
    from energy_real.model import LM,Config
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('format')=='wiki_pilot_v21':
        c=Config(kind=d['kind'],width=d['width'],layers=d['layers'],ff_width=d['ff_width'],chunk=16)
        weights=d['weights']
    else:
        cfg=d.get('config')
        weights=d.get('model',d.get('weights'))
        if not isinstance(cfg,dict) or not isinstance(weights,dict):
            raise ValueError('Need wiki_pilot_v21 or explicit config + model/weights dictionary; convert unknown format first')
        allowed=set(Config.__dataclass_fields__)
        unknown=set(cfg)-allowed
        if unknown:raise ValueError(f'Unexpected model config keys: {sorted(unknown)}')
        c=Config(**cfg)
    if c.kind!='energy':raise ValueError('An energy checkpoint is required')
    m=LM(c).to(device);m.load_state_dict(weights,strict=True);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m
