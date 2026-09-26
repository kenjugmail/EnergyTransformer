"""Causal, checkpoint-agnostic sketches of the current energy_real LM.

Uses the declared final-layer P/S. This is a lossy 2H-dimensional sketch, NOT a
forecast of every coefficient or a replacement state. Does not alter LM weights.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import Tensor

from energy_real.summaries import state_sketch, energy_layer, forward_with_summaries

@torch.no_grad()
def trace_states(model, tokens: Tensor, layer=None) -> Tensor:
    """Frozen causal reference trace, including hybrids with attention last."""
    was_training = model.training
    try:
        model.eval()
        return forward_with_summaries(model, tokens, layer=layer)[2]
    finally:
        model.train(was_training)


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
    energy_layer(c)
    m=LM(c).to(device);m.load_state_dict(weights,strict=True);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m
