"""Enumerated future studies; no execution or budget authorization."""
from dataclasses import asdict,replace
from .spec import config,SEEDS,source_identity
from energy_real.model import LM
import torch


def count(cfg):
    with torch.device('meta'):return LM(cfg).parameters_count()


def describe(cfg):
    energy=sum(cfg.mixer_kind(i)=='energy' for i in range(cfg.layers))
    return {'config':asdict(cfg),'parameters':count(cfg),
            'fp32_recurrent_bytes_per_sequence':4*energy*cfg.heads*cfg.slots*(cfg.width//cfg.heads+1),
            'bf16_local_kv_bytes_per_sequence':4*(cfg.layers-energy)*cfg.attention_window*cfg.width}


def design(d0=None,selected_family=None):
    if d0 is not None and (type(d0)is not int or d0<2**18 or d0>2**24 or d0&(d0-1)):raise ValueError('D0 must be a pilot power-of-two budget from 2^18 through 2^24')
    if selected_family not in (None,'energy','hybrid'):raise ValueError('Select energy family using validation evidence')
    scaling=[]
    for family in ('softmax',selected_family or 'validation_selected_energy_family'):
        for size in range(4):
            for numerator,denominator in ((1,4),(1,2),(1,1),(2,1)):
                for seed in SEEDS:
                    scaling.append({'family':family,'size':size,'seed':seed,'budget_ratio':[numerator,denominator],
                                    'targets':None if d0 is None else d0*numerator//denominator,
                                    'independent_run':True,'schedule':'warmup/cosine ending at this data budget'})
    target=count(config('hybrid',1));slots=[]
    for family in ('energy','hybrid'):
        for k in (8,16,32,64):
            cfg=replace(config(family,1),slots=k)
            cfg=replace(cfg,ff_width=cfg.ff_width+round((target-count(cfg))/(cfg.layers*(2*cfg.width+1))))
            if cfg.ff_width<1 or abs(count(cfg)/target-1)>.005:raise ValueError('Slot study cannot match parameters')
            for seed in SEEDS:slots.append({'family':family,'seed':seed,**describe(cfg)})
    for seed in SEEDS:slots.append({'family':'softmax','seed':seed,**describe(config('softmax',1))})
    return {'source_sha256':source_identity(),'execution_authorized':False,'d0':d0,'selected_family':selected_family,
            'scaling_runs':scaling,'slot_capacity_runs':slots,
            'slot_protocol':'Paired data and equal validation-only tuning; fixed parameter budget trades FF capacity for slots. Report this tradeoff. Delayed-copy/associative recall plus LM loss and elapsed cost.',
            'slot_additional_control':'A later fixed-FF slot sweep is required to separate memory capacity from FF tradeoffs; not part of these 27 fits.',
            'cost_estimate':None,'cost_status':'Await measured GPU profiles; do not extrapolate CPU rates',
            'scaling_limits':'96 independent fits; re-profile selected family. Need larger measured size after this grid before extrapolation. No billion-parameter claim.'}
