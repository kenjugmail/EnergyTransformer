from dataclasses import asdict
from pathlib import Path
import hashlib
import json
from energy_real.model import Config
from energy_real.train import matched_ff

STAGES = {'acceptance':7200, 'backbone':43200, 'forecast':14400, 'conversion':14400, 'verification':7200}
SIZES = [(160,3), (208,6), (384,6), (512,9)]
FAMILIES = ('softmax','local_softmax','energy','hybrid')
SEEDS = (404,505,606)
MODEL = {'id':'HuggingFaceTB/SmolLM2-135M','revision':'93efa2f097d58c2a74874c7e644dbc9b0cee75a2','layers':[4,9,14,19,24,29]}
DATA = {'id':'HuggingFaceFW/fineweb-edu','revision':'87f09149ef4734204d70ed1d046ddc9ca3f2b8f9','config':'sample-10BT','bytes':256*2**20}


def source_identity():
    root=Path(__file__).resolve().parents[1]; h=hashlib.sha256()
    paths=[]
    for folder in ('energy_real','energy_forecast','energy_campaign','energy_transformer','spark_kit/energy_spark'):
        paths.extend((root/folder).glob('*.py'))
    if (root/'pyproject.toml').is_file():paths.append(root/'pyproject.toml')
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode()+b'\0'+p.read_bytes())
    return h.hexdigest()


def config(family, size, chunk=16, checkpointing=False):
    if family not in FAMILIES or not 0 <= size < len(SIZES):raise ValueError('Unknown family/size')
    w,l=SIZES[size]; n=l if 'softmax' in family else l//3 if family=='hybrid' else 0
    ff=4*w+round((matched_ff(w,4,16,4*w)-4*w)*n/l)
    return Config(kind='softmax' if family=='local_softmax' else family,width=w,layers=l,heads=4,
                  slots=16,ff_width=ff,chunk=chunk,attention_window=128 if family in ('local_softmax','hybrid') else 0,
                  activation_checkpointing=checkpointing)


def campaign_spec():
    return {'format':'energy_campaign_v1','source_sha256':source_identity(),'stage_seconds':STAGES,
            'data':DATA,'donor':MODEL,'seeds':SEEDS,'families':FAMILIES,
            'configs':{f'{f}_{i}':asdict(config(f,i)) for i in range(4) for f in FAMILIES},
            'context':512,'effective_targets':8192,'precision':'bf16_outer_fp32_state',
            'learning_rates':[.0003,.001,.003],'calibration_seed':1701,
            'optimizer':{'name':'AdamW','betas':[.9,.95],'weight_decay':.1,'clip':1.},
            'evaluation':'validation_selection_test_after_freeze','no_automatic_promotion':True}


def choose_budget(seconds_per_target, fixed_seconds, available=32400):
    """All four families and three seeds remain paired as largest sizes drop."""
    for n in range(4,0,-1):
        required={(f,i) for f in FAMILIES for i in range(n)}
        if not required.issubset(seconds_per_target):raise ValueError('Incomplete profile grid')
        for power in range(24,17,-1):
            targets=2**power
            cost=3*sum(targets*seconds_per_target[k]+fixed_seconds[k] for k in required)
            if cost*1.2 <= available:return {'sizes':list(range(n)),'targets':targets,'estimated_seconds':cost,'margin':.2}
    raise RuntimeError('Minimum replicated grid does not fit; profile/optimization required')
