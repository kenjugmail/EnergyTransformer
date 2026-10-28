"""Replay one saved training update without changing saved weights or data."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.data import Corpus,file_sha256
from rewt.evaluate import model_output
from rewt.experiment import AUXILIARY_WEIGHTS,curriculum_batch,initialization_noise,load_model
from rewt.improve import get_tasks
from rewt.optim import GradientFilter


def scalar(value):
    value=float(value)
    return value if math.isfinite(value) else str(value)


def main(seed,interval=0):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan=json.loads((ROOT/'selected_protocol.json').read_text())
    path=Path(plan['runs'])/f'seed_{seed}'/'full_rewt'/'reasoning'/'last.pt'
    original=file_sha256(path)
    model,cp=load_model(path)
    teacher,_=load_model(Path(plan['runs'])/f'seed_{seed}'/'transformer'/'reasoning'/'best.pt')
    teacher.eval();teacher.requires_grad_(False)
    index=cp['step'];recipe=plan['variants']['full_rewt']['recipe']
    rng=np.random.default_rng(seed*1_000_000+index+77_000_000)
    generator=torch.Generator().manual_seed(seed*1_000_000+index+19)
    progress=index/plan['settings']['reason_updates']
    stage=1 if progress<.25 else 2 if progress<.55 else 3
    batch=curriculum_batch(get_tasks(plan,'train'),rng,plan['settings']['reason_batch_size'],
        stage=stage,process_seed=seed*1_000_000+index,process_supervision=True)
    rounds=int(np.random.default_rng(seed*1_000_000+index+90_000_000).integers(1,5))
    initial,corruption=initialization_noise(model,batch.tokens,seed=seed*1_000_000+index+91_000_000,recipe=recipe)
    model.train();model.zero_grad(set_to_none=True)
    out=model_output(model,batch.tokens,generator,details=True,rounds=rounds,initial_noise=initial,trace_energy=False,
                     state_gradient_interval=interval)
    ce=F.cross_entropy(out.logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100)
    extra=dict(ranking=model.ranking_loss(batch.tokens,batch.targets,out,batch.valid,hard_negatives=True,generator=generator),
               repair=model.repair_loss(out,batch.targets,batch.valid))
    with torch.no_grad():hidden=teacher(batch.tokens,details=True).details['hidden'].detach()
    extra.update(model.auxiliary_losses(out,batch.targets,valid=batch.valid,teacher_hidden=hidden,
        noise=torch.randn(*out.details['clean'].shape,generator=generator)))
    extra['process']=model.process_loss(out,batch.targets,batch.process_negatives)
    weights={**AUXILIARY_WEIGHTS,'repair':recipe['repair_weight'],'process':recipe['process_weight']}
    loss=ce+sum(weights[k]*v for k,v in extra.items())
    loss.backward()
    gradients=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.parameters() if p.requires_grad])
    value=dict(seed=seed,replayed_step=index+1,checkpoint_sha256=original,rounds=rounds,curriculum_stage=stage,state_gradient_interval=interval,
        loss=scalar(loss.detach()),ce=scalar(ce.detach()),all_gradient_elements_finite=bool(torch.isfinite(gradients).all()),
        nonfinite_gradient_elements=int((~torch.isfinite(gradients)).sum()),
        float32_norm=scalar(gradients.norm()),float64_norm=scalar(gradients.double().norm()),
        maximum_absolute_gradient=scalar(gradients.abs().max()),initialization_corruption_fraction=corruption)
    filtering=GradientFilter(model.parameters(),method=plan['variants']['full_rewt']['method'])
    filtering.load_state_dict(cp['filter'])
    try:
        value['original_filter_diagnostics']=filtering.apply()
        value['original_diagnostics_json_finite']=True
        json.dumps(value['original_filter_diagnostics'],allow_nan=False)
    except Exception as error:
        value.pop('original_filter_diagnostics',None)
        value['original_filter_error']=f'{type(error).__name__}: {error}'
    assert file_sha256(path)==original
    suffix=f'_truncated{interval}' if interval else ''
    (ROOT/f'stability_replay_seed_{seed}{suffix}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--state-gradient-interval',type=int,default=0)
    args=parser.parse_args();main(args.seed,args.state_gradient_interval)
