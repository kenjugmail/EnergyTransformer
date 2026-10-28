"""Distill the converted model on short and long, answer-weighted conversations."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from rewt.lfm_conversion import load_converted
from rewt.lfm_distill import RecoveryTokens, vocabulary_losses
from scripts.prepare_rewt_lfm3 import TEACHER_REVISION, sha, save_json


def hidden_loss(hidden,reference,weights):
    numerator=((hidden.float()-reference.float()).square().mean(-1)*weights).sum()
    denominator=(reference.float().square().mean(-1)*weights).sum().clamp_min(1e-6)
    return numerator/denominator


@torch.no_grad()
def evaluate(student,teacher,data,args):
    student.eval(); rng=np.random.default_rng(args.seed+100)
    totals={k:0. for k in ('kl','nll','teacher_nll','agreement','hidden_relative_mse')}; count=0.; tokens=0
    with torch.autocast(args.device,dtype=torch.bfloat16):
        for _ in range(args.eval_batches):
            x,y,w,active=data.batch(args.batch,rng,args.device,prefix_fraction=.75)
            reference=teacher.model(x,use_cache=False).last_hidden_state
            hidden,_=student.hidden(x)
            metrics=vocabulary_losses(hidden,reference,y,student.embedding.weight,teacher.model.embed_tokens.weight,
                chunk=args.logit_chunk,token_weights=w,checkpoint_chunks=False)
            metrics['hidden_relative_mse']=hidden_loss(hidden,reference,w)
            weight=float(w.sum()); count+=weight; tokens+=int(active.sum())
            for name,value in metrics.items():totals[name]+=float(value)*weight
    return {**{k:v/count for k,v in totals.items()},'weighted_tokens':count,'tokens':tokens}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus',type=Path,default=ROOT/'data/lfm3_capability_recovery_v1')
    p.add_argument('--steps',type=int,default=1024); p.add_argument('--batch',type=int,default=2)
    p.add_argument('--length',type=int,default=512); p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--ffn-layers',type=int,default=6); p.add_argument('--inherited-lr-scale',type=float,default=.2)
    p.add_argument('--prompt-weight',type=float,default=.1); p.add_argument('--hard-weight',type=float,default=.2)
    p.add_argument('--hidden-weight',type=float,default=.05); p.add_argument('--prefix-fraction',type=float,default=.75)
    p.add_argument('--eval-batches',type=int,default=16); p.add_argument('--eval-every',type=int,default=128)
    p.add_argument('--logit-chunk',type=int,default=32); p.add_argument('--seed',type=int,default=281710)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if a.output.exists():p.error('use a fresh output directory')
    if min(a.steps,a.batch,a.length,a.eval_batches,a.eval_every,a.logit_chunk)<1 or a.ffn_layers<0:p.error('invalid sizes')
    if not 0<a.prompt_weight<=1 or not 0<=a.prefix_fraction<=1 or min(a.lr,a.inherited_lr_scale)<=0:p.error('invalid weights')
    a.output.mkdir(parents=True); torch.set_num_threads(4); torch.manual_seed(a.seed)
    sources=['rewt/lfm_conversion.py','rewt/lfm_distill.py','scripts/recover_lfm3_capabilities.py','scripts/prepare_lfm3_recovery.py']
    import shutil
    for name in sources:
        target=a.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    plan=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        initial_sha256=sha(a.initial),corpus_manifest_sha256=sha(a.corpus/'manifest.json'),
        source_sha256={name:sha(ROOT/name) for name in sources},
        selection='Lowest fixed validation answer-weighted KL; capability evaluation is performed only after the training run and never used for checkpoint selection.')
    save_json(a.output/'plan.json',plan)
    teacher=AutoModelForCausalLM.from_pretrained(ROOT/'data/lfm3_1_7b_teacher'/TEACHER_REVISION,
        trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,device_map=a.device,attn_implementation='sdpa').eval().requires_grad_(False)
    student,metadata=load_converted(a.initial,device=a.device)
    if a.ffn_layers>len(student.blocks):raise ValueError('too many FFN layers')
    conversion=student.trainable_conversion_parameters()
    for block in list(student.blocks)[len(student.blocks)-a.ffn_layers:]:
        block.feed_forward.requires_grad_(True);block.ffn_norm.requires_grad_(True);block.post_ffn_norm.requires_grad_(True)
    student.norm.requires_grad_(True)
    conversion_ids={id(x) for x in conversion}
    inherited=[x for x in student.parameters() if x.requires_grad and id(x) not in conversion_ids]
    parameters=conversion+inherited
    for x in parameters:x.data=x.data.float()
    student.gradient_checkpointing=True
    train=RecoveryTokens(a.corpus,'train',a.length,student.config.pad_token_id,a.prompt_weight)
    valid=RecoveryTokens(a.corpus,'valid',a.length,student.config.pad_token_id,a.prompt_weight)
    start=time.perf_counter(); exposures=0; weighted_exposures=0.; best_step=0
    def log(row):
        row['elapsed_seconds']=time.perf_counter()-start
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def save(name,step):
        student.save(a.output/name,metadata=dict(metadata,stage='capability_recovery',
            training_steps=metadata.get('training_steps',0)+step,run_training_steps=step,
            run=str(a.output.resolve()),corpus_manifest_sha256=plan['corpus_manifest_sha256'],
            run_token_exposures=exposures,ffn_layers_unfrozen=a.ffn_layers))
    baseline=evaluate(student,teacher,valid,a);best=baseline['kl'];log(dict(stage='initial',validation=baseline))
    save('best.pt',0)
    optimizer=torch.optim.AdamW([{'params':conversion,'lr_scale':1.},{'params':inherited,'lr_scale':a.inherited_lr_scale}],
        lr=a.lr,weight_decay=.01,foreach=False)
    rng=np.random.default_rng(a.seed)
    for step in range(1,a.steps+1):
        student.train();optimizer.zero_grad(set_to_none=True)
        x,y,w,active=train.batch(a.batch,rng,a.device,prefix_fraction=a.prefix_fraction)
        with torch.no_grad(),torch.autocast(a.device,dtype=torch.bfloat16):
            reference=teacher.model(x,use_cache=False).last_hidden_state
        with torch.autocast(a.device,dtype=torch.bfloat16):
            hidden,_=student.hidden(x)
            losses=vocabulary_losses(hidden,reference,y,student.embedding.weight,teacher.model.embed_tokens.weight,
                chunk=a.logit_chunk,token_weights=w)
            hmse=hidden_loss(hidden,reference,w)
            loss=losses['kl']+a.hard_weight*losses['nll']+a.hidden_weight*hmse
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite recovery loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        factor=min(1.,step/32)*(.2+.8*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in optimizer.param_groups:group['lr']=a.lr*factor*group['lr_scale']
        optimizer.step();exposures+=int(active.sum());weighted_exposures+=float(w.sum())
        if step==1 or step%16==0 or step==a.steps:
            log(dict(stage='training',step=step,loss=float(loss.detach()),grad_norm=float(norm),
                token_exposures=exposures,learning_rate=a.lr*factor,
                **{k:float(v.detach()) for k,v in losses.items()}))
        if step%a.eval_every==0 or step==a.steps:
            metrics=evaluate(student,teacher,valid,a);log(dict(stage='validation',step=step,validation=metrics))
            if metrics['kl']<best:
                best=metrics['kl'];best_step=step;save('best.pt',step)
    save('last.pt',a.steps)
    summary=dict(best_step=best_step,initial_validation=baseline,best_validation_kl=best,
        token_exposures=exposures,weighted_token_exposures=weighted_exposures,
        trainable_parameters=sum(x.numel() for x in parameters),elapsed_seconds=time.perf_counter()-start,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if a.device=='cuda' else None,
        best_sha256=sha(a.output/'best.pt'),selection=plan['selection'])
    save_json(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
