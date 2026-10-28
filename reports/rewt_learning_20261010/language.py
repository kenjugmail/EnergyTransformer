"""From-scratch, document-bounded English learning with matched energy controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.data import file_sha256
from rewt.evaluate import generate_batch, language_metrics, refinement_audit, text_evaluation
from rewt.experiment import event, load_model, save_checkpoint, save_json, save_lines
from rewt.model import EnergyConfig, JointEnergyLM, TransformerConfig, TransformerLM
from rewt.optim import GradientFilter
from rewt.story_data import StoryCorpus

PARENT=Path(__file__).resolve().parent
OUT=PARENT/'language'
RUNS=ROOT/'runs/rewt_learning_20261010/language'
DATA=ROOT/'data/tinystories_rewt_20261010'


def cases():
    spec=importlib.util.spec_from_file_location('rewt_story_configuration',PARENT/'micro.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    settings=module.cases()
    return {name:dict(physical_arm='fourier_rewt',config=settings[name]['config'])
            for name in ('control','memory','combined')} | {
            'transformer':dict(physical_arm='transformer',config=asdict(TransformerConfig(
                width=80,layers=2,heads=4,feedforward=453,context=128)))}


def prepare(updates):
    OUT.mkdir(parents=True,exist_ok=True)
    corpus=StoryCorpus(DATA)
    plan=dict(created=datetime.now(timezone.utc).isoformat(),seed=991,updates=updates,
        batch_size=32,context=128,learning_rate=.003,warm_updates=16,validation_every=128,
        cases=cases(),corpus=corpus.identity(),data_directory=str(DATA),
        validation_offsets=corpus.offsets('valid',32,128,993),
        test_offsets=corpus.offsets('test',64,128,995),text_prompts=corpus.text_prompts(30,seed=997),
        source_sha256={f.name:file_sha256(f) for f in sorted((ROOT/'rewt').glob('*.py'))},
        script_sha256=file_sha256(Path(__file__)),configuration_script_sha256=file_sha256(PARENT/'micro.py'),
        rounds=2,loss='Next-byte cross entropy only; no synthetic response labels or future teacher.',
        selection='Lowest validation bpb under the common two-round energy policy; Transformer has its ordinary forward.',
        scope='Capped TinyStories subset, ASCII-normalized, document-disjoint. A domain learning experiment, not a paired enwik8 result.')
    save_json(OUT/'plan.json',plan)
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for f in (ROOT/'rewt').glob('*.py'):(snapshot/f.name).write_bytes(f.read_bytes())
    save_json(OUT/'text_prompts.json',plan['text_prompts'])
    return plan


def new_model(settings,seed):
    torch.manual_seed(seed)
    if settings['physical_arm']=='transformer':return TransformerLM(TransformerConfig(**settings['config']))
    return JointEnergyLM(EnergyConfig(**settings['config']))


def fit(name,settings,plan):
    torch.set_num_threads(1)
    corpus=StoryCorpus(plan['data_directory'])
    model=new_model(settings,plan['seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
    clip=GradientFilter(model.parameters(),method='clip',max_norm=1.,spike_threshold=5.)
    directory=RUNS/name;directory.mkdir(parents=True,exist_ok=True)
    trace=[];best=float('inf');best_step=0;max_norm=0.;training_cpu=0.;validation_cpu=0.
    event('story_fit_started',case=name,parameters=model.parameters_count(),updates=plan['updates'],
          target_bytes=plan['updates']*plan['batch_size']*plan['context'])
    for step in range(plan['updates']):
        rng=np.random.default_rng(plan['seed']*1000000+step+998)
        batch=corpus.batch('train',rng,plan['batch_size'],plan['context'])
        model.train();optimizer.zero_grad(set_to_none=True)
        # Warm-up then gradual decay, declared before any test output.
        scale=min(1.,(step+1)/16)*(.2+.8*(1+math.cos(math.pi*step/plan['updates']))/2)
        for group in optimizer.param_groups:group['lr']=plan['learning_rate']*scale
        start=time.process_time()
        output=model(batch.tokens,**({'rounds':0 if step<plan['warm_updates'] else plan['rounds']}
                                     if isinstance(model,JointEnergyLM) else {}))
        loss=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten())
        if not torch.isfinite(loss):raise FloatingPointError(f'{name}/{step}: nonfinite loss')
        loss.backward();g=clip.apply();optimizer.step()
        training_cpu+=time.process_time()-start;max_norm=max(max_norm,g['raw_gradient_norm'])
        row=dict(step=step+1,loss=float(loss.detach()),processed_bytes=(step+1)*plan['batch_size']*plan['context'],**g)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            start=time.process_time()
            validation=language_metrics(model,corpus,plan['validation_offsets'],context=plan['context'],
                batch_size=8,split='valid',**({'rounds':plan['rounds']} if isinstance(model,JointEnergyLM) else {}))
            row['validation']=validation;validation_cpu+=time.process_time()-start
            if validation['bpb']<best:
                best=validation['bpb'];best_step=step+1
                save_checkpoint(directory/'best.pt',dict(arm=settings['physical_arm'],config=asdict(model.config),
                    seed=plan['seed'],weights=model.state_dict(),step=step+1,validation=validation))
            save_checkpoint(directory/'last.pt',dict(arm=settings['physical_arm'],config=asdict(model.config),
                seed=plan['seed'],weights=model.state_dict(),step=step+1,optimizer=optimizer.state_dict(),filter=clip.state_dict()))
            # A fixed progress prompt is separate from the sealed held-out stories.
            sample=generate_batch(model,[b'Once upon a time, there was a '],max_new=80,
                                  sampling=True,temperature=.7,seed=1001)[0]
            row['progress_prompt_generation']=sample.decode('ascii',errors='replace')
            event('story_validation',case=name,step=step+1,bpb=validation['bpb'],max_raw_norm=max_norm,
                  processed_bytes=row['processed_bytes'],progress_sample=row['progress_prompt_generation'])
        elif (step+1)%64==0:
            event('story_progress',case=name,step=step+1,loss=row['loss'],training_cpu=training_cpu)
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    model,checkpoint=load_model(directory/'best.pt')
    test=language_metrics(model,corpus,plan['test_offsets'],context=plan['context'],batch_size=8,split='test')
    text_summary,text_rows=text_evaluation(model,plan['text_prompts'],seed=1003)
    batch=corpus.batch('valid',None,2,32,starts=plan['validation_offsets'][:2])
    audit=refinement_audit(model,batch.tokens)
    save_lines(OUT/f'{name}_text.jsonl',text_rows);save_lines(directory/'trace.jsonl',trace)
    result=dict(case=name,seed=plan['seed'],parameters=model.parameters_count(),updates=plan['updates'],
        processed_target_bytes=plan['updates']*plan['batch_size']*plan['context'],best_step=best_step,
        best_validation_bpb=best,language=test,text=text_summary,energy_audit=audit,
        maximum_raw_gradient_norm=max_norm,training_cpu_seconds=training_cpu,validation_cpu_seconds=validation_cpu,
        checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'),config=asdict(model.config))
    save_json(OUT/f'{name}.json',result);event('story_fit_complete',case=name,test_bpb=test['bpb'],validation_bpb=best)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args();torch.set_num_threads(1)
    plan=prepare(args.updates)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,name,settings,plan) for name,settings in plan['cases'].items()]
        results=[future.result() for future in as_completed(futures)]
    for name,expected in plan['source_sha256'].items():assert file_sha256(ROOT/'rewt'/name)==expected
    save_json(OUT/'results.json',results)


if __name__=='__main__':main()
