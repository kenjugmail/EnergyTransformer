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
from rewt.tokenizer import StoryTokenizer

PARENT=Path(__file__).resolve().parent
OUT=PARENT/'language_capacity'
RUNS=ROOT/'runs/rewt_learning_20261010/language_capacity'
DATA=ROOT/'data/tinystories_tokens_20261010'


def cases():
    spec=importlib.util.spec_from_file_location('rewt_story_configuration',PARENT/'micro.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stable=dict(recurrent_type='gru',context_kernel=3,history=128,
        compatibility_geometry='anchor_relative',wave_geometry='anchor_relative',initial_carry_scale=0.)
    config64=asdict(module.sized_config(width=64,**stable))
    config80=asdict(module.sized_config(width=80,**stable))
    return {'fourier64':dict(physical_arm='fourier_rewt',config=config64),
            'fourier80':dict(physical_arm='fourier_rewt',config=config80),
            'neural80':dict(physical_arm='neural_energy',config={**config80,'activation':'mlp'})}



def prepare(updates):
    OUT.mkdir(parents=True,exist_ok=True)
    corpus=StoryCorpus(DATA)
    tokenizer=StoryTokenizer.load(corpus.manifest['tokenizer']['path'])
    rng=np.random.default_rng(997)
    eligible=np.flatnonzero(corpus.documents['test'][:,1]-corpus.documents['test'][:,0]>=128)
    documents=rng.choice(eligible,30,replace=False)
    prompts=[]
    for index in documents:
        start=int(corpus.documents['test'][index,0])
        prefix=bytes(corpus.data['test'][start:start+32])
        reference=bytes(corpus.data['test'][start+32:start+128])
        prompts.append(dict(id=f'story_{len(prompts):03d}',document_index=int(index),offset=start,
            prompt=tokenizer.decode(prefix).decode('ascii'),prompt_hex=prefix.hex(),
            reference_hex=tokenizer.decode(reference).hex(),token_prefix_length=32))
    plan=dict(created=datetime.now(timezone.utc).isoformat(),seed=991,updates=updates,
        batch_size=32,context=128,learning_rate=.0015,warm_updates=16,validation_every=256,
        state_gradient_interval=64,tokenizer=corpus.manifest['tokenizer'],
        cases=cases(),corpus=corpus.identity(),data_directory=str(DATA),
        validation_offsets=corpus.offsets('valid',32,128,993),
        test_offsets=corpus.offsets('test',64,128,995),text_prompts=prompts,
        source_sha256={f.name:file_sha256(f) for f in sorted((ROOT/'rewt').glob('*.py'))},
        script_sha256=file_sha256(Path(__file__)),configuration_script_sha256=file_sha256(PARENT/'micro.py'),
        rounds=2,loss='Next-token cross entropy only; no synthetic response labels or future teacher.',
        selection='Lowest validation bpb under the common two-round energy policy; Transformer has its ordinary forward.',
        scope='Matched 250K parameter budget reallocated toward larger recurrent states (64/80 versus preceding 48); paired seed, training batches, token budget, context, learning rate and vocabulary. A training-only learned 256-piece vocabulary on document-disjoint TinyStories. Equal target-token budget, actual raw byte volume recorded; 128 token context, not 128 raw bytes. This is a separate vocabulary/learning experiment.')
    save_json(OUT/'plan.json',plan)
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for f in (ROOT/'rewt').glob('*.py'):(snapshot/f.name).write_bytes(f.read_bytes())
    save_json(OUT/'text_prompts.json',plan['text_prompts'])
    return plan


def new_model(settings,seed):
    torch.manual_seed(seed)
    if settings['physical_arm']=='transformer':return TransformerLM(TransformerConfig(**settings['config']))
    return JointEnergyLM(EnergyConfig(**settings['config']))


@torch.inference_mode()
def token_metrics(model,corpus,offsets,tokenizer,*,context=128,batch_size=8,split='valid'):
    model.eval();nll=0.;raw_bytes=0;tokens=0;correct=0
    lengths=torch.tensor(tokenizer.byte_lengths,dtype=torch.long)
    for first in range(0,len(offsets),batch_size):
        starts=offsets[first:first+batch_size]
        batch=corpus.batch(split,None,len(starts),context,starts=starts)
        output=model(batch.tokens)
        nll+=float(F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten(),reduction='sum'))
        raw_bytes+=int(lengths[batch.targets].sum());tokens+=batch.targets.numel()
        correct+=int((output.logits.argmax(-1)==batch.targets).sum())
    return dict(bpb=nll/raw_bytes/math.log(2),bits_per_token=nll/tokens/math.log(2),
        next_token_accuracy=correct/tokens,evaluated_tokens=tokens,evaluated_raw_bytes=raw_bytes)


def fit(name,settings,plan):
    torch.set_num_threads(1)
    corpus=StoryCorpus(plan['data_directory'])
    tokenizer=StoryTokenizer.load(plan['tokenizer']['path'])
    assert file_sha256(Path(plan['tokenizer']['path']))==plan['tokenizer']['sha256']
    byte_lengths=torch.tensor(tokenizer.byte_lengths,dtype=torch.long)
    model=new_model(settings,plan['seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
    clip=GradientFilter(model.parameters(),method='clip',max_norm=1.,spike_threshold=5.)
    directory=RUNS/name;directory.mkdir(parents=True,exist_ok=True)
    trace=[];best=float('inf');best_step=0;max_norm=0.;training_cpu=0.;validation_cpu=0.;raw_bytes=0
    event('story_fit_started',case=name,parameters=model.parameters_count(),updates=plan['updates'],
          target_tokens=plan['updates']*plan['batch_size']*plan['context'])
    for step in range(plan['updates']):
        rng=np.random.default_rng(plan['seed']*1000000+step+998)
        batch=corpus.batch('train',rng,plan['batch_size'],plan['context'])
        raw_bytes+=int(byte_lengths[batch.targets].sum())
        model.train();optimizer.zero_grad(set_to_none=True)
        # Warm-up then gradual decay, declared before any test output.
        scale=min(1.,(step+1)/16)*(.2+.8*(1+math.cos(math.pi*step/plan['updates']))/2)
        for group in optimizer.param_groups:group['lr']=plan['learning_rate']*scale
        start=time.process_time()
        output=model(batch.tokens,**({'rounds':0 if step<plan['warm_updates'] else plan['rounds'],
                                    'state_gradient_interval':plan['state_gradient_interval']}
                                     if isinstance(model,JointEnergyLM) else {}))
        loss=F.cross_entropy(output.logits.flatten(0,1),batch.targets.flatten())
        if not torch.isfinite(loss):raise FloatingPointError(f'{name}/{step}: nonfinite loss')
        loss.backward();g=clip.apply();optimizer.step()
        training_cpu+=time.process_time()-start;max_norm=max(max_norm,g['raw_gradient_norm'])
        row=dict(step=step+1,loss=float(loss.detach()),processed_tokens=(step+1)*plan['batch_size']*plan['context'],
                 processed_bytes=raw_bytes,**g)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            start=time.process_time()
            validation=token_metrics(model,corpus,plan['validation_offsets'],tokenizer,context=plan['context'],split='valid')
            row['validation']=validation;validation_cpu+=time.process_time()-start
            if validation['bpb']<best:
                best=validation['bpb'];best_step=step+1
                save_checkpoint(directory/'best.pt',dict(arm=settings['physical_arm'],config=asdict(model.config),
                    seed=plan['seed'],weights=model.state_dict(),step=step+1,validation=validation,tokenizer=plan['tokenizer']))
            save_checkpoint(directory/'last.pt',dict(arm=settings['physical_arm'],config=asdict(model.config),
                seed=plan['seed'],weights=model.state_dict(),step=step+1,optimizer=optimizer.state_dict(),filter=clip.state_dict(),tokenizer=plan['tokenizer']))
            # A fixed progress prompt is separate from the sealed held-out stories.
            sample=generate_batch(model,[tokenizer.encode(b'Once upon a time, there was a ')],max_new=64,
                                  sampling=True,temperature=.7,seed=1001)[0]
            row['progress_prompt_generation']=tokenizer.decode(sample).decode('ascii',errors='replace')
            event('story_validation',case=name,step=step+1,bpb=validation['bpb'],max_raw_norm=max_norm,
                  processed_bytes=row['processed_bytes'],progress_sample=row['progress_prompt_generation'])
        elif (step+1)%64==0:
            event('story_progress',case=name,step=step+1,loss=row['loss'],training_cpu=training_cpu)
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    model,checkpoint=load_model(directory/'best.pt')
    batch=corpus.batch('valid',None,2,32,starts=plan['validation_offsets'][:2])
    audit=refinement_audit(model,batch.tokens)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=name,seed=plan['seed'],parameters=model.parameters_count(),updates=plan['updates'],
        processed_target_tokens=plan['updates']*plan['batch_size']*plan['context'],processed_target_raw_bytes=raw_bytes,best_step=best_step,
        best_validation_bpb=best,test_not_evaluated=True,energy_audit=audit,
        maximum_raw_gradient_norm=max_norm,training_cpu_seconds=training_cpu,validation_cpu_seconds=validation_cpu,
        checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'),config=asdict(model.config))
    save_json(OUT/f'{name}.json',result);event('story_fit_complete',case=name,validation_bpb=best)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args();torch.set_num_threads(1)
    plan=prepare(args.updates)
    # Exercise checkpoint serialization for every arm before a long fit.
    for name,settings in plan['cases'].items():
        model=new_model(settings,plan['seed'])
        smoke=OUT/f'{name}_smoke.pt'
        save_checkpoint(smoke,dict(arm=settings['physical_arm'],config=asdict(model.config),
            seed=plan['seed'],weights=model.state_dict()))
        loaded,_=load_model(smoke)
        x=torch.tensor([list(b'Once there was a')],dtype=torch.long)
        with torch.inference_mode():
            torch.testing.assert_close(loaded(x).logits,model(x).logits,rtol=0,atol=0)
        smoke.unlink()
    event('story_checkpoint_smoke_passed',cases=list(plan['cases']))
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fit,name,settings,plan) for name,settings in plan['cases'].items()]
        results=[future.result() for future in as_completed(futures)]
    for name,expected in plan['source_sha256'].items():assert file_sha256(ROOT/'rewt'/name)==expected
    save_json(OUT/'results.json',results)


if __name__=='__main__':main()
