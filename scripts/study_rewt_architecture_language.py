"""Cold architecture language screening with new document-disjoint holdouts."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.bpe import StoryBPE
from rewt.story_data import StoryCorpus
from rewt.architecture import ArchitectureConfig, ArchitectureEnergyLM, load_architecture_checkpoint
from rewt.data import file_sha256
from rewt.experiment import save_json,save_lines,save_checkpoint,event
from study_rewt_architecture import VARIANTS

OUT=ROOT/'reports/rewt_architecture_20261022/language'
RUNS=ROOT/'runs/rewt_architecture_20261022/language'
DATA=ROOT/'data/rewt_architecture_20261022'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():raise FileExistsError('preserve prior language plan')
    old=json.loads((ROOT/'data/tinystories_rewt_20261010/manifest.json').read_text())
    excluded={i for split in old['splits'].values() for i in split['document_ids']}
    bpe_source=ROOT/'data/tinystories_bpe_20261010'
    bpe_manifest=json.loads((bpe_source/'manifest.json').read_text())
    tokenizer=StoryBPE.load(bpe_source/'vocabulary.json')
    raw=(DATA/'new_valid_source.txt').read_bytes()
    text=raw.decode('utf-8',errors='ignore')
    docs={'valid':[],'test':[]};seen=set(excluded)
    for text in text.split('<|endoftext|>')[1:-1]:
        normalized=text.strip().replace('\r','').translate(str.maketrans({'“':'"','”':'"','‘':"'",'’':"'",'—':'-','–':'-','\u00a0':' '}))
        encoded=normalized.encode('ascii',errors='replace')
        identity=hashlib.sha256(encoded).hexdigest()
        if len(encoded)<200 or identity in seen:continue
        seen.add(identity)
        tokens=tokenizer.encode(encoded)
        if len(tokens)<=160:continue
        docs['valid' if int(identity[:12],16)%2==0 else 'test'].append((identity,tokens))
    splits={'train':bpe_manifest['splits']['train']}
    for split,rows in docs.items():
        output=bytearray();records=[]
        for identity,tokens in rows:
            start=len(output);output.extend(tokens)
            records.append(dict(id=identity,start=start,end=len(output)))
        path=DATA/f'{split}_bpe.bin';path.write_bytes(output)
        index=DATA/f'{split}_bpe_documents.json';save_json(index,records)
        splits[split]=dict(path=str(path),index_path=str(index),sha256=file_sha256(path),index_sha256=file_sha256(index),documents=len(records),bytes=len(output))
    manifest=dict(dataset=old['dataset'],revision=old['revision'],splits=splits,
        source=json.loads((DATA/'source.json').read_text()),vocabulary=str(bpe_source/'vocabulary.json'),
        vocabulary_sha256=file_sha256(bpe_source/'vocabulary.json'),
        note='Fresh byte range 8..12 MiB of the pinned official validation file. Drop boundary fragments, exclude all prior train/valid/test normalized document hashes. Tokenizer learned on old training text only. Training reuses the original official training partition.')
    save_json(DATA/'manifest.json',manifest)
    corpus=StoryCorpus(DATA)
    taskplan=json.loads((ROOT/'reports/rewt_architecture_20261022/plan.json').read_text())
    names=('control','precond4','scan','feedback8','delta','patch4','lowrank16','combined')
    cases=[c for c in taskplan['cases'] if c['variant'] in names]
    plan=dict(config=taskplan['config'],cases=cases,updates=512,batch_size=24,context=128,
        validation_every=128,learning_rate=.001,data_manifest_sha256=file_sha256(DATA/'manifest.json'),
        valid_offsets=corpus.offsets('valid',128,128,22182),test_offsets=corpus.offsets('test',256,128,22183),
        tokenizer_sha256=manifest['vocabulary_sha256'],
        selection='Minimum validation bits per original byte over four checkpoints. All 8 variants, 3 seeds reported on fresh heldout documents after every run finishes.',
        scope='512-update cold-start screening, 1,572,864 observed BPE tokens per run. This does not establish fluent text or match the earlier 16.8M-token language budget.',
        test_not_evaluated=True)
    save_json(OUT/'plan.json',plan)
    return plan


@torch.inference_mode()
def evaluate(model,corpus,offsets,split,tokenizer):
    model.eval().prepare_inference();nll=0.;byte_count=correct=count=0
    lengths=torch.tensor(tokenizer.byte_lengths)
    for first in range(0,len(offsets),16):
        starts=offsets[first:first+16];batch=corpus.batch(split,None,len(starts),128,starts=starts)
        logits=model(batch.tokens).logits
        nll+=float(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='sum'))
        byte_count+=int(lengths[batch.targets].sum());count+=batch.targets.numel()
        correct+=int((logits.argmax(-1)==batch.targets).sum())
    return dict(bpb=nll/byte_count/math.log(2),token_nll=nll/count,token_accuracy=correct/count,
                original_bytes=byte_count,tokens=count)


def fit(case,plan):
    torch.set_num_threads(1);out=OUT/f'{case["name"]}.json';directory=RUNS/case['name'];directory.mkdir(parents=True,exist_ok=True)
    if out.exists():return json.loads(out.read_text())
    if (directory/'last.pt').exists():raise RuntimeError('partial language run exists')
    torch.manual_seed(case['seed']);model=ArchitectureEnergyLM(ArchitectureConfig(**dict(plan['config'],**case['overrides'])))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01)
    corpus=StoryCorpus(DATA);tokenizer=StoryBPE.load(ROOT/'data/tinystories_bpe_20261010/vocabulary.json')
    best=float('inf');best_step=0;cpu=eval_cpu=0.;trace=[]
    event('language_architecture_started',case=case['name'])
    for step in range(plan['updates']):
        start=time.process_time();rng=np.random.default_rng(case['seed']*100000+step)
        batch=corpus.batch('train',rng,24,128)
        model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=.001*min(1.,(step+1)/32)*(.3+.7*(1+math.cos(math.pi*step/plan['updates']))/2)
        logits=model(batch.tokens,state_gradient_interval=32).logits
        loss=F.cross_entropy(logits.flatten(0,1),batch.targets.flatten())
        if not torch.isfinite(loss):raise FloatingPointError(case['name'])
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
        cpu+=time.process_time()-start;row=dict(step=step+1,loss=float(loss.detach()),gradient=gradient)
        if (step+1)%128==0:
            start=time.process_time();score=evaluate(model,corpus,plan['valid_offsets'],'valid',tokenizer);eval_cpu+=time.process_time()-start
            saved=dict(arm='architecture_energy',config=asdict(model.config),seed=case['seed'],weights=model.state_dict(),step=step+1,selection=score,case=case)
            if score['bpb']<best:
                best,best_step=score['bpb'],step+1;save_checkpoint(directory/'best.pt',saved)
            save_checkpoint(directory/'last.pt',dict(saved,optimizer=optimizer.state_dict()))
            row['validation']=score;event('language_architecture_validation',case=case['name'],step=step+1,bpb=score['bpb'])
        trace.append(row)
        if (step+1)%64==0:save_lines(directory/'trace.jsonl',trace)
    result=dict(case=case,parameters=model.parameters_count(),validation_bpb=best,best_step=best_step,training_cpu_seconds=cpu,
                validation_cpu_seconds=eval_cpu,checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'))
    save_json(out,result);return result


def test(case,plan):
    torch.set_num_threads(1);result=json.loads((OUT/f'{case["name"]}.json').read_text())
    model,_=load_architecture_checkpoint(result['checkpoint']);tokenizer=StoryBPE.load(ROOT/'data/tinystories_bpe_20261010/vocabulary.json')
    corpus=StoryCorpus(DATA);result['test']=evaluate(model,corpus,plan['test_offsets'],'test',tokenizer)
    generations=[]
    with torch.inference_mode():
        for start in plan['test_offsets'][:6]:
            prompt=bytes(corpus.data['test'][start:start+32]);x=torch.tensor([list(prompt)])
            output=model(x);state=output.state;tokens=[]
            for i in range(64):
                next_token=output.logits[:,-1].argmax(-1);tokens.append(int(next_token))
                output=model(next_token[:,None],state);state=output.state
            generations.append(dict(offset=start,prompt=tokenizer.decode(prompt).decode('ascii'),generation=tokenizer.decode(tokens).decode('ascii')))
    save_lines(OUT/f'{case["name"]}_generations.jsonl',generations);save_json(OUT/f'{case["name"]}_test.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--test',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    plan=prepare() if a.prepare else json.loads((OUT/'plan.json').read_text())
    if a.train or a.test:
        if a.test and not all((OUT/f'{c["name"]}.json').exists() for c in plan['cases']):raise RuntimeError('finish all fits before test')
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit if a.train else test,c,plan) for c in plan['cases']]
            results=[f.result() for f in as_completed(futures)]
        save_json(OUT/('training_results.json' if a.train else 'test_results.json'),results)

if __name__=='__main__':main()
