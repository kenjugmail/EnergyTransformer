"""Frozen, matched streaming/Transformer comparison. Test only after all fits.

Cold language and cold autoregressive task fits are separate experiments. Equal
update runs share exact per-seed batches. Equal-process-CPU language runs measure
the full optimizer update (including data preparation), excluding validation/I/O.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.bpe import StoryBPE
from rewt.curriculum import generate_curriculum, render_task
from rewt.data import file_sha256, task_batch
from rewt.experiment import save_checkpoint, save_json, save_lines, event
from rewt.story_data import StoryCorpus
from rewt.streaming import StreamingConfig, StreamingLM, load_streaming
from rewt.tasks import Task, semantic_identity, semantic_partition, score_response

DATA=ROOT/'data/rewt_streaming_20261026'
SOURCES=[ROOT/p for p in ['rewt/streaming.py','rewt/curriculum.py','rewt/tasks.py','rewt/data.py',
    'rewt/story_data.py','rewt/bpe.py','scripts/study_rewt_streaming.py']]


def read_tasks(path):
    return [Task(**json.loads(s)) for s in Path(path).read_text().splitlines()]


def prepare_data():
    if (DATA/'manifest.json').exists():return
    old=json.loads((ROOT/'data/tinystories_bpe_20261010/manifest.json').read_text())
    tokenizer=StoryBPE.load(ROOT/'data/tinystories_bpe_20261010/vocabulary.json')
    excluded=set()
    for path in (ROOT/'data').rglob('*documents.json'):
        if DATA in path.parents:continue
        for row in json.loads(path.read_text()):excluded.add(row['id'])
    raw=(DATA/'heldout_source.txt').read_bytes()
    source=json.loads((DATA/'source.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==source['sha256']
    rows={'valid':[],'test':[]};seen=set(excluded)
    for text in raw.decode('utf-8',errors='ignore').split('<|endoftext|>')[1:-1]:
        normalized=text.strip().replace('\r','').translate(str.maketrans({'“':'"','”':'"','‘':"'",'’':"'",'—':'-','–':'-','\u00a0':' '}))
        encoded=normalized.encode('ascii',errors='replace');identity=hashlib.sha256(encoded).hexdigest()
        if identity in seen or len(encoded)<200:continue
        tokens=tokenizer.encode(encoded)
        if len(tokens)<=256:continue
        seen.add(identity);rows['valid' if int(identity[:12],16)%2==0 else 'test'].append((identity,tokens))
    splits={'train':old['splits']['train']}
    for split,docs in rows.items():
        raw=bytearray();records=[]
        for identity,tokens in docs:
            start=len(raw);raw.extend(tokens);records.append(dict(id=identity,start=start,end=len(raw)))
        path=DATA/f'{split}.bin';index=DATA/f'{split}_documents.json'
        path.write_bytes(raw);save_json(index,records)
        splits[split]=dict(path=str(path),index_path=str(index),sha256=file_sha256(path),index_sha256=file_sha256(index),documents=len(records),bytes=len(raw))
    save_json(DATA/'manifest.json',dict(dataset=old['dataset'],revision=old['revision'],splits=splits,source=source,
        excluded_previous_document_ids=len(excluded),vocabulary=str(ROOT/'data/tinystories_bpe_20261010/vocabulary.json'),
        note='New official validation byte range 12..16MiB; complete documents only; all previously indexed story IDs excluded; training and train-only tokenizer reused.'))


def prepare(args):
    out=args.output.resolve();runs=args.runs.resolve()
    if (out/'plan.json').exists():raise FileExistsError('use a new output directory')
    out.mkdir(parents=True,exist_ok=True);runs.mkdir(parents=True,exist_ok=True);prepare_data()
    corpus=StoryCorpus(DATA)
    excluded=set();previous=[]
    for path in (ROOT/'reports').rglob('tasks_test*.jsonl'):
        if out in path.parents:continue
        ids={json.loads(s)['id'] for s in path.read_text().splitlines()}
        excluded |= ids;previous.append(dict(path=str(path),sha256=file_sha256(path),identities=len(ids)))
    tasks={s:generate_curriculum(s,n,seed=202610263,exclude_ids=excluded if s=='test' else ())
           for s,n in [('train',450),('valid',12),('test',32)]}
    rng=random.Random(261363);long=[];used={t.id for t in tasks['test']}|excluded
    for depth in (32,64,128):
        accepted=0
        while accepted<24:
            payload=dict(family='recall',value=rng.randrange(10),distractors=[rng.randrange(10) for _ in range(depth)])
            identity=semantic_identity(payload)
            if identity in used or semantic_partition(identity)!='test':continue
            used.add(identity);facts=[f'X={v}' for v in payload['distractors']]
            facts.insert(rng.randrange(len(facts)+1),f'K={payload["value"]}')
            prompt=';'.join(facts)+';K=?\n';response=f'steps={payload["value"]};ans={payload["value"]}\n'
            long.append(Task(identity,'recall','test',0,depth,prompt,response,payload,'long_context'))
            accepted+=1
    tasks['test_long']=long
    for split,rows in tasks.items():save_lines(out/f'tasks_{split}.jsonl',[t.to_dict() for t in rows])
    ids={s:{t.id for t in rows} for s,rows in tasks.items()}
    assert all(ids[a].isdisjoint(ids[b]) for i,a in enumerate(ids) for b in list(ids)[i+1:])
    variants={
        'transformer':StreamingConfig(hybrid=False,hidden=162,window=0,rounds=0),
        'window_transformer':StreamingConfig(hybrid=False,hidden=162,window=64,rounds=0),
        'hybrid':StreamingConfig(hybrid=True,hidden=144,window=64,rounds=0),
        'hybrid_energy':StreamingConfig(hybrid=True,hidden=144,window=64,rounds=2)}
    cases=[dict(name=f'{objective}_{variant}_{seed}',objective=objective,variant=variant,seed=seed,config=asdict(config))
        for objective in args.objectives for seed in args.seeds for variant,config in variants.items()]
    plan=dict(output=str(out),runs=str(runs),cases=cases,language_updates=args.language_updates,task_updates=args.task_updates,
        cpu_budget=args.cpu_budget,learning_rate=.003,batch_language=24,batch_tasks=36,context=128,
        valid_offsets=corpus.offsets('valid',96,128,261360),
        test_offsets=corpus.offsets('test',192,128,261361),test_long_offsets=corpus.offsets('test',96,256,261362),
        data_manifest_sha256=file_sha256(DATA/'manifest.json'),
        data_sha256={s:file_sha256(out/f'tasks_{s}.jsonl') for s in tasks},
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in SOURCES},
        previous_tests=previous,excluded_test_ids=len(excluded),
        tokenizer_path=str(ROOT/'data/tinystories_bpe_20261010/vocabulary.json'),
        tokenizer_sha256=file_sha256(ROOT/'data/tinystories_bpe_20261010/vocabulary.json'),
        protocol='Cold starts; 3 seeds; same per-seed batches and learning-rate schedule; fresh documents and semantic tests. Language uses the shared training-only 256 BPE vocabulary. Tasks are separate raw-byte cold fits with identical answer-only weighted CE and no extra symbolic/pointer supervision. All variants reported.',
        selection='Language: minimum validation bpb. Tasks: maximum validation complete accuracy, lower response CE breaks ties. Four validation checkpoints per fit. No test access until every declared fit finishes.',
        fairness='Two-layer RMSNorm/RoPE/SwiGLU/tied-embedding full and window attention controls. Transformer FF width 162 vs hybrid 144 matches allocated parameters within 0.1%. Hybrid zero-round control retains inactive energy tensors, reported separately. Equal CPU is process time, not FLOPs; all measured training preprocessing included; validation and save time excluded. CPU single-thread workers; no GPU performance claim.',
        frozen_interventions=['hybrid_energy at zero refinement rounds'],
        limitations='Small-model bounded screening on stories, synthetic tasks and CPU; not general LLM superiority. Local attention training uses dense masked SDPA and is quadratic in chunk length.',test_not_evaluated=True)
    save_json(out/'plan.json',plan)
    (out/'source').mkdir(exist_ok=True)
    for p in SOURCES:(out/'source'/p.name).write_bytes(p.read_bytes())
    return plan


def verify(plan):
    assert file_sha256(DATA/'manifest.json')==plan['data_manifest_sha256']
    assert file_sha256(plan['tokenizer_path'])==plan['tokenizer_sha256']
    for path,sha in plan['source_sha256'].items():assert file_sha256(ROOT/path)==sha,path
    for s,sha in plan['data_sha256'].items():assert file_sha256(Path(plan['output'])/f'tasks_{s}.jsonl')==sha,s


@torch.inference_mode()
def language_evaluate(model,corpus,offsets,split,tokenizer,context=128,rounds=None):
    model.eval();total=original_bytes=correct=0;rows=[]
    lengths=torch.tensor(tokenizer.byte_lengths)
    for first in range(0,len(offsets),16):
        starts=offsets[first:first+16];batch=corpus.batch(split,None,len(starts),context,starts=starts)
        logits,_,_=model(batch.tokens,rounds=rounds)
        losses=F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none').view(len(starts),-1)
        for j,start in enumerate(starts):
            nll=float(losses[j].sum());nbytes=int(lengths[batch.targets[j]].sum())
            rows.append(dict(offset=start,nll=nll,bytes=nbytes))
            total+=nll;original_bytes+=nbytes
        correct+=int((logits.argmax(-1)==batch.targets).sum())
    return dict(bpb=total/original_bytes/math.log(2),token_accuracy=correct/(len(offsets)*context),rows=rows)


@torch.inference_mode()
def task_evaluate(model,tasks,rounds=None):
    model.eval();groups={};rows=[]
    for index,task in enumerate(tasks):groups.setdefault(len(task.prompt),[]).append((index,task))
    for length,items in groups.items():
        for first in range(0,len(items),24):
            part=items[first:first+24];tokens=torch.tensor([list(t.prompt.encode()) for _,t in part])
            logits,state,_=model(tokens,return_state=True,rounds=rounds)
            generated=[[] for _ in part];finished=[False]*len(part)
            for _ in range(40):
                nxt=logits[:,-1].argmax(-1)
                for i,b in enumerate(nxt.tolist()):
                    if not finished[i]:generated[i].append(b);finished[i]=b==10
                if all(finished):break
                logits,state,_=model(nxt[:,None],state,return_state=True,rounds=rounds)
            for (index,task),raw in zip(part,generated):
                response=bytes(raw).decode('latin1');score=score_response(task,response)
                rows.append(dict(index=index,id=task.id,family=task.family,depth=task.steps,template=task.template,
                                 difficulty=task.difficulty,**score))
    rows.sort(key=lambda r:r['index']);by={}
    for key in ('family','difficulty','depth'):
        by[key]={str(v):sum(r['both_correct'] for r in rows if r[key]==v)/sum(r[key]==v for r in rows) for v in sorted({r[key] for r in rows})}
    nll=count=0
    if max(len(t.prompt+t.response) for t in tasks)<=128:
        for first in range(0,len(tasks),36):
            batch=task_batch(tasks,range(first,min(first+36,len(tasks))))
            logits,_,_=model(batch.tokens,rounds=rounds)
            nll+=float(F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,reduction='sum'))
            count+=int(batch.valid.sum())
    return dict(complete_accuracy=sum(r['both_correct'] for r in rows)/len(rows),
        answer_accuracy=sum(r['answer_correct'] for r in rows)/len(rows),response_ce=nll/count if count else None,groups=by,rows=rows)


def init_worker():
    torch.set_num_threads(1)


def fit(case,plan):
    verify(plan);out=Path(plan['output']);directory=Path(plan['runs'])/case['name'];path=out/f'{case["name"]}.json'
    if path.exists():return json.loads(path.read_text())
    if directory.exists() and any(directory.iterdir()):raise RuntimeError('preserve partial run '+str(directory))
    directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(case['seed']);model=StreamingLM(StreamingConfig(**case['config']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],betas=(.9,.95),weight_decay=.01)
    # Decoupled decay excludes normalization and bias tensors, identically for every arm.
    optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.ndim>=2],weight_decay=.01),
                                dict(params=[p for p in model.parameters() if p.ndim<2],weight_decay=0.)],lr=plan['learning_rate'],betas=(.9,.95))
    objective=case['objective'];language=objective!='tasks';timed=objective=='compute'
    corpus=StoryCorpus(DATA) if language else None;tokenizer=StoryBPE.load(plan['tokenizer_path']) if language else None
    tasks=read_tasks(out/'tasks_train.jsonl') if not language else None
    valid=read_tasks(out/'tasks_valid.jsonl') if not language else None
    pools={(f,d):[i for i,t in enumerate(tasks) if t.family==f and t.steps==d] for f in ('arithmetic','rules','recall')
           for d in ((4,8,16) if f=='recall' else (1,2,3))} if tasks else None
    updates=plan['language_updates'] if language else plan['task_updates'];budget=plan['cpu_budget']
    step=0;cpu=wall=eval_cpu=0.;trace=[];best=None;best_step=0;quarter=1;digest=hashlib.sha256();target_tokens=0
    event('streaming_started',case=case['name'])
    while (cpu<budget if timed else step<updates):
        begin=time.process_time();wall_begin=time.perf_counter();fraction=cpu/budget if timed else step/updates
        rng=np.random.default_rng(case['seed']*100000+step)
        if language:batch=corpus.batch('train',rng,plan['batch_language'],128)
        else:
            chosen=[]
            for (family,depth),pool in pools.items():
                for _ in range(4):
                    task=tasks[int(rng.choice(pool))]
                    chosen.append(render_task(task.payload,'train',int(rng.integers(5)),seed=int(rng.integers(2**32))))
            rng.shuffle(chosen);batch=task_batch(chosen,range(len(chosen)),semantic_weight=4.)
        digest.update(batch.tokens.numpy().tobytes());digest.update(batch.targets.numpy().tobytes())
        model.train();optimizer.zero_grad(set_to_none=True)
        rate=plan['learning_rate']*min(1.,(fraction+.005)/.05)*(.2+.8*(1+math.cos(math.pi*min(1.,fraction)))/2)
        for group in optimizer.param_groups:group['lr']=rate
        logits,_,_=model(batch.tokens)
        loss=F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),ignore_index=-100,reduction='none').view_as(batch.targets)
        weights=batch.valid if language else batch.token_weights
        loss=(loss*weights).sum()/weights.sum()
        if not torch.isfinite(loss):raise FloatingPointError(case['name'])
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
        cpu+=time.process_time()-begin;wall+=time.perf_counter()-wall_begin;step+=1;target_tokens+=int(batch.valid.sum())
        row=dict(step=step,loss=float(loss.detach()),gradient=gradient,training_cpu=cpu)
        progress=cpu/budget if timed else step/updates
        if progress>=quarter/4:
            before=time.process_time()
            score=language_evaluate(model,corpus,plan['valid_offsets'],'valid',tokenizer) if language else task_evaluate(model,valid)
            eval_cpu+=time.process_time()-before
            selection=(-score['bpb'],) if language else (score['complete_accuracy'],-score['response_ce'])
            saved=dict(arm='streaming_lm',config=asdict(model.config),weights=model.state_dict(),case=case,step=step,
                       selection={k:v for k,v in score.items() if k!='rows'},training_cpu=cpu,tokenizer=plan['tokenizer_path'] if language else None)
            if best is None or selection>best:best=selection;best_step=step;save_checkpoint(directory/'best.pt',saved)
            save_checkpoint(directory/'last.pt',dict(saved,optimizer=optimizer.state_dict()))
            row['validation']=saved['selection'];quarter+=1
            event('streaming_validation',case=case['name'],step=step,cpu=cpu,score=saved['selection'])
        trace.append(row)
    save_lines(directory/'trace.jsonl',trace)
    parameters=sum(p.numel() for p in model.parameters());inactive=sum(p.numel() for p in model.energy_head.parameters()) if model.energy_head and model.config.rounds==0 else 0
    result=dict(case=case,parameters=parameters,active_parameters=parameters-inactive,best_step=best_step,selection=list(best),updates=step,
        target_tokens=target_tokens,batch_sha256=digest.hexdigest(),training_cpu_seconds=cpu,training_wall_seconds=wall,validation_cpu_seconds=eval_cpu,
        checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'))
    save_json(path,result);return result


def test(case,plan):
    verify(plan);out=Path(plan['output']);result=json.loads((out/f'{case["name"]}.json').read_text())
    assert file_sha256(result['checkpoint'])==result['checkpoint_sha256']
    model,_=load_streaming(result['checkpoint']);model.eval()
    if case['objective']!='tasks':
        corpus=StoryCorpus(DATA);tokenizer=StoryBPE.load(plan['tokenizer_path'])
        result['test']=language_evaluate(model,corpus,plan['test_offsets'],'test',tokenizer)
        result['test_long']=language_evaluate(model,corpus,plan['test_long_offsets'],'test',tokenizer,256)
        if case['variant']=='hybrid_energy':result['zero_rounds']=language_evaluate(model,corpus,plan['test_offsets'],'test',tokenizer,rounds=0)
    else:
        result['test']=task_evaluate(model,read_tasks(out/'tasks_test.jsonl'))
        result['test_long']=task_evaluate(model,read_tasks(out/'tasks_test_long.jsonl'))
        if case['variant']=='hybrid_energy':result['zero_rounds']=task_evaluate(model,read_tasks(out/'tasks_test.jsonl'),rounds=0)
    save_json(out/f'{case["name"]}_test.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports/rewt_streaming_20261026')
    p.add_argument('--runs',type=Path,default=ROOT/'runs/rewt_streaming_20261026')
    for name in ('prepare','train','test'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--language-updates',type=int,default=768)
    p.add_argument('--task-updates',type=int,default=1536);p.add_argument('--cpu-budget',type=float,default=90.)
    p.add_argument('--seeds',type=int,nargs='+',default=[26301,26302,26303])
    p.add_argument('--objectives',nargs='+',choices=['language','tasks','compute'],default=['language','tasks','compute'])
    args=p.parse_args();plan=prepare(args) if args.prepare else json.loads((args.output/'plan.json').read_text());verify(plan)
    if args.train and args.test:raise ValueError('train and test are separate stages')
    if args.test and not all((args.output/f'{c["name"]}.json').exists() for c in plan['cases']):raise RuntimeError('all declared fits must finish before test')
    if args.train or args.test:
        if args.test:save_json(args.output/'test_opened.json',dict(time=time.time(),plan_sha256=file_sha256(args.output/'plan.json')))
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            jobs=[pool.submit(fit if args.train else test,c,plan) for c in plan['cases']]
            results=[job.result() for job in as_completed(jobs)]
        save_json(args.output/('training_results.json' if args.train else 'test_results.json'),results)


if __name__=='__main__':main()
