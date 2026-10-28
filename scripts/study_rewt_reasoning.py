"""Controlled learned multi-read study; raw bytes in, predicted trace out.

Prepare and train before test. All three arms share initialization, batches,
supervision and update budget. Labels are used here only, never in the model.
The structured answer head changes the task interface: do not compare its
percentages directly with old autoregressive language-model experiments.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.curriculum import render_task
from rewt.data import file_sha256
from rewt.experiment import save_checkpoint, save_json, save_lines, event
from rewt.reasoning import ReadReviseReasoner, ReasoningConfig, load_reasoner
from rewt.tasks import Task, semantic_identity, semantic_partition, solve, score_response


def read_tasks(path):
    return [Task(**json.loads(line)) for line in Path(path).read_text().splitlines()]


def make_tasks(split, per_depth, seed, excluded, distractors=2):
    rng = random.Random(seed)
    used, result = set(excluded), []
    for depth in (1, 2, 3):
        count = 0
        while count < per_depth:
            # A disconnected distractor path has exactly `distractors` edges.
            nodes = rng.sample(list('ABCDEFGHIJKLMNOPQRSTUVWX'), depth+distractors+2)
            edges = [[nodes[i], nodes[i+1]] for i in range(depth)]
            edges += [[nodes[depth+1+i], nodes[depth+2+i]] for i in range(distractors)]
            payload = dict(family='rules', start=nodes[0], hops=depth, edges=sorted(edges))
            identity = semantic_identity(payload)
            if identity in used or semantic_partition(identity) != split:
                continue
            templates = [rng.randrange(5)]
            if split != 'train':
                templates += [rng.choice((5,6) if split == 'valid' else (7,8))]
            result.extend(render_task(payload, split, template, seed=rng.randrange(2**32)) for template in templates)
            used.add(identity); count += 1
    return result


def prepare(args):
    out, runs = Path(args.output).resolve(), Path(args.runs).resolve()
    if (out/'plan.json').exists():
        raise FileExistsError('study exists; do not overwrite it')
    excluded, prior = set(), []
    for path in sorted((ROOT/'reports').rglob('tasks_test*.jsonl')):
        if out in path.parents:
            continue
        try:
            ids = {t.id for t in read_tasks(path)}
        except (TypeError, KeyError):
            continue
        excluded.update(ids)
        prior.append(dict(path=str(path), sha256=file_sha256(path), identities=len(ids)))
    sets = dict(train=make_tasks('train', args.train_per_depth, 261001, set()),
                valid=make_tasks('valid', args.valid_per_depth, 261002, set()),
                test=make_tasks('test', args.test_per_depth, 261003, excluded))
    test_ids = {t.id for t in sets['test']}
    sets['test_distractors'] = make_tasks('test', args.test_per_depth//2, 261004, excluded|test_ids, distractors=8)
    ids = {s:{t.id for t in ts} for s,ts in sets.items()}
    for s,a in ids.items():
        for t,b in ids.items():
            if s != t:
                assert a.isdisjoint(b)
    assert (ids['test']|ids['test_distractors']).isdisjoint(excluded)
    for split,tasks in sets.items():
        save_lines(out/f'tasks_{split}.jsonl',[t.to_dict() for t in tasks])
    variants = dict(reread_energy=dict(), fixed_query_energy=dict(reread=False),
                    reread_no_energy=dict(energy_rounds=0))
    cases = [dict(name=f'{variant}_{seed}',variant=variant,seed=seed,config=asdict(ReasoningConfig(**overrides)))
             for variant,overrides in variants.items() for seed in args.seeds]
    paths = [ROOT/'rewt/reasoning.py',ROOT/'rewt/tasks.py',ROOT/'rewt/curriculum.py',Path(__file__)]
    plan = dict(output=str(out),runs=str(runs),updates=args.updates,batch_size=64,learning_rate=.003,
                validation_every=max(1,args.updates//4),seeds=args.seeds,cases=cases,
                data_sha256={s:file_sha256(out/f'tasks_{s}.jsonl') for s in sets},
                source_sha256={str(p):file_sha256(p) for p in paths},previous_tests=prior,
                excluded_previous_test_identities=len(excluded),
                sizes={s:dict(prompts=len(ts),identities=len(ids[s])) for s,ts in sets.items()},
                selection='Per-seed highest validation complete trace+answer accuracy; lower final-answer CE breaks ties. All arms reported. Fresh test once all fits are finished.',
                protocol='Same initialized tensors, per-seed batches, supervision, optimizer and updates. Structured head, not autoregressive output; predicted hop count selects decoded intermediate states and final answer. All annotation and symbolic solving restricted to training/scoring.',
                supervision=dict(final=1.,trace=1.,pointer=.5,key=1.,value=1.,gate=.5,start=1.,count=1.),
                limitations='Narrow acyclic letter-rule grammar, at most three hops, 24 node symbols. Two training distractor edges; eight only in extra test. Same allocated parameters but zero-energy arm has unused energy-only parameters. Equal updates, not compute. No Transformer-superiority or general-language claim.',
                test_not_evaluated=True)
    save_json(out/'plan.json',plan)
    return plan


def check_plan(plan):
    for path,digest in plan['source_sha256'].items():
        assert file_sha256(path)==digest, f'source changed while study active: {path}'
    for s,digest in plan['data_sha256'].items():
        assert file_sha256(Path(plan['output'])/f'tasks_{s}.jsonl')==digest


def annotations(task):
    patterns = [r'from ([A-Z]) go',r'walk ([A-Z]) ',r';([A-Z]) after',r'from ([A-Z]):',
                r'follow ([A-Z])/',r'to ([A-Z])=',r'hops \d+:([A-Z]):',r'from ([A-Z])=',r'find ([A-Z]) in']
    query = re.search(patterns[task.template],task.prompt)
    assert query and query[1]==task.payload['start']
    facts = [(m.end()-1,ord(m[1]),ord(m[2])) for m in re.finditer(r'([A-Z])>([A-Z])',task.prompt)]
    positions = {chr(src):at for at,src,dst in facts}
    trace,answer = solve(task.payload)
    sources = [task.payload['start'],*trace[:-1]]
    return dict(start=query.start(1),facts=facts,trace=[ord(v) for v in trace],
                pointers=[positions[s] for s in sources],answer=ord(answer),hops=task.steps)


def tensor_batch(tasks):
    raw = [t.prompt.encode('ascii') for t in tasks]
    lengths = torch.tensor([len(p) for p in raw])
    tokens = torch.zeros(len(tasks),int(lengths.max()),dtype=torch.long)
    for i,p in enumerate(raw):
        tokens[i,:len(p)] = torch.tensor(list(p))
    return tokens,lengths


def losses(model,tasks):
    tokens,lengths = tensor_batch(tasks)
    output = model(tokens,lengths)
    ann = [annotations(t) for t in tasks]
    target = torch.tensor([a['answer'] for a in ann])
    result = dict(final=F.cross_entropy(output['answer_logits'],target),
                  count=F.cross_entropy(output['count_logits'],torch.tensor([a['hops']-1 for a in ann])),
                  start=F.cross_entropy(output['start_logits'],torch.tensor([a['start'] for a in ann])))
    row,stage,truth,ptr,fr,fp,src,dst = [],[],[],[],[],[],[],[]
    gates = torch.zeros_like(output['fact_logits'])
    for i,a in enumerate(ann):
        for j,v in enumerate(a['trace']):
            row.append(i); stage.append(j); truth.append(v); ptr.append(a['pointers'][j])
        for at,s,v in a['facts']:
            fr.append(i); fp.append(at); src.append(s); dst.append(v); gates[i,at]=1
    result['trace'] = F.cross_entropy(output['step_logits'][row,stage],torch.tensor(truth))
    result['pointer'] = F.cross_entropy(output['pointer_logits'][row,stage],torch.tensor(ptr))
    result['key'] = F.cross_entropy(output['key_logits'][fr,fp],torch.tensor(src))
    result['value'] = F.cross_entropy(output['value_logits'][fr,fp],torch.tensor(dst))
    result['gate'] = F.binary_cross_entropy_with_logits(output['fact_logits'][output['mask']],gates[output['mask']],pos_weight=torch.tensor(8.))
    return result


@torch.inference_mode()
def evaluate(model,tasks,**kwargs):
    model.eval(); rows=[]; nll=0.
    for first in range(0,len(tasks),64):
        selected=tasks[first:first+64]
        predictions=model.predict([t.prompt for t in selected],**kwargs)
        tokens,lengths=tensor_batch(selected)
        output=model(tokens,lengths,**kwargs)
        truth=torch.tensor([ord(solve(t.payload)[1]) for t in selected])
        nll+=float(F.cross_entropy(output['answer_logits'],truth,reduction='sum'))
        for t,p in zip(selected,predictions):
            rows.append(dict(id=t.id,prompt=t.prompt,template=t.template,steps=t.steps,difficulty=t.difficulty,
                             **{**p,**score_response(t,p['response'])}))
    groups={}
    for depth in (1,2,3):
        for kind in ('familiar_format','withheld_wording'):
            group=[r for r in rows if r['steps']==depth and r['difficulty']==kind]
            if group:
                groups[f'{depth}/{kind}']=sum(r['both_correct'] for r in group)/len(group)
    return dict(complete=sum(r['both_correct'] for r in rows)/len(rows),
                answer=sum(r['answer_correct'] for r in rows)/len(rows),
                valid_format=sum(r['valid_format'] for r in rows)/len(rows),
                count_accuracy=sum(r['predicted_hops']==r['steps'] for r in rows)/len(rows),
                response_ce=nll/len(rows),groups=groups,prompts=len(rows)),rows


def fit(case,plan):
    torch.set_num_threads(1); check_plan(plan)
    out=Path(plan['output']);directory=Path(plan['runs'])/case['name']
    if (out/f'{case["name"]}.json').exists():
        return json.loads((out/f'{case["name"]}.json').read_text())
    if directory.exists():
        raise FileExistsError(f'partial fit preserved: {directory}')
    directory.mkdir(parents=True)
    torch.manual_seed(case['seed']);model=ReadReviseReasoner(ReasoningConfig(**case['config']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],betas=(.9,.95),weight_decay=.01)
    train,valid=read_tasks(out/'tasks_train.jsonl'),read_tasks(out/'tasks_valid.jsonl')
    best=(-1.,-float('inf')); trace=[];cpu=validation_cpu=0.;maximum=0.;example_hash=__import__('hashlib').sha256()
    event('read_revise_started',case=case['name'],parameters=sum(p.numel() for p in model.parameters()))
    for step in range(plan['updates']):
        begin=time.process_time();rng=np.random.default_rng(case['seed']*1000000+step)
        selected=[render_task(train[int(i)].payload,'train',int(rng.integers(5)),seed=int(rng.integers(2**32)))
                  for i in rng.integers(len(train),size=plan['batch_size'])]
        for task in selected:
            example_hash.update(task.prompt.encode('ascii'))
        rate=plan['learning_rate']*min(1.,(step+1)/16)*(.3+.7*(1+math.cos(math.pi*step/plan['updates']))/2)
        for group in optimizer.param_groups:group['lr']=rate
        model.train();optimizer.zero_grad(set_to_none=True)
        parts=losses(model,selected);loss=sum(plan['supervision'][k]*v for k,v in parts.items())
        if not torch.isfinite(loss):raise FloatingPointError(case['name'])
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(norm):raise FloatingPointError('gradient')
        optimizer.step();cpu+=time.process_time()-begin;maximum=max(maximum,norm)
        row=dict(step=step+1,loss=float(loss.detach()),losses={k:float(v.detach()) for k,v in parts.items()},gradient_norm=norm,training_cpu=cpu)
        if (step+1)%plan['validation_every']==0 or step+1==plan['updates']:
            begin=time.process_time();score,rows=evaluate(model,valid);validation_cpu+=time.process_time()-begin
            rank=score['complete'],-score['response_ce']
            if rank>best:
                best=rank
                save_checkpoint(directory/'best.pt',dict(arm='read_revise_reasoner',config=asdict(model.config),
                    weights=model.state_dict(),seed=case['seed'],step=step+1,selection=score,case=case))
                save_lines(out/f'{case["name"]}_validation_outputs.jsonl',rows)
            row['validation']=score
            event('read_revise_validation',case=case['name'],step=step+1,complete=score['complete'],best=best[0],groups=score['groups'])
        elif (step+1)%100==0:event('read_revise_progress',case=case['name'],step=step+1,loss=float(loss.detach()))
        trace.append(row)
    save_lines(directory/'trace.jsonl',trace)
    check_plan(plan)
    result=dict(case=case,parameters=sum(p.numel() for p in model.parameters()),
                active_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
                validation_complete=best[0],validation_ce=-best[1],updates=plan['updates'],
                training_cpu_seconds=cpu,validation_cpu_seconds=validation_cpu,maximum_gradient=maximum,
                training_examples_sha256=example_hash.hexdigest(),checkpoint=str(directory/'best.pt'),
                checkpoint_sha256=file_sha256(directory/'best.pt'))
    save_json(out/f'{case["name"]}.json',result)
    return result


def test(case,plan):
    torch.set_num_threads(1);check_plan(plan);out=Path(plan['output'])
    result=json.loads((out/f'{case["name"]}.json').read_text());model,_=load_reasoner(result['checkpoint'])
    assert file_sha256(result['checkpoint'])==result['checkpoint_sha256']
    before={k:v.clone() for k,v in model.state_dict().items()}
    for split in ('test','test_distractors'):
        tasks=read_tasks(out/f'tasks_{split}.jsonl');begin=time.process_time()
        score,rows=evaluate(model,tasks);score['generation_and_scoring_cpu']=time.process_time()-begin
        save_lines(out/f'{case["name"]}_{split}_outputs.jsonl',rows)
        result[split]=score
    # Prespecified frozen-policy intervention: independent rereads and waves.
    if case['variant']=='reread_energy':
        for name,kwargs in [('zero_energy',dict(energy_rounds=0)),('fixed_query',dict(reread=False))]:
            score,rows=evaluate(model,read_tasks(out/'tasks_test.jsonl'),**kwargs)
            result[name]=score
            save_lines(out/f'{case["name"]}_{name}_outputs.jsonl',rows)
    tasks=read_tasks(out/'tasks_valid.jsonl')[:18];tokens,lengths=tensor_batch(tasks)
    with torch.inference_mode():
        e=model(tokens,lengths,details=True)['energy_trace']
    result['max_within_read_energy_increase']=float((e[...,1:]-e[...,:-1]).max()) if e.shape[-1]>1 else 0.
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert file_sha256(result['checkpoint'])==result['checkpoint_sha256']
    result['inference_weights_unchanged']=True
    save_json(out/f'{case["name"]}_test.json',result)
    event('read_revise_test',case=case['name'],complete=result['test']['complete'],extra=result['test_distractors']['complete'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/rewt_reasoning_20261026');p.add_argument('--runs',default='runs/rewt_reasoning_20261026')
    p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--test',action='store_true')
    p.add_argument('--updates',type=int,default=800);p.add_argument('--seeds',type=int,nargs='+',default=[26101,26102,26103])
    p.add_argument('--train-per-depth',type=int,default=2000);p.add_argument('--valid-per-depth',type=int,default=60)
    p.add_argument('--test-per-depth',type=int,default=160);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();torch.set_num_threads(1)
    plan=prepare(args) if args.prepare else json.loads((Path(args.output)/'plan.json').read_text())
    check_plan(plan)
    if args.train:
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
            results=list(pool.map(fit,plan['cases'],[plan]*len(plan['cases'])))
        for seed in plan['seeds']:
            assert len({r['training_examples_sha256'] for r in results if r['case']['seed']==seed})==1
        save_json(Path(plan['output'])/'training_results.json',results)
    if args.test:
        out=Path(plan['output'])
        if (out/'test_plan.json').exists():raise FileExistsError('test already opened; preserve the frozen result')
        trained=[json.loads((out/f'{c["name"]}.json').read_text()) for c in plan['cases']]
        save_json(out/'test_plan.json',dict(cases=trained,data=plan['data_sha256'],source=plan['source_sha256'],
            interventions=['zero_energy','fixed_query'],selection='All declared fits completed before test; no further training selected using these results.'))
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
            results=list(pool.map(test,plan['cases'],[plan]*len(plan['cases'])))
        save_json(out/'test_results.json',results)
    check_plan(plan)


if __name__=='__main__':main()
