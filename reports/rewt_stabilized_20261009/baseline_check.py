"""Supplementary validation-only Transformer allocation/LR check.

This preserves the five-arm primary protocol. All supplemental choices are
declared before inspecting the replacement study's held-out outputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))

import torch
from rewt.data import Corpus,file_sha256
from rewt.evaluate import language_metrics,reasoning_evaluation,text_evaluation
from rewt.experiment import event,new_model,save_json,save_lines,train
from rewt.improve import digest,get_tasks,worker_setup
from rewt.model import TransformerConfig,TransformerLM
from rewt.report import review_pack,statistics


def allocation(width):
    base=TransformerConfig(width=width,layers=2,feedforward=1)
    one=TransformerLM(base).parameters_count()
    two=TransformerLM(replace(base,feedforward=2)).parameters_count()
    cfg=replace(base,feedforward=round(1+(250_000-one)/(two-one)))
    assert abs(TransformerLM(cfg).parameters_count()-250_000)<1250
    return asdict(cfg)


def prepare():
    out=ROOT/'baseline_supplement'
    path=out/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        assert file_sha256(Path(__file__))==plan['supplement_script_sha256']
        for name,value in plan['source_sha256'].items():
            assert file_sha256(REPO/'rewt'/name)==value
        return plan
    primary=json.loads((ROOT/'selected_protocol.json').read_text())
    # Do not read result files, the text identity key, or generated examples.
    cases={f'w{w}_lr{lr:g}':dict(config=allocation(w),learning_rate=lr)
           for w in (48,80,96) for lr in (.001,.002,.004)}
    plan={**primary,'created':datetime.now(timezone.utc).isoformat(),'out':str(out),
          'runs':str(Path(primary['runs'])/'baseline_supplement'),
          'primary_report':str(ROOT),'screening_cases':cases,
          'supplement_script_sha256':file_sha256(Path(__file__)),
          'selection_rule':'Lowest 64-update language validation bpb; one seed; no test outputs',
          'settings':{**primary['settings'],'screen_updates':64},
          'protocol':'Supplementary baseline width and learning-rate selection, separately reported; primary protocol unchanged'}
    plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    out.mkdir()
    save_json(path,plan)
    for split in ('train','valid','test'):
        (out/f'tasks_{split}.jsonl').write_bytes((ROOT/f'tasks_{split}.jsonl').read_bytes())
    return plan


def fit(plan,seed,case):
    corpus=Corpus(plan['corpus']['path'])
    local={**plan,'settings':{**plan['settings'],'learning_rate':case['learning_rate']}}
    base=Path(plan['runs'])/f'seed_{seed}'/'transformer'
    out=Path(plan['out'])/f'seed_{seed}'
    recipe=plan['variants']['transformer']['recipe']
    lm,meta=train(new_model('transformer',case['config'],seed),'transformer',seed,'supplement_language',
                  local,corpus,base/'language',recipe=recipe,method='clip')
    reason,rmeta=train(new_model('transformer',case['config'],seed),'transformer',seed,'supplement_reasoning',
        local,corpus,base/'reasoning',initial_weights=lm.state_dict(),tasks=get_tasks(plan,'train'),
        updates=plan['settings']['reason_updates'],warm_updates=0,recipe=recipe,method='clip')
    language=language_metrics(lm,corpus,plan['test_offsets'],batch_size=plan['settings']['batch_size'],seed=1830)
    text,texts=text_evaluation(lm,plan['text_prompts'],seed=seed+1831)
    reasoning,records=reasoning_evaluation(reason,get_tasks(plan,'test'),seed=seed+1832)
    save_lines(out/'transformer_text.jsonl',texts)
    save_lines(out/'transformer_reasoning.jsonl',records)
    equal,eqmeta=train(new_model('transformer',case['config'],seed),'transformer',seed,'supplement_equal_time',
        local,corpus,base/'equal_time',updates=20_000,cpu_budget=plan['settings']['equal_cpu_seconds'],
        warm_updates=4,recipe=recipe,method='clip')
    eq=language_metrics(equal,corpus,plan['test_offsets'],batch_size=plan['settings']['batch_size'],seed=1834)
    tasks=[r for r in records if r['family']!='recall']
    row=dict(arm='transformer',seed=seed,config=case['config'],training=meta,reasoning_training=rmeta,
        language=language,text=text,adapted_reasoning=reasoning,
        reasoning_only=dict(count=len(tasks),answer_correct=sum(r['answer_correct'] for r in tasks)/len(tasks),
                            both_correct=sum(r['both_correct'] for r in tasks)/len(tasks)),
        equal_cpu_time=dict(training=eqmeta,language=eq),language_policy=dict(rounds=None,halting_tolerance=0.,validation=[]))
    save_json(out/'transformer.json',row)
    event('supplement_baseline_evaluated',seed=seed,test_bpb=language['bpb'],reasoning=row['reasoning_only'])
    return row


def main():
    torch.set_num_threads(1)
    plan=prepare()
    corpus=Corpus(plan['corpus']['path'])
    screen=[]
    for name,case in plan['screening_cases'].items():
        local={**plan,'settings':{**plan['settings'],'learning_rate':case['learning_rate'],
               'lm_updates':64,'validate_every':32,'budget_milestones':[]}}
        _,meta=train(new_model('transformer',case['config'],101),'transformer',101,'supplement_validation_screen',
            local,corpus,Path(plan['runs'])/'screen'/name,recipe=plan['variants']['transformer']['recipe'],method='clip',updates=64)
        row=dict(name=name,case=case,training=meta,validation_bpb=meta['best_validation_score'])
        screen.append(row)
        save_json(Path(plan['out'])/'screen'/f'{name}.json',row)
        event('supplement_screen',name=name,validation_bpb=row['validation_bpb'])
    chosen=min(screen,key=lambda r:(r['validation_bpb'],r['name']))
    selected={**plan,'selected_case':chosen,'settings':{**plan['settings'],'learning_rate':chosen['case']['learning_rate']},
              'variants':{'transformer':{'arm':'transformer','config':chosen['case']['config'],
                                       'recipe':plan['variants']['transformer']['recipe'],'method':'clip'}}}
    save_json(Path(plan['out'])/'selected_protocol.json',selected)
    event('supplement_selected',name=chosen['name'],validation_bpb=chosen['validation_bpb'])
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3,mp_context=context,initializer=worker_setup,initargs=(1,)) as pool:
        rows=[future.result() for future in as_completed([pool.submit(fit,selected,seed,chosen['case']) for seed in plan['settings']['seeds']])]
    aggregate={key:statistics(values) for key,values in dict(
        test_bpb=[r['language']['bpb'] for r in rows],
        equal_time_test_bpb=[r['equal_cpu_time']['language']['bpb'] for r in rows],
        reasoning_answer=[r['reasoning_only']['answer_correct'] for r in rows],
        reasoning_complete=[r['reasoning_only']['both_correct'] for r in rows],
        recall=[r['adapted_reasoning']['by_family']['recall']['both_correct'] for r in rows]).items()}
    save_json(Path(plan['out'])/'aggregate.json',aggregate)
    review_pack(Path(plan['out']),rows)
    text=['# Supplemental Transformer validation check','',
        'The five-arm primary protocol remains unchanged. Its Transformer was chosen by minimum parameter-count error, producing width 48 and feedforward width 1,029. This separate study screens hidden width and learning rate using the same language training/validation data, then evaluates the selected allocation across three seeds on the replacement study\'s already-declared fresh test samples. All weights start from scratch; language bytes, context resets, reasoning curriculum and adaptation budget match the revised primary models. This is a stronger baseline check, not an isolated estimate of a REWT repair effect.','',
        '| Width / learning rate | Validation bpb at 64 updates |','|---|---:|']
    for r in sorted(screen,key=lambda r:r['validation_bpb']):
        text.append(f"| {r['name']} | {r['validation_bpb']:.4f} |")
    text += ['',f"Selected **{chosen['name']}**; {TransformerLM(TransformerConfig(**chosen['case']['config'])).parameters_count():,} active parameters. No held-out score entered this selection.",'',
             '| Metric | Mean ± seed standard deviation |','|---|---:|']
    for key,scale in [('test_bpb',1),('equal_time_test_bpb',1),('reasoning_answer',100),('reasoning_complete',100),('recall',100)]:
        v=aggregate[key];text.append(f"| {key}{' %' if scale==100 else ''} | {v['mean']*scale:.3f} ± {v['std']*scale:.3f} |")
    text += ['','Nine additional single-seed screens are a supplemental tuning budget and are charged separately. The original primary baseline had no independent architecture/LR search; the primary REWT recipe screened fourteen cases. Short screening need not choose the best longer-run configuration. The full model\'s teacher remains the original primary Transformer; this supplemental baseline does not change its targets. The sampled supplemental continuations are saved and anonymized separately, with human scores pending.','']
    (Path(plan['out'])/'REPORT.md').write_text('\n'.join(text))
    save_json(Path(plan['out'])/'complete.json',dict(completed=datetime.now(timezone.utc).isoformat(),seeds=plan['settings']['seeds']))


if __name__=='__main__':main()
