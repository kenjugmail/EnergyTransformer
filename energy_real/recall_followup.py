"""Bounded all-parameter versus controller-only recall training on fresh episodes."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import torch
from . import diagnostics as data
from . import recall_study as study
from .memory_eval import paired_interval
from .runtime import atomic_json, sha256, source_identity

MODES=('all','controller_only')
SEEDS=(11,22,33)
DATA_SEEDS={'train':271101,'valid':271102,'test':271103}


def choose(records,kind):
    # Seed-11 validation only chooses the mode; replication/test never choose it.
    return max((r for r in records if r['kind']==kind and r['seed']==11),
               key=lambda r:(r['validation']['selected']['accuracy'],
                             -r['validation']['selected']['unsupported_answers'],
                             r['validation']['selected']['support_accuracy'],
                             r['training_mode']=='controller_only'))['training_mode']


def compare_tests(tests,kind_a,mode_a,kind_b,mode_b,initial_b=False):
    a=[next(t for t in tests if (t['kind'],t['training_mode'],t['seed'])==(kind_a,mode_a,s)) for s in SEEDS]
    b=[next(t for t in tests if (t['kind'],t['training_mode'],t['seed'])==(kind_b,mode_b,s)) for s in SEEDS]
    arrays=[]
    for group,field in ((a,'trained'),(b,'initial' if initial_b else 'trained')):
        arrays.append([[float(r['correct']) for r in t[field]['rows']] for t in group])
    for ta,tb in zip(a,b):
        assert [r['episode_id'] for r in ta['trained']['rows']]==[r['episode_id'] for r in tb['trained']['rows']]
    av,bv=(torch.tensor(x,dtype=torch.float64) for x in arrays)
    return {'paired_interval':paired_interval(av.mean(0).tolist(),bv.mean(0).tolist()),
            'per_seed_differences':(av.mean(1)-bv.mean(1)).tolist()}


def cost_gate(costs,modes,quality):
    chosen={}
    for kind in study.KINDS:
        mode=modes[kind]
        rows=[r for r in costs[mode]['rows'] if r['backend']==kind]
        chosen[kind]={key:statistics.median(r[key] for r in rows)
                      for key in ('median_query_s','state_bytes','accounted_total_bytes')}
    energy,control=(chosen[k] for k in study.KINDS)
    reductions={key:1-energy[key]/control[key] for key in energy}
    supported=quality['paired_interval']['difference']>=-.01 and min(quality['per_seed_differences'])>=-.01
    latency=supported and reductions['median_query_s']>=.2
    storage=supported and reductions['accounted_total_bytes']>=.2
    return {'cpu_latency_passed':latency,'accounted_storage_passed':storage,'reductions':reductions,
            'selected_costs':chosen,'scope':'CPU query latency; complete component storage accounting. No GPU efficiency claim.'}


def run(args):
    if not 1<=args.steps<=500 or not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1800:
        raise ValueError('Follow-up requires 1–500 steps and a positive ceiling of at most 1800 seconds')
    if min(args.train_per_family,args.valid_per_family,args.test_per_family,args.threads)<1:
        raise ValueError('Positive dataset sizes and threads required')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    started=time.perf_counter();deadline=started+args.max_seconds;identity=source_identity()
    records=[];tests=[];selected=[];status='running';selection_sha256=None
    def progress():
        atomic_json({'status':status,'experiment':'controller_freeze_v1','source_sha256':identity,
                     'records':records,'selected':selected,'test_results':tests,
                     'elapsed_s':time.perf_counter()-started,'max_seconds':args.max_seconds,
                     'gpu_seconds':0,'selection_sha256':selection_sha256},args.out/'summary.json')
    progress()
    try:
        train=data.generate('train',args.train_per_family,DATA_SEEDS['train'])
        valid=data.generate('valid',args.valid_per_family,DATA_SEEDS['valid'])
        manifests={name:data.manifest(episodes) for name,episodes in [('train',train),('valid',valid)]}
        for name,manifest in manifests.items():atomic_json(manifest,args.out/f'{name}_episodes.json')
        recipe={'version':data.VERSION,**{name:{'seed':DATA_SEEDS[name],'sha256':manifest['sha256'],
                   'episodes':len(manifest['episodes'])} for name,manifest in manifests.items()},
                'test_recipe':{'split':'test','seed':DATA_SEEDS['test'],'per_family':args.test_per_family}}
        atomic_json(recipe,args.out/'datasets.json')
        for seed in SEEDS:
            for kind in study.KINDS:
                for mode in MODES:
                    record=study.fit(kind,64,.001,seed,args.steps,args.out/f'{kind}_{mode}_seed{seed}',
                                     train,valid,deadline,training_mode=mode)
                    records.append(record);progress()
        modes={kind:choose(records,kind) for kind in study.KINDS}
        selected.extend(r for r in records if r['training_mode']==modes[r['kind']])
        frozen={'status':'frozen_for_test','source_sha256':identity,'dataset_sha256':sha256(args.out/'datasets.json'),
                'selected':selected,'selected_modes':modes,'all_records':records,
                'selection_rule':'Seed-11 validation accuracy, fewer unsupported answers, support accuracy; ties prefer controller-only. Checkpoints selected on each seed validation; all settings frozen before test.',
                'test_recipe':recipe['test_recipe']}
        atomic_json(frozen,args.out/'selection.json');selection_sha256=sha256(args.out/'selection.json')
        test=data.generate('test',args.test_per_family,DATA_SEEDS['test'])
        atomic_json(data.manifest(test),args.out/'test_episodes.json')
        atomic_json(study.evaluate_exact(test,deadline),args.out/'exact_oracle.json')
        for record in records:
            row={'kind':record['kind'],'training_mode':record['training_mode'],'seed':record['seed'],
                 'trained':study.evaluate_config(record,test,deadline),
                 'initial':study.evaluate_config(record,test,deadline,initial=True)}
            row['learning_interval']=paired_interval([r['correct'] for r in row['trained']['rows']],
                                                     [r['correct'] for r in row['initial']['rows']])
            tests.append(row);progress()
        costs={}
        # Balance mode order by episode as well as mechanism order within each episode.
        episodes=data.generate('test',2,DATA_SEEDS['test'])
        for mode in MODES:costs[mode]={'rows':[]}
        for index,episode in enumerate(episodes):
            for mode in MODES[index%2:]+MODES[:index%2]:
                reps=[r for r in records if r['seed']==11 and r['training_mode']==mode]
                costs[mode]['rows'].extend(study.cost_comparison(reps,[episode],deadline,arm_offset=index)['rows'])
        for mode in MODES:
            reps=[r for r in records if r['seed']==11 and r['training_mode']==mode]
            costs[mode]['isolated_process_memory']=study.measure_peaks(reps,args.out/f'peak_{mode}',deadline,test_seed=DATA_SEEDS['test'])
        atomic_json(costs,args.out/'costs.json')
        mode_comparisons={kind:compare_tests(tests,kind,'controller_only',kind,'all') for kind in study.KINDS}
        learning={f'{kind}/{mode}':compare_tests(tests,kind,mode,kind,mode,True) for kind in study.KINDS for mode in MODES}
        quality=compare_tests(tests,'precision',modes['precision'],'preconditioned_delta',modes['preconditioned_delta'])
        quality_gate=quality['paired_interval']['difference']>=.05 and quality['paired_interval']['low']>0 and min(quality['per_seed_differences'])>0
        efficiency=cost_gate(costs,modes,quality)
        decision={'decision':'candidate_for_replication' if quality_gate or efficiency['cpu_latency_passed'] or efficiency['accounted_storage_passed'] else 'retain_bounded_result',
                  'selected_modes':modes,'energy_vs_classical':quality,'quality_gate_passed':quality_gate,
                  'efficiency':efficiency,'controller_only_vs_all':mode_comparisons,'learning':learning,
                  'learning_claims':{key:value['paired_interval']['low']>0 and min(value['per_seed_differences'])>0 for key,value in learning.items()},
                  'gpu_and_live_reasoning':'separate pending stages; no larger training authorized'}
        atomic_json(decision,args.out/'decision.json')
        if source_identity()!=identity:raise ValueError('Source changed during experiment')
        status='completed'
    except study.BudgetExpired as exc:
        status='budget_exhausted';atomic_json({'status':status,'reason':str(exc)},args.out/'incomplete.json')
    except BaseException as exc:
        status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        atomic_json({'status':status,'reason':f'{type(exc).__name__}: {exc}'},args.out/'incomplete.json')
        raise
    finally:progress()
    return json.loads((args.out/'summary.json').read_text())


def parser():
    p=study.parser()
    p.description=__doc__
    p.set_defaults(max_seconds=1800)
    return p


if __name__=='__main__':
    from .stages import interruption_handler
    with interruption_handler():result=run(parser().parse_args())
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','selected','test_results')}))
