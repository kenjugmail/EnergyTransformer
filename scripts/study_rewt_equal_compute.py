"""Declared equal-process-CPU language comparison with a fresh document holdout."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import argparse,json,math,multiprocessing as mp,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM,load_architecture_checkpoint
from rewt.bpe import StoryBPE
from rewt.story_data import StoryCorpus
from rewt.experiment import save_json,save_lines,save_checkpoint,event
from rewt.data import file_sha256
from study_rewt_architecture_language import DATA,evaluate
BASE=ROOT/'reports/rewt_architecture_20261022'
OUT=BASE/'equal_compute';RUNS=ROOT/'runs/rewt_architecture_20261022/equal_compute'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():raise FileExistsError('preserve the declared compute study')
    original=json.loads((BASE/'language/plan.json').read_text());corpus=StoryCorpus(DATA)
    spans=corpus.documents['test'];used={int(np.searchsorted(spans[:,0],o,side='right'))-1 for o in original['test_offsets']}
    eligible=[i for i,(a,b) in enumerate(spans) if b-a>128 and i not in used]
    rng=np.random.default_rng(22383);selected=rng.choice(eligible,256,replace=False)
    offsets=[int(rng.integers(spans[i,0],spans[i,1]-128)) for i in selected]
    variants={'control':{},'precond2':dict(solver='preconditioned',step_size=.75),
        'precond4':dict(solver='preconditioned',step_size=.75,rounds=4),
        'precond8':dict(solver='preconditioned',step_size=.75,rounds=8),
        'precond4_structured':dict(solver='preconditioned',step_size=.75,rounds=4,preconditioner_factorization='structured')}
    plan=dict(config=original['config'],cpu_budget_seconds=150.,validation_fractions=[.25,.5,.75,1.],
        cases=[dict(name=f'{n}_{seed}',variant=n,seed=seed,overrides=o) for n,o in variants.items() for seed in (22301,22302,22303)],
        context=128,batch_size=24,learning_rate=.001,valid_offsets=original['valid_offsets'],test_offsets=offsets,
        excluded_previous_test_documents=sorted(used),test_documents=selected.tolist(),
        data_manifest_sha256=file_sha256(DATA/'manifest.json'),tokenizer_sha256=original['tokenizer_sha256'],
        note='Separate CPU-budget screening. 150 measured training process seconds per fit, excluding initialization, validation and checkpoint I/O; at most one update overshoot. LR and validation schedule follow measured consumed CPU budget. Matched seeds and per-update batches; faster arms can execute more updates. Same model parameter counts. Test documents exclude the 256 already evaluated language documents; no test outcomes used for selection. This shorter budget does not establish fluent text.',
        source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')})
    save_json(OUT/'plan.json',plan);return plan


def init_worker():
    torch.set_num_threads(1)
    os.sched_setaffinity(0,{(5,6,8,9)[(mp.current_process()._identity[0]-1)%4]})


def fit(case,plan):
    path=OUT/f'{case["name"]}.json';directory=RUNS/case['name'];directory.mkdir(parents=True,exist_ok=True)
    if path.exists():return json.loads(path.read_text())
    if (directory/'last.pt').exists():raise RuntimeError('partial compute fit requires inspection')
    torch.manual_seed(case['seed']);model=ArchitectureEnergyLM(ArchitectureConfig(**dict(plan['config'],**case['overrides'])))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01)
    corpus=StoryCorpus(DATA);vocabulary=ROOT/'data/tinystories_bpe_20261010/vocabulary.json';tokenizer=StoryBPE.load(vocabulary)
    assert file_sha256(DATA/'manifest.json')==plan['data_manifest_sha256']
    cpu=wall=0.;step=0;threshold=0;best=float('inf');best_step=0;trace=[]
    event('equal_compute_started',case=case['name'])
    while cpu<plan['cpu_budget_seconds']:
        begin=time.process_time();wall_begin=time.perf_counter()
        fraction=cpu/plan['cpu_budget_seconds'];rate=.001*min(1.,(fraction+.005)/.05)*(.3+.7*(1+math.cos(math.pi*fraction))/2)
        rng=np.random.default_rng(case['seed']*100000+step);batch=corpus.batch('train',rng,24,128)
        model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=rate
        logits=model(batch.tokens,state_gradient_interval=32).logits
        loss=F.cross_entropy(logits.flatten(0,1),batch.targets.flatten())
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
        cpu+=time.process_time()-begin;wall+=time.perf_counter()-wall_begin;step+=1
        row=dict(step=step,loss=float(loss.detach()),gradient=gradient,cumulative_training_cpu_seconds=cpu)
        if cpu>=plan['cpu_budget_seconds']*plan['validation_fractions'][threshold]:
            score=evaluate(model,corpus,plan['valid_offsets'],'valid',tokenizer)
            saved=dict(arm='architecture_energy',config=asdict(model.config),seed=case['seed'],weights=model.state_dict(),
                step=step,training_cpu_seconds=cpu,selection=score,case=case,
                tokenizer=dict(path=str(vocabulary),sha256=plan['tokenizer_sha256']))
            if score['bpb']<best:best=score['bpb'];best_step=step;save_checkpoint(directory/'best.pt',saved)
            save_checkpoint(directory/'last.pt',dict(saved,optimizer=optimizer.state_dict()))
            row['validation']=score;threshold+=1;event('equal_compute_validation',case=case['name'],step=step,cpu=cpu,bpb=score['bpb'])
        trace.append(row)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=case,parameters=model.parameters_count(),validation_bpb=best,best_step=best_step,updates=step,
        training_tokens=step*24*128,training_cpu_seconds=cpu,training_wall_seconds=wall,
        checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'))
    save_json(path,result);return result


def test(case,plan):
    result=json.loads((OUT/f'{case["name"]}.json').read_text());model,_=load_architecture_checkpoint(result['checkpoint'])
    corpus=StoryCorpus(DATA);tokenizer=StoryBPE.load(ROOT/'data/tinystories_bpe_20261010/vocabulary.json')
    result['test']=evaluate(model,corpus,plan['test_offsets'],'test',tokenizer)
    save_json(OUT/f'{case["name"]}_test.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--test',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    plan=prepare() if a.prepare else json.loads((OUT/'plan.json').read_text())
    if a.train or a.test:
        if a.test and not all((OUT/f'{c["name"]}.json').exists() for c in plan['cases']):raise RuntimeError('complete every fit before opening the compute holdout')
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            futures=[pool.submit(fit if a.train else test,c,plan) for c in plan['cases']]
            rows=[f.result() for f in as_completed(futures)]
        save_json(OUT/('training_results.json' if a.train else 'test_results.json'),rows)

if __name__=='__main__':main()
