"""Matched supervision ablation with a third, newly declared memory holdout.

Reuses the six binding-supervised checkpoints unchanged, trains response-only
controls with their exact seeds/batches/update count, and evaluates all twelve
on a fresh test set. The first two studies' test outputs are not reused here.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse,json,multiprocessing as mp
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.experiment import save_json,save_lines
from rewt.data import file_sha256
from rewt.architecture import load_architecture_checkpoint
import study_rewt_long_memory as memory
BASE=ROOT/'reports/rewt_architecture_20261022'
OUT=BASE/'memory_ablation';RUNS=ROOT/'runs/rewt_architecture_20261022/memory_ablation'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():raise FileExistsError('ablation already declared')
    plan=json.loads((BASE/'binding_memory/plan.json').read_text())
    (OUT/'valid.jsonl').write_bytes((BASE/'binding_memory/valid.jsonl').read_bytes())
    test=[r for length in (16,32,64) for near in (False,True)
          for r in memory.examples(24192+length+int(near)*1000,length,64,near)]
    ids={r['id'] for r in test}
    for folder in ('long_memory','binding_memory'):
        for split in ('valid','test'):
            old={json.loads(l)['id'] for l in (BASE/folder/f'{split}.jsonl').read_text().splitlines()}
            assert ids.isdisjoint(old)
    save_lines(OUT/'test.jsonl',test)
    plan['cases']=[dict(name=f'{name}_response_{seed}',seed=seed,delta_size=size)
                   for name,size in [('local',0),('delta',16)] for seed in (22201,22202,22203)]
    plan['supervised_controls']=[dict(source=str(BASE/'binding_memory'/f'{name}_{seed}.json'),family=name,seed=seed)
                                for name in ('local','delta') for seed in (22201,22202,22203)]
    plan['note']='Matched 1536 updates, seeds 22201/2/3, training batches, architecture, optimizer and validation selection. Response-only losses compared with the already-frozen binding/copy (and delta write-gate) supervision. A third test set is declared before any new fits or scores.'
    plan['data_sha256']={s:file_sha256(OUT/f'{s}.jsonl') for s in ('valid','test')}
    save_json(OUT/'plan.json',plan);return plan


def fit(case,plan):
    memory.OUT=OUT;memory.RUNS=RUNS
    return memory.fit(case,plan)


def test(item):
    torch.set_num_threads(1);memory.OUT=OUT;memory.RUNS=RUNS
    if 'source' in item:
        result=json.loads(Path(item['source']).read_text());family=item['family']+'_binding'
    else:
        result=json.loads((OUT/f'{item["name"]}.json').read_text());family=item['name'].rsplit('_',1)[0]
    model,_=load_architecture_checkpoint(result['checkpoint']);score,rows=memory.score(model,memory.read('test'))
    name=f'{family}_{result["case"]["seed"]}'
    result['ablation_family']=family;result['test']=score
    save_lines(OUT/f'{name}_test_outputs.jsonl',rows);save_json(OUT/f'{name}_test.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--test',action='store_true');a=p.parse_args()
    plan=prepare() if a.prepare else json.loads((OUT/'plan.json').read_text())
    if a.train or a.test:
        if a.test and not all((OUT/f'{c["name"]}.json').exists() for c in plan['cases']):raise RuntimeError('finish matched fits before test')
        with ProcessPoolExecutor(1,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit,c,plan) for c in plan['cases']] if a.train else [pool.submit(test,c) for c in plan['cases']+plan['supervised_controls']]
            results=[f.result() for f in futures]
        save_json(OUT/('training_results.json' if a.train else 'test_results.json'),results)

if __name__=='__main__':main()
