"""Local-window versus persistent delta memory on learned multi-key overwrites.

Only prompt bytes enter the model. The last-write target is training/evaluation
metadata, never an inference module. Fresh seeds, fixed validation/test examples.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse,hashlib,json,multiprocessing as mp
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.architecture import ArchitectureConfig,ArchitectureEnergyLM,load_architecture_checkpoint
from rewt.experiment import save_json,save_lines,save_checkpoint,event
from rewt.data import file_sha256
OUT=ROOT/'reports/rewt_architecture_20261022/long_memory';RUNS=ROOT/'runs/rewt_architecture_20261022/long_memory'


def examples(seed,records,batch,near=False):
    rng=np.random.default_rng(seed);result=[]
    for _ in range(batch):
        query=int(rng.integers(8));old=int(rng.integers(10));new=(old+int(rng.integers(1,10)))%10
        overwrite=records-2 if near else max(1,records//4)
        facts=[]
        for i in range(records):
            if i==0:key,value=query,old
            elif i==overwrite:key,value=query,new
            else:
                key=int(rng.integers(7));key+=key>=query;value=int(rng.integers(10))
            facts.append(f'{chr(65+key)}={value};')
        prompt=''.join(facts)+f'?{chr(65+query)}='
        result.append(dict(prompt=prompt,target=ord(str(new)),query=chr(65+query),records=records,
                           gap=4*(records-1-overwrite)+3,near=near,id=hashlib.sha256(prompt.encode()).hexdigest()))
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():raise FileExistsError('study already exists')
    rows={split:[t for records in lengths for near in (False,True) for t in examples(seed+records+int(near)*1000,records,64,near)]
          for split,lengths,seed in [('valid',(8,16,24),22191),('test',(16,32,64),22192)]}
    assert {r['id'] for r in rows['valid']}.isdisjoint({r['id'] for r in rows['test']})
    for split,data in rows.items():save_lines(OUT/f'{split}.jsonl',data)
    cfg=ArchitectureConfig(width=16,hidden=96,modes=8,history=16,position_period=16,
        context_kernel=3,recurrent_type='gru',output_geometry='centered',compatibility_geometry='anchor_relative',
        wave_geometry='anchor_relative',initial_carry_scale=0.,backbone='scan',solver='krylov2')
    plan=dict(config=asdict(cfg),updates=768,batch_size=24,validation_every=192,
        cases=[dict(name=f'{name}_{seed}',seed=seed,delta_size=size) for name,size in [('local',0),('delta',16)] for seed in (22101,22102,22103)],
        train_records=[8,16,24],test_records=[16,32,64],selection='validation accuracy on balanced near/far overwrite queries',
        note='Keys A..H and values 0..9. Requested key written twice with a different value; distractors contain other keys only. Correct answer is the latest write. Far queries exceed the 16-byte local window; test includes 259-byte prompts. Cold learned memory, no symbolic executor or labels at inference.',
        data_sha256={s:file_sha256(OUT/f'{s}.jsonl') for s in rows})
    save_json(OUT/'plan.json',plan);return plan


def read(split):return [json.loads(l) for l in (OUT/f'{split}.jsonl').read_text().splitlines()]


@torch.inference_mode()
def score(model,rows):
    model.eval().prepare_inference();records=[]
    for length in sorted({len(r['prompt']) for r in rows}):
        subset=[r for r in rows if len(r['prompt'])==length]
        for start in range(0,len(subset),32):
            group=subset[start:start+32];x=torch.tensor([list(r['prompt'].encode()) for r in group])
            prediction=model(x).logits[:,-1].argmax(-1).tolist()
            records.extend(dict(**r,prediction=p,correct=p==r['target']) for r,p in zip(group,prediction))
    groups={f'{r}/{kind}':sum(t['correct'] for t in records if t['records']==r and t['near']==kind)/sum(t['records']==r and t['near']==kind for t in records)
            for r in sorted({t['records'] for t in records}) for kind in (False,True)}
    return dict(accuracy=sum(t['correct'] for t in records)/len(records),groups=groups),records


def fit(case,plan):
    torch.set_num_threads(1);path=OUT/f'{case["name"]}.json';directory=RUNS/case['name'];directory.mkdir(parents=True,exist_ok=True)
    if path.exists():return json.loads(path.read_text())
    torch.manual_seed(case['seed']);model=ArchitectureEnergyLM(ArchitectureConfig(**dict(plan['config'],delta_size=case['delta_size'])))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);best=-1.;trace=[];cpu=0.
    heldout={r['id'] for r in read('valid')+read('test')}
    for step in range(plan['updates']):
        start=time.process_time();length=plan['train_records'][step%3]
        rows=examples(case['seed']*100000+step,length,24,near=step%4==0)
        if any(r['id'] in heldout for r in rows):raise ValueError('training overlaps declared heldout')
        x=torch.tensor([list(r['prompt'].encode()) for r in rows]);y=torch.tensor([r['target'] for r in rows])
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x,state_gradient_interval=0).logits[:,-1],y)
        loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step();cpu+=time.process_time()-start
        record=dict(step=step+1,loss=float(loss.detach()),gradient=gradient)
        if (step+1)%plan['validation_every']==0:
            result,_=score(model,read('valid'));record['validation']=result
            if result['accuracy']>best:
                best=result['accuracy'];save_checkpoint(directory/'best.pt',dict(arm='architecture_energy',config=asdict(model.config),seed=case['seed'],weights=model.state_dict(),selection=result,step=step+1))
            event('long_memory_validation',case=case['name'],step=step+1,accuracy=result['accuracy'])
        trace.append(record)
    save_lines(directory/'trace.jsonl',trace)
    result=dict(case=case,validation_accuracy=best,parameters=model.parameters_count(),training_cpu_seconds=cpu,checkpoint=str(directory/'best.pt'),checkpoint_sha256=file_sha256(directory/'best.pt'))
    save_json(path,result);return result


def test(case):
    torch.set_num_threads(1);result=json.loads((OUT/f'{case["name"]}.json').read_text());model,_=load_architecture_checkpoint(result['checkpoint'])
    result['test'],rows=score(model,read('test'));save_lines(OUT/f'{case["name"]}_test_outputs.jsonl',rows);save_json(OUT/f'{case["name"]}_test.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--test',action='store_true');p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    plan=prepare() if a.prepare else json.loads((OUT/'plan.json').read_text())
    if a.train or a.test:
        if a.test and not all((OUT/f'{c["name"]}.json').exists() for c in plan['cases']):raise RuntimeError('finish training before test')
        with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit,c,plan) if a.train else pool.submit(test,c) for c in plan['cases']]
            results=[f.result() for f in futures]
        save_json(OUT/('training_results.json' if a.train else 'test_results.json'),results)

if __name__=='__main__':main()
