"""Audit completed read/revise outputs and summarize without model calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rewt.data import file_sha256
from rewt.experiment import save_json
from rewt.tasks import score_response
from study_rewt_reasoning import read_tasks,check_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',default='reports/rewt_reasoning_20261026')
    args=p.parse_args();out=Path(args.study)
    plan=json.loads((out/'plan.json').read_text());check_plan(plan)
    results=json.loads((out/'test_results.json').read_text());arms={};audited=0
    for result in results:
        assert result['inference_weights_unchanged']
        assert file_sha256(result['checkpoint'])==result['checkpoint_sha256']
        for split in ('test','test_distractors','zero_energy','fixed_query'):
            if split not in result:continue
            tasks=read_tasks(out/f'tasks_{split if split.startswith("test") else "test"}.jsonl')
            rows=[json.loads(s) for s in (out/f'{result["case"]["name"]}_{split}_outputs.jsonl').read_text().splitlines()]
            assert len(tasks)==len(rows)
            for task,row in zip(tasks,rows):
                assert task.id==row['id'] and task.prompt==row['prompt']
                assert score_response(task,row['response'])['both_correct']==row['both_correct']
                audited+=1
            assert sum(r['both_correct'] for r in rows)/len(rows)==result[split]['complete']
    for variant in sorted({r['case']['variant'] for r in results}):
        selected=[r for r in results if r['case']['variant']==variant]
        data=dict(parameters=selected[0]['parameters'],active_parameters=selected[0]['active_parameters'],
                  seeds=[r['case']['seed'] for r in selected],
                  training_cpu_mean=float(np.mean([r['training_cpu_seconds'] for r in selected])),
                  validation_mean=float(np.mean([r['validation_complete'] for r in selected])),
                  max_energy_increase=max(r['max_within_read_energy_increase'] for r in selected))
        for split in ('test','test_distractors','zero_energy','fixed_query'):
            if split not in selected[0]:continue
            values=[r[split]['complete'] for r in selected]
            data[split]=dict(mean=float(np.mean(values)),min=min(values),max=max(values),values=values,
                groups={key:float(np.mean([r[split]['groups'][key] for r in selected])) for key in selected[0][split]['groups']},
                count_accuracy=float(np.mean([r[split]['count_accuracy'] for r in selected])))
        arms[variant]=data
    paired={}
    for variant in ('fixed_query_energy','reread_no_energy'):
        # Bootstrap semantic IDs while retaining both wordings and averaging
        # fixed seeds. The interval does not estimate all possible fit seeds.
        differences=defaultdict(list)
        for seed in plan['seeds']:
            a=[json.loads(s) for s in (out/f'reread_energy_{seed}_test_outputs.jsonl').read_text().splitlines()]
            b=[json.loads(s) for s in (out/f'{variant}_{seed}_test_outputs.jsonl').read_text().splitlines()]
            for x,y in zip(a,b):
                assert x['id']==y['id'] and x['template']==y['template']
                differences[x['steps'],x['id']].append(int(x['both_correct'])-int(y['both_correct']))
        strata=[np.array([np.mean(v) for (depth,key),v in differences.items() if depth==d]) for d in (1,2,3)]
        rng=np.random.default_rng(261026);boot=[]
        for _ in range(5000):
            boot.append(np.mean([rng.choice(s,len(s),replace=True).mean() for s in strata]))
        paired[variant]=dict(difference=float(np.mean([s.mean() for s in strata])),
                             ci95=np.quantile(boot,[.025,.975]).tolist(),
                             scope='Paired semantic-ID bootstrap, depth-stratified, both wordings and fixed three seeds kept together; not training-seed uncertainty.')
    chosen=max([r for r in results if r['case']['variant']=='reread_energy'],key=lambda r:(r['validation_complete'],-r['validation_ce'],-r['case']['seed']))
    for seed in plan['seeds']:
        assert len({r['training_examples_sha256'] for r in results if r['case']['seed']==seed})==1
    summary=dict(arms=arms,paired_differences=paired,audited_response_rows=audited,
                 selected_demo_checkpoint=chosen['checkpoint'],selected_demo_checkpoint_sha256=chosen['checkpoint_sha256'],
                 selected_demo_seed=chosen['case']['seed'],demo_selection='Validation only; best complete then lower answer CE then lower seed.',
                 training_cpu_total=sum(r['training_cpu_seconds'] for r in results),
                 validation_cpu_total=sum(r['validation_cpu_seconds'] for r in results),
                 protocol=plan['protocol'],limitations=plan['limitations'],test_sizes=plan['sizes'])
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
