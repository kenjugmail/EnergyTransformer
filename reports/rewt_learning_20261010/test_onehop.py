"""Locked fresh-test evaluation; this file makes no parameter updates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from rewt.data import file_sha256, task_batch
from rewt.evaluate import reasoning_evaluation, refinement_audit, runtime_profile, task_loss
from rewt.experiment import save_json, save_lines
from rewt.memory import load_rewt_checkpoint
from rewt.tasks import Task, task_manifest

OUT=Path(__file__).resolve().parent/'onehop_test'
PARENT=OUT.parent


def weights_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.shape).encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def metrics(rows):
    groups={}
    for template_group,templates in [('known',[0,1]),('withheld_wording',[2])]:
        for family in ('arithmetic','rules','recall'):
            selected=[r for r in rows if r['family']==family and r['template'] in templates]
            groups[f'{template_group}/{family}']=dict(count=len(selected),correct=sum(r['both_correct'] for r in selected),
                complete_accuracy=sum(r['both_correct'] for r in selected)/max(1,len(selected)),
                answer_accuracy=sum(r['answer_correct'] for r in selected)/max(1,len(selected)))
    return groups


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    record=json.loads((PARENT/'binding/aligned.json').read_text())
    tasks=[Task(**json.loads(line)) for line in (PARENT/'foundation/tasks_test.jsonl').read_text().splitlines()]
    plan=dict(created=datetime.now(timezone.utc).isoformat(),checkpoint=record,
        checkpoint_sha256=file_sha256(Path(record['checkpoint'])),task_manifest=task_manifest(tasks),
        policies=[0,2],primary_policy=2,selection='Aligned one-head checkpoint selected by validation family macro; two rounds are the training/default policy and won the prior validation-only refinement sweep.',
        comparison='Zero rounds is the same trained model with its inference refinement disabled, not a separately trained matched competitor.',
        model_inputs='Raw ASCII prompts only. The generator receives no task payload, answer, trace, auxiliary head, solver or reference.',
        source_sha256={p.name:file_sha256(p) for p in (ROOT/'rewt').glob('*.py')},script_sha256=file_sha256(Path(__file__)))
    assert plan['checkpoint_sha256']==record['checkpoint_sha256']
    save_json(OUT/'plan.json',plan)
    model,_=load_rewt_checkpoint(record['checkpoint']);model.eval();before=weights_sha(model)
    results=[]
    for rounds in plan['policies']:
        start=time.process_time();summary,rows=reasoning_evaluation(model,tasks,seed=1751,rounds=rounds)
        result=dict(rounds=rounds,summary=summary,groups=metrics(rows),cpu_seconds=time.process_time()-start)
        save_lines(OUT/f'rounds_{rounds}_outputs.jsonl',rows);results.append(result)
        print(json.dumps(result),flush=True)
    batch=task_batch(tasks,range(4))
    audit=refinement_audit(model,batch.tokens)
    profile=runtime_profile(model,batch.tokens[:2,:32],repeats=3)
    after=weights_sha(model)
    assert before==after and file_sha256(Path(record['checkpoint']))==plan['checkpoint_sha256']
    assert audit['all_accepted_states_nonincreasing']
    result=dict(results=results,energy_audit=audit,runtime_profile=profile,parameters=model.parameters_count(),
        weights_before=before,weights_after=after,weights_unchanged=before==after,
        lineage_initialization_seeds=1,auxiliary_training_parameters=record['auxiliary_training_parameters'],
        limitations='One selected random-initialization lineage after validation exploration; bounded one-hop arithmetic/rules and recall delays 1/4. Three independent-seed replication and longer-chain generalization are not established by this test.')
    save_json(OUT/'results.json',result)


if __name__=='__main__':main()
