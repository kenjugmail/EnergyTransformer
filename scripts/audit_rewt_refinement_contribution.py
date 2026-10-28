"""Validation-only removal of refinement from frozen trained checkpoints.

No retraining or test selection. Removing refinement also changes future carried
states in the GRU arm, so this is an intervention on the full execution policy.
"""
from dataclasses import replace
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.architecture import ArchitectureEnergyLM,load_architecture_checkpoint
from rewt.experiment import save_json,save_lines
from study_rewt_architecture import OUT,read_tasks,validation


def main():
    torch.set_num_threads(1);tasks=read_tasks(OUT/'tasks_valid.jsonl');records=[]
    for name in ('control','precond4','delta'):
        for seed in (22101,22102,22103):
            case=f'{name}_{seed}';record=json.loads((OUT/f'{case}.json').read_text())
            model,_=load_architecture_checkpoint(record['checkpoint'])
            original={n:t.clone() for n,t in model.state_dict().items()}
            zero=ArchitectureEnergyLM(replace(model.config,rounds=0));zero.load_state_dict(model.state_dict())
            score,rows=validation(zero,tasks,seed)
            saved_rows=[json.loads(l) for l in (OUT/f'{case}_validation_outputs.jsonl').read_text().splitlines()]
            assert len(rows)==len(saved_rows)
            assert all(torch.equal(t,original[n]) for n,t in zero.state_dict().items())
            def row_key(r):return r['id'],r['prompt']
            normal={row_key(r):r for r in saved_rows}
            # Preserve full outputs so changes in formatting and correctness can
            # be inspected, including cases where accuracy happens to tie.
            save_lines(OUT/f'{case}_no_refinement_validation.jsonl',rows)
            changed=sum(r['response']!=normal[row_key(r)]['response'] for r in rows)
            records.append(dict(case=case,normal_rounds=model.config.rounds,normal_complete=record['validation_complete'],
                no_refinement_complete=score['macro_complete'],normal_response_ce=record['validation_ce'],
                no_refinement_response_ce=score['response_ce'],changed_generations=changed,total_generations=len(rows),
                weights_unchanged=True))
    save_json(OUT/'refinement_contribution.json',dict(rows=records,validation_only=True,
        note='Post-training zero-round intervention, no retraining, no policy selection. GRU future carry also changes when refinement is removed. Results are conditional on the learned parameters and this small validation set.'))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
