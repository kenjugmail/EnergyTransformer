"""Geometry and energy audits of validation-selected trained checkpoints."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rewt.architecture import load_architecture_checkpoint
from rewt.data import file_sha256
OUT=ROOT/'reports/rewt_architecture_20261022'


@torch.no_grad()
def audit(path,prompts):
    model,saved=load_architecture_checkpoint(path);model.eval().prepare_inference()
    tokens=torch.zeros(len(prompts),max(map(len,prompts)),dtype=torch.long)
    valid=torch.zeros_like(tokens,dtype=torch.bool)
    for row,prompt in enumerate(prompts):tokens[row,:len(prompt)]=torch.tensor(list(prompt.encode()));valid[row,:len(prompt)]=True
    original={n:t.clone() for n,t in model.state_dict().items()}
    with torch.inference_mode():
        output=model(tokens,details=True,trace_energy=True)
        ctx=output.details['conditioning'];e0=model.energy(ctx.anchor,ctx);ef=model.energy(output.details['joint'],ctx)
    assert all(torch.equal(t,original[n]) for n,t in model.state_dict().items())
    frequency=model.frequencies().double();gram=frequency@frequency.T;eigen=torch.linalg.eigvalsh(gram)
    trace=output.details['energy_trace'];increases=(trace[...,1:]-trace[...,:-1]).clamp_min(0)
    update=output.details['update_mask'][None].expand_as(valid)&valid
    delta=(ef-e0)[valid];updated=(ef-e0)[update]
    skipped=(ef-e0)[valid&~update]
    result=dict(checkpoint=str(path),checkpoint_sha256=file_sha256(path),parameters=model.parameters_count(),
        frequency_rank=int(torch.linalg.matrix_rank(frequency)),frequency_stable_rank=float(eigen.sum()/eigen[-1]),
        frequency_eigen_min=float(eigen[0]),frequency_eigen_max=float(eigen[-1]),
        active_frequency_coordinates=int(model.mode_mask.count_nonzero()),
        largest_round_energy_increase=float(increases.max()) if increases.numel() else 0.,
        max_updated_energy_change=float(updated.max()) if updated.numel() else None,
        mean_updated_energy_change=float(updated.mean()) if updated.numel() else None,
        skipped_positions_above_anchor_fraction=float((skipped>1e-6).float().mean()) if skipped.numel() else None,
        anchor_logit_rms=float(ctx.anchor[...,model.blocks['output']][valid].square().mean().sqrt()),
        correction_logit_rms=float((output.logits-ctx.anchor[...,model.blocks['output']])[valid].square().mean().sqrt()),
        prompt_argmax_changed_fraction=float((output.logits.argmax(-1)!=ctx.anchor[...,model.blocks['output']].argmax(-1))[valid].float().mean()),
        total_positions=int(valid.sum()),refinement_positions=int(update.sum()),weights_unchanged=True,
        finite_logits=bool(torch.isfinite(output.logits).all()))
    if 'certified_step' in output.details:result.update(certified_step=float(output.details['certified_step']),curvature_upper=float(output.details['curvature_upper']))
    return result


def main():
    torch.set_num_threads(1)
    plan=json.loads((OUT/'plan.json').read_text())
    if not all((OUT/f'{c["name"]}.json').exists() for c in plan['cases']):raise RuntimeError('finish every fit before auditing selected models')
    tasks=[json.loads(l) for l in (OUT/'tasks_valid.jsonl').read_text().splitlines()]
    prompts=[r['prompt'] for r in tasks[::max(1,len(tasks)//18)]]
    results=[]
    for case in plan['cases']:
        record=json.loads((OUT/f'{case["name"]}.json').read_text());result=audit(Path(record['checkpoint']),prompts);result['case']=case;results.append(result)
    (OUT/'trained_geometry.json').write_text(json.dumps(dict(rows=results,validation_only=True),indent=2)+'\n')
    print(json.dumps(dict(checkpoints=len(results),all_finite=all(r['finite_logits'] for r in results),max_energy_increase=max(r['largest_round_energy_increase'] for r in results))))

if __name__=='__main__':main()
