"""Validation-only energy signal decomposition of frozen selected checkpoints."""
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
import torch
from torch.nn import functional as F
from rewt.data import Corpus,file_sha256
from rewt.evaluate import model_output
from rewt.experiment import load_model
from rewt.model import Conditioning


@torch.inference_mode()
def probe(model,corpus,offsets,seed=1835):
    batch=corpus.batch("valid",None,len(offsets),128,starts=offsets)
    out=model_output(model,batch.tokens,torch.Generator().manual_seed(seed),details=True)
    source=out.details['conditioning']
    cond=Conditioning(source.anchor.flatten(0,1),source.amplitudes.flatten(0,1),
        source.routing.flatten(0,1) if source.routing is not None else None,
        source.features.flatten(0,1),{},source.frequency,source.precision)
    initial=model.canonical_state(out.details['initial_joint'].flatten(0,1))
    cut=model.blocks['output'].start
    projected=F.linear(initial,cond.frequency,model.phase)
    derivative=-projected.sin() if model.config.activation=='fourier' else 1-projected.tanh().square()
    wave=model.canonical_state((cond.amplitudes*derivative)@cond.frequency)
    total=model.gradient(initial,cond)
    quadratic=total-wave
    ratio=wave[:,cut:].norm(dim=-1)/quadratic[:,cut:].norm(dim=-1).clamp_min(1e-12)
    target=batch.targets.flatten()
    base=out.details['joint'].flatten(0,1)
    wrong_scores=out.logits.flatten(0,1).clone()
    wrong_scores.scatter_(-1,target[:,None],-torch.inf)
    hard=wrong_scores.argmax(-1)
    random=(target+17)%256
    def candidate(label):
        return torch.cat((base[:,:cut],2*F.one_hot(label,256).to(base.dtype)),dim=-1)
    positive=model.energy(candidate(target),cond)
    random_energy=model.energy(candidate(random),cond)
    hard_energy=model.energy(candidate(hard),cond)
    incorrect=out.logits.flatten(0,1).argmax(-1)!=target
    return dict(positions=int(target.numel()),
        mean_output_wave_to_quadratic_gradient_norm_ratio=float(ratio.mean()),
        energy_ranks_true_below_fixed_wrong_fraction=float((positive<random_energy).float().mean()),
        energy_ranks_true_below_strongest_wrong_fraction=float((positive<hard_energy).float().mean()),
        when_prediction_wrong_count=int(incorrect.sum()),
        when_prediction_wrong_hard_ranking_fraction=float((positive[incorrect]<hard_energy[incorrect]).float().mean()),
        predicted_byte_accuracy=float((~incorrect).float().mean()))


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan=json.loads((ROOT/'selected_protocol.json').read_text())
    corpus=Corpus(plan['corpus']['path'])
    records=[]
    for seed in plan['settings']['seeds']:
        for arm in ('legacy_fourier','neural_energy','fourier_rewt','full_rewt'):
            path=Path(plan['runs'])/f'seed_{seed}'/arm/'language'/'best.pt'
            before=file_sha256(path)
            model,_=load_model(path)
            record=dict(arm=arm,seed=seed,checkpoint_sha256=before,
                        **probe(model,corpus,plan['validation_offsets'][:4]))
            assert before==file_sha256(path)
            records.append(record)
    groups=defaultdict(list)
    for r in records: groups[r['arm']].append(r)
    fields=('mean_output_wave_to_quadratic_gradient_norm_ratio','energy_ranks_true_below_fixed_wrong_fraction',
            'energy_ranks_true_below_strongest_wrong_fraction','when_prediction_wrong_hard_ranking_fraction')
    aggregate={arm:{f:sum(r[f] for r in rows)/len(rows) for f in fields} for arm,rows in groups.items()}
    value=dict(scope='Frozen validation prefixes; one-hot candidate ranking is a training-objective diagnostic, not a generative accuracy metric',
               records=records,mean_across_seeds=aggregate)
    (ROOT/'energy_signal.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
