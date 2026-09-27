from pathlib import Path
import json,hashlib
from energy_forecast.conditional_training import train,KINDS,source_hash
from energy_forecast.experiment import digest
ROOT=Path(__file__).resolve().parent
out=ROOT/'results/conditional'
# A fresh reproduction owns a new result directory; never overwrite shipped runs.
out.mkdir(parents=True,exist_ok=False)
protocol={'date':'2026-09-26','base_commit':'3f6384a69454b1c4a0b90441b936a173596b881d',
 'question':'Does explicitly scoring matured-prefix conditionals improve forecast updates?',
 'arms':list(KINDS),'seeds':[11,22,33],'updates':350,'batch':64,'initialization':'Same seed-matched anchored/calibrated parent; anchor and calibration buffers frozen',
 'primary_selection':'.5 joint NLL/coordinate + .5 mean conditional NLL/coordinate on original 15 selection documents, step zero eligible',
 'training_mean':'Inherited document-crossfit mean for TRAIN rows; fitted mean for inference',
 'conditioning':'Fixed-origin targets; reveal +1/+4/+16 only after arrival; no new inference information',
 'controls':['joint-NLL warm continuation','unchanged parent raw and guarded','classical correlated Gaussian'],
 'test':'634 previously inspected origins; development, not pristine confirmation',
 'secondary':'Gensim separate lee.cor with exact/64-byte train overlap audit; extracted after selection frozen; historical language-task use disclosed',
 'deployment':'Do not change the baseline language logits or automatically enable a new gate',
 'gpu':'not used','source_sha256':source_hash()}
if (out/'PROTOCOL.json').exists():raise FileExistsError('Do not overwrite the protocol')
(out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
records=[]
for seed in (11,22,33):
 for kind in KINDS:
  records.append(train(ROOT/'results/state_data.pt',ROOT/'results/repair/fits',out/f'{kind}_seed{seed}',kind,seed,350))
means={k:sum(r['selection_score'] for r in records if r['kind']==k)/3 for k in KINDS}
choice=min(means,key=means.get)
tied=[k for k in KINDS if means[k]==means[choice]]
frozen={'status':'frozen_before_test','best_validation_family':choice,'validation_scores':means,'tied_validation_families':tied,'source_sha256':source_hash(),
        'heads':{str(p.relative_to(ROOT)):digest(p) for p in sorted(out.glob('*/best.pt'))},
        'note':'No test fitting, calibration, or per-horizon gate selection this pass.'}
(out/'FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
print('FROZEN',json.dumps(frozen),flush=True)
