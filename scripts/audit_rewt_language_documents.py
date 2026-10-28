"""Paired document uncertainty for the already-frozen language test models.

This descriptive audit does not select models or tune hyperparameters. Each
bootstrap draw keeps all three trained seeds and the same documents across arms.
"""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from rewt.architecture import load_architecture_checkpoint
from rewt.bpe import StoryBPE
from rewt.story_data import StoryCorpus
from rewt.experiment import save_json
OUT=ROOT/'reports/rewt_architecture_20261022/language'


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    plan=json.loads((OUT/'plan.json').read_text());results=json.loads((OUT/'test_results.json').read_text())
    corpus=StoryCorpus(ROOT/'data/rewt_architecture_20261022')
    tokenizer=StoryBPE.load(ROOT/'data/tinystories_bpe_20261010/vocabulary.json')
    lengths=torch.tensor(tokenizer.byte_lengths);records=[];arrays={};bytes_per_doc=None
    for result in results:
        model,_=load_architecture_checkpoint(result['checkpoint']);model.eval().prepare_inference()
        nll=[];byte_counts=[]
        for first in range(0,len(plan['test_offsets']),16):
            starts=plan['test_offsets'][first:first+16]
            batch=corpus.batch('test',None,len(starts),128,starts=starts)
            logits=model(batch.tokens).logits
            loss=F.cross_entropy(logits.flatten(0,1),batch.targets.flatten(),reduction='none').reshape(len(starts),128)
            nll.extend(loss.double().sum(-1).tolist());byte_counts.extend(lengths[batch.targets].sum(-1).tolist())
        nll=np.array(nll);byte_counts=np.array(byte_counts)
        measured=nll.sum()/byte_counts.sum()/math.log(2)
        assert abs(measured-result['test']['bpb'])<1e-6
        if bytes_per_doc is None:bytes_per_doc=byte_counts
        assert np.array_equal(bytes_per_doc,byte_counts)
        case=result['case'];arrays[case['variant'],case['seed']]=nll
        records.append(dict(case=case,nll_per_document=nll.tolist(),original_bytes_per_document=byte_counts.tolist(),bpb=measured))
    seeds=sorted({r['case']['seed'] for r in results});names=sorted({r['case']['variant'] for r in results})
    draws=np.random.default_rng(22219).integers(0,len(bytes_per_doc),(5000,len(bytes_per_doc)))
    intervals={}
    for name in names:
        differences=np.stack([arrays[name,s]-arrays['control',s] for s in seeds])
        mean=differences.mean(0)
        samples=mean[draws].sum(-1)/bytes_per_doc[draws].sum(-1)/math.log(2)
        intervals[name]=dict(delta_bpb=float(mean.sum()/bytes_per_doc.sum()/math.log(2)),
            ci95=np.quantile(samples,[.025,.975]).tolist(),
            seed_deltas=(differences.sum(-1)/bytes_per_doc.sum()/math.log(2)).tolist())
    save_json(OUT/'document_audit.json',dict(rows=records,test_offsets=plan['test_offsets'],paired_deltas=intervals,
        method='5000 paired document bootstraps; identical documents across all arms and three fixed seeds. Ratio of total NLL to original bytes. Conditional on these trained seeds; no training-seed population CI or multiple-comparison correction. Descriptive audit after test, no reselection.'))
    print(json.dumps(intervals,indent=2))

if __name__=='__main__':main()
