#!/usr/bin/env python3
"""Same-session CPU component timing. NOT language-model token latency."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time,random,json,statistics
import torch
from scripts.evaluate_recall import load
from energy_transformer.recall_data import episodes
ROOT=Path(__file__).resolve().parents[1]

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    kinds=['diagonal','euclidean','precision','preconditioned_delta','projected','delta','softmax']
    models={k:load(ROOT/f'checkpoints/v18/{k}_delayed_seed11.pt')[0] for k in kinds}
    b=episodes(18981000,batch=16)
    def execute(m):return m(b.keys,b.values,b.operations,b.queries)
    for m in models.values():
        for _ in range(3):execute(m)
    rng=random.Random(18982000);records=[]
    for rep in range(15):
        order=kinds.copy();rng.shuffle(order)
        for k in order:
            t=time.perf_counter();_,s,_=execute(models[k]);elapsed=1000*(time.perf_counter()-t)
            records.append({'model':k,'repeat':rep,'batch16_ms':elapsed,'returned_array_bytes_per_episode':(s.precision[0].numel()+s.information[0].numel())*4})
    summaries=[{'model':k,'median_batch16_ms':statistics.median(r['batch16_ms'] for r in records if r['model']==k),'raw_state_bytes':next(r['returned_array_bytes_per_episode'] for r in records if r['model']==k)} for k in kinds]
    (ROOT/'results/v18/timing.json').write_text(json.dumps({'summary':summaries,'records':records,'threads':1,'scope':'16 independent short episodes per batch, including writes and final read. Not next-token LM or optimized GPU results. Softmax returns full prefix arrays including holds; no streaming cache is implemented.'},indent=2))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
