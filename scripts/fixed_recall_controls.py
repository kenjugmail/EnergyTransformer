#!/usr/bin/env python3
"""Post-hoc stronger no-training controls; do not claim all gains require learning."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,json
from scripts.evaluate_recall import CONDITIONS
from energy_transformer.recall_data import episodes
from energy_transformer.geometric_memory import EditMemory
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/v18'

@torch.inference_mode()
def main():
    torch.set_num_threads(1);records=[]
    for condition in CONDITIONS:
        b=np.load(OUT/f'test_inputs/{condition}.npz');keys=torch.from_numpy(b['keys']);values=torch.from_numpy(b['values']);op=torch.from_numpy(b['operations']);q=torch.from_numpy(b['queries']);target=torch.from_numpy(b['targets'])
        p=.01*torch.eye(12).expand(192,-1,-1).clone();m=torch.zeros(192,12,4)
        for t in range(keys.shape[1]):
            active=(op[:,t]!=2).float();k=keys[:,t];v=values[:,t]*(op[:,t]==0).float()[:,None]
            p=p+active[:,None,None]*k[:,:,None]*k[:,None,:]
            d=torch.linalg.solve(p,k[:,:,None]).squeeze(-1);d=d/(d*k).sum(-1,keepdim=True)
            residual=v-(k[:,None,:]@m).squeeze(1)
            m=m+active[:,None,None]*d[:,:,None]*residual[:,None,:]
        mse=((q@m-target)**2).mean((-1,-2))
        records.extend({'model':'fixed_preconditioned','condition':condition,'episode':i,'mse':float(e)} for i,e in enumerate(mse))
    pd.DataFrame(records).to_csv(OUT/'fixed_control.csv',index=False)
    print(pd.DataFrame(records).groupby('condition').mse.mean().to_string())
    # Raw serial same-prefix hold audit for the representative converted checkpoint.
    from scripts.evaluate_recall import load
    model,_=load(ROOT/'checkpoints/v18/projected_delayed_seed11.pt')
    a=episodes(188201,batch=4,gap=0);_,state,_=model(a.keys,a.values,a.operations,a.queries)
    before=(state.precision.clone(),state.information.clone());g=torch.Generator().manual_seed(188202)
    for _ in range(4096):
        state=model.step(torch.randn(4,12,generator=g),torch.randn(4,4,generator=g),torch.full((4,),2),state)
    assert torch.equal(state.precision,before[0]) and torch.equal(state.information,before[1])
    (OUT/'hold_audit.json').write_text(json.dumps({'same_prefix_batch':4,'extra_hold_steps':4096,'max_precision_difference':0.,'max_information_difference':0.,'boundary':'HOLD flag is supplied to every method; not learned long-context language behavior.'},indent=2))
if __name__=='__main__':main()
