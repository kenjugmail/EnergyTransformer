#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd,json,time,subprocess,hashlib
from scripts.evaluate_recall import load,CONDITIONS,sliced
from energy_transformer.recall_data import EpisodeBatch,episodes
from energy_transformer.geometric_memory import MemoryState
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/v18'

@torch.inference_mode()
def main():
    torch.set_num_threads(1);records=[];audit=[]
    for seed in (11,22,33):
        model,c=load(ROOT/f'checkpoints/v18/projected_delayed_seed{seed}.pt')
        for cond in CONDITIONS:
            z=np.load(OUT/f'test_inputs/{cond}.npz');b=EpisodeBatch(**{k:torch.from_numpy(z[k]) for k in z.files});outputs=[]
            for start in range(0,192,24):
                a=sliced(b,start,start+24);o,_,_=model(a.keys,a.values,a.operations,a.queries);outputs.append(o)
            pred=torch.cat(outputs);mse=(pred-b.targets).square().mean(-1)
            ref=np.load(OUT/f'predictions/preconditioned_delta_delayed_seed{seed}_{cond}.npz')['output']
            audit.append({'seed':seed,'condition':cond,'max_output_difference':float(np.abs(pred.numpy()-ref).max()),'mse':float(mse.mean())})
            np.savez_compressed(OUT/f'predictions/projected_delayed_seed{seed}_{cond}.npz',output=pred.numpy(),mse=mse.numpy(),changed=b.changed.numpy())
            for ep in range(192):
                ch=b.changed[ep];records.append({'model':'projected','objective':'delayed','seed':seed,'condition':cond,'episode':ep,'mse':float(mse[ep].mean()),'changed_mse':float(mse[ep][ch].mean()) if ch.any() else np.nan,'untouched_mse':float(mse[ep][~ch].mean()),'fraction_rmse_below_point1':float((mse[ep]<.01).float().mean())})
        # Prefix state only to a separate interpreter, with entirely new future records.
        b=episodes(18800111,batch=2,nkeys=6,edits=8,gap=0);state=model.initial(2);cut=8
        for t in range(cut):state=model.step(b.keys[:,t],b.values[:,t],b.operations[:,t],state)
        case=OUT/f'restart_seed{seed}.pt'
        torch.save({'state':{'precision':state.precision.clone(),'information':state.information.clone()},'keys':b.keys[:,cut:].clone(),'values':b.values[:,cut:].clone(),'operations':b.operations[:,cut:].clone(),'queries':b.queries.clone()},case)
        for t in range(cut,b.keys.shape[1]):state=model.step(b.keys[:,t],b.values[:,t],b.operations[:,t],state)
        expected=model.read(b.queries,state)
        code=f"import torch;from scripts.evaluate_recall import load;from energy_transformer.geometric_memory import MemoryState;torch.set_num_threads(1);m,c=load({str(ROOT/f'checkpoints/v18/projected_delayed_seed{seed}.pt')!r});x=torch.load({str(case)!r},weights_only=True);s=MemoryState(**x['state']);\nwith torch.inference_mode():\n for t in range(x['keys'].shape[1]):s=m.step(x['keys'][:,t],x['values'][:,t],x['operations'][:,t],s)\n torch.save(m.read(x['queries'],s),{str(OUT/f'restored_seed{seed}.pt')!r})"
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
        actual=torch.load(OUT/f'restored_seed{seed}.pt',weights_only=True)
        assert torch.equal(actual,expected)
        audit.append({'seed':seed,'fresh_process_restart_max_error':float((actual-expected).abs().max()),'saved_state_bytes':(state.precision[0].numel()+state.information[0].numel())*4})
    (OUT/'projected_audit.json').write_text(json.dumps(audit,indent=2))
    df=pd.DataFrame(records);df.to_csv(OUT/'projected_raw.csv',index=False)
    joined=pd.concat([pd.read_csv(OUT/'raw.csv'),df],ignore_index=True);joined.to_csv(OUT/'all_raw.csv',index=False)
    summary=joined.groupby(['model','objective','condition'])[['mse','changed_mse','untouched_mse','fraction_rmse_below_point1']].mean().reset_index()
    summary.to_csv(OUT/'all_summary.csv',index=False)
    print(summary[summary.model=='projected'].to_string(index=False))
    print('Max conversion difference',max(x.get('max_output_difference',0) for x in audit))
if __name__=='__main__':main()
