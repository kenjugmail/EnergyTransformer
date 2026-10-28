"""Verify a recovery checkpoint changed only the intended parameter groups."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.prepare_rewt_lfm3 import sha,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ffn-layers',type=int,default=6)
    a=p.parse_args();torch.set_num_threads(4)
    before=torch.load(a.before,map_location='cpu',mmap=True,weights_only=True)
    after=torch.load(a.after,map_location='cpu',mmap=True,weights_only=True)
    assert before['config']==after['config'],'architecture changed'
    assert before['state_dict'].keys()==after['state_dict'].keys(),'state keys changed'
    layers=len(before['config']['donor_layers']);groups=defaultdict(lambda:dict(tensors=0,parameters=0,changed_tensors=0,delta_squared=0.,before_squared=0.))
    for name,old in before['state_dict'].items():
        new=after['state_dict'][name];assert old.shape==new.shape,name
        if name.startswith('energy.'):
            group='energy'
        elif '.mixer.' in name or '.operator_norm.' in name or '.post_operator_norm.' in name:
            group='mixers_and_operator_norms'
        elif name=='norm.weight':
            group='final_norm'
        elif name.startswith('blocks.') and int(name.split('.')[1])>=layers-a.ffn_layers:
            group='last_ffns_and_ffn_norms'
        else:
            group='frozen'
        changed=not torch.equal(old,new)
        if group=='frozen':
            assert not changed and old.dtype==new.dtype,f'frozen tensor changed: {name}'
        g=groups[group];g['tensors']+=1;g['parameters']+=old.numel();g['changed_tensors']+=int(changed)
        if group!='frozen':
            oldf=old.float();newf=new.float()
            assert torch.isfinite(newf).all(),f'nonfinite weights: {name}'
            g['delta_squared']+=float((newf-oldf).square().sum(dtype=torch.float64))
            g['before_squared']+=float(oldf.square().sum(dtype=torch.float64))
    for group,g in groups.items():
        g['relative_l2_change']=(g.pop('delta_squared')/max(g.pop('before_squared'),1e-30))**.5
    report=dict(before_sha256=sha(a.before),after_sha256=sha(a.after),architecture_unchanged=True,
        frozen_weights_exactly_unchanged=True,groups=dict(groups),after_metadata=after['metadata'])
    save_json(a.output,report);print(a.output)

if __name__=='__main__':main()
