#!/usr/bin/env python3
"""Re-express the learned classical covariance-delta model as energy coefficients.
No fitting; exact real-arithmetic equivalence, not bitwise float32 equivalence.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]

def main():
    records=[]
    for seed in (11,22,33):
        src=ROOT/f'checkpoints/v18/preconditioned_delta_delayed_seed{seed}.pt'
        c=torch.load(src,map_location='cpu',weights_only=True);c['kind']='projected'
        c['origin']='Algebraic conversion of trained preconditioned_delta. No new optimization.'
        dst=ROOT/f'checkpoints/v18/projected_delayed_seed{seed}.pt';torch.save(c,dst)
        records.append({'seed':seed,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'converted_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
    (ROOT/'results/v18/conversion.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
