#!/usr/bin/env python3
"""Run the continuous-valued energy-memory example; no language parsing or writes to user facts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,torch
from scripts.evaluate_recall import load
from energy_transformer.recall_data import episodes

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/v18/projected_delayed_seed11.pt');a=p.parse_args()
    torch.set_num_threads(1);m,_=load(Path(a.checkpoint));b=episodes(18991001,batch=1)
    out,state,_=m(b.keys,b.values,b.operations,b.queries)
    print('Current target values:',b.targets[0].tolist())
    print('Retrieved values:',out[0].tolist())
    print('MSE:',float((out-b.targets).square().mean()))
    print('Raw state bytes:',4*(state.precision[0].numel()+state.information[0].numel()))
    print('Structured numerical recall demonstration, not a chat assistant.')
if __name__=='__main__':main()
