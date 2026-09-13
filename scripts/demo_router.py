#!/usr/bin/env python3
"""Learned token-only demonstration; grammar is displayed, not parsed by the model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,torch
from energy_transformer.write_router import WriteRouterLM,RouterConfig
from energy_transformer.scaling_data import BOS,SET,DELETE,NOISE,ASK,SEP,KEY_START,VALUE_START,UNKNOWN
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'results/v20/hard_aux_seed11/best.pt'));a=p.parse_args()
    torch.set_num_threads(1);d=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m=WriteRouterLM(RouterConfig(**d['config']));m.load_state_dict(d['model']);m.eval()
    seq=[BOS,SET,KEY_START+3,VALUE_START+7,SEP,SET,KEY_START+9,VALUE_START+2,SEP]
    for _ in range(64):seq +=[NOISE,KEY_START+3,VALUE_START+5,SEP]
    seq += [ASK,KEY_START+3]
    with torch.inference_mode():out,s,_=m(torch.tensor([seq]));pred=int(out[0,-1].argmax())
    print('Store key3=value7, key9=value2; then 64 conflicting NOISE records.')
    print('ASK key3 ->',f'value{pred-VALUE_START}' if VALUE_START<=pred<VALUE_START+16 else 'UNKNOWN' if pred==UNKNOWN else f'token{pred}')
    print('State tensors:',[tuple(t.shape) for t in s],'raw bytes:',sum(t.numel()*t.element_size() for t in s))
    print('This is a bounded symbolic-memory model, not an English assistant.')
if __name__=='__main__':main()
