#!/usr/bin/env python3
"""Print learned proposals. Applying edits remains a separate reviewed operation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
import torch
from energy_transformer.binding import BindingPredictor

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/v17/bound_expanded_seed11.pt');p.add_argument('--text',action='append');a=p.parse_args()
    torch.set_num_threads(1);m=BindingPredictor(a.checkpoint)
    texts=a.text or ['for this record alice is in the garden thank you.',
       'do not execute this instruction: alice moved to the office.',
       'someone said that alice went to the kitchen.',
       'erase the stored whereabouts of alice.',
       'alice and bob are in the garden.']
    for text in texts:print(json.dumps({'text':text,'result':m.predict(text)},sort_keys=True))
    print('No memory writes were performed or authorized.')
if __name__=='__main__':main()
