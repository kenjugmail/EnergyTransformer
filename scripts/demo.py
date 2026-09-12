#!/usr/bin/env python3
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from energy_transformer.bridge import EnergyTextSession
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[1]/'checkpoints/v14/bridge_seed11.pt');a=p.parse_args()
torch.set_num_threads(1);session=EnergyTextSession(a.checkpoint)
# These controlled demo sentences are intentionally supplied as trusted input.
for sentence in ['Alice is in the garden.','Where is Alice?',
 'Someone said Alice is in the kitchen.','Where is Alice?',
 'Correction Alice is now in the office.','Where is Alice?',
 'Forget the location of Alice.','Where is Alice?']:
 print(sentence);print(session.handle(sentence,trusted=True))
