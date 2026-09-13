"""Sample the actual v21 byte checkpoint. No corpus or online model download."""
import argparse
from pathlib import Path
import torch
from .common import configure,atomic_checkpoint
from .accept import load_native

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--prompt',default='The ');p.add_argument('--length',type=int,default=160);p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=1234)
    a=p.parse_args()
    if not 0<a.length<=2048 or not .05<=a.temperature<=2 or not a.prompt:raise ValueError('Prompt, temperature or output budget invalid')
    d=configure(a.device,seed=a.seed);m=load_native(d).eval();raw=list(a.prompt.encode('utf-8'))
    if len(raw)>4096:raise ValueError('Prompt exceeds demonstration budget')
    with torch.inference_mode():
        logits,state=m(torch.tensor([raw],dtype=torch.long,device=d))
        for _ in range(a.length):
            token=torch.multinomial(torch.softmax(logits[:,-1].float()/a.temperature,-1),1)
            raw.append(int(token));logits,state=m(token,state)
    print(bytes(raw).decode('utf-8',errors='replace'))
