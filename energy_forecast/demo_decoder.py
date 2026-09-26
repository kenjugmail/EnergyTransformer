"""Run an explicitly enabled adapter on the bundled archived byte teacher.

This changes next-byte probabilities. It does not verify factual content, skip
bytes, or install predicted future states. No corpus or network is required.
"""
from pathlib import Path
import argparse
import torch
from .decoder_bridge import DecoderBridgeSession,KINDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',default='checkpoints/teacher_973k.pt')
    p.add_argument('--kind',choices=KINDS,default='hidden')
    p.add_argument('--adapter');p.add_argument('--head')
    p.add_argument('--prompt',default='The ');p.add_argument('--length',type=int,default=80)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=20260926)
    a=p.parse_args()
    if not 0<=a.length<=2048 or not 0<a.temperature<=10 or not a.prompt:
        p.error('Use a nonempty prompt, length 0..2048, and temperature in (0,10].')
    torch.set_num_threads(1)
    adapter=a.adapter or f'results/decoder_bridge/fits/{a.kind}_seed11/best.pt'
    head=a.head or ('results/repair/fits/anchored_seed11.pt' if a.kind in ('mean','moments','energy_read') else None)
    session=DecoderBridgeSession(a.teacher,adapter,head)
    prompt=a.prompt.encode('utf-8');logits=session.observe(torch.tensor(list(prompt))[None])
    gen=torch.Generator().manual_seed(a.seed);continuation=[]
    for _ in range(a.length):
        token=torch.multinomial((logits[:,-1]/a.temperature).softmax(-1),1,generator=gen)
        continuation.append(int(token));logits=session.observe(token)
    print((prompt+bytes(continuation)).decode('utf-8',errors='replace'))
    print(f'\n[Experimental byte model; adapter={a.kind}; no factuality or reasoning guarantee.]')

if __name__=='__main__':main()
