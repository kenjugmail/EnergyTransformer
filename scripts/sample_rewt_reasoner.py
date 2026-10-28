"""Run the opt-in learned rule head on a raw prompt, with fixed weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rewt.reasoning import load_reasoner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--energy-rounds',type=int)
    p.add_argument('--fixed-query',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    model,_=load_reasoner(args.checkpoint)
    before={k:v.clone() for k,v in model.state_dict().items()}
    kwargs={}
    if args.energy_rounds is not None:kwargs['energy_rounds']=args.energy_rounds
    if args.fixed_query:kwargs['reread']=False
    prediction=model.predict([args.prompt],**kwargs)[0]
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    print(json.dumps(dict(prompt=args.prompt,**prediction,inference_weights_unchanged=True,
          interface='Structured rule head: learned byte extraction, read count, intermediate nodes and final node; not free-form language generation.',
          checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
