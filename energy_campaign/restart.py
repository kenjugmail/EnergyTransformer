"""Fresh-process deterministic AdamW/RNG/state restart audit."""
import argparse
from pathlib import Path
import torch
from energy_real.model import LM,Config
from energy_real.runtime import atomic_checkpoint


def run(path,mode,device):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(831);model=LM(Config(kind='hybrid',width=16,layers=3,heads=2,slots=4,ff_width=32,chunk=8,attention_window=8)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
    start=0;hashes=[]
    if mode=='finish':
        d=torch.load(path/'half.pt',map_location='cpu',weights_only=True);model.load_state_dict(d['weights']);optimizer.load_state_dict(d['optimizer'])
        torch.set_rng_state(d['rng']);start=d['step'];hashes=d['samples']
        if device=='cuda':torch.cuda.set_rng_state_all(d['cuda_rng'])
    for step in range(start,3 if mode=='half' else 6):
        tokens=torch.randint(256,(2,18),device=device);optimizer.zero_grad(set_to_none=True)
        import hashlib
        hashes.append(hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest())
        y,_=model(tokens[:,:-1]);loss=torch.nn.functional.cross_entropy(y.flatten(0,1),tokens[:,1:].flatten());loss.backward();optimizer.step()
    atomic_checkpoint({'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                       'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],'step':step+1,'samples':hashes},path/(mode+'.pt'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);p.add_argument('mode');p.add_argument('device');a=p.parse_args();run(a.path,a.mode,a.device)
