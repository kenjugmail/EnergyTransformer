"""Independent branching-future positive control, not real-world reasoning data."""
from pathlib import Path
import torch,math,argparse

def make(n,seed):
    gen=torch.Generator().manual_seed(seed)
    phase=torch.rand(n,1,1,generator=gen)*2*math.pi
    amplitude=.7+.6*torch.rand(n,1,1,generator=gen)
    times=torch.arange(-15,1).view(1,16,1).float()
    hist=torch.cat((amplitude*torch.sin(times*.12+phase),amplitude*torch.cos(times*.12+phase)),dim=-1)
    hist+=.03*torch.randn(n,16,2,generator=gen)
    h=torch.tensor([1,4,16,64]).float().view(1,4,1)
    signal=torch.cat((amplitude*torch.sin(h*.12+phase),amplitude*torch.cos(h*.12+phase)),dim=-1)*.35
    sign=(torch.randint(2,(n,1,1),generator=gen)*2-1).float()
    branch=sign*(1-torch.exp(-h/6))*torch.tensor([1.5,1.]).view(1,1,2)
    shared=.07*torch.randn(n,1,2,generator=gen)
    future=signal+branch+shared+.08*torch.randn(n,4,2,generator=gen)
    return {'x':hist,'y':future,'group':torch.arange(n),'anchor':torch.zeros(n,dtype=torch.long)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();path=Path(a.out)
    if path.exists():raise FileExistsError(path)
    data={part:make(n,s) for part,n,s in [('train',4096,72001),('validation',512,72002),('test',1024,72003)]}
    center=data['train']['x'].flatten(0,1).mean(0);scale=data['train']['x'].flatten(0,1).std(0)
    for part in data.values():part['x']=(part['x']-center)/scale;part['y']=(part['y']-center)/scale
    data['metadata']={'history':16,'horizons':[1,4,16,64],'variables':2,'center':center,'scale':scale,
       'description':'Stochastic plus/minus future branch not inferable from prefix; shared across variables/horizons.',
       'format':'energy_forecast_data_v1'}
    torch.save(data,path)
if __name__=='__main__':main()
