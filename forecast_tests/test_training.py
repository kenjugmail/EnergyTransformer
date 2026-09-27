from pathlib import Path
import torch
from energy_forecast.experiment import train

def test_exact_resume_for_optimizer_trajectory(tmp_path):
    gen=torch.Generator().manual_seed(119)
    data={part:{'x':torch.randn(24,4,2,generator=gen),'y':torch.randn(24,2,2,generator=gen)} for part in ['train','validation']}
    data['metadata']={'history':4,'variables':2,'horizons':[1,4],'center':torch.zeros(2),'scale':torch.ones(2),'teacher_sha256':'fixture'}
    path=tmp_path/'data.pt';torch.save(data,path)
    train(path,tmp_path/'full','mixture',11,steps=8)
    train(path,tmp_path/'paused','mixture',11,steps=8,stop_at=4)
    train(path,tmp_path/'paused','mixture',11,steps=8,resume=True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=True);b=torch.load(tmp_path/'paused/last.pt',weights_only=True)
    assert a['batch_sha256']==b['batch_sha256'] and a['step']==b['step']==8
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for k,items in a['optimizer']['state'].items():
        for name,v in items.items():assert torch.equal(v,b['optimizer']['state'][k][name])
