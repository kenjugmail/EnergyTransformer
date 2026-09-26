from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import pytest
from energy_real.model import LM,Config
from energy_campaign.data import prepare
from energy_campaign.forecast_study import extract,joint_train
from energy_campaign.restart import run as restart


def test_joint_workflow_frozen_targets_and_resume(tmp_path):
    torch.manual_seed(4);c=Config(kind='hybrid',width=16,layers=3,heads=2,slots=4,ff_width=32,attention_window=8,chunk=8)
    model=LM(c);checkpoint=tmp_path/'model.pt';torch.save({'weights':model.state_dict(),'config':asdict(c)},checkpoint)
    rng=np.random.default_rng(32);records=[{'id':str(i),'text':' '.join(str(x) for x in rng.integers(0,999999,300))} for i in range(100)]
    prepare(records,tmp_path/'data',300000)
    forecast=tmp_path/'forecast.pt';data=extract(checkpoint,tmp_path/'data',forecast,'cpu',counts=(2,2,2),length=96)
    assert data['metadata']['summary_layer']==1
    job={'device':'cpu','out':str(tmp_path/'joint'),'checkpoint':str(checkpoint),'forecast_data':str(forecast),'data':str(tmp_path/'data'),
         'mode':'forecast','seed':71,'seconds':60,'steps':1}
    joint_train(job);saved=torch.load(tmp_path/'joint/last.pt',weights_only=True)
    assert saved['step']==1 and saved['history'][0]['auxiliary_backbone_gradient_norm']>0
    before={k:v.clone() for k,v in saved['weights'].items()};joint_train(job)
    after=torch.load(tmp_path/'joint/last.pt',weights_only=True)
    assert all(torch.equal(v,after['weights'][k]) for k,v in before.items())
    changed=torch.load(forecast,weights_only=True);changed['metadata']['scale']*=2;torch.save(changed,forecast)
    with pytest.raises(ValueError,match='resume identity'):joint_train(job)


def test_fresh_process_audit_worker_parity(tmp_path):
    restart(tmp_path,'whole','cpu');restart(tmp_path,'half','cpu');restart(tmp_path,'finish','cpu')
    a=torch.load(tmp_path/'whole.pt',weights_only=True);b=torch.load(tmp_path/'finish.pt',weights_only=True)
    assert a['samples']==b['samples'] and torch.equal(a['rng'],b['rng'])
    assert all(torch.equal(v,b['weights'][k]) for k,v in a['weights'].items())
