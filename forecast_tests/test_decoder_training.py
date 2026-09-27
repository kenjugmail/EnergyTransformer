from pathlib import Path
import json,os,subprocess,sys
import pytest
import torch
from energy_forecast.train_decoder import train_one

def fixture():
    g=torch.Generator().manual_seed(6001)
    def part(n):return {'h':torch.randn(n,16,generator=g),'x':torch.randn(n,16,8,generator=g),'y':torch.randint(0,8,(n,),generator=g)}
    return {'train':part(30),'validation':part(12),'weight':torch.randn(8,16,generator=g),
            'metadata':{'width':16,'history':16,'variables':8,'horizons':[1,4,16,64],
                        'teacher_sha256':'fixture-not-a-trained-model','normalization':{'center':torch.zeros(8),'scale':torch.ones(8)}}}

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def test_fresh_process_training_resume(tmp_path):
    data=fixture();cache={'train':None,'validation':None};torch.set_num_threads(1)
    train_one(data,'fixture',cache,None,tmp_path/'full','hidden',steps=8,batch=6)
    train_one(data,'fixture',cache,None,tmp_path/'resume','hidden',steps=8,batch=6,stop_after=3)
    torch.save(data,tmp_path/'data.pt')
    code=f"import torch;from energy_forecast.train_decoder import train_one;d=torch.load({str(tmp_path/'data.pt')!r},weights_only=True);train_one(d,'fixture',{{'train':None,'validation':None}},None,{str(tmp_path/'resume')!r},'hidden',steps=8,batch=6,resume=True)"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    a=torch.load(tmp_path/'full/last.pt',weights_only=True);b=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    for key in ('weights','optimizer','generator','rng','step','best','best_step','history','sample_hash','best_payload'):
        assert equal(a[key],b[key]),key

def test_resume_changed_schedule_rejected(tmp_path):
    data=fixture();cache={'train':None,'validation':None}
    train_one(data,'fixture',cache,None,tmp_path/'run','hidden',steps=8,batch=6,stop_after=3)
    with pytest.raises(ValueError):train_one(data,'fixture',cache,None,tmp_path/'run','hidden',steps=9,batch=6,resume=True)

def test_overwrite_rejected(tmp_path):
    data=fixture();cache={'train':None,'validation':None};out=tmp_path/'run';out.mkdir()
    with pytest.raises(FileExistsError):train_one(data,'fixture',cache,None,out,'hidden',steps=1,batch=6)

def test_headless_cli_training_and_evaluation(tmp_path):
    data=fixture();g=torch.Generator().manual_seed(101)
    data['test']={'h':torch.randn(20,16,generator=g),'x':torch.randn(20,16,8,generator=g),'y':torch.randint(0,8,(20,),generator=g),
                  'group':torch.arange(20)//10,'anchor':torch.arange(20)}
    inp=tmp_path/'input.pt';torch.save(data,inp)
    subprocess.run([sys.executable,'-m','energy_forecast.train_decoder','--data',str(inp),'--out',str(tmp_path/'fits'),
                    '--kinds','hidden','history','--steps','2','--batch','6'],check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,'-m','energy_forecast.evaluate_decoder','--data',str(inp),'--fits',str(tmp_path/'fits'),
                    '--out',str(tmp_path/'evaluation')],check=True,capture_output=True,text=True)
    assert (tmp_path/'evaluation/summary.csv').exists()
