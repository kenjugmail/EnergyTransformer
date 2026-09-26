from pathlib import Path
from dataclasses import asdict
import json,os,subprocess,sys
import pytest
import torch
from energy_forecast.model import ForecastConfig
from energy_forecast.calibrated import AnchoredForecast
from energy_forecast.conditioning import GaussianMixture
from energy_forecast.conditional_training import train,objective_terms
from energy_forecast.experiment import digest


def setup_function():torch.set_num_threads(1)


def test_conditional_score_excludes_observed_coordinates():
    torch.manual_seed(17)
    m=AnchoredForecast(ForecastConfig(history=3,variables=2,width=8,components=2))
    x=torch.randn(5,3,2);y=torch.randn(5,8,dtype=torch.float64)
    f=GaussianMixture.from_landscape(m(x));nll,mse=objective_terms(f,y,2)
    assert nll.shape==mse.shape==(5,4)
    for stage,n in enumerate((0,2,4,6)):
        if n:
            marginal=f.marginal(list(range(n)))
            torch.testing.assert_close(nll[:,stage]*(8-n),f.energy(y)-marginal.energy(y[:,:n]))
    loss=.5*nll[:,0].mean()+.5*nll[:,1:].mean()+.25*mse[:,1:].mean()
    loss.backward()
    assert m.raw_correlation.grad is not None and torch.isfinite(m.raw_correlation.grad).all()
    assert float(m.output.weight.grad.norm())>0
    # Forecast coefficients are functions of x, not of the future targets.
    f2=m(x)
    assert torch.equal(f.means.float(),f2.means)


def make_data(tmp_path):
    torch.manual_seed(20)
    c=ForecastConfig(history=3,variables=2,width=8,components=2)
    m=AnchoredForecast(c)
    train_x=torch.randn(32,3,2);train_y=torch.randn(32,4,2)
    validation_x=torch.randn(16,3,2);validation_y=torch.randn(16,4,2)
    meta={'history':3,'variables':2,'horizons':[1,4,16,64],
          'center':torch.zeros(2),'scale':torch.ones(2),'teacher_sha256':'0'*64}
    data={'train':{'x':train_x,'y':train_y,'group':torch.arange(32)//4},
          'validation':{'x':validation_x,'y':validation_y,'group':torch.arange(16)//2},
          'test':{'x':torch.randn(4,3,2),'y':torch.randn(4,4,2),'group':torch.arange(4)},'metadata':meta}
    path=tmp_path/'data.pt';torch.save(data,path);parents=tmp_path/'parents';parents.mkdir()
    checkpoint={'format':'calibrated_trajectory_v2','variant':'anchored','config':asdict(c),'weights':m.state_dict(),
                'normalization':{k:meta[k] for k in ('center','scale')},'teacher_sha256':meta['teacher_sha256'],
                'identity':{'data_sha256':digest(path)},'selection':{},'summary_only':True}
    torch.save(checkpoint,parents/'anchored_seed11.pt')
    torch.save({'weights':m.anchor.state_dict(),'oof_mean':torch.zeros(32,8)},parents/'anchor.pt')
    return path,parents


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_full_optimizer_resume_in_fresh_process(tmp_path):
    data,parents=make_data(tmp_path)
    full=tmp_path/'full';split=tmp_path/'split'
    train(data,parents,full,'conditional_mse',11,8)
    train(data,parents,split,'conditional_mse',11,8,stop_at=3)
    cmd=[sys.executable,'-m','energy_forecast.conditional_training','--data',str(data),'--parents',str(parents),
         '--out',str(split),'--kind','conditional_mse','--seed','11','--steps','8','--resume']
    subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=os.environ.copy())
    a=torch.load(full/'last.pt',weights_only=True);b=torch.load(split/'last.pt',weights_only=True)
    for k in ('weights','optimizer','generator','rng','step','best','selected','best_weights','batch_sha256','logs'):equal(a[k],b[k])


def test_no_overwrite_or_changed_resume(tmp_path):
    data,parents=make_data(tmp_path);out=tmp_path/'run'
    train(data,parents,out,'joint',11,2)
    with pytest.raises(FileExistsError):train(data,parents,out,'joint',11,2)
    with pytest.raises(ValueError):train(data,parents,out,'joint',11,3,resume=True)


def test_training_streams_paired_across_objectives(tmp_path):
    data,parents=make_data(tmp_path)
    a=train(data,parents,tmp_path/'a','joint',11,3)
    b=train(data,parents,tmp_path/'b','conditional',11,3)
    assert a['batch_sha256']==b['batch_sha256']
