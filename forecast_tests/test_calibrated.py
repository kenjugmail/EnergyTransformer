import inspect,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from energy_forecast.model import ForecastConfig,ForecastEnergy,EnergyLandscape
from energy_forecast.calibrated import HorizonAnchor,AnchoredForecast,affine_landscape,load_calibrated
from energy_forecast.repair import split_validation,fit_anchor
from energy_forecast.repair_evaluate import marginal_quantile
from energy_forecast.experiment import load_head

ROOT=Path(__file__).resolve().parents[1]
FIT=ROOT/'results/repair/fits/anchored_seed11.pt'


def setup_function():torch.set_num_threads(1)


def fixture(dtype=torch.float64):
    torch.manual_seed(19)
    m=ForecastEnergy(ForecastConfig(history=3,variables=2,horizons=(1,4),width=8,components=3)).to(dtype)
    return m(torch.randn(2,3,2,dtype=dtype))


def test_affine_normalized_energy_jacobian():
    f=fixture();new_mean=torch.randn(2,4,dtype=torch.float64);t=torch.tensor([.6,1.2,1.8,.9],dtype=torch.float64)
    new=affine_landscape(f,new_mean,t);y=torch.randn_like(new_mean)
    torch.testing.assert_close(new.energy(y),f.energy(f.mean+(y-new_mean)/t)+t.log().sum())
    torch.testing.assert_close(new.mean,new_mean)


def test_candidate_and_temperature_gradients():
    f=fixture().detached();mean=torch.randn(2,4,dtype=torch.float64)
    t=torch.ones(4,dtype=torch.float64,requires_grad=True);y=torch.randn_like(mean,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:affine_landscape(f,mean,b).energy(a),(y,t))


def test_joint_sampling_affine_identity():
    f=fixture().detached();mean=torch.randn_like(f.mean);t=torch.tensor([.5,1.,1.3,1.8],dtype=torch.float64)
    a=f.sample(300,generator=torch.Generator().manual_seed(20))
    b=affine_landscape(f,mean,t).sample(300,generator=torch.Generator().manual_seed(20))
    torch.testing.assert_close(b,mean[:,None]+(a-f.mean[:,None])*t)


def test_two_basins_preserved_in_control():
    f=EnergyLandscape(torch.zeros(1,2),torch.tensor([[[-2.],[2.]]]),torch.full((1,2,1),.2),torch.ones(2,1,1))
    changed=affine_landscape(f,torch.tensor([[3.]]),torch.tensor([1.4]))
    assert changed.energy(torch.tensor([[3.]])).item()>changed.energy(changed.means[:,0]).item()+10


def test_mean_cannot_drift_when_shape_weights_change():
    c=ForecastConfig(history=4,variables=2,horizons=(1,2),width=8)
    model=AnchoredForecast(c);model.anchor.gain.fill_(.75);model.anchor.weight.normal_(std=.1)
    x=torch.randn(5,4,2);a=model.anchor(x)
    torch.testing.assert_close(model(x).mean,a,atol=1e-6,rtol=1e-6)
    with torch.no_grad():model.output.weight.normal_();model.temperature.copy_(torch.tensor([.5,2.]))
    torch.testing.assert_close(model(x).mean,a,atol=2e-6,rtol=2e-6)
    assert list(inspect.signature(model.forward).parameters)==['history']


def test_zero_gain_has_no_context_dependence_in_mean():
    a=HorizonAnchor(3,2,(1,64));a.weight.normal_();a.constant.normal_();a.gain.copy_(torch.tensor([1.,0.]))
    p=a(torch.randn(4,3,2));torch.testing.assert_close(p[:,2:],a.constant[2:].expand(4,-1))


@pytest.mark.parametrize('bad',[-1.,0.,float('nan'),float('inf')])
def test_bad_temperatures(bad):
    f=fixture();t=torch.ones(4,dtype=torch.float64);t[0]=bad
    with pytest.raises(ValueError):affine_landscape(f,f.mean,t)


def test_document_calibration_split():
    g=torch.tensor([3]*5+[4]*2+[8]*8+[9]*3+[10]*10+[11]*2)
    a,b=split_validation(g)
    assert not set(g[a].tolist())&set(g[b].tolist())
    assert torch.equal(a,split_validation(g)[0]) and (a|b).all()


def test_exact_gaussian_marginal_intervals():
    f=EnergyLandscape(torch.zeros(3,1),torch.ones(3,1,2),torch.tensor([[[.5,2.]]]).expand(3,-1,-1),torch.eye(2)[None])
    q=marginal_quantile(f,.95);expected=1+1.6448536269514722*torch.tensor([.5,2.])
    torch.testing.assert_close(q,expected.expand_as(q),atol=1e-5,rtol=1e-5)


def test_crossfit_only_train_and_selection_inputs():
    # A small fixture verifies the mean-fitting API has no final-test argument.
    gen=torch.Generator().manual_seed(71)
    def make(n,offset):
        x=torch.randn(n,2,1,generator=gen)
        y=torch.stack((.8*x[:,-1],.4*x[:,0]),dim=1)+.02*torch.randn(n,2,1,generator=gen)
        return {'x':x,'y':y,'group':torch.arange(n)//4+offset}
    train,valid=make(40,0),make(20,20)
    model,oof,details,raw=fit_anchor(train,valid,{'history':2,'variables':1,'horizons':[1,4]})
    assert torch.isfinite(oof).all() and oof.shape==(40,2)
    assert len(set(details['fold_by_group'].values()))==5
    assert model(valid['x']).shape==(20,2)


def require_artifacts():
    if not FIT.exists():pytest.skip('Standalone trained repair artifacts required')


def test_frozen_repair_and_parent_batch_stream_match():
    require_artifacts();frozen=json.loads((FIT.parent/'FROZEN.json').read_text())
    from energy_forecast.experiment import digest
    for n,s in frozen['files'].items():assert digest(FIT.parent/n)==s
    rows=json.loads((FIT.parent/'training.json').read_text())
    for r in rows:
        if r['variant']=='anchored':
            parent=json.loads((ROOT/f"results/native_runs/spectral_seed{r['seed']}/training.json").read_text())
            assert parent['batch_sha256']==r['batch_sha256']


def test_trained_distribution_and_failed_horizon():
    require_artifacts();m,info=load_head(FIT)
    data=torch.load(ROOT/'results/state_data.pt',weights_only=True)
    x=data['test']['x'][:4];f=m(x)
    torch.testing.assert_close(f.mean,m.anchor(x))
    assert m.anchor.gain[-1]==0
    assert not info['selection']['horizons'][-1]['use_context']


def test_snapshot_schema_and_teacher_unchanged(tmp_path):
    require_artifacts()
    sys.path.insert(0,str(ROOT/'reference'))
    from energy_forecast.session import ForecastSession
    from energy_forecast.probe import load_energy_model
    teacher=ROOT/'checkpoints/teacher_973k.pt'
    session=ForecastSession(teacher,FIT)
    tokens=torch.tensor([list(b'The stored energy predicts an uncertain future.')],dtype=torch.long)
    actual=session.observe(tokens)
    model=load_energy_model(teacher);state=None;logits=[]
    with torch.no_grad():
        for i in range(tokens.shape[1]):
            out,state=model(tokens[:,i:i+1],state);logits.append(out)
    assert torch.equal(actual,torch.cat(logits,1))
    before=[(p.clone(),s.clone()) for p,s in session.state]
    fun=session.forecast();fun.refine(fun.mean,steps=2)
    assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(before,session.state))
    snapshot=tmp_path/'session.pt';session.save(snapshot);saved=torch.load(snapshot,weights_only=True)
    saved['steps']=0;torch.save(saved,tmp_path/'bad.pt')
    with pytest.raises(ValueError):session.restore(tmp_path/'bad.pt')
    # A malformed restore must not mutate the existing session.
    assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(before,session.state))
    continuation=torch.tensor([[32,97,110,100]],dtype=torch.long)
    expected=session.observe(continuation);forecast=session.forecast().energy(session.forecast().mean)
    code=f'''import sys,torch
sys.path.insert(0,{str(ROOT)!r});sys.path.insert(0,{str(ROOT/'reference')!r})
from energy_forecast.session import ForecastSession
torch.set_num_threads(1)
s=ForecastSession({str(teacher)!r},{str(FIT)!r});s.restore({str(snapshot)!r})
y=s.observe(torch.tensor([[32,97,110,100]],dtype=torch.long));f=s.forecast()
torch.save({{'logits':y,'energy':f.energy(f.mean),'status':s.point_forecast_status()}},{str(tmp_path/'fresh.pt')!r})
'''
    subprocess.run([sys.executable,'-c',code],check=True,cwd=ROOT)
    result=torch.load(tmp_path/'fresh.pt',weights_only=True)
    assert torch.equal(expected,result['logits']) and torch.equal(forecast,result['energy'])
    assert result['status']['horizons'][-1]['point_mean']=='training_target_mean'
