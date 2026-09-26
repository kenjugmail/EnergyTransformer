import torch
import pytest
from energy_forecast.decoder_bridge import DecoderConfig,DecoderAdapter
from energy_forecast.calibrated import HorizonAnchor
from energy_forecast.fold_decoder_mean import fold_mean

def fixture():
    torch.manual_seed(171)
    m=DecoderAdapter(DecoderConfig('mean')).double();a=HorizonAnchor(16,8,(1,4,16,64)).double()
    with torch.no_grad():
        m.last.weight.normal_(std=.01);m.last.bias.normal_(std=.01)
        m.feature_center.normal_();m.feature_scale.uniform_(.2,2.)
        a.weight.normal_(std=.2);a.constant.normal_();a.gain.copy_(torch.tensor([1.,.7,.4,0.]))
    return m,a

def test_mean_fold_forward():
    m,a=fixture();new=fold_mean(m,a)
    h=torch.randn(5,160,dtype=torch.float64);x=torch.randn(5,16,8,dtype=torch.float64)
    torch.testing.assert_close(m.delta(h,x,mean=a(x)),new.delta(h,x),atol=3e-15,rtol=1e-11)
    assert new.config.kind=='history'

def test_mean_fold_input_gradients():
    m,a=fixture();new=fold_mean(m,a)
    h=torch.randn(2,160,dtype=torch.float64,requires_grad=True);x=torch.randn(2,16,8,dtype=torch.float64,requires_grad=True)
    old=m.delta(h,x,mean=a(x)).square().sum();actual=new.delta(h,x).square().sum()
    ga=torch.autograd.grad(old,(h,x));gb=torch.autograd.grad(actual,(h,x))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=1e-14,rtol=1e-10)

def test_fold_rejects_nonlinear_read():
    _,a=fixture()
    with pytest.raises(ValueError):fold_mean(DecoderAdapter(DecoderConfig('energy_read')),a)
