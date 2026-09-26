from dataclasses import asdict,replace
import torch
from energy_real.model import Config,FastFunctionMixer
from energy_real.train import parser,model_config

def test_protected_projection_roundtrip_and_amp():
    torch.manual_seed(606)
    base=Config(width=16,heads=2,slots=4,ff_width=32,chunk=8)
    cfg=Config(**asdict(replace(base,energy_projection_fp32=True)))
    old=FastFunctionMixer(base);new=FastFunctionMixer(cfg);new.load_state_dict(old.state_dict())
    x=torch.randn(1,17,16,requires_grad=True)
    torch.testing.assert_close(old(x)[0],new(x)[0],rtol=0,atol=0)
    seen=[]
    handle=new.projection.register_forward_hook(lambda m,a,o:seen.append(o.dtype))
    with torch.autocast('cpu',dtype=torch.bfloat16):y,state=new(x)
    handle.remove();y.float().square().mean().backward()
    assert seen==[torch.float32] and y.dtype==torch.bfloat16
    assert all(s.dtype==torch.float32 for s in state)
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_training_precision_policy_is_explicit():
    args=parser().parse_args(['--out','unused','--energy-projection-fp32'])
    assert model_config(args).energy_projection_fp32
    assert not Config().energy_projection_fp32
