from dataclasses import asdict, replace

import pytest
import torch

from rewt.memory import MemoryConfig, MemoryEnergyLM
from rewt.model import EnergyConfig, JointEnergyLM


def configuration(**kwargs):
    defaults=dict(width=8,hidden=20,modes=4,history=8,context_kernel=3,recurrent_type='gru',
        output_geometry='centered',compatibility_geometry='anchor_relative',
        wave_geometry='anchor_relative',initial_carry_scale=0.)
    return MemoryConfig(**{**defaults,**kwargs})


def test_one_head_matches_existing_attention_without_extra_parameters():
    torch.set_num_threads(1);torch.manual_seed(382)
    config=configuration(attention_heads=1)
    model=MemoryEnergyLM(config).double().eval()
    payload=asdict(config);payload.pop('attention_heads')
    existing=JointEnergyLM(EnergyConfig(**payload)).double().eval()
    existing.load_state_dict(model.state_dict())
    tokens=torch.randint(256,(2,15))
    with torch.inference_mode():
        a=model(tokens).logits;b=existing(tokens).logits
    assert model.parameters_count()==existing.parameters_count()
    torch.testing.assert_close(a,b,rtol=2e-12,atol=2e-12)


def test_multiple_heads_are_causal_chunkable_and_refine_the_same_energy():
    torch.set_num_threads(1);torch.manual_seed(383)
    model=MemoryEnergyLM(configuration()).double().eval()
    x=torch.randint(256,(2,25));changed=x.clone();changed[:,12:]=(changed[:,12:]+49)%256
    with torch.inference_mode():
        whole=model(x,details=True);other=model(changed)
        first=model(x[:,:9]);second=model(x[:,9:18],state=first.state)
        third=model(x[:,18:],state=second.state)
    torch.testing.assert_close(whole.logits[:,:12],other.logits[:,:12],rtol=0,atol=0)
    torch.testing.assert_close(whole.logits,torch.cat((first.logits,second.logits,third.logits),1),rtol=2e-12,atol=2e-12)
    assert (whole.details['energy_trace'][...,1:]<=whole.details['energy_trace'][...,:-1]+1e-12).all()
    cond,_=model.condition(x[:,0],model.initial_state(2))
    s=torch.randn_like(cond.anchor).requires_grad_()
    expected=torch.autograd.grad(model.energy(s,cond).sum(),s)[0]
    torch.testing.assert_close(model.gradient(s,cond),expected,rtol=2e-9,atol=2e-9)


@pytest.mark.parametrize('kwargs',[dict(attention_heads=3),dict(attention_heads=0)])
def test_invalid_head_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):MemoryEnergyLM(configuration(**kwargs))
