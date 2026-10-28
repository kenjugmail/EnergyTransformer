from dataclasses import replace
import pytest
import torch
from rewt.memory import MemoryConfig,MemoryEnergyLM
from rewt.position_memory import PositionMemoryConfig,PositionMemoryEnergyLM,load_learning_checkpoint


def config():
    return PositionMemoryConfig(width=8,hidden=12,modes=4,history=8,position_period=8,
        attention_heads=1,context_kernel=3,recurrent_type='gru',output_geometry='centered',
        compatibility_geometry='anchor_relative',wave_geometry='anchor_relative',initial_carry_scale=0.)


def test_position_memory_causal_chunked_and_rollover():
    torch.manual_seed(2201);model=PositionMemoryEnergyLM(config()).double()
    x=torch.randint(0,128,(2,27));changed=x.clone();changed[:,13:]=7
    whole=model(x,details=True);altered=model(changed)
    torch.testing.assert_close(whole.logits[:,:13],altered.logits[:,:13],rtol=0,atol=0)
    first=model(x[:,:11]);last=model(x[:,11:],state=first.state)
    torch.testing.assert_close(whole.logits,torch.cat((first.logits,last.logits),1),rtol=0,atol=0)
    torch.testing.assert_close(whole.details['features'][...,:8],model.encode_tokens(x),rtol=1e-12,atol=1e-12)


def test_zero_positions_preserve_memory_model_and_exact_cost():
    c=replace(config(),position_initial_scale=0.)
    torch.manual_seed(2203);base=MemoryEnergyLM(MemoryConfig(**{k:v for k,v in vars(c).items() if not k.startswith('position_')})).double()
    torch.manual_seed(2203);position=PositionMemoryEnergyLM(c).double()
    x=torch.randint(0,128,(2,13))
    torch.testing.assert_close(base(x).logits,position(x).logits,rtol=0,atol=0)
    assert position.parameters_count()-base.parameters_count()==c.position_period*c.width


def test_position_energy_gradient_and_checkpoint(tmp_path):
    torch.manual_seed(2205);model=PositionMemoryEnergyLM(config()).double()
    state=model.initial_state(2);condition,_=model.condition(torch.tensor([65,66]),state)
    candidate=(condition.anchor+torch.randn_like(condition.anchor)*.02).requires_grad_(True)
    auto=torch.autograd.grad(model.energy(candidate,condition).sum(),candidate,retain_graph=True)[0]
    torch.testing.assert_close(model.gradient(candidate,condition),auto,rtol=1e-10,atol=1e-10)
    model=model.float();path=tmp_path/'position.pt'
    from dataclasses import asdict
    torch.save(dict(arm='position_memory_energy',config=asdict(model.config),seed=2205,weights=model.state_dict()),path)
    loaded,_=load_learning_checkpoint(path);x=torch.tensor([[65,66,67]])
    torch.testing.assert_close(model(x).logits,loaded(x).logits,rtol=0,atol=0)


def test_invalid_position_config():
    with pytest.raises(ValueError,match='position_period'):PositionMemoryEnergyLM(replace(config(),position_period=7))
