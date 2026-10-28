import pytest
import torch
from torch.nn import functional as F
from rewt.streaming import StreamingConfig, StreamingLM, LatentEnergy, gated_delta_chunk


@pytest.mark.parametrize('length',[1,7,32])
def test_gated_delta_matches_recurrence_and_gradients(length):
    torch.manual_seed(12)
    k=F.normalize(torch.randn(2,length,4,dtype=torch.double),dim=-1).requires_grad_()
    q=F.normalize(torch.randn(2,length,4,dtype=torch.double),dim=-1).requires_grad_()
    v=torch.randn(2,length,4,dtype=torch.double,requires_grad=True)
    beta=torch.rand(2,length,1,dtype=torch.double,requires_grad=True)
    decay=(.9+.1*torch.rand(2,length,1,dtype=torch.double)).requires_grad_()
    initial=torch.randn(2,4,4,dtype=torch.double,requires_grad=True)
    actual,final=gated_delta_chunk(k,v,beta,decay,q,initial)
    memory=initial;rows=[]
    for t in range(length):
        memory=decay[:,t,:,None]*memory
        error=v[:,t]-(k[:,t,:,None]*memory).sum(-2)
        memory=memory+beta[:,t,:,None]*k[:,t,:,None]*error[:,None]
        rows.append((q[:,t,:,None]*memory).sum(-2))
    expected=torch.stack(rows,1)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(final,memory,atol=1e-10,rtol=1e-9)
    inputs=(k,v,beta,decay,q,initial)
    ga=torch.autograd.grad(actual.square().sum()+final.square().sum(),inputs,retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum()+memory.square().sum(),inputs)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-8)


@pytest.mark.parametrize('hybrid,window',[(False,0),(False,8),(True,8),(True,1)])
def test_causal_chunk_and_decode_parity(hybrid,window):
    torch.manual_seed(4)
    model=StreamingLM(StreamingConfig(width=16,heads=2,hidden=24,hybrid=hybrid,window=window,chunk=5,modes=4)).eval()
    x=torch.randint(256,(2,27))
    with torch.inference_mode():
        expected,_,_=model(x)
        altered=x.clone();altered[:,12:]=torch.randint(256,(2,15))
        changed,_,_=model(altered)
        torch.testing.assert_close(expected[:,:12],changed[:,:12],atol=1e-6,rtol=1e-5)
        rows=[];state=None
        for a,b in [(0,3),(3,14),(14,15),(15,27)]:
            logits,state,_=model(x[:,a:b],state,return_state=True);rows.append(logits)
        torch.testing.assert_close(torch.cat(rows,1),expected,atol=2e-6,rtol=2e-5)
        rows=[];state=None
        for token in x.unbind(1):
            logits,state,_=model(token[:,None],state,return_state=True);rows.append(logits)
        torch.testing.assert_close(torch.cat(rows,1),expected,atol=2e-6,rtol=2e-5)


def test_energy_gradient_descent_and_parameter_gradients():
    torch.manual_seed(42)
    config=StreamingConfig(width=16,heads=2,hidden=24,modes=4)
    model=StreamingLM(config)
    logits,_,energy=model(torch.randint(256,(2,21)),details=True,rounds=8)
    assert (energy.diff(dim=-1)<=2e-6).all()
    logits.square().mean().backward()
    for name,p in model.named_parameters():
        assert p.grad is not None,name
        assert torch.isfinite(p.grad).all(),name
    delta=torch.randn(2,16,dtype=torch.double,requires_grad=True)
    a=torch.randn(2,4,dtype=torch.double);w=torch.randn(4,16,dtype=torch.double)
    phase=torch.randn(4,dtype=torch.double);q=torch.rand(16,dtype=torch.double)
    analytic=q*delta-(a*F.linear(delta,w,phase).sin())@w
    actual=torch.autograd.grad(LatentEnergy.energy(delta,a,w,phase,q).sum(),delta)[0]
    torch.testing.assert_close(actual,analytic)


def test_bounded_owned_state_and_stale_state_rejection():
    model=StreamingLM(StreamingConfig(width=16,heads=2,hidden=24,window=8,modes=4)).eval()
    with torch.no_grad():
        _,first,_=model(torch.randint(256,(1,32)),return_state=True)
        snapshot=[tuple(t.clone() for t in layer) for layer in first.layers]
        _,second,_=model(torch.randint(256,(1,61)),first,return_state=True)
        assert first.tensor_bytes()==second.tensor_bytes()
        for layer,old in zip(first.layers,snapshot):
            for a,b in zip(layer,old):torch.testing.assert_close(a,b,rtol=0,atol=0)
        model.embedding.weight.add_(.01)
        with pytest.raises(ValueError,match='changed weights'):model(torch.zeros(1,1,dtype=torch.long),second)


def test_zero_rounds_exact_anchor():
    head=LatentEnergy(StreamingConfig(width=16,heads=2,modes=4))
    anchor=torch.randn(2,13,16)
    output,trace=head(anchor,0,trace=True)
    torch.testing.assert_close(output,anchor,atol=0,rtol=0)
    assert trace.shape==(2,13,1)


@pytest.mark.parametrize('kwargs',[{'width':15},{'rounds':-1},{'window':-1},{'chunk':0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):StreamingLM(StreamingConfig(**kwargs))
