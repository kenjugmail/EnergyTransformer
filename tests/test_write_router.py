from pathlib import Path
import json,subprocess,sys,dataclasses
import pytest
import torch
from energy_transformer.write_router import RouterConfig,WriteRouterLM,choice,affine_prefix,training_labels,router_loss
from energy_transformer.scaling_data import batch,SET,DELETE,NOISE,ASK,SEP,BOS,UNKNOWN,KEY_START,VALUE_START

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]

def model(kind='hard'):
    torch.manual_seed(7001);return WriteRouterLM(RouterConfig(kind=kind,width=32,hidden=48)).eval()

def leaves(s):
    if torch.is_tensor(s):yield s
    elif isinstance(s,(tuple,list)):
        for a in s:yield from leaves(a)

@pytest.mark.parametrize('kind',['hard','soft','attention'])
def test_causality(kind):
    m=model(kind);x=torch.randint(128,(2,27));changed=x.clone();changed[:,13:]=torch.randint(128,(2,14))
    with torch.no_grad():a,_,_=m(x);b,_,_=m(changed)
    torch.testing.assert_close(a[:,:13],b[:,:13],atol=0,rtol=0)

@pytest.mark.parametrize('kind',['hard','soft','attention'])
def test_chunk_and_reset(kind):
    m=model(kind);x=torch.randint(128,(2,37));reset=torch.zeros_like(x,dtype=torch.bool);reset[:,0]=True;reset[0,19]=True
    with torch.no_grad():
        expected,full,_=m(x,reset=reset);ys=[];s=None
        for i,j in [(0,1),(1,5),(5,19),(19,21),(21,37)]:
            y,s,_=m(x[:,i:j],s,reset=reset[:,i:j]);ys.append(y)
    torch.testing.assert_close(torch.cat(ys,1),expected,atol=2e-5,rtol=2e-5)
    for a,b in zip(leaves(s),leaves(full)):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    with torch.no_grad():fresh,_,_=m(x[0:1,19:])
    torch.testing.assert_close(fresh,expected[0:1,19:],atol=2e-5,rtol=2e-5)


def test_exact_hard_forward_and_surrogate_gradient():
    p=torch.randn(3,4,requires_grad=True);a=choice(p,True)
    assert set(a.detach().flatten().tolist())<={0.,1.}
    assert torch.equal(a.sum(-1),torch.ones(3))
    (a*torch.arange(4)).sum().backward();assert p.grad.abs().sum()>0


def test_scan_serial_and_gradients():
    torch.manual_seed(18);f=torch.rand(2,6,4,dtype=torch.float64,requires_grad=True);b=torch.randn(2,6,4,3,dtype=torch.float64,requires_grad=True);s=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True)
    out=affine_prefix(f,b,s);state=s;hist=[]
    for i in range(6):state=f[:,i,:,None]*state+b[:,i];hist.append(state)
    ref=torch.stack(hist,1);torch.testing.assert_close(out,ref,atol=1e-12,rtol=1e-12)
    gs=torch.autograd.grad(out.square().sum(),(f,b,s),retain_graph=True);rs=torch.autograd.grad(ref.square().sum(),(f,b,s))
    for g,r in zip(gs,rs):torch.testing.assert_close(g,r,atol=1e-11,rtol=1e-11)


def test_labels_causal_only():
    x=torch.tensor([[BOS,SET,KEY_START+3,VALUE_START+5,SEP,NOISE,KEY_START+1,VALUE_START+2,SEP,DELETE,KEY_START+3,SEP,ASK,KEY_START+3,UNKNOWN]])
    op,k,v=training_labels(x)
    assert op.tolist()==[[0,0,0,0,1,0,0,0,0,0,0,2,0,3,0]]
    assert k[0,4]==3 and v[0,4]==5 and k[0,11]==3 and k[0,13]==3
    xx=x.clone();xx[:,12:]=99
    old=training_labels(xx)
    for a,b in zip((op,k,v),old):assert torch.equal(a[:,:12],b[:,:12])

@pytest.mark.parametrize('kind',['hard','soft','attention'])
def test_loss_and_nonmutation(kind):
    m=model(kind).train();z=batch(20441,0,4);_,s,_=m(z['x'][:,:8]);saved=[t.detach().clone() for t in leaves(s)]
    logits,_,aux=m(z['x'],reset=z['reset']);loss=router_loss(logits,aux,z,1);loss.backward()
    assert torch.isfinite(loss)
    assert m.embedding.weight.grad.abs().sum()>0
    assert m.address.weight.grad.abs().sum()>0
    m(z['x'][:,8:12],s)
    for a,b in zip(saved,leaves(s)):torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('kind',['hard','soft'])
def test_finite_positive_constant_state(kind):
    m=model(kind);x=torch.randint(128,(2,81))
    with torch.no_grad():_,s,_=m(x)
    assert (s[1]>=m.config.prior-1e-6).all()
    assert [tuple(t.shape) for t in s]==[(2,3,32),(2,32),(2,32,17)]
    for t in s:assert t.untyped_storage().nbytes()==t.numel()*t.element_size()


def test_trained_labels_absent_and_fresh_process(tmp_path):
    file=ROOT/'results/v20/hard_aux_seed11/best.pt'
    if not file.exists():pytest.skip('final trained checkpoint not yet available')
    d=torch.load(file,weights_only=True);m=WriteRouterLM(RouterConfig(**d['config']));m.load_state_dict(d['model']);m.eval()
    z=batch(201501,1,1);x=z['x'];split=21
    with torch.no_grad():_,s,_=m(x[:,:split]);ref,end,_=m(x[:,split:],s)
    snapshot=tmp_path/'restart.pt';torch.save({'state':tuple(t.clone() for t in s),'new':x[:,split:]},snapshot)
    target=tmp_path/'output.pt'
    code="import torch;from energy_transformer.write_router import *;torch.set_num_threads(1);d=torch.load("+repr(str(file))+",weights_only=True);m=WriteRouterLM(RouterConfig(**d['config']));m.load_state_dict(d['model']);m.eval();s=torch.load("+repr(str(snapshot))+",weights_only=True);\nwith torch.inference_mode(): y,t,_=m(s['new'],s['state']);torch.save((y,t),"+repr(str(target))+")"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    output,outstate=torch.load(target,weights_only=True)
    assert torch.equal(ref,output)
    for a,b in zip(end,outstate):assert torch.equal(a,b)


def test_no_gold_arguments_in_signature():
    import inspect
    params=set(inspect.signature(WriteRouterLM.forward).parameters)
    assert not params.intersection({'labels','operations','keys','values','answers','query'})
