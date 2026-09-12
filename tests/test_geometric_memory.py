import pytest
import torch
from energy_transformer.geometric_memory import EditMemory,congruence_edit,KINDS
from energy_transformer.recall_data import episodes

@pytest.mark.parametrize('kind',KINDS)
def test_noop_does_not_change_storage(kind):
    torch.manual_seed(1);m=EditMemory(kind).double();s=m.initial(2)
    x=torch.randn(2,12,dtype=torch.double);v=torch.randn(2,4,dtype=torch.double)
    s=m.step(x,v,torch.zeros(2,dtype=torch.long),s)
    before=(s.precision.clone(),s.information.clone())
    out=m.step(x,v,torch.full((2,),2),s)
    assert torch.equal(out.precision,before[0]);assert torch.equal(out.information,before[1])

@pytest.mark.parametrize('geometry',[False,True])
def test_congruence_dense_formula_and_gradient(geometry):
    torch.manual_seed(2);a=torch.randn(2,4,4,dtype=torch.double);p=.01*torch.eye(4)+a@a.transpose(-1,-2)
    s=torch.randn(2,4,3,dtype=torch.double);k=torch.nn.functional.normalize(torch.randn(2,4,dtype=torch.double),dim=-1)
    v=torch.randn(2,3,dtype=torch.double);erase=torch.tensor([.3,.9],dtype=torch.double);write=torch.tensor([.5,2.],dtype=torch.double)
    pn,sn=congruence_edit(p,s,k,v,erase,write,geometry=geometry)
    if geometry:
        d=torch.linalg.solve(p,k[:,:,None]).squeeze(-1);d=d/(k*d).sum(-1,keepdim=True)
    else:d=k
    A=torch.eye(4)-erase[:,None,None]*d[:,:,None]*k[:,None,:]
    exact=.01*torch.eye(4)+A.transpose(-1,-2)@(p-.01*torch.eye(4))@A+write[:,None,None]*k[:,:,None]*k[:,None,:]
    torch.testing.assert_close(pn,exact);torch.testing.assert_close(sn,A.transpose(-1,-2)@s+write[:,None,None]*k[:,:,None]*v[:,None,:])
    assert torch.linalg.eigvalsh(pn).min()>=.01-1e-10
    k.requires_grad_();v.requires_grad_()
    assert torch.autograd.gradcheck(lambda k,v:congruence_edit(p,s,k,v,erase,write,geometry=geometry),(k,v))


def test_minimum_metric_erase_direction():
    torch.manual_seed(3);a=torch.randn(5,5,dtype=torch.double);p=.1*torch.eye(5)+a@a.T
    k=torch.nn.functional.normalize(torch.randn(5,dtype=torch.double),dim=0)
    d=torch.linalg.solve(p,k);d=d/(k@d)
    assert abs(k@d-1)<1e-12
    for _ in range(20):
        v=torch.randn(5,dtype=torch.double);v=v-k*(k@v);other=d+v
        assert d@p@d<=other@p@other+1e-12

@pytest.mark.parametrize('kind',KINDS)
def test_state_only_streaming_and_finite_gradients(kind):
    torch.manual_seed(4);m=EditMemory(kind);b=episodes(991,batch=2,nkeys=4,edits=2)
    out,s,_=m(b.keys,b.values,b.operations,b.queries)
    z=m.initial(2)
    for t in range(b.keys.shape[1]):z=m.step(b.keys[:,t],b.values[:,t],b.operations[:,t],z)
    assert torch.equal(out,m.read(b.queries,z))
    ((out-b.targets)**2).mean().backward()
    assert m.key.weight.grad is not None and torch.isfinite(m.key.weight.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_target_generation_not_provided_by_key_id():
    b=episodes(992,batch=5,nkeys=6,edits=7)
    state=torch.zeros_like(b.targets)
    for t in range(b.keys.shape[1]):
        for n in range(5):
            op=int(b.operations[n,t]);i=int(b.key_ids[n,t])
            if op==0:state[n,i]=b.values[n,t]
            if op==1:state[n,i]=0
    assert torch.equal(state,b.targets)
    assert torch.equal(b.targets[:,3:],b.initial_values[:,3:])


def test_data_reproducibility():
    a=episodes(33);b=episodes(33);c=episodes(34)
    assert torch.equal(a.keys,b.keys) and not torch.equal(a.keys,c.keys)


def test_raw_energy_not_cached_mode():
    m=EditMemory('precision');s=m.initial(1)
    assert set(s.__dataclass_fields__)=={'precision','information'}
    before=s.information.clone();_=m.read(torch.randn(1,2,12),s)
    assert torch.equal(before,s.information)

@pytest.mark.parametrize('dtype,atol',[(torch.float64,1e-9),(torch.float32,2e-4)])
def test_projected_energy_matches_preconditioned_delta(dtype,atol):
    torch.manual_seed(51);a=EditMemory('preconditioned_delta').to(dtype);b=EditMemory('projected').to(dtype)
    b.load_state_dict(a.state_dict());ep=episodes(1201,batch=3,edits=20,correlation=.9)
    p=a.initial(3);q=b.initial(3)
    for t in range(ep.keys.shape[1]):
        x=ep.keys[:,t].to(dtype);v=ep.values[:,t].to(dtype);op=ep.operations[:,t]
        p=a.step(x,v,op,p);q=b.step(x,v,op,q)
        torch.testing.assert_close(q.precision,p.precision,rtol=1e-6,atol=atol)
    torch.testing.assert_close(a.read(ep.queries.to(dtype),p),b.read(ep.queries.to(dtype),q),rtol=atol,atol=atol)


def test_projection_obeys_query_constraint():
    torch.manual_seed(2);m=EditMemory('projected').double();s=m.initial(1)
    ep=episodes(45,batch=1,nkeys=4,edits=0,gap=0)
    for t in range(4):s=m.step(ep.keys[:,t].double(),ep.values[:,t].double(),ep.operations[:,t],s)
    x=ep.queries[:,0].double();v=torch.tensor([[2.,-1.,3.,-4.]],dtype=torch.double);op=torch.zeros(1,dtype=torch.long)
    k=m.encode_key(x);old=(k[:,None,:]@torch.linalg.solve(s.precision,s.information)).squeeze(1)
    alpha=torch.sigmoid(m.controller(torch.cat([x,torch.nn.functional.one_hot(op,3).double()],-1))[:,0])
    new=m.step(x,v,op,s);got=(k[:,None,:]@torch.linalg.solve(new.precision,new.information)).squeeze(1)
    torch.testing.assert_close(got,old+alpha[:,None]*(v-old),atol=1e-10,rtol=1e-10)
