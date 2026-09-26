"""Independent checks for ordered square-root conditioning and score identities."""
import pytest
import torch
from energy_forecast.conditioning import GaussianMixture


def fixture(dtype=torch.float64,dim=9):
    g=torch.Generator().manual_seed(729)
    logits=torch.randn(2,3,generator=g,dtype=dtype)
    means=torch.randn(2,3,dim,generator=g,dtype=dtype)
    raw=torch.randn(2,3,dim,dim,generator=g,dtype=dtype)*.15
    L=torch.tril(raw,-1)+torch.eye(dim,dtype=dtype)*.9
    return GaussianMixture(logits,means,L)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('count',[2,4,8])
def test_prefix_equals_dense_reordered(dtype,count):
    f=fixture(dtype);y=torch.linspace(-.4,.7,2*count,dtype=dtype).view(2,count)
    a=f.condition_prefix(count,y)
    b=f.condition(list(reversed(range(count))),y.flip(-1))
    for aa,bb in [(a.distribution.mean,b.distribution.mean),
                  (a.distribution.component_covariance,b.distribution.component_covariance),
                  (a.log_evidence,b.log_evidence)]:
        torch.testing.assert_close(aa,bb,atol=2e-6 if dtype==torch.float32 else 1e-12,rtol=2e-5)


def test_density_identity_and_independent_distribution():
    f=fixture();obs=torch.randn(2,4,dtype=torch.float64);rest=torch.randn(2,5,dtype=torch.float64)
    result=f.condition_prefix(4,obs)
    torch.testing.assert_close(result.distribution.energy(rest),f.energy(torch.cat([obs,rest],-1))+result.log_evidence)
    from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal
    d=MixtureSameFamily(Categorical(logits=f.logits),MultivariateNormal(f.means,scale_tril=f.scale_tril))
    torch.testing.assert_close(-d.log_prob(torch.cat([obs,rest],-1))+result.log_evidence,result.distribution.energy(rest))


def test_prefix_gradients():
    f=fixture(dim=4)
    logits=f.logits.requires_grad_();means=f.means.requires_grad_();raw=torch.zeros_like(f.scale_tril,requires_grad=True)
    obs=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    target=torch.randn(2,2,dtype=torch.float64)
    def fun(l,m,r,o):
        L=torch.tril(r,-1)+torch.diag_embed(torch.diagonal(r,dim1=-2,dim2=-1).exp())
        return GaussianMixture(l,m,L).condition_prefix(2,o).distribution.energy(target)
    assert torch.autograd.gradcheck(fun,(logits,means,raw,obs),fast_mode=True)


def test_square_root_avoids_schur_cancellation():
    L=torch.tensor([[[[1.,0.],[1.,1e-4]]]],dtype=torch.float32)
    f=GaussianMixture(torch.zeros(1,1),torch.zeros(1,1,2),L)
    cov=f.component_covariance
    assert float(cov[0,0,1,1]-cov[0,0,1,0]**2/cov[0,0,0,0])==0.
    p=f.condition_prefix(1,torch.zeros(1,1)).distribution
    torch.testing.assert_close(p.variance,torch.tensor([[1e-8]]))
    assert torch.isfinite(p.energy(torch.zeros(1,1))).all()


@pytest.mark.parametrize('count',[-1,9,True,1.5])
def test_bad_prefix(count):
    with pytest.raises(ValueError):fixture().condition_prefix(count,torch.zeros(2,1,dtype=torch.float64))


def test_sequential_prefix_condition_and_nonmutation():
    f=fixture();before=f.payload();o=torch.randn(2,6,dtype=torch.float64)
    joint=f.condition_prefix(6,o)
    a=f.condition_prefix(2,o[:,:2]);b=a.distribution.condition_prefix(4,o[:,2:])
    torch.testing.assert_close(joint.distribution.mean,b.distribution.mean)
    torch.testing.assert_close(joint.log_evidence,a.log_evidence+b.log_evidence)
    for k,v in before.items():assert torch.equal(v,getattr(f,k))
