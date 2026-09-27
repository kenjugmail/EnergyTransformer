import math
import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from energy_forecast.conditioning import GaussianMixture
from energy_forecast.model import ForecastConfig, ForecastEnergy


def setup_function():
    torch.set_num_threads(1)


def example(batch=3):
    torch.manual_seed(2401)
    model = ForecastEnergy(ForecastConfig(history=3, variables=2, horizons=(1, 4), width=8)).double()
    with torch.no_grad():
        model.raw_correlation.normal_(std=.5)
    return GaussianMixture.from_landscape(model(torch.randn(batch, 3, 2, dtype=torch.float64)))


def test_density_matches_independent_torch_distribution():
    fun=example(); y=torch.randn(3,4,dtype=torch.float64)
    reference=MixtureSameFamily(Categorical(logits=fun.logits),MultivariateNormal(fun.means,scale_tril=fun.scale_tril))
    torch.testing.assert_close(fun.energy(y),-reference.log_prob(y))


def test_conditional_joint_minus_marginal_identity():
    fun=example(); y=torch.randn(3,4,dtype=torch.float64)
    result=fun.condition([2,0],y[:,[2,0]])
    expected=fun.energy(y)-fun.marginal([2,0]).energy(y[:,[2,0]])
    torch.testing.assert_close(result.distribution.energy(y[:,[1,3]]),expected,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(result.log_evidence,-fun.marginal([2,0]).energy(y[:,[2,0]]))
    assert result.remaining_indices==(1,3)


def test_condition_order_and_sequential_equivalence():
    f=example(); y=torch.randn(3,4,dtype=torch.float64)
    joint=f.condition([0,2],y[:,[0,2]])
    one=f.condition([0],y[:,[0]])
    two=one.distribution.condition([1],y[:,[2]])
    for name in ('means','scale_tril','logits'):
        torch.testing.assert_close(getattr(joint.distribution,name),getattr(two.distribution,name),atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(joint.log_evidence,one.log_evidence+two.log_evidence)


def test_single_gaussian_reference():
    covariance=torch.tensor([[4.,1.2],[1.2,2.]],dtype=torch.float64)[None,None]
    fun=GaussianMixture(torch.zeros(1,1,dtype=torch.float64),torch.zeros(1,1,2,dtype=torch.float64),torch.linalg.cholesky(covariance))
    conditioned=fun.condition([0],torch.tensor([[2.]],dtype=torch.float64)).distribution
    torch.testing.assert_close(conditioned.mean,torch.tensor([[.6]],dtype=torch.float64))
    torch.testing.assert_close(conditioned.variance,torch.tensor([[1.64]],dtype=torch.float64))


def test_conditioning_can_increase_mixture_variance():
    # Within-component covariance decreases, TOTAL posterior variance need not.
    means=torch.tensor([[[0.,0.],[5.,0.]]],dtype=torch.float64)
    scale=torch.diag_embed(torch.tensor([[[1.,.1],[1.,10.]]],dtype=torch.float64))
    f=GaussianMixture(torch.log(torch.tensor([[.999,.001]],dtype=torch.float64)),means,scale)
    post=f.condition([0],torch.tensor([[5.]],dtype=torch.float64)).distribution
    assert post.variance.item()>f.variance[0,1]


def test_candidate_observation_and_parameter_gradients():
    f=example(batch=1).cloned()
    obs=torch.tensor([[.4]],dtype=torch.float64,requires_grad=True)
    target=torch.randn(1,3,dtype=torch.float64,requires_grad=True)
    means=f.means.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x,y,m:GaussianMixture(f.logits,m,f.scale_tril).condition([1],x).distribution.energy(y),(obs,target,means))


def test_samples_moments():
    f=example(batch=1).condition([0],torch.tensor([[.5]],dtype=torch.float64)).distribution
    x=f.sample(80000,generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(x.mean(1),f.mean,atol=.018,rtol=.02)
    torch.testing.assert_close(x.var(1),f.variance,atol=.025,rtol=.03)


def test_quantile_gaussian_and_cdf():
    f=example(batch=1).condition([0],torch.tensor([[.2]],dtype=torch.float64)).distribution
    for q in (.05,.5,.95):
        x=f.quantile(q);sigma=f.scale_tril.square().sum(-1).sqrt()
        cdf=(f.logits.softmax(-1)[...,None]*.5*torch.erfc(-(x[:,None]-f.means)/sigma/math.sqrt(2))).sum(1)
        torch.testing.assert_close(cdf,torch.full_like(cdf,q),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('idx',[[0,0],[-1],[4],[0,1,2,3],[True]])
def test_bad_indices(idx):
    f=example()
    with pytest.raises((ValueError,TypeError)):
        f.condition(idx,torch.zeros(3,len(idx),dtype=torch.float64))


def test_empty_observations_identity_and_nonmutation():
    f=example(); old=f.payload()
    result=f.condition([],torch.zeros(3,0,dtype=torch.float64))
    assert result.distribution is f and torch.count_nonzero(result.log_evidence)==0
    f.condition([0],torch.zeros(3,1,dtype=torch.float64))
    for key in old:assert torch.equal(old[key],getattr(f,key))


def test_reject_invalid_observations_and_matrices():
    f=example()
    with pytest.raises(ValueError):f.condition([0],torch.full((3,1),float('nan'),dtype=torch.float64))
    bad=f.scale_tril.clone();bad[0,0,0,0]=-1
    with pytest.raises(ValueError):GaussianMixture(f.logits,f.means,bad)


def test_payload_defensive_copy():
    f=example(); old=f.mean.clone();payload=f.payload();payload['means'].zero_()
    assert torch.equal(f.mean,old)
    with pytest.raises(ValueError):GaussianMixture.restore({'means':f.means})


def test_extreme_representable_gaussian_quantiles_remain_finite():
    fun=GaussianMixture(torch.zeros(1,1),torch.zeros(1,1,1),torch.ones(1,1,1,1))
    for q in (1e-12,1.-1e-12):
        actual=fun.quantile(q)
        expected=torch.special.ndtri(torch.tensor(q,dtype=torch.float64)).float()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual,expected.reshape(1,1),rtol=1e-6,atol=1e-6)
