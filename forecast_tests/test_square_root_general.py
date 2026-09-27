"""Independent density, gradient and cancellation checks for arbitrary observations."""
import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from energy_forecast.conditioning import GaussianMixture, _row_factor
from energy_forecast.conditioning_reference import dense_condition, dense_marginal


def fixture(dtype=torch.float64, dim=5):
    g = torch.Generator().manual_seed(261407)
    logits = torch.randn(2, 3, generator=g, dtype=dtype)
    means = torch.randn(2, 3, dim, generator=g, dtype=dtype)
    raw = torch.randn(2, 3, dim, dim, generator=g, dtype=dtype) * .2
    factor = torch.tril(raw, -1) + torch.eye(dim, dtype=dtype) * .8
    return GaussianMixture(logits, means, factor)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('given', [(3,), (4, 1), (3, 1, 0), (1, 0)])
def test_general_condition_matches_independent_covariance(dtype, given):
    f = fixture(dtype)
    obs = torch.linspace(-.3, .9, 2 * len(given), dtype=dtype).view(2, -1)
    a = f.condition(list(given), obs)
    b = dense_condition(f, list(given), obs)
    assert a.remaining_indices == b.remaining_indices
    for x, y in [(a.distribution.mean, b.distribution.mean),
                 (a.distribution.component_covariance, b.distribution.component_covariance),
                 (a.log_evidence, b.log_evidence)]:
        torch.testing.assert_close(x, y, atol=3e-6 if dtype == torch.float32 else 2e-12, rtol=3e-5)
    full = torch.linspace(-1., 1., 10, dtype=dtype).view(2, 5)
    full[:, given] = obs
    dist = MixtureSameFamily(Categorical(logits=f.logits), MultivariateNormal(f.means, scale_tril=f.scale_tril))
    torch.testing.assert_close(a.distribution.energy(full[:, a.remaining_indices]),
                               -dist.log_prob(full) + a.log_evidence,
                               atol=1e-5 if dtype == torch.float32 else 1e-11, rtol=1e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('keep', [(0,), (0, 1, 2), (4, 1), (4, 3, 2, 1, 0)])
def test_marginal_preserves_requested_order_and_density(dtype, keep):
    f = fixture(dtype)
    a = f.marginal(list(keep)); b = dense_marginal(f, list(keep))
    torch.testing.assert_close(a.mean, b.mean)
    torch.testing.assert_close(a.component_covariance, b.component_covariance, atol=2e-6, rtol=1e-5)
    x = torch.zeros(2, len(keep), dtype=dtype)
    torch.testing.assert_close(a.energy(x), b.energy(x), atol=4e-6, rtol=1e-5)


def test_nonprefix_condition_gradient_check():
    torch.manual_seed(7)
    logits = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    means = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    raw = (torch.randn(1, 2, 4, 4, dtype=torch.float64) * .1).requires_grad_()
    obs = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[.1, -.2]], dtype=torch.float64)
    def fn(l, m, r, o):
        chol = torch.tril(r, -1) + torch.diag_embed(r.diagonal(dim1=-2, dim2=-1).exp())
        result = GaussianMixture(l, m, chol).condition([3, 1], o)
        return result.distribution.energy(target) + .3 * result.log_evidence
    assert torch.autograd.gradcheck(fn, (logits, means, raw, obs), fast_mode=True)


def test_marginal_qr_gradient_check():
    raw = torch.zeros(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    means = torch.ones(1, 2, 4, dtype=torch.float64, requires_grad=True)
    def fn(r, m):
        chol = torch.tril(r, -1) + torch.diag_embed(r.diagonal(dim1=-2, dim2=-1).exp())
        f = GaussianMixture(torch.zeros(1, 2, dtype=torch.float64), m, chol)
        return f.marginal([3, 1]).energy(torch.zeros(1, 2, dtype=torch.float64))
    assert torch.autograd.gradcheck(fn, (raw, means), fast_mode=True)


def cancellation_fixture():
    chol = torch.tensor([[[[1., 0., 0.], [1., 1e-4, 0.], [0., 0., 1.]]]])
    return GaussianMixture(torch.zeros(1, 1), torch.zeros(1, 1, 3), chol)


def test_cancellation_is_fixed_for_nonprefix_and_reordered_evidence():
    f = cancellation_fixture()
    with pytest.raises(torch.linalg.LinAlgError):
        dense_condition(f, [1], torch.zeros(1, 1))
    with pytest.raises(torch.linalg.LinAlgError):
        dense_condition(f, [1, 0], torch.zeros(1, 2))
    posterior = f.condition([1], torch.zeros(1, 1)).distribution
    torch.testing.assert_close(posterior.variance, torch.tensor([[1e-8, 1.]]), atol=1e-12, rtol=5e-5)
    assert torch.isfinite(posterior.energy(torch.zeros(1, 2))).all()
    torch.testing.assert_close(f.condition([1, 0], torch.zeros(1, 2)).distribution.variance, torch.ones(1, 1))


def test_cancellation_is_fixed_for_marginals():
    f = cancellation_fixture()
    for keep in ([0, 1], [1, 0]):
        with pytest.raises(torch.linalg.LinAlgError):
            dense_marginal(f, keep)
        reduced = f.marginal(keep)
        assert torch.isfinite(reduced.energy(torch.zeros(1, 2))).all()
        torch.testing.assert_close(reduced.scale_tril[..., 1, 1], torch.tensor([[1e-4]]), rtol=1e-5, atol=1e-9)


def test_arbitrary_sequential_conditioning_consistent():
    f = fixture()
    obs = torch.tensor([[.1, -.2], [-.4, .7]], dtype=torch.float64)
    combined = f.condition([3, 1], obs)
    first = f.condition([3], obs[:, :1])
    second = first.distribution.condition([first.remaining_indices.index(1)], obs[:, 1:])
    torch.testing.assert_close(combined.distribution.mean, second.distribution.mean)
    torch.testing.assert_close(combined.distribution.component_covariance, second.distribution.component_covariance)
    torch.testing.assert_close(combined.log_evidence, first.log_evidence + second.log_evidence)


def test_prefix_and_reordered_prefix_do_not_call_qr(monkeypatch):
    f = fixture()
    def fail(*args, **kwargs): raise AssertionError('Unneeded QR')
    monkeypatch.setattr(torch.linalg, 'qr', fail)
    a = f.condition([0, 1], torch.zeros(2, 2, dtype=torch.float64))
    b = f.condition([1, 0], torch.zeros(2, 2, dtype=torch.float64))
    assert torch.equal(a.distribution.mean, b.distribution.mean)
    f.marginal([0, 1])


def test_no_covariance_factorization_in_production_paths(monkeypatch):
    f = fixture()
    def fail(*args, **kwargs): raise AssertionError('Unneeded Cholesky')
    monkeypatch.setattr(torch.linalg, 'cholesky', fail)
    f.condition([3, 1], torch.zeros(2, 2, dtype=torch.float64))
    f.marginal([4, 2])


def test_reduced_factors_own_only_declared_storage():
    f = fixture(); original = f.scale_tril.clone()
    for sub in (f.marginal([0, 1]), f.marginal([4, 2]), f.condition_prefix(3, torch.zeros(2, 3, dtype=torch.float64)).distribution):
        assert sub.scale_tril.untyped_storage().nbytes() == sub.scale_tril.numel() * sub.scale_tril.element_size()
        sub.scale_tril.zero_()
        assert torch.equal(f.scale_tril, original)


def test_qr_explicit_numerical_rank_failure():
    with pytest.raises(FloatingPointError): _row_factor(torch.zeros(1, 1, 3, 4))


@pytest.mark.parametrize('given', [[0], [2]])
def test_observation_overflow_rejected(given):
    f = fixture(torch.float32)
    with pytest.raises(ValueError, match='overflows'):
        f.condition(given, torch.full((2, 1), 1e300, dtype=torch.float64))


@pytest.mark.parametrize('bad', [None, 'not a tensor', [[1., 2.]]])
def test_observation_type_rejected(bad):
    with pytest.raises(ValueError): fixture().condition([3, 1], bad)


def test_exact_online_policy_does_not_construct_prior_marginal(monkeypatch):
    from types import SimpleNamespace
    from energy_forecast.online import ObservedForecastSession
    online = object.__new__(ObservedForecastSession)
    online._base = SimpleNamespace(steps=11, head=SimpleNamespace(config=SimpleNamespace(horizons=(1, 4), variables=2)))
    online._ticket = {'origin':10, 'distribution':fixture(dim=4),
                      'observations':{1:torch.zeros(2, 2, dtype=torch.float64)}}
    online._gates = {}; online._gate_hash = None
    def fail(*args): raise AssertionError('Exact posterior does not use prior marginal')
    monkeypatch.setattr(GaussianMixture, 'marginal', fail)
    assert online.distribution(policy='exact_conditional').dimension == 2
