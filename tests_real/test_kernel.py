"""Correctness tests for the fast energy kernel.

Direct kernel comparisons use independent serial recurrences and finite
differences. Model parity checks additionally compare the original Spark
implementation; they are compatibility checks, not a novelty claim.
"""
import math, sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'spark_kit'))

from energy_real.kernel import chunk_gla, read_again           # noqa: E402
from energy_real.model import LM, Config                        # noqa: E402
from energy_spark.model import ScaleLM, ScaleConfig, diagonal_scan, transported_scan  # noqa: E402


def naive_gla(u, wk, v, f, s0, u2=None):
    """S_t = f_t*S_{t-1} + wk_t (x) v_t ; r_t = u_t . S_t"""
    S = s0.clone()
    r, r2 = [], []
    for t in range(u.shape[1]):
        S = f[:, t, ..., None] * S + wk[:, t, ..., None] * v[:, t, :, None, :]
        r.append(torch.einsum('bhk,bhkv->bhv', u[:, t], S))
        if u2 is not None:
            r2.append(torch.einsum('bhk,bhkv->bhv', u2[:, t], S))
    return torch.stack(r, 1), (torch.stack(r2, 1) if u2 is not None else None), S


@pytest.fixture
def case():
    torch.manual_seed(7)
    B, T, H, K, V = 2, 37, 3, 8, 5
    g = dict(u=torch.randn(B, T, H, K, dtype=torch.float64),
             wk=torch.randn(B, T, H, K, dtype=torch.float64),
             v=torch.randn(B, T, H, V, dtype=torch.float64),
             f=torch.rand(B, T, H, K, dtype=torch.float64).clamp(1e-3, .999),
             s0=torch.randn(B, H, K, V, dtype=torch.float64),
             u2=torch.randn(B, T, H, K, dtype=torch.float64))
    return g


@pytest.mark.parametrize('chunk', [1, 2, 4, 8, 16, 64])
def test_chunk_gla_matches_serial_loop(case, chunk):
    want_r, want_r2, want_S = naive_gla(case['u'], case['wk'], case['v'], case['f'], case['s0'], case['u2'])
    got_r, got_S, ctx = chunk_gla(case['u'], case['wk'], case['v'], case['f'].log(), case['s0'], chunk)
    got_r2 = read_again(ctx, case['u2'])
    assert (got_r - want_r).abs().max() < 1e-10
    assert (got_r2 - want_r2).abs().max() < 1e-10
    assert (got_S - want_S).abs().max() < 1e-10


@pytest.mark.parametrize('chunk', [4, 16])
def test_chunk_gla_gradients_match_serial_loop(case, chunk):
    names = ('u', 'wk', 'v', 's0', 'f')
    a = {n: case[n].clone().requires_grad_(True) for n in names}
    r, _, _ = chunk_gla(a['u'], a['wk'], a['v'], a['f'].log(), a['s0'], chunk)
    r.square().sum().backward()
    b = {n: case[n].clone().requires_grad_(True) for n in names}
    rn, _, _ = naive_gla(b['u'], b['wk'], b['v'], b['f'], b['s0'])
    rn.square().sum().backward()
    for n in names:
        assert (a[n].grad - b[n].grad).abs().max() < 1e-9, n


@pytest.mark.parametrize('backend', ['dmat', 'matmul'])
def test_single_step_both_reads_and_state_gradients(case, backend):
    names = ('u', 'wk', 'v', 's0', 'f', 'u2')
    values = {n: case[n] if n == 's0' else case[n][:, :1] for n in names}
    a = {n: values[n].clone().requires_grad_(True) for n in names}
    r, state, ctx = chunk_gla(a['u'], a['wk'], a['v'], a['f'].log(), a['s0'], 64, backend)
    outputs = (r, read_again(ctx, a['u2']), state)
    b = {n: values[n].clone().requires_grad_(True) for n in names}
    expected = naive_gla(b['u'], b['wk'], b['v'], b['f'], b['s0'], b['u2'])
    for got, want in zip(outputs, expected):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)
    sum(v.square().sum() for v in outputs).backward()
    sum(v.square().sum() for v in expected).backward()
    for name in names:
        torch.testing.assert_close(a[name].grad, b[name].grad, atol=1e-9, rtol=1e-9)


def test_reference_diagonal_scan_matches_serial_loop():
    """Guards the original spark_kit scan, with an additional direct serial test."""
    torch.manual_seed(0)
    f = torch.rand(2, 17, 3, 8, dtype=torch.float64).clamp(.05, .99)
    b = torch.randn(2, 17, 3, 8, 4, dtype=torch.float64)
    init = torch.randn(2, 3, 8, 4, dtype=torch.float64)
    x, out = init, []
    for t in range(f.shape[1]):
        x = f[:, t, ..., None] * x + b[:, t]
        out.append(x)
    assert (diagonal_scan(f, b, init) - torch.stack(out, 1)).abs().max() < 1e-12


def test_reference_transported_scan_matches_serial_loop():
    torch.manual_seed(0)
    f = torch.rand(2, 17, 3, 8, dtype=torch.float64).clamp(.05, .99)
    b = torch.randn(2, 17, 3, 8, 4, dtype=torch.float64)
    init = torch.randn(2, 3, 8, 4, dtype=torch.float64)
    strides = torch.tensor([0, 1, 3])
    x, out = init.clone(), []
    for t in range(f.shape[1]):
        rolled = torch.stack([torch.roll(x[:, h], shifts=int(strides[h]), dims=1) for h in range(3)], 1)
        x = f[:, t, ..., None] * rolled + b[:, t]
        out.append(x)
    assert (transported_scan(f, b, init, strides) - torch.stack(out, 1)).abs().max() < 1e-12


@pytest.mark.parametrize('slots,T,chunk', [(8, 37, 32), (8, 128, 32), (16, 101, 64), (16, 512, 32)])
def test_fast_model_matches_reference_model(slots, T, chunk):
    torch.manual_seed(3)
    kw = dict(vocab=256, width=64, layers=2, heads=4, slots=slots, ff_width=128)
    ref = ScaleLM(ScaleConfig(kind='energy', chunk=8192, **kw)).eval()
    fast = LM(Config(kind='energy', chunk=chunk, **kw)).eval()
    fast.load_state_dict(ref.state_dict(), strict=True)
    tok = torch.randint(0, 256, (3, T))
    with torch.no_grad():
        a, sa = ref(tok)
        b, sb = fast(tok)
    assert (a.softmax(-1) - b.softmax(-1)).abs().max() < 1e-5
    for la, lb in zip(sa, sb):
        for x, y in zip(la, lb):
            assert (x - y).abs().max() < 1e-4


@pytest.mark.parametrize('kind', ['energy', 'softmax'])
@pytest.mark.parametrize('split', [5, 13, 32, 37, 50])
def test_state_carry_equals_full_sequence(kind, split):
    torch.manual_seed(5)
    m = LM(Config(kind=kind, vocab=256, width=64, layers=2, heads=4, slots=8, ff_width=128, chunk=16)).eval()
    tok = torch.randint(0, 256, (3, 96))
    with torch.no_grad():
        full, _ = m(tok)
        a, s = m(tok[:, :split])
        b, _ = m(tok[:, split:], s)
    assert (full - torch.cat([a, b], 1)).abs().max() < 2e-4


@pytest.mark.parametrize('kind', ['energy', 'softmax'])
def test_strictly_causal(kind):
    torch.manual_seed(11)
    m = LM(Config(kind=kind, vocab=256, width=64, layers=2, heads=4, slots=8, ff_width=128, chunk=16)).eval()
    tok = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        base, _ = m(tok)
        alt = tok.clone(); alt[0, 20] = (alt[0, 20] + 7) % 256
        pert, _ = m(alt)
    delta = (base - pert).abs().amax(-1)[0]
    assert delta[:20].max() == 0, 'information leaked backwards in time'
    assert delta[20:].max() > 0, 'perturbation had no forward effect'


def test_energy_and_softmax_parameter_counts_match_within_half_percent():
    e = LM(Config(kind='energy', width=384, layers=6, heads=4, slots=16, ff_width=1536)).parameters_count()
    s = LM(Config(kind='softmax', width=384, layers=6, heads=4, slots=16, ff_width=1506)).parameters_count()
    assert abs(e - s) / e < 0.005, (e, s)


def test_precision_stays_positive():
    """mode = S/P divides by precision; P must never approach zero."""
    torch.manual_seed(13)
    c = Config(kind='energy', vocab=256, width=64, layers=1, heads=4, slots=8, ff_width=128, chunk=16)
    m = LM(c).eval()
    mixer = m.blocks[0].mixer
    with torch.no_grad():
        # drive the decay gate to its floor so precision decays as fast as allowed
        mixer.projection.bias.view(c.heads, -1)[:, 2 * c.slots + mixer.v:3 * c.slots + mixer.v] = -20.0
        out, states = m(torch.randint(0, 256, (2, 64)))
    P, S = states[0]
    assert torch.isfinite(out).all()
    assert (P >= c.prior - 1e-6).all(), f'precision fell below prior: {P.min()}'


@pytest.mark.parametrize('chunk', [4, 8, 16, 32])
def test_matmul_and_dmat_paths_agree(case, chunk):
    """The fast GEMM path and the always-valid decay-tensor path must match.

    Uses a decay mild enough that the GEMM path is inside its validity domain
    for every chunk size tested.
    """
    torch.manual_seed(21)
    f = torch.rand_like(case['f']) * .199 + .8          # log-span <= 32*0.223 = 7.1
    a, _, _ = chunk_gla(case['u'], case['wk'], case['v'], f.log(), case['s0'], chunk, force='matmul')
    b, _, _ = chunk_gla(case['u'], case['wk'], case['v'], f.log(), case['s0'], chunk, force='dmat')
    assert (a - b).abs().max() < 1e-10


def test_fixed_dmat_supports_extreme_decay():
    """A decay span beyond float32 range must fall back, not overflow."""
    from energy_real.kernel import SAFE_LOG_SPAN
    torch.manual_seed(2)
    B, T, H, K, V, chunk = 1, 32, 2, 4, 3, 32
    f = torch.full((B, T, H, K), 1e-4, dtype=torch.float64)   # span = 32*9.21 = 295 >> 80
    u = torch.randn(B, T, H, K, dtype=torch.float64)
    wk = torch.randn(B, T, H, K, dtype=torch.float64)
    v = torch.randn(B, T, H, V, dtype=torch.float64)
    s0 = torch.randn(B, H, K, V, dtype=torch.float64)
    assert float((f.log().view(B, 1, T, H, K).cumsum(2)[:, :, :1]
                  - f.log().view(B, 1, T, H, K).cumsum(2)[:, :, -1:]).max()) > SAFE_LOG_SPAN
    actual, _, ctx = chunk_gla(u, wk, v, f.log(), s0, chunk)     # must pick dmat
    assert ctx.dmat is not None, 'fixed dmat backend expected'
    want, _, _ = naive_gla(u, wk, v, f, s0)[0], None, None
    assert torch.isfinite(actual).all()
    assert (actual - want).abs().max() < 1e-10


@pytest.mark.parametrize('force', ['dmat'])
def test_gradients_finite_under_extreme_decay(force):
    """Regression: masking after exp() overflowed, and where() backpropagated 0*inf=NaN.

    The forward pass is unaffected -- the masked entries are discarded either
    way -- so this only shows up in the backward pass.
    """
    torch.manual_seed(4)
    B, T, H, K, V, chunk = 2, 64, 2, 8, 6, 32
    # decay small enough that the within-chunk log span is huge
    f = torch.full((B, T, H, K), 1e-4, dtype=torch.float32)
    u = torch.randn(B, T, H, K, requires_grad=True)
    wk = torch.randn(B, T, H, K, requires_grad=True)
    v = torch.randn(B, T, H, V, requires_grad=True)
    s0 = torch.zeros(B, H, K, V)
    r, _, _ = chunk_gla(u, wk, v, f.log(), s0, chunk, force=force)
    assert torch.isfinite(r).all(), 'forward produced non-finite values'
    r.square().sum().backward()
    for name, g in (('u', u.grad), ('wk', wk.grad), ('v', v.grad)):
        assert torch.isfinite(g).all(), f'non-finite gradient in {name} ({force} path)'


def test_forced_matmul_outside_validity_domain_raises():
    """Misuse must be loud, not silently non-finite."""
    torch.manual_seed(4)
    B, T, H, K, V, chunk = 1, 32, 2, 4, 3, 32
    f = torch.full((B, T, H, K), 1e-4)
    args = (torch.randn(B, T, H, K), torch.randn(B, T, H, K), torch.randn(B, T, H, V),
            f.log(), torch.zeros(B, H, K, V), chunk)
    with pytest.raises(ValueError, match='validity domain'):
        chunk_gla(*args, force='matmul')
