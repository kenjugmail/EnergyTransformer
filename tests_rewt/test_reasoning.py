from dataclasses import replace
import torch
import pytest
from rewt.reasoning import ReadReviseReasoner, ReasoningConfig

torch.set_num_threads(1)

def model():
    torch.manual_seed(2601)
    return ReadReviseReasoner(ReasoningConfig(width=8, modes=5, max_hops=3)).double()

def test_analytic_gradient_and_per_read_descent():
    m = model()
    x = torch.randn(2, 272, dtype=torch.float64, requires_grad=True)
    a = torch.randn(2, 5, dtype=torch.float64)*.2
    f = m.frequencies(); p = .15+1.85*m.raw_precision.sigmoid()
    expected, = torch.autograd.grad(m.energy(x, a, f, m.phase, p).sum(), x)
    torch.testing.assert_close(m.gradient(x, a, f, m.phase, p), expected)
    out = m(torch.randint(0,256,(2,30)), details=True)
    e = out['energy_trace']
    assert e.shape == (2,3,3)
    assert bool((e[...,1:] <= e[...,:-1]+1e-12).all())
    out['answer_logits'].square().mean().backward()
    for name, param in m.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name

def test_prefix_encoding_and_padding_do_not_leak():
    m = model().eval(); tokens = torch.randint(0,256,(2,41))
    torch.testing.assert_close(m.encode(tokens[:,:17]), m.encode(tokens)[:,:17])
    short = m(tokens[:,:17], details=True)
    padded = m(tokens, torch.tensor([17,17]), details=True)
    for k in ('answer_logits','step_logits','count_logits'):
        torch.testing.assert_close(short[k],padded[k],atol=1e-12,rtol=1e-12)
    assert torch.isneginf(padded['pointer_logits'][...,17:]).all()

def test_controls_identical_parameters_and_effective_reread():
    m = model().eval(); x = torch.randint(0,256,(2,29))
    fixed = m(x, reread=False, energy_rounds=0)
    assert torch.equal(fixed['step_logits'][:,0],fixed['step_logits'][:,1])
    live = m(x, energy_rounds=0)
    torch.testing.assert_close(live['step_logits'][:,0],fixed['step_logits'][:,0])
    assert not torch.equal(live['step_logits'][:,1],fixed['step_logits'][:,1])
    zero = ReadReviseReasoner(replace(m.config,energy_rounds=0)).double()
    zero.load_state_dict(m.state_dict())
    torch.testing.assert_close(zero(x)['answer_logits'],live['answer_logits'])

def test_predict_is_batch_padding_invariant_and_weights_fixed():
    m = model(); before={k:v.clone() for k,v in m.state_dict().items()}
    a=m.predict(['A>B;from A go 1=?\n'])
    b=m.predict(['A>B;from A go 1=?\n','walk C 2:D>E;C>D;H>I=?\n'])
    assert a[0]==b[0]
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())

@pytest.mark.parametrize('kwargs',[dict(width=0),dict(max_hops=0),dict(energy_rounds=-1),dict(step_size=float('nan')),dict(reread=1)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ReadReviseReasoner(ReasoningConfig(**kwargs))
