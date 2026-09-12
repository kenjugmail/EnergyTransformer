import pytest
import torch
from energy_transformer.quadratic_edit import QuadraticState

def test_exact_identity_no_erase_no_write():
    s=QuadraticState.initial(4,3);e=torch.ones(4,dtype=torch.float64);v=torch.ones(3,dtype=torch.float64)
    updated=s.edit(e,e,v,weight=0)
    assert torch.equal(s.precision,updated.precision) and torch.equal(s.information,updated.information)

def test_positive_floor_after_random_edits():
    g=torch.Generator().manual_seed(17200);s=QuadraticState.initial(8,4)
    for _ in range(100):
        e,k=torch.randn(2,8,generator=g,dtype=torch.float64);v=torch.randn(4,generator=g,dtype=torch.float64)
        s=s.edit(e,k,v,erase_strength=.8,weight=.7)
        assert torch.linalg.eigvalsh(s.precision).min()>=.25-1e-12
        assert torch.isfinite(s.mode()).all()

def test_distinct_erase_write_directions():
    a,b,c=torch.eye(3,dtype=torch.float64);v=torch.tensor([1.,2.],dtype=torch.float64)
    s=QuadraticState.initial(3,2).edit(a,a,v).edit(a,c,v*2)
    new=s.edit(a,b,v*3,erase_strength=1.)
    assert torch.equal(new.information[0],torch.zeros_like(v))
    torch.testing.assert_close(new.information[1],v*3)
    torch.testing.assert_close(new.information[2],s.information[2])
    # In this diagonal, orthogonal example the unrelated query is preserved exactly.
    torch.testing.assert_close(new.mode()[2],s.mode()[2])

def test_energy_congruence_identity_up_to_factor_constant():
    torch.manual_seed(17201);s=QuadraticState.initial(4,3)
    for _ in range(4):s=s.edit(torch.randn(4,dtype=torch.float64),torch.randn(4,dtype=torch.float64),torch.randn(3,dtype=torch.float64))
    e,k=torch.randn(2,4,dtype=torch.float64);v=torch.randn(3,dtype=torch.float64);w=torch.randn(4,3,dtype=torch.float64)
    alpha=.4;weight=.7;a=torch.eye(4,dtype=torch.float64)-alpha*torch.outer(e/e.norm(),e/e.norm())
    new=s.edit(e,k,v,erase_strength=alpha,weight=weight)
    expected=s.energy(a@w)-.5*s.prior*(a@w).square().sum()+.5*s.prior*w.square().sum()+.5*weight*(k@w-v).square().sum()-.5*weight*v.square().sum()
    torch.testing.assert_close(new.energy(w),expected)

def test_autograd_through_update_and_read():
    s=QuadraticState.initial(3,2);e=torch.ones(3,dtype=torch.float64);k=torch.randn(3,dtype=torch.float64,requires_grad=True);v=torch.randn(2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda k,v:s.edit(e,k,v,erase_strength=.3).mode(),(k,v))

@pytest.mark.parametrize('alpha,weight',[(-.1,1),(1.1,1),(0,-1)])
def test_invalid_gates(alpha,weight):
    s=QuadraticState.initial(2,2);t=torch.ones(2,dtype=torch.float64)
    with pytest.raises(ValueError):s.edit(t,t,t,erase_strength=alpha,weight=weight)
