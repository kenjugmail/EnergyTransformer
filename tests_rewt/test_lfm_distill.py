import numpy as np
import pytest
import torch
from torch.nn import functional as F

from rewt.lfm_distill import ConversationTokens, RecoveryTokens, relative_mse, vocabulary_losses


def test_distillation_exact_full_vocabulary_loss_and_gradient():
    torch.manual_seed(281703)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    reference = torch.randn(2, 7, 8)
    weights = torch.randn(31, 8, requires_grad=True)
    donor = torch.randn(31, 8)
    targets = torch.randint(31, (2, 7)); temperature = 1.7
    actual = vocabulary_losses(hidden, reference, targets, weights, donor, chunk=3, temperature=temperature)
    logits = F.linear(hidden, weights); teacher = F.linear(reference, donor)
    teacher_logp = F.log_softmax(teacher/temperature, -1)
    expected_kl = (teacher_logp.exp()*(teacher_logp-F.log_softmax(logits/temperature, -1))).sum(-1).mean()*temperature**2
    expected_nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(actual['kl'], expected_kl)
    torch.testing.assert_close(actual['nll'], expected_nll)
    grads = torch.autograd.grad(actual['kl']+.1*actual['nll'], (hidden, weights), retain_graph=True)
    expected_grads = torch.autograd.grad(expected_kl+.1*expected_nll, (hidden, weights))
    for a, b in zip(grads, expected_grads):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_sampler_never_crosses_document_boundary_and_is_reproducible(tmp_path):
    tokens = np.array([11]*5+[22]*7+[33]*9, dtype=np.uint32)
    np.save(tmp_path/'train_tokens.npy', tokens)
    np.save(tmp_path/'train_offsets.npy', np.array([0, 5, 12, 21]))
    data = ConversationTokens(tmp_path, 'train', 5)
    x, y = data.batch(30, np.random.default_rng(1), 'cpu')
    x2, y2 = data.batch(30, np.random.default_rng(1), 'cpu')
    assert torch.equal(x, x2) and torch.equal(y, y2)
    assert torch.equal(x, y)
    assert set(x[:, 0].tolist()) == {22, 33}
    assert (x == x[:, :1]).all()


def test_hidden_alignment_relative_scale():
    x = torch.randn(3, 5)
    assert relative_mse(x, x) == 0
    torch.testing.assert_close(relative_mse(torch.zeros_like(x), x), torch.tensor(1.))


def test_prefix_sampling_keeps_the_start_of_each_conversation(tmp_path):
    np.save(tmp_path/'train_tokens.npy', np.array(list(range(10, 17))+list(range(20, 29)), dtype=np.uint32))
    np.save(tmp_path/'train_offsets.npy', np.array([0, 7, 16]))
    data = ConversationTokens(tmp_path, 'train', 5)
    x, y = data.batch(30, np.random.default_rng(281707), 'cpu', prefix_fraction=1.)
    assert set(x[:, 0].tolist()) == {10, 20}
    assert torch.equal(x+1, y)
    with pytest.raises(ValueError, match='prefix fraction'):
        data.batch(1, np.random.default_rng(1), 'cpu', prefix_fraction=1.1)


def test_weighted_distillation_masks_padding_and_preserves_gradients():
    torch.manual_seed(12)
    hidden=torch.randn(2,5,8,requires_grad=True); teacher=torch.randn_like(hidden)
    weight=torch.randn(31,8,requires_grad=True); donor=torch.randn(31,8)
    targets=torch.randint(31,(2,5)); mask=torch.tensor([[.1,1,1,0,0],[.1,.1,1,1,0]])
    result=vocabulary_losses(hidden,teacher,targets,weight,donor,chunk=3,token_weights=mask)
    logits=F.linear(hidden,weight);reference=F.linear(teacher,donor)
    logp=reference.log_softmax(-1)
    kl=((logp.exp()*(logp-logits.log_softmax(-1))).sum(-1)*mask).sum()/mask.sum()
    nll=(F.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view(2,5)*mask).sum()/mask.sum()
    torch.testing.assert_close(result['kl'],kl);torch.testing.assert_close(result['nll'],nll)
    grads=torch.autograd.grad(result['kl']+.2*result['nll'],(hidden,weight),retain_graph=True)
    expected=torch.autograd.grad(kl+.2*nll,(hidden,weight))
    for x,y in zip(grads,expected):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    assert torch.count_nonzero(grads[0][mask==0])==0


def test_recovery_sampler_includes_short_documents_and_masks_shifted_targets(tmp_path):
    np.save(tmp_path/'train_tokens.npy',np.array([10,11,12,20,21,22,23,24,25]))
    np.save(tmp_path/'train_assistant.npy',np.array([0,0,1,0,0,1,1,1,1]))
    np.save(tmp_path/'train_offsets.npy',np.array([0,3,9]))
    data=RecoveryTokens(tmp_path,'train',5,99,.1)
    x,y,w,active=data.batch(20,np.random.default_rng(1),'cpu',prefix_fraction=1)
    assert set(x[:,0].tolist())=={10,20}
    for i in range(20):
        n=2 if x[i,0]==10 else 5
        assert torch.equal(y[i,:n],x[i,:n]+1)
        torch.testing.assert_close(w[i,0],torch.tensor(.1))
        assert (w[i,1:n]==1).all() and (w[i,n:]==0).all()
        assert (active[i,:n]==1).all() and (active[i,n:]==0).all()
