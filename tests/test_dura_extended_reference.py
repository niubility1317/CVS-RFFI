import numpy as np
import pytest
import torch
from experiments.pamf_dura_efficiency.extended_reference import extended_basis,auxiliary_decision
from experiments.pamf_dura_efficiency.soft_reference_tangent import minimum_shift,prototype_tangent_loss,ALPHA


def test_source_axes_keep_original_cache_and_remove_shared_RX_day_offset():
    rng=np.random.default_rng(42);head=np.eye(160,5,dtype=np.float32)
    centers=rng.normal(size=(6,160));cells=np.broadcast_to(centers,(5,3,6,160)).copy()
    q,info=extended_basis(cells,head)
    assert np.array_equal(q[:,:5],head) and np.max(abs(q.T@q-np.eye(10)))<3e-6
    shared=rng.normal(size=(5,3,1,160))*4
    changed,_=extended_basis(cells+shared,head)
    np.testing.assert_allclose(q[:,5:]@q[:,5:].T,changed[:,5:]@changed[:,5:].T,atol=1e-6)
    assert info['fit_cells']==90 and info['fit_RX_day_equal']
    with pytest.raises(ValueError):extended_basis(np.zeros_like(cells),head)


def test_explicit_source10_shift_and_real_autograd_cap_default_stays_rank5():
    torch.manual_seed(7);x=torch.randn(3,2,256);r=(x+.1*torch.randn_like(x)).requires_grad_()
    j=torch.randn(3,10,2,256);d=torch.randn(3,10)*.2
    shift,info=minimum_shift(j,d,dimension=10)
    torch.testing.assert_close((j*shift[:,None]).sum((2,3)),d,atol=2e-5,rtol=2e-5)
    assert (info['rank']==10).all()
    value,stat=prototype_tangent_loss(r,x,j,d,dimension=10)
    grad=torch.autograd.grad((ALPHA*value).sum(),r)[0]
    assert (grad.flatten(1).norm(dim=1)<=stat['IQ_gradient_norm']*(1+2e-5)).all()
    with pytest.raises(ValueError):minimum_shift(j,d)
    with pytest.raises(ValueError):prototype_tangent_loss(r,x,j,d)


def test_auxiliary_directions_must_pass_their_own_fixed_geometry():
    row=dict(count=512,defined=512,defined_IQ=512,mean_cosine=.6,mean_IQ_cosine=.1,nonnegative=400,
        RX_mean_cosine={str(r):.5 for r in [1,3,4,6,8]})
    assert auxiliary_decision({'1':row,'20':row})['eligible']
    assert not auxiliary_decision({'1':row,'20':dict(row,mean_cosine=.49)})['eligible']
    assert not auxiliary_decision({'1':row,'20':dict(row,nonnegative=383)})['eligible']
    assert not auxiliary_decision({'1':row,'20':dict(row,RX_mean_cosine=dict(row['RX_mean_cosine'],**{'3':-.01}))})['eligible']
