import pytest
import torch
from experiments.pamf_phase3_restoration.covariance_inverse import dense_action


@pytest.mark.parametrize('n,k',[(1,1),(12,5),(12,12),(256,64)])
def test_compact_action_error_and_gradient_match_real_matrix(n,k):
    # Catches wrong conjugation, missing reflected tap collisions, or wrong IQ norm.
    from experiments.pamf_phase3_restoration.factorized_action_loss import relative_action_mse
    generator=torch.Generator().manual_seed(41)
    make=lambda shape:torch.randn(shape,dtype=torch.complex128,generator=generator)
    h=make((2,k)).requires_grad_();m=make((2,n)).requires_grad_();q=make((2,n)).requires_grad_()
    ht,mt,qt=make((2,k)),make((2,n)),make((2,n))
    actual=relative_action_mse(h,m,q,ht,mt,qt)
    reference=dense_action(ht,mt,qt)
    wanted=(dense_action(h,m,q)-reference).square().sum((1,2))/reference.square().sum((1,2)).clamp_min(1e-8)
    torch.testing.assert_close(actual,wanted,rtol=2e-13,atol=2e-13)
    got=torch.autograd.grad(actual.sum(),(h,m,q),retain_graph=True)
    expected=torch.autograd.grad(wanted.sum(),(h,m,q))
    for x,y in zip(got,expected):torch.testing.assert_close(x,y,rtol=2e-12,atol=2e-12)


def test_equal_actions_and_reflection_cancellation_have_zero_loss():
    # Catches cancellation-sensitive Gram expansion or accidental diagonal doubling.
    from experiments.pamf_phase3_restoration.factorized_action_loss import relative_action_mse
    h=torch.zeros(1,5,dtype=torch.complex128);h[0,0]=1;h[0,2]=-1
    m=torch.zeros(1,8,dtype=torch.complex128);m[0,1]=1
    q=torch.zeros_like(m);zero=torch.zeros_like(m)
    assert torch.count_nonzero(dense_action(h,m,q))==0
    assert relative_action_mse(h,m,q,h,zero,zero).item()==0
    h.requires_grad_();m.requires_grad_();q.requires_grad_()
    loss=relative_action_mse(h,m,q,h.detach(),m.detach(),q.detach())
    assert loss.item()==0
    assert all(torch.count_nonzero(g)==0 for g in torch.autograd.grad(loss.sum(),(h,m,q)))


def test_compact_action_error_uses_original_denominator_floor():
    from experiments.pamf_phase3_restoration.factorized_action_loss import relative_action_mse
    h=torch.ones(1,1,dtype=torch.complex128);m=torch.ones(1,2,dtype=torch.complex128);q=torch.zeros_like(m)
    torch.testing.assert_close(relative_action_mse(h,m,q,h,2*m,q),torch.tensor([.25],dtype=torch.float64))
    torch.testing.assert_close(relative_action_mse(h,m,q,h,0*m,q),torch.tensor([4e8],dtype=torch.float64))
    with pytest.raises(ValueError):relative_action_mse(h,m,q,h,m[:,:1],q)
    with pytest.raises(ValueError):relative_action_mse(h.real,m,q,h,m,q)


def test_factor_only_estimator_preserves_source_objective_and_all_parameter_gradients():
    # Catches a different estimator path, changed source weights, or detached factors.
    from experiments.pamf_phase3_restoration.covariance_model import scratch_covariance,source_action_loss
    model=scratch_covariance(41).eval()
    generator=torch.Generator().manual_seed(42)
    with torch.no_grad():
        for head in [model.kernel_head,model.coefficient_head,model.noise_head]:
            head.weight.copy_(.001*torch.randn(head.weight.shape,generator=generator))
    y=torch.randn(2,2,256,generator=generator)
    h=torch.randn(2,2,64,generator=generator);c=torch.randn(2,4,256,generator=generator)
    variance=torch.tensor([1e-8,.2])
    dense=model.estimate(y);factors=model.estimate(y,dense=False)
    assert set(factors)=={'kernel','main','image','variance'}
    torch.testing.assert_close(dense['action_matrix'],dense_action(factors['kernel'].to(torch.complex128),factors['main'].to(torch.complex128),factors['image'].to(torch.complex128)),rtol=0,atol=0)
    loss,terms=source_action_loss(factors,h,c,variance);reference,expected=source_action_loss(dense,h,c,variance)
    torch.testing.assert_close(loss,reference,rtol=0,atol=0)
    for key in terms:torch.testing.assert_close(terms[key],expected[key],rtol=0,atol=0)
    actual_grad=torch.autograd.grad(loss.sum(),tuple(model.parameters()))
    expected_grad=torch.autograd.grad(reference.sum(),tuple(model.parameters()))
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)
