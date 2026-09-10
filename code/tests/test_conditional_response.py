import pytest
import torch

from cvsrffi.conditional_response import (ConditionalResponse, fit_support_response,
                                         matched_state_cross_rx_differences)


def test_same_variance_ridge_and_defensive_snapshot():
    torch.manual_seed(7)
    phi = torch.randn(8, 3, dtype=torch.double)
    z = torch.randn(8, 2, dtype=torch.double)
    prior = torch.randn(3, 2, dtype=torch.double)
    sigma2, lam = 2.5, 0.7
    cov = sigma2 * torch.eye(2, dtype=torch.double).expand(8, 2, 2)
    post = fit_support_response(z, phi, cov, torch.ones_like(z, dtype=torch.bool),
                                list(range(8)), prior_mean=prior, prior_precision=lam/sigma2)
    c = phi.T @ phi + lam * torch.eye(3, dtype=torch.double)
    expected = torch.linalg.solve(c, phi.T @ z + lam * prior)
    torch.testing.assert_close(post.mean, expected)
    _, variance = post.predict(phi)
    leverage = (phi * torch.linalg.solve(c, phi.T).T).sum(-1)
    torch.testing.assert_close(variance, sigma2*leverage[:,None,None]*torch.eye(2,dtype=torch.double))
    copied = post.mean
    copied.zero_()
    z.zero_()
    torch.testing.assert_close(post.mean, expected)


def test_missing_correlated_heteroscedastic_matches_full_joint_conditioning():
    phi = torch.tensor([[1., 0.], [1., 2.]], dtype=torch.double)
    z = torch.tensor([[2., float('nan')], [1., 3.]], dtype=torch.double)
    cov = torch.tensor([[[2., 1.], [1., 3.]], [[1., .4], [.4, 2.]]], dtype=torch.double)
    mask = torch.tensor([[True, False], [True, True]])
    prior_cov = torch.tensor([[2.,.2,0.,0.],[.2,1.,0.,0.],[0.,0.,3.,.3],[0.,0.,.3,2.]],dtype=torch.double)
    prior_mean = torch.ones(2,2,dtype=torch.double)*.2
    post = fit_support_response(z,phi,cov,mask,['a','b'],prior_mean=prior_mean,prior_precision=torch.linalg.inv(prior_cov))
    h = torch.tensor([[1.,0.,0.,0.],[1.,0.,2.,0.],[0.,1.,0.,2.]],dtype=torch.double)
    noise = torch.block_diag(cov[0,:1,:1],cov[1])
    gain = prior_cov @ h.T @ torch.linalg.inv(noise+h@prior_cov@h.T)
    expected = prior_mean.flatten()+gain@(torch.tensor([2.,1.,3.],dtype=torch.double)-h@prior_mean.flatten())
    torch.testing.assert_close(post.mean.flatten(),expected)
    torch.testing.assert_close(torch.linalg.inv(post.precision),prior_cov-gain@h@prior_cov)


def test_single_state_rank_and_duplicate_rejection():
    phi = torch.tensor([[1.,0.]]).expand(10,2)
    z = torch.ones(10,1)
    cov = torch.ones(10,1,1)
    observed = torch.ones_like(z,dtype=torch.bool)
    post = fit_support_response(z,phi,cov,observed,list(range(10)))
    assert post.coverage()['design_rank'] == 1
    assert not post.coverage()['full_state_rank']
    assert post.precision[1,1] == 1
    with pytest.raises(ValueError,match='distinct'):
        fit_support_response(z,phi,cov,observed,[0]*10)


def test_clamped_response_query_purity_batch_invariance_and_gradient():
    m = ConditionalResponse(3,2,2).double()
    with pytest.raises(RuntimeError):
        m(torch.zeros(1,2,dtype=torch.double))
    m.fit_state_domain(torch.tensor([[-1.,-1.],[1.,1.]],dtype=torch.double), source_training=True)
    with torch.no_grad():
        m.shared_slopes.normal_()
        m.class_slopes.normal_()
    e = torch.tensor([[.2,.3],[4.,-.4]],dtype=torch.double,requires_grad=True)
    before = {k:v.clone() for k,v in m.state_dict().items()}
    out = m(e)
    torch.testing.assert_close(out[:1],m(e[:1]))
    torch.testing.assert_close(out[1:],m(e[1:].clamp(-1,1)))
    assert m.domain_diagnostics(e)['in_domain'].tolist() == [True,False]
    u = torch.eye(2,dtype=torch.double).expand(2,2,2).clone().requires_grad_()
    propagated = m.state_uncertainty(e,u)
    (out.square().sum()+propagated.square().sum()).backward()
    assert torch.isfinite(u.grad).all()
    assert torch.isfinite(m.class_slopes.grad).all()
    assert torch.isfinite(e.grad).all()
    for k,v in m.state_dict().items():
        torch.testing.assert_close(before[k],v)
    for d in range(2):
        actual = torch.autograd.functional.jacobian(lambda x:m(x)[0,0,d],e)[0]
        torch.testing.assert_close(actual,m.jacobian(e)[0,0,d])


def test_validation_and_no_observations():
    z,phi,cov = torch.zeros(1,2),torch.ones(1,1),torch.eye(2)[None]
    post=fit_support_response(z,phi,cov,torch.zeros_like(z,dtype=torch.bool),['a'])
    assert post.coverage()['design_rank']==0
    torch.testing.assert_close(post.precision,torch.eye(2,dtype=post.precision.dtype))
    with pytest.raises(ValueError,match='positive definite'):
        fit_support_response(z,phi,-cov,torch.ones_like(z,dtype=torch.bool),['a'])
    with pytest.raises(ValueError,match='source_training'):
        ConditionalResponse(2,2,1).fit_state_domain(phi,source_training=False)


def test_matched_state_diagnostic_does_not_pair_unmatched_samples():
    z=torch.tensor([[1.],[3.],[6.],[8.],[99.]])
    result=matched_state_cross_rx_differences(z,['a','b','a','b','a'],['r1','r1','r2','r2','r2'],['s','s','s','s','unmatched'])
    assert len(result)==1
    assert result[0]['squared_norm']==0


def test_support_gradients_basis_and_tensor_snapshot():
    m=ConditionalResponse(2,2,1).double()
    m.fit_state_domain(torch.tensor([[-2.],[2.]],dtype=torch.double),source_training=True)
    e=torch.tensor([[-1.],[1.]],dtype=torch.double)
    torch.testing.assert_close(m(e),torch.einsum('bp,cpd->bcd',m.basis(e),m.coefficients()))
    z=torch.randn(2,2,dtype=torch.double,requires_grad=True)
    post=fit_support_response(z,m.basis(e),torch.eye(2,dtype=torch.double).expand(2,2,2),
                              torch.ones_like(z,dtype=torch.bool),['a','b'],prior_mean=m.coefficients()[0])
    mean,cov=post.predict(m.basis(e))
    (mean.square().sum()+cov.sum()).backward()
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(m.mean.grad).all()
    restored=type(post).from_dict(post.to_dict())
    torch.testing.assert_close(restored.predict(m.basis(e))[0],mean)
    assert restored.coverage()['physical_shots']==2


def test_fp32_large_correlated_precision_snapshot_and_strict_input_symmetry():
    torch.manual_seed(27)
    d,p,n=160,3,2
    phi=torch.tensor([[1.,.21,.32],[1.,.22,.33]])
    z=torch.randn(n,d)
    low=torch.randn(d,5)*.04
    covariance=(torch.eye(d)*.001+low@low.T).expand(n,d,d)
    observed=torch.ones_like(z,dtype=torch.bool)
    post=fit_support_response(z,phi,covariance,observed,['a','b'])
    assert torch.equal(post.precision,post.precision.T)
    restored=type(post).from_dict(post.to_dict())
    means,variance=restored.predict(phi)
    assert means.dtype==phi.dtype and variance.dtype==phi.dtype
    assert torch.isfinite(variance).all()
    assert torch.linalg.eigvalsh(variance).min()>0
    invalid=covariance.clone()
    invalid[0,0,1]+=.001
    with pytest.raises(ValueError,match='symmetric'):
        fit_support_response(z,phi,invalid,observed,['a','b'])


def test_state_uncertainty_direction_changes_decision_not_only_temperature():
    from cvsrffi.partial_gaussian_head import gaussian_scores
    response=ConditionalResponse(2,2,2).double()
    response.fit_state_domain(torch.tensor([[-1.,-1.],[1.,1.]],dtype=torch.double),source_training=True)
    with torch.no_grad():
        response.mean.copy_(torch.eye(2,dtype=torch.double))
        response.shared_slopes.copy_(torch.eye(2,dtype=torch.double))
    state=torch.zeros(1,2,dtype=torch.double)
    z=torch.zeros(1,2,dtype=torch.double)
    mask=torch.ones(1,2,dtype=torch.bool)
    base=torch.eye(2,dtype=torch.double)*.1
    u=torch.diag(torch.tensor([1.,.01],dtype=torch.double))[None]
    first=gaussian_scores(z,response(state),base+response.state_uncertainty(state,u),mask)['scores']
    second=gaussian_scores(z,response(state),base+response.state_uncertainty(state,u.flip(-1).flip(-2)),mask)['scores']
    assert first.argmax(-1).item()==0
    assert second.argmax(-1).item()==1
