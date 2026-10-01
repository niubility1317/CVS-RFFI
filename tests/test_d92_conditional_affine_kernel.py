"""Independent finite-feature constrained primal oracle; synthetic data only."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.d92_conditional_affine_kernel import (
    fit_conditional_affine, conditional_affine_adjoint, NumericalFailure, EPS, ROUNDING_MULTIPLIER,
)


def centered(x):
    return x-x.mean(axis=1,keepdims=True)


def fixture(seed=217):
    rng=np.random.default_rng(seed);m,p,h,c,d=3,4,5,4,12
    O=rng.normal(size=(m,d));N=rng.normal(size=(p,d));H=rng.normal(size=(h,d))
    priors=dict(M_O=centered(rng.normal(size=(m,c))),M_N=centered(rng.normal(size=(p,c))),
        M_H=centered(rng.normal(size=(h,c))),R_N=centered(rng.normal(size=(p,c))))
    return O,N,H,priors


def blocks(O,N,H,priors):
    return dict(A=O@O.T,B=O@N.T,D=N@N.T,F=H@O.T,E=H@N.T,**priors)


def fit(O,N,H,priors):
    return fit_conditional_affine(**blocks(O,N,H,priors))


def primal_oracle(O,N,H,priors):
    """Solve weights, free b, and old equality multipliers directly.

    This uses neither conditional Gram nor the implementation's Schur solve.
    """
    p,d=N.shape;m=len(O);C=priors['R_N'].shape[1]
    design=np.column_stack((N,np.ones(p)));constraints=np.column_stack((O,np.ones(m)))
    penalty=np.diag(np.r_[np.ones(d),0.])
    system=np.block([[design.T@design+penalty,constraints.T],
                     [constraints,np.zeros((m,m))]])
    rhs=np.vstack((design.T@priors['R_N'],np.zeros((m,C))))
    solution=np.linalg.solve(system,rhs);w,b=solution[:d],solution[d]
    return w,b,priors['M_H']+H@w+b


def test_independent_equality_constrained_primal_scores_coefficients_and_old_zero():
    O,N,H,priors=fixture();state=fit(O,N,H,priors);a=state.arrays
    w,b,held=primal_oracle(O,N,H,priors)
    np.testing.assert_allclose(a['held_scores'],held,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(N.T@a['alpha']-O.T@a['beta'],w,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(a['v'],b,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(a['train_new_scores'],priors['M_N']+N@w+b,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(a['old_residual'],0,atol=3e-12)
    np.testing.assert_allclose(a['old_scores'],priors['M_O'],rtol=0,atol=3e-12)
    np.testing.assert_allclose(a['beta'].sum(0),a['alpha'].sum(0),atol=3e-12)
    np.testing.assert_allclose(a['A']@a['beta'],a['B']@a['alpha']+a['v'],atol=3e-12)
    assert np.linalg.eigvalsh(a['K_perp'])[0]>=-1e-12
    for key in ('alpha','beta','v','held_scores','old_scores','train_new_scores'):
        np.testing.assert_allclose(a[key].sum(axis=-1),0,atol=3e-12)
    # The RKHS norm is the norm of the primal weights, never a penalty on b.
    np.testing.assert_allclose(state.audit['residual_norm_squared'],np.sum(w*w),rtol=3e-12,atol=3e-12)


def test_per_record_expansion_and_old_new_held_and_class_permutations():
    O,N,H,priors=fixture();state=fit(O,N,H,priors)
    single=np.vstack([state.expansion.score(k_old=(H[i:i+1]@O.T),k_new=(H[i:i+1]@N.T),M=priors['M_H'][i:i+1]) for i in range(len(H))])
    np.testing.assert_allclose(single,state.arrays['held_scores'],rtol=3e-12,atol=3e-12)
    po=np.array([2,0,1]);pn=np.array([3,1,0,2]);ph=np.array([4,0,3,1,2]);pc=np.array([2,0,3,1])
    perm=dict(M_O=priors['M_O'][po][:,pc],M_N=priors['M_N'][pn][:,pc],M_H=priors['M_H'][ph][:,pc],R_N=priors['R_N'][pn][:,pc])
    other=fit(O[po],N[pn],H[ph],perm)
    for name,order in (('alpha',pn),('beta',po),('held_scores',ph)):
        np.testing.assert_allclose(other.arrays[name],state.arrays[name][order][:,pc],rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(other.arrays['v'],state.arrays['v'][pc],rtol=3e-12,atol=3e-12)
    G=centered(np.random.default_rng(802).normal(size=priors['M_H'].shape))
    g=conditional_affine_adjoint(state,G).arrays;gp=conditional_affine_adjoint(other,G[ph][:,pc]).arrays
    for name,left,right in (('A',po,po),('B',po,pn),('D',pn,pn),('F',ph,po),('E',ph,pn)):
        np.testing.assert_allclose(gp[name],g[name][left][:,right],rtol=3e-11,atol=3e-12)


@pytest.mark.parametrize('block',['A','B','D','F','E'])
def test_each_raw_kernel_block_directional_derivative_with_frozen_priors(block):
    O,N,H,priors=fixture();args=blocks(O,N,H,priors);state=fit_conditional_affine(**args)
    rng=np.random.default_rng(192);G=centered(rng.normal(size=priors['M_H'].shape));adj=conditional_affine_adjoint(state,G)
    assert np.linalg.norm(adj.arrays['g_b'])>.1
    direction=rng.normal(size=args[block].shape)
    if block in ('A','D'): direction=.5*(direction+direction.T)
    direction/=np.linalg.norm(direction);step=1e-5
    plus=dict(args,**{block:args[block]+step*direction});minus=dict(args,**{block:args[block]-step*direction})
    fd=np.sum(G*(fit_conditional_affine(**plus).arrays['held_scores']-fit_conditional_affine(**minus).arrays['held_scores']))/(2*step)
    np.testing.assert_allclose(np.sum(adj.arrays[block]*direction),fd,rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize('endpoint',['old','new','held'])
def test_all_endpoint_derivatives_against_independent_primal_KKT(endpoint):
    O,N,H,priors=fixture();state=fit(O,N,H,priors)
    rng=np.random.default_rng(905);G=centered(rng.normal(size=priors['M_H'].shape));g=conditional_affine_adjoint(state,G).arrays
    endpoint_gradients=dict(old=(g['A']+g['A'].T)@O+g['B']@N+g['F'].T@H,
        new=(g['D']+g['D'].T)@N+g['B'].T@O+g['E'].T@H,held=g['F']@O+g['E']@N)
    points=dict(old=O,new=N,held=H);direction=rng.normal(size=points[endpoint].shape);direction/=np.linalg.norm(direction)
    plus=dict(points);minus=dict(points);step=1e-5
    plus[endpoint]=points[endpoint]+step*direction;minus[endpoint]=points[endpoint]-step*direction
    sp=primal_oracle(plus['old'],plus['new'],plus['held'],priors)[2]
    sm=primal_oracle(minus['old'],minus['new'],minus['held'],priors)[2]
    fd=np.sum(G*(sp-sm))/(2*step)
    np.testing.assert_allclose(np.sum(endpoint_gradients[endpoint]*direction),fd,rtol=2e-6,atol=3e-8)


def test_rank_one_detach_A_and_missing_constant_adjoint_are_detectable():
    O,N,H,priors=fixture();state=fit(O,N,H,priors);a=state.arrays
    _,_,oracle=primal_oracle(O,N,H,priors)
    # Deliberately wrong: removing the free constant rank-one conditional term.
    simpleK=a['D']-a['B'].T@np.linalg.solve(a['A'],a['B'])
    simpleL=a['E']-a['F']@np.linalg.solve(a['A'],a['B'])
    wrong=priors['M_H']+simpleL@np.linalg.solve(simpleK+np.eye(len(N)),priors['R_N'])
    assert np.linalg.norm(wrong-oracle)>1e-3
    rng=np.random.default_rng(4);G=centered(rng.normal(size=priors['M_H'].shape));adj=conditional_affine_adjoint(state,G).arrays
    direction=adj['A']/np.linalg.norm(adj['A']);args=blocks(O,N,H,priors);step=1e-5
    fd=np.sum(G*(fit_conditional_affine(**dict(args,A=args['A']+step*direction)).arrays['held_scores']-
                 fit_conditional_affine(**dict(args,A=args['A']-step*direction)).arrays['held_scores']))/(2*step)
    assert abs(fd)>1e-3  # A detached would incorrectly give zero here.
    rhs=a['F'].T@G;wrong_nu=(a['z']@rhs)/float(a['s'])
    wrong_lambda=np.linalg.solve(a['A'],rhs)-a['z'][:,None]*wrong_nu
    raw=(wrong_lambda-adj['X_T'][:-1])@a['beta'].T;wrong_A=.5*(raw+raw.T)
    assert abs(np.sum(wrong_A*direction)-fd)>1e-4
    np.testing.assert_allclose(adj['Lambda'][:-1].sum(0),G.sum(0),atol=3e-12)


def test_real_rhs_work_and_immutable_state_no_free_intercept_refit():
    O,N,H,priors=fixture();args=blocks(O,N,H,priors);state=fit_conditional_affine(**args);m,p,c=len(O),len(N),4
    assert state.audit['factorization_count']==state.audit['completed_factorization_count']==2
    assert state.audit['spectral_diagnostic_count']==3 and state.audit['optimizer_steps']==0
    for prefix,n,width in (('projection',m,p+1),('residual',p,c)):
        assert state.audit[prefix+'_factorization_count']==1
        assert state.audit[prefix+'_triangular_solve_count']==2
        assert state.audit[prefix+'_triangular_rhs_count']==2*width
        assert state.audit[prefix+'_triangular_rhs_element_count']==2*n*width
        assert state.audit[prefix+'_triangular_dense_work_unit_count']==2*n*n*width
    adj=conditional_affine_adjoint(state,np.ones_like(priors['M_H']))
    assert adj.audit['factorization_count']==0
    for prefix,n in (('projection_adjoint',m),('residual_adjoint',p)):
        assert adj.audit[prefix+'_triangular_solve_count']==2
        assert adj.audit[prefix+'_triangular_rhs_count']==2*c
        assert adj.audit[prefix+'_triangular_rhs_element_count']==2*n*c
        assert adj.audit[prefix+'_triangular_dense_work_unit_count']==2*n*n*c
    assert state.audit['deployment_coefficient_bytes']==8*((m+p)*c+c)
    saved=state.arrays['R_N'].copy();args['R_N'].fill(5.)
    np.testing.assert_array_equal(state.arrays['R_N'],saved)
    with pytest.raises(ValueError):state.arrays['alpha'][0,0]=2.
    with pytest.raises(ValueError):state.expansion.alpha[0,0]=2.
    with pytest.raises(TypeError):fit_conditional_affine(**args,labels=np.zeros(p))


@pytest.mark.parametrize('kind',['indefinite','singular','ill_conditioned','negative_schur'])
def test_nonSPD_or_unresolvable_projection_fails_without_jitter(kind):
    O,N,H,priors=fixture();args=blocks(O,N,H,priors)
    if kind=='indefinite':args['A']=np.diag([1.,1.,-1.])
    elif kind=='singular':args['A']=np.diag([1.,1.,0.])
    elif kind=='ill_conditioned':args['A']=np.diag([1.,1.,1e-16])
    else:args['D']=-np.eye(len(N))
    with pytest.raises(NumericalFailure) as caught:fit_conditional_affine(**args)
    audit=caught.value.audit
    assert audit['jitter']==0. and audit['factorization_count']==(1 if kind=='negative_schur' else 0)
    assert audit['completed_factorization_count']==audit['factorization_count']
    assert 'phase' in audit


def test_shapes_dtype_nonfinite_and_no_old_or_new_are_explicit_errors():
    O,N,H,priors=fixture();args=blocks(O,N,H,priors)
    for name,value in [('A',args['A'].astype(np.float32)),('B',args['B'][:2]),('E',args['E'][:,:2]),
                       ('M_H',args['M_H'][:,:3]),('R_N',np.full_like(args['R_N'],np.nan)),('A',np.empty((0,0)))]:
        with pytest.raises(ValueError):fit_conditional_affine(**dict(args,**{name:value}))
    with pytest.raises(ValueError):fit_conditional_affine(**dict(args,D=np.empty((0,0))))
    state=fit_conditional_affine(**args)
    with pytest.raises(ValueError):conditional_affine_adjoint(state,np.ones((len(H),3)))
    with pytest.raises(ValueError):state.expansion.score(k_old=args['F'],k_new=args['E'],M=args['M_H'][:,:3])
    assert state.audit['projection_minimum_reciprocal_condition']==np.sqrt(ROUNDING_MULTIPLIER*EPS*len(O))


def test_empty_held_and_zero_upstream_have_valid_shapes_and_actual_solve_counts():
    O,N,H,priors=fixture();priors['M_H']=np.empty((0,4));state=fit(O,N,H[:0],priors)
    assert state.arrays['held_scores'].shape==(0,4)
    adj=conditional_affine_adjoint(state,np.empty((0,4)))
    for name in ('A','B','D','F','E'):np.testing.assert_array_equal(adj.arrays[name],0.)
    assert adj.audit['projection_adjoint_triangular_rhs_count']==8


def test_sufficient_full_margin_bound_and_old_anchor_wrong_or_tied_preservation():
    O,N,H,priors=fixture();state=fit(O,N,H,priors);a=state.arrays
    norm=np.sqrt(state.audit['residual_norm_squared']);F=H@O.T
    variance=np.sum(H*H,axis=1)-np.sum(F*np.linalg.solve(a['A'],F.T).T,axis=1)+(1-F@a['z'])**2/float(a['s'])
    bound=np.sqrt(2*np.maximum(variance,0))*norm
    target=bound+1.;M=np.column_stack((.75*target,-.25*target,-.25*target,-.25*target))
    scores=state.expansion.score(k_old=F,k_new=H@N.T,M=M)
    residual=state.expansion.residual(k_old=F,k_new=H@N.T)
    assert np.all(np.linalg.norm(residual,axis=1)<=np.sqrt(variance)*norm+1e-11)
    assert np.all(scores.argmax(1)==0)
    final_margin=scores[:,0]-scores[:,1:].max(1)
    assert np.all(final_margin>=target-bound-1e-11)
    # A wrong decision or a tie at an old anchor is preserved, not corrected.
    prior=np.array([[0.,1.,-1.,0.],[0.,0.,0.,0.],[1.,-1.,0.,0.]])
    old=state.expansion.score(k_old=a['A'],k_new=a['B'],M=prior)
    np.testing.assert_allclose(old,prior,atol=3e-12)
    assert old[0].argmax()==1  # No ground-truth-dependent correction.


def test_far_from_old_and_new_gaussian_limit_has_nonzero_v_and_can_flip_margin():
    O=np.array([[-1.],[0.]]);N=np.array([[2.],[3.]]);H=np.array([[100.]])
    def kernel(x,y):return np.exp(-np.sum((x[:,None]-y[None,:])**2,axis=2))
    args=dict(A=kernel(O,O),B=kernel(O,N),D=kernel(N,N),F=kernel(H,O),E=kernel(H,N),
        M_O=np.zeros((2,2)),M_N=np.zeros((2,2)),M_H=np.array([[.001,-.001]]),R_N=np.zeros((2,2)))
    zero=fit_conditional_affine(**args);alpha=np.tile(np.array([-2.,2.]),(2,1))
    args['R_N']=(zero.arrays['K_perp']+np.eye(2))@alpha
    state=fit_conditional_affine(**args);a=state.arrays
    assert np.linalg.norm(a['v'])>1.
    np.testing.assert_array_equal(state.expansion.residual(k_old=args['F'],k_new=args['E']),a['v'][None])
    assert args['M_H'].argmax(1)[0]==0 and a['held_scores'].argmax(1)[0]==1
    sigma2=1.+1./float(a['s'])
    assert .002-np.sqrt(2*sigma2*state.audit['residual_norm_squared'])<0
