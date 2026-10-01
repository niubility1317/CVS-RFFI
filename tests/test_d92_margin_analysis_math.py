"""Independent feature-primal and full-active KKT oracle for archive analysis."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import summarize_d92_margin_joint_probe as analysis
from test_d92_margin_joint_math_certificate import _feature_primal,_constraints


def oracle(phi,M,y,old):
    n,c=M.shape;K=phi@phi.T;R=np.eye(c)[y]-1/c-M
    D,delta,pairs,margins=_constraints(M,y,old)
    primal=_feature_primal(phi,R,M,D,delta);mu=primal['mu']
    V=(D.T@mu).reshape((n,c),order='F')
    A=K+np.eye(n);H=np.block([[A,np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    solved=np.linalg.solve(H,np.vstack((R+V,np.zeros((1,c)))))
    alpha,b=solved[:n],solved[n]
    # The independently optimized feature coefficients determine the function.
    np.testing.assert_allclose(phi.T@alpha,primal['theta'][:-1],atol=3e-11,rtol=2e-11)
    np.testing.assert_allclose(b,primal['theta'][-1],atol=3e-11,rtol=2e-11)
    np.testing.assert_allclose(K@alpha+b,primal['residual'],atol=3e-11,rtol=2e-11)
    inverse=np.linalg.solve(A,np.eye(n));z=inverse@np.ones(n);s=z.sum()
    J=inverse-np.outer(z,z)/s;P=np.eye(n)-J
    alpha0=np.linalg.solve(H,np.vstack((R,np.zeros((1,c)))))
    base=K@alpha0[:n]+alpha0[n]
    W=np.flatnonzero(mu>1e-8)
    Q=D@np.kron(np.eye(c),P)@D.T
    scores=M+K@alpha+b
    return dict(K=K,M=M,labels=y,old_indices=old,R=R,alpha=alpha,b=b,z=z,s=np.asarray(s),
        P_old=P[np.ix_(old,old)],pair_rows=np.repeat(np.arange(len(old)),c-1),
        pair_truth=np.repeat(y[old],c-1),pair_other=np.array([v[2] for v in pairs]),
        margins=margins,delta=delta,s0=D@(M+base).reshape(-1,order='F')-delta,
        multipliers=mu,working_set=W,slack=D@scores.reshape(-1,order='F')-delta,
        train_scores=scores,chol_A=np.linalg.cholesky(A),
        chol_working=np.linalg.cholesky(Q[np.ix_(W,W)]) if len(W) else np.empty((0,0))),primal


def active_fixture(c=2):
    K=np.array([[1.4,.25],[.25,.9]])
    phi=np.linalg.cholesky(K)
    M=np.array([[2.,0.],[.7,0.]]) if c==2 else np.array([[2.,.1,0.],[.7,-.2,0.]])
    y=np.array([0,1 if c==2 else 2],dtype=np.int64);old=np.array([0],dtype=np.int64)
    data,primal=oracle(phi,M,y,old)
    L=np.array([[.3,.8],[-.2,.4]])
    G=np.array([[.7,-.7],[.3,-.3]]) if c==2 else np.array([[.4,-.1,-.3],[.2,.3,-.5]])
    return data,primal,L,G


@pytest.mark.parametrize('classes',[2,3])
def test_feature_primal_matches_independent_archive_primal_dual_all_constraints(classes):
    data,primal,_,_=active_fixture(classes)
    work={};certificate=analysis.verify_margin_head(data,analysis_work=work)
    assert certificate['primal_objective']==pytest.approx(primal['objective'],abs=5e-12)
    assert certificate['dual_objective']==pytest.approx(primal['objective'],abs=5e-12)
    assert len(certificate['W'])==classes-1 and np.min(data['multipliers'])>0
    assert work['independent_general_solve_count']==work['independent_spectral_diagnostic_count']==1
    assert work['independent_general_rhs_column_count']==classes+len(data['old_indices'])


@pytest.mark.parametrize('classes',[2,3])
def test_full_K_L_directional_derivative_nonzero_free_intercept_and_multiplier(classes):
    data,_,L,G=active_fixture(classes);adj=analysis.margin_head_vjp(data,L,G)
    assert np.linalg.norm(adj['g_b'])>0 and np.linalg.norm(adj['W_eta'])>0
    direction=np.array([[.1,-.17],[-.17,.07]]);cross=np.array([[.03,-.09],[.04,.08]])
    eps=2e-6;scores=[]
    for sign in (1.,-1.):
        phi=np.linalg.cholesky(data['K']+sign*eps*direction)
        changed,_=oracle(phi,data['M'],data['labels'],data['old_indices'])
        np.testing.assert_array_equal(changed['working_set'],data['working_set'])
        scores.append((L+sign*eps*cross)@changed['alpha']+changed['b'])
    finite=np.sum(G*(scores[0]-scores[1]))/(2*eps)
    expected=np.sum(adj['K']*direction)+np.sum(adj['L']*cross)
    np.testing.assert_allclose(expected,finite,rtol=3e-7,atol=3e-9)
    # Omitting g_b must change this nonzero-intercept companion.
    n,c=data['alpha'].shape
    H=np.block([[data['K']+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    wrong=np.linalg.solve(H,np.vstack((L.T@G,np.zeros((1,c)))))[:n]
    assert np.linalg.norm(wrong-adj['T_G'])>1e-4


@pytest.mark.parametrize('phi',[np.array([[1.,.2],[1.,.2],[-.3,.4]]),np.zeros((3,0))])
def test_singular_or_zero_K_preserves_unique_function_and_all_physical_rows(phi):
    y=np.array([0,1,1],dtype=np.int64);M=np.zeros((3,2));old=np.array([0],dtype=np.int64)
    data,primal=oracle(phi,M,y,old)
    cert=analysis.verify_margin_head(data)
    np.testing.assert_allclose(cert['scores'],M+primal['residual'],atol=2e-11)
    assert data['alpha'].shape==(3,2) and len(data['slack'])==1
    altered=deepcopy(data);altered['alpha'][0,0]+=.1
    with pytest.raises(ValueError):analysis.verify_margin_head(altered)


@pytest.mark.parametrize('margin',[1.,0.,-1.])
def test_original_positive_tie_and_negative_margins_are_not_clamped(margin):
    phi=np.array([[.8,.1],[-.2,.6],[.3,-.1]])
    M=np.array([[margin,0.],[.2,0.],[0.,.1]])
    y=np.array([0,1,1],dtype=np.int64);old=np.array([0],dtype=np.int64)
    data,_=oracle(phi,M,y,old);analysis.verify_margin_head(data)
    assert data['margins'][0]==margin
    if margin!=0:
        bad=deepcopy(data);bad['margins'][0]=0
        with pytest.raises(ValueError):analysis.verify_margin_head(bad)


def test_weak_tight_region_refuses_unique_Jacobian_without_pinv():
    phi=np.zeros((2,0));M=np.zeros((2,2));y=np.array([0,1],dtype=np.int64);old=np.array([0,1],dtype=np.int64)
    data,_=oracle(phi,M,y,old);analysis.verify_margin_head(data)
    assert np.allclose(data['slack'],0) and len(data['working_set'])==0
    with pytest.raises(ValueError,match='UNSUPPORTED_ACTIVE_JACOBIAN'):
        analysis.margin_head_vjp(data,np.zeros((1,2)),np.array([[.5,-.5]]))


def test_one_physical_row_free_intercept_and_empty_working_factor():
    phi=np.array([[.3,-.4]]);M=np.zeros((1,2));y=np.array([0],dtype=np.int64);old=np.array([0],dtype=np.int64)
    data,_=oracle(phi,M,y,old);cert=analysis.verify_margin_head(data)
    np.testing.assert_allclose(cert['b'],[.5,-.5],atol=1e-12)
    assert data['s'].shape==() and data['chol_working'].shape==(0,0)
    adj=analysis.margin_head_vjp(data,np.array([[.7]]),np.array([[.2,-.2]]))
    np.testing.assert_allclose(adj['K'],0,atol=1e-12)


@pytest.mark.parametrize('key',['b','alpha','delta','margins','multipliers','P_old','slack','chol_A','chol_working'])
def test_every_saved_certificate_field_tampering_is_rejected(key):
    data,_,_,_=active_fixture(3);data=deepcopy(data);data[key].flat[0]+=.01
    with pytest.raises(ValueError):analysis.verify_margin_head(data)


def test_singular_active_factor_is_not_repaired_or_given_a_Jacobian():
    data,_,L,G=active_fixture(3);data=deepcopy(data);data['chol_working'].fill(0)
    with pytest.raises(ValueError):analysis.margin_head_vjp(data,L,G)
