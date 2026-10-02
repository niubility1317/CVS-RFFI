"""Independent synthetic weighted gate oracles; root owns numerical execution."""
if __name__=='__main__':
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),root/'code/cvsrffi/d92_group_balanced_barrier_gate.py'):
        raw=path.read_bytes();source=raw.decode('utf-8',errors='strict')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source
        ast.parse(source,filename=str(path))
    print('UTF8/AST PASS: group-balanced gate/test; no numerical imports')
    raise SystemExit(0)

from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from cvsrffi import d92_group_balanced_barrier_gate as gate
from cvsrffi import d92_group_barrier_gate as original

LIMITS=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=167772160)


def data():
    X=np.array([[1.,.2,-.3],[.8,.4,-.2],[.3,1.,.4],[-.5,.2,1.],
        [.1,-.7,-.4],[.2,-.6,-.3],[-.1,-.8,-.2]])
    labels=np.array([0,0,1,2,3,3,3]);targets=(labels<2).astype(float)
    return X,dict(K=X@X.T,targets=targets,labels=labels,old_indices=np.array([0,1,2]),
        lower_bounds=np.array([[.2,-.3],[.1,.1],[-.1,.05]]))


def fit(args,**limits):
    return gate.fit_group_balanced_barrier_gate(**args,**dict(LIMITS,**limits))


def independent_weights(labels,targets):
    # Class averaging within each physical Bernoulli group, expressed without
    # the factory's unique/inverse indexing algorithm.
    weights=np.zeros(len(labels))
    for group in (0,1):
        classes=sorted(set(labels[targets==group].tolist()))
        for value in classes:
            where=(labels==value)
            weights[where]=len(labels)/2/len(classes)/np.count_nonzero(where)
    return weights


def feature_oracle(X,args):
    weights=independent_weights(args['labels'],args['targets']);t=args['targets']
    old=args['old_indices'];bounds=args['lower_bounds'];zeta=len(t)*gate.AVERAGE_DUAL_GAP/bounds.size
    def objective(v):
        f=X@v[:-1]+v[-1];slack=f[old,None]-bounds
        if np.any(slack<=0):return np.inf
        return .5*float(v[:-1]@v[:-1])+float(weights@np.logaddexp(0.,np.where(t==1,-f,f)))-zeta*np.log(slack).sum()
    def derivative(v):
        f=X@v[:-1]+v[-1]
        q=weights*np.where(t==1,-expit(-f),expit(f));q[old]-=(zeta/(f[old,None]-bounds)).sum(axis=1)
        return np.r_[v[:-1]+X.T@q,q.sum()]
    result=minimize(objective,np.r_[np.zeros(X.shape[1]),bounds.max()+1.],jac=derivative,
        method='SLSQP',constraints=[dict(type='ineq',fun=lambda v:((X@v[:-1]+v[-1])[old,None]-bounds).ravel())],
        options=dict(ftol=1e-12,maxiter=1000))
    assert result.success,result.message
    return result


def test_unequal_class_counts_group_mass_and_independent_feature_primal():
    X,args=data();state=fit(args);weights=independent_weights(args['labels'],args['targets'])
    np.testing.assert_allclose(state.weights,weights,rtol=2e-16,atol=0)
    for group in (0,1):np.testing.assert_allclose(state.weights[args['targets']==group].sum(),len(X)/2,rtol=2e-16)
    oracle=feature_oracle(X,args)
    np.testing.assert_allclose(X.T@state.alpha,oracle.x[:-1],rtol=4e-6,atol=4e-6)
    np.testing.assert_allclose(state.b,oracle.x[-1],rtol=4e-6,atol=4e-6)
    np.testing.assert_allclose(state.audit['objective'],oracle.fun,atol=2e-9,rtol=0)
    q=state.weights*np.where(args['targets']==1,-expit(-state.f),expit(state.f))
    q[args['old_indices']]-=(state.zeta/state.slacks).sum(axis=1)
    D=state.weights*expit(state.f)*expit(-state.f)
    D[args['old_indices']]+=(state.zeta/state.slacks**2).sum(axis=1)
    np.testing.assert_allclose(state.qeff,q,atol=4e-16,rtol=2e-15)
    np.testing.assert_allclose(state.D_eff,D,atol=4e-16,rtol=2e-15)
    assert state.slacks.shape==args['lower_bounds'].shape and np.all(state.slacks>0)
    assert state.zeta==len(X)*1e-4/args['lower_bounds'].size
    assert state.audit['weight_constructions_completed']==1
    assert state.audit['weighted_logistic_record_evaluations']==state.audit['objective_evaluations']*len(X)
    assert state.audit['weighted_entropy_record_evaluations']==len(X)
    # Weighted conjugate entropy is necessary for the candidate dual value.
    p=expit(state.f);entropy=p*np.log(p)+(1-p)*np.log1p(-p)
    dual=-.5*float((-state.qeff)@(args['K']@(-state.qeff)))-float(weights@entropy)+float(np.sum((state.zeta/state.slacks)*args['lower_bounds']))
    np.testing.assert_allclose(state.audit['candidate_dual_objective'],dual,rtol=2e-15,atol=2e-15)
    np.testing.assert_allclose(state.audit['candidate_primal_dual_gap'],len(X)*1e-4,atol=2e-9)


def test_complete_weighted_JVP_VJP_two_kernel_ends_all_bounds_free_intercept():
    X,args=data();state=fit(args)
    rng=np.random.default_rng(4507);r=2
    dX=rng.normal(scale=.1,size=X.shape+(r,));dk=np.empty(args['K'].shape+(r,))
    for j in range(r):dk[...,j]=dX[...,j]@X.T+X@dX[...,j].T
    L=rng.normal(scale=.2,size=(3,len(X)));dl=rng.normal(scale=.1,size=L.shape+(r,))
    dbounds=rng.normal(scale=.1,size=args['lower_bounds'].shape+(r,));G=np.array([.4,-.2,.9])
    jvp=gate.group_balanced_barrier_gate_jvp(state,K_jacobian=dk,L=L,L_jacobian=dl,bounds_jacobian=dbounds)
    vjp=gate.group_balanced_barrier_gate_vjp(state,L=L,G=G)
    assert G.sum()!=0 and np.linalg.norm(vjp.gradlower_bounds)>0
    for j in range(r):
        eps=2e-5;changed=[]
        for sign in (-1,1):
            xx=X+sign*eps*dX[...,j]
            s=fit(dict(args,K=xx@xx.T,lower_bounds=args['lower_bounds']+sign*eps*dbounds[...,j]))
            changed.append(gate.predict_group_balanced_barrier_gate(s,L=L+sign*eps*dl[...,j]))
        finite=(changed[1]-changed[0])/(2*eps)
        np.testing.assert_allclose(jvp.score_jacobian[:,j],finite,atol=4e-6,rtol=4e-5)
        contraction=np.sum(vjp.gradK*dk[...,j])+np.sum(vjp.gradL*dl[...,j])+np.sum(vjp.gradlower_bounds*dbounds[...,j])
        np.testing.assert_allclose(G@jvp.score_jacobian[:,j],contraction,atol=2e-10,rtol=2e-10)
    assert np.array_equal(vjp.gradK,vjp.gradK.T)
    assert jvp.audit['factorizations_completed']==1 and jvp.audit['triangular_rhs_columns']==2*(r+1)
    assert jvp.audit['weight_constructions_completed']==vjp.audit['weight_constructions_completed']==0
    assert abs(vjp.audit['intercept_adjoint'])>1e-7
    # Each separate tied or untied physical constraint remains differentiated.
    for i,j in np.ndindex(args['lower_bounds'].shape):
        eps=2e-5;unit=np.zeros_like(args['lower_bounds']);unit[i,j]=eps
        plus=gate.predict_group_balanced_barrier_gate(fit(dict(args,lower_bounds=args['lower_bounds']+unit)),L=L)
        minus=gate.predict_group_balanced_barrier_gate(fit(dict(args,lower_bounds=args['lower_bounds']-unit)),L=L)
        np.testing.assert_allclose(vjp.gradlower_bounds[i,j],G@(plus-minus)/(2*eps),atol=4e-6,rtol=4e-5)


def test_unit_weights_equivalence_and_original_globals_untouched():
    X=np.array([[1.,.2],[.3,1.],[-.5,.2],[.1,-.7]])
    args=dict(K=X@X.T,targets=np.array([1.,1.,0.,0.]),labels=np.arange(4),
        old_indices=np.array([0,1]),lower_bounds=np.array([[.2,-.3],[.1,.1]]))
    new=fit(args);old=original.fit_group_barrier_gate(**{k:v for k,v in args.items() if k!='labels'},**LIMITS)
    assert np.array_equal(new.weights,np.ones(4))
    np.testing.assert_allclose(new.alpha,old.alpha,atol=3e-12,rtol=3e-12)
    np.testing.assert_allclose(new.b,old.b,atol=3e-12,rtol=3e-12)
    assert original.fit_group_barrier_gate.__globals__['_evaluate'] is original._evaluate
    assert gate._factor is original._factor and gate._solve is original._solve


def test_zero_kernel_six_old_one_new_free_b_not_sample_count_prior():
    labels=np.arange(7);args=dict(K=np.zeros((7,7)),labels=labels,targets=(labels<6).astype(float),
        old_indices=np.arange(6),lower_bounds=np.full((6,1),-30.))
    state=fit(args)
    np.testing.assert_allclose(state.weights[:6],7/12,rtol=0,atol=0)
    assert state.weights[-1]==3.5 and 0<state.b<.01
    assert np.linalg.norm(state.alpha)>0 and np.all(state.slacks>0)
    assert state.audit['last_invalid_trial_code']=='LOGISTIC_CURVATURE_UNDERFLOW'
    assert state.audit['last_invalid_trial_step_size']>0
    assert state.audit['line_search_trials']>state.audit['accepted_steps']
    assert state.audit['canonical_residual']<=state.audit['stationarity_tolerance']
    result=gate.group_balanced_barrier_gate_vjp(state,L=np.zeros((1,7)),G=np.ones(1))
    assert np.isfinite(result.gradlower_bounds).all()


def test_saturated_trial_consumes_budget_without_changing_current_state():
    labels=np.arange(7);args=dict(K=np.zeros((7,7)),labels=labels,targets=(labels<6).astype(float),
        old_indices=np.arange(6),lower_bounds=np.full((6,1),-30.))
    with pytest.raises(gate.GroupBalancedBarrierFailure) as caught:
        fit(args,max_line_search_trials=1)
    failed=caught.value
    assert failed.code=='LINE_SEARCH_LIMIT'
    assert failed.audit['last_invalid_trial_code']=='LOGISTIC_CURVATURE_UNDERFLOW'
    assert failed.audit['last_invalid_trial_step_size']==1.
    assert failed.audit['line_search_trials']==1 and failed.audit['accepted_steps']==0
    assert failed.audit['objective_evaluations']==2
    assert failed.audit['weighted_logistic_record_evaluations']==14
    assert failed.audit['factorizations_completed']==1 and failed.audit['triangular_calls']==2
    np.testing.assert_array_equal(failed.arrays['alpha'],np.zeros(7))
    assert failed.arrays['b'].shape==() and float(failed.arrays['b'])==-29.
    np.testing.assert_array_equal(failed.arrays['f'],np.full(7,-29.))
    np.testing.assert_array_equal(failed.arrays['slacks'],np.ones((6,1)))


def test_failure_keeps_actual_weight_work_and_immutable_arrays():
    _,args=data()
    with pytest.raises(gate.GroupBalancedBarrierFailure) as caught:fit(args,max_factor_buffer_bytes=1)
    failed=caught.value
    assert failed.audit['weight_constructions_completed']==1 and failed.audit['spectral_checks']==1
    assert failed.audit['factorization_attempts']==0
    np.testing.assert_allclose(failed.arrays['weights'],independent_weights(args['labels'],args['targets']))
    with pytest.raises(ValueError):failed.arrays['weights'].setflags(write=True)
    state=fit(args)
    with pytest.raises(FrozenInstanceError):state.weights=np.ones(7)
    with pytest.raises(ValueError):state.weights.setflags(write=True)


@pytest.mark.parametrize('fault',['float_labels','mixed_class','missing_new_bound','wrong_old'])
def test_train_only_class_and_group_input_guards(fault):
    _,args=data()
    if fault=='float_labels':args['labels']=args['labels'].astype(float)
    elif fault=='mixed_class':args['labels']=np.zeros(len(args['labels']),dtype=int)
    elif fault=='missing_new_bound':args['lower_bounds']=args['lower_bounds'][:,:1]
    else:args['old_indices']=args['old_indices'][:2]
    with pytest.raises((ValueError,gate.GroupBalancedBarrierFailure)):fit(args)
