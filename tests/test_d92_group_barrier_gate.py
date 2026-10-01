"""Synthetic-only numerical contracts; execution belongs to the root test owner."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from cvsrffi import d92_group_barrier_gate as gate

from cvsrffi.d92_group_barrier_gate import (
    AVERAGE_DUAL_GAP, GroupBarrierFailure, fit_group_barrier_gate,
    predict_group_barrier_gate, group_barrier_gate_vjp,
)


LIMITS = dict(max_newton_iterations=100, max_line_search_trials=60,
              max_factor_buffer_bytes=1_000_000)  # Artificial test capacity only.


def data():
    X = np.array([[1., .2, -.3], [.3, 1., .4], [-.5, .2, 1.], [.1, -.7, -.4]])
    return X, dict(K=X@X.T, targets=np.array([1.,1.,0.,0.]),
        old_indices=np.array([0,1]), lower_bounds=np.array([[.2,-.3],[.1,.1]]))


def fit(args, **limits):
    return fit_group_barrier_gate(**args, **dict(LIMITS, **limits))


def independent_feature_oracle(X, args):
    """Optimize w,b directly, without canonical-alpha Newton or kernel solves."""
    t, old, a = args['targets'], args['old_indices'], args['lower_bounds']
    zeta = len(t)*AVERAGE_DUAL_GAP/a.size
    def objective(theta):
        w,b = theta[:-1],theta[-1]
        f = X@w+b
        slack = f[old,None]-a
        if np.any(slack <= 0):
            return np.inf
        return .5*(w@w)+np.logaddexp(0.,np.where(t==1,-f,f)).sum()-zeta*np.log(slack).sum()
    def jacobian(theta):
        w,b = theta[:-1],theta[-1]
        f = X@w+b
        q = np.where(t==1,-expit(-f),expit(f))
        q[old] -= (zeta/(f[old,None]-a)).sum(axis=1)
        return np.r_[w+X.T@q,q.sum()]
    def slack(theta):
        return ((X@theta[:-1]+theta[-1])[old,None]-a).ravel()
    start=np.r_[np.zeros(X.shape[1]),a.max()+1.]
    result=minimize(objective,start,jac=jacobian,method='SLSQP',
        constraints=[dict(type='ineq',fun=slack)],
        options=dict(ftol=1e-12,maxiter=1000))
    assert result.success, result.message
    return result


def test_forward_matches_independent_finite_feature_optimum_and_full_certificate():
    X,args=data();state=fit(args);oracle=independent_feature_oracle(X,args)
    np.testing.assert_allclose(X.T@state.alpha,oracle.x[:-1],atol=3e-6,rtol=3e-6)
    np.testing.assert_allclose(state.b,oracle.x[-1],atol=3e-6,rtol=3e-6)
    np.testing.assert_allclose(state.audit['objective'],oracle.fun,atol=1e-9)
    np.testing.assert_allclose(state.f,args['K']@state.alpha+state.b,atol=0,rtol=0)
    np.testing.assert_allclose(state.slacks,state.f[args['old_indices'],None]-args['lower_bounds'],atol=0,rtol=0)
    assert np.all(state.slacks>0)
    assert abs(state.alpha.sum()) <= state.audit['intercept_tolerance']
    assert np.max(np.abs(state.alpha+state.qeff)) <= state.audit['stationarity_tolerance']
    multipliers=state.zeta/state.slacks
    assert np.all(multipliers>0)
    np.testing.assert_allclose(multipliers*state.slacks,state.zeta,atol=1e-18)
    assert state.audit['theoretical_center_path_gap']==args['lower_bounds'].size*state.zeta
    np.testing.assert_allclose(state.audit['complementarity_gap'],len(state.alpha)*1e-4,atol=1e-18)
    np.testing.assert_allclose(state.audit['candidate_primal_dual_gap'],len(state.alpha)*1e-4,atol=1e-9)
    assert state.audit['candidate_dual_box_feasible']
    assert state.audit['dual_intercept_feasibility_residual']<=state.audit['intercept_tolerance']
    assert state.audit['factorization_attempts']==state.audit['factorizations_completed']
    assert state.audit['triangular_calls']==2*state.audit['factorizations_completed']
    assert state.audit['triangular_rhs_columns']==4*state.audit['factorizations_completed']
    assert state.audit['spectral_checks']==1


def test_full_VJP_including_symmetric_K_L_all_bounds_and_free_intercept():
    X,args=data();state=fit(args)
    L=np.array([[.7,-.3,.2,.5],[-.1,.4,.8,.2],[.2,.1,-.2,.3]])
    G=np.array([.4,-.2,.9])  # Nonzero sum specifically tests free b.
    vjp=group_barrier_gate_vjp(state,L=L,G=G)
    dX=np.array([[.2,-.1,.3],[-.2,.1,.1],[.1,.3,-.2],[.2,.2,.1]])
    dK=dX@X.T+X@dX.T
    dL=np.array([[.1,.2,-.1,.3],[.4,-.2,.1,.2],[-.2,.3,.1,.4]])
    da=np.array([[.3,-.2],[.1,.4]])
    step=2e-5
    def scalar(sign):
        XX=X+sign*step*dX
        changed=dict(args,K=XX@XX.T,lower_bounds=args['lower_bounds']+sign*step*da)
        return float(G@predict_group_barrier_gate(fit(changed),L=L+sign*step*dL))
    finite=(scalar(1)-scalar(-1))/(2*step)
    analytic=float(np.sum(vjp.gradK*dK)+np.sum(vjp.gradL*dL)+np.sum(vjp.gradlower_bounds*da))
    np.testing.assert_allclose(analytic,finite,atol=3e-6,rtol=3e-5)
    assert np.array_equal(vjp.gradK,vjp.gradK.T)
    assert vjp.gradlower_bounds.shape==args['lower_bounds'].shape
    # Separate every bound endpoint, including tied bounds; no max selection.
    for i,j in np.ndindex(args['lower_bounds'].shape):
        basis=np.zeros_like(args['lower_bounds']);basis[i,j]=step
        plus=fit(dict(args,lower_bounds=args['lower_bounds']+basis))
        minus=fit(dict(args,lower_bounds=args['lower_bounds']-basis))
        numerical=float(G@(predict_group_barrier_gate(plus,L=L)-predict_group_barrier_gate(minus,L=L)))/(2*step)
        np.testing.assert_allclose(vjp.gradlower_bounds[i,j],numerical,atol=3e-6,rtol=3e-5)
    assert vjp.audit['adjoint_residual']<=vjp.audit['adjoint_tolerance']
    assert abs(vjp.audit['intercept_adjoint'])>1e-6
    assert abs(np.sum(vjp.gradlower_bounds*da))>1e-6  # Detaching constraints fails this direction.


@pytest.mark.parametrize('zero', [False,True])
def test_singular_duplicate_physical_records_tied_bounds_and_constant_kernel(zero):
    X=np.array([[1.,.3],[1.,.3],[-.2,.5],[-.2,.5]])
    K=np.zeros((4,4)) if zero else X@X.T
    args=dict(K=K,targets=np.array([1.,1.,0.,0.]),old_indices=np.array([0,1]),
              lower_bounds=np.array([[.2,.2],[.2,.2]]))
    state=fit(args)
    assert state.alpha.shape==(4,) and state.slacks.shape==(2,2)
    assert np.all(state.slacks>0) and state.audit['constraint_count']==4
    np.testing.assert_allclose(state.f[0],state.f[1],atol=0)
    if zero:
        np.testing.assert_allclose(state.f,state.b,atol=0)
        assert state.b>.2 and np.linalg.norm(state.alpha)>0
    vjp=group_barrier_gate_vjp(state,L=np.zeros((1,4)),G=np.ones(1))
    assert np.all(np.isfinite(vjp.gradlower_bounds))
    np.testing.assert_allclose(vjp.gradlower_bounds[:,0],vjp.gradlower_bounds[:,1],atol=0)


def test_single_new_class_and_label_aware_extreme_logits():
    _,args=data();args['lower_bounds']=args['lower_bounds'][:,:1]
    state=fit(args)
    assert state.slacks.shape==(2,1)
    # K=0, high but representable finite logits: p-1 would erase old residual.
    high=dict(K=np.zeros((2,2)),targets=np.array([1.,0.]),old_indices=np.array([0]),
              lower_bounds=np.array([[40.]]))
    extreme=fit(high)
    assert np.all(extreme.D_eff>0) and extreme.f[0]>40.
    assert extreme.qeff[0]<0 and extreme.audit['minimum_slack']>0


@pytest.mark.parametrize('field,value', [('max_newton_iterations',True),('max_line_search_trials',0),
                                       ('max_factor_buffer_bytes',None)])
def test_explicit_positive_integer_guards(field,value):
    _,args=data()
    with pytest.raises(ValueError):fit(args,**{field:value})


@pytest.mark.parametrize('bad', ['asymmetric','nonfinite','wrong_old','one_group','empty_bounds'])
def test_invalid_input_rejected(bad):
    _,args=data()
    if bad=='asymmetric':args['K'][0,1]+=.01
    elif bad=='nonfinite':args['K'][0,0]=np.nan
    elif bad=='wrong_old':args['old_indices']=np.array([0,2])
    elif bad=='one_group':args['targets'][:]=1
    else:args['lower_bounds']=np.empty((2,0))
    with pytest.raises(ValueError):fit(args)


@pytest.mark.parametrize('reason', ['factor','iteration','underflow','nonpsd'])
def test_explicit_technical_failure_preserves_current_state_and_partial_work(reason):
    _,args=data();limits={}
    if reason=='factor':limits['max_factor_buffer_bytes']=1
    elif reason=='iteration':limits['max_newton_iterations']=1
    elif reason=='underflow':args['lower_bounds'][:]=1000.
    else:args['K']=-np.eye(4)
    with pytest.raises(GroupBarrierFailure) as caught:fit(args,**limits)
    failure=caught.value
    expected=dict(factor='FACTOR_BUFFER_LIMIT',iteration='NEWTON_ITERATION_LIMIT',
                  underflow='LOGISTIC_CURVATURE_UNDERFLOW',nonpsd='K_NOT_PSD')
    assert failure.code==expected[reason]
    assert failure.audit['status']=='TECHNICAL_FAILURE'
    saved=failure.arrays
    np.testing.assert_allclose(saved['slacks'],(saved['K']@saved['alpha']+saved['b'])[saved['old_indices'],None]-saved['lower_bounds'],atol=0,rtol=0)
    assert failure.audit['spectral_checks']==1
    if reason=='iteration':
        assert failure.audit['factorizations_completed']==1 and failure.audit['line_search_trials']>=1
    if reason=='underflow':assert failure.audit['logistic_record_evaluations']==4
    if reason=='factor':assert failure.audit['factorization_attempts']==0


def test_outputs_are_sealed_no_input_alias_and_predictions_are_per_record():
    _,args=data();state=fit(args);original=state.K.copy();args['K'][:]=0
    np.testing.assert_array_equal(state.K,original)
    L=np.array([[.2,.4,.3,.1],[.7,-.2,.1,.4]])
    scores=predict_group_barrier_gate(state,L=L)
    np.testing.assert_allclose(scores,np.r_[predict_group_barrier_gate(state,L=L[:1]),
                                          predict_group_barrier_gate(state,L=L[1:])],atol=1e-14)
    for array in (state.K,state.alpha,state.slacks,state.targets,state.lower_bounds,scores):
        with pytest.raises(ValueError):array.setflags(write=True)
    with pytest.raises(TypeError):state.audit['status']='edited'
    vjp=group_barrier_gate_vjp(state,L=L,G=np.ones(2))
    for array in (vjp.gradK,vjp.gradL,vjp.gradlower_bounds):
        with pytest.raises(ValueError):array.setflags(write=True)


def test_failed_factor_and_rejected_trials_are_charged_without_state_mixing(monkeypatch):
    _,args=data()
    original=gate.cholesky
    def broken(*args,**kwargs):
        raise np.linalg.LinAlgError('synthetic factor failure')
    monkeypatch.setattr(gate,'cholesky',broken)
    with pytest.raises(GroupBarrierFailure) as caught:fit(args)
    assert caught.value.audit['factorization_attempts']==1
    assert caught.value.audit['factorizations_completed']==0
    assert caught.value.audit['objective_evaluations']==1
    monkeypatch.setattr(gate,'cholesky',original)
    evaluate=gate._evaluate
    def reject(*values):
        result=evaluate(*values)
        if values[-1]['line_search_trials']:
            result['objective']+=1e10  # Deterministic simulated objective readback rejection.
        return result
    monkeypatch.setattr(gate,'_evaluate',reject)
    with pytest.raises(GroupBarrierFailure,match='LINE_SEARCH_LIMIT') as caught:
        fit(args,max_line_search_trials=2)
    audit=caught.value.audit
    assert audit['factorizations_completed']==1 and audit['line_search_trials']==2
    assert audit['objective_evaluations']==3 and audit['accepted_steps']==0
    np.testing.assert_array_equal(caught.value.arrays['alpha'],0.)
    assert float(caught.value.arrays['b'])==args['lower_bounds'].max()+1.


def test_literal_seed7201_gate_objective_resolution_stagnation_regression():
    """Literal synthetic head only, from the fixed seed7201 joint fixture.

    No real data and no external fixture path at test runtime. Reproduces
    canonical residual stagnation from objective differences below eps.
    """
    K=np.array((4.584378686806434, 1.6522239897592599, 1.609219822243154, 1.5324503817865056, 1.5713431507600772, 1.6315969177259575, 1.5273436330377275, 1.666899354512403, 1.5695534772038535, 1.6613584128175025, 1.6080216543651848, 1.5052245705431087, 1.5696075070237323, 1.5048393634544603, 1.6465832999604735, 1.5169755514044505, 1.6522239897592599, 4.584378686806434, 1.5634662695502537, 1.61426137204519, 1.4680380483003241, 1.5019322714455923, 1.591744886109906, 1.5345383990957502, 1.5094692004721615, 1.6773771274298954, 1.579910773383677, 1.58262214206739, 1.607239237908816, 1.5679287724659194, 1.6659913910292137, 1.449949452890891, 1.609219822243154, 1.5634662695502537, 4.584378686806434, 1.6251512839355853, 1.58252246463719, 1.6487812007547002, 1.5868339211120148, 1.6193431858416, 1.5283219835709556, 1.557126920288495, 1.6277057162957846, 1.6773381579604585, 1.6406873686246317, 1.5766622681678055, 1.5851519668666898, 1.6390600645565083, 1.5324503817865056, 1.61426137204519, 1.6251512839355853, 4.584378686806434, 1.6096116023193117, 1.581630262405138, 1.5730098823930279, 1.5607754096621602, 1.529600551607855, 1.4894990545803546, 1.6631839300361853, 1.596339500612106, 1.457887867716204, 1.5596285703522021, 1.6567129550693886, 1.5497996571549548, 1.5713431507600772, 1.4680380483003241, 1.58252246463719, 1.6096116023193117, 4.584378686806434, 1.7522365071744137, 1.4926144537057036, 1.659796960662472, 1.503403273402392, 1.4856095810910581, 1.5792214802798383, 1.6993073106431065, 1.613454008544924, 1.821054208957774, 1.573302756226929, 1.7238128625363278, 1.6315969177259575, 1.5019322714455923, 1.6487812007547002, 1.581630262405138, 1.7522365071744137, 4.584378686806434, 1.4897801050944504, 1.665073223692438, 1.5798631374728276, 1.5906007912752365, 1.5606077014856852, 1.527310220515329, 1.5797745556581753, 1.737573218448419, 1.5366518383101277, 1.5949304050941797, 1.5273436330377275, 1.591744886109906, 1.5868339211120148, 1.5730098823930279, 1.4926144537057036, 1.4897801050944504, 4.584378686806434, 1.7276447152504548, 1.5700877119559775, 1.5665957183579626, 1.6194755634862898, 1.658049199931539, 1.7131104147325276, 1.5912008352526956, 1.5287699815212337, 1.6407507622605675, 1.666899354512403, 1.5345383990957502, 1.6193431858416, 1.5607754096621602, 1.659796960662472, 1.665073223692438, 1.7276447152504548, 4.584378686806434, 1.5477410653410195, 1.5755368754596122, 1.5981773977401768, 1.6766113160498233, 1.6863134133394155, 1.7312568443136223, 1.580864501646881, 1.7312357582022029, 1.5695534772038535, 1.5094692004721615, 1.5283219835709556, 1.529600551607855, 1.503403273402392, 1.5798631374728276, 1.5700877119559775, 1.5477410653410195, 4.584378686806434, 1.6430996298218747, 1.5559120975527445, 1.6464558717546542, 1.591938716792877, 1.5583365856566125, 1.5364834567098227, 1.5411355965237092, 1.6613584128175025, 1.6773771274298954, 1.557126920288495, 1.4894990545803546, 1.4856095810910581, 1.5906007912752365, 1.5665957183579626, 1.5755368754596122, 1.6430996298218747, 4.584378686806434, 1.641440869406614, 1.5898104254029795, 1.695394451497181, 1.5375835464633663, 1.5760214720021144, 1.5993056447954397, 1.6080216543651848, 1.579910773383677, 1.6277057162957846, 1.6631839300361853, 1.5792214802798383, 1.5606077014856852, 1.6194755634862898, 1.5981773977401768, 1.5559120975527445, 1.641440869406614, 4.584378686806434, 1.688635373927926, 1.563078055713289, 1.4782878251280107, 1.639318182712276, 1.7022104673565883, 1.5052245705431087, 1.58262214206739, 1.6773381579604585, 1.596339500612106, 1.6993073106431065, 1.527310220515329, 1.658049199931539, 1.6766113160498233, 1.6464558717546542, 1.5898104254029795, 1.688635373927926, 4.584378686806434, 1.6646775666529106, 1.6308591074397805, 1.6207036376488042, 1.6795228960869817, 1.5696075070237323, 1.607239237908816, 1.6406873686246317, 1.457887867716204, 1.613454008544924, 1.5797745556581753, 1.7131104147325276, 1.6863134133394155, 1.591938716792877, 1.695394451497181, 1.563078055713289, 1.6646775666529106, 4.584378686806434, 1.590409004820607, 1.6071779570996358, 1.61484681952471, 1.5048393634544603, 1.5679287724659194, 1.5766622681678055, 1.5596285703522021, 1.821054208957774, 1.737573218448419, 1.5912008352526956, 1.7312568443136223, 1.5583365856566125, 1.5375835464633663, 1.4782878251280107, 1.6308591074397805, 1.590409004820607, 4.584378686806434, 1.589769733966647, 1.632789689442096, 1.6465832999604735, 1.6659913910292137, 1.5851519668666898, 1.6567129550693886, 1.573302756226929, 1.5366518383101277, 1.5287699815212337, 1.580864501646881, 1.5364834567098227, 1.5760214720021144, 1.639318182712276, 1.6207036376488042, 1.6071779570996358, 1.589769733966647, 4.584378686806434, 1.5651499391275114, 1.5169755514044505, 1.449949452890891, 1.6390600645565083, 1.5497996571549548, 1.7238128625363278, 1.5949304050941797, 1.6407507622605675, 1.7312357582022029, 1.5411355965237092, 1.5993056447954397, 1.7022104673565883, 1.6795228960869817, 1.61484681952471, 1.632789689442096, 1.5651499391275114, 4.584378686806434)).reshape((16, 16))
    targets=np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)).reshape((16,))
    old_indices=np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)).reshape((12,))
    lower_bounds=np.array((1.1320717044425113, 1.1576696478127668, 1.1514094374667676, 1.1392168438464885, 1.1419062958999224, 1.146921302727913, 1.1190125628917826, 1.169995518049854, 1.1615460224526377, 1.1300941863207663, 1.1674417663336083, 1.1235654385582954, 1.160899097621591, 1.1300539871771251, 1.1564612418819167, 1.1331157218424541, 1.153158095920432, 1.1379100365031867, 1.1507482437202872, 1.1393416734030946, 1.103892808869527, 1.1833684565120968, 1.1419165497998423, 1.1463872557627297)).reshape((12, 2))
    state=fit_group_barrier_gate(K=K,targets=targets,old_indices=old_indices,lower_bounds=lower_bounds,
        max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=16_000_000)
    audit=state.audit
    assert audit["status"]=="COMPLETE"
    assert audit["round_off_residual_acceptances"]>=1
    assert audit["canonical_residual"]<=audit["stationarity_tolerance"]
    assert audit["intercept_residual"]<=audit["intercept_tolerance"]
    assert audit["primal_gradient_residual"]<=audit["primal_gradient_tolerance"]
    assert np.all(state.slacks>0)
    assert audit["last_acceptance_rule"]=="ROUND_OFF_RESIDUAL_ACCEPTANCE"
    assert audit["last_predicted_decrease"]<=audit["last_objective_roundoff_bound"]
    assert audit["last_objective_increase"]<=audit["last_objective_roundoff_bound"]
    assert audit["last_residual_merit_after"]<audit["last_residual_merit_before"]
    assert audit["last_objective_increased"] is (audit["last_objective_increase"]>0)
