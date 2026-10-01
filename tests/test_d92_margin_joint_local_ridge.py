"""Synthetic joint correctness only. Direct execution performs static checks."""
if __name__ == '__main__':
    import ast
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    paths = (root/'code/cvsrffi/d92_margin_joint_local_ridge.py',Path(__file__),
             root/'docs/D92_MARGIN_JOINT_IMPLEMENTATION_20261001.md')
    for path in paths:
        raw = path.read_bytes()
        source = raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source and '\r' not in source
        if path.suffix=='.py':
            ast.parse(source,filename=str(path))
    print(json.dumps(dict(status='STATIC_CHECKS_ONLY',checked=[str(path) for path in paths]),ensure_ascii=False))
    raise SystemExit(0)

from dataclasses import replace
import json
from types import MappingProxyType

import numpy as np
import pytest

from cvsrffi import d92_margin_joint_local_ridge as mj
from cvsrffi import d92_margin_qp_head as qp
from cvsrffi import d92_affine_joint_local_ridge as af
from cvsrffi import d92_conditional_joint_local_ridge as cj
from test_d92_margin_joint_math_certificate import _feature_primal, _constraints


# Artificial per-head limits for these small fixtures; not a proposed run budget.
LIMITS = dict(max_transitions=128,max_factor_buffer_bytes=1_000_000)


def fixture(k=3,c=3,seed=341):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(c),k)
    raw = {}
    for name,width in zip(mj._NAMES,(160,96,160,160,160)):
        table = rng.normal(size=(c*k,width))
        table[:,0] += labels*.4
        table[:,1] += (labels==0)*.6
        raw[name] = table
    classes = tuple('c'+str(i) for i in range(c))
    ids = tuple(f'synthetic-physical-{i}-{j:02}' for i in range(c) for j in range(k))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=classes,old_classes=classes)


def scope(**extra):
    return dict(run_id='synthetic-margin-only',row_id='row-one',split_id='synthetic-split',scope='synthetic_support',**extra)


def old_args(args):
    mask = args['support_labels']<2
    result = {name:args[name][mask] for name in mj._NAMES}
    result.update(support_labels=args['support_labels'][mask],
        support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),
        classes=('c0','c1'),old_classes=('c0','c1'))
    return result


@pytest.fixture(scope='module')
def sequence():
    args = fixture()
    oa = old_args(args)
    bp = mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope())
    b = mj.fit_margin_joint_local_ridge(bp)
    args['old_classes'] = ('c0','c1')
    cp = mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope())
    c = mj.fit_margin_joint_local_ridge(cp,mode='C_seq')
    return bp,b,cp,c,oa,args


def test_public_schema_readonly_numeric_state_and_unchanged_existing_methods(sequence):
    bp,b,cp,c,_,_ = sequence
    assert mj.FROZEN_CONFIG['schema']=='d92_margin_joint_local_ridge_v1'
    assert mj.FROZEN_CONFIG['method']=='D92-MarginJointLocalRidge-v1'
    assert mj.FROZEN_CONFIG['proximal_coefficient']==0 and mj.FROZEN_CONFIG['coordinate_ball_radius']==.5
    assert af.FROZEN_CONFIG['objective'].endswith('+0.5*Z_frobenius_squared')
    assert cj.FROZEN_CONFIG['C_head'].endswith('old_residual_zero_all_registered_columns')
    assert b.labels.dtype==c.labels.dtype==cp.problems[0].held_labels.dtype==np.int64
    assert cp.full_problem.old_indices.dtype==np.int64
    assert isinstance(c.final_cache['numeric'],MappingProxyType)
    for array in (b.U,c.Z,c.final_cache['numeric']['s'],c.final_cache['numeric']['multipliers']):
        with pytest.raises(ValueError):
            array[...] = 1
    assert c.final_cache['numeric']['s'].shape==()
    audit = json.loads(json.dumps(cp.audit_dict(),allow_nan=False))
    assert audit['classes']==list(cp.classes) and audit['training_physical_ids']==list(cp.ids)
    assert audit['actual_parameters']['qp_max_transitions']==LIMITS['max_transitions']


def test_resource_limits_are_explicit_without_experiment_defaults():
    args = old_args(fixture())
    with pytest.raises(TypeError):
        mj.prepare_margin_joint_training(**args,context=scope())
    for name in LIMITS:
        invalid = dict(LIMITS)
        invalid[name] = True
        with pytest.raises(ValueError,match='explicit positive integer'):
            mj.prepare_margin_joint_training(**args,**invalid,context=scope())


def test_actual_current_B_anchor_frozen_prior_mapping_and_inner_train_isolation(sequence):
    _,b,p,_,_,args = sequence
    old_columns = [p.classes.index(name) for name in b.classes]
    np.testing.assert_array_equal(p.anchor_U,b.U)
    np.testing.assert_array_equal(p.full_problem.M_train[:,old_columns],b.score(**p.raw))
    new_columns = [i for i,name in enumerate(p.classes) if name not in b.classes]
    np.testing.assert_array_equal(p.full_problem.M_train[:,new_columns],0.)
    for problem,prior in zip(p.problems,p.audit['prior_folds']):
        train = set(prior['training_physical_ids'])
        held = set(prior['held_physical_ids'])
        assert train==set(problem.audit['old_reference_physical_ids']) and not train&held
        assert train|held==set(b.ids)
        assert train.isdisjoint(problem.audit['held_physical_ids'])
        assert problem.audit['prior_ref']==prior['head_state_ref']
        arrays = p.records[prior['head_state_ref']['key']]
        assert len(arrays['train_labels'])==len(train)
        np.testing.assert_array_equal(arrays['U'],b.U)
    before = [np.array(pr.M_train,copy=True) for pr in p.problems]
    mj.evaluate_margin_joint_objective(p,np.full((736,p.r),.0001),gradient=False)
    for original,problem in zip(before,p.problems):
        np.testing.assert_array_equal(problem.M_train,original)
    for key in ('run_id','row_id','split_id','scope'):
        bad = scope()
        bad[key] = 'different'
        with pytest.raises(ValueError,match='crosses'):
            mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=bad)
    with pytest.raises(ValueError,match='explicit'):
        mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b)
    bad = dict(args,z_id=args['z_id'].copy())
    bad['z_id'][0,0] += .01
    with pytest.raises(ValueError,match='raw features'):
        mj.prepare_margin_joint_training(**bad,**LIMITS,inherited=b,context=scope())
    # Trial numbers are trace coordinates, not a new B authority condition.
    mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope(trial=91))


def test_B_analytic_affine_head_and_C_full_physical_primal_oracle(sequence):
    bp,_,p,c,_,_ = sequence
    cache = mj._forward(bp.problems[0],np.full((736,8),.0007))
    alpha,b,*_ = af._solve_affine_head(cache['K'],cache['residual_target'])
    np.testing.assert_array_equal(cache['numeric']['alpha'],alpha)
    np.testing.assert_array_equal(cache['numeric']['intercept'],b)
    cf,num,pr = c.final_cache,c.final_cache['numeric'],p.full_problem
    n,classes = len(p.ids),len(p.classes)
    assert num['K'].shape==(n,n) and num['alpha'].shape==(n,classes)
    assert num['labels'].shape==(n,) and num['old_indices'].shape==pr.old_indices.shape
    np.testing.assert_array_equal(cf['K'],pr.gamma*cf['radial'])
    D,delta,_,_ = _constraints(pr.M_train,pr.train_labels,pr.old_indices)
    assert np.min(D@cf['train_scores'].reshape(-1,order='F')-delta)>=-3e-10
    np.testing.assert_allclose(c.score(**p.raw),cf['train_scores'],atol=3e-10,rtol=2e-10)
    # This equality also detects adding actual B twice at inference.
    np.testing.assert_allclose(cf['train_scores'],pr.M_train+cf['K']@num['alpha']+num['b'],atol=3e-10,rtol=2e-10)
    # Enumerate only the separate K1 oracle's FOUR constraints, not 2^12 sets.
    # The large physical sequence above still receives full KKT/constraint checks.
    args = fixture(k=1)
    oa = old_args(args)
    small_b = mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope()))
    args['old_classes'] = ('c0','c1')
    small_p = mj.prepare_margin_joint_training(**args,**LIMITS,inherited=small_b,context=scope())
    small_c = mj.fit_margin_joint_local_ridge(small_p,mode='C_seq')
    small_cf,small_pr = small_c.final_cache,small_p.full_problem
    D,delta,_,_ = _constraints(small_pr.M_train,small_pr.train_labels,small_pr.old_indices)
    phi = np.linalg.cholesky(small_cf['K'])
    oracle = _feature_primal(phi,small_cf['residual_target'],small_pr.M_train,D,delta)
    np.testing.assert_allclose(oracle['residual'],small_cf['train_scores']-small_pr.M_train,atol=3e-10,rtol=2e-10)
    np.testing.assert_allclose(oracle['theta'][-1],small_cf['numeric']['b'],atol=3e-10,rtol=2e-10)
    assert oracle['objective']==pytest.approx(small_cf['audit']['margin_qp_audit']['final_residuals']['primal_objective'],rel=2e-10,abs=3e-10)


def _active_table_problem(problem):
    # Artificial frozen numeric prior for a differentiability certificate only.
    # It is not an actual registered B sequence and is not used in sequence tests.
    M = np.zeros_like(problem.M_train)
    for row in problem.old_indices:
        truth = int(problem.train_labels[row])
        M[row,truth] = 1.2
        M[row,1-truth] = -1.2
    return replace(problem,M_train=M,M_held=np.zeros_like(problem.M_held),
        audit=dict(problem.audit,prior_source='SYNTHETIC_FROZEN_TABLE_FOR_DERIVATIVE_ONLY'))


@pytest.mark.parametrize('mode',['B','C_seq','C_active_table'])
def test_complete_real_736_Z_directional_derivative_CE_only(sequence,mode):
    bp,_,cp,_,_,_ = sequence
    p = bp if mode=='B' else cp
    if mode=='C_active_table':
        p = replace(cp,problems=tuple(_active_table_problem(pr) for pr in cp.problems))
    rng = np.random.default_rng(91)
    Z = rng.normal(size=(736,p.r))
    Z *= .017/np.linalg.norm(Z)
    direction = rng.normal(size=Z.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-5
    loss,g,audit,cache = mj.evaluate_margin_joint_objective(p,Z)
    high = mj.evaluate_margin_joint_objective(p,Z+epsilon*direction,gradient=False)
    low = mj.evaluate_margin_joint_objective(p,Z-epsilon*direction,gradient=False)
    if mode!='B':
        for middle,upper,lower in zip(cache.folds,high[3].folds,low[3].folds):
            np.testing.assert_array_equal(middle['numeric']['working_set'],upper['numeric']['working_set'])
            np.testing.assert_array_equal(middle['numeric']['working_set'],lower['numeric']['working_set'])
        assert audit['margin_qp_adjoint_factorization_attempts']==0
        assert any(np.linalg.norm(cf['adjoint_arrays']['g_b'])>1e-8 for cf in cache.folds)
    if mode=='C_active_table':
        assert all(len(cf['numeric']['working_set'])>0 for cf in cache.folds)
        assert any(np.linalg.norm(cf['adjoint_arrays']['W_eta'])>0 for cf in cache.folds)
        for cf in cache.folds:
            up = cf['adjoint_arrays']['raw_train_upstream']
            oi,ni = cf['problem'].old_indices,cf['problem'].new_indices
            assert np.linalg.norm(up[np.ix_(oi,oi)])>0
            assert np.linalg.norm(up[np.ix_(oi,ni)])>0
            assert np.linalg.norm(up[np.ix_(ni,ni)])>0
            assert np.linalg.norm(cf['adjoint_arrays']['raw_cross_upstream'][:,oi])>0
    np.testing.assert_allclose(np.sum(g*direction),(high[0]-low[0])/(2*epsilon),rtol=5e-4,atol=4e-8)
    assert loss==audit['RMSCE']==audit['loss_total'] and audit['loss_proximal']==0.
    assert np.linalg.norm(Z)<.5


@pytest.mark.parametrize('mode',['B','C_active_table'])
def test_full_unwhitened_U_directional_derivative_all_adapter_entries(sequence,mode):
    bp,_,cp,_,_,_ = sequence
    p = bp.problems[0] if mode=='B' else _active_table_problem(cp.problems[0])
    rng = np.random.default_rng(137)
    perturbation = rng.normal(size=(736,8))
    perturbation *= .011/np.linalg.norm(perturbation)
    anchor = bp.anchor_U if mode=='B' else cp.anchor_U
    U = anchor+perturbation
    direction = rng.normal(size=U.shape)
    direction /= np.linalg.norm(direction)
    cache = mj._forward(p,U)
    values,G = af._ce(cache['score'],p.held_labels)
    G /= len(values)
    audit = {key:0 for key in mj.STAGE_COUNTERS}
    gradient = mj._backward(cache,G,audit)
    epsilon = 1e-5
    high,low = mj._forward(p,U+epsilon*direction),mj._forward(p,U-epsilon*direction)
    if mode!='B':
        np.testing.assert_array_equal(cache['numeric']['working_set'],high['numeric']['working_set'])
        np.testing.assert_array_equal(cache['numeric']['working_set'],low['numeric']['working_set'])
    finite = (af._ce(high['score'],p.held_labels)[0].mean()-af._ce(low['score'],p.held_labels)[0].mean())/(2*epsilon)
    np.testing.assert_allclose(np.sum(gradient*direction),finite,rtol=5e-4,atol=4e-8)
    assert gradient.shape==(736,8)


def test_C_raw_head_is_independent_of_B_centering_gauge(sequence):
    _,_,p,_,_,_ = sequence
    pr = p.problems[0]
    U = p.anchor_U+np.full((736,8),.0002)
    baseline = mj._forward(pr,U)
    uniform = replace(pr,q=np.full(len(pr.train_labels),1/len(pr.train_labels)))
    changed = mj._forward(uniform,U)
    np.testing.assert_array_equal(changed['score'],baseline['score'])
    np.testing.assert_array_equal(changed['numeric']['alpha'],baseline['numeric']['alpha'])
    np.testing.assert_array_equal(changed['numeric']['multipliers'],baseline['numeric']['multipliers'])


def test_RMSCE_uses_legal_held_supervision_and_exact_cache_billing(sequence):
    _,_,p,_,_,_ = sequence
    Z = np.zeros((736,p.r))
    loss,_,audit,cache = mj.evaluate_margin_joint_objective(p,Z,gradient=False)
    scores = np.concatenate([cf['score'] for cf in cache.folds])
    labels = np.concatenate([cf['problem'].held_labels for cf in cache.folds])
    shifted = scores-scores.max(axis=1,keepdims=True)
    ce = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),labels]
    means = np.array([ce[labels==col].mean() for col in range(len(p.classes))])
    assert loss==pytest.approx(np.sqrt(np.mean(means**2)),abs=2e-15)
    _,g,again,_ = mj.evaluate_margin_joint_objective(p,Z,forward_cache=cache)
    assert again['inner_head_fit_count']==again['inner_factorization_count']==0
    assert again['margin_qp_forward_transitions']==again['margin_qp_forward_spectral_checks']==0
    assert again['ce_adjoint_solve_count']==len(p.problems)
    assert again['derivative_triangular_solve_count']==again['margin_qp_adjoint_triangular_calls']
    changed = replace(p,problems=tuple(replace(pr,held_labels=(pr.held_labels+1)%len(p.classes)) for pr in p.problems))
    different,_,_,changed_cache = mj.evaluate_margin_joint_objective(changed,Z,gradient=False)
    for old,new in zip(cache.folds,changed_cache.folds):
        np.testing.assert_array_equal(old['numeric']['alpha'],new['numeric']['alpha'])
        np.testing.assert_array_equal(old['score'],new['score'])
    changed_labels = (labels+1)%len(p.classes)
    changed_ce = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),changed_labels]
    changed_means = np.array([changed_ce[changed_labels==col].mean() for col in range(len(p.classes))])
    assert different==pytest.approx(np.sqrt(np.mean(changed_means**2)),abs=2e-15)
    assert not np.array_equal(changed_ce,ce)  # Labels influence CE, never head fit.
    with pytest.raises(ValueError,match='cache'):
        mj.evaluate_margin_joint_objective(p,Z+.001,forward_cache=cache)
    with pytest.raises(ValueError,match='anchor'):
        mj.evaluate_margin_joint_objective(p,Z,anchor_U=p.anchor_U+.001)


def test_projected_budget_last_accepted_and_true_forward_adjoint_resources(sequence):
    _,b,_,c,_,_ = sequence
    for state in (b,c):
        audit = state.audit
        assert audit['optimizer_steps']<=4 and audit['trial_count']<=12*audit['optimizer_iterations']
        assert audit['trial_count']==audit['accepted_trial_count']+audit['rejected_trial_count']
        assert audit['accepted_trial_count']==audit['optimizer_steps']
        assert np.linalg.norm(state.Z)<=.5+1e-14
        assert audit['ajlr_forward_evaluation_count']==audit['inner_head_fit_count']+audit['final_head_fit_count']
        for trial in audit['trials']:
            if trial['accepted']:
                assert trial['loss_after']<=trial['loss_before']+trial['comparison_tolerance']
        if audit['steps']:
            np.testing.assert_array_equal(state.Z,state.records[audit['steps'][-1]['state_ref']['key']]['Z'])
        assert audit['resident_numeric_state_bytes']>=audit['deployment_numeric_state_bytes']>0
        assert audit['trained_parameter_count']==(736*state.W.shape[1] if audit['optimizer_steps'] else 0)
        assert audit['source_validation'] is None and audit['device_model'] is None
    audit = c.audit
    assert audit['margin_qp_forward_spectral_checks']==audit['ajlr_forward_evaluation_count']
    assert audit['margin_qp_forward_factorization_attempts']==audit['inner_factorization_count']+audit['final_factorization_count']
    assert audit['margin_qp_forward_factorizations_completed']==audit['completed_factorization_count']
    assert audit['margin_qp_forward_triangular_calls']==audit['head_triangular_solve_count']
    assert audit['margin_qp_forward_triangular_rhs_columns']==audit['head_triangular_rhs_count']
    assert audit['margin_qp_adjoint_triangular_calls']==audit['derivative_triangular_solve_count']
    assert audit['margin_qp_adjoint_triangular_rhs_elements']==audit['derivative_triangular_rhs_element_count']
    assert audit['margin_qp_forward_compact_snapshot_rebuilds']>0
    assert audit['margin_qp_forward_compact_snapshot_dense_work_units']>0
    assert audit['margin_qp_adjoint_factorization_attempts']==0
    assert audit['margin_qp_peak_factor_buffer_bytes']<=LIMITS['max_factor_buffer_bytes']


def test_rejected_trials_are_all_paid_and_do_not_change_last_accepted(sequence,monkeypatch):
    _,_,p,_,_,_ = sequence
    # Only the new joint wrapper's predicate is overridden in this synthetic test.
    original = mj._trial_acceptance
    def reject(before,after,dot):
        return dict(original(before,after,dot),accepted=False)
    monkeypatch.setattr(mj,'_trial_acceptance',reject)
    state = mj.fit_margin_joint_local_ridge(p,mode='C_seq')
    assert state.audit['stop_reason']=='TRIAL_BUDGET_EXHAUSTED'
    assert state.audit['trial_count']==state.audit['rejected_trial_count']==12
    assert state.audit['optimizer_steps']==0
    np.testing.assert_array_equal(state.Z,np.zeros_like(state.Z))
    assert state.audit['inner_head_fit_count']==13*len(p.problems)
    assert state.audit['margin_qp_forward_spectral_checks']==13*len(p.problems)+1


def test_K1_rank0_and_no_inner_held_still_compute_complete_heads():
    args = fixture(k=1)
    oa = old_args(args)
    b = mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope()))
    args['old_classes'] = ('c0','c1')
    p = mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope())
    c = mj.fit_margin_joint_local_ridge(p,mode='C_seq')
    assert not p.problems and c.audit['optimizer_steps']==0
    assert c.audit['final_head_fit_count']==1 and c.audit['final_factorization_count']>=1
    assert c.audit['margin_qp_forward_spectral_checks']==1
    rank0 = replace(p,H=np.zeros_like(p.H),W=np.empty((8,0)),r=0,
        audit=dict(p.audit,no_information=True,no_information_reason='ZERO_DICTIONARY_RANK'))
    state = mj.fit_margin_joint_local_ridge(rank0,mode='C_seq')
    assert state.problem.gamma is not None and state.audit['final_head_fit_count']==1
    assert state.audit['trained_parameter_count']==0 and state.Z.shape==(736,0)
    noheld = replace(p,problems=(),audit=dict(p.audit,no_information=True,no_information_reason='NO_INNER_HELD'))
    state = mj.fit_margin_joint_local_ridge(noheld,mode='C_seq')
    assert state.audit['final_head_fit_count']==1 and state.audit['ce_adjoint_solve_count']==0


def test_new0_reuses_exact_current_B_object_before_QP(sequence):
    _,b,_,_,oa,_ = sequence
    p = mj.prepare_margin_joint_training(**oa,**LIMITS,inherited=b,context=scope())
    assert p.audit['ajlr_preparation_count']==p.audit['prior_head_fit_count']==0
    assert p.audit['no_information_reason']=='N0_REUSE_ACTUAL_B'
    events = []
    assert mj.fit_margin_joint_local_ridge(p,mode='C_seq',log_callback=events.append) is b
    assert events==[]


def test_zero_kernel_still_fits_constrained_free_b_and_does_not_claim_kernel_gradient():
    args = fixture()
    oa = old_args(args)
    for name in mj._NAMES:
        oa[name][:] = oa[name][0]
        args[name][:6] = oa[name]
    b = mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope()))
    args['old_classes'] = ('c0','c1')
    p = mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope())
    c = mj.fit_margin_joint_local_ridge(p,mode='C_seq')
    assert p.full_problem.gamma is None and c.audit['optimizer_steps']==0
    assert c.audit['final_head_fit_count']==1 and c.audit['final_factorization_count']==1
    assert c.audit['margin_qp_forward_spectral_checks']==1
    np.testing.assert_array_equal(c.final_cache['K'],0.)
    # Hand-made, imbalanced physical problem: zero K does NOT force b=0.
    pr = p.full_problem
    keep = np.array([0,1,6])
    contexts = lambda ctx:{key:value[keep] for key,value in ctx.items()}
    manual = replace(pr,train_context=contexts(pr.train_context),train_labels=np.array([0,0,1]),classes=('c0','c1'),
        old_indices=np.array([0,1]),new_indices=np.array([2]),q=np.array([.5,.5,0.]),d0=np.zeros((3,3)),
        held_context={key:value[:0] for key,value in pr.held_context.items()},held_labels=np.empty(0,dtype=np.int64),
        cross_d0=np.empty((0,3)),M_train=np.zeros((3,2)),M_held=np.empty((0,2)))
    cache = mj._forward(manual,np.zeros((736,8)))
    np.testing.assert_allclose(cache['numeric']['b'],[1/6,-1/6],atol=2e-12)
    np.testing.assert_allclose(cache['train_scores'],np.tile([1/6,-1/6],(3,1)),atol=2e-12)
    audit = {key:0 for key in mj.STAGE_COUNTERS}
    gradient = mj._backward(cache,np.empty((0,2)),audit)
    np.testing.assert_array_equal(gradient,0.)
    assert audit['derivative_triangular_solve_count']==0
    assert audit['kernel_gradient_status'].startswith('NOT_REQUESTED')
    # A mathematical zero scale is also zero kernel, not an ordinary Jacobian.
    zero_scale = replace(manual,gamma=0.,tau=1.)
    zero_cache = mj._forward(zero_scale,np.zeros((736,8)))
    np.testing.assert_allclose(zero_cache['numeric']['b'],cache['numeric']['b'],atol=2e-12)
    zero_audit = {key:0 for key in mj.STAGE_COUNTERS}
    np.testing.assert_array_equal(mj._backward(zero_cache,np.ones((0,2)),zero_audit),0.)
    assert zero_audit['derivative_triangular_solve_count']==0


def test_tau0_equivalence_retains_all_physical_SSE_and_constraints():
    args = fixture()
    oa = old_args(args)
    for name in mj._NAMES:
        oa[name][3:] = oa[name][:3]
        args[name][:6] = oa[name]
        args[name][6] = oa[name][0]
    b = mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope()))
    args['old_classes'] = ('c0','c1')
    p = mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope())
    c = mj.fit_margin_joint_local_ridge(p,mode='C_seq')
    assert p.full_problem.tau==0 and p.full_problem.gamma is not None
    assert c.audit['optimizer_steps']==c.audit['derivative_triangular_solve_count']==0
    assert c.final_cache['numeric']['alpha'].shape==(9,3)
    assert c.audit['final_fit']['old_physical_constraint_count']==6*2
    assert len(c.final_cache['numeric']['slack'])==6*2
    assert len(c.problem.audit['old_reference_physical_ids'])==6
    np.testing.assert_array_equal(c.final_cache['radial'],(p.full_problem.d0==0).astype(float))
    scores = c.score(**p.raw)
    np.testing.assert_allclose(scores[0],scores[3],atol=3e-10)
    np.testing.assert_allclose(scores,c.final_cache['train_scores'],atol=3e-10)


def test_unsupported_active_jacobian_is_explicit_failure_not_fallback():
    args = fixture()
    for name in mj._NAMES:
        args[name][:] = args[name][0]
    oa = old_args(args)
    b = mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope()))
    args['old_classes'] = ('c0','c1')
    p = mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope())
    # Synthetic positive scale on an exactly rank-one kernel exercises the
    # unavailable Jacobian path. It is NOT a replacement for a real scale rule.
    problems = tuple(replace(pr,gamma=1.,tau=1.) for pr in p.problems)
    p = replace(p,problems=problems,full_problem=replace(p.full_problem,gamma=1.,tau=1.),
        audit=dict(p.audit,no_information=False,no_information_reason=None))
    with pytest.raises(mj.NumericalFailure,match='UNSUPPORTED_ACTIVE_JACOBIAN') as caught:
        mj.fit_margin_joint_local_ridge(p,mode='C_seq')
    audit = caught.value.audit
    assert audit['status']=='TECHNICAL_FAILURE' and audit['failure_code']=='UNSUPPORTED_ACTIVE_JACOBIAN'
    assert audit['inner_head_fit_count']==len(p.problems) and audit['margin_qp_forward_triangular_calls']>0
    assert audit['margin_qp_adjoint_triangular_calls']==0 and audit['optimizer_steps']==0
    assert 'failure' in caught.value.records
    assert 'G' in caught.value.records['failure']
    assert isinstance(caught.value.arrays,MappingProxyType) and 'G' in caught.value.arrays
    np.testing.assert_array_equal(caught.value.arrays['G'],caught.value.records['failure']['G'])
    with pytest.raises(ValueError):
        caught.value.arrays['G'][...] = 0
    json.dumps(caught.value.audit_dict(),allow_nan=False)


def test_transition_and_factor_buffer_failures_preserve_actual_work_and_arrays(sequence):
    _,b,_,_,_,args = sequence
    p = mj.prepare_margin_joint_training(**args,max_transitions=1,max_factor_buffer_bytes=LIMITS['max_factor_buffer_bytes'],
        inherited=b,context=scope())
    with pytest.raises(mj.NumericalFailure,match='TRANSITION_LIMIT') as caught:
        mj.fit_margin_joint_local_ridge(p,mode='C_seq')
    audit = caught.value.audit
    assert audit['margin_qp_forward_transitions']==1
    assert audit['margin_qp_forward_factorization_attempts']==1
    assert audit['margin_qp_forward_triangular_calls']>0
    failed = caught.value.records['failure']
    assert 'qp_failure_slack' in failed and 'qp_failure_rho' in failed
    np.testing.assert_array_equal(failed['last_accepted_Z'],np.zeros_like(failed['last_accepted_Z']))
    tiny = mj.prepare_margin_joint_training(**args,max_transitions=128,max_factor_buffer_bytes=1,inherited=b,context=scope())
    with pytest.raises(mj.NumericalFailure,match='FACTOR_BUFFER_LIMIT') as caught:
        mj.fit_margin_joint_local_ridge(tiny,mode='C_seq')
    assert caught.value.audit['margin_qp_forward_factorization_attempts']==0
    assert caught.value.audit['margin_qp_forward_triangular_calls']==0
    assert caught.value.audit['inner_head_fit_count']==1


def test_production_callbacks_finite_archive_readback_full_text_and_scalar_surface(sequence,tmp_path):
    _,_,p,_,_,_ = sequence
    seen,events = {},[]
    def archive(key,arrays):
        path = tmp_path/(key+'.npz')
        assert all(value.dtype.kind in 'fibu' and np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False) as loaded:
            assert set(loaded.files)==set(arrays)
            for name,value in arrays.items():
                np.testing.assert_array_equal(loaded[name],value)
                assert loaded[name].shape==value.shape
        seen[key] = {name:tuple(value.shape) for name,value in arrays.items()}
        return dict(storage='synthetic_npz',path=str(path))
    state = mj.fit_margin_joint_local_ridge(p,mode='C_seq',state_callback=archive,log_callback=events.append)
    audit = json.loads(json.dumps(state.audit_dict(),allow_nan=False))
    assert audit['final_state_ref']['arrays']['s']['shape']==[]
    assert seen['final']['s']==()
    assert seen['final']['prior_old_class_indices']==(2,)
    assert {'prior_B_U','prior_B_alpha','prior_B_intercept','M_train','multipliers','working_set',
            'K','L','chol_A','chol_working','margins','delta','P_old','raw_train','adapted_train_b'}<=set(seen['final'])
    assert any(key.startswith('gradient_') for key in seen)
    assert any(key.startswith('trial_') for key in seen)
    assert audit['final_objective_state_ref']['key']=='final_objective'
    for record in events:
        assert 'source_validation=N/A' in record['text']
        compact = mj.compact_training_record(record)
        assert all(value is None or isinstance(value,(str,int,float,bool)) for value in compact.values())
        json.dumps(compact,allow_nan=False)
        if record['event']=='MARGIN_JOINT_GRADIENT':
            assert compact['counter_ownership']=='OBJECTIVE_INCREMENT'
            assert compact['margin_qp_forward_transitions']==0
    assert events[-1]['event']=='MARGIN_JOINT_FINAL'


def test_full_registered_samplewise_scores_and_actual_physical_bytes(sequence):
    _,_,p,c,_,_ = sequence
    together,resource = c.score_with_audit(**p.raw)
    individual = np.vstack([c.score(**{name:value[i:i+1] for name,value in p.raw.items()}) for i in range(len(p.ids))])
    np.testing.assert_array_equal(together,individual)
    np.testing.assert_array_equal(c.predict(**p.raw),np.asarray(c.classes)[np.argmax(together,axis=1)])
    assert resource['score_physical_count']==len(p.ids)
    assert resource['residual']['kernel_pair_count']==len(p.ids)**2
    assert resource['prior']['kernel_pair_count']==len(p.ids)*len(c.prior.ids)
    assert resource['kernel_pair_count']==resource['residual']['kernel_pair_count']+resource['prior']['kernel_pair_count']
    assert c.audit['resident_numeric_state_bytes']==mj._unique_bytes(c._resident_values())
    assert c.audit['deployment_numeric_state_bytes']==mj._unique_bytes(c._deployment_values())
    array = np.arange(12,dtype=np.float64)
    mapping = MappingProxyType(dict(original=array,view=array[2:9]))
    assert mj._unique_bytes((mapping,array))==array.nbytes
    bytes_backed = np.frombuffer(array.tobytes(),dtype=np.float64)
    assert mj._unique_bytes((bytes_backed,bytes_backed[1:]))==array.nbytes
