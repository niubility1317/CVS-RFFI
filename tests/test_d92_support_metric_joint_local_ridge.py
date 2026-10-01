"""Literal synthetic composition tests. Numerical execution belongs to root."""
if __name__=='__main__':
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),root/'code/cvsrffi/d92_support_metric_joint_local_ridge.py'):
        raw=path.read_bytes();source=raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source
        ast.parse(source,filename=str(path));print('STATIC_AST_UTF8_VERIFIED',path)
    raise SystemExit(0)

from dataclasses import replace
import json
import numpy as np
import pytest
from cvsrffi import d92_support_metric_joint_local_ridge as joint
from cvsrffi import d92_proto_frame_primitives as primitive
from cvsrffi import d92_proto_frame_joint_local_ridge as old_method

LIMITS=dict(max_integer_bits=16384,max_fraction_operations=1_000_000,
    max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=16_000_000,
    max_secular_iterations=128)
CONTEXT=dict(run_id='synthetic-support-metric',row_id='row',split_id='literal',scope='support',fold=None,trial=None)


def fixture(k=3,c=8,rank=2):
    rng=np.random.default_rng(7301);classes=tuple('class-'+str(i) for i in range(c));labels=np.repeat(np.arange(c),k)
    raw={}
    for name,width in zip(joint.BRANCHES,(160,96,160,160,160)):
        center=rng.normal(size=(c,width));raw[name]=center[labels]+.7*rng.normal(size=(c*k,width))
    prototypes=np.zeros((6,160))
    if rank==0:prototypes[:,0]=1
    elif rank==1:prototypes[:,0]=[1,-1,1,-1,1,-1]
    else:
        prototypes[[0,2,4],0]=[1,-1,1];prototypes[[1,3,5],1]=[1,-1,1]
    frame=primitive.build_proto_frame_dictionary(prototypes=prototypes,classes=classes[:6])
    return dict(**raw,support_labels=labels,support_ids=tuple(f'physical-{i}-{j}' for i in range(c) for j in range(k)),
        classes=classes,old_classes=classes[:6],prototype_frame=frame)


def old_args(args):
    mask=args['support_labels']<6
    return dict(**{k:args[k][mask] for k in joint.BRANCHES},support_labels=args['support_labels'][mask],
        support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),
        classes=args['old_classes'],old_classes=args['old_classes'],prototype_frame=args['prototype_frame'])


def prepare(args,**kwargs):return joint.prepare_support_metric_joint_training(**args,**LIMITS,context=CONTEXT,**kwargs)


@pytest.fixture(scope='module')
def sequence():
    args=fixture();old=old_args(args);bp=prepare(old);b=joint.fit_support_metric_joint_local_ridge(bp)
    cp=prepare(args,inherited=b);c=joint.fit_support_metric_joint_local_ridge(cp,mode='C_seq')
    return args,old,bp,b,cp,c


def test_independent_U_coordinate_ABI_and_actual_B(sequence):
    args,old,bp,b,cp,c=sequence
    assert b.audit['method']==joint.METHOD and joint.METHOD!=old_method.METHOD
    assert bp.basis.rank==2 and b.theta.shape==c.theta.shape==(2,)
    assert cp.basis is b.basis and c.prior is b
    np.testing.assert_array_equal(cp.anchor,b.theta)
    np.testing.assert_array_equal(cp.physical_gram,cp.basis.basis.T@cp.basis.basis)
    assert cp.audit['actual_work']['basis_calls']==0
    assert bp.audit['actual_work']['basis_calls']==1
    assert cp.audit['full_prior_ref']==b.audit['final_state_ref']==c.audit['final_prior_ref']
    assert c.audit['effective_parameter_rank']==2 and c.audit['nominal_parameter_count']==5
    np.testing.assert_array_equal(c.to_arrays()['actual_B_theta'],b.theta)
    assert c.to_arrays()['U'].shape==(160,2)
    assert c.to_arrays()['head_theta_padded'].shape==(5,)
    with pytest.raises(ValueError,match='actual current U-coordinate B'):
        prepare(args,inherited=object())
    changed=dict(args,z_id=args['z_id'].copy());changed['z_id'][0,0]+=.01
    with pytest.raises(ValueError,match='records/labels changed'):prepare(changed,inherited=b)
    context=dict(CONTEXT,row_id='other')
    with pytest.raises(ValueError,match='lineage crosses row_id'):
        joint.prepare_support_metric_joint_training(**args,**LIMITS,context=context,inherited=b)


def test_old_inner_teacher_never_uses_held_and_keeps_actual_B_anchor(sequence):
    p=sequence[4];b=sequence[3]
    for audit,problem in zip(p.audit['folds'],p.folds):
        assert set(audit['old_inner_train_ids']).isdisjoint(problem.held_ids)
        saved=p.records[audit['prior_ref']['key']]
        np.testing.assert_array_equal(saved['theta'],b.theta)
        assert saved['K'].shape==(12,12)
        K,Y,L=saved['K'],saved['Y'],saved['L'];n=len(K)
        saddle=np.block([[K+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        sol=np.linalg.solve(saddle,np.vstack((Y,np.zeros((1,6)))))
        np.testing.assert_allclose(np.vstack((problem.prior_train,problem.prior_held)),L@sol[:-1]+sol[-1],atol=2e-12)


@pytest.mark.parametrize('stage',['B','C'])
def test_complete_U_kernel_ridge_threshold_gate_JVP_and_RMS_gradient(sequence,stage):
    p=sequence[2] if stage=='B' else sequence[4];theta=p.anchor+np.array([.019,-.011])
    direction=np.array([.3,-.4]);epsilon=2e-5
    loss,g,info,cache=joint.evaluate_support_metric_joint_objective(p,theta)
    plus=joint.evaluate_support_metric_joint_objective(p,theta+epsilon*direction,jacobian=False)
    minus=joint.evaluate_support_metric_joint_objective(p,theta-epsilon*direction,jacobian=False)
    np.testing.assert_allclose(g@direction,(plus[0]-minus[0])/(2*epsilon),rtol=3e-4,atol=2e-7)
    np.testing.assert_allclose(np.einsum('ncr,r->nc',cache.score_jacobian,direction),
        (plus[3].scores-minus[3].scores)/(2*epsilon),rtol=5e-4,atol=2e-6)
    assert cache.score_jacobian.shape==(len(p.ids),len(p.classes),2)
    assert info['RMSCE']==info['loss_total']==loss and info['loss_proximal']==0
    assert cache.arrays['curvature'].shape==(2,2) and 'damped_hessian' not in cache.arrays
    if stage=='C':
        assert len(p.classes)-len(p.old_classes)==2
        args=sequence[0]
        assert not np.array_equal(args['z_id'][args['support_labels']==6],args['z_id'][args['support_labels']==7])
        assert info['actual_work']['gate_jvp_calls']==3
        assert info['actual_work']['gate_jvp_triangular_rhs_columns']==36
        assert np.linalg.norm(cache.arrays['fold_0_lower_bounds_jacobian'][...,:2])>0
        assert np.count_nonzero(cache.arrays['fold_0_K_jacobian'][...,2:])==0
        for fold in range(len(p.folds)):
            prefix='fold_'+str(fold)+'_'
            for value,derivative in (('lower_bounds','lower_bounds_jacobian'),
                    ('new_intercept','new_intercept_jacobian'),('gate_b','gate_b_jacobian')):
                analytic=np.einsum('...r,r->...',cache.arrays[prefix+derivative][...,:2],direction)
                numerical=(plus[3].arrays[prefix+value]-minus[3].arrays[prefix+value])/(2*epsilon)
                np.testing.assert_allclose(analytic,numerical,rtol=5e-4,atol=2e-6)
    assert plus[3].score_jacobian is None and plus[1] is None


def test_single_new_class_has_exactly_zero_conditional_softmax_threshold_JVP(sequence):
    args,b=sequence[0],sequence[3];mask=args['support_labels']<7
    single=dict(**{name:args[name][mask] for name in joint.BRANCHES},
        support_labels=args['support_labels'][mask],
        support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),
        classes=args['classes'][:7],old_classes=args['old_classes'],prototype_frame=args['prototype_frame'])
    p=prepare(single,inherited=b)
    _,_,_,cache=joint.evaluate_support_metric_joint_objective(p,p.anchor+np.array([.019,-.011]))
    for fold in range(len(p.folds)):
        prefix='fold_'+str(fold)+'_'
        bounds=cache.arrays[prefix+'lower_bounds_jacobian']
        assert bounds.shape==(12,1,5)
        np.testing.assert_array_equal(bounds,np.zeros_like(bounds))
        for key in ('new_alpha','new_intercept','new_score_jacobian'):
            value=cache.arrays[prefix+key]
            np.testing.assert_array_equal(value,np.zeros_like(value))


def test_Fisher_metric_quadratic_is_actual_class_balanced_score_geometry(sequence):
    p=sequence[2];_,g,_,cache=joint.evaluate_support_metric_joint_objective(p,p.anchor)
    step=joint.metric_module.solve_support_metric_step(gradient=g,ggn=cache.arrays['curvature'],
        physical_gram=p.physical_gram,score_jvp=cache.score_jacobian,
        probabilities=cache.arrays['probabilities'],labels=p.labels,max_secular_iterations=128)
    J,P=cache.score_jacobian,cache.arrays['probabilities'];C=len(p.classes)
    counts=np.bincount(p.labels,minlength=C);F=np.zeros((2,2))
    for i in range(len(J)):
        F+=J[i].T@(np.diag(P[i])-np.outer(P[i],P[i]))@J[i]/(C*counts[p.labels[i]])
    np.testing.assert_allclose(step.fisher,F,atol=3e-15)
    np.testing.assert_allclose(step.M,p.physical_gram+F,atol=3e-15)
    np.testing.assert_allclose(step.H,cache.arrays['curvature']+step.M,atol=3e-15)
    assert step.direction@step.M@step.direction<=.25+2e-14
    residual=step.H@step.direction+g+step.multiplier*step.M@step.direction
    np.testing.assert_allclose(residual,0,atol=3e-12)


def test_eta_full_direction_actual_Armijo_one_step_and_SUM_MAX(sequence):
    for state in (sequence[3],sequence[5]):
        a=state.audit_dict();json.dumps(a,allow_nan=False)
        assert a['optimizer_steps'] in (0,1) and a['trial_count']<=12
        assert a['actual_work']['support_metric_step_calls']==1
        assert a['actual_work']['ggn_step_calls']==0
        assert a['parameter_buffer_bytes']==16
        assert a['complete_head_jvp_error_bound'] is None and a['device_cost'] is None
        for t in a['trials']:
            assert t['step_size']==.5**t['trial']
            assert t['metric_ball_value']<=.25+2e-14
            if t['accepted']:assert t['loss_after']<=t['armijo_rhs']+t['comparison_tolerance']
        if a['trials']:assert a['trials'][0]['step_size']==1.
        for key in joint.WORK_SUM_KEYS:
            actual=sum(op['audit'].get(key[len(op['operation'])+1:],0) for op in a['operation_audits']
                if key.startswith(op['operation']+'_') and key!=op['operation']+'_calls')
            if not key.endswith('_calls'):assert a['actual_work'][key]==pytest.approx(actual)
        assert set(a['actual_work'])==set(joint.WORK_SUM_KEYS)|set(joint.WORK_MAX_KEYS)
        for key in joint.WORK_MAX_KEYS:
            values=[op['audit'].get(key[len(op['operation'])+1:],0) for op in a['operation_audits'] if key.startswith(op['operation']+'_')]
            assert a['actual_work'][key]==max(values,default=0)


@pytest.mark.parametrize('rank,k',[(0,3),(0,1),(1,1),(1,3)])
def test_exact_rank_and_K1_complete_final_head(rank,k):
    args=fixture(k=k,c=6,rank=rank);p=prepare(args);state=joint.fit_support_metric_joint_local_ridge(p)
    assert p.basis.rank==rank and state.theta.shape==(rank,)
    assert state.audit['final_head_complete'] and state.head['alpha'].shape==(6*k,6)
    assert state.audit['parameter_buffer_bytes']==8*rank
    if rank==0 or k==1:
        assert state.audit['optimizer_steps']==state.audit['actual_work']['support_metric_step_calls']==0
        assert state.audit['trainable_parameter_count']==0
    assert state.audit['actual_work']['ridge_calls']>=1
    sample={name:args[name][:1] for name in joint.BRANCHES}
    scores,work=state.score_with_audit(**sample)
    assert scores.shape==(1,6) and work['single_record_all_registered_classes']
    with pytest.raises(ValueError,match='exactly one'):state.score(**{n:args[n][:2] for n in joint.BRANCHES})


def test_new0_exact_original_object_reuse_before_head(sequence,monkeypatch):
    old,b=sequence[1],sequence[3];p=prepare(old,inherited=b)
    def forbidden(*a,**kw):raise AssertionError('new0 invoked numeric work')
    monkeypatch.setattr(joint,'_head',forbidden)
    monkeypatch.setattr(joint.metric_module,'solve_support_metric_step',forbidden)
    assert joint.fit_support_metric_joint_local_ridge(p,mode='C_seq') is b
    assert all(v==0 for v in p.audit['actual_work'].values())


@pytest.mark.parametrize('boundary',['zero','tau0'])
def test_zero_kernel_and_exact_equivalence_keep_complete_free_intercept_head(boundary):
    args=fixture(k=2,c=6,rank=1)
    for name in joint.BRANCHES:
        width=args[name].shape[1];values=np.zeros_like(args[name]);values[:,0]=1.
        if boundary=='tau0':values[1::2,1]=1.
        args[name]=values
    p=prepare(args);state=joint.fit_support_metric_joint_local_ridge(p)
    assert state.audit['optimizer_steps']==0 and state.audit['final_head_complete']
    assert all(f.gamma is None for f in p.folds) if boundary=='zero' else p.full_problem.tau==0
    assert p.full_problem.gamma is None if boundary=='zero' else p.full_problem.gamma>0
    K,Y=state.head['arrays']['K'],state.head['arrays']['Y'];n=len(K)
    saddle=np.block([[K+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    sol=np.linalg.solve(saddle,np.vstack((Y,np.zeros((1,6)))))
    np.testing.assert_allclose(state.head['alpha'],sol[:-1],atol=2e-12)
    np.testing.assert_allclose(state.head['intercept'],sol[-1],atol=2e-12)


def test_basis_budget_failure_preserves_actual_arithmetic_and_snapshot():
    args=fixture(c=6);limits=dict(LIMITS,max_fraction_operations=1)
    with pytest.raises(joint.SupportMetricJointFailure) as caught:
        joint.prepare_support_metric_joint_training(**args,**limits,context=CONTEXT)
    failure=caught.value
    assert failure.audit['actual_work']['basis_calls']==1
    assert failure.audit['actual_work']['basis_fraction_operations_attempted']>1
    assert failure.audit['status']=='TECHNICAL_FAILURE' and 'original_dictionary_Q' in failure.arrays
    assert failure.audit['failure_state_ref']['key']=='preparation_failure'


def test_explicit_prebuilt_basis_is_bound_without_recharging_construction():
    args=fixture(k=1,c=6,rank=1)
    basis=joint.basis_module.build_support_metric_basis(Q=args['prototype_frame'].Q,
        max_integer_bits=LIMITS['max_integer_bits'],max_fraction_operations=LIMITS['max_fraction_operations'])
    p=prepare(args,support_metric_basis=basis)
    assert p.basis is basis and not p.audit['basis_construction_charged_here']
    assert p.audit['actual_work']['basis_calls']==0
    assert p.audit['actual_work']['basis_fraction_operations_attempted']==0
    assert p.audit['actual_work']['basis_binding_calls']==1
    assert p.audit['actual_work']['basis_binding_physical_gram_evaluation_count']==1
    np.testing.assert_array_equal(p.physical_gram,basis.basis.T@basis.basis)
    wrong=fixture(k=1,c=6,rank=2)
    with pytest.raises(joint.SupportMetricJointFailure,match='exact frozen dictionary'):
        prepare(wrong,support_metric_basis=basis)


def test_step_failure_preserves_initial_head_and_actual_failed_work(sequence,monkeypatch):
    p=sequence[2]
    def fail(**kwargs):
        audit={k:0 for k in joint.metric_module.AUDIT_SUM_KEYS};audit['factorization_attempts']=1
        raise joint.metric_module.SupportMetricFailure('SYNTHETIC_FACTOR_FAILURE','literal',audit,
            {'gradient':kwargs['gradient'],'physical_gram':kwargs['physical_gram'],'rhs':np.ones((2,1))})
    monkeypatch.setattr(joint.metric_module,'solve_support_metric_step',fail)
    with pytest.raises(joint.SupportMetricJointFailure) as caught:joint.fit_support_metric_joint_local_ridge(p)
    failure=caught.value;work=failure.audit['actual_work']
    assert work['support_metric_step_calls']==work['support_metric_step_factorization_attempts']==1
    assert work['support_metric_step_factorizations_completed']==0 and work['ridge_calls']==3
    assert 'initial' in failure.records and 'failure' in failure.records
    assert 'failed_rhs' in failure.arrays and 'last_accepted_theta' in failure.arrays
    np.testing.assert_array_equal(failure.arrays['last_accepted_theta'],p.anchor)


def test_callback_full_state_finite_shapes_and_unmodifiable_arrays(sequence):
    p=sequence[2];seen={};events=[]
    def callback(key,arrays):
        seen[key]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return dict(storage='literal',key=key,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),nbytes=v.nbytes) for k,v in arrays.items()})
    state=joint.fit_support_metric_joint_local_ridge(p,state_callback=callback,log_callback=events.append)
    assert state.audit['final_state_ref']['storage']=='literal'
    assert seen['final']['theta'].shape==(2,) and seen['final']['U'].shape==(160,2)
    assert any(e['event']=='SUPPORT_METRIC_FINAL' for e in events)
    assert joint.compact_training_record(events[-1])['method']==joint.METHOD
    json.dumps(state.audit_dict(),allow_nan=False)
    with pytest.raises(ValueError):state.theta[0]=0
    with pytest.raises(AttributeError):state.basis.basis=np.zeros((160,2))
    with pytest.raises(TypeError):state.records['fake']={}
