"""Synthetic/query-blind joint contracts. Parent owns numerical execution."""
if __name__ == '__main__':
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),root/'code/cvsrffi/d92_group_barrier_joint_local_ridge.py'):
        raw=path.read_bytes(); source=raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source and '\r' not in source
        ast.parse(source,filename=str(path))
        print('STATIC_AST_UTF8_VERIFIED',path)
    raise SystemExit(0)

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.optimize import minimize

from cvsrffi import d92_group_barrier_joint_local_ridge as joint
from cvsrffi import d92_group_barrier_gate as gate
from cvsrffi import d92_margin_joint_local_ridge as margin

LIMITS=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=16_000_000)


def scope(**extra):
    return dict(run_id='synthetic-group-barrier-only',row_id='row-one',split_id='artificial-split',scope='legal-support',**extra)


def fixture(k=3,c=8,seed=7201):
    rng=np.random.default_rng(seed); labels=np.repeat(np.arange(c),k)
    raw={}
    for name,width in zip(joint._NAMES,(160,96,160,160,160)):
        rows=rng.normal(size=(k*c,width)); rows[:,0]+=labels*.3; rows[:,1]+=(labels%2)*.4
        raw[name]=rows
    classes=tuple('c'+str(i) for i in range(c))
    return dict(**raw,support_labels=labels,support_ids=tuple(f'artificial-{i}-{j:02}' for i in range(c) for j in range(k)),
        classes=classes,old_classes=classes[:6])


def old_args(args):
    old=args['old_classes']; mask=args['support_labels']<len(old)
    return dict(**{name:args[name][mask] for name in joint._NAMES},support_labels=args['support_labels'][mask],
        support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)


def sequence_args(k=3,c=8,**kwargs):
    args=fixture(k=k,c=c,**kwargs); oa=old_args(args)
    bp=joint.prepare_group_barrier_joint_training(**oa,**LIMITS,context=scope())
    b=joint.fit_group_barrier_joint_local_ridge(bp,**LIMITS)
    cp=joint.prepare_group_barrier_joint_training(**args,**LIMITS,inherited=b,context=scope())
    return bp,b,cp,args,oa


@pytest.fixture(scope='module')
def sequence():
    bp,b,p,args,oa=sequence_args()
    c=joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',**LIMITS)
    return bp,b,p,c,args,oa


def test_actual_B_delegation_lineage_and_exact_new0(sequence,monkeypatch):
    bp,b,p,c,args,oa=sequence
    assert isinstance(b,margin.MarginJointState) and b.audit['mode']=='B'
    assert b.audit['config']['proximal_coefficient']==0
    np.testing.assert_array_equal(p.anchor_U,b.U)
    assert c.prior is b and c.audit['schema']=='d92_group_barrier_joint_local_ridge_v1'
    assert c.audit['config']['proximal_coefficient']==0 and c.audit['coordinate_norm']<=.5+1e-14
    n0=joint.prepare_group_barrier_joint_training(**oa,**LIMITS,inherited=b,context=scope())
    def forbidden(*a,**kw): raise AssertionError('new0 fitted a gate')
    monkeypatch.setattr(gate,'fit_group_barrier_gate',forbidden)
    assert joint.fit_group_barrier_joint_local_ridge(n0,mode='C_seq',**LIMITS) is b
    for key in ('run_id','row_id','scope','split_id'):
        bad=scope(); bad[key]='another-parent'
        with pytest.raises(ValueError,match='crosses'):
            joint.prepare_group_barrier_joint_training(**args,**LIMITS,inherited=b,context=bad)
    modified=dict(args,z_id=args['z_id'].copy()); modified['z_id'][0,0]+=.01
    with pytest.raises(ValueError,match='raw features'):
        joint.prepare_group_barrier_joint_training(**modified,**LIMITS,inherited=b,context=scope())


def test_explicit_frozen_resources_and_no_C_QP(sequence,monkeypatch):
    _,b,p,_,args,_=sequence
    with pytest.raises(TypeError): joint.prepare_group_barrier_joint_training(**args,inherited=b,context=scope())
    for name in LIMITS:
        invalid=dict(LIMITS); invalid[name]=True
        with pytest.raises(ValueError,match='explicit positive integer'):
            joint.prepare_group_barrier_joint_training(**args,**invalid,inherited=b,context=scope())
    def forbidden(*a,**kw): raise AssertionError('C called margin QP')
    monkeypatch.setattr(margin.qp,'fit_margin_qp_head',forbidden)
    joint._forward(p.problems[0],p.anchor_U,LIMITS)
    changed=dict(LIMITS,max_newton_iterations=99)
    with pytest.raises(ValueError,match='frozen preparation'):
        joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',**changed)


def test_inner_prior_physical_isolation_and_final_actual_B(sequence):
    _,b,p,c,_,_=sequence
    oldcols=[p.classes.index(name) for name in p.old_classes]
    np.testing.assert_array_equal(p.full_problem.M_train[:,oldcols],b.score(**p.raw))
    for problem,teacher in zip(p.problems,p.audit['prior_folds']):
        train=set(teacher['training_physical_ids']); held=set(teacher['held_physical_ids'])
        assert train==set(problem.audit['old_reference_physical_ids'])
        assert not train&held and train|held==set(b.ids)
        assert train.isdisjoint(problem.audit['held_physical_ids'])
        rec=p.records[teacher['head_state_ref']['key']]
        np.testing.assert_array_equal(rec['U'],b.U)
    for array in (c.U,c.Z,c.final_cache['bounds'],c.final_cache['gate'].alpha):
        with pytest.raises(ValueError): array.flat[0]=9
        with pytest.raises(ValueError): array.setflags(write=True)
    json.dumps(c.audit_dict(),allow_nan=False)


def test_new_only_analytic_head_and_independent_feature_primal_gate(sequence):
    _,_,p,_,_,_=sequence
    cf=joint._forward(p.problems[0],p.anchor_U,LIMITS)
    ni,oi=cf['new_indices'],cf['old_indices']; K=cf['K']; Y=cf['Ynew']; n=len(ni)
    saddle=np.block([[np.eye(n)+K[np.ix_(ni,ni)],np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    solved=np.linalg.solve(saddle,np.vstack((Y,np.zeros((1,Y.shape[1])))))
    np.testing.assert_allclose(cf['new_alpha'],solved[:-1],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(cf['htrain'],K[:,ni]@solved[:-1]+solved[-1],rtol=2e-12,atol=2e-12)
    assert len(cf['new_labels'])==len(ni) and cf['new_labels'].max()<len(cf['new_columns'])
    # Independent finite-feature primal: dense Cholesky features, no imported
    # gate solver formula or representer Newton system.
    Phi=np.linalg.cholesky(K); target=np.isin(cf['problem'].train_labels,cf['old_columns']).astype(float)
    gs=cf['gate']; bounds=cf['bounds']; zeta=gs.zeta
    def objective(theta):
        t=Phi@theta[:-1]+theta[-1]; slack=t[oi,None]-bounds
        if np.any(slack<=0): return np.inf,np.zeros_like(theta)
        q=1/(1+np.exp(-t))-target; mu=zeta/slack; q[oi]-=mu.sum(axis=1)
        val=.5*theta[:-1]@theta[:-1]+np.logaddexp(0.,np.where(target==1,-t,t)).sum()-zeta*np.log(slack).sum()
        return val,np.r_[theta[:-1]+Phi.T@q,q.sum()]
    theta=np.r_[Phi.T@gs.alpha,gs.b]
    independent=minimize(objective,theta,jac=True,method='BFGS',options=dict(gtol=1e-8,maxiter=500))
    np.testing.assert_allclose(Phi@independent.x[:-1]+independent.x[-1],gs.f,rtol=1e-6,atol=2e-6)
    assert np.all(gs.slacks>0)
    oldcols=cf['old_columns']; logold=joint._logsoftmax(cf['problem'].M_train[:,oldcols])
    for j,row in enumerate(oi):
        own=np.flatnonzero(oldcols==cf['problem'].train_labels[row])[0]
        cross=gs.f[row]+logold[row,own]-joint._logsoftmax(cf['htrain'])[row]
        assert np.min(cross)>cf['margin'][j]-1e-10


def test_full_nonzero_C_direction_gradient_includes_new_head_RHS(sequence):
    _,_,p,_,_,_=sequence
    rng=np.random.default_rng(171); Z=rng.normal(size=(736,p.r))*.0001
    direction=rng.normal(size=Z.shape); direction/=np.linalg.norm(direction)
    loss,g,info,cache=joint.evaluate_group_barrier_joint_objective(p,Z,**LIMITS)
    step=2e-5
    plus=joint.evaluate_group_barrier_joint_objective(p,Z+step*direction,gradient=False,**LIMITS)[0]
    minus=joint.evaluate_group_barrier_joint_objective(p,Z-step*direction,gradient=False,**LIMITS)[0]
    np.testing.assert_allclose(np.sum(g*direction),(plus-minus)/(2*step),rtol=5e-3,atol=2e-6)
    assert loss==info['RMSCE'] and info['loss_proximal']==0 and info['gradient_norm']>0
    for cf in cache.folds:
        ar=cf['adjoint_arrays']
        assert np.any(ar['bounds_upstream']) and np.any(ar['new_old_G'])
        assert ar['K_upstream'].shape==cf['K'].shape and ar['L_upstream'].shape==cf['L'].shape
    _,_,reused,_=joint.evaluate_group_barrier_joint_objective(p,Z,gradient=False,forward_cache=cache,**LIMITS)
    assert reused['inner_head_fit_count']==0 and reused['group_gate_forward_newton_iterations']==0
    with pytest.raises(ValueError,match='different exact objective'):
        joint.evaluate_group_barrier_joint_objective(p,Z+direction*.0001,forward_cache=cache,**LIMITS)


def test_single_physical_allclass_batching_and_structured_argmax(sequence):
    _,b,p,c,_,_=sequence
    rows={name:p.raw[name][:5] for name in joint._NAMES}
    scores,info=c.score_with_audit(**rows)
    single=np.concatenate([c.score(**{name:rows[name][i:i+1] for name in joint._NAMES}) for i in range(5)])
    np.testing.assert_array_equal(scores,single)
    predictions=c.predict(**rows)
    np.testing.assert_array_equal(predictions,np.concatenate([c.predict(**{name:rows[name][i:i+1] for name in joint._NAMES}) for i in range(5)]))
    assert scores.shape==(5,len(c.classes)) and info['score_physical_count']==5
    assert info['prediction_policy']=='all_registered_classes_both_groups_evaluated'
    np.testing.assert_allclose(np.exp(scores).sum(axis=1),1.,rtol=1e-12,atol=1e-12)
    oldcols=c.final_cache['old_columns']; raw=b.score(**rows)
    np.testing.assert_allclose(scores[:,oldcols[0]]-scores[:,oldcols[1]],raw[:,0]-raw[:,1],rtol=1e-12,atol=1e-12)


def test_K1_no_held_new_one_and_rank0():
    _,b,p,args,_=sequence_args(k=1,c=7,seed=193)
    assert p.audit['no_information'] and not p.problems
    c=joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',**LIMITS)
    assert c.audit['optimizer_steps']==0 and c.audit['final_head_fit_count']==1
    np.testing.assert_array_equal(c.U,b.U)
    np.testing.assert_array_equal(c.final_cache['new_alpha'],0.)
    np.testing.assert_array_equal(c.final_cache['new_intercept'],0.)
    # Rank0 is a coordinate boundary; the nonzero raw kernel/head remains.
    p0=replace(p,r=0,W=np.empty((8,0)),audit=dict(p.audit_dict(),no_information=True,no_information_reason='ZERO_DICTIONARY_RANK'))
    c0=joint.fit_group_barrier_joint_local_ridge(p0,mode='C_seq',**LIMITS)
    assert c0.Z.shape==(736,0) and np.any(c0.final_cache['K'])


@pytest.mark.parametrize('gamma',[None,0.])
def test_zero_kernel_full_free_constants(sequence,gamma):
    _,_,p,_,_,_=sequence
    problem=replace(p.full_problem,gamma=gamma)
    cf=joint._forward(problem,p.anchor_U,LIMITS)
    np.testing.assert_array_equal(cf['K'],0.)
    np.testing.assert_allclose(cf['new_intercept'],cf['Ynew'].mean(axis=0),atol=1e-14)
    assert cf['gate'].b>np.max(cf['bounds'])
    assert cf['gate'].audit['logistic_record_evaluations']>=len(problem.train_labels)
    g=joint._backward(cf,np.ones(len(problem.held_labels)),np.empty_like(cf['hheld']),{})
    np.testing.assert_array_equal(g,0.)


def test_tau0_exact_original_equivalence_and_no_adapter_gradient(sequence):
    _,_,p,_,_,_=sequence
    problem=replace(p.problems[0],tau=0.)
    U=p.anchor_U+np.ones_like(p.anchor_U)*.001
    cf=joint._forward(problem,U,LIMITS)
    np.testing.assert_array_equal(cf['distance'],problem.d0)
    np.testing.assert_array_equal(cf['K'],problem.gamma*(problem.d0==0))
    g=joint._backward(cf,np.ones(len(problem.held_labels)),np.zeros_like(cf['hheld']),{})
    np.testing.assert_array_equal(g,0.)


def test_complete_callbacks_cost_scope_and_failure_retention(sequence):
    _,_,p,_,_,_=sequence; emitted={}; logs=[]
    def save(key,arrays):
        emitted[key]={name:np.array(value,copy=True) for name,value in arrays.items()}
        return dict(storage='synthetic-memory',key=key)
    c=joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',state_callback=save,log_callback=logs.append,**LIMITS)
    final=emitted[c.audit['final_state_ref']['key']]
    for name in ('gate_slacks','gate_lower_bounds','gate_D_eff','gate_b','new_alpha','new_intercept','bounds','prior_B_U'):
        assert name in final
    assert c.audit['group_gate_forward_newton_iterations']>=c.audit['final_fit']['group_gate_forward_newton_iterations']
    assert c.audit['new_ridge_fit_count']>=1 and c.audit['gate_peak_factor_buffer_bytes']>0
    assert c.audit['extra_ground_data_stat_payload_bytes']==0 and c.audit['incremental_transmission_bytes'] is None
    for record in logs:
        json.dumps(joint.compact_training_record(record),allow_nan=False)
    bad_limits=dict(LIMITS,max_factor_buffer_bytes=1)
    # Small physical K1 case isolates a final-head resource failure.
    _,b,short,args,_=sequence_args(k=1,c=7,seed=593)
    short=replace(short,audit=dict(short.audit_dict(),group_barrier_resources=bad_limits))
    with pytest.raises(joint.NumericalFailure) as err:
        joint.fit_group_barrier_joint_local_ridge(short,mode='C_seq',**bad_limits)
    failure=err.value
    assert failure.audit['status']=='TECHNICAL_FAILURE' and failure.audit['failure_state_ref']
    np.testing.assert_array_equal(failure.arrays['last_accepted_U'],b.U)
    assert 'gate_failure_lower_bounds' in failure.arrays
    assert failure.audit['new_ridge_fit_count']==1


def test_new_ridge_second_solve_failure_keeps_actual_work_and_partial_arrays(sequence,monkeypatch):
    _,b,p,_,_,_=sequence
    real=joint.solve_triangular; calls=[]
    def fail_second(chol,rhs,**kwargs):
        calls.append(rhs.shape)
        if len(calls)==2: raise np.linalg.LinAlgError('synthetic second new-ridge solve failure')
        return real(chol,rhs,**kwargs)
    monkeypatch.setattr(joint,'solve_triangular',fail_second)
    with pytest.raises(joint.NumericalFailure) as err:
        joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',**LIMITS)
    failure=err.value; audit=failure.audit
    assert len(calls)==2
    assert audit['new_ridge_factorization_attempts']==audit['new_ridge_factorizations_completed']==1
    assert audit['new_ridge_triangular_calls']==2 and audit['new_ridge_triangular_completed_calls']==1
    assert audit['new_ridge_triangular_rhs_columns']==sum(shape[1] for shape in calls)
    assert audit['new_ridge_triangular_rhs_elements']==sum(n*q for n,q in calls)
    assert audit['new_ridge_triangular_dense_work_units']==sum(n*n*q for n,q in calls)
    assert audit['new_ridge_seconds']>0 and audit['group_gate_forward_factorization_attempts']==0
    for name in ('new_rhs','new_factor_input','new_chol','new_lower_solution'):
        assert name in failure.arrays
    assert 'new_alpha' not in failure.arrays
    np.testing.assert_array_equal(failure.arrays['last_accepted_U'],b.U)


def test_new_ridge_adjoint_second_solve_failure_keeps_actual_work(sequence,monkeypatch):
    _,_,p,_,_,_=sequence
    cf=joint._forward(p.problems[0],p.anchor_U,LIMITS)
    v=np.full(len(cf['gheld']),.1); F=np.full_like(cf['hheld'],.01)
    joint._backward(cf,v,F,{})
    old_rhs=cf['adjoint_arrays']['new_adjoint_rhs'].copy()
    assert 'new_T' in cf['adjoint_arrays'] and 'new_intercept_adjoint' in cf['adjoint_arrays']
    real=joint.solve_triangular; calls=[]
    def fail_second(chol,rhs,**kwargs):
        calls.append(rhs.shape)
        if len(calls)==2: raise np.linalg.LinAlgError('synthetic second new-ridge adjoint failure')
        return real(chol,rhs,**kwargs)
    monkeypatch.setattr(joint,'solve_triangular',fail_second)
    progress={}
    with pytest.raises(np.linalg.LinAlgError) as err:
        joint._backward(cf,2*v,2*F,progress)
    failure=err.value
    assert len(calls)==2 and progress['new_ridge_adjoint_calls']==1
    assert progress['new_ridge_adjoint_triangular_calls']==2
    assert progress['new_ridge_adjoint_triangular_completed_calls']==1
    assert progress['new_ridge_adjoint_rhs_columns']==sum(shape[1] for shape in calls)
    assert progress['new_ridge_adjoint_rhs_elements']==sum(n*q for n,q in calls)
    assert progress['new_ridge_adjoint_seconds']>0
    assert failure.group_head_audit is progress
    for name in ('new_adjoint_rhs','new_g_b','new_adjoint_lower_solution'):
        assert name in failure.group_failed_arrays
    assert 'new_adjoint_solved_rhs' not in failure.group_failed_arrays
    assert 'new_T' not in failure.group_failed_arrays and 'new_intercept_adjoint' not in failure.group_failed_arrays
    assert 'adjoint_arrays' not in cf
    np.testing.assert_allclose(failure.group_failed_arrays['new_adjoint_rhs'],2*old_rhs,rtol=1e-12,atol=1e-12)


def test_reused_successful_cache_gate_failure_excludes_old_ridge_adjoint(sequence,monkeypatch):
    _,_,p,_,_,_=sequence
    cf=joint._forward(p.problems[0],p.anchor_U,LIMITS)
    v=np.full(len(cf['gheld']),.1); F=np.full_like(cf['hheld'],.01)
    joint._backward(cf,v,F,{})
    names=('new_adjoint_rhs','new_g_b','new_adjoint_lower_solution','new_adjoint_solved_rhs',
           'new_T','new_intercept_adjoint')
    assert all(name in cf for name in names)
    def gate_failure(state,*,L,G):
        raise gate.GroupBarrierFailure('synthetic gate-before-new-ridge failure',gate._ledger(),dict(L=L,G=G))
    monkeypatch.setattr(gate,'group_barrier_gate_vjp',gate_failure)
    progress={}
    with pytest.raises(gate.GroupBarrierFailure) as err:
        joint._backward(cf,2*v,2*F,progress)
    failure=err.value
    assert 'adjoint_arrays' not in cf
    assert all(name not in cf and name not in failure.group_failed_arrays for name in names)
    # The current immutable forward head still belongs to this legal cache;
    # only prior backward products disappear.
    assert 'new_rhs' in failure.group_failed_arrays and 'new_alpha' in failure.group_failed_arrays
    np.testing.assert_array_equal(failure.group_failed_arrays['gate_failure_G'],2*v)
    assert progress.get('new_ridge_adjoint_calls',0)==0


def test_fit_nonfinite_failure_reaches_real_archive_callback_and_preserves_origin(sequence,monkeypatch,tmp_path):
    _,b,p,_,_,_=sequence
    original=np.array([np.nan,np.inf,-np.inf,2.],dtype=np.float64)
    def fail_gate(**kwargs):
        raise gate.GroupBarrierFailure('SYNTHETIC_NONFINITE_GATE_EVIDENCE',gate._ledger(),dict(bad_input=original))
    monkeypatch.setattr(gate,'fit_group_barrier_gate',fail_gate)
    saved={}
    def archive(key,arrays):
        path=tmp_path/(key+'.npz')
        np.savez_compressed(path,**arrays)
        saved[key]=path
        return dict(storage='npz',path=str(path))
    with pytest.raises(joint.NumericalFailure) as err:
        joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',state_callback=archive,**LIMITS)
    failure=err.value
    assert failure.code=='SYNTHETIC_NONFINITE_GATE_EVIDENCE'
    assert failure.audit['failure_code']==failure.code and failure.audit['failure_archive_status']=='CALLBACK_RETURNED'
    assert failure.__cause__.code==failure.code
    assert failure.original_failure_audit['failure_code']==failure.code
    np.testing.assert_array_equal(failure.original_failure_arrays['bad_input'],original)
    np.testing.assert_array_equal(failure.arrays['gate_failure_bad_input'],original)
    np.testing.assert_array_equal(failure.arrays['last_accepted_U'],b.U)
    assert failure.audit['new_ridge_factorizations_completed']==1 and failure.audit['new_ridge_triangular_completed_calls']==2
    path=saved['failure']
    with np.load(path,allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded['gate_failure_bad_input'],original)
    assert failure.audit['failure_state_ref']['arrays']['gate_failure_bad_input']['nonfinite_count']==3
    with pytest.raises(ValueError): failure.arrays['gate_failure_bad_input'].setflags(write=True)
    # The failure-only route cannot weaken successful state validation.
    recorder=joint._Recorder(archive)
    with pytest.raises(FloatingPointError): recorder.save('success',bad_input=original)
    assert 'success' not in saved


def test_failure_archive_error_keeps_primary_code_and_nonfinite_arrays(sequence,monkeypatch):
    _,_,p,_,_,_=sequence
    def fail_gate(**kwargs):
        raise gate.GroupBarrierFailure('SYNTHETIC_ORIGINAL_GATE_FAILURE',gate._ledger(),dict(raw=np.array([np.inf,np.nan])))
    monkeypatch.setattr(gate,'fit_group_barrier_gate',fail_gate)
    def broken_archive(key,arrays):
        if key.startswith('failure'): raise OSError('synthetic archive device failure')
        return dict(storage='synthetic-only')
    with pytest.raises(joint.NumericalFailure) as err:
        joint.fit_group_barrier_joint_local_ridge(p,mode='C_seq',state_callback=broken_archive,**LIMITS)
    failure=err.value
    assert failure.code=='SYNTHETIC_ORIGINAL_GATE_FAILURE'
    assert failure.audit['failure_archive_status']=='FAILED'
    assert 'synthetic archive device failure' in failure.audit['failure_archive_error']
    assert failure.audit['failure_state_ref'] is None
    assert np.isinf(failure.arrays['gate_failure_raw'][0]) and np.isnan(failure.arrays['gate_failure_raw'][1])
