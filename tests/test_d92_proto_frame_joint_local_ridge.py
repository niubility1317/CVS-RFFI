"""Literal synthetic contracts; root owns all numerical execution."""
if __name__ == '__main__':
    import ast
    from pathlib import Path
    base=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),base/'code/cvsrffi/d92_proto_frame_joint_local_ridge.py'):
        raw=path.read_bytes();source=raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source
        ast.parse(source,filename=str(path))
        print('STATIC_AST_UTF8_VERIFIED',path)
    raise SystemExit(0)

from dataclasses import replace
import json

import numpy as np
import pytest

from cvsrffi import d92_proto_frame_joint_local_ridge as joint
from cvsrffi import d92_proto_frame_primitives as primitive
from cvsrffi import d92_group_barrier_gate as gate

LIMITS=dict(max_newton_iterations=100,max_line_search_trials=64,max_factor_buffer_bytes=16_000_000)
CONTEXT=dict(run_id='synthetic-proto',row_id='row',split_id='literal',scope='support',fold=None,trial=None)


def fixture(k=3,c=8):
    rng=np.random.default_rng(7301);classes=tuple('class-'+str(i) for i in range(c))
    labels=np.repeat(np.arange(c),k);raw={}
    for name,width in zip(joint.BRANCHES,(160,96,160,160,160)):
        centers=rng.normal(size=(c,width))
        raw[name]=centers[labels]+.7*rng.normal(size=(k*c,width))
    frame=primitive.build_proto_frame_dictionary(prototypes=rng.normal(size=(6,160)),classes=classes[:6])
    return dict(**raw,support_labels=labels,support_ids=tuple(f'physical-{i}-{j}' for i in range(c) for j in range(k)),
        classes=classes,old_classes=classes[:6],prototype_frame=frame)


def old_args(args):
    mask=args['support_labels']<6
    return dict(**{k:args[k][mask] for k in joint.BRANCHES},support_labels=args['support_labels'][mask],
        support_ids=tuple(v for i,v in enumerate(args['support_ids']) if mask[i]),classes=args['old_classes'],
        old_classes=args['old_classes'],prototype_frame=args['prototype_frame'])


def prepare(args,**kwargs):
    return joint.prepare_proto_frame_joint_training(**args,**LIMITS,context=CONTEXT,**kwargs)


@pytest.fixture(scope='module')
def sequence():
    args=fixture();old=old_args(args);bp=prepare(old)
    b=joint.fit_proto_frame_joint_local_ridge(bp)
    cp=prepare(args,inherited=b);c=joint.fit_proto_frame_joint_local_ridge(cp,mode='C_seq')
    return args,old,bp,b,cp,c


def test_actual_independent_B_C_and_exact_new0(sequence,monkeypatch):
    args,old,bp,b,cp,c=sequence
    assert b.mode=='B' and c.prior is b
    assert b.audit['method']=='D92-ProtoFrameTangent-GGN1-LocalRidge'
    np.testing.assert_array_equal(cp.anchor,b.theta)
    assert np.linalg.norm(b.theta)<=.5 and np.linalg.norm(c.theta-b.theta)<=.5
    assert c.audit['final_prior_ref']==b.audit['final_state_ref']
    n0=prepare(old,inherited=b)
    def forbidden(*a,**kw):raise AssertionError('new0 executed a head')
    monkeypatch.setattr(joint,'_head',forbidden)
    assert joint.fit_proto_frame_joint_local_ridge(n0,mode='C_seq') is b
    assert all(v==0 for v in n0.audit['actual_work'].values())
    for key in ('run_id','row_id','split_id','scope','fold','trial'):
        context=dict(CONTEXT);context[key]='other'
        with pytest.raises(ValueError,match='lineage crosses'):
            joint.prepare_proto_frame_joint_training(**args,**LIMITS,context=context,inherited=b)
    changed=dict(args,z_id=args['z_id'].copy());changed['z_id'][0,0]+=.01
    with pytest.raises(ValueError,match='records/labels changed'):prepare(changed,inherited=b)


def test_physical_fold_teacher_isolation_and_frozen_theta(sequence):
    _,_,_,b,p,c=sequence
    for index,problem in enumerate(p.folds):
        audit=p.audit['folds'][index]
        oldtrain=set(audit['old_inner_train_ids'])
        assert oldtrain<=set(problem.ids) and oldtrain.isdisjoint(problem.held_ids)
        assert len(oldtrain)==12 and len(problem.held_ids)==8
        saved=p.records[audit['prior_ref']['key']]
        np.testing.assert_array_equal(saved['theta'],b.theta)
        assert saved['K'].shape==(12,12)
        # Independently rebuild this legal teacher's free-intercept system.
        K,Y,L=saved['K'],saved['Y'],saved['L'];n=len(K)
        Q=np.block([[np.eye(n)+K,np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
        solution=np.linalg.solve(Q,np.vstack((Y,np.zeros((1,6)))))
        oracle=L@solution[:-1]+solution[-1]
        np.testing.assert_allclose(np.vstack((problem.prior_train,problem.prior_held)),oracle,atol=2e-12)
    assert any(key.startswith('preparation_fold_') for key in c.state_records())


def test_fixed_original_old_geometry_matches_existing_rule_and_counts_QR(sequence):
    p=sequence[2]
    for problem in p.folds+(p.full_problem,):
        geometry=primitive.branch_geometry(**problem.train)
        d,s0,tau,_,_=joint.local._geometry(geometry.b,geometry.a,problem.labels,6)
        radial=-2*np.sum(joint.local._radial_minus_one(d,tau)[np.triu_indices(len(d),1)])/len(d)
        np.testing.assert_allclose(problem.tau,tau,rtol=2e-14)
        np.testing.assert_allclose(problem.gamma,s0/radial,rtol=2e-14)
    assert p.audit['actual_work']['fixed_old_distances_distance_evaluation_count']==4
    assert p.audit['actual_work']['fixed_old_distances_distance_qr_pair_count']>0


@pytest.mark.parametrize('stage',['B','C'])
def test_complete_score_and_RMS_jacobian_directional_difference(sequence,stage):
    p=sequence[2] if stage=='B' else sequence[4]
    theta=p.anchor+np.array([.021,-.013,.017,.006,-.009])
    direction=np.array([.3,-.4,.2,.1,-.25]);epsilon=2e-5
    loss,g,info,cache=joint.evaluate_proto_frame_joint_objective(p,theta)
    plus=joint.evaluate_proto_frame_joint_objective(p,theta+epsilon*direction,jacobian=False)
    minus=joint.evaluate_proto_frame_joint_objective(p,theta-epsilon*direction,jacobian=False)
    np.testing.assert_allclose(g@direction,(plus[0]-minus[0])/(2*epsilon),rtol=3e-4,atol=2e-7)
    np.testing.assert_allclose(np.einsum('ncp,p->nc',cache.score_jacobian,direction),
        (plus[3].scores-minus[3].scores)/(2*epsilon),rtol=5e-4,atol=2e-6)
    assert np.linalg.eigvalsh(cache.arrays['damped_hessian']).min()>=1-1e-12
    assert info['loss_proximal']==0 and info['RMSCE']==loss
    assert plus[1] is None and plus[3].score_jacobian is None
    if stage=='C':
        assert info['actual_work']['gate_jvp_calls']==3
        assert info['actual_work']['gate_jvp_factorization_attempts']==3
        assert info['actual_work']['gate_jvp_triangular_rhs_columns']==36  # 3 folds x 2 triangles x 6 RHS
        assert plus[2]['actual_work']['gate_jvp_calls']==0
        assert np.linalg.norm(cache.arrays['fold_0_lower_bounds_jacobian'])>0


def gate_fixture():
    rng=np.random.default_rng(7401);X=rng.normal(size=(6,4));K=X@X.T/4+.3*np.eye(6)
    L=rng.normal(size=(3,6))*.1;bounds=np.array([[.1,-.1],[.2,.05],[-.2,.15],[.1,.1]])
    t=np.array([1,1,1,1,0,0.]);old=np.arange(4)
    state=gate.fit_group_barrier_gate(K=K,targets=t,old_indices=old,lower_bounds=bounds,**LIMITS)
    dK=rng.normal(size=(6,6,5))*.04;dK=(dK+dK.transpose(1,0,2))/2
    return state,L,dK,rng.normal(size=(3,6,5))*.08,rng.normal(size=(4,2,5))*.09


def test_gate_full_RHS_free_intercept_JVP_and_detached_negative_control():
    state,L,dK,dL,da=gate_fixture();ledger=joint._Ledger()
    J,arrays=joint._gate_jvp(state,dK,L,dL,da,ledger);direction=np.array([.2,.4,-.3,.1,.25]);epsilon=1e-5
    values=[]
    for sign in (1,-1):
        K=state.K+sign*epsilon*np.einsum('ijp,p->ij',dK,direction)
        a=state.lower_bounds+sign*epsilon*np.einsum('ijp,p->ij',da,direction)
        s=gate.fit_group_barrier_gate(K=K,targets=state.targets,old_indices=state.old_indices,lower_bounds=a,**LIMITS)
        values.append((L+sign*epsilon*np.einsum('ijp,p->ij',dL,direction))@s.alpha+s.b)
    np.testing.assert_allclose(J@direction,(values[0]-values[1])/(2*epsilon),rtol=3e-5,atol=1e-7)
    detached,_=joint._gate_jvp(state,dK,L,dL,np.zeros_like(da),joint._Ledger())
    assert np.linalg.norm(J-detached)>1e-8
    assert np.linalg.norm(arrays['gate_b_jacobian'])>1e-8
    G=np.array([.7,-.2,.4]);reverse=gate.group_barrier_gate_vjp(state,L=L,G=G)
    expected=np.einsum('ij,ijp->p',reverse.gradK,dK)+np.einsum('ij,ijp->p',reverse.gradL,dL)+np.einsum('ij,ijp->p',reverse.gradlower_bounds,da)
    np.testing.assert_allclose(G@J,expected,rtol=1e-9,atol=1e-10)
    assert ledger.work['gate_jvp_triangular_calls']==2 and ledger.work['gate_jvp_triangular_rhs_columns']==12


def test_gate_JVP_second_solve_failure_charges_partial(monkeypatch):
    state,L,dK,dL,da=gate_fixture();real=gate.solve_triangular;calls=[];ledger=joint._Ledger()
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise ArithmeticError('SYNTHETIC_SECOND_TRIANGLE')
        return real(*args,**kwargs)
    monkeypatch.setattr(gate,'solve_triangular',fail_second)
    with pytest.raises(ArithmeticError,match='SYNTHETIC_SECOND_TRIANGLE') as caught:
        joint._gate_jvp(state,dK,L,dL,da,ledger)
    assert ledger.work['gate_jvp_factorizations_completed']==1
    assert ledger.work['gate_jvp_triangular_calls']==2 and ledger.work['gate_jvp_triangular_rhs_columns']==12
    assert 'gate_jvp_rhs' in caught.value.proto_partial_arrays
    assert 'gate_jvp_solved' not in caught.value.proto_partial_arrays


def test_actual_one_update_trials_archive_and_native_scalar_logs(sequence):
    _,_,_,b,_,c=sequence
    for state in (b,c):
        a=state.audit_dict();json.dumps(a,allow_nan=False)
        assert a['optimizer_steps'] in (0,1) and a['trial_count']<=12
        assert a['nominal_parameter_count']==5 and a['parameter_buffer_bytes']==40
        assert a['resident_numeric_state_bytes']==joint._numeric_bytes(state)
        assert a['process_peak_bytes'] is None and a['wire_bytes'] is None
        assert a['actual_work']['ridge_factorization_attempts']>=4
        assert 'initial' in state.records and 'final' in state.records
        for i,trial in enumerate(a['trials']):
            assert trial['step_size']==.125*.5**i
            assert trial['accepted']==(trial['loss_after']<=trial['armijo_rhs']+trial['comparison_tolerance'])
        assert sum(x['accepted'] for x in a['trials'])==a['optimizer_steps']
    assert c.audit['actual_work']['gate_forward_calls']>=4


@pytest.mark.parametrize('q',[1,2])
def test_K1_full_final_heads_no_OOF_and_single_sample_all_classes(q):
    args=fixture(k=1,c=6+q);bp=prepare(old_args(args));b=joint.fit_proto_frame_joint_local_ridge(bp)
    cp=prepare(args,inherited=b);events=[];saved={}
    def callback(key,arrays):
        saved[key]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return dict(key=key,storage='synthetic',arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
    c=joint.fit_proto_frame_joint_local_ridge(cp,mode='C_seq',state_callback=callback,log_callback=events.append)
    for state in (b,c):
        assert state.audit['no_held'] and state.audit['initial_objective'] is None
        assert state.audit['optimizer_steps']==0 and state.audit['final_head_complete']
    assert c.audit['actual_work']['gate_forward_calls']==1
    assert c.audit['actual_work']['ridge_factorization_attempts']==1
    assert set(saved)=={'final'} and events[-1]['event']=='PROTO_FRAME_FINAL'
    compact=joint.compact_training_record(events[-1]);json.dumps(compact,allow_nan=False)
    assert all(v is None or type(v) in (str,int,float,bool) for v in compact.values())
    row={k:args[k][0:1] for k in joint.BRANCHES};scores,audit=c.score_with_audit(**row)
    assert scores.shape==(1,6+q) and audit['single_record_all_registered_classes']
    np.testing.assert_allclose(np.exp(scores).sum(),1,atol=2e-14)
    values,parts=c._score(joint._raw(**row),joint._Ledger());B,new,g,lo,ln=parts
    np.testing.assert_array_equal(B,b.score(**row))
    oi,ni=int(B[0].argmax()),int(new[0].argmax());gap=g[0]+lo[0,oi]-ln[0,ni]
    old=c.classes[c.head['old_columns'][oi]];novel=c.classes[c.head['new_columns'][ni]]
    expected=old if gap>0 else novel if gap<0 else min(old,novel)
    assert c.predict(**row).tolist()==[expected]
    if q==1:np.testing.assert_array_equal(ln,np.zeros((1,1)))
    with pytest.raises(ValueError,match='exactly one'):c.score(**{k:args[k][:2] for k in joint.BRANCHES})


def test_zero_dictionary_and_zero_kernel_keep_complete_constant_heads():
    args=fixture(k=2,c=7)
    args['prototype_frame']=primitive.build_proto_frame_dictionary(prototypes=np.zeros((6,160)),classes=args['old_classes'])
    for k in joint.BRANCHES:args[k]=np.zeros_like(args[k])
    b=joint.fit_proto_frame_joint_local_ridge(prepare(old_args(args)))
    p=prepare(args,inherited=b);c=joint.fit_proto_frame_joint_local_ridge(p,mode='C_seq')
    np.testing.assert_array_equal(b.theta,np.zeros(5));np.testing.assert_array_equal(c.theta,np.zeros(5))
    assert p.full_problem.gamma is None
    assert c.audit['actual_work']['gate_forward_calls']==3  # 2 folds + final
    assert len(c.head['gate'].targets)==14 and c.head['gate'].lower_bounds.shape==(12,1)
    assert np.all(c.head['gate'].slacks>0)
    assert np.isfinite(c.head['gate'].b)


def test_negative_raw_old_margin_and_all_tied_pair_bounds_are_preserved(sequence):
    prepared=sequence[4];problem=prepared.folds[0]
    prior=np.array(problem.prior_train,copy=True)
    oi=np.flatnonzero(problem.labels<6)
    prior[oi]=0.
    for i in oi:prior[i,int(problem.labels[i])]=-.4
    changed=replace(problem,prior_train=joint._seal(prior));arrays={}
    head=joint._head(changed,prepared,prepared.anchor,False,joint._Ledger(),arrays)
    np.testing.assert_array_equal(arrays['raw_old_margin'],np.full(len(oi),-.4))
    h=arrays['new_train_scores'];lognew=joint._logsoftmax(h)
    lse=np.log(np.exp(prior[oi]).sum(axis=1))
    np.testing.assert_allclose(arrays['lower_bounds'],lse[:,None]+lognew[oi],atol=2e-14)
    assert head['gate'].slacks.shape==(len(oi),2)
    assert np.all(head['gate'].slacks>0)


def test_exact_equivalence_kernel_keeps_duplicate_physical_rows():
    args=fixture(k=2,c=7)
    for name in joint.BRANCHES:args[name][1]=args[name][0]
    raw=joint._raw(**{k:args[k] for k in joint.BRANCHES});ledger=joint._Ledger()
    original=joint._geometry(raw,ledger)
    result=joint._kernel(raw,None,original,None,args['prototype_frame'],np.ones(5)*.04,0.,1.,True,ledger)
    assert result.kernel.shape==(14,14) and result.kernel[0,1]==1.
    np.testing.assert_array_equal(result.jacobian,np.zeros((14,14,5)))


def test_gate_buffer_failure_does_not_return_B_or_hide_new_ridge_work():
    args=fixture(k=1,c=7);limits=dict(LIMITS,max_factor_buffer_bytes=1)
    bp=joint.prepare_proto_frame_joint_training(**old_args(args),**limits,context=CONTEXT)
    b=joint.fit_proto_frame_joint_local_ridge(bp)
    cp=joint.prepare_proto_frame_joint_training(**args,**limits,context=CONTEXT,inherited=b)
    with pytest.raises(joint.ProtoFrameFailure) as caught:joint.fit_proto_frame_joint_local_ridge(cp,mode='C_seq')
    failure=caught.value
    assert failure.audit['status']=='TECHNICAL_FAILURE'
    assert failure.audit['actual_work']['ridge_factorizations_completed']==1
    assert failure.audit['actual_work']['gate_forward_calls']==1
    assert 'K' in failure.arrays and failure.arrays['K'].shape==(7,7)


def test_immutable_sealed_state_and_no_caller_alias(sequence):
    args,_,_,b,_,c=sequence
    for value in (c.theta,c.labels,c.head['gate'].alpha,c.raw['z_id'],c.prototype_frame.Q):
        with pytest.raises(ValueError):value.flat[0]=99
        with pytest.raises(ValueError):value.setflags(write=True)
    with pytest.raises(TypeError):c.audit['status']='OTHER'
    assert not np.shares_memory(args['z_id'],c.raw['z_id'])
    values=c.to_arrays();assert 'actual_B_theta' in values and values['gate_b'].shape==()
    assert values['gate_zeta'].shape==() and not values['theta'].flags.writeable
    assert c.audit['source_validation'] is None


@pytest.mark.parametrize('name',list(LIMITS))
@pytest.mark.parametrize('value',[True,0,-1,1.5])
def test_explicit_positive_resource_guards(name,value):
    limits=dict(LIMITS);limits[name]=value
    with pytest.raises(ValueError,match='explicit positive integer'):
        joint.prepare_proto_frame_joint_training(**old_args(fixture(k=1)),**limits)


def test_failure_keeps_primary_nonfinite_snapshot_and_actual_completed_work(sequence,monkeypatch):
    _,_,_,_,prepared,_=sequence
    original=gate.fit_group_barrier_gate
    def fail(**kwargs):
        audit=gate._ledger();audit['factorization_attempts']=1
        raise gate.GroupBarrierFailure('SYNTHETIC_GATE_FAILURE',audit,dict(attempt=np.array([np.nan,np.inf,-np.inf])))
    monkeypatch.setattr(gate,'fit_group_barrier_gate',fail)
    archived={}
    def callback(key,arrays):
        archived[key]=arrays
        raise OSError('SYNTHETIC_ARCHIVE_DEVICE_FAILURE')
    with pytest.raises(joint.ProtoFrameFailure) as caught:
        joint.fit_proto_frame_joint_local_ridge(prepared,mode='C_seq',state_callback=callback)
    failure=caught.value;assert 'SYNTHETIC_GATE_FAILURE' in str(failure)
    assert failure.audit['actual_work']['ridge_factorizations_completed']==1
    assert failure.audit['actual_work']['gate_forward_factorization_attempts']==1
    assert failure.audit['failure_archive_error']=='SYNTHETIC_ARCHIVE_DEVICE_FAILURE'
    assert any(np.isnan(a).any() for a in failure.arrays.values())
    assert 'failure' in archived
    assert isinstance(failure.__cause__,joint.ProtoFrameFailure)
    assert isinstance(failure.__cause__.__cause__,gate.GroupBarrierFailure)
