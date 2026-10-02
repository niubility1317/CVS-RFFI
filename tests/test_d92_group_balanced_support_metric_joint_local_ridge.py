"""Synthetic weighted-gate composition; no actual data or numerical child run."""
if __name__=='__main__':
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),root/'code/cvsrffi/d92_group_balanced_support_metric_joint_local_ridge.py'):
        raw=path.read_bytes();source=raw.decode('utf-8',errors='strict')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source
        ast.parse(source,filename=str(path))
    print('UTF8/AST PASS: group-balanced joint/test; no numerical imports')
    raise SystemExit(0)

from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_d92_support_metric_joint_local_ridge as literal
from cvsrffi import d92_group_balanced_support_metric_joint_local_ridge as joint
from cvsrffi import d92_support_metric_joint_local_ridge as original
from cvsrffi import d92_group_balanced_barrier_gate as gate

CONTEXT=dict(literal.CONTEXT,run_id='synthetic-group-balanced-support-metric')
LIMITS=literal.LIMITS


def prepare(args,**kwargs):
    return joint.prepare_group_balanced_support_metric_joint_training(**args,**LIMITS,context=CONTEXT,**kwargs)


@pytest.fixture(scope='module')
def sequence():
    args=literal.fixture(k=3,c=8,rank=2);old=literal.old_args(args)
    bp=prepare(old);b=joint.fit_group_balanced_support_metric_joint_local_ridge(bp)
    cp=prepare(args,inherited=b);c=joint.fit_group_balanced_support_metric_joint_local_ridge(cp,mode='C_seq')
    return args,old,bp,b,cp,c


def test_distinct_method_exact_current_B_and_no_prior_method_adapter(sequence):
    args,old,bp,b,cp,c=sequence
    assert b.audit['schema']==joint.SCHEMA and b.audit['method']==joint.METHOD
    assert joint.FROZEN_CONFIG['gate_weight_rule']=='N/(2*C_group*n_class)'
    for key,value in original.FROZEN_CONFIG.items():
        if key not in ('schema','method'):assert joint.FROZEN_CONFIG[key]==value
    assert c.prior is b and cp.basis is b.basis and cp.anchor.shape==(2,)
    np.testing.assert_array_equal(cp.anchor,b.theta)
    assert c.audit['final_prior_ref']==cp.audit['full_prior_ref']==b.audit['final_state_ref']
    # B has precisely the existing head/CE-only metric definition; the old
    # method is fitted solely on this literal support to test formula identity.
    before=original.fit_support_metric_joint_local_ridge(original.prepare_support_metric_joint_training(
        **old,**LIMITS,context=CONTEXT))
    np.testing.assert_allclose(b.theta,before.theta,rtol=0,atol=0)
    np.testing.assert_allclose(b.head['alpha'],before.head['alpha'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='actual current U-coordinate B'):prepare(args,inherited=before)
    np.testing.assert_array_equal(c.to_arrays()['actual_B_theta'],b.theta)


def test_weights_only_actual_inner_train_and_full_old_new_barrier(sequence):
    _,_,_,b,p,c=sequence
    _,_,info,cache=joint.evaluate_group_balanced_support_metric_joint_objective(p,p.anchor)
    for i,(problem,head) in enumerate(zip(p.folds,cache.heads)):
        saved=cache.arrays;prefix=f'fold_{i}_'
        labels=problem.labels;weights=saved[prefix+'gate_weights'];targets=saved[prefix+'gate_targets']
        np.testing.assert_array_equal(saved[prefix+'gate_labels'],labels)
        for y in np.unique(labels):
            where=labels==y;group=targets[where][0]
            group_classes=len(np.unique(labels[targets==group]))
            np.testing.assert_allclose(weights[where],len(labels)/(2*group_classes*np.count_nonzero(where)),rtol=0,atol=0)
        assert saved[prefix+'gate_slacks'].shape==(len(problem.labels)*6//8,2)
        assert np.all(saved[prefix+'gate_slacks']>0)
        assert head['gate'].audit['schema']==gate.SCHEMA
        for group in (0,1):np.testing.assert_allclose(weights[targets==group].sum(),len(labels)/2,rtol=3e-16)
        assert set(p.audit['folds'][i]['old_inner_train_ids']).isdisjoint(problem.held_ids)
        teacher=p.records[p.audit['folds'][i]['prior_ref']['key']]
        np.testing.assert_array_equal(teacher['theta'],b.theta)
        assert teacher['K'].shape==(12,12)
    assert info['actual_work']['gate_forward_weight_constructions_completed']==len(p.folds)
    assert info['actual_work']['gate_jvp_weight_constructions_completed']==0
    assert info['actual_work']['gate_jvp_triangular_rhs_columns']==12*len(p.folds)
    # A held-label permutation changes neither the weighted head input nor its
    # inner train prior. The legally supervised outer labels stay in p.labels.
    changed=replace(p,folds=tuple(replace(problem,held_labels=problem.held_labels[::-1]) for problem in p.folds))
    other=joint.evaluate_group_balanced_support_metric_joint_objective(changed,p.anchor)[3]
    for i in range(len(p.folds)):
        np.testing.assert_array_equal(other.arrays[f'fold_{i}_gate_weights'],cache.arrays[f'fold_{i}_gate_weights'])
    final=c.to_arrays()
    np.testing.assert_array_equal(final['gate_labels'],c.labels)
    assert final['gate_slacks'].shape==(18,2) and np.all(final['gate_slacks']>0)


def test_full_U_C_kernel_free_ridge_weighted_gate_and_bounds_JVP(sequence):
    p=sequence[4];theta=p.anchor+np.array([.019,-.011]);direction=np.array([.3,-.4]);eps=2e-5
    loss,g,info,cache=joint.evaluate_group_balanced_support_metric_joint_objective(p,theta)
    plus=joint.evaluate_group_balanced_support_metric_joint_objective(p,theta+eps*direction,jacobian=False)
    minus=joint.evaluate_group_balanced_support_metric_joint_objective(p,theta-eps*direction,jacobian=False)
    np.testing.assert_allclose(g@direction,(plus[0]-minus[0])/(2*eps),rtol=4e-4,atol=3e-7)
    np.testing.assert_allclose(np.einsum('ncr,r->nc',cache.score_jacobian,direction),
        (plus[3].scores-minus[3].scores)/(2*eps),rtol=6e-4,atol=3e-6)
    assert np.linalg.norm(cache.arrays['fold_0_lower_bounds_jacobian'][...,:2])>0
    for fold in range(len(p.folds)):
        prefix=f'fold_{fold}_'
        for value,derivative in (('lower_bounds','lower_bounds_jacobian'),('new_intercept','new_intercept_jacobian'),
                                 ('gate_b','gate_b_jacobian'),('gate_alpha','gate_alpha_jacobian')):
            analytic=np.einsum('...r,r->...',cache.arrays[prefix+derivative][...,:2],direction)
            numeric=(plus[3].arrays[prefix+value]-minus[3].arrays[prefix+value])/(2*eps)
            np.testing.assert_allclose(analytic,numeric,rtol=6e-4,atol=3e-6)
        np.testing.assert_array_equal(plus[3].arrays[prefix+'gate_weights'],minus[3].arrays[prefix+'gate_weights'])
    assert info['loss_proximal']==0 and loss==info['RMSCE']


def test_fixed_one_metric_step_halving_and_actual_work(sequence):
    for state in (sequence[3],sequence[5]):
        a=state.audit_dict();trials=a['trials'];work=a['actual_work']
        assert 0<=a['optimizer_steps']<=1 and a['trial_count']==len(trials)<=12
        assert work['support_metric_step_calls']==1
        assert all(t['step_size']==.5**i for i,t in enumerate(trials))
        assert all(t['accepted']==(t['loss_after']<=t['armijo_rhs']+t['comparison_tolerance']) for t in trials)
        assert all(not t['accepted'] or i==len(trials)-1 for i,t in enumerate(trials))
        assert a['final_objective']['RMSCE']==(trials[-1]['loss_after'] if a['optimizer_steps'] else a['initial_objective']['RMSCE'])
        assert set(work)==set(joint.WORK_SUM_KEYS)|set(joint.WORK_MAX_KEYS)
        assert a['source_validation'] is None and a['complete_head_jvp_error_bound'] is None
    work=sequence[5].audit['actual_work']
    operations=sequence[5].audit['operation_audits']
    for key in gate.WEIGHT_SUM_KEYS:
        actual=sum(op['audit'].get(key,0) for op in operations if op['operation']=='gate_forward')
        assert work['gate_forward_'+key]==actual
    assert work['gate_forward_weight_constructions_completed']==work['gate_forward_calls']


@pytest.mark.parametrize('k,rank',[(1,2),(3,0)])
def test_K1_and_rank0_no_adapter_but_complete_weighted_C(k,rank):
    args=literal.fixture(k=k,c=8,rank=rank);old=literal.old_args(args)
    bp=prepare(old);b=joint.fit_group_balanced_support_metric_joint_local_ridge(bp)
    cp=prepare(args,inherited=b);c=joint.fit_group_balanced_support_metric_joint_local_ridge(cp,mode='C_seq')
    assert c.audit['optimizer_steps']==c.audit['trial_count']==0
    assert c.audit['final_head_complete'] and c.head['gate'].weights.shape==(k*8,)
    assert c.head['gate'].slacks.shape==(k*6,2)
    assert c.audit['actual_work']['gate_forward_calls']==1
    assert c.audit['actual_work']['gate_forward_weight_constructions_completed']==1
    assert c.audit['trainable_parameter_count']==0


def test_new0_same_B_object_no_gate_or_second_fit(sequence):
    old,b=sequence[1],sequence[3];p=prepare(old,inherited=b)
    assert p.new0 and p.inherited is b
    assert joint.fit_group_balanced_support_metric_joint_local_ridge(p,mode='C_seq') is b
    assert p.audit['actual_work']['gate_forward_calls']==0


def test_frozen_old_function_and_public_single_record_prediction(sequence):
    args,b,c=sequence[0],sequence[3],sequence[5]
    for row in (0,7,19):
        feature={name:args[name][row:row+1] for name in joint.BRANCHES}
        scores,audit=c.score_with_audit(**feature)
        old=b.score(**feature);old_columns=np.asarray([c.classes.index(v) for v in b.classes])
        for i in range(1,len(b.classes)):
            np.testing.assert_allclose(scores[0,old_columns[i]]-scores[0,old_columns[0]],old[0,i]-old[0,0],atol=3e-14,rtol=3e-14)
        assert joint.predict_group_balanced_support_metric_joint_local_ridge(c,**feature).shape==(1,)
        assert scores.shape==(1,8) and audit['single_record_all_registered_classes']
        assert audit['actual_work']['gate_forward_calls']==0
    with pytest.raises(ValueError,match='one physical'):c.score(**{name:args[name][:2] for name in joint.BRANCHES})


def test_full_numeric_archive_and_failure_weight_cost(tmp_path,sequence):
    old=sequence[1];args=sequence[0];paths=[]
    def save(key,arrays):
        path=tmp_path/(str(len(paths))+'_'+key+'.npz');np.savez_compressed(path,**arrays);paths.append((key,path))
        return dict(path=str(path),key=key)
    bp=prepare(old,state_callback=save);b=joint.fit_group_balanced_support_metric_joint_local_ridge(bp,state_callback=save)
    cp=prepare(args,inherited=b,state_callback=save);c=joint.fit_group_balanced_support_metric_joint_local_ridge(cp,mode='C_seq',state_callback=save)
    final=next(path for key,path in reversed(paths) if key=='final')
    with np.load(final,allow_pickle=False) as values:
        for key in ('gate_labels','gate_weights','gate_class_labels','gate_class_counts','gate_class_targets','gate_qeff','gate_D_eff','actual_B_theta'):
            np.testing.assert_array_equal(values[key],c.to_arrays()[key])
    assert json.loads(json.dumps(c.audit_dict(),allow_nan=False))['method']==joint.METHOD
    # A resource failure is preserved after its actual weights were constructed.
    changed=replace(cp,limits=joint._freeze(dict(cp.limits,max_factor_buffer_bytes=1)))
    with pytest.raises(joint.GroupBalancedSupportMetricFailure) as caught:
        joint.fit_group_balanced_support_metric_joint_local_ridge(changed,mode='C_seq',state_callback=save)
    failure=caught.value
    assert failure.audit['status']=='TECHNICAL_FAILURE'
    assert failure.audit['actual_work']['gate_forward_weight_constructions_completed']==1
    assert any('weights' in key for key in failure.arrays)
    assert any(key=='failure' for key,path in paths)


def test_old_source_and_immutable_arrays_not_shared_mutably(sequence):
    assert original.fit_support_metric_joint_local_ridge.__globals__['_head'] is original._head
    assert original.FROZEN_CONFIG['method']!=joint.FROZEN_CONFIG['method']
    weights=sequence[5].to_arrays()['gate_weights']
    with pytest.raises(ValueError):weights.setflags(write=True)
    with pytest.raises(TypeError):sequence[5].head['gate'].audit['weight_rule']='other'
