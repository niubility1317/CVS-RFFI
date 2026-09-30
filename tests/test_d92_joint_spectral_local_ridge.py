import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi import d92_joint_spectral_local_ridge as mod
from cvsrffi import d92_branch_local_ridge as local


def fixture(c=3,k=3,seed=118):
    rng=np.random.default_rng(seed)
    raw={name:rng.normal(size=(c*k,dim)) for name,dim in [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160)]}
    return dict(**raw,support_labels=np.repeat(np.arange(c),k),support_ids=[f'p{i:04}' for i in range(c*k)],
        classes=[f'c{i}' for i in range(c)],old_classes=[f'c{i}' for i in range(c)])


def raw(args,ix=slice(None)):return {key:args[key][ix] for key in mod._NAMES}


def test_frozen_config_and_projection():
    assert json.loads((ROOT/'configs/d92_joint_spectral_frozen_20260930.json').read_text(encoding='utf-8'))=={'algorithm':mod.FROZEN_CONFIG}
    np.testing.assert_array_equal(mod.project_theta([-.1,.2]),[0,.2])
    np.testing.assert_allclose(mod.project_theta([.9,.5]),[.7,.3])
    np.testing.assert_array_equal(mod.project_theta([5.,-3.]),[1.,0.])


def test_smooth_analytic_gradient_matches_finite_difference():
    prepared=mod.prepare_joint_training(**fixture(k=4))
    theta=np.array([.23,.17]);anchor=np.array([.02,.04])
    _,g,_=mod.evaluate_joint_objective(prepared,theta,anchor)
    numerical=[]
    for p in range(2):
        delta=np.eye(2)[p]*1e-5
        plus=mod.evaluate_joint_objective(prepared,theta+delta,anchor,gradient=False)[0]
        minus=mod.evaluate_joint_objective(prepared,theta-delta,anchor,gradient=False)[0]
        numerical.append((plus-minus)/2e-5)
    np.testing.assert_allclose(g,numerical,rtol=2e-5,atol=2e-8)


def test_zero_theta_exact_original_inner_forward_and_active_gradient():
    args=fixture(k=4);prepared=mod.prepare_joint_training(**args)
    _,grad,_=mod.evaluate_joint_objective(prepared,np.zeros(2),np.zeros(2))
    assert np.linalg.norm(grad)>1e-10
    for problem in prepared.problems:
        loss,g,audit,parts,actual=mod._evaluate_kernel(problem,np.zeros(2),return_scores=True)
        info=problem.audit
        keep=np.array([pid in info['training_physical_ids'] for pid in prepared.ids])
        base=local.fit_branch_local_ridge(**{key:np.asarray(prepared.raw[key])[keep] for key in mod._NAMES},
            support_labels=np.asarray(prepared.labels,dtype=int)[keep],support_ids=np.array(prepared.ids)[keep],
            classes=prepared.classes,old_classes=prepared.old_classes)
        expected=base.score(**{key:np.asarray(prepared.raw[key])[~keep] for key in mod._NAMES})
        np.testing.assert_array_equal(actual,expected)
    for p in range(2):
        h=np.eye(2)[p]*1e-6
        finite=(mod.evaluate_joint_objective(prepared,h,np.zeros(2),gradient=False)[0]-
            mod.evaluate_joint_objective(prepared,np.zeros(2),np.zeros(2),gradient=False)[0])/1e-6
        np.testing.assert_allclose(grad[p],finite,rtol=5e-4,atol=2e-7)


def test_tied_generalized_derivative_is_symmetric_not_all_directional_derivatives():
    d=np.ones((3,3))-np.eye(3);derivative=np.zeros((2,3,3))
    derivative[0,0,1]=derivative[0,1,0]=1
    derivative[0,0,2]=derivative[0,2,0]=-1
    labels=np.arange(3)
    tau,g=mod._bandwidth(d,derivative,labels)
    assert tau==1 and g[0]==0
    h=1e-6
    positive=(mod._bandwidth(d+h*derivative[0],derivative,labels)[0]-tau)/h
    negative_direction=(mod._bandwidth(d-h*derivative[0],derivative,labels)[0]-tau)/h
    np.testing.assert_allclose([positive,negative_direction],[-1,-1],atol=1e-9)
    order=[2,0,1]
    _,permuted=mod._bandwidth(d[np.ix_(order,order)],derivative[:,order][:,:,order],labels[order])
    np.testing.assert_array_equal(g,permuted)


def test_inner_held_features_cannot_build_its_geometry():
    args=fixture(k=4);first=mod.prepare_joint_training(**args)
    held=set(first.problems[0].audit['held_physical_ids'])
    modified={**args,**{key:value.copy() for key,value in raw(args).items()}}
    ix=np.array([pid in held for pid in args['support_ids']])
    for value in raw(modified).values():value[ix]*=-2
    second=mod.prepare_joint_training(**modified)
    one,two=first.problems[0],second.problems[0]
    assert not held&set(one.geometry.metric.support_ids)
    np.testing.assert_array_equal(one.geometry.metric.T,two.geometry.metric.T)
    np.testing.assert_array_equal(one.geometry.eigenvalues,two.geometry.eigenvalues)
    np.testing.assert_array_equal(one.d0,two.d0)
    assert not np.array_equal(one.cross_d0,two.cross_d0)


def test_unequal_folds_use_physical_sum_and_single_proximal_penalty():
    prepared=mod.prepare_joint_training(**fixture(k=5))
    theta=np.array([.2,.1]);anchor=np.array([.1,.05])
    value,g,audit=mod.evaluate_joint_objective(prepared,theta,anchor)
    pieces=[mod._evaluate_kernel(p,theta) for p in prepared.problems]
    n=len(prepared.ids)
    expected=(sum(x[0] for x in pieces)+.5*np.sum((theta-anchor)**2))/n
    np.testing.assert_allclose(value,expected,rtol=1e-14)
    np.testing.assert_allclose(g,(sum(x[1] for x in pieces)+(theta-anchor))/n,rtol=1e-14)
    assert sorted(len(p.held_labels) for p in prepared.problems)==[3,6,6]


@pytest.mark.parametrize('k',[1,2])
def test_no_information_exact_ridge_without_empty_optimizer_updates(k):
    args=fixture(k=k);prepared=mod.prepare_joint_training(**args)
    base=local.fit_branch_local_ridge(**args)
    state=mod.fit_joint_spectral_local_ridge(prepared,baseline_state=base)
    audit=state.audit_dict()
    assert audit['optimizer_steps']==audit['inner_objective_evaluation_count']==0
    assert audit['no_information'] and audit['identity_forward']
    assert audit['final_head_fit_count']==0
    np.testing.assert_array_equal(state.score(**raw(args)),base.score(**raw(args)))
    fixed=mod.fit_joint_spectral_local_ridge(prepared,mode='fixed',baseline_state=base)
    assert fixed.audit_dict()['optimizer_steps']==0
    assert fixed.audit_dict()['identity_forward']==(k==1)


def test_eight_updates_nine_objectives_real_logs_immutable_and_batch_independent():
    args=fixture(k=3);prepared=mod.prepare_joint_training(**args);events=[]
    state=mod.fit_joint_spectral_local_ridge(prepared,log_callback=events.append)
    audit=state.audit_dict()
    assert audit['optimizer_steps']==8 and audit['inner_objective_evaluation_count']==9
    assert audit['inner_head_fit_count']==27 and audit['inner_factorization_count']==27
    assert audit['derivative_triangular_solve_count']==8*3*4
    steps=[event for event in events if event.get('step') is not None]
    assert [event['step'] for event in steps]==list(range(1,9))
    assert events[-1]['event']=='JOINT_SPECTRAL_FIT' and 'step' not in events[-1]
    assert audit['nonzero_projected_update_count']==sum(event['update_norm']>0 for event in steps)
    joint=state.score(**raw(args))
    singles=np.concatenate([state.score(**raw(args,slice(i,i+1))) for i in range(9)])
    np.testing.assert_array_equal(joint,singles)
    with pytest.raises(ValueError):state.theta[0]=1
    with pytest.raises(ValueError):prepared.problems[0].q[0,0,0]=1
    json.dumps(audit,allow_nan=False)


def test_C_inherits_B_geometry_but_inner_geometry_is_refitted_and_reset_anchor_zero():
    allargs=fixture(c=4,k=3)
    oldargs=dict(**raw(allargs,slice(0,9)),support_labels=allargs['support_labels'][:9],
        support_ids=allargs['support_ids'][:9],classes=allargs['classes'][:3],old_classes=allargs['classes'][:3])
    b=mod.fit_joint_spectral_local_ridge(mod.prepare_joint_training(**oldargs,context={'row_id':'r','split_id':'s'}))
    allargs['old_classes']=oldargs['classes']
    prepared=mod.prepare_joint_training(**allargs,inherited=b,context={'row_id':'r','split_id':'s'})
    assert prepared.geometry is b.geometry
    assert prepared.audit_dict()['geometry_fit_count']==3
    assert all(p.geometry is not b.geometry for p in prepared.problems)
    seq=mod.fit_joint_spectral_local_ridge(prepared,mode='C_seq')
    reset=mod.fit_joint_spectral_local_ridge(prepared,mode='C_reset')
    np.testing.assert_array_equal(seq.audit_dict()['anchor'],b.theta)
    np.testing.assert_array_equal(reset.audit_dict()['anchor'],[0,0])
    assert seq.geometry is reset.geometry is b.geometry
    for problem in prepared.problems:
        assert not set(problem.audit['held_physical_ids'])&set(problem.geometry.metric.support_ids)
    with pytest.raises(ValueError,match='row_id'):
        mod.prepare_joint_training(**allargs,inherited=b,context={'row_id':'other','split_id':'s'})
    allargs['z_id'][0,0]+=.1
    with pytest.raises(ValueError,match='features changed'):mod.prepare_joint_training(**allargs,inherited=b)


def test_row_class_permutations_preserve_shared_spectral_objective():
    args=fixture(k=3);a=mod.prepare_joint_training(**args)
    order=np.arange(9)[::-1]
    changed=dict(**raw(args,order),support_ids=np.array(args['support_ids'])[order],
        support_labels=np.array([2,0,1])[args['support_labels'][order]],classes=['c1','c2','c0'],old_classes=['c2','c0','c1'])
    b=mod.prepare_joint_training(**changed)
    for x,y in zip(a.problems,b.problems):
        np.testing.assert_array_equal(x.q,y.q)
    av,ag,_=mod.evaluate_joint_objective(a,[.2,.3],[0,0]);bv,bg,_=mod.evaluate_joint_objective(b,[.2,.3],[0,0])
    assert av==bv;np.testing.assert_array_equal(ag,bg)


def test_partial_inner_failure_preserves_completed_and_current_costs(monkeypatch):
    prepared=mod.prepare_joint_training(**fixture(k=3))
    original=np.linalg.cholesky;calls=[]
    def fail_second(matrix):
        calls.append(1)
        if len(calls)==2:raise np.linalg.LinAlgError('synthetic second fold failure')
        return original(matrix)
    monkeypatch.setattr(np.linalg,'cholesky',fail_second)
    with pytest.raises(mod.NumericalFailure) as error:mod.fit_joint_spectral_local_ridge(prepared)
    audit=error.value.audit_dict()
    assert audit['optimizer_steps']==0 and audit['inner_objective_evaluation_count']==1
    assert audit['inner_head_fit_count']==2 and audit['inner_factorization_count']==2
    assert audit['derivative_triangular_solve_count']==4
    assert len(audit['failed_stage']['completed_inner_folds'])==1
    assert audit['failed_stage']['current_inner_fold']['inner_fold']==1
    json.dumps(audit,allow_nan=False)
