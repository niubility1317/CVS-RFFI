import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import minimize
from cvsrffi import d92_branch_orbit_ce as core
from cvsrffi import d92_branch_interaction as base

KEYS=('z_id','fft','t_emb','f_emb','pa_local')


def data(k=2,c=3,seed=930):
    rng=np.random.default_rng(seed)
    values={key:rng.normal(size=(k*c,96) if key=='fft' else (k*c,4,160)).astype(np.float32) for key in KEYS}
    values.update(support_labels=np.repeat(np.arange(c),k),support_ids=['id%03d'%i for i in range(k*c)],
        classes=['class%02d'%i for i in range(c)],old_classes=['class00'])
    return values


def features(value):return {key:value[key] for key in KEYS}
def first(value):return {key:value[key] if key=='fft' else value[key][:,0] for key in KEYS}


def test_orbit_kernel_is_normalized_mean_of_full_tensor_features():
    rng=np.random.default_rng(531);b=rng.normal(size=(7,4,3));a=rng.normal(size=(7,4,2))
    phi=np.concatenate((b,a,np.einsum('nvi,nvj->nvij',b,a).reshape(7,4,-1)),axis=2).mean(axis=1)
    scales,diagonal=core._scale(b,a)
    np.testing.assert_allclose(diagonal,np.sum(phi*phi,axis=1),atol=1e-14)
    phi=phi*scales[:,None]
    np.testing.assert_allclose(core._kernel(b,a,b,a,scales,scales,True),phi@phi.T,atol=2e-14)
    assert np.linalg.eigvalsh(core._kernel(b,a,b,a,scales,scales,True)).min()>-1e-12


def test_true_rkhs_gradient_and_ce_solution_match_independent_primal():
    rng=np.random.default_rng(92);x=rng.normal(size=(8,5));x-=x.mean(axis=0)
    g=x@x.T;labels=np.repeat(np.arange(4),2);alpha=rng.normal(size=(8,4))
    logits,loss,penalty,norm=core._ce_stats(alpha,g,labels)
    probabilities=np.exp(logits-logits.max(axis=1,keepdims=True));probabilities/=probabilities.sum(axis=1,keepdims=True)
    expected=x.T@(alpha+probabilities-np.eye(4)[labels])
    assert norm==pytest.approx(np.linalg.norm(expected))
    def objective(flat):
        w=flat.reshape(5,4);s=x@w;m=s.max(axis=1,keepdims=True);e=np.exp(s-m);p=e/e.sum(axis=1,keepdims=True)
        value=float(np.sum(m[:,0]+np.log(e.sum(axis=1))-s[np.arange(8),labels])+.5*np.sum(w*w))
        return value,(x.T@(p-np.eye(4)[labels])+w).ravel()
    expected=minimize(objective,np.zeros(20),jac=True,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-15,'maxiter':1000})
    fitted,audit=core._ce_fit(g,labels,4)
    np.testing.assert_allclose(g@fitted,x@expected.x.reshape(5,4),atol=2e-6)
    assert audit['loss_total']==pytest.approx(expected.fun,abs=1e-11)
    assert audit['gradient_norm']<=audit['gradient_tolerance']
    assert audit['lipschitz_bound']==pytest.approx(1+.5*np.trace(g))
    assert len(audit['steps'])==audit['optimizer_steps']+1
    assert [r['iteration'] for r in audit['steps']]==list(range(audit['iterations']+1))
    for record in audit['steps']:assert record['loss_total']==pytest.approx(record['loss_data']+record['loss_ridge'])


@pytest.mark.parametrize('tensor',[False,True])
def test_nearly_cancelled_orbits_have_stable_explicit_normalization(tensor):
    rng=np.random.default_rng(8);v=rng.normal(size=(3,2,256));v/=np.linalg.norm(v,axis=2,keepdims=True)
    last=-v[:,1]+1e-8*v[:,0];last/=np.linalg.norm(last,axis=1,keepdims=True)
    b=np.stack((v[:,0],v[:,1],-v[:,0],last),axis=1)
    a=np.zeros((3,4,480))
    if tensor:
        # Nonzero bilinear means also nearly cancel, while their individual terms are large.
        u=rng.normal(size=(3,480));u/=np.linalg.norm(u,axis=1,keepdims=True)
        a[:]=u[:,None,:]
    explicit=np.concatenate((b.mean(axis=1),a.mean(axis=1),
        np.einsum('nvi,nvj->nij',b,a).reshape(3,-1)/4.),axis=1)
    scale,diagonal=core._scale(b,a)
    np.testing.assert_allclose(diagonal,np.sum(explicit*explicit,axis=1),rtol=1e-7,atol=1e-30)
    gram=core._kernel(b,a,b,a,scale,scale,True)
    expected=(explicit*scale[:,None])@(explicit*scale[:,None]).T
    np.testing.assert_allclose(gram,expected,rtol=1e-7,atol=1e-12)
    np.testing.assert_allclose(np.diag(gram),3.,atol=2e-14)
    np.testing.assert_allclose(gram,gram.T,atol=2e-14)
    assert np.linalg.eigvalsh(gram).min()>-1e-12


def test_tensor_cancellation_self_norm_uses_stable_factorization():
    rng=np.random.default_rng(91);u=rng.normal(size=(3,256));v=rng.normal(size=(3,480))
    u/=np.linalg.norm(u,axis=1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True)
    b=np.stack((u,u,-u,-u),axis=1)
    a=np.stack((v,-v,v,-v),axis=1)
    a[:,3]+=1e-8*v
    explicit=np.concatenate((b.mean(axis=1),a.mean(axis=1),
        np.einsum('nvi,nvj->nij',b,a).reshape(3,-1)/4.),axis=1)
    scale,diagonal=core._scale(b,a)
    gram=core._kernel(b,a,b,a,scale,scale,True)
    np.testing.assert_allclose(diagonal,np.sum(explicit*explicit,axis=1),rtol=2e-7,atol=1e-29)
    np.testing.assert_allclose(gram,(explicit*scale[:,None])@(explicit*scale[:,None]).T,rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(np.diag(gram),3.,atol=2e-14)


@pytest.mark.parametrize('k,c',[(1,1),(1,26),(5,3),(20,3)])
def test_single_ridge_exact_legacy_equivalence(k,c):
    inputs=data(k,c)
    state=core.fit_branch_orbit_ce(**inputs,arm='single_ridge')
    previous=base.fit_branch_interaction(**first(inputs),**{key:inputs[key] for key in ('support_labels','support_ids','classes','old_classes')})
    query=features(data(2,c,888))
    np.testing.assert_array_equal(state.alpha,previous.alpha)
    np.testing.assert_array_equal(state.score(**query),previous.score(**first(query)))
    assert state.audit_dict()['final_fit']['loss_total']==previous.audit_dict()['final_fit']['loss_total']


@pytest.mark.parametrize('arm',['orbit_ridge','orbit_ce'])
def test_c4_view_permutation_invariance_and_query_batch_independence(arm):
    inputs=data(3,3);state=core.fit_branch_orbit_ce(**inputs,arm=arm)
    query=features(data(3,3,551));score=state.score(**query)
    rotated={key:np.roll(v,1,axis=1) if key!='fft' else v for key,v in query.items()}
    np.testing.assert_allclose(score,state.score(**rotated),atol=2e-12)
    changed=copy.deepcopy(inputs)
    for key in KEYS:
        if key!='fft':changed[key]=np.roll(changed[key],2,axis=1)
    other=core.fit_branch_orbit_ce(**changed,arm=arm)
    np.testing.assert_allclose(score,other.score(**query),atol=2e-12)
    chunks=np.concatenate([state.score(**{key:v[i:i+2] for key,v in query.items()}) for i in range(0,9,2)])
    np.testing.assert_array_equal(score,chunks)
    order=np.random.default_rng(42).permutation(9)
    np.testing.assert_array_equal(score[order],state.score(**{key:v[order] for key,v in query.items()}))


def test_duplicated_views_do_not_multiply_physical_loss_or_claim_shots():
    inputs=data(3,3)
    for key in KEYS:
        if key!='fft':inputs[key]=np.repeat(inputs[key][:,:1],4,axis=1)
    original=core.fit_branch_orbit_ce(**inputs,arm='single_ce')
    orbit=core.fit_branch_orbit_ce(**inputs,arm='orbit_ce')
    np.testing.assert_allclose(original.score(**features(inputs)),orbit.score(**features(inputs)),atol=2e-12)
    assert original.audit_dict()['final_fit']['physical_loss_mass']==orbit.audit_dict()['final_fit']['physical_loss_mass']==9
    assert orbit.audit_dict()['k']==3


@pytest.mark.parametrize('constant',[0.,1e-30,1.,1e30])
def test_constant_support_exact_ties_and_empty_queries(constant):
    inputs=data(1,3);inputs['classes'],inputs['old_classes']=['z','a','m'],['a']
    for key in KEYS:inputs[key].fill(constant)
    state=core.fit_branch_orbit_ce(**inputs)
    query=features(data(2,3))
    np.testing.assert_array_equal(state.score(**query),np.zeros((6,3)))
    np.testing.assert_array_equal(state.predict(**query),np.full(6,'a'))
    assert state.audit_dict()['optimizer_steps']==0
    assert state.score(**{key:v[:0] for key,v in query.items()}).shape==(0,3)


def test_state_bytes_immutability_and_role_class_order():
    inputs=data(3,3);state=core.fit_branch_orbit_ce(**inputs)
    names=['support_background','support_auxiliary','alpha','reference_kernel','center_mean','target_mean','support_scale']
    assert state.audit_dict()['persistent_state_bytes']==sum(getattr(state,k).nbytes for k in names)+16
    for name in names:
        with pytest.raises(ValueError):getattr(state,name).setflags(write=True)
    changed=copy.deepcopy(inputs);order=np.random.default_rng(83).permutation(9)
    for key in KEYS+('support_labels',):changed[key]=changed[key][order]
    changed['support_ids']=[changed['support_ids'][i] for i in order]
    changed['classes']=changed['classes'][::-1];changed['support_labels']=2-changed['support_labels'];changed['old_classes']=['class02']
    other=core.fit_branch_orbit_ce(**changed)
    np.testing.assert_array_equal(state.alpha[:,::-1],other.alpha)
    np.testing.assert_array_equal(state.predict(**features(inputs)),other.predict(**features(inputs)))
    json.dumps(state.audit_dict(),allow_nan=False)


def test_iteration_cap_failure_keeps_optimization_evidence(monkeypatch):
    monkeypatch.setitem(core._CONFIG,'ce_max_iterations',0)
    with pytest.raises(core.OrbitCEConvergenceError) as captured:core.fit_branch_orbit_ce(**data())
    assert captured.value.audit['status']=='TECHNICAL_FAILURE'
    assert captured.value.audit['converged'] is False and captured.value.audit['steps'][0]['iteration']==0
    json.dumps(captured.value.audit,allow_nan=False)


def test_nonfinite_iteration_failure_retains_prior_finite_trace(monkeypatch):
    original=core._ce_stats;calls=[]
    def fail_second(*args):
        calls.append(True)
        if len(calls)==2:raise FloatingPointError('synthetic arithmetic failure')
        return original(*args)
    monkeypatch.setattr(core,'_ce_stats',fail_second)
    with pytest.raises(core.OrbitCEConvergenceError) as captured:core.fit_branch_orbit_ce(**data())
    audit=captured.value.audit
    assert audit['status']=='TECHNICAL_FAILURE' and audit['converged'] is False
    assert audit['attempted_iteration']==1 and [s['iteration'] for s in audit['steps']]==[0]
    json.dumps(audit,allow_nan=False)


@pytest.mark.parametrize('failed_ce,scope,fold,trial,completed',[(1,'support_oof',0,None,1),(3,'support_oof',1,None,5),(5,'support_oneshot_proxy',None,0,9)])
def test_later_fit_failure_keeps_episode_context_and_completed_stages(monkeypatch,failed_ce,scope,fold,trial,completed):
    original=core._ce_fit;calls=[]
    def fail_later(*args):
        calls.append(True)
        if len(calls)==failed_ce:raise core.OrbitCEConvergenceError('synthetic later fit failure',dict(status='TECHNICAL_FAILURE',converged=False,steps=[]))
        return original(*args)
    monkeypatch.setattr(core,'_ce_fit',fail_later)
    with pytest.raises(core.OrbitCEConvergenceError) as captured:core.probe_branch_orbit_ce(**data(2,3))
    audit=captured.value.audit
    assert (audit['scope'],audit['fold'],audit['trial'])==(scope,fold,trial)
    assert audit['arm']=='single_ce' and audit['parent_k']==2 and audit['train_k']==1
    assert len(audit['training_physical_ids'])==3 and len(audit['held_ids'])==3
    assert set(audit['training_physical_ids']).isdisjoint(audit['held_ids'])
    assert len(audit['completed_stages'])==completed
    assert all(stage['converged'] is True for stage in audit['completed_stages'])
    json.dumps(audit,allow_nan=False)


def test_true_k1_no_fit_proxy_and_complete_probe_physical_coverage(monkeypatch):
    original=core._fit
    def reject(*args,**kwargs):raise AssertionError('true K1 probe fit forbidden')
    monkeypatch.setattr(core,'_fit',reject)
    result=core.probe_branch_orbit_ce(**data(1,3))
    assert result['oof'] is result['oneshot_proxy'] is None and result['optimizer_steps']==0
    monkeypatch.setattr(core,'_fit',original)
    inputs=data(2,3);result=core.probe_branch_orbit_ce(**inputs)
    assert result['standard_factorization_count']==4 and result['factorization_count']==8
    stages=[]
    for entry in result['folds']+result['oneshot_proxy']['trials']:
        assert set(entry['training_ids']).isdisjoint(entry['held_ids'])
        assert set(entry['training_ids'])|set(entry['held_ids'])==set(inputs['support_ids'])
        assert len(entry['stages'])==4
        stages+=entry['stages']
        for stage in entry['stages']:
            assert stage['converged'] is True
            assert set(stage['training_physical_ids'])==set(entry['training_ids'])
            if stage['arm'].endswith('_ce'):
                assert stage['gradient_norm']<=stage['gradient_tolerance']
                assert len(stage['steps'])==stage['iterations']+1
        if 'trial' in entry:
            assert entry['proxy_train_k']==1 and entry['parent_k']==2
            for arm in core._ARMS:
                assert 'rows' not in entry['oof'][arm]
                assert sum(map(sum,entry['oof'][arm]['confusion']))==3
    assert result['optimizer_steps']==sum(s['optimizer_steps'] for s in stages)
    assert len(result['paired'])==5
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('fault',['shape','nan','duplicate','imbalance','arm'])
def test_invalid_inputs(fault):
    inputs=data()
    if fault=='shape':inputs['z_id']=inputs['z_id'][:,0]
    if fault=='nan':inputs['f_emb'][0,2,0]=np.nan
    if fault=='duplicate':inputs['support_ids'][0]=inputs['support_ids'][1]
    if fault=='imbalance':inputs['support_labels'][0]=1
    if fault=='arm':inputs['arm']='query_select'
    with pytest.raises((ValueError,FloatingPointError)):core.fit_branch_orbit_ce(**inputs)


def test_frozen_config_exact():
    path=Path(__file__).resolve().parents[1]/'configs/d92_branch_orbit_ce_frozen_20260929.json'
    assert json.loads(path.read_text(encoding='utf-8'))=={'algorithm':core.FROZEN_CONFIG}


def test_single_class_ce_has_exact_zero_function():
    state=core.fit_branch_orbit_ce(**data(5,1))
    np.testing.assert_array_equal(state.score(**features(data(2,1))),np.zeros((2,1)))
    assert state.audit_dict()['optimizer_steps']==0


@pytest.mark.parametrize('k',[1,20])
def test_synthetic_full_c26_budget(k):
    from threadpoolctl import threadpool_limits
    inputs=data(k,26,seed=930)
    with threadpool_limits(limits=2):
        for arm in core._ARMS:
            state=core.fit_branch_orbit_ce(**inputs,arm=arm)
            audit=state.audit_dict();stage=audit['final_fit']
            assert stage['converged'] is True
            print(json.dumps(dict(synthetic_only=True,c=26,k=k,arm=arm,blas_threads=2,
                iterations=stage['iterations'],fit_seconds=audit['fit_seconds'],
                solve_seconds=stage['solve_seconds'],gradient_norm=stage['gradient_norm'],
                gradient_tolerance=stage.get('gradient_tolerance'),state_bytes=audit['persistent_state_bytes']),allow_nan=False))
