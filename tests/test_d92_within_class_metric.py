import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi import d92_within_class_metric as mod
from cvsrffi import d92_branch_local_ridge as local
from cvsrffi import d92_branch_interaction as interaction


def fixture(c=3,k=3,seed=18):
    rng=np.random.default_rng(seed)
    features={name:rng.normal(size=(c*k,dim)) for name,dim in
        [('z_id',160),('fft',96),('t_emb',160),('f_emb',160),('pa_local',160)]}
    return dict(**features,support_labels=np.repeat(np.arange(c),k),
        support_ids=[f'p{i:04}' for i in range(c*k)],classes=[f'c{i}' for i in range(c)])


def features(args,index=slice(None)):
    return {key:args[key][index] for key in ('z_id','fft','t_emb','f_emb','pa_local')}


def phi(b,a):return np.concatenate((b,a,np.einsum('ni,nj->nij',b,a).reshape(len(b),-1)),axis=1)


def explicit_oracle(b,a,labels,xb,xa,yb,ya):
    x=phi(b,a)
    means=np.stack([x[labels==j].mean(axis=0) for j in sorted(set(labels))])
    means-=means.mean(axis=0)
    e=np.stack([x[i]-x[labels==labels[i]].mean(axis=0) for i in range(len(x))])
    r=e-np.linalg.lstsq(means.T,e.T,rcond=1e-12)[0].T@means
    f=r/np.linalg.norm(r)
    delta=phi(xb,xa)[:,None,:]-phi(yb,ya)[None,:,:]
    dots=np.einsum('ijd,nd->ijn',delta,f)
    solution=np.linalg.solve(np.eye(len(f))+f@f.T,dots.reshape(-1,len(f)).T).T.reshape(dots.shape)
    return np.sum(delta*delta,axis=-1)-np.sum(dots*solution,axis=-1)


def test_frozen_json_matches_module():
    assert json.loads((ROOT/'configs/d92_within_class_metric_frozen_20260930.json').read_text(encoding='utf-8'))=={'algorithm':mod.FROZEN_CONFIG}


def test_metric_matches_explicit_full_phi_oracle_and_bounds():
    args=fixture();state=mod.fit_within_class_metric(**args)
    b,a=interaction._blocks(**features(args))
    rng=np.random.default_rng(8)
    xb,yb=rng.normal(size=(2,256)),rng.normal(size=(3,256))
    xa,ya=rng.normal(size=(2,480)),rng.normal(size=(3,480))
    actual,_=mod._metric_distances(state,xb,xa,yb,ya)
    expected=explicit_oracle(b,a,args['support_labels'],xb,xa,yb,ya)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
    base=local._distances(xb,xa,yb,ya)
    assert np.all(actual>=.5*base) and np.all(actual<=base)
    audit=state.audit_dict()
    assert audit['metric_factorization_count']==1 and audit['protected_rank']==2
    assert audit['protection_relative_error']<=audit['numerical_tolerance']
    assert sum(audit['state_array_bytes'].values())==audit['persistent_state_bytes']
    with pytest.raises(ValueError):state.T[0,0]=2


def test_unchanged_old_mean_directions_and_nearest_centroid_ranking():
    args=fixture();state=mod.fit_within_class_metric(**args)
    b,a=interaction._blocks(**features(args));labels=args['support_labels']
    coords=mod._delta_psi(b,a,b[:1],a[:1],state.Qb,state.Qa)
    means=np.stack([coords[labels==j].mean(axis=0) for j in range(3)])
    differences=means-means[0]
    np.testing.assert_allclose(differences@state.T.T,0,atol=1e-12)
    x=coords[0]+.4*coords[-1]
    d0=np.sum((x-means)**2,axis=1)
    dw=d0-np.sum(((x-means)@state.T.T)**2,axis=1)
    np.testing.assert_allclose(dw-dw[0],d0-d0[0],atol=1e-12)


def test_row_and_class_permutation_equivariance():
    args=fixture();a=mod.fit_within_class_metric(**args)
    order=np.arange(9)[::-1];rename=[2,0,1]
    changed=dict(**features(args,order),support_labels=np.array(rename)[args['support_labels'][order]],
        support_ids=np.array(args['support_ids'])[order].tolist(),classes=['c1','c2','c0'])
    b=mod.fit_within_class_metric(**changed)
    np.testing.assert_allclose(a.distances(**features(args)),b.distances(**features(args)),rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize('k',[1,3])
def test_identity_exact_baseline_and_zero_state_work(k):
    args=fixture(k=k)
    if k>1:
        for val in features(args).values():
            for j in range(3):val[j*k:(j+1)*k]=val[j*k]
    metric=mod.fit_within_class_metric(**args)
    assert metric.identity and metric.T.size==0
    base=local.fit_branch_local_ridge(**args,old_classes=args['classes'])
    state=mod.fit_within_class_local_ridge(**args,old_classes=args['classes'],metric=metric,baseline_state=base)
    assert state.base_state is base
    np.testing.assert_array_equal(state.score(**features(args)),base.score(**features(args)))
    assert state.audit_dict()['metric_head_fit_count']==0
    assert metric.audit_dict()['metric_factorization_count']==0


def test_nonzero_residual_inside_mean_span_is_explicit_failure():
    args=fixture(c=2,k=3)
    for value in features(args).values():value[:]=0
    args['z_id'][[0,1,3],0]=1
    args['z_id'][[2,4,5],1]=1
    with pytest.raises(mod.NumericalFailure,match='UNRESOLVED_ZERO_RESIDUAL') as error:
        mod.fit_within_class_metric(**args,context={'fold':2,'row_id':'test'})
    audit=error.value.audit_dict()
    assert audit['fold']==2 and audit['training_physical_ids']==args['support_ids']
    json.dumps(audit,allow_nan=False)


def test_near_orthogonal_near_duplicate_uses_unprojected_error_scale():
    qb=np.zeros((256,1));qb[:2,0]=1/np.sqrt(2)
    qa=np.zeros((480,1));qa[0,0]=1
    t=np.array([[.5,0.,0.]])
    state=mod.WithinClassMetricState(False,('old',),('zero',),np.array([0]),
        np.zeros((1,256)),np.zeros((1,480)),qb,qa,t,
        {'numerical_tolerance':128*np.finfo(float).eps*480})
    x=np.zeros((1,256));x[0,:2]=[1.,-1.]
    y=x.copy();y[0,0]=np.nextafter(1.,2.)
    a=np.zeros((1,480))
    actual,counts=mod._metric_distances(state,x,a,y,a)
    delta=mod._delta_psi(x,a,y,a,qb,qa)[0]
    d0=local._distances(x,a,y,a)[0,0]
    expected=d0-np.sum((t@delta)**2)
    assert counts['direct_difference_pair_count']==1
    assert actual[0,0]==expected
    assert .5*d0<=actual[0,0]<d0


def test_head_trace_and_original_ridge_equations_and_batch_independence():
    args=fixture();metric=mod.fit_within_class_metric(**args)
    state=mod.fit_within_class_local_ridge(**args,old_classes=args['classes'],metric=metric)
    b,a=interaction._blocks(**features(args));distances=local._distances(b,a)
    final=state.audit_dict()['final_fit']
    s0=np.sum(distances[np.triu_indices(9,1)])/9
    assert final['interaction_centered_trace']==s0
    assert final['transformed_interaction_centered_trace']<s0
    assert final['normal_equation_residual']<=final['numerical_tolerance']
    assert final['loss_total']==final['loss_data']+final['loss_ridge']
    joint=state.score(**features(args))
    single=np.concatenate([state.score(**features(args,slice(i,i+1))) for i in range(9)])
    np.testing.assert_array_equal(joint,single)
    np.testing.assert_array_equal(state.score(**features(args,np.arange(9)[::-1])),joint[::-1])
    assert state.score(**features(args,slice(0,0))).shape==(0,3)


def test_C_inherits_metric_but_rejects_changed_old_binding():
    args=fixture(c=4);old=fixture(c=3)
    # The fixture RNG generated each block at a different total size, so bind
    # the first three classes explicitly to exactly the old feature cache.
    for key in features(args):args[key][:9]=old[key]
    metric=mod.fit_within_class_metric(**old)
    state=mod.fit_within_class_local_ridge(**args,old_classes=old['classes'],metric=metric)
    assert state.metric is metric and len(state.classes)==4
    args['z_id'][0,0]+=.1
    with pytest.raises(ValueError,match='features changed'):
        mod.fit_within_class_local_ridge(**args,old_classes=old['classes'],metric=metric)


def test_LOCO_all_classes_train_only_and_identity_skip():
    args=fixture();events=[]
    result=mod.diagnose_leave_one_class_out(**args,log_callback=events.append)
    assert result['diagnostic_fit_count']==3 and len(events)==3
    for row in result['folds']:
        assert not set(row['held_physical_ids'])&set(row['training_physical_ids'])
        assert len(row['training_physical_ids'])==6
        assert row['held_class'] not in row['metric_audit']['classes']
        assert 0<=row['held_class_contraction_fraction']<=.5
    skip=mod.diagnose_leave_one_class_out(**fixture(k=1))
    assert skip['diagnostic_fit_count']==0 and skip['folds']==[]


def test_failure_state_json_nonfinite_tagged():
    error=mod.NumericalFailure('x',{'partial':np.array([np.inf,np.nan]),'completed_stages':[{'stage':'B'}]})
    assert json.loads(json.dumps(error.audit_dict(),allow_nan=False))['partial']==['inf','nan']


def test_context_derived_fields_are_authoritative_without_duplicate_keywords():
    args=fixture();context=dict(train_k=999,train_physical_count=999,classes=['wrong'],parent_k=20,fold=2)
    metric=mod.fit_within_class_metric(**args,context=context)
    assert metric.audit_dict()['train_k']==3
    assert metric.audit_dict()['train_physical_count']==9
    assert metric.audit_dict()['classes']==args['classes']
    head=mod.fit_within_class_local_ridge(**args,old_classes=args['classes'],metric=metric,context=context)
    assert head.audit_dict()['train_k']==3
    assert head.audit_dict()['parent_k']==20
    assert head.audit_dict()['final_fit']['train_k']==3
    loco=mod.diagnose_leave_one_class_out(**args,context=context)
    assert all(row['metric_audit']['train_k']==3 for row in loco['folds'])
    one=fixture(k=1);identity=mod.fit_within_class_metric(**one,context=context)
    base=local.fit_branch_local_ridge(**one,old_classes=one['classes'])
    reused=mod.fit_within_class_local_ridge(**one,old_classes=one['classes'],metric=identity,
        baseline_state=base,context=context)
    assert reused.audit_dict()['train_k']==1 and reused.audit_dict()['classes']==one['classes']


@pytest.mark.parametrize('n,k',[(26,1),(364,14),(520,20)])
def test_synthetic_registered_head_sizes(n,k):
    args=fixture(c=26,k=k,seed=27)
    old=dict(**features(args,slice(0,6*k)),support_labels=args['support_labels'][:6*k],
        support_ids=args['support_ids'][:6*k],classes=args['classes'][:6])
    metric=mod.fit_within_class_metric(**old)
    base=local.fit_branch_local_ridge(**args,old_classes=old['classes']) if k==1 else None
    head=mod.fit_within_class_local_ridge(**args,old_classes=old['classes'],metric=metric,baseline_state=base)
    assert head.score(**features(args,slice(0,1))).shape==(1,26)
    assert len(head.base_state.support_background)==n
    assert head.audit_dict()['optimizer_steps']==0
