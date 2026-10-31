"""Full-cluster association and support fallback, without semantic labels."""
from copy import deepcopy
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_cluster_registry_core')


def density():
    c=api();return dict(schema=np.asarray(c.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160),log_abs_det_whitener=np.asarray(0.),pooled_L_variance=np.asarray(1.),
        known_means=np.zeros((90,160)),known_variances=np.ones(90),known_weights=np.full(90,1/90),background_means=np.zeros((1,160)),background_variances=np.ones(1),background_weights=np.ones(1))


def test_source_own_class_population_radius_and_no_mutation():
    c=api();d=density();before=deepcopy(d);x=np.eye(160)[np.repeat(np.arange(6),3)].astype(np.float32);x[:,10]=np.tile([-.1,0,.2],6);y=np.repeat(np.arange(6),3).astype(np.int64)
    source,diag=c.source_statistics(d,x,y);white=x.astype(np.float64)/np.linalg.norm(x.astype(np.float64),axis=1,keepdims=True);means=np.stack([white[y==i].mean(0) for i in range(6)])
    np.testing.assert_allclose(source['source_centers'],means,atol=1e-15);assert float(source['source_radius'])==pytest.approx(np.quantile(np.linalg.norm(white-means[y],axis=1),.95))
    assert diag['source_count']==18 and diag['source_labels_used'] and all(np.array_equal(d[k],before[k]) for k in d)
    with pytest.raises(ValueError):c.source_statistics(d,x[y!=5],y[y!=5])


def test_all_clusters_80_percent_zero_margin_boundary_and_canonical_slots(monkeypatch):
    c=api();d=density();source=dict(schema=np.asarray(c.SOURCE_SCHEMA),source_centers=np.eye(160)[:6].copy(),source_radius=np.asarray(.3))
    rng=np.random.default_rng(7);x=np.repeat(np.eye(160)[[0,1]],20,axis=0)+rng.normal(0,.0001,(40,160));ids=np.asarray([f'id{i:03}' for i in range(40)])
    closed=np.r_[np.zeros(16,np.int64),np.ones(4,np.int64),np.ones(20,np.int64)];margin=np.ones((40,2),np.float64);margin[:20,1]=0; margin[20:,1]=-.1
    monkeypatch.setattr(c,'_cluster',lambda white:np.repeat(np.arange(2),20))
    state,assignment,diag=c.discover(d,source,x.astype(np.float32),closed,margin,ids)
    assert state['labels'].tolist()==[0,6] and state['anonymous_ids'].tolist()==['anon-0001'] and len(state['centers'])==2
    assert np.all(assignment[:20]==0) and np.all(assignment[20:]==1) and diag['all_U_rows_used']
    assert diag['clusters'][0]['head_vote_fraction']==.8 and diag['clusters'][0]['median_predicted_margin']==0
    assert np.all(state['radii']<=source['source_radius'])


def test_actual_HDBSCAN_stable_ID_permutation_and_free_cluster_count():
    c=api();d=density();rng=np.random.default_rng(17);x=np.repeat(np.eye(160)[[0,1]],24,axis=0)+rng.normal(0,.0001,(48,160));x=x.astype(np.float32)
    source=dict(schema=np.asarray(c.SOURCE_SCHEMA),source_centers=np.eye(160)[:6].copy(),source_radius=np.asarray(.5));ids=np.asarray([f'opaque{i:03}' for i in range(48)]);closed=np.repeat(np.arange(2),24).astype(np.int64);m=np.zeros((48,2),np.float64)
    state,a,diag=c.discover(d,source,x,closed,m,ids);p=rng.permutation(48);other,b,_=c.discover(d,source,x[p],closed[p],m[p],ids[p])
    assert len(state['centers'])==2 and len(state['anonymous_ids'])==0 and diag['algorithm']=='sklearn.cluster.HDBSCAN'
    for key in state:np.testing.assert_array_equal(state[key],other[key])
    np.testing.assert_array_equal(a[p],b)


def test_prototypes_override_identity_and_reject_old_unsupported_with_batch_independence(subtests):
    c=api();d=density();state=dict(schema=np.asarray(c.REGISTRY_SCHEMA),source_centers=np.eye(160)[:6].copy(),source_radius=np.asarray(.2),
        centers=np.eye(160)[[1,7]].copy(),radii=np.asarray([.05,.05]),labels=np.asarray([2,6],np.int64),anonymous_ids=np.asarray(['anon-0001']))
    x=np.eye(160)[[1,7,8,0]].astype(np.float32);closed=np.zeros(4,np.int64);m=np.asarray([[-1.,-1.],[-1.,-1.],[1.,1.],[0.,0.]])
    out=c.predict(d,state,x,closed,m);assert out['raw_prediction'][:,3].tolist()==[2,6,-1,0] and out['raw_prediction'][:,1].tolist()==[-1,-1,0,0]
    assert np.array_equal(out['raw_prediction']>=0,out['decision_score']>=0)
    for i in range(4):
        row=c.predict(d,state,x[i:i+1],closed[i:i+1],m[i:i+1])
        for key in out:np.testing.assert_array_equal(row[key],out[key][i:i+1])
    for defect in ('label','radius','nonfinite'):
        with subtests.test(defect=defect):
            bad=deepcopy(state)
            if defect=='label':bad['labels'][0]=7
            elif defect=='radius':bad['radii'][0]=-.1
            else:bad['source_centers'][0,0]=np.nan
            with pytest.raises(ValueError):c.validate_registry(bad)
