"""Fixed PCD MD/RMD equations, source aggregation and strict input boundaries."""
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_relative_metric_core')


def source():
    rng=np.random.default_rng(71);y=np.repeat(np.arange(6,dtype=np.int64),20);x=rng.normal(0,.1,(120,160));x[np.arange(120),y]+=1.
    x=(x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32);return x,y


def test_fit_exact_population_moments_shrinkage_and_aggregate_only_state():
    m=api();x,y=source();before=x.copy();state=m.fit_metric(x,y);z=x.astype(np.float64);z/=np.linalg.norm(z,axis=1,keepdims=True)
    means=np.stack([z[y==c].mean(0) for c in range(6)]);background=z.mean(0);residual=z-means[y];global_residual=z-background
    assert set(state)=={'schema','class_means','background_mean','within_cholesky','background_cholesky'} and state['schema'].shape==() and state['schema'].dtype.kind=='U'
    assert state['schema'].item()==m.SCHEMA and m.METHODS==('MD_pooled','MD_predicted','RMD_pooled','RMD_predicted')
    np.testing.assert_array_equal(x,before);np.testing.assert_allclose(state['class_means'],means,rtol=0,atol=1e-15)
    np.testing.assert_allclose(state['background_mean'],background,rtol=0,atol=1e-15);assert np.all(np.linalg.norm(means,axis=1)<1.)
    for name,r in (('within_cholesky',residual),('background_cholesky',global_residual)):
        c=r.T@r/len(r);shrunk=.9*c+.1*np.trace(c)/160*np.eye(160)
        np.testing.assert_allclose(state[name]@state[name].T,shrunk,rtol=1e-13,atol=1e-16)
        assert state[name].dtype==np.float64 and np.array_equal(state[name],np.tril(state[name])) and np.all(np.diag(state[name])>0)
    assert m.validate_metric(state) is state


def test_scores_direct_equation_routing_row_and_bank_permutation_and_no_mutation():
    m=api();x,y=source();state=m.fit_metric(x,y);snap={k:v.copy() for k,v in state.items()};q=x[[7,28,59,96]];closed=np.asarray([5,0,1,2],np.int64);before=closed.copy()
    scores=m.metric_scores(q,closed,state);z=q.astype(np.float64);z/=np.linalg.norm(z,axis=1,keepdims=True)
    distances=np.column_stack([np.square(np.linalg.solve(state['within_cholesky'],(z-center).T)).sum(0) for center in state['class_means']])
    d0=np.square(np.linalg.solve(state['background_cholesky'],(z-state['background_mean']).T)).sum(0)
    minimum=distances.min(1);routed=distances[np.arange(len(q)),closed];expected=np.column_stack((-minimum,-routed,d0-minimum,d0-routed))
    assert scores.dtype==np.float64 and scores.shape==(4,4);np.testing.assert_allclose(scores,expected,rtol=1e-12,atol=1e-12)
    assert np.any(scores[:,0]!=scores[:,1]) and np.any(scores[:,2]!=scores[:,3]);np.testing.assert_array_equal(closed,before)
    for i in range(len(q)):np.testing.assert_allclose(m.metric_scores(q[i:i+1],closed[i:i+1],state),scores[i:i+1],rtol=1e-12,atol=1e-12)
    p=np.asarray([3,1,0,2]);np.testing.assert_allclose(m.metric_scores(q[p],closed[p],state),scores[p],rtol=1e-12,atol=1e-12)
    order=np.random.default_rng(7).permutation(len(x));permuted=m.fit_metric(x[order],y[order]);relabeled=m.fit_metric(x,(y+1)%6)
    np.testing.assert_allclose(m.metric_scores(q,closed,permuted),scores,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(m.metric_scores(q,(closed+1)%6,relabeled),scores,rtol=1e-12,atol=1e-12)
    assert all(np.array_equal(state[k],v) for k,v in snap.items())


def test_background_cancels_common_orthogonal_distance_but_MD_retains_it():
    m=api();means=np.zeros((6,160),np.float64);means[:,0]=[.8,.6,.4,.2,0.,-.2];chol=np.eye(160);chol[2,2]=2.
    state=dict(schema=np.asarray(m.SCHEMA),class_means=means,background_mean=np.zeros(160),within_cholesky=chol,background_cholesky=chol.copy())
    q=np.zeros((2,160),np.float32);q[:,0]=.5;q[0,1]=np.sqrt(.75);q[1,2]=np.sqrt(.75)
    scores=m.metric_scores(q,np.ones(2,np.int64),state)
    assert not np.isclose(scores[0,0],scores[1,0]) and not np.isclose(scores[0,1],scores[1,1])
    np.testing.assert_allclose(scores[0,2:],scores[1,2:],rtol=0,atol=1e-14)


def test_illegal_source_query_and_degenerate_covariance_are_rejected(subtests):
    m=api();x,y=source();state=m.fit_metric(x,y)
    for defect in ('float64','shape','nan','not_unit','missing_class','negative_label','float_label','label_shape','degenerate'):
        with subtests.test(defect=defect):
            a=x.copy();labels=y.copy()
            if defect=='float64':a=a.astype(np.float64)
            elif defect=='shape':a=a[:,:159]
            elif defect=='nan':a[0,0]=np.nan
            elif defect=='not_unit':a*=2
            elif defect=='missing_class':labels[labels==5]=4
            elif defect=='negative_label':labels[0]=-1
            elif defect=='float_label':labels=labels.astype(np.float64)
            elif defect=='label_shape':labels=labels[:,None]
            else:a=np.repeat(np.eye(160,dtype=np.float32)[:6],20,axis=0)
            with pytest.raises(ValueError):m.fit_metric(a,labels)
    with pytest.raises(TypeError):m.fit_metric(x,y,OE_features=x)
    for closed in (np.asarray([6]),np.asarray([-1]),np.asarray([0.]),np.zeros((1,1),np.int64)):
        with subtests.test(closed=closed.tolist()),pytest.raises(ValueError):m.metric_scores(x[:1],closed,state)


def test_exact_state_schema_covariance_and_finite_values_reject_extra_truth(subtests):
    m=api();x,y=source();valid=m.fit_metric(x,y)
    for defect in ('extra_truth','schema','float32','shape','nan','upper','zero_diagonal','negative_diagonal','impossible_mean'):
        with subtests.test(defect=defect):
            state={k:v.copy() for k,v in valid.items()}
            if defect=='extra_truth':state['truth']=y
            elif defect=='schema':state['schema']=np.asarray('wrong')
            elif defect=='float32':state['class_means']=state['class_means'].astype(np.float32)
            elif defect=='shape':state['background_mean']=state['background_mean'][:-1]
            elif defect=='nan':state['background_cholesky'][0,0]=np.nan
            elif defect=='upper':state['within_cholesky'][0,1]=.1
            elif defect=='zero_diagonal':state['within_cholesky'][0,0]=0.
            elif defect=='negative_diagonal':state['background_cholesky'][0,0]*=-1
            else:state['class_means'][0]*=10
            with pytest.raises(ValueError):m.validate_metric(state)
            with pytest.raises(ValueError):m.metric_scores(x[:1],np.zeros(1,np.int64),state)
