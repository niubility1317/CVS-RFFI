"""Full Gaussian shapes, volume, source statistics and frozen prediction."""
from copy import deepcopy
import importlib
import inspect
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_local_cov_core')


@pytest.fixture(scope='module',autouse=True)
def single_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def ratio(known):
    state={k:v.copy() for k,v in known.items()}
    state.update(background_means=np.zeros((2,160),np.float64),background_variances=np.asarray([.7,1.3]),background_weights=np.asarray([.25,.75]))
    state['background_means'][1,0]=.2
    return state


@pytest.fixture(scope='module')
def fitted():
    m=api();rng=np.random.default_rng(39)
    groups=np.asarray([(c,r,d) for c in range(6) for r in (1,3,4,6,8) for d in (1,2,3)],np.int64)
    meta=np.repeat(groups,70,axis=0);x=rng.normal(scale=.08,size=(6300,160))
    x[:,0]+=.03*meta[:,1];x[:,1]+=.1*meta[:,2];x[np.arange(6300),meta[:,0]+2]+=1.
    x[:,10]=x[:,11]*.8+x[:,10]*.2;x=x.astype(np.float32)
    known,_=m.density.fit_known(x,*meta.T);state=ratio(known);before=deepcopy(state);original=x.copy()
    local,diag=m.fit(state,x,*meta.T)
    return m,x,meta,state,local,diag,before,original


def test_fit_population_covariance_trace_PSD_and_original_arrays_unchanged(fitted):
    m,x,meta,state,local,diag,before,original=fitted
    assert diag['source_count']==6300 and diag['known_modes']==90 and diag['cell_size']==70
    assert diag['source_state_refitted'] is False and diag['query_inputs_used'] is False
    assert set(local)=={'schema','transforms','log_determinants','trace_variances','alpha'}
    assert float(local['alpha'])==70/230 and local['transforms'].shape==(90,160,160)
    np.testing.assert_allclose(local['trace_variances'],state['known_variances'],rtol=1e-5,atol=1e-7)
    white=(m.density.unit(x)-state['origin'])@state['whitener'];population=np.cov(white[:70],rowvar=False,bias=True)
    expected=(70/230)*population+(160/230)*float(state['pooled_L_variance'])*np.eye(160)
    inverse=np.linalg.inv(local['transforms'][0]);actual=inverse.T@inverse
    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(actual,actual.T,rtol=0,atol=1e-14);assert np.linalg.eigvalsh(actual).min()>0
    for k,v in before.items():np.testing.assert_array_equal(state[k],v)
    np.testing.assert_array_equal(x,original);assert all(not v.flags.writeable for v in local.values())


def test_independent_direct_multivariate_Gaussian_volume_weights_and_predicted_routing(fitted):
    m,x,meta,state,_,_,_,_=fitted;state=deepcopy(state)
    covariance=np.diag(np.linspace(.5,1.5,160));covariance[0,1]=covariance[1,0]=.2
    variance=float(np.trace(covariance)/160);state['known_variances'][:]=variance
    transform=np.linalg.inv(np.linalg.cholesky(covariance)).T;logdet=np.linalg.slogdet(covariance)[1]
    local=dict(schema=np.asarray(m.SCHEMA),transforms=np.tile(transform,(90,1,1)),log_determinants=np.full(90,logdet),trace_variances=np.full(90,variance),alpha=np.asarray(70/230))
    features=x[[2,78,143,264,401]];closed=np.asarray([0,5,2,4,1],np.int64);result=m.raw_scores(state,local,features,closed)
    baseline=m.conditional.raw_scores(state,features,closed)
    np.testing.assert_array_equal(result[:,:2],baseline[:,[1,3]])
    w=(m.density.unit(features)-state['origin'])@state['whitener'];precision=np.linalg.inv(covariance)
    def logsum(values):
        high=max(values);return high+np.log(sum(np.exp(v-high) for v in values))
    for i,point in enumerate(w):
        terms=[]
        for center in state['known_means']:
            residual=point-center
            terms.append(-.5*(160*np.log(2*np.pi)+logdet+residual@precision@residual))
        background=[]
        for center,v,p in zip(state['background_means'],state['background_variances'],state['background_weights']):
            background.append(np.log(p)-80*np.log(2*np.pi*v)-np.dot(point-center,point-center)/(2*v))
        bg=logsum(background);pooled=logsum(terms)-np.log(90);predicted=logsum(terms[15*closed[i]:15*(closed[i]+1)])-np.log(15)
        # The shared whitening Jacobian cancels only after both full densities.
        np.testing.assert_allclose(result[i,2:],np.asarray([pooled-bg,predicted-bg]),rtol=2e-12,atol=1e-9)


def test_spherical_reduction_and_constant_floor_preserve_old_variance(fitted):
    m,x,meta,state,_,_,_,_=fitted;v=state['known_variances']
    local=dict(schema=np.asarray(m.SCHEMA),transforms=np.eye(160)[None]/np.sqrt(v[:,None,None]),log_determinants=160*np.log(v),trace_variances=v.copy(),alpha=np.asarray(70/230))
    query=x[:9];closed=np.arange(9,dtype=np.int64)%6;result=m.raw_scores(state,local,query,closed)
    np.testing.assert_allclose(result[:,2:],result[:,:2],rtol=1e-11,atol=1e-8)
    constant=np.ones_like(x);known,_=m.density.fit_known(constant,*meta.T);zero_state=ratio(known)
    zero,diag=m.fit(zero_state,constant,*meta.T)
    assert len(diag['floor_components'])==90
    np.testing.assert_allclose(zero['trace_variances'],zero_state['known_variances'],rtol=0,atol=1e-20)
    assert np.isfinite(m.raw_scores(zero_state,zero,constant[:2],np.zeros(2,np.int64))).all()


def test_source_role_missing_grid_labels_means_and_nonfinite_rejected(fitted,subtests):
    m,x,meta,state,_,_,_,_=fitted
    for defect in ('label','RX','day','missing_cell','mean','NaN'):
        with subtests.test(defect=defect):
            features=x.copy();labels=meta.copy();source=deepcopy(state)
            if defect=='label':labels[0,0]=6
            elif defect=='RX':labels[0,1]=26
            elif defect=='day':labels[0,2]=4
            elif defect=='missing_cell':features=features[:-1];labels=labels[:-1]
            elif defect=='mean':source['known_means'][0,0]+=.01
            else:features[0,0]=np.nan
            with pytest.raises(ValueError):m.fit(source,features,*labels.T)


def test_prediction_no_fit_batch_invariant_and_bad_aggregate_scale_rejected(fitted,monkeypatch,subtests):
    m,x,meta,state,local,_,_,_=fitted;query=x[:11];closed=np.arange(11,dtype=np.int64)%6;before=deepcopy(local)
    monkeypatch.setattr(m,'fit',lambda *a:pytest.fail('Prediction must never fit'))
    full=m.raw_scores(state,local,query,closed);parts=np.concatenate([m.raw_scores(state,local,query[:4],closed[:4]),m.raw_scores(state,local,query[4:],closed[4:])])
    np.testing.assert_allclose(parts,full,rtol=1e-12,atol=1e-9)
    for k,v in local.items():np.testing.assert_array_equal(v,before[k])
    assert list(inspect.signature(m.raw_scores).parameters)==['state','local','x','closed']
    for defect in ('negative_diagonal','logdet','trace','alpha','shape','nonfinite'):
        with subtests.test(defect=defect):
            bad=deepcopy(local)
            if defect=='negative_diagonal':bad['transforms'][0,0,0]*=-1
            elif defect=='logdet':bad['log_determinants'][0]+=1
            elif defect=='trace':bad['trace_variances'][0]*=2
            elif defect=='alpha':bad['alpha']=np.asarray(.5)
            elif defect=='shape':bad['transforms']=bad['transforms'][:89]
            else:bad['transforms'][0,0,0]=np.nan
            with pytest.raises(ValueError):m.validate_local(bad,state)
