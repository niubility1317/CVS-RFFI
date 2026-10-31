"""An explicit equal-variance Gaussian oracle for the cancelled linear form."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_tied_density_core')


@pytest.fixture(scope='module',autouse=True)
def one_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def inputs():
    rng=np.random.default_rng(318);W=np.diag(np.linspace(.8,1.2,160));means=rng.normal(scale=.03,size=(90,160))
    for c in range(6):means[c*15:(c+1)*15,c]+=.2+.1*c
    state=dict(schema=np.asarray('pamf_phase3_star_density_state_v1'),origin=np.linspace(-.02,.02,160),whitener=W,
        log_abs_det_whitener=np.asarray(np.log(np.diag(W)).sum()),pooled_L_variance=np.asarray(.35),known_means=means,
        known_variances=np.linspace(.03,.8,90),known_weights=np.full(90,1/90),background_means=rng.normal(scale=.08,size=(3,160)),
        background_variances=np.asarray([.02,.5,2.]),background_weights=np.asarray([.15,.25,.6]))
    x=rng.normal(size=(4,160)).astype(np.float32);closed=np.asarray([0,4,2,5],np.int64)
    return state,x,closed


def logsum(values):
    top=max(values);return top+math.log(sum(math.exp(float(v)-top) for v in values))


def test_explicit_common_covariance_Gaussian_density_difference_oracle():
    m=api();state,x,closed=inputs();s=float(state['pooled_L_variance']);expected=[]
    for point,c in zip(x,closed):
        q=point.astype(np.float64);z=(q/np.linalg.norm(q)-state['origin'])@state['whitener']
        def logpdf(mean):
            residual=z-mean
            return -80*math.log(2*math.pi*s)-float(np.dot(residual,residual))/(2*s)+float(state['log_abs_det_whitener'])
        known=[logpdf(mean) for mean in state['known_means']]
        bg=logsum([math.log(weight)+logpdf(mean) for mean,weight in zip(state['background_means'],state['background_weights'])])
        expected.append([logsum([math.log(w)+v for w,v in zip(state['known_weights'],known)])-bg,
            logsum(known[15*c:15*(c+1)])-math.log(15)-bg])
    result=m.raw_scores(state,x,closed)
    assert result.shape==(4,2) and result.dtype==np.float64
    np.testing.assert_allclose(result,expected,atol=1e-12,rtol=0)
    assert m.METHODS==('pooled_ratio','predicted_ratio','tied_pooled_ratio','tied_predicted_ratio') and m.PRIMARY_METHOD==m.METHODS[3] and m.PRIMARY_INDEX==3


def test_only_pooled_variance_used_fixed_original_class_routing_and_no_mutation():
    m=api();state,x,c=inputs();before=deepcopy(state);xx=x.copy();cc=c.copy();result=m.raw_scores(state,x,c)
    changed=deepcopy(state);changed['known_variances']*=100;changed['background_variances']*=.01
    np.testing.assert_array_equal(m.raw_scores(changed,x,c),result)
    changed['pooled_L_variance']*=2;assert np.any(abs(m.raw_scores(changed,x,c)-result)>1e-4)
    rerouted=m.raw_scores(state,x,(c+1)%6);np.testing.assert_array_equal(rerouted[:,0],result[:,0]);assert np.any(abs(rerouted[:,1]-result[:,1])>1e-4)
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,x[i:i+1],c[i:i+1]) for i in range(len(x))]),result,atol=1e-12,rtol=0)
    permutation=np.asarray([4,1,0,5,3,2]);reordered=deepcopy(state)
    for key in ('known_means','known_variances','known_weights'):
        reordered[key]=state[key].reshape((6,15)+state[key].shape[1:])[permutation].reshape(state[key].shape).copy()
    np.testing.assert_allclose(m.raw_scores(reordered,x,np.argsort(permutation)[c]),result,atol=1e-12,rtol=0)
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(c,cc)


def test_invalid_common_scale_features_class_or_original_density(subtests):
    m=api();state,x,c=inputs()
    for defect in ('zero_s','negative_s','nan_s','shape_s','nan_x','zero_x','width','class','class_dtype','modes'):
        with subtests.test(defect=defect):
            s=deepcopy(state);xx=x.copy();cc=c.copy()
            if defect=='zero_s':s['pooled_L_variance']=np.asarray(0.)
            elif defect=='negative_s':s['pooled_L_variance']=np.asarray(-1.)
            elif defect=='nan_s':s['pooled_L_variance']=np.asarray(np.nan)
            elif defect=='shape_s':s['pooled_L_variance']=np.asarray([.35])
            elif defect=='nan_x':xx[0,0]=np.nan
            elif defect=='zero_x':xx[0]=0
            elif defect=='width':xx=xx[:,:159]
            elif defect=='class':cc[0]=6
            elif defect=='class_dtype':cc=cc.astype(np.float64)
            else:s['known_means']=s['known_means'][:-1]
            with pytest.raises(ValueError):m.raw_scores(s,xx,cc)
