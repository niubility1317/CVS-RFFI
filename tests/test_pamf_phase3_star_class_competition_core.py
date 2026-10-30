"""Independent Gaussian arithmetic and a fixed original-head routing contract."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_class_competition_core')


@pytest.fixture(scope='module',autouse=True)
def one_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def fixture():
    from test_pamf_phase3_star_signal_register_core import model
    state=model();rng=np.random.default_rng(401)
    state['origin']=np.linspace(-.01,.01,160);state['whitener']=np.diag(np.linspace(.8,1.2,160))
    state['log_abs_det_whitener']=np.asarray(np.log(np.diag(state['whitener'])).sum())
    state['known_means']=rng.normal(scale=.02,size=(90,160))
    for c in range(6):state['known_means'][c*15:(c+1)*15,c]+=.6
    state['known_variances']=np.linspace(.08,.25,90)
    state['background_means']=rng.normal(scale=.03,size=(2,160));state['background_variances']=np.asarray([.11,.32]);state['background_weights']=np.asarray([.3,.7])
    x=rng.normal(size=(4,160));x/=np.linalg.norm(x,axis=1,keepdims=True)
    return state,x.astype(np.float32),np.asarray([0,5,2,3],np.int64)


def logsum(values):
    peak=max(values);return peak+math.log(sum(math.exp(float(v)-peak) for v in values))


def test_manual_six_class_equal_mass_Gaussian_oracle_and_shared_jacobian():
    m=api();state,x,closed=fixture();result=m.raw_scores(state,x,closed);expected=[]
    for point,c in zip(x,closed):
        unit=point.astype(np.float64)/np.linalg.norm(point.astype(np.float64));white=(unit-state['origin'])@state['whitener']
        terms=[-80*math.log(2*math.pi*v)-float(np.dot(white-center,white-center))/(2*v) for center,v in zip(state['known_means'],state['known_variances'])]
        classes=[logsum(terms[j*15:(j+1)*15])-math.log(15)+float(state['log_abs_det_whitener']) for j in range(6)]
        background=logsum([math.log(w)-80*math.log(2*math.pi*v)-float(np.dot(white-center,white-center))/(2*v)
            for center,v,w in zip(state['background_means'],state['background_variances'],state['background_weights'])])+float(state['log_abs_det_whitener'])
        other=logsum([classes[j] for j in range(6) if j!=c])-math.log(5)
        expected.append([classes[c]-other,classes[c]-logsum([background,other])+math.log(2)])
    assert result.shape==(4,2) and result.dtype==np.float64
    np.testing.assert_allclose(result,expected,rtol=0,atol=1e-12)
    assert m.METHODS==('pooled_ratio','predicted_ratio','class_competition','hybrid_competition') and m.PRIMARY_METHOD==m.METHODS[-1]


def test_OE_changes_hybrid_only_no_hidden_refitting_or_mutation():
    m=api();state,x,closed=fixture();before=deepcopy(state);xx=x.copy();cc=closed.copy();initial=m.raw_scores(state,x,closed)
    altered=deepcopy(state);altered['background_means']+=2
    changed=m.raw_scores(altered,x,closed)
    np.testing.assert_array_equal(changed[:,0],initial[:,0]);assert np.any(abs(changed[:,1]-initial[:,1])>1e-4)
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(closed,cc)
    np.testing.assert_allclose(m.raw_scores(state,x[::-1],closed[::-1])[::-1],initial,rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,x[i:i+1],closed[i:i+1]) for i in range(len(x))]),initial,rtol=0,atol=1e-12)


def test_class_permutation_equivariance_and_original_closed_routing():
    m=api();state,x,closed=fixture();original=m.raw_scores(state,x,closed);permutation=np.asarray([4,2,5,1,0,3]);changed=deepcopy(state)
    for key in ('known_means','known_variances','known_weights'):
        changed[key]=state[key].reshape((6,15)+state[key].shape[1:])[permutation].reshape(state[key].shape).copy()
    np.testing.assert_allclose(m.raw_scores(changed,x,np.argsort(permutation)[closed]),original,rtol=0,atol=1e-12)
    rerouted=m.raw_scores(state,x,(closed+1)%6)
    assert np.any(abs(rerouted-original)>1e-6)
    np.testing.assert_array_equal(closed,[0,5,2,3])


def test_invalid_density_features_or_original_class_rejected(subtests):
    m=api();state,x,closed=fixture()
    for defect in ('nan','zero','shape','label','floatlabel','means','weights','schema','private'):
        with subtests.test(defect=defect):
            s=deepcopy(state);xx=x.copy();c=closed.copy()
            if defect=='nan':xx[0,0]=np.nan
            elif defect=='zero':xx[0]=0
            elif defect=='shape':xx=xx[:,:159]
            elif defect=='label':c[0]=6
            elif defect=='floatlabel':c=c.astype(np.float64)
            elif defect=='means':s['known_means']=s['known_means'][:-1]
            elif defect=='weights':s['known_weights'][0]=.5
            elif defect=='schema':s['schema']=np.asarray(m.density.CONTRIBUTION_SCHEMA)
            else:s['source_y']=np.zeros(1,np.int64)
            with pytest.raises(ValueError):m.raw_scores(s,xx,c)
