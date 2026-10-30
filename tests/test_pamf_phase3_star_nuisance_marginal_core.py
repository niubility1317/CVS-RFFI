"""Analytic source nuisance convolution, independent dense Gaussian checks."""
from copy import deepcopy
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_marginal_core')


@pytest.fixture(scope='module',autouse=True)
def single_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def source_state():
    m=api();rng=np.random.default_rng(71);q=np.linalg.qr(rng.normal(size=(160,2)))[0].T
    angle=np.arange(15)*2*np.pi/15;offset=np.column_stack((.4*np.cos(angle),.2*np.sin(angle)))@q
    means=np.zeros((6,15,160));means[:,:,0]=np.arange(6)[:,None]*.03;means+=offset[None]
    state=dict(schema=np.asarray(m.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160),log_abs_det_whitener=np.asarray(0.),pooled_L_variance=np.asarray(1.),
        known_means=means.reshape(90,160),known_variances=np.linspace(.3,.8,90),known_weights=np.full(90,1/90),
        background_means=np.stack((q[0]*.7,q[1]*-.4)),background_variances=np.asarray([.5,1.1]),background_weights=np.asarray([.35,.65]))
    return state,q


def test_prior_population_spectrum_and_independent_dense160_logpdf_ratios():
    m=api();state,q=source_state();before=deepcopy(state);prior,diag=m.build_prior(state)
    assert set(prior)=={'schema','basis','variances'} and prior['basis'].shape==(2,160)
    np.testing.assert_allclose(prior['variances'],[.08,.02],rtol=1e-13,atol=1e-15)
    covariance=prior['basis'].T@(prior['variances'][:,None]*prior['basis'])
    np.testing.assert_allclose(covariance,q.T@np.diag([.08,.02])@q,atol=1e-16)
    assert diag['rank']==2 and diag['population_divisor']==15 and diag['source_state_refitted'] is False
    rng=np.random.default_rng(72);x=rng.normal(size=(4,160));closed=np.asarray([0,5,2,3],np.int64);actual=m.raw_scores(state,prior,x,closed)
    white=m.density.unit(x)
    def logsum(v):a=max(v);return a+np.log(sum(np.exp(value-a) for value in v))
    def dense(point,center,variance):
        full=variance*np.eye(160)+covariance;residual=point-center
        sign,logdet=np.linalg.slogdet(full);assert sign==1
        return -.5*(160*np.log(2*np.pi)+logdet+residual@np.linalg.solve(full,residual))
    for i,point in enumerate(white):
        known=[dense(point,center,v) for center,v in zip(state['known_means'],state['known_variances'])]
        background=[np.log(p)+dense(point,center,v) for center,v,p in zip(state['background_means'],state['background_variances'],state['background_weights'])]
        expected=[logsum(known)-np.log(90)-logsum(background),logsum(known[15*closed[i]:15*(closed[i]+1)])-np.log(15)-logsum(background)]
        np.testing.assert_allclose(actual[i],expected,rtol=1e-12,atol=1e-10)
    for k,v in state.items():np.testing.assert_array_equal(v,before[k])
    assert all(not a.flags.writeable for a in prior.values())


def test_rank_zero_is_legal_and_exactly_preserves_both_old_ratios():
    m=api();state,_=source_state();state['known_means']=np.repeat(state['known_means'].reshape(6,15,160).mean(1)[:,None],15,axis=1).reshape(90,160)
    prior,diag=m.build_prior(state);assert prior['basis'].shape==(0,160) and prior['variances'].shape==(0,) and diag['rank']==0
    x=np.eye(160)[[0,1,2]];closed=np.asarray([0,3,5],np.int64)
    np.testing.assert_array_equal(m.raw_scores(state,prior,x,closed),m.conditional.raw_scores(state,x,closed)[:,[1,3]])
    assert m.validate_prior(prior) is prior


def test_rotation_row_and_batch_invariance_without_query_fit_or_mutation(monkeypatch):
    m=api();state,_=source_state();prior,_=m.build_prior(state);rng=np.random.default_rng(73);x=rng.normal(size=(7,160));closed=np.arange(7,dtype=np.int64)%6
    old=deepcopy(prior);expected=m.raw_scores(state,prior,x,closed);rotation=np.linalg.qr(rng.normal(size=(160,160)))[0]
    rotated=deepcopy(state);rotated['known_means']=state['known_means']@rotation;rotated['background_means']=state['background_means']@rotation
    rp,_=m.build_prior(rotated);np.testing.assert_allclose(m.raw_scores(rotated,rp,x@rotation,closed),expected,rtol=2e-12,atol=1e-10)
    monkeypatch.setattr(m,'build_prior',lambda *a:pytest.fail('Prediction cannot fit any prior'))
    order=np.asarray([6,1,4,0,5,2,3]);np.testing.assert_allclose(m.raw_scores(state,prior,x[order],closed[order]),expected[order],rtol=1e-12,atol=1e-10)
    batch=np.concatenate((m.raw_scores(state,prior,x[:2],closed[:2]),m.raw_scores(state,prior,x[2:],closed[2:])))
    np.testing.assert_allclose(batch,expected,rtol=1e-12,atol=1e-10)
    for k,v in prior.items():np.testing.assert_array_equal(v,old[k])


def test_orthogonal_residual_quadratic_retains_small_positive_value():
    m=api();basis=np.eye(160)[:1];residual=basis.copy();value=m._quadratic(residual,basis,np.asarray([1e12]),1e-8)
    assert value[0]>0;np.testing.assert_allclose(value,[1/(1e12+1e-8)],rtol=1e-15,atol=0)


def test_invalid_geometry_labels_nonfinite_and_negative_variance(subtests):
    m=api();state,_=source_state();prior,_=m.build_prior(state);x=np.eye(160)[:2];closed=np.asarray([0,5],np.int64)
    for label in (np.asarray([-1,0]),np.asarray([0,6]),closed.astype(float),closed[:1]):
        with subtests.test(label=label.tolist()),pytest.raises(ValueError):m.raw_scores(state,prior,x,label)
    for value in (np.zeros_like(x),x[:,:159],np.full_like(x,np.nan)):
        with subtests.test(shape=value.shape),pytest.raises(ValueError):m.raw_scores(state,prior,value,closed)
    for defect in ('negative','zero','nonfinite','basis','rank','extra'):
        with subtests.test(defect=defect):
            bad=deepcopy(prior)
            if defect=='negative':bad['variances'][0]=-1
            elif defect=='zero':bad['variances'][0]=0
            elif defect=='nonfinite':bad['variances'][0]=np.inf
            elif defect=='basis':bad['basis'][0]*=2
            elif defect=='rank':bad['basis']=np.eye(160)[:15];bad['variances']=np.ones(15)
            else:bad['ids']=np.asarray(['private-row'])
            with pytest.raises(ValueError):m.validate_prior(bad)
    wrong=deepcopy(state);wrong['known_variances'][0]=-1
    with pytest.raises(ValueError):m.raw_scores(wrong,prior,x,closed)
