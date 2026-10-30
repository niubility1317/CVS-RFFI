from copy import deepcopy
import numpy as np
import pytest
from experiments.pamf_phase3_open_world import star_class_density_core as core


def state():
    rng=np.random.default_rng(123);means=rng.normal(0,.2,(90,160));variance=np.linspace(.7,1.4,90)
    w=np.diag(np.linspace(.8,1.2,160))
    return dict(schema=np.asarray(core.density.SCHEMA),origin=np.zeros(160),whitener=w,
        log_abs_det_whitener=np.asarray(np.linalg.slogdet(w)[1]),pooled_L_variance=np.asarray(1.),
        known_means=means,known_variances=variance,known_weights=np.full(90,1/90),
        background_means=rng.normal(0,.3,(3,160)),background_variances=np.array([.8,1.,1.3]),background_weights=np.array([.2,.3,.5]))


def test_four_scores_match_pooled_and_explicit_conditional_gaussians():
    s=state();x=np.random.default_rng(456).normal(size=(8,160)).astype(np.float32);closed=np.array([0,1,2,3,4,5,2,0])
    before=deepcopy(s);out=core.raw_scores(s,x,closed);z=(core.density.unit(x)-s['origin'])@s['whitener']
    known={k:s[k] for k in core.density.KNOWN_FIELDS}
    assert np.array_equal(out[:,0],core.density.score(known,x,'known_only'))
    assert np.array_equal(out[:,1],core.density.score(s,x,'density_ratio'))
    for i,c in enumerate(closed):
        js=np.arange(c*15,(c+1)*15);v=s['known_variances'][js]
        terms=-np.log(15.)-80*np.log(2*np.pi*v)-np.square(z[i]-s['known_means'][js]).sum(1)/(2*v)
        peak=terms.max();expected=peak+np.log(np.exp(terms-peak).sum())+s['log_abs_det_whitener']
        assert out[i,2]==pytest.approx(expected,abs=1e-12)
    assert np.allclose(out[:,3]-out[:,2],out[:,1]-out[:,0],rtol=0,atol=1e-12)
    assert all(np.array_equal(v,before[k]) for k,v in s.items())


def test_other_class_modes_cannot_compensate_predicted_class_support():
    s=state();x=np.eye(160,dtype=np.float32)[:4];c=np.zeros(4,np.int64);a=core.raw_scores(s,x,c)
    changed=deepcopy(s);changed['known_means'][15:]=100.;b=core.raw_scores(changed,x,c)
    assert np.array_equal(a[:,2:],b[:,2:]) and np.any(a[:,:2]!=b[:,:2])
    # Label routing changes the numerator, without changing background or fitting.
    shifted=core.raw_scores(s,x,np.ones(4,np.int64))
    assert np.array_equal(a[:,:2],shifted[:,:2]) and np.any(a[:,2]!=shifted[:,2])


def test_query_batch_does_not_change_a_record_and_class_weights_are_normalized():
    s=state();x=np.eye(160,dtype=np.float32)[:3];c=np.array([5,0,2]);a=core.raw_scores(s,x,c)
    assert np.allclose(a[1:2],core.raw_scores(s,x[1:2],c[1:2]),rtol=0,atol=1e-12)
    identical=deepcopy(s);identical['known_means'][:]=0;identical['known_variances'][:]=1
    b=core.raw_scores(identical,x,c)
    assert np.allclose(b[:,0],b[:,2],rtol=0,atol=1e-12)


@pytest.mark.parametrize('closed',[np.array([0.,1.]),np.array([-1,1]),np.array([0,6]),np.array([0]),np.array([[0],[1]])])
def test_invalid_identity_routing_is_rejected(closed):
    with pytest.raises(ValueError):core.raw_scores(state(),np.eye(160,dtype=np.float32)[:2],closed)
