"""Hand-computable Gaussian densities and synthetic source-only fit behavior."""
from copy import deepcopy
import importlib
import inspect
import io
import math

import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_density_core')


@pytest.fixture
def toy(monkeypatch):
    s=api()
    for name,value in dict(DIMENSION=2,CLASSES=(0,),RECEIVERS=(1,3),DAYS=(1,),KNOWN_CELL_SIZE=2,OE_COUNT=9,CAPACITY=3).items():
        monkeypatch.setattr(s,name,value)
    x=np.asarray([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]],np.float32)
    return s,x,np.zeros(4,np.int64),np.asarray([1,1,3,3],np.int64),np.ones(4,np.int64)


def manual_state(s):
    return dict(schema=np.asarray(s.SCHEMA),origin=np.zeros(2,np.float64),whitener=np.diag([2.,3.]),
        log_abs_det_whitener=np.asarray(math.log(6.),np.float64),pooled_L_variance=np.asarray(1.,np.float64),
        known_means=np.asarray([[0.,0.],[1.,0.]],np.float64),known_variances=np.asarray([1.,4.]),known_weights=np.asarray([.5,.5]))


def test_hand_density_includes_component_weight_volume_jacobian_and_ratio(toy):
    s,*_=toy;k=manual_state(s);x=np.asarray([[3.,0.]])  # Canonical unit -> whitened [2,0].
    pL=6*(.5/(2*math.pi)*math.exp(-2)+.5/(2*math.pi*4)*math.exp(-1/8))
    np.testing.assert_allclose(s.score(k,x,'known_only'),[math.log(pL)],atol=1e-13,rtol=0.)
    b=dict(k,background_means=np.asarray([[2.,0.],[-2.,0.]]),background_variances=np.asarray([.25,1.]),background_weights=np.asarray([.8,.2]))
    pOE=6*(.8/(2*math.pi*.25)+.2/(2*math.pi)*math.exp(-8))
    value=s.score(b,x,'density_ratio')
    np.testing.assert_allclose(value,[math.log(pL)-math.log(pOE)],atol=1e-13,rtol=0.)
    assert value.dtype==np.float64 and value[0]<0
    with pytest.raises(ValueError):s.validate_state(b,'known_only')


def test_known_four_cardinal_points_have_analytic_within_variance_and_correct_whitener(toy):
    s,x,y,rx,day=toy;state,diag=s.fit_known(x,y,rx,day)
    # Within C has eigenvalues 0 and 1/2. Shrinkage gives 1/40 and 19/40,
    # so whitened residual variance per dimension is (20/19)/2 = 10/19.
    np.testing.assert_allclose(state['pooled_L_variance'],10/19,atol=1e-14)
    np.testing.assert_allclose(state['known_variances'],[10/19,10/19],atol=1e-14)
    np.testing.assert_array_equal(state['known_weights'],[.5,.5]);np.testing.assert_array_equal(state['origin'],[0.,0.])
    covariance=np.asarray([[.25,-.225],[-.225,.25]])
    np.testing.assert_allclose(state['whitener'].T@covariance@state['whitener'],np.eye(2),atol=1e-14)
    assert diag['known_mode_count']==2 and diag['known_cell_count']==2
    scaled=s.fit_known(x*np.asarray([[2.],[3.],[4.],[5.]]),y,rx,day)[0]
    for name in state:np.testing.assert_array_equal(state[name],scaled[name])


def test_background_domain_mass_not_raw_frequency_empty_modes_removed_and_exact_25_steps(toy):
    s,x,y,rx,day=toy;k,_=s.fit_known(x,y,rx,day);before={a:b.copy() for a,b in k.items()}
    oe=np.asarray([[1.,0.]]*2+[[-1.,0.]]*7,np.float32);orx=np.asarray([1]*2+[3]*7,np.int64);oday=np.ones(9,np.int64)
    b,diag=s.fit_background(k,oe,orx,oday)
    assert len(diag['lloyd_trace'])==25 and [r['iteration'] for r in diag['lloyd_trace']]==list(range(1,26))
    assert len(b['background_weights'])==2 and diag['empty_modes_removed']==1
    np.testing.assert_allclose(b['background_weights'],[.5,.5],atol=1e-15)
    np.testing.assert_allclose(sorted(diag['effective_counts']),[2.,7.],atol=1e-13)
    assert all(r['empty_modes']==1 for r in diag['lloyd_trace'])
    assert np.all(np.diff([r['weighted_sse'] for r in diag['lloyd_trace']])<=1e-12)
    assert diag['kmeans_seed']==2026103019 and diag['background_modes_are_TX_classes'] is False
    for name in k:np.testing.assert_array_equal(k[name],before[name])
    assert set(k).isdisjoint({'background_means','background_variances','background_weights'})
    again,_=s.fit_background(k,oe,orx,oday)
    for name in b:np.testing.assert_array_equal(b[name],again[name])


def test_constant_background_uses_weighted_effective_count_and_known_variance_shrink(toy):
    s,x,y,rx,day=toy;k,_=s.fit_known(x,y,rx,day)
    b,diag=s.fit_background(k,np.tile(np.asarray([[1.,0.]]),(9,1)),np.asarray([1]*2+[3]*7),np.ones(9,np.int64))
    # Domain weights are 1/4 twice and 1/14 seven times: n_eff = 56/9.
    np.testing.assert_allclose(diag['effective_counts'],[56/9],atol=1e-13)
    np.testing.assert_allclose(b['background_variances'],[90/703],atol=1e-14)
    assert len(b['background_weights'])==1
    np.testing.assert_allclose(b['background_weights'],[1.],rtol=0.,atol=1e-15)
    assert len(diag['lloyd_trace'])==25 and all(r['empty_modes']==2 for r in diag['lloyd_trace'])
    assert np.isfinite(s.score(b,x,'density_ratio')).all()


def test_constant_and_nearly_singular_known_source_remain_finite_and_no_mutation(toy):
    s,x,y,rx,day=toy
    for values in (np.tile(np.asarray([[1.,0.]]),(4,1)),np.asarray([[1.,1e-10],[1.,2e-10],[1.,3e-10],[1.,4e-10]])):
        frozen=values.copy();k,_=s.fit_known(values,y,rx,day)
        assert np.all(k['known_variances']>=1e-8) and np.isfinite(k['whitener']).all()
        assert np.isfinite(s.score(k,x,'known_only')).all();np.testing.assert_array_equal(values,frozen)


def test_production_160d_API_and_class_label_permutation_preserve_density(monkeypatch):
    s=api();monkeypatch.setattr(s,'KNOWN_CELL_SIZE',2)
    rng=np.random.default_rng(9);rows=[(c,r,d) for c in range(6) for r in s.RECEIVERS for d in s.DAYS for _ in range(2)]
    y=np.asarray([a[0] for a in rows],np.int64);rx=np.asarray([a[1] for a in rows],np.int64);day=np.asarray([a[2] for a in rows],np.int64)
    x=rng.standard_normal((180,160)).astype(np.float32);original=x.copy()
    k,_=s.fit_known(x,y,rx,day);permuted,_=s.fit_known(x,np.asarray([3,0,5,1,4,2])[y],rx,day)
    np.testing.assert_allclose(s.score(k,x[:7],'known_only'),s.score(permuted,x[:7],'known_only'),atol=1e-11,rtol=1e-12)
    np.testing.assert_array_equal(x,original);assert k['known_means'].shape==(90,160)
    assert inspect.signature(s.fit_background).parameters.keys()=={'known_state','x','rx','day'}


def test_source_grid_shapes_zero_nonfinite_and_aggregate_schema_reject_bad_inputs(toy,subtests):
    s,x,y,rx,day=toy;k,_=s.fit_known(x,y,rx,day)
    for values in (x[:3],x[:,:1],np.zeros_like(x),np.full_like(x,np.nan)):
        with subtests.test(shape=values.shape),pytest.raises(ValueError):s.fit_known(values,y,rx,day)
    for yy,rr,dd in ((y+1,rx,day),(y.astype(float),rx,day),(y,np.asarray([1,1,1,3]),day),(y,rx,day+1)):
        with subtests.test(y=yy.tolist(),rx=rr.tolist()),pytest.raises(ValueError):s.fit_known(x,yy,rr,dd)
    with pytest.raises(ValueError):s.fit_background(k,np.zeros((9,2)),np.asarray([1]*2+[3]*7),np.ones(9,np.int64))
    with pytest.raises(ValueError):s.score(k,np.zeros((1,2)),'known_only')
    for field,value in (('known_weights',np.asarray([.2,.8])),('known_variances',np.asarray([0.,1.])),
                        ('whitener',np.zeros((2,2))),('source_rows',x),('log_abs_det_whitener',np.asarray(0.))):
        bad={a:b.copy() for a,b in k.items()};bad[field]=value
        with subtests.test(field=field),pytest.raises(ValueError):s.validate_state(bad,'known_only')
    buffer=io.BytesIO();np.savez(buffer,**k);buffer.seek(0)
    with np.load(buffer,allow_pickle=False) as z:loaded={a:z[a] for a in z.files}
    assert s.validate_state(loaded,'known_only') is loaded
    assert all(isinstance(v,np.ndarray) for v in loaded.values()) and not {'ids','y','features'}&set(loaded)
