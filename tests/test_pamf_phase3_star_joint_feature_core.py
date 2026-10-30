"""Two final frozen features, D320 density mathematics and legacy schemas."""
from copy import deepcopy
import importlib
import inspect
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_feature_core')


@pytest.fixture(scope='module',autouse=True)
def single_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def test_join_equal_feature_weight_precision_scaling_and_boundaries(subtests):
    m=api();u=np.zeros((2,160),np.float32);z=u.copy();u[:,0]=[2,4];z[:,1]=[8,16];before=(u.copy(),z.copy())
    value=m.join(u,z);expected=np.zeros((2,320),np.float32);expected[:,[0,161]]=1/math.sqrt(2)
    np.testing.assert_array_equal(value,expected);np.testing.assert_array_equal(value,m.join(u*8,z*.25))
    assert value.dtype==np.float32 and value.shape==(2,320)
    np.testing.assert_allclose(np.linalg.norm(value.astype(np.float64),axis=1),1,atol=5e-8)
    for a,b in zip((u,z),before):np.testing.assert_array_equal(a,b)
    for bad in (np.zeros_like(u),u[:1],u[:,:159],np.full_like(u,np.nan),u.astype(np.int64)):
        with subtests.test(shape=bad.shape,dtype=bad.dtype),pytest.raises(ValueError):m.join(bad,z)


@pytest.fixture(scope='module')
def fitted():
    m=api();rng=np.random.default_rng(61);rows=np.asarray([(c,r,d) for c in range(6) for r in (1,3,4,6,8) for d in (1,2,3) for _ in range(70)],np.int64)
    Lu=rng.normal(size=(6300,160)).astype(np.float32);Lz=rng.normal(size=(6300,160)).astype(np.float32)
    oe=np.repeat(np.asarray([(r,d) for r in (1,3,4,6,8) for d in (1,2,3)],np.int64),[409]*14+[408],axis=0)
    OEu=rng.normal(size=(6134,160)).astype(np.float32);OEz=rng.normal(size=(6134,160)).astype(np.float32)
    state,diag=m.fit(Lu,Lz,*rows.T,OEu,OEz,*oe.T)
    original,_=m.density.fit_known(Lz,*rows.T)
    original=dict(original,background_means=np.zeros((1,160)),background_variances=np.ones(1),background_weights=np.ones(1))
    return m,(Lu,Lz,rows,OEu,OEz,oe),state,diag,original


def test_production_grid320_schema_population_fit_domain_weights_and_fixed_budget(fitted):
    m,(Lu,Lz,rows,OEu,OEz,oe),state,diag,original=fitted
    assert m.density.DIMENSION==160 and state['schema'].item()==m.density.JOINT_SCHEMA
    assert set(state)==m.density.KNOWN_FIELDS|m.density.BACKGROUND_FIELDS and len(state)==11
    assert state['origin'].shape==(320,) and state['whitener'].shape==(320,320) and state['known_means'].shape==(90,320)
    assert diag['known']['dimension']==diag['background']['dimension']==320
    assert diag['known']['known_count']==6300 and diag['known']['variance_shrinkage_alpha']==70/390
    assert diag['background']['background_count']==6134 and diag['background']['lloyd_iterations']==25
    np.testing.assert_allclose(diag['background']['domain_total_weights'],np.full(15,1/15),atol=1e-15)
    assert len(diag['background']['lloyd_trace'])==25 and diag['background']['kmeans_seed']==2026103019
    assert all(not v.flags.writeable for v in state.values()) and not {'ids','y','features','source_rows'}&set(state)
    assert m.validate_joint(state) is state
    with pytest.raises(ValueError):m.validate_joint(original)
    assert list(inspect.signature(m.fit).parameters)==['Lu','Lz','Ly','Lrx','Lday','OEu','OEz','OE_rx','OE_day']


def model(width,schema):
    means=np.zeros((90,width));means[:,0]=np.repeat(np.linspace(-.5,.5,6),15)
    return dict(schema=np.asarray(schema),origin=np.zeros(width),whitener=np.eye(width)*1.5,log_abs_det_whitener=np.asarray(width*math.log(1.5)),pooled_L_variance=np.asarray(1.),
        known_means=means,known_variances=np.repeat([.5,.7,1.,1.5,2.,3.],15),known_weights=np.full(90,1/90),
        background_means=np.vstack((np.zeros(width),np.full(width,.03))),background_variances=np.asarray([.6,1.4]),background_weights=np.asarray([.3,.7]))


def test_direct_D320_Gaussian_volume_conditional_mass_and_exact_old_controls():
    m=api();original=model(160,m.density.SCHEMA);joint=model(320,m.density.JOINT_SCHEMA)
    u=np.eye(160,dtype=np.float32)[[0,1,2]];z=np.eye(160,dtype=np.float32)[[2,1,0]];closed=np.asarray([5,0,3],np.int64)
    result=m.raw_scores(original,joint,u,z,closed);base=m.conditional.raw_scores(original,z,closed)
    np.testing.assert_array_equal(result[:,:2],base[:,[1,3]])
    w=m.density.unit(m.join(u,z),dimension=320)*1.5
    def sumlog(v):peak=max(v);return peak+math.log(sum(math.exp(a-peak) for a in v))
    for i,point in enumerate(w):
        terms=[-160*math.log(2*math.pi*v)-np.dot(point-center,point-center)/(2*v) for center,v in zip(joint['known_means'],joint['known_variances'])]
        bg=[math.log(p)-160*math.log(2*math.pi*v)-np.dot(point-center,point-center)/(2*v) for center,v,p in zip(joint['background_means'],joint['background_variances'],joint['background_weights'])]
        expected=[sumlog(terms)-math.log(90)-sumlog(bg),sumlog(terms[closed[i]*15:(closed[i]+1)*15])-math.log(15)-sumlog(bg)]
        np.testing.assert_allclose(result[i,2:],expected,rtol=1e-12,atol=1e-10)


def test_legacy160_and640_schemas_numeric_volume_and_default_are_retained():
    m=api();s=m.density
    assert s.DIMENSION==160 and s.SCHEMA=='pamf_phase3_star_density_state_v1' and s.CONTRIBUTION_SCHEMA=='pamf_phase3_star_contribution_density_state_v1'
    for width,schema in ((160,s.SCHEMA),(640,s.CONTRIBUTION_SCHEMA)):
        state=dict(schema=np.asarray(schema),origin=np.zeros(width),whitener=np.eye(width),log_abs_det_whitener=np.asarray(0.),pooled_L_variance=np.asarray(2.),
            known_means=np.zeros((90,width)),known_variances=np.full(90,2.),known_weights=np.full(90,1/90))
        x=np.eye(width)[:1];expected=-width/2*math.log(4*math.pi)-.25
        np.testing.assert_allclose(s.score(state,x,'known_only'),[expected],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(s.unit(np.eye(160)[:1]),s.unit(np.eye(160)[:1],dimension=160))
    with pytest.raises(ValueError):s.unit(np.ones((2,320)))
    with pytest.raises(ValueError):s.unit(np.ones((2,480)),dimension=480)


def test_source_roles_and_prediction_without_fitting_or_state_mutation(fitted,monkeypatch,subtests):
    m,(Lu,Lz,rows,OEu,OEz,oe),state,diag,original=fitted
    bad=rows.copy();bad[0,0]=6
    with pytest.raises(ValueError):m.fit(Lu,Lz,*bad.T,OEu,OEz,*oe.T)
    with pytest.raises(ValueError):m.fit(Lu[:-1],Lz[:-1],*rows[:-1].T,OEu,OEz,*oe.T)
    bad_oe=oe.copy();bad_oe[0,0]=26
    with pytest.raises(ValueError):m.fit(Lu,Lz,*rows.T,OEu,OEz,*bad_oe.T)
    with pytest.raises(TypeError):m.fit(Lu,Lz,*rows.T,OEu,OEz,*oe.T,OE_y=np.zeros(6134))
    before=deepcopy(state);old=deepcopy(original);inputs=(Lu[:9].copy(),Lz[:9].copy());closed=np.arange(9,dtype=np.int64)%6
    monkeypatch.setattr(m.density,'fit_known',lambda *a,**k:pytest.fail('Prediction cannot fit known'))
    monkeypatch.setattr(m.density,'fit_background',lambda *a,**k:pytest.fail('Prediction cannot fit OE'))
    full=m.raw_scores(original,state,*inputs,closed)
    parts=np.concatenate([m.raw_scores(original,state,inputs[0][:3],inputs[1][:3],closed[:3]),m.raw_scores(original,state,inputs[0][3:],inputs[1][3:],closed[3:])])
    np.testing.assert_allclose(parts,full,rtol=1e-12,atol=1e-9)
    for k,v in state.items():np.testing.assert_array_equal(v,before[k])
    for k,v in original.items():np.testing.assert_array_equal(v,old[k])
    np.testing.assert_array_equal(inputs[0],Lu[:9]);np.testing.assert_array_equal(inputs[1],Lz[:9])
