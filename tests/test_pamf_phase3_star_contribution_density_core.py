"""Only the fixed four-contribution geometry; no model or real source rows."""
import importlib
import io
import math

import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_density_core')


def test_contribution_unit_branch_scales_zeros_precision_and_input_boundaries(subtests):
    s=api();c=np.zeros((2,4,160),np.float32)
    c[0,0,0]=2;c[0,1,1]=3;c[0,3,2]=4;c[1,:,0]=[1,2,4,8];before=c.copy()
    z=s.contribution_unit(c)
    assert z.shape==(2,640) and z.dtype==np.float32 and s.DIMENSION==160
    expected=np.zeros((2,640),np.float32);expected[0,[0,161,482]]=1/math.sqrt(3);expected[1,[0,160,320,480]]=.5
    np.testing.assert_array_equal(z,expected);assert not z[0,320:480].any()
    np.testing.assert_array_equal(z,s.contribution_unit(c*np.asarray([.25,2.,16.,.5],np.float32)[None,:,None]))
    np.testing.assert_array_equal(c,before)
    np.testing.assert_allclose(np.linalg.norm(s.unit(z,dimension=640),axis=1),1.,atol=1e-15)
    for bad in (np.zeros((1,4,160),np.float32),c[:,:3],c[:,:,:159],c.astype(np.int64),np.full_like(c,np.nan)):
        with subtests.test(shape=bad.shape,dtype=bad.dtype),pytest.raises(ValueError):s.contribution_unit(bad)


def test_640_schema_covariance_volume_background_and_fixed_state_fields(monkeypatch):
    s=api()
    for name,value in dict(CLASSES=(0,),RECEIVERS=(1,3),DAYS=(1,),KNOWN_CELL_SIZE=2,OE_COUNT=9,CAPACITY=3).items():
        monkeypatch.setattr(s,name,value)
    x=np.zeros((4,640));x[:,:2]=[[1,0],[0,1],[-1,0],[0,-1]]
    y=np.zeros(4,np.int64);rx=np.asarray([1,1,3,3]);day=np.ones(4,np.int64)
    state,diag=s.fit_known(x,y,rx,day,dimension=640)
    assert s.DIMENSION==160 and state['schema'].item()=='pamf_phase3_star_contribution_density_state_v1'
    assert state['origin'].shape==(640,) and state['whitener'].shape==(640,640) and state['known_means'].shape==(2,640)
    assert set(state)==s.KNOWN_FIELDS and len(state)==8
    # The only nonzero within eigenvalue is 1/2. D=640 shrinkage gives
    # eigenvalues 5761/12800 and (1/12800)*639; pooled variance = 10/5761.
    pooled=10/5761;logdet=.5*(640*math.log(12800)-math.log(5761))
    np.testing.assert_allclose(state['pooled_L_variance'],pooled,atol=1e-14)
    np.testing.assert_allclose(state['known_variances'],pooled,atol=1e-14)
    assert diag['dimension']==640 and diag['variance_shrinkage_alpha']==2/642
    np.testing.assert_allclose(state['log_abs_det_whitener'],logdet,atol=1e-9,rtol=0.)
    # At the first cardinal point, the own-mode exponent is -D/2=-320;
    # the other mode contributes exp(-25600/(2*pooled)), below FP64 range.
    expected=-math.log(2)-320*math.log(2*math.pi*pooled)-320+logdet
    np.testing.assert_allclose(s.score(state,x[:1],'known_only'),[expected],atol=1e-9,rtol=0.)
    oe=np.vstack((np.repeat(x[:1],2,axis=0),np.repeat(x[2:3],7,axis=0)))
    background,bdiag=s.fit_background(state,oe,np.asarray([1]*2+[3]*7),np.ones(9,np.int64))
    assert len(bdiag['lloyd_trace'])==25 and set(background)==s.KNOWN_FIELDS|s.BACKGROUND_FIELDS
    np.testing.assert_allclose(np.sort(background['background_variances']),np.sort([(1-2/642)*pooled,(1-7/647)*pooled]),atol=1e-13)
    assert np.isfinite(s.score(background,x,'density_ratio')).all()
    for key in state:np.testing.assert_array_equal(state[key],background[key])
    saved=io.BytesIO();np.savez(saved,**background);saved.seek(0)
    with np.load(saved,allow_pickle=False) as z:loaded={k:z[k] for k in z.files}
    assert s.validate_state(loaded,'density_ratio') is loaded
    wrong={k:v.copy() for k,v in state.items()};wrong['schema']=np.asarray(s.SCHEMA)
    with pytest.raises(ValueError):s.validate_state(wrong,'known_only')
    with pytest.raises(ValueError):s.score(state,np.ones((1,160)),'known_only')


def test_only_explicit160_or640_allowed_without_global_dimension_changes(subtests):
    s=api();x=np.ones((2,160));assert s.unit(x).shape==(2,160)
    for width in (None,160):np.testing.assert_array_equal(s.unit(x,dimension=width),s.unit(x))
    for width in (64,320,1280,True,640.):
        with subtests.test(width=width),pytest.raises(ValueError):s.unit(np.ones((2,640)),dimension=width)
    with pytest.raises(ValueError):s.unit(np.ones((1,640)))
    assert s.DIMENSION==160 and s.SCHEMA=='pamf_phase3_star_density_state_v1'
