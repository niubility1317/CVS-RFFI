import numpy as np
import pytest
from experiments.pamf_dura_efficiency.soft_rx_reference import fit_references,make_reference,head_geometry,eligible


def fixture():
    rx=np.repeat([1,3,4,6,8],3*6*40);day=np.tile(np.repeat([1,2,3],6*40),5)
    coordinate=np.tile(np.repeat(np.arange(6),40),15)
    p=np.full((len(rx),6),.01);p[np.arange(len(rx)),coordinate]=.95
    features=np.zeros((len(rx),160));features[np.arange(len(rx)),coordinate]=1
    features[:,6]=rx/10;features[:,7]=day/10
    return features,p,rx,day


def test_reference_excludes_own_receiver_and_excluded_fit_records():
    f,p,rx,day=fixture();fit=np.ones(len(rx),bool);fit[:5]=False
    a=fit_references(f,p,rx,day,fit)
    changed=f.copy();changed[rx==1]+=10
    b=fit_references(changed,p,rx,day,fit)
    np.testing.assert_allclose(a['leave_RX_centroids'][0],b['leave_RX_centroids'][0])
    changed=f.copy();changed[:5]+=1000
    c=fit_references(changed,p,rx,day,fit)
    np.testing.assert_array_equal(a['cell_means'],c['cell_means'])
    assert a['cell_effective_n'].shape==(5,3,6)


def test_receiver_day_cells_are_averaged_equally_despite_record_imbalance():
    f,p,rx,day=fixture();fit=np.ones(len(rx),bool)
    first=fit_references(f,p,rx,day,fit)
    take=(rx==3)&(day==1)
    f2=np.concatenate((f,np.repeat(f[take],3,axis=0)));p2=np.concatenate((p,np.repeat(p[take],3,axis=0)))
    rx2=np.concatenate((rx,np.repeat(rx[take],3)));day2=np.concatenate((day,np.repeat(day[take],3)))
    second=fit_references(f2,p2,rx2,day2,np.ones(len(rx2),bool))
    np.testing.assert_allclose(first['leave_RX_centroids'],second['leave_RX_centroids'],atol=1e-14)


def test_zero_mixture_is_identity_and_geometry_has_exact_class_contrast_case():
    f,p,rx,day=fixture();f=f/np.linalg.norm(f,axis=1,keepdims=True)
    fit=fit_references(f,p,rx,day,np.ones(len(rx),bool))
    target=make_reference(f,p,rx,fit['leave_RX_centroids'],beta=0)
    np.testing.assert_allclose(target,f,atol=1e-14)
    q=np.eye(160)[:,:5];g=head_geometry(f,p,rx,q)
    assert g['RX_scatter']>0 and g['class_spread']>0


def test_missing_coordinate_mass_rejected_and_fixed_eligibility_not_relaxed():
    f,p,rx,day=fixture();p[:,5]=0;p/=p.sum(1,keepdims=True)
    with pytest.raises(ValueError):fit_references(f,p,rx,day,np.ones(len(rx),bool))
    assert eligible(32,512,dict(RX_scatter=1,class_spread=1),dict(RX_scatter=.8,class_spread=.9))
    assert not eligible(32,511,dict(RX_scatter=1,class_spread=1),dict(RX_scatter=.8,class_spread=.9))
    assert not eligible(31.9,512,dict(RX_scatter=1,class_spread=1),dict(RX_scatter=.8,class_spread=.9))


def test_unknown_rx_cannot_leave_a_reference_row_uninitialized():
    with pytest.raises(ValueError):
        make_reference(np.ones((2,160)),np.full((2,6),1/6),np.array([1,24]),np.ones((5,6,160)))
