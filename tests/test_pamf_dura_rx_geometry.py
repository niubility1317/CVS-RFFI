import numpy as np
import pytest

from experiments.pamf_dura_efficiency.rx_geometry import basis,canonical_feature,conditional_means,geometry,unit


def centroids(shared=True):
    head=np.eye(160)[:,:5];classes=np.zeros((6,160));classes[:5,:5]=np.eye(5);classes[5,:5]=-1
    receiver=np.array([-2.,-1.,0.,1.,2.])
    values=np.broadcast_to(classes,(5,6,160)).copy()
    for r in range(5):
        for k in range(6):values[r,k,10 if shared else 10+k]=receiver[r]
    return values,head


def test_shared_RX_geometry_generalizes_across_soft_coordinates_and_leftout_receivers():
    values,head=centroids();stats,q,loo=geometry(values,values,head)
    assert stats['source_canonical_feature_calibration_eligible'] and q.shape==(160,1)
    assert stats['protected_shared_energy_fraction']==pytest.approx(1.)
    # The middle receiver has exactly zero offset from the other four. Its
    # capture is explicitly undefined and conservatively scored zero.
    assert stats['leave_RX_mean_capture']==pytest.approx(.8)
    assert stats['leave_RX_defined']==[True,True,False,True,True]
    assert all(v.shape==(160,1) for v in loo)


def test_coordinate_specific_interactions_cannot_qualify_as_shared_RX_direction():
    values,head=centroids(False);stats,_,_=geometry(values,values,head)
    assert not stats['source_canonical_feature_calibration_eligible']
    assert stats['leave_posterior_coordinate_mean_cosine']==pytest.approx(0.,abs=1e-12)


def test_transform_preserves_original_centered_head_choices_but_is_not_IQ_correction():
    rng=np.random.default_rng(31);weight=unit(rng.normal(size=(6,160)))
    head=basis(weight-weight.mean(0),5);rows=rng.normal(size=(5,160));rows-=rows.mean(0)
    rows-=rows@head@head.T;q=basis(rows,4)
    features=unit(rng.normal(size=(512,160)));mean=features.mean(0)
    changed=canonical_feature(features,mean,q)
    assert np.max(np.abs(head.T@q))<1e-12
    assert np.array_equal((features@weight.T).argmax(1),(unit(changed)@weight.T).argmax(1))
    assert np.linalg.norm(changed-features)>0


def test_soft_weighted_means_do_not_harden_posterior_or_consume_TX():
    features=np.zeros((10,160));features[:,0]=np.arange(10)
    probabilities=np.tile(np.array([.1,.2,.1,.2,.1,.3]),(10,1))
    rx=np.repeat([1,3,4,6,8],2);means,mass=conditional_means(features,probabilities,rx)
    np.testing.assert_allclose(means[:,:,0],np.repeat(np.arange(.5,10,2)[:,None],6,axis=1))
    np.testing.assert_allclose(mass,2*probabilities[:5])
    with pytest.raises(ValueError):conditional_means(features[:-2],probabilities[:-2],rx[:-2])
