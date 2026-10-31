import numpy as np
import pytest
from experiments.pamf_dura_efficiency.receiver_gradients import balanced_indices,geometry,decision,RECEIVERS,SCENES,PER_CELL


def test_balancing_matches_scenes_and_receivers_with_deterministic_unique_selection():
    rx=np.repeat(RECEIVERS,len(SCENES)*15);scene=np.tile(np.repeat(SCENES,15),5)
    order=np.random.default_rng(91).permutation(len(rx));a=balanced_indices(rx,scene,order)
    assert a.shape==(5,84) and np.unique(a).size==420
    np.testing.assert_array_equal(a,balanced_indices(rx,scene,order))
    for i,r in enumerate(RECEIVERS):
        assert np.all(rx[a[i]]==r)
        for j,s in enumerate(SCENES):assert np.all(scene[a[i,j*PER_CELL:(j+1)*PER_CELL]]==s)
    with pytest.raises(ValueError):balanced_indices(rx,scene,np.zeros_like(order))
    scene[scene=='clean']='bad'
    with pytest.raises(ValueError):balanced_indices(rx,scene,order)


def test_geometry_uses_raw_equal_receiver_mean_and_handles_undefined_directions():
    g=np.array([[4.,1],[-1,0],[-2,-1],[3,2],[-1,-2]],np.float32);r=geometry(g);mean=g.astype(float).mean(0)
    for i in range(5):
        assert r['cosine_with_equal_RX_mean'][i]==pytest.approx(np.dot(g[i],mean)/(np.linalg.norm(g[i].astype(float))*np.linalg.norm(mean)))
        for j in range(5):assert r['pair_cosines'][i][j]==pytest.approx(np.dot(g[i],g[j])/(np.linalg.norm(g[i].astype(float))*np.linalg.norm(g[j].astype(float))))
    assert r['negative_pair_count']>=3 and r['mean_conflict_count']>=1 and r['all_directions_defined']
    z=geometry(np.zeros((5,2)));assert not z['all_directions_defined'] and z['negative_pair_count']==z['mean_conflict_count']==0
    with pytest.raises(ValueError):geometry(np.full((5,2),np.nan))


def test_primary_requires_both_fixed_views_and_cannot_promote_single_view():
    good=geometry(np.array([[4.,1],[-1,0],[-2,-1],[3,2],[-1,-2]],np.float32));bad=geometry(np.ones((5,2)))
    assert decision({'1':good,'20':good})['supports_RX_conflict_hypothesis']
    r=decision({'1':good,'20':bad});assert not r['supports_RX_conflict_hypothesis'] and not r['automatic_training']
    with pytest.raises(ValueError):decision({'1':good})
