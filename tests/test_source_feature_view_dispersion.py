import numpy as np
import pytest

from experiments.pamf_phase3_restoration.source_feature_view_dispersion import decompose, select_rows


def test_three_view_error_is_bias_plus_dispersion():
    clean=np.array([[1.,-2.],[3.,.5]])
    views=np.array([clean+[1.,2.],clean+[-1.,3.],clean+[.5,-2.]])
    scale=np.array([2.,.5])
    total,bias,variation=decompose(clean,views,scale)
    np.testing.assert_allclose(total,bias+variation,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(bias,np.mean((views.mean(axis=0)-clean)**2/scale,axis=1))
    assert np.all(variation>0)


def test_bad_feature_data_is_rejected():
    clean=np.zeros((2,4));views=np.zeros((3,2,4))
    with pytest.raises(ValueError):decompose(clean,views[:2],np.ones(4))
    with pytest.raises(ValueError):decompose(clean,views,np.array([1.,0.,1.,1.]))
    views[0,0,0]=np.nan
    with pytest.raises(ValueError):decompose(clean,views,np.ones(4))


def test_fixed_selection_covers_both_roles_without_duplicates():
    roles=np.array(['L_s']*6300+['U_s']*56700)
    idx=select_rows(roles)
    assert len(idx)==len(np.unique(idx))==512
    assert np.all(roles[idx[:256]]=='L_s') and np.all(roles[idx[256:]]=='U_s')
    with pytest.raises(ValueError):select_rows(roles[:-1])
