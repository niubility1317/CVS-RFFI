import numpy as np
import pytest

from experiments.pamf_phase3_restoration.covariance_capacity_source import capacity


def test_mode_coupling_captures_a_difference_that_all_diagonal_scalings_miss():
    base=np.diag([1.,2.])
    target=base+np.array([[0.,.4],[.4,0.]])
    result=capacity(base,target,rank=2)
    assert result['explained_difference_energy']['diagonal_rank']==pytest.approx(0.)
    assert result['explained_difference_energy']['diagonal_all']==pytest.approx(0.)
    assert result['explained_difference_energy']['bounded_signed_rank']==pytest.approx(0.)
    assert result['explained_difference_energy']['coupled_rank']==pytest.approx(1.)
    assert result['coupled_target_block_min_eigenvalue']>0


def test_actual_bounded_signed_capacity_is_not_overstated_by_an_unbounded_projection():
    result=capacity(np.diag([1.,2.]),np.diag([.02,5.]),rank=2)
    assert result['explained_difference_energy']['diagonal_rank']==pytest.approx(1.)
    assert result['explained_difference_energy']['bounded_signed_rank']<1.
    assert result['bounded_signed_coefficients_at_limit']==2
    assert result['ideal_diagonal_variance_ratio_min']==pytest.approx(.02)
    assert result['ideal_diagonal_variance_ratio_max']==pytest.approx(2.5)


def test_subspace_projection_residual_agrees_with_explicit_matrix_reconstruction():
    base=np.diag([.5,1.,2.])
    target=np.array([[.6,.1,.05],[.1,1.3,.2],[.05,.2,2.5]])
    result=capacity(base,target,rank=2)
    delta=target-base
    coupled=np.zeros((3,3));coupled[1:,1:]=delta[1:,1:]
    assert result['residual_squared_Frobenius']['coupled_rank']==pytest.approx(np.square(delta-coupled).sum())
    assert result['explained_difference_energy']['coupled_rank']>=result['explained_difference_energy']['diagonal_rank']


def test_invalid_covariance_is_rejected_and_zero_difference_is_explicit():
    with pytest.raises(ValueError):capacity(np.eye(2),np.ones((2,3)),rank=1)
    with pytest.raises(ValueError):capacity(np.eye(2),np.array([[1.,.2],[0.,1.]]),rank=1)
    same=capacity(np.eye(2),np.eye(2),rank=1)
    assert same['zero_difference'] and all(value is None for value in same['explained_difference_energy'].values())
