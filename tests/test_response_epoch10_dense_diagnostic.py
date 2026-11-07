import pytest
import torch
from experiments.pamf_phase3_restoration.response_epoch10_dense_diagnostic import summary


def test_comparison_reports_residuals_and_iq_difference():
    low=torch.tensor([[1.,2.],[3.,4.]])
    dense=low+0.25
    a={'relative_solve_residual':torch.tensor([1e-9,2e-8])}
    b={'relative_solve_residual':torch.tensor([2e-10,3e-10])}
    result=summary(low,a,dense,b)
    assert result['max_abs_iq_difference']==0.25
    assert result['low_rank_over_limit']==1 and result['dense_over_limit']==0
    assert result['low_rank_max_residual']==pytest.approx(2e-8)
    with pytest.raises(ValueError,match='Finite'):
        summary(low,{'relative_solve_residual':torch.tensor([float('nan')])},dense,b)
