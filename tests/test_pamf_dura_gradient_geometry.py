import pytest
import torch

from experiments.pamf_dura_gradient.runner import geometry


def test_geometry_reports_opposed_weighted_gradients_and_undefined_zero_norm():
    left=[torch.tensor([3.,4.])]
    opposite=[torch.tensor([-6.,-8.])]
    result=geometry(left,opposite)
    assert result['cosine']==pytest.approx(-1)
    assert result['right_to_left_norm']==pytest.approx(2)
    assert geometry([torch.zeros(2)],opposite)['cosine'] is None
