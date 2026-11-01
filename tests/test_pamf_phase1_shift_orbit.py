import pytest
import torch

from experiments.pamf_phase1_flow.shift_orbit import shift_candidates


def test_shift_candidates_are_ordered_cyclic_shifts_and_preserve_energy():
    x=torch.arange(2*2*256,dtype=torch.float32).reshape(2,2,256)
    y=shift_candidates(x,[0,-1,1,-3,3])
    assert y.shape==(2,5,2,256) and torch.equal(y[:,0],x)
    assert torch.equal(y[:,1],torch.roll(x,-1,dims=2))
    assert torch.equal(y[:,4],torch.roll(x,3,dims=2))
    torch.testing.assert_close(y.square().sum((2,3)),x.square().sum((1,2))[:,None].expand(-1,5))


def test_invalid_shift_order_and_range_fail():
    x=torch.zeros(1,2,256)
    with pytest.raises(ValueError):shift_candidates(x,[0,3,1])
    with pytest.raises(ValueError):shift_candidates(x,[0,256])
