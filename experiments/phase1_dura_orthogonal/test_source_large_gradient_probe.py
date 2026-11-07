import pytest
import torch

from experiments.phase1_dura_orthogonal.source_large_gradient_probe import summary


def test_aligned_and_opposite_gradient_means():
    aligned=torch.tensor([[[1.,0.],[3.,0.]],[[2.,0.],[4.,0.]]])
    result=summary(aligned)
    assert result['split_cosine']==pytest.approx(1.)
    assert result['side_mean_norms']==pytest.approx([2.,3.])
    opposite=aligned.clone();opposite[1]*=-1
    assert summary(opposite)['split_cosine']==pytest.approx(-1.)


def test_zero_or_nonfinite_gradients_are_not_reported_as_valid():
    with pytest.raises(ValueError):
        summary(torch.zeros(2,2,4))
    bad=torch.ones(2,2,4);bad[0,0,0]=float('nan')
    with pytest.raises(ValueError):
        summary(bad)
