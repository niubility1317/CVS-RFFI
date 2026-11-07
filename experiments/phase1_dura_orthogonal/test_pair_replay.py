import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.pair_replay import compare_batch


def test_compare_batch_counts_both_prediction_streams():
    raw = torch.tensor([[0., 2., 1., 0., 0., 0.], [3., 1., 0., 0., 0., 0.]])
    restored = torch.tensor([[0., 0., 5., 0., 0., 0.], [0., 4., 0., 0., 0., 0.]])
    assert compare_batch((raw, restored), np.array([1, 1]), np.array([2, 0])) == (1, 1)
    with pytest.raises(ValueError, match='shape'):
        compare_batch((raw, restored), np.array([1]), np.array([2, 0]))
