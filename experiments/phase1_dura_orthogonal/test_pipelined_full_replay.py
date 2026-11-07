import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.pipelined_full_replay import decision_mismatches
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import encode


def test_decisions_compare_to_fixed_predictions_without_truth():
    raw = torch.tensor([[1., 0.], [0., 1.]])
    restored = torch.tensor([[0., 1.], [1., 0.]])
    out = [torch.zeros(2, 2) for _ in range(7)]
    out[4], out[6] = raw, restored
    actual = decision_mismatches([[encode(out)]], [(np.array([0, 1]), np.array([1, 0]))])
    assert actual == [dict(raw=0, restored=0)]
    actual = decision_mismatches([[encode(out)]], [(np.array([1, 1]), np.array([1, 1]))])
    assert actual == [dict(raw=1, restored=1)]
    with pytest.raises(ValueError, match='lengths'):
        decision_mismatches([[encode(out)]], [(np.array([0]), np.array([1]))])
