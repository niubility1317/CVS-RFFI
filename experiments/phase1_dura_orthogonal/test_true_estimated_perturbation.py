import pytest
import torch

from experiments.phase1_dura_orthogonal.true_estimated_perturbation import action_metrics


def test_identity_opposite_and_scale():
    truth = torch.tensor([[1 + 2j, 3 - 4j], [-1 + .5j, 2 + 1j]], dtype=torch.complex64)
    for estimate, expected in ((truth, [0., 1., 1.]), (-truth, [4., -1., 1.]),
                               (2 * truth, [1., 1., 2.])):
        torch.testing.assert_close(action_metrics(estimate, truth), torch.tensor([expected, expected]))


def test_rejects_bad_shapes_types_and_nonfinite():
    truth = torch.tensor([[1 + 2j, 3 - 4j], [-1 + .5j, 2 + 1j]], dtype=torch.complex64)
    with pytest.raises(ValueError):
        action_metrics(truth[:1], truth)
    with pytest.raises(ValueError):
        action_metrics(truth.real, truth)
    truth[0, 0] = complex(float('nan'), 0.)
    with pytest.raises(FloatingPointError):
        action_metrics(truth, truth)
