import pytest
import torch
from experiments.pamf_physics_branch.source_diagnostic import direction_overlap, identity_basis, run


def test_source_class_basis_and_zero_direction_are_well_defined():
    features = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]])
    labels = torch.tensor([0, 0, 1, 1])
    basis, rank = identity_basis(features, labels, classes=2)
    assert rank == 1 and basis.shape == (3, 1)
    directions = torch.tensor([[1., -1., 0.], [0., 0., 1.], [0., 0., 0.]])
    overlap, valid = direction_overlap(directions, basis)
    assert overlap[0].item() == pytest.approx(1.)
    assert overlap[1].item() == pytest.approx(0.)
    assert not bool(valid[2]) and overlap[2].item() == 0.


def test_source_diagnostic_rejects_target_before_opening_model():
    with pytest.raises(ValueError, match='source-only'):
        run({'truth': '/forbidden'}, device='cpu')
