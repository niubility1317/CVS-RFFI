import torch
import pytest

from experiments.phase1_dura_orthogonal.two_loss_projection import project_two_descent_halfspaces


def test_feasible_update_is_unchanged():
    a = torch.tensor([1., 0.])
    b = torch.tensor([0., 1.])
    d = torch.tensor([2., 3.])
    assert torch.equal(project_two_descent_halfspaces(d, a, b), d)


def test_conflicting_proposed_update_projects_to_closest_feasible_direction():
    a = torch.tensor([1., 0.])
    b = torch.tensor([-1., 1.])
    result = project_two_descent_halfspaces(torch.tensor([2., -1.]), a, b)
    torch.testing.assert_close(result, torch.tensor([.5, .5]))
    assert a @ result >= 0 and b @ result >= 0


def test_opposing_gradients_project_to_shared_tangent():
    a = torch.tensor([1., 0.])
    b = -a
    result = project_two_descent_halfspaces(torch.tensor([2., 3.]), a, b)
    torch.testing.assert_close(result, torch.tensor([0., 3.]))


def test_rejects_nonfinite_or_different_shapes():
    with pytest.raises(ValueError):
        project_two_descent_halfspaces(torch.ones(2), torch.ones(3), torch.ones(2))
    with pytest.raises(ValueError):
        project_two_descent_halfspaces(torch.tensor([float('nan')]), torch.ones(1), torch.ones(1))
