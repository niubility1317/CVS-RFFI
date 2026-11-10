import torch

from experiments.pamf_physics_branch.geometry import soft_distances
from experiments.pamf_physics_branch.reference_routing import stable_reference_directions


def test_reference_routing_retains_stable_directions_and_zeroes_only_changed_routes():
    vectors = torch.tensor([[1., 0.], [0., 1.], [2**-.5, 2**-.5]])
    raw = torch.tensor([1, 2, 3])
    perturbed = torch.tensor([1, 1, 3])
    out = stable_reference_directions(vectors, raw, perturbed)
    assert torch.equal(out, torch.tensor([[1., 0.], [0., 0.], [2**-.5, 2**-.5]]))
    assert torch.equal(vectors[1], torch.tensor([0., 1.]))


def test_route_corrected_distances_keep_all_pairs_and_physical_lower_bounds():
    features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    directions = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    masked = stable_reference_directions(directions, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 2]))
    ordinary, corrected = soft_distances(features, masked, beta=.5)
    assert corrected.shape == (3, 3)
    assert torch.all(corrected <= ordinary + 1e-6)
    assert torch.all(corrected >= .5 * ordinary - 1e-6)
    assert corrected[1, 2] == ordinary[1, 2]
    assert corrected[0, 1] >= .75 * ordinary[0, 1] - 1e-6
