import torch
from experiments.phase1_dura_orthogonal.source_cross_expert_response_gradient import geometry


def test_gradient_geometry_matches_known_directions():
    base=torch.tensor([1.,0.])
    aligned=geometry(base,torch.tensor([1.,0.]),weight=.1)
    opposite=geometry(base,torch.tensor([-1.,0.]),weight=.1)
    assert abs(aligned['cosine']-1.)<1e-6
    assert abs(aligned['base_dot_combined']-1.1)<1e-6
    assert abs(opposite['cosine']+1.)<1e-6
    assert abs(opposite['base_dot_combined']-.9)<1e-6
    assert abs(opposite['aux_dot_combined']+.9)<1e-6
