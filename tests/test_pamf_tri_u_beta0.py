import pytest
import torch
from experiments.pamf_physics_tri_u_beta0 import plan, source
from experiments.pamf_physics_branch.geometry import soft_distances


def test_beta0_ignores_physical_directions_but_preserves_euclidean_relation():
    torch.manual_seed(7)
    x=torch.randn(6,4)
    u=torch.nn.functional.normalize(torch.randn(6,4),dim=1)
    plain, beta_zero=soft_distances(x,u,0.)
    _, beta_changed=soft_distances(x,u,.5)
    assert torch.equal(plain,beta_zero)
    assert not torch.equal(beta_zero,beta_changed)


def test_registered_identity_and_output_root_are_strict():
    assert [r['model_seed'] for r in plan.candidate_rows()] == [2026092701,2026092702,2026092703]
    with pytest.raises(ValueError):
        plan.configuration({}, {}, '/project', 'commit')
    with pytest.raises(ValueError):
        source.parent(dict(model_seed=2026092701,run_id='other',row_id='bad',output_root='/bad'))
