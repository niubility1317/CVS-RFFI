from types import SimpleNamespace

import torch
from torch import nn

from experiments.pamf_physics_tri import physical_branch_dropout as route
from experiments.pamf_physics_tri import physical_branch_dropout_plan as plan


def test_physical_routing_uses_branch_distortion_and_complementary_views():
    raw = {name: torch.tensor([[1., 0.], [1., 0.]]) for name in route.PATHS}
    changed = {name: value.clone() for name, value in raw.items()}
    changed['reference'][0] = torch.tensor([-1., 0.])
    changed['pa'][0] = torch.tensor([0., 1.])
    changed['time'][1] = torch.tensor([-1., 0.])
    changed['frequency'][1] = torch.tensor([0., 1.])
    first, second = route.routed_paths(raw, changed, 'complementary', torch.tensor([5, 8]), 0)
    assert first.tolist() == [4, 0]
    assert second.tolist() == [2, 1]
    first, second = route.routed_paths(raw, changed, 'balanced', torch.tensor([5, 8]), 1)
    assert first.tolist() == second.tolist() == [1, 4]


def test_attenuation_only_changes_selected_branch_and_preserves_gradient(monkeypatch):
    core = SimpleNamespace(t_proj=nn.Identity(), f_proj=nn.Identity(), freq_stats_proj=nn.Identity(),
                           pa_proj=nn.Identity(), pa_stats_proj=nn.Identity())
    sequence, reference = nn.Identity(), nn.Identity()
    monkeypatch.setattr(route, '_actual_paths', lambda _: (core, sequence, reference))
    x = torch.ones(3, 2, requires_grad=True)
    assignments = torch.tensor([0, 1, 4])
    selected = torch.tensor([True, True, False])
    with route.attenuated_branches(None, assignments, selected, .5):
        time = core.t_proj(x)
        frequency = core.f_proj(x) + core.freq_stats_proj(x)
        ref = reference(x)
    assert time[:, 0].tolist() == [.5, 1., 1.]
    assert frequency[:, 0].tolist() == [2., 1., 2.]
    assert ref[:, 0].tolist() == [1., 1., 1.]
    (time.sum() + frequency.sum() + ref.sum()).backward()
    assert x.grad[:, 0].tolist() == [3.5, 3., 4.]
    assert core.t_proj(x).equal(x)


def test_nine_fixed_rows():
    rows = plan.candidate_rows()
    assert len(rows) == 9
    assert {row['arm'] for row in rows} == {'balanced', 'physical', 'complementary'}
