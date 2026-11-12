import torch

from experiments.pamf_physics_tri.hybrid_confidence import HybridConfidence
from experiments.pamf_physics_tri.hybrid_weight import HybridWeight


def test_auxiliary_weight_scales_hybrid_gradient_only(monkeypatch):
    component = torch.tensor(2., requires_grad=True)
    monkeypatch.setattr(HybridConfidence, 'additional_fusion_loss', lambda *_: component)
    objective = HybridWeight.__new__(HybridWeight)
    objective.c = {'hybrid_auxiliary_weight': .25}
    objective.metrics = {}
    result = objective.additional_fusion_loss(None, None)
    result.backward()
    assert result.item() == .5 and component.grad.item() == .25


def test_fixed_matrix_has_four_settings_two_weights_three_seeds():
    from experiments.pamf_physics_tri import hybrid_weight_plan as plan
    rows = plan.candidate_rows()
    assert len(rows) == 24
    assert {row['auxiliary_weight'] for row in rows} == {.25, .5}
    assert len({row['setting'] for row in rows}) == 4
    assert all(plan.baseline_row(row)['arm'] == 'hybrid' for row in rows)
