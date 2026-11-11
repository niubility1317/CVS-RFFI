import torch

from experiments.pamf_physics_tri.multilevel_physics import paired_branch_loss, paired_prediction_loss
from experiments.pamf_physics_tri import multilevel_physics_plan


def test_soft_consistency_only_trains_selected_physical_predictions():
    raw_a = torch.tensor([[3., 0.], [3., 0.]], requires_grad=True)
    raw_b = torch.tensor([[2., 0.], [2., 0.]], requires_grad=True)
    a = torch.tensor([[0., 2.], [0., 2.]], requires_grad=True)
    b = torch.tensor([[0., 3.], [0., 3.]], requires_grad=True)
    loss = paired_prediction_loss(raw_a, raw_b, a, b, torch.tensor([True, False]))
    assert loss.item() > 0
    loss.backward()
    assert raw_a.grad is None and raw_b.grad is None
    assert a.grad[0].abs().sum() > 0 and b.grad[0].abs().sum() > 0
    assert a.grad[1].abs().sum() == 0 and b.grad[1].abs().sum() == 0


def test_branch_consistency_preserves_raw_and_unselected_paths():
    raw = {'time': torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)}
    changed = {'time': torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)}
    loss = paired_branch_loss(raw, raw, changed, changed, torch.tensor([True, False]))
    loss.backward()
    assert raw['time'].grad is None
    assert changed['time'].grad[0].abs().sum() > 0
    assert changed['time'].grad[1].abs().sum() == 0


def test_three_objectives_share_one_setting_and_three_seeds():
    rows = multilevel_physics_plan.candidate_rows()
    assert len(rows) == 9
    assert {r['arm'] for r in rows} == {'branch', 'soft', 'combined'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}
    assert {r['protocol_id'] for r in rows} == {'m1-natural'}
