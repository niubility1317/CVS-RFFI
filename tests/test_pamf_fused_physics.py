import pytest
import torch

from experiments.pamf_physics_tri.fused_physics import paired_fusion_loss
from experiments.pamf_physics_tri import fused_physics_plan


def test_only_selected_physical_packets_train_altered_fusion():
    raw = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    changed = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    loss = paired_fusion_loss(raw, changed, torch.tensor([True, False]))
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert raw.grad is None
    assert changed.grad[0].abs().sum() > 0
    assert torch.equal(changed.grad[1], torch.zeros(2))


def test_rejects_misaligned_selection_and_keeps_three_seed_matrix():
    with pytest.raises(ValueError):
        paired_fusion_loss(torch.ones(2, 3), torch.ones(2, 3), torch.tensor([True]))
    rows = fused_physics_plan.candidate_rows()
    assert {r['protocol_id'] for r in rows} == {'m1-natural'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}
