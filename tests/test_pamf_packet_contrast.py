import pytest
import torch

from experiments.pamf_physics_tri.packet_contrast import packet_contrast_loss
from experiments.pamf_physics_tri import packet_contrast_plan, pair_runner


def test_same_packet_positive_uses_only_other_pseudo_classes_as_negatives():
    raw = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
    changed = torch.tensor([[.7, .7], [0., 1.], [1., 0.]], requires_grad=True)
    pseudo = torch.tensor([0, 1, 0])
    accepted = torch.tensor([True, True, True])
    selected = torch.tensor([True, False, False])
    loss, count = packet_contrast_loss(raw, changed, pseudo, accepted, selected)
    assert count == 1 and loss.item() > 0
    loss.backward()
    assert raw.grad is None
    assert changed.grad[0].abs().sum() > 0
    assert changed.grad[1:].abs().sum() == 0


def test_no_other_class_yields_zero_and_rejected_slot_is_invalid():
    x = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    accepted = torch.tensor([True, True])
    loss, count = packet_contrast_loss(x, x, torch.tensor([0, 0]), accepted, torch.tensor([True, False]))
    assert count == 0 and loss.item() == 0.
    with pytest.raises(ValueError):
        packet_contrast_loss(x, x, torch.tensor([0, 1]), torch.tensor([True, False]), torch.tensor([False, True]))


def test_two_fixed_arms_three_seeds():
    rows = packet_contrast_plan.candidate_rows()
    assert len(rows) == 6
    assert {r['arm'] for r in rows} == {'contrast', 'joint'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}


def test_disabled_cosine_does_not_block_valid_packet_gradient(monkeypatch):
    monkeypatch.setattr(pair_runner, 'check_single_rx', lambda state: {})
    fused = dict(calls=1, selected=3, cosine_enabled=False, smoke_gradients={'signature': 0., 'reference': 0.})
    packet = dict(calls=1, anchors=4, extra_student_forwards=0,
                  smoke_gradients={'signature': .1, 'reference': .2})
    result = pair_runner.check_packet_contrast({'U_fused_physics': fused, 'U_packet_contrast': packet})
    assert result['U_packet_contrast']['anchors'] == 4
