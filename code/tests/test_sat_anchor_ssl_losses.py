import torch

from cvsrffi.muse_ssdg import (
    MUSETrainingHeads,
    sat_anchor_clean_kl,
    trusted_satellite_cross_entropy,
)


def test_sat_anchor_pair_loss_is_symmetric_and_backpropagates_to_both_student_views():
    heads = MUSETrainingHeads(4, 3, 2, 2, 2)
    clean = torch.randn(5, 4, requires_grad=True)
    satellite = torch.randn(5, 4, requires_grad=True)

    loss = heads.sat_anchor_pair_loss(clean, satellite)
    reverse = heads.sat_anchor_pair_loss(satellite, clean)
    loss.backward()

    assert torch.allclose(loss, reverse, atol=1e-6)
    assert clean.grad is not None and clean.grad.abs().sum().item() > 0
    assert satellite.grad is not None and satellite.grad.abs().sum().item() > 0


def test_clean_anchor_kl_never_backpropagates_to_frozen_teacher():
    student = torch.tensor([[2.0, -1.0], [0.5, 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.5, -0.5], [0.8, -0.2]], requires_grad=True)

    loss = sat_anchor_clean_kl(student, teacher, temperature=2.0)
    loss.backward()

    assert student.grad is not None and student.grad.abs().sum().item() > 0
    assert teacher.grad is None


def test_trusted_satellite_ce_uses_full_unlabeled_batch_denominator():
    logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]],
        requires_grad=True,
    )
    pseudo = torch.tensor([0, 1, 0, 1])
    mask = torch.tensor([True, False, False, False])

    loss = trusted_satellite_cross_entropy(
        logits,
        pseudo,
        mask,
        full_unlabeled_batch_size=4,
    )
    selected_mean = torch.nn.functional.cross_entropy(logits[mask], pseudo[mask])

    assert torch.allclose(loss, selected_mean / 4.0)


def test_empty_trusted_satellite_ce_is_graph_safe_zero():
    logits = torch.randn(3, 2, requires_grad=True)
    loss = trusted_satellite_cross_entropy(
        logits,
        torch.tensor([0, 1, 0]),
        torch.zeros(3, dtype=torch.bool),
        full_unlabeled_batch_size=3,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0
