from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from cvsrffi.phase1_bicad_xr.sampler import build_structured_episode
from cvsrffi.phase1_bicad_xr.xdc import (
    donor_query_matrix,
    fit_receiver_donors,
    xdc_losses,
)


def _two_receiver_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_id = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        requires_grad=True,
    )
    tx = torch.tensor([0, 1, 0, 1])
    receiver = torch.tensor([0, 0, 1, 1])
    return z_id, tx, receiver


def test_ridge_uses_solve_and_produces_finite_weights():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = torch.tensor([0, 1, 0, 1])

    bank = fit_receiver_donors(
        z,
        y,
        torch.tensor([0, 0, 0, 0]),
        num_classes=2,
        ridge=1e-2,
    )

    assert torch.isfinite(bank.weights[0]).all()
    assert bank.valid_receivers.tolist() == [0]


def test_low_coverage_donor_is_skipped():
    bank = fit_receiver_donors(
        torch.randn(3, 4),
        torch.tensor([0, 0, 0]),
        torch.zeros(3, dtype=torch.long),
        num_classes=2,
        ridge=1e-2,
    )

    assert bank.valid_receivers.numel() == 0
    assert "coverage" in bank.skip_reasons[0]


def test_nonfinite_condition_donor_is_skipped():
    bank = fit_receiver_donors(
        torch.full((2, 2), torch.finfo(torch.float32).max),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        num_classes=2,
        ridge=1e-2,
    )

    assert bank.valid_receivers.numel() == 0
    assert "condition" in bank.skip_reasons[0]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_z_id_fails_closed_before_zero_path(bad_value):
    z_id = torch.tensor(
        [[bad_value, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    tx = torch.tensor([0, 1])
    receiver = torch.zeros(2, dtype=torch.long)

    with pytest.raises(ValueError, match="z_id.*finite"):
        xdc_losses(
            z_id,
            tx,
            receiver,
            torch.zeros(2, 2),
            num_classes=2,
        )


def test_fit_receiver_donors_rejects_nonfinite_z_id():
    with pytest.raises(ValueError, match="z_id.*finite"):
        fit_receiver_donors(
            torch.tensor([[1.0, 0.0], [0.0, float("nan")]]),
            torch.tensor([0, 1]),
            torch.zeros(2, dtype=torch.long),
            num_classes=2,
        )


def test_support_accuracy_filter_is_applied():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = torch.tensor([0, 1, 0, 1])

    bank = fit_receiver_donors(
        z,
        y,
        torch.zeros(4, dtype=torch.long),
        num_classes=2,
        ridge=1e-2,
        min_support_accuracy=0.75,
    )

    assert bank.valid_receivers.numel() == 0
    assert "support_accuracy" in bank.skip_reasons[0]


def test_quality_uses_one_formula_and_weights_are_detached():
    z = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
        requires_grad=True,
    )
    y = torch.tensor([0, 1, 0, 1])

    bank = fit_receiver_donors(
        z,
        y,
        torch.zeros(4, dtype=torch.long),
        num_classes=2,
        ridge=1e-2,
    )

    index = int(bank.valid_receivers[0])
    expected = (
        bank.support_accuracy[index]
        * bank.support_margin[index].clamp_min(0.0)
        / torch.log1p(bank.condition_numbers[index])
    )
    assert torch.allclose(bank.quality[index], expected)
    assert not bank.weights.requires_grad


def test_query_features_receive_gradient_but_donor_weights_do_not():
    z_id, tx, receiver = _two_receiver_batch()
    public_logits = torch.zeros(4, 2, requires_grad=True)

    output = xdc_losses(
        z_id,
        tx,
        receiver,
        public_logits,
        num_classes=2,
        temperature=2.0,
    )

    assert output.detached_donor_weights
    assert all(weight.grad is None for weight in output.detached_donor_weights)
    output.total.backward()
    assert z_id.grad is not None
    assert public_logits.grad is not None


def test_kd_uses_temperature_squared_and_detached_ensemble_target():
    z_id, tx, receiver = _two_receiver_batch()
    public_logits = torch.tensor(
        [[0.2, -0.1], [0.0, 0.3], [0.4, -0.2], [-0.2, 0.1]],
        requires_grad=True,
    )
    temperature = 2.0

    output = xdc_losses(
        z_id,
        tx,
        receiver,
        public_logits,
        num_classes=2,
        temperature=temperature,
    )

    expected = F.kl_div(
        F.log_softmax(public_logits / temperature, dim=1),
        F.softmax(output.ensemble_logits.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2
    assert torch.allclose(output.knowledge_distillation, expected)
    assert not output.ensemble_logits.requires_grad


def test_no_valid_cross_receiver_donor_returns_connected_zero_and_reason():
    z_id = torch.randn(4, 3, requires_grad=True)
    tx = torch.tensor([0, 1, 0, 1])
    receiver = torch.zeros(4, dtype=torch.long)
    public_logits = torch.randn(4, 2)

    output = xdc_losses(
        z_id,
        tx,
        receiver,
        public_logits,
        num_classes=2,
        temperature=2.0,
    )

    assert output.skip_reason == "no_valid_cross_receiver_donor"
    assert output.total.requires_grad
    assert output.total.item() == pytest.approx(0.0)
    output.total.backward()
    assert z_id.grad is not None


def test_no_donor_zero_stays_finite_when_full_sum_would_overflow():
    largest = torch.finfo(torch.float32).max
    z_id = torch.full((4, 2), largest, requires_grad=True)
    tx = torch.tensor([0, 1, 0, 1])
    receiver = torch.zeros(4, dtype=torch.long)
    public_logits = torch.full((4, 2), largest)

    output = xdc_losses(
        z_id,
        tx,
        receiver,
        public_logits,
        num_classes=2,
    )

    assert output.skip_reason == "no_valid_cross_receiver_donor"
    assert output.total.requires_grad
    assert torch.isfinite(output.total)
    assert output.total.item() == pytest.approx(0.0)
    output.total.backward()
    assert z_id.grad is not None
    assert torch.isfinite(z_id.grad).all()


def test_negative_receiver_ids_are_rejected_by_all_public_entrypoints():
    z_id = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tx = torch.tensor([0, 1])
    receiver = torch.tensor([-1, -1])
    valid_bank = fit_receiver_donors(
        z_id,
        tx,
        torch.zeros(2, dtype=torch.long),
        num_classes=2,
    )

    with pytest.raises(ValueError, match="receiver.*non-negative"):
        fit_receiver_donors(z_id, tx, receiver, num_classes=2)
    with pytest.raises(ValueError, match="receiver.*non-negative"):
        xdc_losses(z_id, tx, receiver, torch.zeros(2, 2), num_classes=2)
    with pytest.raises(ValueError, match="receiver.*non-negative"):
        donor_query_matrix(z_id, tx, receiver, num_classes=2)
    with pytest.raises(ValueError, match="receiver.*non-negative"):
        xdc_losses(
            z_id,
            tx,
            receiver,
            torch.zeros(2, 2),
            num_classes=2,
            bank=valid_bank,
        )
    with pytest.raises(ValueError, match="receiver.*non-negative"):
        donor_query_matrix(
            z_id,
            tx,
            receiver,
            valid_bank,
            num_classes=2,
        )


def test_donor_query_matrix_marks_only_evaluated_sparse_cells():
    z_id = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    tx = torch.tensor([0, 1, 0, 1, 0, 0])
    receiver = torch.tensor([0, 0, 1, 1, 2, 2])

    matrix = donor_query_matrix(z_id, tx, receiver, num_classes=2)

    assert matrix.shape == (3, 3)
    assert torch.isnan(matrix[0, 0])
    assert torch.isnan(matrix[1, 1])
    assert torch.isfinite(matrix[0, 1])
    assert torch.isfinite(matrix[1, 0])
    assert torch.isfinite(matrix[0, 2])
    assert torch.isfinite(matrix[1, 2])
    assert torch.isnan(matrix[2]).all()


def test_sparse_episode_uses_local_feature_rows_not_physical_id_offsets():
    episode = build_structured_episode(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [10, 11, 20, 21],
        samples_per_cell=2,
        generator=torch.Generator().manual_seed(392001),
        physical_indices=[100, 101, 200, 201],
    )
    z_id = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )

    bank = fit_receiver_donors(
        z_id,
        episode.tx,
        episode.receiver,
        num_classes=2,
        physical_indices=episode.indices,
    )
    output = xdc_losses(
        z_id,
        episode.tx,
        episode.receiver,
        torch.zeros(4, 2),
        num_classes=2,
        physical_indices=episode.indices,
    )

    assert bank.weights.shape == (2, 2, 2)
    assert output.total.requires_grad
    assert math.isfinite(float(output.total.detach()))
