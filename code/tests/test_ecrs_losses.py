from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import (  # noqa: E402
    basis_gauge_loss,
    different_tx_response_ranking_loss,
    response_gate_calibration_loss,
    response_pair_cross_prediction_loss,
    response_split_fit_loss,
    response_surface_distance,
    same_tx_cross_response_loss,
    stratified_response_split,
)


def test_stratified_split_fit_is_bidirectional_and_content_only() -> None:
    torch.manual_seed(21)
    amplitude = torch.linspace(0.05, 1.2, 80).repeat(2, 1)
    mask_a, mask_b = stratified_response_split(amplitude, bins=8)
    assert torch.all(mask_a ^ mask_b)
    assert abs(int(mask_a.sum()) - int(mask_b.sum())) <= 2

    phi = torch.randn(2, 80, 28, dtype=torch.complex64)
    coef = torch.randn(2, 28, dtype=torch.complex64)
    target = torch.einsum("bnk,bk->bn", phi, coef)
    good = response_split_fit_loss(phi, target, torch.ones(2, 80), amplitude)
    bad = response_split_fit_loss(phi, target.roll(9, dims=1), torch.ones(2, 80), amplitude)
    assert good < 1e-4
    assert bad > good * 100.0


def test_clean_leo_cross_prediction_and_surface_distance_use_pair_identity() -> None:
    torch.manual_seed(22)
    phi = torch.randn(4, 64, 28, dtype=torch.complex64)
    coef = torch.stack(
        [torch.ones(28, dtype=torch.complex64), torch.full((28,), 2.0 + 0j),
         torch.ones(28, dtype=torch.complex64), torch.full((28,), 2.0 + 0j)]
    )
    target = torch.einsum("bnk,bk->bn", phi, coef)
    pair_ids = ["a", "b", "a", "b"]
    clean = torch.tensor([True, True, False, False])
    leo = ~clean
    loss = response_pair_cross_prediction_loss(
        coef, phi, target, pair_ids, clean, leo, torch.ones(4, 64)
    )
    assert loss < 1e-6

    anchor = torch.randn(3, 32, dtype=torch.complex64)
    anchor[1] = anchor[0] + 0.01
    variance = torch.full((3, 32), 0.1)
    assert response_surface_distance(anchor[0], anchor[1], variance[0], variance[1]) < response_surface_distance(
        anchor[0], anchor[2], variance[0], variance[2]
    )


def test_labeled_response_losses_require_cross_receiver_positive_and_matched_negative() -> None:
    torch.manual_seed(23)
    anchor = torch.zeros(4, 32, dtype=torch.complex64)
    anchor[1] = 0.05
    anchor[2] = 2.0
    anchor[3] = 2.05
    labels = torch.tensor([0, 0, 1, 1])
    receiver = torch.tensor([0, 1, 0, 1])
    day = torch.tensor([0, 0, 0, 0])
    label_mask = torch.tensor([True, True, True, False])
    variance = torch.full((4, 32), 0.1)
    design = torch.randn(4, 48, 28, dtype=torch.complex64)
    coefficient = torch.stack(
        [torch.ones(28, dtype=torch.complex64), torch.ones(28, dtype=torch.complex64),
         torch.full((28,), 2.0 + 0j), torch.full((28,), 2.0 + 0j)]
    )
    target = torch.einsum("bnk,bk->bn", design, coefficient)

    same = same_tx_cross_response_loss(
        coefficient,
        design,
        target,
        torch.ones(4, 48),
        labels,
        receiver,
        day,
        label_mask,
    )
    ranking = different_tx_response_ranking_loss(
        anchor, variance, labels, receiver, day, ["clean"] * 4, label_mask, margin=0.5
    )
    assert same < 0.1
    assert ranking < 0.1


def test_gauge_and_gate_calibration_penalize_cheating_routes() -> None:
    phi = torch.eye(28, dtype=torch.complex64).unsqueeze(0)
    assert basis_gauge_loss(phi, torch.ones(1, 28)) < 1e-6
    raw_correct = torch.tensor([False, True, True, False])
    fused_correct = torch.tensor([True, False, True, False])
    calibrated, stats = response_gate_calibration_loss(
        torch.tensor([0.24, 0.01, 0.1, 0.1]), raw_correct, fused_correct, rho_max=0.25
    )
    reversed_loss, _ = response_gate_calibration_loss(
        torch.tensor([0.01, 0.24, 0.1, 0.1]), raw_correct, fused_correct, rho_max=0.25
    )
    assert calibrated < reversed_loss
    assert stats == {"rescue": 1, "harm": 1, "unchanged": 2}
