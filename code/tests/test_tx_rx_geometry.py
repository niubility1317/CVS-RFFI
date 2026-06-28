import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.tx_rx_geometry import (  # noqa: E402
    masked_supcon_loss,
    pair_masks,
    tx_rx_anova_metrics,
    tx_rx_rectangle_identity_loss,
    tx_rx_rectangle_receiver_loss,
)


def test_pair_masks_and_masked_supcon_use_expected_relationships():
    y = torch.tensor([0, 0, 1, 1])
    d = torch.tensor([0, 1, 0, 1])
    z = torch.randn(4, 3, requires_grad=True)
    masks = pair_masks(y, d)

    loss = masked_supcon_loss(z, masks["same_tx_cross_domain"], masks["valid_pair"])
    loss.backward()

    assert masks["same_tx_cross_domain"].sum().item() == 4
    assert torch.isfinite(loss.detach())
    assert z.grad is not None


def test_rectangle_losses_return_nonzero_coverage_for_full_four_corner_batch():
    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1])
    d = torch.tensor([0, 1, 0, 1])

    loss_tx, n_tx = tx_rx_rectangle_identity_loss(z, y, d)
    loss_rx, n_rx = tx_rx_rectangle_receiver_loss(z, y, d)

    assert n_tx == 1
    assert n_rx == 1
    assert torch.isfinite(loss_tx)
    assert torch.isfinite(loss_rx)


def test_rectangle_loss_empty_coverage_is_graph_safe_zero():
    z = torch.randn(3, 2, requires_grad=True)
    loss, n = tx_rx_rectangle_identity_loss(z, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]))
    loss.backward()

    assert n == 0
    assert float(loss.detach().item()) == 0.0
    assert z.grad is not None


def test_tx_rx_anova_metrics_are_finite_for_constant_features():
    z = torch.ones(4, 3)
    metrics = tx_rx_anova_metrics(z, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))

    assert metrics["var_total"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["var_tx_ratio"]))
