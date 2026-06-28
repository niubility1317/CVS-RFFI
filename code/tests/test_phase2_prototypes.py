import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.phase2_prototypes import (  # noqa: E402
    BalancedPrototypeBank,
    PrototypeRadiusTracker,
    TxDomainPrototypeBank,
    prototype_geometry_summary,
)


def test_balanced_prototype_bank_averages_domain_centers_not_sample_counts():
    bank = BalancedPrototypeBank(num_items=1, feat_dim=2, momentum=0.0, min_count_per_update=1)
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([0, 0, 0])
    d = torch.tensor([0, 0, 1])

    stats = bank.update_from_features(z, y, d)
    proto = bank.get()[0]

    assert stats["updated"] == 1.0
    assert torch.allclose(proto, torch.tensor([0.7071, 0.7071]), atol=1e-3)


def test_tx_domain_bank_computes_public_shift_and_interaction():
    tx_bank = BalancedPrototypeBank(num_items=2, feat_dim=2, momentum=0.0, min_count_per_update=1)
    tx_bank.update_from_features(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0, 1]),
    )
    local = TxDomainPrototypeBank(num_tx=2, num_domains=1, feat_dim=2, momentum=0.0)
    local.update(
        torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )

    shifts = local.compute_domain_shifts(tx_bank)

    assert shifts["mask"].sum().item() == 2
    assert shifts["domain_counts"][0].item() == 2
    assert shifts["domain_shift"].shape == (1, 2)


def test_radius_tracker_and_geometry_summary_report_margin_violations():
    bank = BalancedPrototypeBank(num_items=2, feat_dim=2, momentum=0.0, min_count_per_update=1)
    bank.update_from_features(
        torch.tensor([[1.0, 0.0], [0.5, 0.866]], dtype=torch.float32),
        torch.tensor([0, 1]),
    )
    tracker = PrototypeRadiusTracker(num_classes=2)
    tracker.update(
        torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.5, 0.866], [0.6, 0.8]], dtype=torch.float32),
        torch.tensor([0, 0, 1, 1]),
        bank.get(),
    )

    radii = tracker.radii_tensor()
    summary = prototype_geometry_summary(bank.get(), radii, gamma_open_rad=math.radians(80), initialized=bank.initialized_mask())

    assert summary.initialized == 2
    assert summary.margin_violation_pairs == 1
    assert summary.min_interclass_angle_deg > 0.0


def test_proto_loss_returns_graph_safe_zero_when_no_initialized_class():
    bank = BalancedPrototypeBank(num_items=2, feat_dim=2)
    z = torch.randn(3, 2, requires_grad=True)
    loss, metrics = bank.prototype_pull_margin_loss(z, torch.tensor([0, 1, 1]))
    loss.backward()

    assert float(loss.detach().item()) == 0.0
    assert z.grad is not None
    assert metrics["active"] == 0.0
