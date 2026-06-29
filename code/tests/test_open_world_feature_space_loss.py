import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import open_world_feature_space_loss  # noqa: E402


def test_open_world_feature_space_loss_penalizes_collapsed_class_geometry():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [0.98, 0.05], [0.0, 1.0], [0.05, 0.98]],
        dtype=torch.float32,
        requires_grad=True,
    )
    collapsed = torch.tensor(
        [[1.0, 0.0], [0.98, 0.04], [0.92, 0.12], [0.88, 0.18]],
        dtype=torch.float32,
        requires_grad=True,
    )

    good_loss, good_metrics = open_world_feature_space_loss(
        separated,
        labels,
        radius_rad=math.radians(12.0),
        inter_margin_rad=math.radians(55.0),
        sample_margin_rad=math.radians(5.0),
    )
    bad_loss, bad_metrics = open_world_feature_space_loss(
        collapsed,
        labels,
        radius_rad=math.radians(12.0),
        inter_margin_rad=math.radians(55.0),
        sample_margin_rad=math.radians(5.0),
    )

    assert bad_loss.item() > good_loss.item()
    assert bad_metrics["inter"] > good_metrics["inter"]
    bad_loss.backward()
    assert collapsed.grad is not None


def test_open_world_feature_space_loss_reports_domain_center_misalignment():
    features = torch.tensor(
        [[1.0, 0.0], [0.98, 0.05], [0.0, 1.0], [0.05, 0.98]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 0])
    domains = torch.tensor([0, 0, 1, 1])

    loss, metrics = open_world_feature_space_loss(
        features,
        labels,
        domains,
        radius_rad=math.radians(10.0),
        inter_margin_rad=math.radians(30.0),
        sample_margin_rad=math.radians(5.0),
        domain_align_weight=1.0,
        min_classes=1,
    )

    assert torch.isfinite(loss)
    assert metrics["domain_align"] > 0.0
    assert metrics["active_classes"] == 1.0


def test_open_world_feature_space_loss_has_finite_gradients_on_near_duplicate_features():
    torch.manual_seed(7)
    base = torch.randn(1, 16)
    features = (base.repeat(12, 1) + 1e-4 * torch.randn(12, 16)).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    loss, metrics = open_world_feature_space_loss(
        features,
        labels,
        domains,
        radius_rad=math.radians(12.0),
        inter_margin_rad=math.radians(55.0),
        sample_margin_rad=math.radians(5.0),
        domain_align_weight=0.03,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert metrics["active_classes"] == 6.0


def test_open_world_feature_space_loss_returns_graph_safe_zero_without_enough_classes():
    features = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 0])

    loss, metrics = open_world_feature_space_loss(features, labels, min_classes=2)
    loss.backward()

    assert loss.item() == 0.0
    assert features.grad is not None
    assert metrics["active_classes"] == 1.0
