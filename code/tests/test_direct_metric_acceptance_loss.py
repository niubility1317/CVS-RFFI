import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import direct_metric_acceptance_loss  # noqa: E402


def test_direct_metric_acceptance_loss_reports_targets_and_backpropagates():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.965, 0.260],
            [0.0, 1.0],
            [0.087, 0.996],
            [0.260, 0.965],
            [-1.0, 0.0],
            [-0.996, 0.087],
            [-0.965, 0.260],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    domains = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])

    loss, metrics = direct_metric_acceptance_loss(
        features,
        labels,
        domains,
        virtual_count=9,
        virtual_mode="hard",
        core_quantile=0.50,
        accept_quantile=0.67,
        tail_quantile=0.80,
        overflow_quantile=0.90,
        zid_p50_target_rad=math.radians(5.0),
        zid_p95_target_rad=math.radians(8.0),
        zid_p99_target_rad=math.radians(10.0),
        zid_tail_cvar_target_rad=math.radians(8.0),
        source_overflow_target=0.20,
        proxy_vaccept_target=0.20,
        bridge_accept_target=0.20,
        low_density_accept_target=0.10,
        tail_accept_target=0.25,
        overflow_accept_target=0.20,
        radius_inter_ratio_target=0.20,
        core_accept_target=0.75,
        zid_quantile_weight=1.0,
        source_overflow_weight=1.0,
        proxy_vaccept_weight=1.0,
        bridge_accept_weight=1.0,
        low_density_accept_weight=1.0,
        tail_accept_weight=1.0,
        overflow_accept_weight=1.0,
        radius_inter_ratio_weight=1.0,
        core_accept_weight=0.25,
        accept_cvar_alpha=0.50,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["active"] == 1.0
    assert metrics["active_classes"] == 3.0
    assert metrics["virtual_count"] == 9.0
    assert metrics["zid_p99_deg"] >= metrics["zid_p95_deg"] >= metrics["zid_p50_deg"]
    assert metrics["zid_quantile_loss"] > 0.0
    assert metrics["proxy_vaccept_loss"] >= 0.0
    assert metrics["bridge_accept_loss"] >= 0.0
    assert metrics["radius_inter_ratio_loss"] >= 0.0
    assert "source_overflow" in metrics
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_direct_metric_acceptance_loss_uses_concat_sat_pair_view():
    clean = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.996, 0.087],
            [0.087, 0.996],
            [-0.996, 0.087],
        ],
        dtype=torch.float32,
    )
    sat = torch.tensor(
        [
            [0.985, 0.174],
            [0.174, 0.985],
            [-0.985, 0.174],
            [0.966, 0.259],
            [0.259, 0.966],
            [-0.966, 0.259],
        ],
        dtype=torch.float32,
    )
    features = torch.cat([clean, sat], dim=0).requires_grad_(True)
    labels = torch.tensor([0, 1, 2, 0, 1, 2] * 2)
    domains = torch.tensor([0, 0, 0, 1, 1, 1] * 2)

    loss, metrics = direct_metric_acceptance_loss(
        features,
        labels,
        domains,
        paired_view_count=6,
        virtual_count=6,
        sat_pair_weight=1.0,
        sat_pair_target_rad=math.radians(2.0),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["sat_pair_angle_p95_deg"] > 2.0
    assert metrics["sat_pair_loss"] > 0.0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_direct_metric_acceptance_loss_reports_stable_geometry_contract():
    torch.manual_seed(20260708)
    centers = torch.randn(4, 8)
    centers = torch.nn.functional.normalize(centers, dim=1)
    centers[1] = torch.nn.functional.normalize(0.995 * centers[0] + 0.005 * centers[1], dim=0)
    features = []
    labels = []
    domains = []
    for cls in range(4):
        for dom in range(4):
            noise = 0.025 * torch.randn(8)
            features.append(centers[cls] + noise)
            labels.append(cls)
            domains.append(dom)
    features = torch.stack(features, dim=0)
    features = torch.nn.functional.normalize(features, dim=1).requires_grad_(True)

    loss, metrics = direct_metric_acceptance_loss(
        features,
        torch.tensor(labels),
        torch.tensor(domains),
        virtual_count=24,
        virtual_mode="hard",
        core_quantile=0.50,
        accept_quantile=0.65,
        tail_quantile=0.80,
        overflow_quantile=0.90,
        zid_p50_target_rad=math.radians(4.0),
        zid_p95_target_rad=math.radians(7.0),
        zid_p99_target_rad=math.radians(9.0),
        zid_tail_cvar_target_rad=math.radians(8.0),
        proxy_vaccept_target=0.12,
        bridge_accept_target=0.10,
        low_density_accept_target=0.08,
        tail_accept_target=0.12,
        overflow_accept_target=0.08,
        radius_inter_ratio_target=0.35,
        quantile_temperature_rad=math.radians(2.0),
        accept_temperature=0.025,
        component_temperature_rad=math.radians(2.0),
        density_temperature_rad=math.radians(2.0),
        accept_cvar_alpha=0.20,
    )
    loss.backward()

    assert metrics["geometry_stabilized"] == 1.0
    assert metrics["geometry_reference_detached"] == 1.0
    assert metrics["angle_clamp_eps"] >= 1e-4
    assert torch.isfinite(loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
