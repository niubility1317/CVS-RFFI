import math
import sys
from pathlib import Path

import torch
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import (  # noqa: E402
    direct_metric_acceptance_loss,
    multiview_direct_metric_acceptance_loss,
    multiview_source_episode_three_sigma_loss,
    source_episode_three_sigma_loss,
    tx_conditional_domain_invariance_loss,
)


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


def test_bridge_acceptance_term_has_nonzero_gradient_when_virtual_geometry_is_trainable():
    features = torch.tensor(
        [[1.0, 0.0], [0.98, 0.10], [0.0, 1.0], [0.10, 0.98]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])

    loss, metrics = direct_metric_acceptance_loss(
        features,
        labels,
        virtual_count=6,
        virtual_mode="hard",
        virtual_detach=False,
        zid_quantile_weight=0.0,
        source_overflow_weight=0.0,
        proxy_vaccept_weight=0.0,
        bridge_accept_weight=1.0,
        low_density_accept_weight=0.0,
        tail_accept_weight=0.0,
        overflow_accept_weight=0.0,
        radius_inter_ratio_weight=0.0,
        core_accept_weight=0.0,
        bridge_accept_target=0.0,
    )
    loss.backward()

    assert metrics["bridge_accept_loss"] > 0.0
    assert metrics["geometry_reference_detached"] == 0.0
    assert features.grad is not None
    assert float(features.grad.norm().item()) > 0.0


def test_receiver_local_component_terms_backpropagate_without_leave_domain_episode():
    features = torch.tensor(
        [[1.0, 0.0], [0.98, 0.10], [0.0, 1.0], [0.10, 0.98]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.zeros(4, dtype=torch.long)

    loss, metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        min_samples_per_class_domain=2,
        local_component_compact_weight=1.0,
        local_component_inter_weight=1.0,
        local_component_accept_weight=1.0,
        local_component_density_weight=1.0,
        local_component_inter_margin_rad=math.radians(80.0),
    )
    loss.backward()

    assert metrics["source_episode_receiver_local_component_count"] == 2.0
    assert metrics["source_episode_local_component_structural_active"] == 1.0
    assert torch.isfinite(loss)
    assert features.grad is not None
    assert float(features.grad.norm().item()) > 0.0


def test_multiview_direct_metric_optimizes_clean_and_satellite_geometry_separately():
    clean = torch.tensor(
        [[1.0, 0.0], [0.98, 0.10], [0.0, 1.0], [0.10, 0.98]],
        dtype=torch.float32,
        requires_grad=True,
    )
    satellite = torch.tensor(
        [[0.94, 0.34], [0.90, 0.43], [0.34, 0.94], [0.43, 0.90]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1])

    loss, metrics = multiview_direct_metric_acceptance_loss(
        clean,
        satellite,
        labels,
        domains,
        virtual_count=6,
        virtual_detach=False,
        pair_weight=1.0,
        sat_pair_target_rad=math.radians(5.0),
    )
    loss.backward()

    assert metrics["multiview_separate_geometry"] == 1.0
    assert metrics["clean_active"] == 1.0
    assert metrics["sat_active"] == 1.0
    assert metrics["sat_zid_p95_deg"] != metrics["clean_zid_p95_deg"]
    assert metrics["sat_pair_angle_p95_deg"] > 5.0
    assert clean.grad is not None and satellite.grad is not None
    assert float(clean.grad.norm().item()) > 0.0
    assert float(satellite.grad.norm().item()) > 0.0


def test_multiview_source_episode_does_not_pool_satellite_tail_into_clean_components():
    clean = torch.tensor(
        [[1.0, 0.0], [0.99, 0.03], [0.0, 1.0], [0.03, 0.99]],
        dtype=torch.float32,
        requires_grad=True,
    )
    satellite = torch.tensor(
        [[0.94, 0.34], [0.90, 0.43], [0.34, 0.94], [0.43, 0.90]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1])

    loss, metrics = multiview_source_episode_three_sigma_loss(
        clean,
        satellite,
        labels,
        domains,
        min_domains=2,
        local_component_compact_weight=1.0,
        local_component_accept_weight=1.0,
        local_component_density_weight=1.0,
    )
    loss.backward()

    assert metrics["source_episode_multiview_separate_geometry"] == 1.0
    assert "clean_source_episode_zid_p99_deg" in metrics
    assert "sat_source_episode_zid_p99_deg" in metrics
    clean_loss, _ = source_episode_three_sigma_loss(
        clean,
        labels,
        domains,
        min_domains=2,
        local_component_compact_weight=1.0,
        local_component_accept_weight=1.0,
        local_component_density_weight=1.0,
    )
    sat_loss, _ = source_episode_three_sigma_loss(
        satellite,
        labels,
        domains,
        min_domains=2,
        local_component_compact_weight=1.0,
        local_component_accept_weight=1.0,
        local_component_density_weight=1.0,
    )
    assert metrics["source_episode_multiview_normalized"] == 1.0
    assert loss.detach().item() == pytest.approx(
        0.5 * (clean_loss.detach().item() + sat_loss.detach().item()),
        rel=1e-5,
    )
    assert clean.grad is not None and satellite.grad is not None


def test_source_episode_density_is_bounded_for_tiny_radius_outlier_component():
    class0 = [[1.0, 0.0] for _ in range(20)] + [[0.0, 1.0]]
    class1 = [[0.0, 1.0] for _ in range(20)] + [[-1.0, 0.0]]
    features = torch.tensor(class0 + class1, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0] * 21 + [1] * 21)
    domains = torch.zeros(42, dtype=torch.long)

    loss, metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        local_component_min_samples=2,
        local_component_radius_floor_rad=math.radians(3.0),
        local_component_density_weight=1.0,
        local_component_density_cap=2.0,
        local_component_term_cap=4.0,
    )
    loss.backward()

    assert metrics["source_episode_local_component_density_raw_loss"] > 2.0
    assert 0.0 <= metrics["source_episode_local_component_density_loss"] <= 2.0
    assert metrics["source_episode_local_component_radius_floor_rate"] > 0.0
    assert loss.detach().item() <= metrics["source_episode_loss_upper_bound"] + 1e-6
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_direct_metric_uses_receiver_local_components_without_global_ball_fallback():
    features = torch.tensor(
        [
            [1.0, 0.00], [0.99, 0.03], [0.94, 0.34], [0.92, 0.39],
            [0.0, 1.00], [0.03, 0.99], [0.34, 0.94], [0.39, 0.92],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])

    loss, metrics = direct_metric_acceptance_loss(
        features,
        labels,
        domains,
        virtual_count=8,
        virtual_detach=False,
        use_domain_local_components=True,
        require_domain_local_components=True,
        min_samples_per_component=2,
    )
    loss.backward()

    assert metrics["active"] == 1.0
    assert metrics["domain_local_component_gate"] == 1.0
    assert metrics["global_ball_accept"] == 0.0
    assert metrics["local_component_count"] == 4.0
    assert metrics["local_component_class_coverage"] == 2.0
    assert math.isfinite(metrics["local_zid_p99_deg"])
    assert metrics["quantile_optimization_scope_local"] == 1.0
    assert metrics["global_diag_zid_p95_deg"] > metrics["local_zid_p95_deg"]
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_tx_conditional_invariance_aligns_nuisance_groups_without_cross_tx_collapse():
    features = torch.tensor(
        [
            [1.0, 0.0], [0.98, 0.2], [1.0, 0.0], [0.98, 0.2],
            [0.0, 1.0], [0.2, 0.98], [0.0, 1.0], [0.2, 0.98],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    tx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    receiver = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    day = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    channel = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])

    loss, metrics = tx_conditional_domain_invariance_loss(
        features,
        tx,
        receiver_labels=receiver,
        day_labels=day,
        channel_labels=channel,
        min_samples_per_group=1,
    )
    loss.backward()

    assert metrics["active"] == 1.0
    assert metrics["receiver_active"] == 1.0
    assert metrics["day_active"] == 1.0
    assert metrics["channel_active"] == 1.0
    assert float(loss.item()) > 0.0
    assert features.grad is not None
    assert float(features.grad.norm().item()) > 0.0


def test_multiview_local_tail_projection_uses_worst_satellite_view():
    clean = torch.tensor(
        [
            [1.0, 0.00], [0.999, 0.02], [0.94, 0.34], [0.93, 0.36],
            [0.0, 1.00], [0.02, 0.999], [0.34, 0.94], [0.36, 0.93],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    satellite = torch.tensor(
        [
            [1.0, 0.00], [0.94, 0.34], [0.94, 0.34], [0.75, 0.66],
            [0.0, 1.00], [0.34, 0.94], [0.34, 0.94], [0.66, 0.75],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])

    loss, metrics = multiview_direct_metric_acceptance_loss(
        clean,
        satellite,
        labels,
        domains,
        use_domain_local_components=True,
        require_domain_local_components=True,
        min_samples_per_component=2,
        virtual_count=8,
    )
    loss.backward()

    assert metrics["domain_local_component_gate"] == 1.0
    assert metrics["global_ball_accept"] == 0.0
    assert metrics["sat_local_zid_p99_deg"] > metrics["clean_local_zid_p99_deg"]
    assert metrics["local_zid_p99_deg"] == metrics["sat_local_zid_p99_deg"]
