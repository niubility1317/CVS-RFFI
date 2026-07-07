import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import open_world_feature_space_loss, source_episode_three_sigma_loss  # noqa: E402


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


def test_open_world_feature_space_loss_tail_mode_penalizes_class_tail_and_reports_three_sigma():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    compact = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.985, 0.174], [0.966, 0.259],
         [0.0, 1.0], [0.087, 0.996], [0.174, 0.985], [0.259, 0.966]],
        dtype=torch.float32,
        requires_grad=True,
    )
    tail = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.985, 0.174], [0.0, 1.0],
         [0.0, 1.0], [0.087, 0.996], [0.174, 0.985], [1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    compact_loss, compact_metrics = open_world_feature_space_loss(
        compact,
        labels,
        min_classes=2,
        tail_mode="robust_3sigma",
        tail_weight=1.0,
        cvar_alpha=0.75,
    )
    tail_loss, tail_metrics = open_world_feature_space_loss(
        tail,
        labels,
        min_classes=2,
        tail_mode="robust_3sigma",
        tail_weight=1.0,
        cvar_alpha=0.75,
    )
    tail_loss.backward()

    assert tail_loss.item() > compact_loss.item()
    assert tail_metrics["tail_loss"] > compact_metrics["tail_loss"]
    assert tail_metrics["tail_frac_gt_3sigma"] > compact_metrics["tail_frac_gt_3sigma"]
    assert tail_metrics["pos_angle_p95_deg"] >= tail_metrics["pos_angle_p50_deg"]
    assert tail.grad is not None
    assert torch.isfinite(tail.grad).all()


def test_open_world_feature_space_loss_vacuum_penalizes_foreign_tail_intrusion():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.0, 1.0], [-0.087, 0.996]],
        dtype=torch.float32,
        requires_grad=True,
    )
    intruding = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.999, 0.035], [0.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    good_loss, good_metrics = open_world_feature_space_loss(
        separated,
        labels,
        min_classes=2,
        vacuum_weight=1.0,
        vacuum_width_rad=math.radians(5.0),
        vacuum_hard_k=1,
    )
    bad_loss, bad_metrics = open_world_feature_space_loss(
        intruding,
        labels,
        min_classes=2,
        vacuum_weight=1.0,
        vacuum_width_rad=math.radians(5.0),
        vacuum_hard_k=1,
    )
    bad_loss.backward()

    assert bad_metrics["vacuum_loss"] > good_metrics["vacuum_loss"]
    assert bad_metrics["vacuum_violation_rate"] > good_metrics["vacuum_violation_rate"]
    assert bad_metrics["vacuum_min_neg_angle_deg"] < bad_metrics["vacuum_boundary_deg"]
    assert bad_loss.item() > good_loss.item()
    assert intruding.grad is not None
    assert torch.isfinite(intruding.grad).all()


def test_source_episode_three_sigma_loss_penalizes_leave_domain_tail_without_target_data():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    good = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.985, 0.174], [0.966, 0.259],
         [0.0, 1.0], [0.087, 0.996], [0.174, 0.985], [0.259, 0.966]],
        dtype=torch.float32,
        requires_grad=True,
    )
    bad = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.0, 1.0], [0.087, 0.996],
         [0.0, 1.0], [0.087, 0.996], [1.0, 0.0], [0.996, 0.087]],
        dtype=torch.float32,
        requires_grad=True,
    )

    good_loss, good_metrics = source_episode_three_sigma_loss(good, labels, domains, min_domains=2, radius_mode="three_sigma")
    bad_loss, bad_metrics = source_episode_three_sigma_loss(bad, labels, domains, min_domains=2, radius_mode="three_sigma")
    bad_loss.backward()

    assert bad_loss.item() > good_loss.item()
    assert bad_metrics["source_episode_overflow_rate"] > good_metrics["source_episode_overflow_rate"]
    assert bad_metrics["source_episode_domains"] == 2.0
    assert bad.grad is not None
    assert torch.isfinite(bad.grad).all()


def test_source_episode_core_quantile_mode_does_not_let_tail_define_safe_shell():
    labels = torch.tensor([0, 0, 0, 0, 0, 0])
    domains = torch.tensor([0, 0, 0, 1, 1, 1])
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.966, 0.259],
            [0.906, 0.423],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    shell_loss, shell_metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        radius_mode="three_sigma",
        radius_cap_rad=math.radians(90.0),
    )
    core_loss, core_metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        radius_mode="core_quantile",
        core_quantile=0.50,
        radius_cap_rad=math.radians(90.0),
    )
    core_loss.backward()

    assert core_metrics["source_episode_radius_safe_deg"] < shell_metrics["source_episode_radius_safe_deg"]
    assert core_metrics["source_episode_overflow_rate"] > shell_metrics["source_episode_overflow_rate"]
    assert core_metrics["source_overflow"] == core_metrics["source_episode_overflow_rate"]
    assert core_loss.item() > shell_loss.item()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_source_episode_default_radius_is_core_safe_not_three_sigma_shell():
    labels = torch.tensor([0, 0, 0, 0, 0, 0])
    domains = torch.tensor([0, 0, 0, 1, 1, 1])
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.966, 0.259],
            [0.906, 0.423],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    default_loss, default_metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        core_quantile=0.50,
        radius_cap_rad=math.radians(90.0),
    )
    _, shell_metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        radius_mode="three_sigma",
        core_quantile=0.50,
        radius_cap_rad=math.radians(90.0),
    )
    default_loss.backward()

    assert default_metrics["source_episode_radius_mode_code"] == 2.0
    assert default_metrics["source_episode_radius_safe_deg"] < shell_metrics["source_episode_radius_safe_deg"]
    assert default_metrics["source_overflow"] == default_metrics["source_episode_overflow_rate"]
    assert torch.isfinite(default_loss)


def test_source_episode_reports_receiver_local_component_tri_state_and_quantiles():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.985, 0.174],
            [0.174, 0.985],
            [0.0, 1.0],
            [0.087, 0.996],
            [0.174, 0.985],
            [0.985, 0.174],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, metrics = source_episode_three_sigma_loss(
        features,
        labels,
        domains,
        min_domains=2,
        radius_mode="core_quantile",
        core_quantile=0.50,
        radius_cap_rad=math.radians(90.0),
    )
    loss.backward()

    assert metrics["source_episode_receiver_local_component_count"] == 4.0
    assert (
        metrics["source_episode_core_count"]
        + metrics["source_episode_tail_count"]
        + metrics["source_episode_outside_count"]
    ) == 8.0
    assert metrics["source_episode_tail_count"] >= 0.0
    assert metrics["source_episode_outside_count"] > 0.0
    assert metrics["source_episode_core_tail_outside_ready"] == 1.0
    assert metrics["source_episode_density_gate_active"] == 1.0
    assert metrics["source_episode_zid_p99_deg"] >= metrics["source_episode_zid_p95_deg"] >= metrics["source_episode_zid_p50_deg"]
    assert metrics["source_episode_zid_tail_cvar_deg"] >= metrics["source_episode_zid_p95_deg"]
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
