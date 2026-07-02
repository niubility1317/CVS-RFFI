import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import proxy_unknown_energy_loss  # noqa: E402


def test_proxy_unknown_energy_loss_keeps_holdout_auxiliary_and_penalizes_known_acceptance():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.0, 1.0],
            [0.05, 0.98],
            [-1.0, 0.0],
            [-0.98, 0.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss, metrics = proxy_unknown_energy_loss(
        features,
        labels,
        holdout_label=2,
        virtual_count=4,
        energy_margin=0.25,
        placeholder_weight=0.5,
        virtual_detach=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["proxy_unknown_count"] == 2.0
    assert metrics["known_count"] == 4.0
    assert metrics["virtual_count"] == 4.0
    assert metrics["energy_proxy"] > metrics["energy_known"]
    assert 0.0 <= metrics["proxy_unknown_auc"] <= 1.0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_proxy_unknown_energy_loss_is_graph_safe_when_no_holdout_samples():
    features = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])

    loss, metrics = proxy_unknown_energy_loss(features, labels, holdout_label=2)
    loss.backward()

    assert loss.item() == 0.0
    assert features.grad is not None
    assert metrics["active"] == 0.0


def test_proxy_unknown_energy_loss_vacuum_penalizes_unknown_inside_known_tail():
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    safe_unknown = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.0, 1.0],
            [0.087, 0.996],
            [-1.0, 0.0],
            [-0.996, -0.087],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    intruding_unknown = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.0, 1.0],
            [0.087, 0.996],
            [0.999, 0.035],
            [0.996, 0.087],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    safe_loss, safe_metrics = proxy_unknown_energy_loss(
        safe_unknown,
        labels,
        holdout_label=2,
        virtual_count=0,
        vacuum_weight=1.0,
        vacuum_width_rad=0.25,
        vacuum_hard_k=1,
        vacuum_radius_rad=0.10,
    )
    bad_loss, bad_metrics = proxy_unknown_energy_loss(
        intruding_unknown,
        labels,
        holdout_label=2,
        virtual_count=0,
        vacuum_weight=1.0,
        vacuum_width_rad=0.25,
        vacuum_hard_k=1,
        vacuum_radius_rad=0.10,
    )
    bad_loss.backward()

    assert bad_metrics["vacuum_loss"] > safe_metrics["vacuum_loss"]
    assert bad_metrics["vacuum_violation_rate"] > safe_metrics["vacuum_violation_rate"]
    assert bad_metrics["vacuum_margin_deg"] < 0.0
    assert torch.isfinite(bad_loss)
    assert intruding_unknown.grad is not None
    assert torch.isfinite(intruding_unknown.grad).all()


def test_proxy_unknown_core_radius_mode_reports_smaller_accept_gate_than_three_sigma_tail():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.087, 0.996],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-0.996, 0.087],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    tail_loss, tail_metrics = proxy_unknown_energy_loss(
        features,
        labels,
        holdout_label=2,
        virtual_count=6,
        virtual_mode="hard",
        component_radius_mode="three_sigma",
        core_quantile=0.50,
    )
    core_loss, core_metrics = proxy_unknown_energy_loss(
        features,
        labels,
        holdout_label=2,
        virtual_count=6,
        virtual_mode="hard",
        component_radius_mode="core_quantile",
        component_radius_quantile=0.50,
        core_quantile=0.50,
        low_density_accept_weight=1.0,
        radius_inter_ratio_weight=1.0,
    )
    core_loss.backward()

    assert core_metrics["component_gate_radius_p95_deg"] < tail_metrics["component_gate_radius_p95_deg"]
    assert core_metrics["low_density_accept_rate"] == core_metrics["low_density_accept_prob"]
    assert core_metrics["radius_to_inter_ratio"] == core_metrics["radius_inter_ratio"]
    assert core_metrics["vaccept_surrogate_CVaR"] == core_metrics["vaccept_surrogate"]
    assert core_metrics["proxy_vaccept"] == core_metrics["virtual_accept_rate"]
    assert torch.isfinite(core_loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
