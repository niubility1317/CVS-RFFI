import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def test_soft_unknown_mixup_loss_uses_soft_labels_energy_and_vacuum_terms():
    from cvsrffi.losses import soft_unknown_mixup_loss

    z = torch.tensor(
        [
            [1.0, 0.00],
            [0.99, 0.05],
            [0.00, 1.0],
            [0.05, 0.99],
            [-1.0, 0.00],
            [-0.99, 0.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.tensor(
        [
            [5.0, -1.0, -1.0],
            [4.5, -0.5, -1.0],
            [-1.0, 5.0, -1.0],
            [-0.5, 4.5, -1.0],
            [-1.0, -1.0, 5.0],
            [-1.0, -0.5, 4.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, info = soft_unknown_mixup_loss(
        z,
        y,
        logits=logits,
        mixup_count=4,
        mixup_order=3,
        alpha=0.5,
        ce_weight=1.0,
        energy_weight=1.0,
        energy_margin=0.5,
        vacuum_weight=1.0,
        vacuum_width_rad=math.radians(5.0),
        generator=torch.Generator().manual_seed(7),
    )

    assert loss.item() > 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert info["soft_unknown_mixup_count"] == 4.0
    assert info["soft_unknown_mixup_order"] == 3.0
    assert info["soft_unknown_mixup_ce"] > 0.0
    assert info["soft_unknown_mixup_energy"] >= 0.0
    assert info["soft_unknown_mixup_vacuum"] >= 0.0
    assert 0.0 <= info["soft_unknown_mixup_virtual_accept_rate"] <= 1.0


def test_source_episode_three_sigma_loss_penalizes_mixup_overflow():
    from cvsrffi.losses import source_episode_three_sigma_loss, make_soft_unknown_mixup

    z = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.02],
            [0.98, -0.02],
            [0.00, 1.00],
            [0.02, 0.99],
            [-0.02, 0.98],
            [-1.00, 0.00],
            [-0.99, 0.02],
            [-0.98, -0.02],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    d = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    mix = make_soft_unknown_mixup(
        z,
        y,
        count=4,
        alpha=0.5,
        mixup_order=3,
        generator=torch.Generator().manual_seed(3),
    )
    assert mix.source_labels.shape == (4, 3)
    for row in mix.source_labels:
        assert torch.unique(row).numel() == 3

    loss, info = source_episode_three_sigma_loss(
        z,
        y,
        d,
        radius_cap_rad=math.radians(6.0),
        mixup_features=mix.features,
        mixup_weight=1.0,
    )

    assert loss.item() > 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert info["source_episode_mixup_count"] == 4.0
    assert info["source_episode_mixup_order"] == 3.0
    assert info["source_episode_mixup_loss"] > 0.0
    assert info["source_episode_mixup_overflow_rate"] > 0.0


def test_proxy_unknown_vaccept_surrogate_penalizes_accepted_unknown_more():
    from cvsrffi.losses import proxy_unknown_energy_loss

    known = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.02],
            [0.00, 1.00],
            [0.02, 0.99],
        ],
        dtype=torch.float32,
    )
    near_unknown = torch.tensor([[0.98, 0.03], [0.97, 0.04]], dtype=torch.float32)
    far_unknown = torch.tensor([[-1.00, 0.00], [-0.99, -0.02]], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1, 2, 2])

    z_near = torch.cat([known, near_unknown], dim=0).requires_grad_(True)
    near_loss, near_info = proxy_unknown_energy_loss(
        z_near,
        y,
        holdout_label=2,
        virtual_count=0,
        energy_margin=0.0,
        placeholder_weight=0.0,
        core_quantile=0.80,
        accept_quantile=0.80,
        vaccept_weight=1.0,
        core_accept_weight=0.25,
        component_gate_weight=0.5,
        vaccept_cvar_alpha=0.5,
        unknown_margin=0.05,
        energy_softplus_temperature=0.05,
    )

    z_far = torch.cat([known, far_unknown], dim=0).requires_grad_(True)
    far_loss, far_info = proxy_unknown_energy_loss(
        z_far,
        y,
        holdout_label=2,
        virtual_count=0,
        energy_margin=0.0,
        placeholder_weight=0.0,
        core_quantile=0.80,
        accept_quantile=0.80,
        vaccept_weight=1.0,
        core_accept_weight=0.25,
        component_gate_weight=0.5,
        vaccept_cvar_alpha=0.5,
        unknown_margin=0.05,
        energy_softplus_temperature=0.05,
    )

    assert near_info["vaccept_surrogate"] > far_info["vaccept_surrogate"]
    assert near_info["hard_proxy_accept_rate"] >= far_info["hard_proxy_accept_rate"]
    assert near_loss.item() > far_loss.item()
    near_loss.backward()
    assert z_near.grad is not None
    assert torch.isfinite(z_near.grad).all()


def test_proxy_unknown_hard_virtual_pool_reports_gate_metrics():
    from cvsrffi.losses import proxy_unknown_energy_loss

    z = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.04],
            [0.98, -0.03],
            [0.00, 1.00],
            [0.04, 0.99],
            [-0.03, 0.98],
            [-1.00, 0.00],
            [-0.99, 0.04],
            [-0.98, -0.03],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    loss, info = proxy_unknown_energy_loss(
        z,
        y,
        holdout_label=2,
        virtual_count=9,
        virtual_mode="hard",
        energy_margin=0.0,
        placeholder_weight=0.0,
        core_quantile=0.80,
        accept_quantile=0.80,
        vaccept_weight=1.0,
        component_gate_weight=1.0,
        tail_quarantine_weight=0.1,
        source_safe_weight=0.1,
        vaccept_cvar_alpha=0.30,
        energy_softplus_temperature=0.05,
    )

    assert loss.item() > 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert info["virtual_count"] == 9.0
    assert info["vaccept_surrogate"] > 0.0
    assert info["component_gate_unknown"] >= 0.0
    for key in ("shell_accept_rate", "bridge_accept_rate", "outward_accept_rate", "hard_proxy_accept_rate"):
        assert 0.0 <= info[key] <= 1.0


def test_proxy_unknown_bridge_governance_adds_loss_without_relaxing_vaccept():
    from cvsrffi.losses import proxy_unknown_energy_loss

    z = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.04],
            [0.98, -0.03],
            [0.00, 1.00],
            [0.04, 0.99],
            [-0.03, 0.98],
            [-1.00, 0.00],
            [-0.99, 0.04],
            [-0.98, -0.03],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    base_loss, base_info = proxy_unknown_energy_loss(
        z.clone().requires_grad_(True),
        y,
        holdout_label=2,
        virtual_count=9,
        virtual_mode="hard",
        energy_margin=0.0,
        placeholder_weight=0.0,
        core_quantile=0.80,
        accept_quantile=0.80,
        vaccept_weight=1.0,
        bridge_accept_weight=0.0,
        vaccept_cvar_alpha=0.30,
        energy_softplus_temperature=0.05,
    )
    bridge_loss, bridge_info = proxy_unknown_energy_loss(
        z.clone().requires_grad_(True),
        y,
        holdout_label=2,
        virtual_count=9,
        virtual_mode="hard",
        energy_margin=0.0,
        placeholder_weight=0.0,
        core_quantile=0.80,
        accept_quantile=0.80,
        vaccept_weight=1.0,
        bridge_accept_weight=2.0,
        bridge_accept_target=0.10,
        vaccept_cvar_alpha=0.30,
        energy_softplus_temperature=0.05,
    )

    assert bridge_info["vaccept_surrogate"] >= base_info["vaccept_surrogate"]
    assert bridge_info["bridge_governance_loss"] > 0.0
    assert bridge_loss.item() > base_loss.item()


def test_proxy_unknown_adg_exports_tail_density_energy_and_radius_metrics():
    from cvsrffi.losses import proxy_unknown_energy_loss

    z = torch.tensor(
        [
            [1.00, 0.00],
            [0.98, 0.15],
            [0.90, 0.44],
            [0.00, 1.00],
            [0.15, 0.98],
            [0.44, 0.90],
            [-1.00, 0.00],
            [-0.99, 0.04],
            [-0.98, -0.03],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    loss, info = proxy_unknown_energy_loss(
        z,
        y,
        holdout_label=2,
        virtual_count=9,
        virtual_mode="hard",
        energy_margin=0.0,
        placeholder_weight=0.0,
        core_quantile=0.50,
        accept_quantile=0.80,
        tail_quantile=0.67,
        overflow_quantile=0.67,
        vaccept_weight=1.0,
        tail_quarantine_weight=0.5,
        source_safe_weight=0.5,
        low_density_accept_weight=0.5,
        energy_margin_quantile_weight=0.5,
        radius_budget_weight=0.5,
        radius_inter_ratio_weight=0.5,
        bridge_accept_weight=0.5,
        shell_outward_accept_weight=0.5,
        radius_budget_rad=math.radians(8.0),
        radius_inter_ratio_target=0.10,
        energy_margin_q=0.10,
        energy_margin_target=0.05,
        energy_softplus_temperature=0.05,
    )

    assert loss.item() > 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    for key in (
        "low_density_accept_loss",
        "energy_margin_quantile_loss",
        "radius_budget_loss",
        "radius_inter_ratio_loss",
        "tail_accept_loss",
        "overflow_accept_loss",
        "shell_outward_accept_loss",
    ):
        assert info[key] >= 0.0
    assert math.isfinite(info["energy_margin_q05"])
    assert math.isfinite(info["energy_margin_q10"])
    assert info["component_radius_p95_deg"] >= 0.0
    assert info["component_radius_max_deg"] >= 0.0
    assert info["radius_inter_ratio"] >= 0.0
