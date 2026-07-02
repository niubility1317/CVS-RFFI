import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.phase2_prototypes import (  # noqa: E402
    BalancedPrototypeBank,
    PrototypeFusionConfig,
    build_phase2_prototype_export,
    export_phase2_prototypes,
    extract_phase2_features,
    fuse_tx_domain_prototypes,
    PrototypeRadiusTracker,
    save_phase2_prototype_export,
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


def test_radius_tracker_reports_robust_three_sigma_tail_without_max_inflation():
    tracker = PrototypeRadiusTracker(num_classes=1)
    proto = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    features = torch.tensor(
        [[1.0, 0.0], [0.996, 0.087], [0.985, 0.174], [0.0, 1.0]],
        dtype=torch.float32,
    )
    tracker.update(features, torch.zeros(4, dtype=torch.long), proto)

    stats = tracker.robust_stats(0)

    assert stats["max"] > math.radians(80.0)
    assert stats["robust_sigma"] < math.radians(8.0)
    assert stats["r_3sigma"] < stats["max"]
    assert stats["tail_count_gt_3sigma"] >= 1


def test_proto_loss_returns_graph_safe_zero_when_no_initialized_class():
    bank = BalancedPrototypeBank(num_items=2, feat_dim=2)
    z = torch.randn(3, 2, requires_grad=True)
    loss, metrics = bank.prototype_pull_margin_loss(z, torch.tensor([0, 1, 1]))
    loss.backward()

    assert float(loss.detach().item()) == 0.0
    assert z.grad is not None
    assert metrics["active"] == 0.0


class _AuxRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, y_tx=None, grl_lambda=1.0, return_aux=False, domain_labels=None):
        self.calls.append(
            {
                "return_aux": bool(return_aux),
                "y_tx": None if y_tx is None else y_tx.detach().cpu().clone(),
                "domain_labels": None if domain_labels is None else domain_labels.detach().cpu().clone(),
                "grl_lambda": float(grl_lambda),
            }
        )
        assert return_aux is True
        z = torch.stack([x[:, 0] + 1.0, x[:, 0] * 0.0 + 0.5], dim=1)
        return {"z_id": z, "z_dom": z + 1.0, "tx_logits": z, "dom_logits": z, "adv_dom_logits": z}


def test_extract_phase2_features_uses_return_aux_and_existing_batch_domain_flow():
    model = _AuxRecordingModel()
    loader = [
        (
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
        )
    ]

    payload = extract_phase2_features(model, loader, device=torch.device("cpu"), feature_key="z_id")

    assert model.calls[0]["return_aux"] is True
    assert torch.equal(model.calls[0]["domain_labels"], torch.tensor([2, 3]))
    assert payload["feature_key"] == "z_id"
    assert torch.allclose(payload["features"], torch.tensor([[1.0, 0.5], [2.0, 0.5]]))
    assert torch.equal(payload["labels"], torch.tensor([0, 1]))
    assert torch.equal(payload["domains"], torch.tensor([2, 3]))


def test_build_phase2_export_contains_tx_domain_bounds_and_geometry():
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 1, 0, 1])

    package = build_phase2_prototype_export(
        features,
        labels,
        domains,
        feature_key="z_id",
        metadata={"checkpoint_path": "best_primary_ood_model.pth"},
    )

    assert package["schema_version"] == 1
    assert package["feature_key"] == "z_id"
    assert package["prototypes"].shape == (2, 2)
    assert package["tx_domain_prototypes"].shape == (2, 2, 2)
    assert set(package["radii"].keys()) == {"p50", "p80", "p90", "p95", "p99", "max", "robust_max", "r_1sigma", "r_2sigma", "r_3sigma"}
    assert "radius_robust_sigma" in package
    assert "radius_tail_stats" in package
    assert "geometry" in package
    assert package["metadata"]["checkpoint_path"] == "best_primary_ood_model.pth"


def test_fuse_tx_domain_prototypes_reduces_redundant_domains_and_preserves_tail_audit():
    package = {
        "feature_key": "z_id",
        "prototypes": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "prototype_counts": torch.tensor([40, 40]),
        "tx_domain_prototypes": torch.tensor(
            [
                [[1.0, 0.0], [0.996, 0.087], [0.0, 1.0]],
                [[0.0, 1.0], [0.087, 0.996], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "tx_domain_counts": torch.tensor([[10, 12, 2], [10, 11, 2]]),
        "radii": {
            "p95": torch.tensor([math.radians(5.0), math.radians(5.0)]),
            "p99": torch.tensor([math.radians(8.0), math.radians(8.0)]),
            "max": torch.tensor([math.radians(90.0), math.radians(90.0)]),
            "r_3sigma": torch.tensor([math.radians(12.0), math.radians(12.0)]),
        },
        "radius_robust_sigma": torch.tensor([math.radians(2.0), math.radians(2.0)]),
        "metadata": {},
    }

    fused = fuse_tx_domain_prototypes(
        package,
        PrototypeFusionConfig(max_components_per_tx=2, merge_angle_deg=8.0, tail_abs_deg=30.0),
    )

    assert fused["fused_tx_prototypes"].shape == (2, 2, 2)
    assert fused["fused_tx_mask"].sum(dim=1).tolist() == [2, 2]
    assert fused["fusion_components"][0][0]["domains"] == [0, 1]
    assert fused["fusion_components"][0][1]["tail_sentinel"] is True
    assert fused["fusion_metadata"]["default_training_behavior_changed"] is False


def test_fuse_tx_domain_prototypes_can_exclude_tail_sentinel_from_accept_components():
    package = {
        "feature_key": "z_id",
        "prototypes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "prototype_counts": torch.tensor([40]),
        "tx_domain_prototypes": torch.tensor(
            [[[1.0, 0.0], [0.996, 0.087], [0.0, 1.0]]],
            dtype=torch.float32,
        ),
        "tx_domain_counts": torch.tensor([[10, 12, 2]]),
        "radii": {
            "p50": torch.tensor([math.radians(3.0)]),
            "p80": torch.tensor([math.radians(4.0)]),
            "p95": torch.tensor([math.radians(5.0)]),
            "p99": torch.tensor([math.radians(8.0)]),
            "r_3sigma": torch.tensor([math.radians(12.0)]),
        },
        "metadata": {},
    }

    fused = fuse_tx_domain_prototypes(
        package,
        PrototypeFusionConfig(
            max_components_per_tx=1,
            merge_angle_deg=8.0,
            tail_abs_deg=30.0,
            accept_radius_key="p80",
            keep_tail_sentinel=False,
            tail_auto_accept=False,
        ),
    )

    comp = fused["fusion_components"][0][0]
    assert comp["tail_sentinel"] is False
    assert comp["accept_enabled"] is True
    assert fused["fusion_config"]["tail_auto_accept"] is False
    assert fused["fusion_config"]["keep_tail_sentinel"] is False


def test_tail_auto_accept_request_does_not_enable_tail_sentinel_acceptance():
    package = {
        "feature_key": "z_id",
        "prototypes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "prototype_counts": torch.tensor([40]),
        "tx_domain_prototypes": torch.tensor(
            [[[1.0, 0.0], [0.996, 0.087], [0.0, 1.0]]],
            dtype=torch.float32,
        ),
        "tx_domain_counts": torch.tensor([[10, 12, 2]]),
        "radii": {
            "p80": torch.tensor([math.radians(4.0)]),
            "p95": torch.tensor([math.radians(5.0)]),
            "p99": torch.tensor([math.radians(8.0)]),
            "r_3sigma": torch.tensor([math.radians(12.0)]),
        },
        "metadata": {},
    }

    fused = fuse_tx_domain_prototypes(
        package,
        PrototypeFusionConfig(
            max_components_per_tx=2,
            merge_angle_deg=8.0,
            tail_abs_deg=30.0,
            accept_radius_key="p80",
            keep_tail_sentinel=True,
            tail_auto_accept=True,
        ),
    )

    tail_components = [comp for comp in fused["fusion_components"][0] if comp["tail_sentinel"]]
    assert tail_components
    assert all(comp["accept_enabled"] is False for comp in tail_components)
    assert fused["fusion_config"]["tail_auto_accept_requested"] is True
    assert fused["fusion_config"]["tail_auto_accept_effective"] is False


def test_export_phase2_prototypes_saves_pt_and_json_sidecar(tmp_path):
    model = _AuxRecordingModel()
    loader = [
        (
            torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
            torch.tensor([0, 1, 1]),
            torch.tensor([0, 0, 1]),
        )
    ]
    out_path = tmp_path / "phase2_proto.pt"

    package = export_phase2_prototypes(
        model,
        loader,
        output_path=out_path,
        device=torch.device("cpu"),
        metadata={"split": "synthetic"},
    )
    save_phase2_prototype_export(package, out_path)

    sidecar = tmp_path / "phase2_proto.json"
    assert out_path.is_file()
    assert sidecar.is_file()
    loaded = torch.load(out_path, map_location="cpu")
    assert loaded["metadata"]["split"] == "synthetic"
