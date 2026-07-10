import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG import train_ssdg  # noqa: E402
from cvsrffi.phase2_prototypes import (  # noqa: E402
    PrototypeFusionConfig,
    attach_endpoint_accept_v1_manifest,
    calibrate_endpoint_accept_v1,
    fuse_tx_domain_prototypes,
    verify_endpoint_accept_v1_manifest,
)


def _minimal_phase2_package():
    return {
        "feature_key": "z_id",
        "prototypes": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "prototype_counts": torch.tensor([50, 40]),
        "tx_domain_prototypes": torch.tensor(
            [
                [[1.0, 0.0], [0.996, 0.087], [0.0, 1.0]],
                [[0.0, 1.0], [0.087, 0.996], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "tx_domain_counts": torch.tensor([[20, 20, 5], [20, 20, 5]]),
        "radii": {
            "p95": torch.tensor([math.radians(5.0), math.radians(5.0)]),
            "p99": torch.tensor([math.radians(12.0), math.radians(12.0)]),
            "r_3sigma": torch.tensor([math.radians(18.0), math.radians(18.0)]),
        },
        "metadata": {
            "source_checkpoint_sha256": "0" * 64,
            "run_id": "unit",
            "candidate_id": "fusion",
            "known_class_count": 2,
            "class_id_to_tx": ["tx0", "tx1"],
            "logit_class_order": [0, 1],
            "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
            "checkpoint_load_strict": True,
            "endpoint_runtime_entry_parity_digest": "1" * 64,
            "endpoint_runtime_entry_parity_sample_count": 8,
        },
    }


def test_fuse_tx_domain_prototypes_exports_v2_gate_schema():
    fused = fuse_tx_domain_prototypes(
        _minimal_phase2_package(),
        PrototypeFusionConfig(
            max_components_per_tx=2,
            merge_angle_deg=8.0,
            radius_cap_deg=25.0,
            accept_policy="local_component",
            global_ball_accept=False,
        ),
    )

    assert "fusion_config" in fused
    assert fused["fusion_config"]["enabled"] is True
    assert fused["fusion_config"]["accept_policy"] == "local_component"
    assert fused["fusion_config"]["global_ball_accept"] is False
    assert "fusion_components" in fused
    assert "fused_tx_prototypes" in fused
    for class_components in fused["fusion_components"]:
        assert class_components
        for comp in class_components:
            assert "component_id" in comp
            assert "source_domains" in comp
            assert "n_samples" in comp
            assert "r_core_deg" in comp
            assert "r_accept_deg" in comp
            assert "r_tail_deg" in comp
            assert "density_p05" in comp
            assert "density_p10" in comp
            assert "nll_p95" in comp
            assert "nll_tail_p95" in comp
            assert "nearest_other_deg" in comp
            assert comp["density_p05"] is not None
            assert comp["density_p10"] is not None
            assert comp["nll_p95"] is not None
            assert comp["nll_tail_p95"] is not None
            assert math.isfinite(float(comp["density_p05"]))
            assert math.isfinite(float(comp["density_p10"]))
            assert math.isfinite(float(comp["nll_p95"]))
            assert math.isfinite(float(comp["nll_tail_p95"]))
            if comp.get("tail_sentinel"):
                assert comp["accept_enabled"] is False
            else:
                assert comp["accept_enabled"] is True


def test_ssdg_default_export_path_is_phase1_source_named():
    export_path = train_ssdg._derive_phase2_export_path("runs/candidate/best_joint_safe_ssdg.pth")

    assert export_path.endswith("best_joint_safe_ssdg_phase1_source_prototypes.pt")
    assert "phase2_prototypes" not in export_path


def test_ssdg_phase2_export_executes_fusion_when_flag_enabled(monkeypatch, tmp_path):
    calls = {"fuse": 0, "save": 0}
    base_package = _minimal_phase2_package()

    class FakeModel:
        def state_dict(self):
            return {}

        def load_state_dict(self, state, strict=False):
            return None

    def fake_export(*args, **kwargs):
        package = dict(base_package)
        package["paths"] = {"pt_path": str(tmp_path / "phase2.pt"), "json_path": str(tmp_path / "phase2.json")}
        return package

    def fake_fuse(package, config):
        calls["fuse"] += 1
        fused = dict(package)
        fused["fusion_config"] = {"enabled": True, "global_ball_accept": bool(config.global_ball_accept)}
        fused["fusion_components"] = [[{"component_id": 0}]]
        fused["fused_tx_prototypes"] = torch.zeros(1, 1, 2)
        return fused

    def fake_save(package, output_path):
        calls["save"] += 1
        assert package.get("fusion_config", {}).get("enabled") is True
        return {"pt_path": str(output_path), "json_path": str(Path(output_path).with_suffix(".json"))}

    monkeypatch.setattr(train_ssdg, "export_phase2_prototypes", fake_export)
    monkeypatch.setattr(train_ssdg, "fuse_tx_domain_prototypes", fake_fuse)
    monkeypatch.setattr(train_ssdg, "save_phase2_prototype_export", fake_save)
    monkeypatch.setattr(train_ssdg, "PrototypeFusionConfig", PrototypeFusionConfig)

    args = SimpleNamespace(
        phase2_export_prototypes=True,
        phase2_export_path=str(tmp_path / "out.pt"),
        phase2_export_checkpoint="",
        phase2_export_split="train",
        phase2_export_feature_key="z_id",
        phase2_export_max_batches=0,
        phase2_fuse_prototypes=True,
        phase2_fuse_max_components=2,
        phase2_fuse_merge_angle_deg=8.0,
        phase2_fuse_radius_cap_deg=25.0,
        phase2_fuse_tail_abs_deg=30.0,
        phase2_fuse_accept_policy="local_component",
        phase2_fuse_accept_radius_key="p95",
        phase2_fuse_max_p95_increase_deg=2.0,
        phase2_fuse_keep_tail_sentinel=True,
        phase2_fuse_global_ball_accept=False,
        endpoint_require_artifact_on_export=False,
        dataset="",
        split_mode="",
    )

    out = train_ssdg._maybe_export_phase2_prototypes_ssdg(
        args,
        FakeModel(),
        {"train_loader": [object()], "val_loader": [object()], "split_info": {}},
        torch.device("cpu"),
        default_checkpoint="",
    )

    assert calls == {"fuse": 1, "save": 1}
    assert out["fusion_config"]["enabled"] is True


def test_endpoint_accept_v1_calibrates_thresholds_from_source_val_and_verifies():
    package = {
        "feature_key": "z_id",
        "fused_tx_prototypes": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        "fused_tx_mask": torch.ones(2, 1, dtype=torch.bool),
        "fusion_accept_policy": "local_component",
        "global_fused_radius_is_accept_region": False,
        "metadata": dict(_minimal_phase2_package()["metadata"]),
        "fusion_components": [
            [{"component_id": 0, "source_domains": [0], "accept_enabled": True}],
            [{"component_id": 0, "source_domains": [0], "accept_enabled": True}],
        ],
    }
    features = torch.tensor(
        [[1.0, 0.0], [0.999, 0.03], [0.995, -0.05], [0.998, 0.04],
         [0.0, 1.0], [0.03, 0.999], [-0.05, 0.995], [0.04, 0.998]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    logits = torch.tensor([[5.0, 0.0]] * 4 + [[0.0, 5.0]] * 4)

    calibrated = calibrate_endpoint_accept_v1(
        package,
        features,
        labels,
        logits,
        min_component_samples=2,
        min_class_samples=2,
    )
    artifact = attach_endpoint_accept_v1_manifest(calibrated)
    manifest = verify_endpoint_accept_v1_manifest(artifact)

    assert manifest["threshold_source"] == "source_val_only"
    assert manifest["calibration_evidence"]["num_samples"] == 8
    assert manifest["gate_thresholds"]["energy_max_by_class"].keys() == {"0", "1"}
    assert all(row[0]["calibration_status"] == "source_val_calibrated" for row in artifact["fusion_components"])


def test_ssdg_export_hook_attaches_and_verifies_endpoint_artifact(monkeypatch, tmp_path):
    calls = {"save": 0}

    class FakeModel:
        def state_dict(self):
            return {}

        def load_state_dict(self, state, strict=False):
            return None

    fused = {
        "feature_key": "z_id",
        "fused_tx_prototypes": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        "fused_tx_mask": torch.ones(2, 1, dtype=torch.bool),
        "fusion_accept_policy": "local_component",
        "global_fused_radius_is_accept_region": False,
        "metadata": dict(_minimal_phase2_package()["metadata"]),
        "fusion_components": [
            [{"component_id": 0, "source_domains": [0], "accept_enabled": True}],
            [{"component_id": 0, "source_domains": [0], "accept_enabled": True}],
        ],
    }
    features = torch.tensor(
        [[1.0, 0.0], [0.999, 0.03], [0.995, -0.05], [0.998, 0.04],
         [0.0, 1.0], [0.03, 0.999], [-0.05, 0.995], [0.04, 0.998]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    logits = torch.tensor([[5.0, 0.0]] * 4 + [[0.0, 5.0]] * 4)

    monkeypatch.setattr(train_ssdg, "export_phase2_prototypes", lambda *a, **k: _minimal_phase2_package())
    monkeypatch.setattr(train_ssdg, "fuse_tx_domain_prototypes", lambda *a, **k: dict(fused))
    monkeypatch.setattr(
        train_ssdg,
        "extract_endpoint_calibration_features",
        lambda *a, **k: {"features": features, "labels": labels, "logits": logits},
    )

    def fake_save(package, output_path):
        calls["save"] += 1
        verify_endpoint_accept_v1_manifest(package)
        return {"pt_path": str(output_path), "json_path": str(Path(output_path).with_suffix(".json"))}

    monkeypatch.setattr(train_ssdg, "save_phase2_prototype_export", fake_save)
    args = SimpleNamespace(
        phase2_export_prototypes=True,
        phase2_export_path=str(tmp_path / "out.pt"),
        phase2_export_checkpoint="",
        phase2_export_split="train",
        phase2_export_feature_key="z_id",
        phase2_export_max_batches=0,
        phase2_fuse_prototypes=True,
        phase2_fuse_max_components=2,
        phase2_fuse_merge_angle_deg=8.0,
        phase2_fuse_radius_cap_deg=25.0,
        phase2_fuse_tail_abs_deg=30.0,
        phase2_fuse_accept_policy="local_component",
        phase2_fuse_accept_radius_key="p95",
        phase2_fuse_max_p95_increase_deg=2.0,
        phase2_fuse_keep_tail_sentinel=True,
        phase2_fuse_global_ball_accept=False,
        endpoint_require_artifact_on_export=True,
        endpoint_calibration_min_component_samples=2,
        endpoint_calibration_min_class_samples=2,
        endpoint_calibration_core_quantile=0.80,
        endpoint_calibration_accept_quantile=0.95,
        endpoint_calibration_tail_quantile=0.99,
        endpoint_threshold_source="source_val_only",
        endpoint_calibration_split="source_val",
        endpoint_accept_policy_id="endpoint_accept_v1",
        dataset="wisig",
        split_mode="tx_rx_day_1_7_2",
    )

    package = train_ssdg._maybe_export_phase2_prototypes_ssdg(
        args,
        FakeModel(),
        {"train_loader": [object()], "val_loader": [object()], "split_info": {}},
        torch.device("cpu"),
        default_checkpoint="",
    )

    assert calls["save"] == 1
    assert package["endpoint_accept_v1"]["fail_closed"] is True
