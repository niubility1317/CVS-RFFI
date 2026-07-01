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
from cvsrffi.phase2_prototypes import PrototypeFusionConfig, fuse_tx_domain_prototypes  # noqa: E402


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
        "metadata": {},
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
            assert "nll_p95" in comp
            assert "nearest_other_deg" in comp
            assert comp["accept_enabled"] is True


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

