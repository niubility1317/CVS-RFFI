from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import ResidualMetaAdapter  # noqa: E402
from cvsrffi.stage2_meta_adapter_adaptation import (  # noqa: E402
    MetaAdapterPhase2Config,
    adapt_meta_adapter_on_support,
    predict_with_frozen_meta_adapter,
)


class _ToyBundleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.meta_adapter_time = ResidualMetaAdapter(dim=2, rank=1)
        self.meta_adapter_freq = ResidualMetaAdapter(dim=2, rank=1)
        self.meta_adapter_fusion = ResidualMetaAdapter(dim=2, rank=1)
        self.fixed_backbone_parameter = nn.Parameter(torch.zeros(10_000))

    def forward(self, x, y=None, return_aux=False):
        del y, return_aux
        z = x.mean(dim=-1)
        z = self.meta_adapter_time(z)
        z = self.meta_adapter_freq(z)
        z = self.meta_adapter_fusion(z)
        return {"feat_cls": z}


def _write_row_inputs(tmp_path: Path) -> dict[str, Path]:
    support_path = tmp_path / "support.npz"
    query_path = tmp_path / "query.npz"
    prototype_path = tmp_path / "prototypes.npz"
    checkpoint_path = tmp_path / "meta_bundle.pth"
    checkpoint_path.write_bytes(b"fake-bundle-loaded-through-monkeypatch")
    np.savez(
        support_path,
        received_iq=np.asarray(
            [
                [[1.0, 0.8, 0.6], [0.1, 0.0, 0.2]],
                [[0.9, 0.7, 0.5], [0.0, 0.2, 0.1]],
                [[0.1, 0.0, 0.2], [1.0, 0.8, 0.6]],
                [[0.0, 0.2, 0.1], [0.9, 0.7, 0.5]],
            ],
            dtype=np.float32,
        ),
        support_labels=np.asarray([10, 10, 20, 20], dtype=np.int64),
    )
    np.savez(
        query_path,
        received_iq=np.asarray(
            [
                [[0.8, 0.7, 0.5], [0.1, 0.0, 0.2]],
                [[0.1, 0.2, 0.0], [0.8, 0.7, 0.5]],
            ],
            dtype=np.float32,
        ),
        query_ids=np.asarray(["query-fixed-002", "query-fixed-001"]),
    )
    np.savez(
        prototype_path,
        prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        class_ids=np.asarray([10, 20], dtype=np.int64),
    )
    return {
        "checkpoint_path": checkpoint_path,
        "support_path": support_path,
        "query_path": query_path,
        "prototype_path": prototype_path,
    }


def _row_config(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed-received-iq",
        "split_id": "split-support-query-disjoint",
        "checkpoint_path": str(paths["checkpoint_path"]),
        "support_path": str(paths["support_path"]),
        "query_path": str(paths["query_path"]),
        "prototype_path": str(paths["prototype_path"]),
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "operating_point": "K2/new2",
        "seed": 392002,
        "k_shot": 2,
        "steps": 3,
    }


def _audit() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_load_strict=True,
        trainable_fraction=0.002,
        prototypes={"10": torch.tensor([1.0, 0.0]), "20": torch.tensor([0.0, 1.0])},
        class_mapping={"0": "tx_a", "1": "tx_b"},
    )


def _install_fake_bundle_loader(monkeypatch: pytest.MonkeyPatch, sut, events: list[str]):
    model = _ToyBundleModel()

    def fake_loader(path, device):
        events.append("load_bundle")
        assert Path(path).exists()
        return model, _audit()

    monkeypatch.setattr(sut, "load_meta_bundle_strict", fake_loader)
    return model


def test_runner_adapts_before_query_is_opened_and_emits_two_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import cvsrffi.stage2_meta_adapter_runner as sut
    paths = _write_row_inputs(tmp_path)
    events: list[str] = []
    model = _install_fake_bundle_loader(monkeypatch, sut, events)
    real_load_npz = sut._load_npz

    def traced_load_npz(path, *args, **kwargs):
        if Path(path) == paths["support_path"]:
            events.append("open_support")
        if Path(path) == paths["query_path"]:
            assert "adapt" in events
            assert model.training is False
            assert all(not parameter.requires_grad for parameter in model.parameters())
            events.append("open_query")
        return real_load_npz(path, *args, **kwargs)

    monkeypatch.setattr(sut, "_load_npz", traced_load_npz)
    real_adapt = sut.adapt_meta_adapter_on_support

    def traced_adapt(*args, **kwargs):
        assert "open_support" in events
        assert "open_query" not in events
        handle = real_adapt(*args, **kwargs)
        events.append("adapt")
        return handle

    monkeypatch.setattr(sut, "adapt_meta_adapter_on_support", traced_adapt)

    receipt = sut.run_meta_adapter_stage2_row(_row_config(paths), tmp_path / "out", "cpu")

    assert events.index("open_support") < events.index("adapt") < events.index("open_query")
    assert receipt["query_opened_before_adaptation"] is False
    assert receipt["states"] == ["DA0_REG0", "DA1_REG0"]
    assert receipt["source_opened"] is False
    assert receipt["query_state_update_count"] == 0
    assert receipt["query_ids"] == ["query-fixed-002", "query-fixed-001"]
    assert Path(receipt["receipt_path"]).is_file()


def test_runner_refuses_existing_output_root(tmp_path: Path):
    import cvsrffi.stage2_meta_adapter_runner as sut
    paths = _write_row_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        sut.run_meta_adapter_stage2_row(_row_config(paths), output, "cpu")


@pytest.mark.parametrize("forbidden_key", ["source_path", "query_truth_path", "query_role"])
def test_runner_rejects_source_or_query_truth_role_config(
    tmp_path: Path, forbidden_key: str
):
    import cvsrffi.stage2_meta_adapter_runner as sut
    paths = _write_row_inputs(tmp_path)
    config = _row_config(paths)
    config[forbidden_key] = str(tmp_path / "forbidden")
    with pytest.raises(ValueError, match="allowlist"):
        sut.run_meta_adapter_stage2_row(config, tmp_path / "out", "cpu")
    assert not (tmp_path / "out").exists()


def test_prediction_artifacts_are_same_row_and_truth_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import cvsrffi.stage2_meta_adapter_runner as sut
    paths = _write_row_inputs(tmp_path)
    events: list[str] = []
    _install_fake_bundle_loader(monkeypatch, sut, events)
    receipt = sut.run_meta_adapter_stage2_row(_row_config(paths), tmp_path / "out", "cpu")

    assert receipt["query_state_update_count"] == 0
    da0 = tmp_path / "out" / "predictions_DA0_REG0.npz"
    da1 = tmp_path / "out" / "predictions_DA1_REG0.npz"
    with np.load(da0, allow_pickle=False) as first, np.load(da1, allow_pickle=False) as second:
        assert set(first.files) == {"query_ids", "predicted_class_ids", "scores"}
        assert set(second.files) == {"query_ids", "predicted_class_ids", "scores"}
        np.testing.assert_array_equal(first["query_ids"], second["query_ids"])
        np.testing.assert_array_equal(first["query_ids"], ["query-fixed-002", "query-fixed-001"])
        assert first["scores"].shape == second["scores"].shape == (2, 2)
        assert not {"truth", "query_truth", "query_role", "query_labels"} & set(first.files)
    persisted = json.loads((tmp_path / "out" / "receipt.json").read_text(encoding="utf-8"))
    assert persisted["states"] == ["DA0_REG0", "DA1_REG0"]
    assert persisted["query_role_opened"] is False


def test_no_query_smoke_has_no_query_path_and_three_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    smoke_path = PROJECT_ROOT / "code" / "scripts" / "smoke_stage2_meta_adapter_no_query.py"
    spec = importlib.util.spec_from_file_location(
        "task10_smoke_stage2_meta_adapter_no_query", smoke_path
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = smoke
    spec.loader.exec_module(smoke)
    paths = _write_row_inputs(tmp_path)
    events: list[str] = []
    import cvsrffi.stage2_meta_adapter_runner as runner
    _install_fake_bundle_loader(monkeypatch, runner, events)
    monkeypatch.setattr(smoke, "load_meta_bundle_strict", runner.load_meta_bundle_strict)
    result = smoke.run_meta_adapter_no_query_smoke(
        {
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "capsule-fixed-received-iq",
            "split_id": "split-support-query-disjoint",
            "checkpoint_path": str(paths["checkpoint_path"]),
            "support_path": str(paths["support_path"]),
            "prototype_path": str(paths["prototype_path"]),
            "receiver": "20-1",
            "scenario": "leo_clear_weak",
            "operating_point": "K2/new2",
            "seed": 392002,
            "k_shot": 2,
            "steps": 3,
        },
        tmp_path / "smoke",
        "cpu",
    )
    assert result["status"] == "REAL_META_CHECKPOINT_NO_QUERY_SMOKE_PASS"
    assert result["query_opened"] is False
    assert result["source_opened"] is False
    assert result["backward_count"] == 3
    assert result["checkpoint_load_strict"] is True
    assert result["query_state_update_count"] == 0
    assert not (tmp_path / "smoke" / "query.npz").exists()


def test_task9_public_interfaces_remain_fixed_for_runner():
    assert "steps" not in MetaAdapterPhase2Config.__dataclass_fields__
    assert "steps" not in __import__(
        "inspect"
    ).signature(adapt_meta_adapter_on_support).parameters
    assert "query_truth" not in __import__("inspect").signature(
        predict_with_frozen_meta_adapter
    ).parameters
