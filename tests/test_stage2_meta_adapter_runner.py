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
from cvsrffi.stage2_meta_adapter_scorer import score_meta_adapter_pair  # noqa: E402


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
        support_physical_ids=np.asarray(
            [
                "physical-support-000",
                "physical-support-001",
                "physical-support-002",
                "physical-support-003",
            ]
        ),
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
        "candidate_id": "CVS_META_ADAPTER_TRI_R4_V1",
        "bundle_id": "ADV3B02_CORE90_SOFT_E200_META_TRI_R4_V1",
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


def test_runner_preserves_strict_bundle_class_floor_objective():
    import cvsrffi.stage2_meta_adapter_runner as sut

    audit = _audit()
    audit.adaptation_objective = "frozen_prototype_class_floor_ce_v1"
    audit.support_logit_scale = 8.0
    resolved = sut._require_strict_audit(audit)

    assert resolved["adaptation_objective"] == "frozen_prototype_class_floor_ce_v1"
    assert resolved["support_logit_scale"] == 8.0


def _rewrite_support_ids(paths: dict[str, Path], ids: list[str] | None) -> None:
    with np.load(paths["support_path"], allow_pickle=False) as archive:
        received_iq = np.asarray(archive["received_iq"]).copy()
        support_labels = np.asarray(archive["support_labels"]).copy()
    payload = {
        "received_iq": received_iq,
        "support_labels": support_labels,
    }
    if ids is not None:
        payload["support_physical_ids"] = np.asarray(ids)
    np.savez(paths["support_path"], **payload)


def test_numpy2_torch21_bridge_covers_inputs_and_prediction_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cvsrffi.stage2_meta_adapter_runner as sut

    def incompatible_from_numpy(_value):
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    def incompatible_tensor_numpy(_value):
        raise TypeError("Numpy is not available")

    monkeypatch.setattr(torch, "from_numpy", incompatible_from_numpy)
    monkeypatch.setattr(torch.Tensor, "numpy", incompatible_tensor_numpy)

    iq = sut._received_iq_tensor(
        np.ones((2, 2, 4), dtype=np.float32), label="support"
    )
    ids = sut._integer_tensor(np.asarray([10, 20], dtype=np.int64), label="ids")
    prototypes, prototype_ids = sut._prototype_tensors(
        {
            "prototypes": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "class_ids": np.asarray([10, 20], dtype=np.int64),
        }
    )
    assert tuple(iq.shape) == (2, 2, 4)
    assert ids.tolist() == [10, 20]
    assert prototypes.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert prototype_ids.tolist() == [10, 20]

    output = tmp_path / "predictions.npz"
    sut._write_prediction(
        output,
        query_ids=np.asarray(["query-0", "query-1"]),
        predicted_class_ids=torch.tensor([10, 20], dtype=torch.long),
        scores=torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32),
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["predicted_class_ids"].tolist() == [10, 20]
        np.testing.assert_allclose(
            archive["scores"],
            np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        )


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
        if Path(path) == paths["prototype_path"]:
            events.append("open_prototype")
        if Path(path) == paths["query_path"]:
            assert "freeze_da1" in events
            assert model.training is False
            assert all(not parameter.requires_grad for parameter in model.parameters())
            events.append("open_query")
        return real_load_npz(path, *args, **kwargs)

    monkeypatch.setattr(sut, "_load_npz", traced_load_npz)
    real_snapshot = getattr(sut, "_snapshot_frozen_model", None)

    def traced_snapshot(snapshot_model):
        events.append("freeze_da0")
        if real_snapshot is not None:
            return real_snapshot(snapshot_model)
        return snapshot_model

    monkeypatch.setattr(sut, "_snapshot_frozen_model", traced_snapshot, raising=False)
    real_adapt = sut.adapt_meta_adapter_on_support

    def traced_adapt(*args, **kwargs):
        assert "open_support" in events
        assert "open_prototype" in events
        assert "open_query" not in events
        support_batch = args[1]
        assert support_batch.support_physical_ids == (
            "physical-support-000",
            "physical-support-001",
            "physical-support-002",
            "physical-support-003",
        )
        handle = real_adapt(*args, **kwargs)
        events.append("adapt")
        assert handle.model.training is False
        assert all(not parameter.requires_grad for parameter in handle.model.parameters())
        events.append("freeze_da1")
        return handle

    monkeypatch.setattr(sut, "adapt_meta_adapter_on_support", traced_adapt)

    receipt = sut.run_meta_adapter_stage2_row(_row_config(paths), tmp_path / "out", "cpu")

    assert events.index("load_bundle") < events.index("freeze_da0")
    assert events.index("freeze_da0") < events.index("open_support")
    assert events.index("open_support") < events.index("open_prototype")
    assert events.index("open_prototype") < events.index("adapt")
    assert events.index("adapt") < events.index("freeze_da1") < events.index("open_query")
    assert receipt["query_opened_before_adaptation"] is False
    assert receipt["states"] == ["DA0_REG0", "DA1_REG0"]
    assert receipt["source_opened"] is False
    assert receipt["query_state_update_count"] == 0
    assert receipt["query_ids"] == ["query-fixed-002", "query-fixed-001"]
    assert receipt["candidate_id"] == "CVS_META_ADAPTER_TRI_R4_V1"
    assert receipt["bundle_id"] == "ADV3B02_CORE90_SOFT_E200_META_TRI_R4_V1"
    assert receipt["registered_class_ids"] == [10, 20]
    assert Path(receipt["receipt_path"]).is_file()


@pytest.mark.parametrize("missing_key", ["candidate_id", "bundle_id"])
def test_runner_requires_candidate_and_bundle_ids(tmp_path: Path, missing_key: str):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    config = _row_config(paths)
    config.pop(missing_key)
    with pytest.raises(ValueError, match="allowlist"):
        sut.run_meta_adapter_stage2_row(config, tmp_path / "out", "cpu")


@pytest.mark.parametrize(
    ("key", "invalid"),
    [("candidate_id", ""), ("candidate_id", "   "), ("candidate_id", None),
     ("bundle_id", ""), ("bundle_id", "   "), ("bundle_id", None)],
)
def test_runner_rejects_invalid_candidate_or_bundle_id(
    tmp_path: Path, key: str, invalid: object
):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    config = _row_config(paths)
    config[key] = invalid
    with pytest.raises(ValueError, match=key):
        sut.run_meta_adapter_stage2_row(config, tmp_path / "out", "cpu")


def test_runner_refuses_existing_output_root(tmp_path: Path):
    import cvsrffi.stage2_meta_adapter_runner as sut
    paths = _write_row_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        sut.run_meta_adapter_stage2_row(_row_config(paths), output, "cpu")


def test_runner_rejects_external_prototypes_that_differ_from_strict_bundle_before_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    with np.load(paths["prototype_path"], allow_pickle=False) as archive:
        class_ids = np.asarray(archive["class_ids"]).copy()
        prototypes = np.asarray(archive["prototypes"]).copy()
    prototypes[0, 0] += 0.25
    np.savez(paths["prototype_path"], prototypes=prototypes, class_ids=class_ids)
    events: list[str] = []
    _install_fake_bundle_loader(monkeypatch, sut, events)
    real_load = sut._load_npz

    def traced(path, *args, **kwargs):
        if Path(path) == paths["query_path"]:
            events.append("open_query")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(sut, "_load_npz", traced)
    with pytest.raises(ValueError, match="bundle.*prototype|prototype.*bundle"):
        sut.run_meta_adapter_stage2_row(_row_config(paths), tmp_path / "out-mismatch", "cpu")
    assert "open_query" not in events


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


@pytest.mark.parametrize(
    ("label", "ids"),
    [
        ("missing", None),
        ("duplicate", ["physical-support-000"] * 4),
        (
            "length",
            ["physical-support-000", "physical-support-001", "physical-support-002"],
        ),
    ],
)
def test_runner_requires_real_support_physical_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    ids: list[str] | None,
):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    _rewrite_support_ids(paths, ids)
    events: list[str] = []
    _install_fake_bundle_loader(monkeypatch, sut, events)
    with pytest.raises(ValueError, match="physical_ids|allowlist"):
        sut.run_meta_adapter_stage2_row(_row_config(paths), tmp_path / f"out-{label}", "cpu")


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


def test_runner_receipt_and_predictions_close_before_truth_last_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    events: list[str] = []
    _install_fake_bundle_loader(monkeypatch, sut, events)
    output = tmp_path / "out"
    receipt = sut.run_meta_adapter_stage2_row(_row_config(paths), output, "cpu")

    receipt_path = output / "receipt.json"
    da0_path = output / "predictions_DA0_REG0.npz"
    da1_path = output / "predictions_DA1_REG0.npz"
    assert receipt_path.is_file() and da0_path.is_file() and da1_path.is_file()
    assert receipt["registered_class_ids"] == [10, 20]

    # Truth is deliberately materialized only after both prediction artifacts
    # and their binding receipt are complete.
    truth_path = tmp_path / "truth.npz"
    np.savez(
        truth_path,
        query_ids=np.asarray(receipt["query_ids"]),
        true_class_ids=np.asarray([20, 10], dtype=np.int64),
    )
    score = score_meta_adapter_pair(
        da0_path, da1_path, truth_path, receipt_path=receipt_path
    )
    assert score.candidate_id == receipt["candidate_id"]
    assert score.bundle_id == receipt["bundle_id"]
    assert score.registered_class_ids == (10, 20)


@pytest.mark.parametrize("target_name", ["predictions_DA0_REG0.npz", "predictions_DA1_REG0.npz"])
def test_runner_writes_failed_receipt_when_prediction_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_name: str
):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    events: list[str] = []
    _install_fake_bundle_loader(monkeypatch, sut, events)
    real_write = sut._write_prediction

    def fail_target(path, **kwargs):
        if Path(path).name == target_name:
            raise OSError(f"injected {target_name} write failure")
        return real_write(path, **kwargs)

    monkeypatch.setattr(sut, "_write_prediction", fail_target)
    output = tmp_path / "write-failure"
    with pytest.raises(OSError, match="injected"):
        sut.run_meta_adapter_stage2_row(_row_config(paths), output, "cpu")
    failure_path = output / "failure_receipt.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["error_type"] == "OSError"
    serialized = json.dumps(failure, ensure_ascii=False).lower()
    assert "truth" not in serialized
    assert "role" not in serialized
    assert not (output / "receipt.json").exists()


def test_runner_writes_failed_receipt_when_final_receipt_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import cvsrffi.stage2_meta_adapter_runner as sut

    paths = _write_row_inputs(tmp_path)
    events: list[str] = []
    _install_fake_bundle_loader(monkeypatch, sut, events)

    def fail_receipt(*args, **kwargs):
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(sut, "_write_receipt", fail_receipt, raising=False)
    output = tmp_path / "receipt-failure"
    with pytest.raises(OSError, match="injected receipt"):
        sut.run_meta_adapter_stage2_row(_row_config(paths), output, "cpu")
    failure = json.loads(
        (output / "failure_receipt.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "FAILED"
    assert failure["error_type"] == "OSError"
    assert "DA0_REG0" in failure["completed_stages"]
    assert "DA1_REG0" in failure["completed_stages"]
    assert not (output / "receipt.json").exists()


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
            "candidate_id": "CVS_META_ADAPTER_TRI_R4_V1",
            "bundle_id": "ADV3B02_CORE90_SOFT_E200_META_TRI_R4_V1",
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


def test_no_query_smoke_base_init_reports_distinct_status_without_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    smoke_path = PROJECT_ROOT / "code" / "scripts" / "smoke_stage2_meta_adapter_no_query.py"
    spec = importlib.util.spec_from_file_location(
        "task12_smoke_stage2_meta_adapter_base_init", smoke_path
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
            "candidate_id": "CVS_META_ADAPTER_TRI_R4_V1",
            "bundle_id": "ADV3B02_CORE90_SOFT_E200_META_TRI_R4_V1",
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
        tmp_path / "base-init-smoke",
        "cpu",
        smoke_kind="base_init",
    )
    assert result["status"] == "REAL_BASE_CHECKPOINT_ADAPTER_INIT_NO_QUERY_SMOKE_PASS"
    assert result["query_opened"] is False
    assert result["query_state_update_count"] == 0
    assert not (tmp_path / "base-init-smoke" / "query.npz").exists()


def test_no_query_smoke_rejects_unknown_smoke_kind_before_io(tmp_path: Path):
    smoke_path = PROJECT_ROOT / "code" / "scripts" / "smoke_stage2_meta_adapter_no_query.py"
    spec = importlib.util.spec_from_file_location(
        "task12_smoke_stage2_meta_adapter_invalid_kind", smoke_path
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = smoke
    spec.loader.exec_module(smoke)
    with pytest.raises(ValueError, match="smoke_kind"):
        smoke.run_meta_adapter_no_query_smoke(
            {}, tmp_path / "invalid-smoke", "cpu", smoke_kind="unexpected"
        )
    assert not (tmp_path / "invalid-smoke").exists()


def test_task9_public_interfaces_remain_fixed_for_runner():
    assert "steps" not in MetaAdapterPhase2Config.__dataclass_fields__
    assert "steps" not in __import__(
        "inspect"
    ).signature(adapt_meta_adapter_on_support).parameters
    assert "query_truth" not in __import__("inspect").signature(
        predict_with_frozen_meta_adapter
    ).parameters
