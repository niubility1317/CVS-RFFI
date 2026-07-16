from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_predictor_entry as entry
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_runtime_request import (
    SOMPH_APPLY_REQUEST_SCHEMA,
    SOMPH_ENROLLMENT_REQUEST_SCHEMA,
)


def _request_file(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _apply_request() -> dict:
    return {
        "schema": SOMPH_APPLY_REQUEST_SCHEMA,
        "package_seal_sha256": "1" * 64,
        "head_capsule_sha256": "2" * 64,
        "head_enrollment_binding_sha256": "3" * 64,
        "row_handle": "row_" + "4" * 64,
        "row_manifest_sha256": "5" * 64,
        "prediction_output_leaf": "prediction.cvspred",
        "device": "cpu",
        **PHASE2_FULL_CONTRACT,
    }


def _enrollment_request() -> dict:
    return {
        "schema": SOMPH_ENROLLMENT_REQUEST_SCHEMA,
        "package_seal_sha256": "1" * 64,
        "head_output_leaf": "head_capsule.npz",
        "device": "cpu",
        "support_batch_size": 8,
        **PHASE2_FULL_CONTRACT,
    }


def _manifest(*, profile: str) -> dict:
    apply = profile == entry.APPLY_ONLY
    return {
        "profile": profile,
        "stage": "stage2c",
        "registration_state": "after",
        "receiver": "8-8",
        "seed": 713106,
        "k_shot": 5,
        "registered_classes": [
            {"class_index": index, "class_handle": f"cls_{index + 1:064x}"}
            for index in range(3)
        ],
        "package_root_sha256": "6" * 64,
        "phase1_checkpoint_sha256": "7" * 64,
        "feature_runtime_sha256": "a" * 64,
        "method_lock_sha256": "8" * 64,
        "overlay_provenance_sha256": "9" * 64,
        "head_capsule_sha256": "2" * 64 if apply else None,
        "head_enrollment_binding_sha256": "3" * 64 if apply else None,
        "row_handle": "row_" + "4" * 64 if apply else None,
        "row_manifest_sha256": "5" * 64 if apply else None,
        "members": [],
    }


def test_cuda_memory_audit_initializes_context_before_peak_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    device = entry.torch.device("cuda:0")
    monkeypatch.setattr(
        entry.torch.cuda,
        "set_device",
        lambda value: calls.append(("set_device", value)),
    )
    monkeypatch.setattr(
        entry.torch,
        "empty",
        lambda *args, **kwargs: calls.append(
            ("empty", kwargs.get("device"))
        ),
    )
    monkeypatch.setattr(
        entry.torch.cuda,
        "reset_peak_memory_stats",
        lambda value: calls.append(("reset", value)),
    )

    entry._prepare_cuda_memory_audit(device)

    assert calls == [
        ("set_device", device),
        ("empty", device),
        ("reset", device),
    ]


def test_invalid_request_is_rejected_before_package_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _apply_request()
    request["query_count"] = 20
    request_path = _request_file(tmp_path / "request.json", request)
    opened = []
    monkeypatch.setattr(
        entry,
        "load_verified_somph_predictor_bundle",
        lambda *args, **kwargs: opened.append(True),
    )
    with pytest.raises(ValueError):
        entry.run_somph_apply(
            request_json=request_path,
            package_root=tmp_path,
            detached_seal_path=tmp_path / "seal.json",
            expected_seal_sha256="1" * 64,
            output_root=tmp_path,
        )
    assert opened == []


def test_apply_uses_fixed_loaded_runtime_singleton_and_artifact_v2_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _request_file(tmp_path / "request.json", _apply_request())
    manifest = _manifest(profile=entry.APPLY_ONLY)
    model = object()
    method_lock = {"locked": True}
    calls: dict = {}
    monkeypatch.setattr(
        entry,
        "preflight_somph_predictor_bundle",
        lambda *args, **kwargs: (manifest, {}, {}),
    )
    monkeypatch.setattr(
        entry,
        "load_verified_somph_predictor_bundle",
        lambda *args, **kwargs: ({"scenario": {}}, manifest, {"status": "structural"}),
    )
    monkeypatch.setattr(
        entry,
        "load_verified_somph_head_capsule",
        lambda *args, **kwargs: ({"head": np.asarray([1])}, {}, "3" * 64),
    )
    monkeypatch.setattr(
        entry,
        "_load_fixed_runtime",
        lambda *args, **kwargs: (model, method_lock),
    )

    def fake_apply(loaded_model, payloads, capsule, **kwargs):
        calls["model"] = loaded_model
        calls["batch_size"] = kwargs["batch_size"]
        return {
            "query_tokens": np.asarray(["qid_" + "a" * 64]),
            "scenarios": np.asarray(["leo_clear_weak"]),
            "predicted_class_indices": np.asarray([1], dtype=np.int64),
            "backbone_forward_counts": np.asarray([1], dtype=np.uint8),
        }, {"mean_backbone_forward_count": 1.0}

    monkeypatch.setattr(entry, "apply_somph_heads", fake_apply)

    def fake_publish(path, **kwargs):
        calls["publication"] = kwargs
        return {"artifact_sha256": "a" * 64, "seal_sha256": "b" * 64}

    monkeypatch.setattr(entry, "publish_somph_prediction_artifact", fake_publish)
    monkeypatch.setattr(entry, "_write_readonly_json_new", lambda *args: "c" * 64)
    result = entry.run_somph_apply(
        request_json=request_path,
        package_root=tmp_path,
        detached_seal_path=tmp_path / "seal.json",
        expected_seal_sha256="1" * 64,
        output_root=tmp_path,
    )
    assert calls["model"] is model
    assert calls["batch_size"] == 1
    assert calls["publication"]["registered_class_count"] == 3
    assert calls["publication"]["row_id"] == manifest["row_handle"]
    assert (
        calls["publication"]["feature_runtime_sha256"]
        == manifest["feature_runtime_sha256"]
    )
    assert (
        calls["publication"]["row_manifest_sha256"]
        == manifest["row_manifest_sha256"]
    )
    assert "new_class_count" not in calls["publication"]
    assert "resource_audit_sha256" not in calls["publication"]
    assert result["artifact_sha256"] == "a" * 64
    assert result["formal_launch_authority"] is False


def test_apply_head_trust_root_fails_before_runtime_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _request_file(tmp_path / "request.json", _apply_request())
    manifest = _manifest(profile=entry.APPLY_ONLY)
    manifest["head_capsule_sha256"] = "f" * 64
    loaded = []
    monkeypatch.setattr(
        entry,
        "preflight_somph_predictor_bundle",
        lambda *args, **kwargs: (manifest, {}, {}),
    )
    monkeypatch.setattr(
        entry,
        "load_verified_somph_predictor_bundle",
        lambda *args, **kwargs: ({}, manifest, {}),
    )
    monkeypatch.setattr(
        entry,
        "_load_fixed_runtime",
        lambda *args, **kwargs: loaded.append(True),
    )
    with pytest.raises(entry.SomphPredictorEntryError, match="head capsule"):
        entry.run_somph_apply(
            request_json=request_path,
            package_root=tmp_path,
            detached_seal_path=tmp_path / "seal.json",
            expected_seal_sha256="1" * 64,
            output_root=tmp_path,
        )
    assert loaded == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("row_handle", "row_" + "e" * 64, "row handle"),
        ("row_manifest_sha256", "e" * 64, "row SHA256"),
    ],
)
def test_apply_row_binding_mismatch_fails_before_query_materialization_and_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    request = _apply_request()
    request[field] = value
    request_path = _request_file(tmp_path / "request.json", request)
    manifest = _manifest(profile=entry.APPLY_ONLY)
    materialized = []
    forwarded = []
    monkeypatch.setattr(
        entry,
        "preflight_somph_predictor_bundle",
        lambda *args, **kwargs: (manifest, {}, {}),
    )
    monkeypatch.setattr(
        entry,
        "load_verified_somph_predictor_bundle",
        lambda *args, **kwargs: materialized.append(True),
    )
    monkeypatch.setattr(
        entry,
        "_load_fixed_runtime",
        lambda *args, **kwargs: forwarded.append(True),
    )
    with pytest.raises(entry.SomphPredictorEntryError, match=message):
        entry.run_somph_apply(
            request_json=request_path,
            package_root=tmp_path,
            detached_seal_path=tmp_path / "seal.json",
            expected_seal_sha256="1" * 64,
            output_root=tmp_path,
        )
    assert materialized == []
    assert forwarded == []


def test_enrollment_uses_fixed_runtime_and_support_only_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _request_file(tmp_path / "request.json", _enrollment_request())
    manifest = _manifest(profile=entry.ENROLLMENT_ONLY)
    model = object()
    calls: dict = {}
    monkeypatch.setattr(
        entry,
        "load_verified_somph_predictor_bundle",
        lambda *args, **kwargs: ({"scenario": {}}, manifest, {"status": "structural"}),
    )
    monkeypatch.setattr(
        entry,
        "_load_fixed_runtime",
        lambda *args, **kwargs: (model, {"locked": True}),
    )

    def fake_enroll(loaded_model, payloads, **kwargs):
        calls["model"] = loaded_model
        calls["binding"] = kwargs["enrollment_binding"]
        calls["batch_size"] = kwargs["batch_size"]
        return {"capsule": np.asarray([1])}, {
            "enrollment_binding_sha256": "d" * 64,
        }

    monkeypatch.setattr(entry, "enroll_somph_heads", fake_enroll)
    monkeypatch.setattr(
        entry,
        "publish_somph_head_artifact",
        lambda *args, **kwargs: {
            "head_capsule_sha256": "e" * 64,
            "enrollment_binding_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(entry, "_write_readonly_json_new", lambda *args: "f" * 64)
    result = entry.run_somph_enrollment(
        request_json=request_path,
        package_root=tmp_path,
        detached_seal_path=tmp_path / "seal.json",
        expected_seal_sha256="1" * 64,
        output_root=tmp_path,
    )
    assert calls["model"] is model
    assert calls["batch_size"] == 8
    assert calls["binding"]["registered_class_handles"] == [
        item["class_handle"] for item in manifest["registered_classes"]
    ]
    assert "query" not in json.dumps(calls["binding"]).lower()
    assert result["head_capsule_sha256"] == "e" * 64
    assert result["formal_launch_authority"] is False
