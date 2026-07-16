from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.somph_prediction_artifact import (
    NPZ_FIELD_ALLOWLIST,
    SomphPredictionArtifactError,
    _check_npz,
    publish_somph_prediction_artifact,
    verify_somph_prediction_artifact,
)


def _qid(value: str) -> str:
    return f"qid_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _cls(value: str) -> str:
    return f"cls_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _binding(*, stage: str = "Stage2-C", state: str = "after_registration") -> dict:
    registered_class_count = (
        6 if stage == "Stage2-B" or state == "before_registration" else 11
    )
    return {
        "stage": stage,
        "registration_state": state,
        "row_id": f"row_{hashlib.sha256(b'row').hexdigest()}",
        "receiver": "8-8",
        "seed": 713106,
        "k_shot": 5,
        "registered_class_count": registered_class_count,
        "registry_snapshot_sha256": "0" * 64,
        "method_lock_sha256": "1" * 64,
        "row_manifest_sha256": "2" * 64,
        "stage_input_binding_sha256": "3" * 64,
        "package_root_sha256": "4" * 64,
        "package_seal_sha256": "5" * 64,
        "feature_runtime_sha256": "6" * 64,
        "head_capsule_sha256": "7" * 64,
        "protocol_policy_sha256": "8" * 64,
    }


def _publish(root: Path, *, tokens: list[str] | None = None, **binding) -> tuple[Path, dict]:
    root.mkdir()
    query_tokens = tokens or [_qid("z"), _qid("a"), _qid("m")]
    target = root / "prediction.cvspred"
    result = publish_somph_prediction_artifact(
        target,
        query_tokens=np.asarray(query_tokens),
        scenarios=np.asarray(["leo_rain_weak"] * len(query_tokens)),
        predicted_class_handles=np.asarray([_cls(str(index)) for index in range(len(query_tokens))]),
        backbone_forward_counts=np.ones(len(query_tokens), dtype=np.uint8),
        **(_binding() | binding),
    )
    return target, result


def test_single_stream_artifact_is_atomic_readonly_and_exact(tmp_path: Path) -> None:
    target, publication = _publish(tmp_path / "case")
    verified = verify_somph_prediction_artifact(
        target,
        expected_artifact_sha256=publication["artifact_sha256"],
        expected_seal_sha256=publication["seal_sha256"],
    )
    assert tuple(verified["arrays"]) == NPZ_FIELD_ALLOWLIST
    assert set(NPZ_FIELD_ALLOWLIST) == {
        "query_tokens",
        "scenarios",
        "predicted_class_handles",
        "backbone_forward_counts",
    }
    assert publication["immutable_state"] == "SEALED_READ_ONLY_ATOMIC_NOREPLACE"
    assert publication["readonly"] is True
    with pytest.raises(FileExistsError):
        publish_somph_prediction_artifact(
            target,
            query_tokens=np.asarray([_qid("new")]),
            scenarios=np.asarray(["leo_clear_weak"]),
            predicted_class_handles=np.asarray([_cls("new")]),
            backbone_forward_counts=np.ones(1, dtype=np.uint8),
            **_binding(),
        )


def test_predictor_artifact_accepts_arbitrary_query_count_and_order(tmp_path: Path) -> None:
    tokens = [_qid(value) for value in ("last", "first")]
    target, publication = _publish(tmp_path / "case", tokens=tokens)
    arrays = verify_somph_prediction_artifact(
        target,
        expected_artifact_sha256=publication["artifact_sha256"],
        expected_seal_sha256=publication["seal_sha256"],
    )["arrays"]
    assert arrays["query_tokens"].tolist() == tokens


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("query_tokens", np.asarray(["target_old|14-10"]), "opaque qid"),
        ("predicted_class_handles", np.asarray(["14-10"]), "opaque cls"),
        ("backbone_forward_counts", np.asarray([3]), "exactly one"),
        ("backbone_forward_counts", np.asarray([257]), "exactly one"),
    ],
)
def test_rejects_nonopaque_or_multiview_payload(
    tmp_path: Path, field: str, values: np.ndarray, message: str
) -> None:
    payload = {
        "query_tokens": np.asarray([_qid("one")]),
        "scenarios": np.asarray(["leo_clear_weak"]),
        "predicted_class_handles": np.asarray([_cls("one")]),
        "backbone_forward_counts": np.ones(1, dtype=np.uint8),
    }
    payload[field] = values
    root = tmp_path / field
    root.mkdir()
    with pytest.raises(SomphPredictionArtifactError, match=message):
        publish_somph_prediction_artifact(root / "prediction.cvspred", **payload, **_binding())


def test_rejects_duplicate_query_key_and_forbidden_binding_fields(tmp_path: Path) -> None:
    token = _qid("same")
    root = tmp_path / "duplicate"
    root.mkdir()
    with pytest.raises(SomphPredictionArtifactError, match="must be unique"):
        publish_somph_prediction_artifact(
            root / "prediction.cvspred",
            query_tokens=np.asarray([token, token]),
            scenarios=np.asarray(["leo_clear_weak", "leo_clear_weak"]),
            predicted_class_handles=np.asarray([_cls("a"), _cls("b")]),
            backbone_forward_counts=np.ones(2, dtype=np.uint8),
            **_binding(),
        )
    forbidden = _binding()
    forbidden["query_per_tx"] = 20
    forbidden_root = tmp_path / "forbidden"
    forbidden_root.mkdir()
    with pytest.raises(SomphPredictionArtifactError, match="binding exact schema drift"):
        publish_somph_prediction_artifact(
            forbidden_root / "prediction.cvspred",
            query_tokens=np.asarray([_qid("one")]),
            scenarios=np.asarray(["leo_clear_weak"]),
            predicted_class_handles=np.asarray([_cls("one")]),
            backbone_forward_counts=np.ones(1, dtype=np.uint8),
            **forbidden,
        )
    future_count = _binding(state="before_registration")
    future_count["new_class_count"] = 5
    future_root = tmp_path / "future_count"
    future_root.mkdir()
    with pytest.raises(SomphPredictionArtifactError, match="binding exact schema drift"):
        publish_somph_prediction_artifact(
            future_root / "prediction.cvspred",
            query_tokens=np.asarray([_qid("one")]),
            scenarios=np.asarray(["leo_clear_weak"]),
            predicted_class_handles=np.asarray([_cls("one")]),
            backbone_forward_counts=np.ones(1, dtype=np.uint8),
            **future_count,
        )


def test_stage2b_requires_before_registration(tmp_path: Path) -> None:
    root = tmp_path / "case"
    root.mkdir()
    invalid = _binding(stage="Stage2-B", state="after_registration")
    with pytest.raises(SomphPredictionArtifactError, match="Stage2-B requires"):
        publish_somph_prediction_artifact(
            root / "prediction.cvspred",
            query_tokens=np.asarray([_qid("one")]),
            scenarios=np.asarray(["leo_clear_weak"]),
            predicted_class_handles=np.asarray([_cls("one")]),
            backbone_forward_counts=np.ones(1, dtype=np.uint8),
            **invalid,
        )


def test_tampered_or_writeable_artifact_is_rejected(tmp_path: Path) -> None:
    target, publication = _publish(tmp_path / "case")
    os.chmod(target, 0o600)
    with pytest.raises(SomphPredictionArtifactError, match="not sealed read-only"):
        verify_somph_prediction_artifact(
            target,
            expected_artifact_sha256=publication["artifact_sha256"],
            expected_seal_sha256=publication["seal_sha256"],
        )


def test_npz_preflight_rejects_compressed_members_before_numpy_load() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for field in NPZ_FIELD_ALLOWLIST:
            array = np.asarray([_qid(field)])
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            archive.writestr(f"{field}.npy", member.getvalue())
    with pytest.raises(SomphPredictionArtifactError, match="ZIP_STORED"):
        _check_npz(payload.getvalue())
