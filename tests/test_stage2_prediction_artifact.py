from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.stage2_prediction_artifact import (  # noqa: E402
    NPZ_FIELD_ALLOWLIST,
    NPZ_MEMBER_ALLOWLIST,
    OUTER_MEMBER_ALLOWLIST,
    PredictionArtifactError,
    publish_prediction_artifact,
    verify_prediction_artifact,
)
from cvsrffi import stage2_prediction_artifact as artifact_module  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _kwargs() -> dict:
    return {
        "stage": "Stage2-C",
        "row_id": "rx20-1-seed1-k10",
        "receiver": "20-1",
        "k_shot": 10,
        "candidate_lock_sha256": SHA_A,
        "package_root_sha256": SHA_B,
        "package_seal_sha256": SHA_C,
        "query_tokens": np.asarray(["q_001", "q_002", "q_003"]),
        "scenarios": np.asarray(["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]),
        "candidate_after": np.asarray(["c01", "c02", "c03"]),
        "candidate_before": np.asarray(["c01", "c01", "c03"]),
        "identity_after": np.asarray(["c01", "c02", "c02"]),
        "identity_before": np.asarray(["c01", "c01", "c02"]),
        "direct": np.asarray(["c01", "c01", "c01"]),
        "shared_view_counts": np.asarray([1, 3, 5], dtype=np.uint8),
    }


def _read_outer(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _write_outer(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    os.chmod(path, 0o444)


def test_roundtrip_exact_fields_bindings_and_resource_receipt(tmp_path: Path) -> None:
    path = tmp_path / "predictions.stage2pred"
    receipt = publish_prediction_artifact(path, **_kwargs())
    verified = verify_prediction_artifact(
        path,
        expected_artifact_sha256=receipt["artifact_sha256"],
        expected_seal_sha256=receipt["seal_sha256"],
    )

    assert receipt["readonly"] is True
    assert verified["immutable_state"] == "SEALED_READ_ONLY_ATOMIC_NOREPLACE"
    assert stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    assert tuple(verified["arrays"]) == NPZ_FIELD_ALLOWLIST
    assert verified["arrays"]["query_tokens"].tolist() == ["q_001", "q_002", "q_003"]
    assert verified["manifest"]["stage"] == "Stage2-C"
    assert verified["seal"]["row_id"] == "rx20-1-seed1-k10"
    assert verified["manifest"]["candidate_lock_sha256"] == SHA_A
    assert verified["seal"]["package_root_sha256"] == SHA_B
    assert verified["seal"]["package_seal_sha256"] == SHA_C
    resource = verified["manifest"]["resource_receipt"]
    assert resource == {
        "query_count": 3,
        "total_backbone_forward_count": 9,
        "mean_backbone_forward_count": 3.0,
        "p95_backbone_forward_count": 5,
        "max_backbone_forward_count": 5,
        "view_1_trigger_count": 1,
        "view_3_trigger_count": 1,
        "view_5_trigger_count": 1,
    }
    assert verified["adapter_resource_verification"] == {
        "status": "NOT_PROVABLE_FROM_PREDICTION_ARTIFACT",
        "reason_code": "ADAPTER_MATRIX_NOT_EMBEDDED",
        "adapter_matrix_embedded": False,
        "trainable_parameter_count_verified": False,
        "persistent_state_bytes_verified": False,
        "formal_adapter_resource_claim_allowed": False,
    }
    assert receipt["adapter_resource_verification"] == verified[
        "adapter_resource_verification"
    ]
    members = _read_outer(path)
    assert set(members) == set(OUTER_MEMBER_ALLOWLIST)
    with zipfile.ZipFile(io.BytesIO(members["payload.npz"]), "r") as payload_zip:
        assert tuple(payload_zip.namelist()) == NPZ_MEMBER_ALLOWLIST


def test_existing_target_fails_without_overwrite_or_temp_leak(tmp_path: Path) -> None:
    path = tmp_path / "predictions.stage2pred"
    original = b"do-not-overwrite"
    path.write_bytes(original)
    with pytest.raises(FileExistsError):
        publish_prediction_artifact(path, **_kwargs())
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_rejects_outer_extra_member(tmp_path: Path) -> None:
    source = tmp_path / "source.stage2pred"
    publish_prediction_artifact(source, **_kwargs())
    members = _read_outer(source)
    members["neutral.txt"] = b"extra"
    mutated = tmp_path / "extra.stage2pred"
    _write_outer(mutated, members)
    with pytest.raises(PredictionArtifactError, match="members must exactly"):
        verify_prediction_artifact(mutated)


def test_rejects_npz_extra_member_before_hash_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source.stage2pred"
    publish_prediction_artifact(source, **_kwargs())
    members = _read_outer(source)
    with np.load(io.BytesIO(members["payload.npz"]), allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    arrays["neutral"] = np.asarray([0, 0, 0])
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    members["payload.npz"] = payload.getvalue()
    mutated = tmp_path / "extra-npz.stage2pred"
    _write_outer(mutated, members)
    with pytest.raises(PredictionArtifactError, match="payload NPZ members must exactly"):
        verify_prediction_artifact(mutated)


@pytest.mark.parametrize("member", ["payload.npz", "manifest.json", "seal.json"])
def test_rejects_each_layer_tamper(tmp_path: Path, member: str) -> None:
    source = tmp_path / "source.stage2pred"
    publish_prediction_artifact(source, **_kwargs())
    members = _read_outer(source)
    if member == "payload.npz":
        with np.load(io.BytesIO(members[member]), allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name]) for name in archive.files}
        arrays["direct"][0] = "c99"
        payload = io.BytesIO()
        np.savez(payload, **arrays)
        members[member] = payload.getvalue()
    else:
        document = json.loads(members[member].decode("utf-8"))
        document["row_id"] = "tampered-row"
        members[member] = json.dumps(
            document, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    mutated = tmp_path / f"tampered-{member.replace('.', '-')}.stage2pred"
    _write_outer(mutated, members)
    with pytest.raises(PredictionArtifactError):
        verify_prediction_artifact(mutated)


def test_rejects_writable_artifact_even_when_hashes_match(tmp_path: Path) -> None:
    path = tmp_path / "predictions.stage2pred"
    publish_prediction_artifact(path, **_kwargs())
    os.chmod(path, 0o644)
    with pytest.raises(PredictionArtifactError, match="write bits are set"):
        verify_prediction_artifact(path)


def test_rejects_symlink_and_nonregular_path(tmp_path: Path) -> None:
    source = tmp_path / "source.stage2pred"
    publish_prediction_artifact(source, **_kwargs())
    link = tmp_path / "link.stage2pred"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PredictionArtifactError, match="must not be a symlink"):
        verify_prediction_artifact(link)
    with pytest.raises(PredictionArtifactError, match="must be a regular file"):
        verify_prediction_artifact(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shared_view_counts", np.asarray([1, 2, 5]), "drawn from 1/3/5"),
        (
            "shared_view_counts",
            np.asarray([1, 257, 5], dtype=np.int64),
            "drawn from 1/3/5",
        ),
        ("scenarios", np.asarray(["leo_clear_weak", "clean", "leo_rain_weak"]), "unsupported LEO"),
        ("direct", np.asarray(["c01", "c02"]), "inconsistent lengths"),
    ],
)
def test_rejects_invalid_prediction_schema(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    kwargs = _kwargs()
    kwargs[field] = value
    with pytest.raises(PredictionArtifactError, match=message):
        publish_prediction_artifact(tmp_path / "bad.stage2pred", **kwargs)


def test_rejects_duplicate_scenario_query_token_key(tmp_path: Path) -> None:
    kwargs = _kwargs()
    kwargs["query_tokens"] = np.asarray(["q_001", "q_001", "q_003"])
    kwargs["scenarios"] = np.asarray(
        ["leo_clear_weak", "leo_clear_weak", "leo_rain_weak"]
    )
    with pytest.raises(PredictionArtifactError, match="scenario/query_token keys"):
        publish_prediction_artifact(tmp_path / "duplicate-key.stage2pred", **kwargs)


def test_external_seal_digest_detects_whole_seal_replacement(tmp_path: Path) -> None:
    path = tmp_path / "predictions.stage2pred"
    receipt = publish_prediction_artifact(path, **_kwargs())
    assert receipt["seal_sha256"] == hashlib.sha256(_read_outer(path)["seal.json"]).hexdigest()
    with pytest.raises(PredictionArtifactError, match="expected digest"):
        verify_prediction_artifact(path, expected_seal_sha256="d" * 64)


def test_adapter_bearing_schema_fails_closed_without_content_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_module,
        "NPZ_FIELD_ALLOWLIST",
        (*NPZ_FIELD_ALLOWLIST, "adapter_matrix"),
    )
    with pytest.raises(PredictionArtifactError, match="content-based resource"):
        artifact_module._adapter_resource_verification()
