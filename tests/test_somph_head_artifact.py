from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi.somph_head_artifact import (
    PROTOCOL_EVIDENCE_STATUS,
    SomphHeadArtifactError,
    _npz_bytes,
    publish_somph_head_artifact,
    verify_somph_head_artifact,
)
from cvsrffi.somph_predictor_runtime import (
    ADV3B02_CHECKPOINT_SHA256,
    SOMPH_ENROLLMENT_BINDING_SCHEMA,
    canonical_sha256,
    enroll_somph_heads,
    expected_somph_method_lock,
    somph_head_capsule_members,
)
from cvsrffi.stage2_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


def _token(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


class _Runtime(torch.nn.Module):
    def forward(self, rows: torch.Tensor):
        flat = rows.flatten(1)
        repeats = (160 + flat.shape[1] - 1) // flat.shape[1]
        features = flat.repeat(1, repeats)[:, :160]
        return features, features[:, :2]


def _support() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    labels = np.asarray([0, 1], dtype=np.int64)
    ranks = np.asarray([0, 0], dtype=np.int64)
    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        iq = np.asarray(
            [
                [[1.0 + index, 0.1], [0.2, 0.3]],
                [[11.0 + index, 1.1], [1.2, 1.3]],
            ],
            dtype=np.float32,
        )
        result[scenario] = {
            "support_leo_weak_iq": iq,
            "support_class_indices": labels,
            "support_rank_within_class": ranks,
            "support_tokens": np.asarray(
                [_token("sid", "class-0"), _token("sid", "class-1")]
            ),
            "support_overlay_tokens": np.asarray(
                [_token("oid", f"{scenario}-0"), _token("oid", f"{scenario}-1")]
            ),
            "support_satellite_seeds": np.asarray([1, 2], dtype=np.int64),
            "support_post_channel_iq_sha256": np.asarray(["a" * 64, "b" * 64]),
        }
    return result


def _capsule() -> tuple[dict[str, np.ndarray], dict, str]:
    lock = expected_somph_method_lock()
    binding = {
        "schema": SOMPH_ENROLLMENT_BINDING_SCHEMA,
        "stage": "stage2c",
        "registration_state": "after",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 1,
        "registered_class_handles": [
            _token("cls", "class-0"),
            _token("cls", "class-1"),
        ],
        "enrollment_package_root_sha256": "1" * 64,
        "enrollment_package_seal_sha256": "2" * 64,
        "checkpoint_sha256": ADV3B02_CHECKPOINT_SHA256,
        "method_lock_sha256": canonical_sha256(lock),
    }
    capsule, receipt = enroll_somph_heads(
        _Runtime().eval(),
        _support(),
        enrollment_binding=binding,
        method_lock=lock,
        device=torch.device("cpu"),
        batch_size=2,
    )
    return capsule, lock, receipt["enrollment_binding_sha256"]


def _publish(root: Path) -> tuple[Path, dict, dict, dict[str, np.ndarray], str]:
    root.mkdir()
    capsule, lock, binding_sha256 = _capsule()
    target = root / "somph_head.npz"
    publication = publish_somph_head_artifact(
        target,
        capsule=capsule,
        method_lock=lock,
        expected_enrollment_binding_sha256=binding_sha256,
    )
    return target, publication, lock, capsule, binding_sha256


def test_publish_and_verify_raw_npz_exact_readonly_artifact(tmp_path: Path) -> None:
    target, publication, lock, capsule, binding_sha256 = _publish(
        tmp_path / "case"
    )
    verified = verify_somph_head_artifact(
        target,
        method_lock=lock,
        expected_enrollment_binding_sha256=binding_sha256,
        expected_head_capsule_sha256=publication["head_capsule_sha256"],
    )
    assert tuple(verified["capsule"]) == somph_head_capsule_members()
    assert all(
        np.array_equal(verified["capsule"][name], capsule[name])
        for name in somph_head_capsule_members()
    )
    assert publication["head_capsule_sha256"] == verified["head_capsule_sha256"]
    assert publication["enrollment_binding_sha256"] == binding_sha256
    assert publication["readonly"] is True
    assert publication["formal_launch_authority"] is False
    assert publication["protocol_evidence_status"] == PROTOCOL_EVIDENCE_STATUS
    with zipfile.ZipFile(target, "r") as archive:
        assert tuple(archive.namelist()) == tuple(
            f"{name}.npy" for name in somph_head_capsule_members()
        )


def test_publish_never_overwrites_existing_artifact(tmp_path: Path) -> None:
    target, publication, lock, capsule, binding_sha256 = _publish(
        tmp_path / "case"
    )
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        publish_somph_head_artifact(
            target,
            capsule=capsule,
            method_lock=lock,
            expected_enrollment_binding_sha256=binding_sha256,
        )
    assert target.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == publication["head_capsule_sha256"]


def test_verify_rejects_writable_or_tampered_artifact(tmp_path: Path) -> None:
    target, publication, lock, _capsule_value, binding_sha256 = _publish(
        tmp_path / "case"
    )
    os.chmod(target, 0o600)
    with pytest.raises(SomphHeadArtifactError, match="not sealed read-only"):
        verify_somph_head_artifact(
            target,
            method_lock=lock,
            expected_enrollment_binding_sha256=binding_sha256,
        )
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(data)
    os.chmod(target, 0o444)
    with pytest.raises(SomphHeadArtifactError, match="SHA256 mismatch"):
        verify_somph_head_artifact(
            target,
            method_lock=lock,
            expected_enrollment_binding_sha256=binding_sha256,
            expected_head_capsule_sha256=publication["head_capsule_sha256"],
        )


def test_verify_rejects_extra_npz_member(tmp_path: Path) -> None:
    capsule, lock, binding_sha256 = _capsule()
    payload = io.BytesIO(_npz_bytes(capsule))
    rewritten = io.BytesIO()
    with zipfile.ZipFile(payload, "r") as source, zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr("query_labels.npy", b"forbidden")
    path = tmp_path / "extra.npz"
    path.write_bytes(rewritten.getvalue())
    os.chmod(path, 0o444)
    with pytest.raises(SomphHeadArtifactError, match="exact member/order"):
        verify_somph_head_artifact(
            path,
            method_lock=lock,
            expected_enrollment_binding_sha256=binding_sha256,
        )


def test_binding_and_method_lock_drift_fail_closed(tmp_path: Path) -> None:
    target, _publication, lock, capsule, binding_sha256 = _publish(
        tmp_path / "case"
    )
    with pytest.raises(SomphHeadArtifactError, match="binding digest mismatch"):
        verify_somph_head_artifact(
            target,
            method_lock=lock,
            expected_enrollment_binding_sha256="f" * 64,
        )
    drift = dict(lock)
    drift["hubness_weight"] = 0.5
    with pytest.raises(SomphHeadArtifactError, match="method lock drift"):
        verify_somph_head_artifact(
            target,
            method_lock=drift,
            expected_enrollment_binding_sha256=binding_sha256,
        )
    root = tmp_path / "publish-drift"
    root.mkdir()
    with pytest.raises(SomphHeadArtifactError, match="method lock drift"):
        publish_somph_head_artifact(
            root / "head.npz",
            capsule=capsule,
            method_lock=drift,
            expected_enrollment_binding_sha256=binding_sha256,
        )


def test_semantic_tamper_is_rejected_even_without_expected_file_sha(
    tmp_path: Path,
) -> None:
    capsule, lock, binding_sha256 = _capsule()
    tampered = dict(capsule)
    scalar_name = f"{FORMAL_LEO_WEAK_SCENARIOS[0]}__scalars_fp16"
    tampered[scalar_name] = np.asarray([0.5, 0.25], dtype=np.float16)
    path = tmp_path / "semantic-tamper.npz"
    path.write_bytes(_npz_bytes(tampered))
    os.chmod(path, 0o444)
    with pytest.raises(SomphHeadArtifactError, match="scoring scalar"):
        verify_somph_head_artifact(
            path,
            method_lock=lock,
            expected_enrollment_binding_sha256=binding_sha256,
        )
