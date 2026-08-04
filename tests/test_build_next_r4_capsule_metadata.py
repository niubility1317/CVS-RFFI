from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cvsrffi import stage2_zid_student_t_qknn as qknn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "code" / "scripts" / "build_next_r4_capsule_metadata.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location(
        "test_next_r4_capsule_metadata_builder", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lock(k: int) -> dict[str, object]:
    return dataclasses.asdict(
        qknn.Phase1ZIDStudentTLock(
            active_k=k,
            student_nu=3.0,
            kernel_effective_dim=12,
            kernel_volume_gamma=1.0,
            shared_h0=0.35,
            scale_prior_strength=2.0,
            scale_min_ratio=0.5,
            scale_max_ratio=2.0,
            temperature=0.85,
            phase1_lodo_receipt_sha256="1" * 64,
            quantization_margin_audit_sha256="2" * 64,
        )
    )


def _write_npz(path: Path, **arrays: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    path.write_bytes(buffer.getvalue())
    return _sha(path.read_bytes())


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    module = _module()
    receivers = ("1-1", "18-2", "1-19", "2-1", "2-19", "14-7", "19-2")
    classes = module.FIXED_CLASSES
    physical: list[str] = []
    observation: list[str] = []
    receiver_ids: list[str] = []
    day_ids: list[str] = []
    scenarios: list[str] = []
    tx_labels: list[str] = []
    for receiver in receivers:
        for class_id in classes:
            for index in range(14):
                physical.append(f"opaque|{receiver}|{class_id}|{index}")
                observation.append(f"obs|{receiver}|{class_id}|{index}")
                receiver_ids.append(receiver)
                day_ids.append(f"day-{index % 4}")
                scenarios.append("leo_clear_weak")
                tx_labels.append(class_id)
    assert len(physical) == 588
    tap_path = (tmp_path / "d106_ls_strict_tap.npz").resolve()
    tap_sha = _write_npz(
        tap_path,
        pre_relu=np.ones((588, 160), dtype="<f4"),
        tx_labels=np.asarray(tx_labels, dtype="<U8"),
        receiver_ids=np.asarray(receiver_ids, dtype="<U8"),
        day_ids=np.asarray(day_ids, dtype="<U8"),
        physical_ids=np.asarray(physical, dtype="<U64"),
        scenario_names=np.asarray(scenarios, dtype="<U32"),
        observation_ids=np.asarray(observation, dtype="<U64"),
    )
    received_path = (tmp_path / "d106_ls_received_iq.npz").resolve()
    received_sha = _write_npz(
        received_path,
        received_iq=np.zeros((588, 2, 16), dtype="<f4"),
        receiver_ids=np.asarray(receiver_ids, dtype="<U8"),
        day_ids=np.asarray(day_ids, dtype="<U8"),
        physical_ids=np.asarray(physical, dtype="<U64"),
        scenario_names=np.asarray(scenarios, dtype="<U32"),
        observation_ids=np.asarray(observation, dtype="<U64"),
    )
    lock_path = (tmp_path / "qknn_locks.json").resolve()
    lock_path.write_bytes(
        json.dumps(
            {"schema": qknn.LOCK_SCHEMA, "K1": _lock(1), "K5": _lock(5)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return tap_path, received_path, lock_path, tap_sha, received_sha


def test_builds_real_shape_metadata_and_closes_all_rows(tmp_path: Path) -> None:
    module = _module()
    tap, received, locks, tap_sha, received_sha = _fixture(tmp_path)
    output = (tmp_path / "capsule_metadata.json").resolve()
    result = module.build_next_r4_capsule_metadata(
        strict_tap=tap,
        received_iq=received,
        qknn_locks=locks,
        validator_receipt_sha256="d" * 64,
        output_path=output,
        strict_tap_sha256=tap_sha,
        received_iq_sha256=received_sha,
    )
    metadata = result["metadata"]
    assert result["status"] == module.BUILD_STATUS
    assert result["output_path"] == str(output)
    assert output.read_bytes() == json.dumps(
        metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert metadata["received_iq_sha256"] == received_sha
    assert metadata["physical_id_count"] == 588
    assert metadata["receiver_count"] == 7
    assert metadata["class_count"] == 6
    assert metadata["row_count"] == 24
    assert metadata["phase1_fit_count_per_row"] == 420
    assert metadata["class_registry"] == list(module.FIXED_CLASSES)
    assert metadata["held_receivers"] == ["1-1", "18-2"]
    assert metadata["qknn_lock_by_k"]["1"] == _lock(1)
    assert metadata["qknn_lock_by_k"]["5"] == _lock(5)
    assert metadata["qknn_locks"]["K1"] == _lock(1)
    assert metadata["qknn_locks"]["K5"] == _lock(5)
    assert len({row["row_id"] for row in metadata["rows"]}) == 24
    for row in metadata["rows"]:
        assert len(row["phase1_fit_ids"]) == 420
        assert all(len(row["k1_support_ids_by_class"][c]) == 1 for c in module.FIXED_CLASSES)
        assert all(len(row["k5_support_ids_by_class"][c]) == 5 for c in module.FIXED_CLASSES)
        assert all(len(row["query_ids_by_class"][c]) == 9 for c in module.FIXED_CLASSES)
        for class_id in module.FIXED_CLASSES:
            assert row["k1_support_ids_by_class"][class_id] == row["k5_support_ids_by_class"][class_id][:1]
            assert not set(row["k5_support_ids_by_class"][class_id]) & set(row["query_ids_by_class"][class_id])
        assert not set(row["phase1_fit_ids"]) & {
            item
            for values in row["k5_support_ids_by_class"].values()
            for item in values
        }
    assert metadata["split_identity"]["row_id_set_sorted"] == sorted(
        metadata["split_identity"]["row_id_order"]
    )
    assert metadata["split_identity"]["physical_id_sort_policy"] == module.PHYSICAL_ID_SORT_POLICY


def test_rejects_id_order_mismatch_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _module()
    tap, received, locks, tap_sha, received_sha = _fixture(tmp_path)
    with np.load(received, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["receiver_ids"] = arrays["receiver_ids"].copy()
    arrays["receiver_ids"][0] = "wrong"
    bad_received = (tmp_path / "bad_received.npz").resolve()
    bad_received_sha = _write_npz(bad_received, **arrays)
    with pytest.raises(module.NextR4CapsuleMetadataError, match="receiver_ids"):
        module.prepare_next_r4_capsule_metadata(
            strict_tap=tap,
            received_iq=bad_received,
            qknn_locks=locks,
            validator_receipt_sha256="d" * 64,
            strict_tap_sha256=tap_sha,
            received_iq_sha256=bad_received_sha,
        )
    output = (tmp_path / "capsule_metadata.json").resolve()
    module.build_next_r4_capsule_metadata(
        strict_tap=tap,
        received_iq=received,
        qknn_locks=locks,
        validator_receipt_sha256="d" * 64,
        output_path=output,
        strict_tap_sha256=tap_sha,
        received_iq_sha256=received_sha,
    )
    with pytest.raises(module.NextR4CapsuleMetadataError, match="new absolute"):
        module.build_next_r4_capsule_metadata(
            strict_tap=tap,
            received_iq=received,
            qknn_locks=locks,
            validator_receipt_sha256="d" * 64,
            output_path=output,
            strict_tap_sha256=tap_sha,
            received_iq_sha256=received_sha,
        )


def test_cli_surface_is_mechanical_and_does_not_expose_tuning(tmp_path: Path) -> None:
    module = _module()
    names = {action.dest for action in module._parser()._actions}
    assert {"strict_tap", "received_iq", "qknn_locks", "validator_receipt_sha256", "output_path"}.issubset(names)
    assert not {"truth", "query", "rank", "threshold", "temperature", "alpha", "lambda"}.intersection(names)
