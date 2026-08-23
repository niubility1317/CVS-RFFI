from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_stage2_structured_late_block_no_query_smoke import (
    prepare,
    prepare_query,
    run_row,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _row_binding() -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
        "receiver": "20-1",
        "method_seed": 7282101,
        "support_seed": 7282201,
        "query_seed": 7282301,
        "new_class_draw_seed": 7282401,
        "k_shot": 5,
        "scenario": "leo_clear_weak",
    }


def _package_manifest(filename: str) -> dict[str, object]:
    return {
        "stage": "stage2b",
        "receiver": "20-1",
        "seed": 7282101,
        "target_channel_scenarios": ["leo_clear_weak"],
        "package_root_sha256": "package-root-a",
        "members": [
            {
                "artifact_role": "query:leo_clear_weak",
                "scenario": "leo_clear_weak",
                "relative_path": filename,
            }
        ],
        "phase2_source_sample_access": False,
        "phase2_source_cache_access": False,
        "phase2_source_derived_signal_access": False,
        "phase2_source_replay": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
    }


def _validated_manifest(*, package_root_sha256: str) -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
        "receiver": "20-1",
        "method_seed": 7282101,
        "support_seed": 7282201,
        "query_seed": 7282301,
        "k_shot": 5,
        "stage_scope": "stage2b",
        "scenarios": ["leo_clear_weak"],
        "package_root_sha256": package_root_sha256,
        "phase2_source_sample_access": False,
        "phase2_source_cache_access": False,
        "phase2_source_derived_signal_access": False,
        "phase2_source_replay": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "query_truth_present": False,
        "query_role_present": False,
    }


def test_prepare_query_requires_same_validated_package_root(tmp_path: Path) -> None:
    query_path = tmp_path / "query_leo_clear_weak.npz"
    np.savez(query_path, query_leo_weak_iq=np.zeros((2, 2, 8), dtype=np.float32))
    package_path = tmp_path / "package_manifest.json"
    row_path = tmp_path / "row.json"
    validated_path = tmp_path / "validated.json"
    output_path = tmp_path / "query_only.npz"
    _write_json(package_path, _package_manifest(query_path.name))
    _write_json(row_path, _row_binding())
    _write_json(
        validated_path,
        _validated_manifest(package_root_sha256="different-package-root"),
    )

    with pytest.raises(ValueError, match="VALIDATED_ONCE"):
        prepare_query(
            SimpleNamespace(
                query_package=query_path,
                package_manifest=package_path,
                validated_row_manifest=validated_path,
                row_binding=row_path,
                output=output_path,
            )
        )


def test_prepare_query_emits_iq_only_after_full_row_binding(tmp_path: Path) -> None:
    query_path = tmp_path / "query_leo_clear_weak.npz"
    received_iq = np.ones((2, 2, 8), dtype=np.float32)
    np.savez(query_path, query_leo_weak_iq=received_iq)
    package_path = tmp_path / "package_manifest.json"
    row_path = tmp_path / "row.json"
    validated_path = tmp_path / "validated.json"
    output_path = tmp_path / "query_only.npz"
    _write_json(package_path, _package_manifest(query_path.name))
    _write_json(row_path, _row_binding())
    _write_json(
        validated_path,
        _validated_manifest(package_root_sha256="package-root-a"),
    )

    result = prepare_query(
        SimpleNamespace(
            query_package=query_path,
            package_manifest=package_path,
            validated_row_manifest=validated_path,
            row_binding=row_path,
            output=output_path,
        )
    )

    assert result["status"] == "PREPARED"
    with np.load(output_path, allow_pickle=False) as query_only:
        assert query_only.files == ["received_iq"]
        assert np.array_equal(query_only["received_iq"], received_iq)


def test_prepare_query_rejects_self_reported_protocol_without_builder_field(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "query_leo_clear_weak.npz"
    np.savez(query_path, query_leo_weak_iq=np.zeros((2, 2, 8), dtype=np.float32))
    package_path = tmp_path / "package_manifest.json"
    row_path = tmp_path / "row.json"
    validated_path = tmp_path / "validated.json"
    output_path = tmp_path / "query_only.npz"
    _write_json(package_path, _package_manifest(query_path.name))
    _write_json(row_path, _row_binding())
    validated = _validated_manifest(package_root_sha256="package-root-a")
    validated.pop("protocol_schema")
    _write_json(validated_path, validated)

    with pytest.raises(ValueError, match="VALIDATED_ONCE"):
        prepare_query(
            SimpleNamespace(
                query_package=query_path,
                package_manifest=package_path,
                validated_row_manifest=validated_path,
                row_binding=row_path,
                output=output_path,
            )
        )


def test_formal_row_first_opens_raw_query_after_adaptation_freezes() -> None:
    source = inspect.getsource(run_row)
    assert source.index("_adapt_from_whitelist") < source.index(
        "_load_query_received_iq"
    )
    assert "query_only" not in source


def test_prepare_does_not_export_torch_arrays_by_numpy_identity() -> None:
    assert ".numpy()" not in inspect.getsource(prepare)
