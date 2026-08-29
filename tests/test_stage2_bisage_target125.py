from __future__ import annotations

from copy import deepcopy

import pytest

from cvsrffi.stage2_bisage_target125 import (
    BiSAGETarget125Error,
    build_bisage_target125_manifest,
    canonical_target125_rows,
    validate_bisage_target125_manifest,
    validate_target125_config,
)


def _config() -> dict:
    return {
        "schema": "cvs.phase2.bisage_d92_target125.method_lock.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "historical_source": {
            "run_id": "d92_registration_balanced_125_retry2_20260720",
            "matrix_manifest_path": "/runs/d92_registration_balanced_125_retry2_20260720/matrix_manifest.json",
            "matrix_manifest_sha256": "b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c",
        },
        "receivers": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "seeds": [713102, 713103, 713104, 713105, 713106],
        "slices": [
            {"k_shot": 1, "new_class_count": 20},
            {"k_shot": 5, "new_class_count": 20},
            {"k_shot": 10, "new_class_count": 5},
            {"k_shot": 10, "new_class_count": 10},
            {"k_shot": 10, "new_class_count": 20},
        ],
        "scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "matrix": {"outer_count": 125, "scene_count": 3, "scene_unit_count": 375},
        "query_contract": {
            "truth_access": False,
            "fit_access": False,
            "update_access": False,
            "selection_access": False,
            "role_oracle_access": False,
            "class_quota_access": False,
            "global_reassignment": False,
        },
    }


def _source_manifest() -> dict:
    jobs = []
    for row in canonical_target125_rows():
        jobs.append(
            {
                **row,
                "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE",
                "capsule_id": f"capsule-{row['outer_key']}",
                "split_id": f"split-{row['outer_key']}",
                "source_job_root": f"/sealed/{row['outer_key']}",
                "scenarios": list(row["scenarios"]),
            }
        )
    return {"jobs": jobs}


def test_target125_reuses_historical_d92_axes() -> None:
    rows = canonical_target125_rows()
    assert len(rows) == 125
    assert len({row["outer_key"] for row in rows}) == 125
    assert {row["receiver"] for row in rows} == {"20-1", "3-19", "7-14", "7-7", "8-8"}
    assert {row["seed"] for row in rows} == set(range(713102, 713107))
    assert {(row["k_shot"], row["new_class_count"]) for row in rows} == {
        (1, 20),
        (5, 20),
        (10, 5),
        (10, 10),
        (10, 20),
    }


def test_config_locks_validated_once_and_query_denials() -> None:
    audit = validate_target125_config(_config())
    assert audit["outer_count"] == 125
    assert audit["scene_unit_count"] == 375
    drift = _config()
    drift["query_contract"]["selection_access"] = True
    with pytest.raises(BiSAGETarget125Error, match="query contract"):
        validate_target125_config(drift)


def test_manifest_reuses_capsule_and_split_bindings() -> None:
    manifest = build_bisage_target125_manifest(_config(), _source_manifest(), "/new/run")
    audit = validate_bisage_target125_manifest(manifest, _config())
    assert audit == {"outer_count": 125, "scene_unit_count": 375, "k1_fallback_count": 25}
    assert manifest["jobs"][0]["selected_mode_policy"] == "S0_K1_FALLBACK"
    assert manifest["jobs"][-1]["selected_mode_policy"] == "SUPPORT_ONLY_S2_S1_S0"


def test_manifest_rejects_capsule_or_split_drift() -> None:
    manifest = build_bisage_target125_manifest(_config(), _source_manifest(), "/new/run")
    for field in ("capsule_id", "split_id"):
        drift = deepcopy(manifest)
        drift["jobs"][0][field] = "wrong"
        with pytest.raises(BiSAGETarget125Error, match=field):
            validate_bisage_target125_manifest(drift, _config())


def test_manifest_rejects_duplicate_outer_and_output_collision() -> None:
    manifest = build_bisage_target125_manifest(_config(), _source_manifest(), "/new/run")
    drift = deepcopy(manifest)
    drift["jobs"][1] = deepcopy(drift["jobs"][0])
    with pytest.raises(BiSAGETarget125Error, match="outer coverage"):
        validate_bisage_target125_manifest(drift, _config())
    collision = deepcopy(manifest)
    collision["jobs"][1]["output_root"] = collision["jobs"][0]["output_root"]
    with pytest.raises(BiSAGETarget125Error, match="output root"):
        validate_bisage_target125_manifest(collision, _config())
