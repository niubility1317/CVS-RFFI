from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi import stage2_d92_continuous_session_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage2_d92_e0_continuous_session_v1.json"


def test_frozen_matrix_constants_and_schedules() -> None:
    assert matrix.PROTOCOL_SCHEMA == "p2_min_v1"
    assert matrix.CLAIM_SCOPE == "DEVELOPMENT_ONLY_CONTINUOUS_SESSION_SCREEN"
    assert matrix.METHOD_ID == "D92_E0_CUMULATIVE_REPLAY_SESSION_V1"
    assert matrix.RECEIVERS == ("20-1", "3-19", "7-14", "7-7", "8-8")
    assert matrix.SCENES == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )
    assert matrix.SEED == 713106
    assert matrix.K_SHOT == 10
    assert matrix.OLD_CLASS_COUNT == 6
    assert matrix.NEW_CLASS_COUNT == 5
    assert matrix.RESOURCE_GATE == {
        "registration_wall_target_ns": 300_000_000,
        "registration_incremental_peak_hard_max_bytes": 4 * 1024 * 1024,
        "query_state_bytes_equal": True,
        "query_macs_equal": True,
    }
    assert matrix.SCHEDULES == {
        "batch_5": (5,),
        "singleton_forward": (1, 1, 1, 1, 1),
        "singleton_reverse": (1, 1, 1, 1, 1),
        "chunk_2_2_1": (2, 2, 1),
    }
    assert matrix.ARRIVAL_ORDERS["batch_5"] == (0, 1, 2, 3, 4)
    assert matrix.ARRIVAL_ORDERS["singleton_forward"] == (0, 1, 2, 3, 4)
    assert matrix.ARRIVAL_ORDERS["singleton_reverse"] == (4, 3, 2, 1, 0)
    assert matrix.ARRIVAL_ORDERS["chunk_2_2_1"] == (0, 1, 2, 3, 4)


def test_config_is_a_truth_free_method_lock_with_four_sealed_layouts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["schema"] == "cvs.phase2.d92_e0_continuous_session.method_lock.v1"
    assert config["matrix_schema"] == matrix.MATRIX_SCHEMA
    assert config["method_id"] == matrix.METHOD_ID
    assert config["protocol_schema"] == matrix.PROTOCOL_SCHEMA
    assert config["claim_scope"] == matrix.CLAIM_SCOPE
    assert config["matrix"] == {
        "receiver_count": 5,
        "scene_count": 3,
        "schedule_count": 4,
        "job_count": 5,
        "session_fit_count": 210,
    }
    assert config["resource_gate"] == matrix.RESOURCE_GATE
    assert config["resource_gate"]["registration_wall_target_ns"] == 300_000_000
    assert config["query_contract"] == {
        "decision": "per_sample_all_registered_classes",
        "truth_access": False,
        "fit_access": False,
        "update_access": False,
        "selection_access": False,
        "role_oracle_access": False,
        "class_quota_access": False,
        "global_reassignment": False,
    }
    assert set(config["sealed_inputs"]["package_layout"]) == {
        "before_enrollment",
        "before_apply",
        "after_enrollment",
        "after_apply",
    }
    assert all(
        set(entry) == {"package_relative_path", "seal_relative_path"}
        for entry in config["sealed_inputs"]["package_layout"].values()
    )
    # The configuration carries only package/seal identities and no truth
    # sidecar or truth payload.  Truth is opened only by the later scorer.
    encoded = json.dumps(config, ensure_ascii=False).lower()
    assert "truth_sidecar" not in encoded
    assert "truth_payload" not in encoded


def test_build_manifest_expands_five_outer_rows_three_scenes_four_schedules(
    tmp_path: Path,
) -> None:
    manifest = matrix.build_continuous_session_manifest(
        method_lock_path=CONFIG,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["schema"] == matrix.MATRIX_SCHEMA
    assert manifest["method_lock_sha256"] == matrix.sha256_file(CONFIG)
    assert manifest["outer_count"] == 5
    assert manifest["scene_count"] == 3
    assert manifest["schedule_count"] == 4
    assert manifest["job_count"] == 5
    assert manifest["session_fit_count"] == 210
    assert manifest["schedules"] == [
        "batch_5",
        "singleton_forward",
        "singleton_reverse",
        "chunk_2_2_1",
    ]
    assert len(manifest["jobs"]) == 5
    assert len({job["job_id"] for job in manifest["jobs"]}) == 5
    for job in manifest["jobs"]:
        assert job["receiver"] in matrix.RECEIVERS
        assert job["scenes"] == list(matrix.SCENES)
        assert job["schedules"] == list(matrix.SCHEDULE_NAMES)
        assert job["scene_schedule_count"] == 12
        assert job["k_shot"] == 10
        assert job["seed"] == 713106
        assert job["old_class_count"] == 6
        assert job["new_class_count"] == 5
        assert set(job["packages"]) == set(matrix.PACKAGE_LAYOUT)
        assert "truth_sidecar" not in job
        for name, package in job["packages"].items():
            assert package["expected_seal_sha256"] is None
            assert package["package_root"].endswith(
                "/".join(matrix.PACKAGE_LAYOUT[name][0])
            )
            assert package["detached_seal_path"].endswith(
                "/".join(matrix.PACKAGE_LAYOUT[name][1])
            )
    assert matrix.validate_continuous_session_manifest(manifest)["job_count"] == 5


def test_manifest_validation_rejects_schedule_and_key_drift(tmp_path: Path) -> None:
    manifest = matrix.build_continuous_session_manifest(
        method_lock_path=CONFIG,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    jobs = [dict(job) for job in manifest["jobs"]]
    jobs[0] = {**jobs[0], "schedules": ["batch_5"]}
    with pytest.raises(ValueError, match="schedule|job"):
        matrix.validate_continuous_session_manifest({**manifest, "jobs": jobs})
    with pytest.raises(ValueError, match="allowed-key|identity"):
        matrix.validate_continuous_session_manifest({**manifest, "unexpected": True})


def test_manifest_validation_rejects_noncanonical_output_root(tmp_path: Path) -> None:
    manifest = matrix.build_continuous_session_manifest(
        method_lock_path=CONFIG,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    jobs = [dict(job) for job in manifest["jobs"]]
    jobs[0] = {**jobs[0], "output_root": str(tmp_path / "other")}
    with pytest.raises(ValueError, match="path|identity"):
        matrix.validate_continuous_session_manifest({**manifest, "jobs": jobs})
