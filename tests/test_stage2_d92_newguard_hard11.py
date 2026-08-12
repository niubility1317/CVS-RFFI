from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_newguard_hard11 import (
    ARM_ID,
    ARM_ORDER,
    CANDIDATE_ID,
    CLAIM_SCOPE,
    HARD11_ROWS,
    LIVENESS_OUTER_KEY,
    SCENES,
    SHARD_COUNT,
    SMOKE_OUTER_KEY,
    build_hard11_manifest,
    canonical_selection_sha256,
    validate_hard11_manifest,
    validate_method_lock,
)


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d108_cbrrc_smme_target125_20260801_r3\artifacts\remote_r1\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_full_bidirectional_newguard_hard11_v1.json")

EXPECTED_PERFORMANCE = (
    "rx_7_7__seed_713106__k_10__new_5",
    "rx_7_7__seed_713104__k_5__new_20",
    "rx_7_7__seed_713103__k_10__new_5",
    "rx_8_8__seed_713103__k_5__new_20",
    "rx_8_8__seed_713103__k_10__new_5",
    "rx_8_8__seed_713106__k_5__new_20",
    "rx_7_14__seed_713104__k_10__new_10",
    "rx_3_19__seed_713102__k_10__new_5",
    "rx_7_7__seed_713105__k_10__new_20",
    "rx_7_7__seed_713104__k_10__new_5",
)


def test_hard11_identity_is_single_arm_with_k_gt_2_smoke_and_k1_liveness() -> None:
    assert tuple(row["outer_key"] for row in HARD11_ROWS[:-1]) == EXPECTED_PERFORMANCE
    assert HARD11_ROWS[-1]["outer_key"] == LIVENESS_OUTER_KEY
    assert sum(row["role"] == "performance" for row in HARD11_ROWS) == 10
    assert sum(row["role"] == "liveness" for row in HARD11_ROWS) == 1
    assert ARM_ORDER == (ARM_ID,)
    assert ARM_ID == "E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN"
    assert CANDIDATE_ID == "d92_e0_full_bidirectional_newguard_maxmin"
    assert CLAIM_SCOPE == "DEVELOPMENT_ONLY_HARD_SCREEN"
    assert SMOKE_OUTER_KEY == EXPECTED_PERFORMANCE[0]
    assert SHARD_COUNT == 8
    assert SCENES == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    assert canonical_selection_sha256()


def test_manifest_expands_to_11_jobs_33_scene_arms_and_8_shards(tmp_path: Path) -> None:
    manifest = build_hard11_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["outer_count"] == 11
    assert manifest["performance_outer_count"] == 10
    assert manifest["liveness_outer_count"] == 1
    assert manifest["job_count"] == 11
    assert manifest["scene_arm_count"] == 33
    assert manifest["scene_count"] == 3
    assert manifest["shard_count"] == 8
    assert manifest["smoke_outer_key"] == SMOKE_OUTER_KEY
    assert manifest["arms"] == [ARM_ID]
    assert manifest["candidate_ids"] == {ARM_ID: CANDIDATE_ID}
    assert {tuple(job["scenarios"]) for job in manifest["jobs"]} == {SCENES}
    assert all(re.fullmatch(r"[0-9a-f]{64}", str(job["truth_sidecar_sha256"])) for job in manifest["jobs"])
    assert validate_hard11_manifest(manifest)["job_count"] == 11


def test_manifest_rejects_identity_drift(tmp_path: Path) -> None:
    manifest = build_hard11_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    for mutated in (
        {**manifest, "unexpected": True},
        {**manifest, "job_count": 12},
        {**manifest, "smoke_outer_key": "rx_20_1__seed_713106__k_1__new_20"},
    ):
        with pytest.raises(ValueError):
            validate_hard11_manifest(mutated)
    broken_truth = {**manifest, "jobs": [dict(job) for job in manifest["jobs"]]}
    broken_truth["jobs"][0]["truth_sidecar"] = "wrong/truth_sidecar.json"
    with pytest.raises(ValueError, match="truth"):
        validate_hard11_manifest(broken_truth)
    broken_truth_hash = {**manifest, "jobs": [dict(job) for job in manifest["jobs"]]}
    broken_truth_hash["jobs"][0]["truth_sidecar_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="truth"):
        validate_hard11_manifest(broken_truth_hash)
    broken_seal = {**manifest, "jobs": [dict(job) for job in manifest["jobs"]]}
    broken_seal["jobs"][0]["packages"] = dict(broken_seal["jobs"][0]["packages"])
    broken_seal["jobs"][0]["packages"]["before_enrollment"] = dict(broken_seal["jobs"][0]["packages"]["before_enrollment"])
    broken_seal["jobs"][0]["packages"]["before_enrollment"]["detached_seal_path"] = "wrong/seal.json"
    with pytest.raises(ValueError, match="package"):
        validate_hard11_manifest(broken_seal)


def test_config_freezes_raw_score_and_per_old_sha_identities() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["historical_baseline"]["paired_rows_sha256"] == (
        "6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a"
    )
    assert lock["historical_baseline"]["per_old_class_rows_sha256"] == (
        "c0fc1e02b66b01d06da68bdd824594f3281e601d72b32726fa1e97a1e49788e6"
    )
    assert len(lock["historical_baseline"]["e0_raw_scores"]) == 11
    assert lock["historical_baseline"]["e0_raw_scores"][SMOKE_OUTER_KEY]["sha256"] == (
        "c4b90161d18482b0eedf978389557871cbf9676197f0a2889d547c95c76fbf97"
    )
    assert validate_method_lock(lock)["claim_scope"] == CLAIM_SCOPE


def test_config_freezes_deployment_backtrack_lock() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["deployment_backtrack"] == {
        "scales": [
            128.0,
            64.0,
            32.0,
            16.0,
            8.0,
            4.0,
            2.0,
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
            0.001953125,
            0.0009765625,
            0.00048828125,
            0.000244140625,
        ],
        "selection": "largest_safe_nonzero_after_actual_d42_roundtrip",
        "all_fail": "exact_e0_fallback",
        "protection_tolerance_formula": "1024*float32_eps",
        "protection_tolerance_value": 0.0001220703125,
        "negative_tail_accepted": False,
        "max_trust_region_fraction": 0.0128,
    }


def test_method_lock_rejects_deployment_backtrack_scale_drift(tmp_path: Path) -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    lock["deployment_backtrack"]["scales"][-1] = 0.0
    with pytest.raises(ValueError, match="method lock"):
        validate_method_lock(lock)
