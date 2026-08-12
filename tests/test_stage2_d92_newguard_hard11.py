from __future__ import annotations

import json
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
