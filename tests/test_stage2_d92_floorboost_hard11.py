from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_floorboost_hard11 import (
    ARM_ID,
    ARM_ORDER,
    CANDIDATE_ID,
    CLAIM_SCOPE,
    HARD11_ROWS,
    MARGIN_QUANTILE,
    RETENTION_BIAS_KAPPA,
    SCENES,
    SMOKE_OUTER_KEY,
    build_hard11_manifest,
    canonical_selection_sha256,
    validate_hard11_manifest,
    validate_method_lock,
)


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d108_cbrrc_smme_target125_20260801_r3\artifacts\remote_r1\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_full_maxmin_floorboost_hard11_v1.json")


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


def test_hard11_selection_is_exact_single_arm_and_three_scene() -> None:
    assert tuple(row["outer_key"] for row in HARD11_ROWS[:-1]) == EXPECTED_PERFORMANCE
    assert HARD11_ROWS[-1]["outer_key"] == SMOKE_OUTER_KEY
    assert sum(row["role"] == "performance" for row in HARD11_ROWS) == 10
    assert sum(row["role"] == "liveness" for row in HARD11_ROWS) == 1
    assert ARM_ORDER == (ARM_ID,)
    assert ARM_ID == "E0_FULL_MAXMIN_FLOORBOOST"
    assert CANDIDATE_ID == "d92_e0_full_maxmin_floorboost"
    assert CLAIM_SCOPE == "DEVELOPMENT_ONLY_FLOOR_HARD_SCREEN"
    assert MARGIN_QUANTILE == 0.20
    assert RETENTION_BIAS_KAPPA == 0.35
    assert canonical_selection_sha256()


def test_hard11_manifest_expands_to_11_jobs_and_33_scene_arms(tmp_path: Path) -> None:
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
    assert manifest["arms"] == [ARM_ID]
    assert manifest["candidate_ids"] == {ARM_ID: CANDIDATE_ID}
    assert {job["arm_id"] for job in manifest["jobs"]} == {ARM_ID}
    assert {tuple(job["scenarios"]) for job in manifest["jobs"]} == {SCENES}
    assert validate_hard11_manifest(manifest)["job_count"] == 11


def test_hard11_manifest_rejects_identity_drift(tmp_path: Path) -> None:
    manifest = build_hard11_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    for mutated in (
        {**manifest, "unexpected": True},
        {**manifest, "job_count": 12},
        {**manifest, "arms": ["E0_FULL_ONLY"]},
    ):
        with pytest.raises(ValueError):
            validate_hard11_manifest(mutated)


def test_hard11_method_lock_matches_frozen_contract() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["arms"] == {
        ARM_ID: {
            "candidate_id": CANDIDATE_ID,
            "role": "primary",
            "contrast_lambda": 0.25,
            "margin_quantile": 0.20,
            "quantile_method": "lower",
            "retention_bias_kappa": 0.35,
        }
    }
    assert lock["matrix"] == {
        "outer_count": 11,
        "performance_outer_count": 10,
        "liveness_outer_count": 1,
        "job_count": 11,
        "scene_count": 3,
        "scene_arm_count": 33,
        "shard_count": 8,
    }
    assert lock["selection_sha256"] == canonical_selection_sha256()
    assert validate_method_lock(lock)["claim_scope"] == CLAIM_SCOPE
