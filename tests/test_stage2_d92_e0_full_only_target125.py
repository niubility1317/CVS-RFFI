from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_e0_full_only_target125 import (
    ARM_ID,
    ARM_ORDER,
    CANDIDATE_ID,
    CONTEXT_SHA256,
    SCENES,
    SMOKE_OUTER_KEY,
    TARGET125_ROWS,
    build_target125_manifest,
    canonical_selection_sha256,
    validate_method_lock,
    validate_target125_manifest,
)


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d108_cbrrc_smme_target125_20260801_r3\artifacts\remote_r1\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_e0_full_only_target125_v1.json")


def test_target125_selection_is_complete_cartesian_and_deterministic() -> None:
    assert len(TARGET125_ROWS) == 125
    assert len(ARM_ORDER) == 1
    assert ARM_ID == "E0_FULL_ONLY"
    assert CANDIDATE_ID == "d92_e0d_e0_full_only"
    assert sum(row["role"] == "liveness" for row in TARGET125_ROWS) == 25
    assert sum(row["role"] == "performance" for row in TARGET125_ROWS) == 100
    assert len({row["outer_key"] for row in TARGET125_ROWS}) == 125
    assert canonical_selection_sha256()
    assert CONTEXT_SHA256 == "067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f"
    assert SMOKE_OUTER_KEY == "rx_20_1__seed_713106__k_1__new_20"


def test_target125_manifest_expands_one_arm_and_three_scenes(tmp_path: Path) -> None:
    manifest = build_target125_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["outer_count"] == 125
    assert manifest["job_count"] == 125
    assert manifest["scene_arm_count"] == 375
    assert manifest["scene_count"] == len(SCENES) == 3
    assert manifest["arms"] == [ARM_ID]
    assert manifest["candidate_ids"] == {ARM_ID: CANDIDATE_ID}
    assert {job["arm_id"] for job in manifest["jobs"]} == {ARM_ID}
    assert {tuple(job["scenarios"]) for job in manifest["jobs"]} == {SCENES}
    assert manifest["coverage"]["receiver_counts"] == {
        "20-1": 25,
        "3-19": 25,
        "7-14": 25,
        "7-7": 25,
        "8-8": 25,
    }
    assert manifest["coverage"]["seed_counts"] == {
        "713102": 25,
        "713103": 25,
        "713104": 25,
        "713105": 25,
        "713106": 25,
    }
    assert manifest["coverage"]["slice_counts"] == {
        "K1_new20": 25,
        "K5_new20": 25,
        "K10_new5": 25,
        "K10_new10": 25,
        "K10_new20": 25,
    }
    smoke = [job for job in manifest["jobs"] if job["outer_key"] == SMOKE_OUTER_KEY]
    assert len(smoke) == 1
    assert smoke[0]["arm_id"] == ARM_ID
    assert smoke[0]["k_shot"] == 1


def test_target125_manifest_rejects_extra_key_path_and_arm_drift(tmp_path: Path) -> None:
    manifest = build_target125_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    for mutated in (
        {**manifest, "unexpected": True},
        {**manifest, "output_root": str(tmp_path / "other")},
        {**manifest, "arms": ["D92_FULL"]},
    ):
        with pytest.raises(ValueError):
            validate_target125_manifest(mutated)


def test_method_lock_matches_target125_contract() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["arms"] == {ARM_ID: {"candidate_id": CANDIDATE_ID, "role": "primary"}}
    assert lock["primary_arm"] == ARM_ID
    assert lock["only_promotion_candidate"] == ARM_ID
    assert lock["matrix"]["outer_count"] == 125
    assert lock["matrix"]["job_count"] == 125
    assert lock["matrix"]["scene_arm_count"] == 375
    assert lock["matrix"]["shard_count"] == 8
    assert lock["selection_sha256"] == canonical_selection_sha256()
    assert validate_method_lock(lock)["claim_scope"] == "TARGET125_CONFIRMATION"

