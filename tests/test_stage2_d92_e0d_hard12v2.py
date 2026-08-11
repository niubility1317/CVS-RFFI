from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

from cvsrffi.stage2_d92_e0d_hard12 import (
    ARM_ORDER,
    CANONICAL_SELECTION_SHA256,
    HARD12_ROWS,
    build_hard12v2_manifest,
    canonical_selection_sha256,
)
from cvsrffi.stage2_d92_e0d_hard12 import SELECTION_PAYLOAD, _canonical_bytes


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_e0d_5arm_hard12v2_v1.json")


def test_hard12v2_selection_digest_and_rows_are_frozen() -> None:
    assert canonical_selection_sha256() == (
        "2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a"
    )
    assert CANONICAL_SELECTION_SHA256 == canonical_selection_sha256()
    assert hashlib.sha256(_canonical_bytes(SELECTION_PAYLOAD)).hexdigest() == (
        "2e3b3333a4a325bd0443a31065d3340d6a650a3e89620951a786637e6bce8d3a"
    )
    assert len(HARD12_ROWS) == 12
    assert sum(row["role"] == "performance" for row in HARD12_ROWS) == 10
    assert sum(row["role"] == "liveness" for row in HARD12_ROWS) == 2
    assert set(ARM_ORDER) == {
        "D92_FULL",
        "E0_FUSION",
        "E0_FULL_ONLY",
        "E0_BLOCK_ONLY",
        "E0_FIXED50",
    }


def test_hard12v2_manifest_joins_context_and_expands_five_arms(tmp_path: Path) -> None:
    manifest = build_hard12v2_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["schema"] == "cvs.phase2.d92_e0d_hard12v2.matrix.v1"
    assert manifest["job_count"] == 60
    assert manifest["scene_arm_count"] == 180
    assert manifest["outer_count"] == 12
    assert manifest["performance_outer_count"] == 10
    assert manifest["liveness_outer_count"] == 2
    assert manifest["selection_sha256"] == CANONICAL_SELECTION_SHA256
    assert {job["arm_id"] for job in manifest["jobs"]} == set(ARM_ORDER)
    counts = Counter(job["outer_role"] for job in manifest["jobs"])
    assert counts == {"performance": 50, "liveness": 10}
    for outer in HARD12_ROWS:
        selected = [
            job for job in manifest["jobs"] if job["outer_key"] == outer["outer_key"]
        ]
        assert len(selected) == 5
        assert len({job["planned_shard_index"] for job in selected}) == 1
        assert len({job["source_job_root"] for job in selected}) == 1


def test_hard12v2_k5_jobs_use_original_d92_k5_packages(tmp_path: Path) -> None:
    manifest = build_hard12v2_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    k5 = [job for job in manifest["jobs"] if job["k_shot"] == 5]
    assert len(k5) == 15
    assert all("__k_5__new_20" in job["source_job_root"] for job in k5)
    assert all(
        "__k_5__new_20" in job["packages"]["after_enrollment"]["package_root"]
        for job in k5
    )


def test_hard12v2_coverage_has_zero_hard12_v1_outer_intersection(
    tmp_path: Path,
) -> None:
    manifest = build_hard12v2_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["coverage"] == {
        "receiver_counts": {"20-1": 3, "3-19": 3, "7-14": 2, "7-7": 2, "8-8": 2},
        "seed_counts": {
            "713102": 2,
            "713103": 3,
            "713104": 3,
            "713105": 2,
            "713106": 2,
        },
        "slice_counts": {
            "K1_new20": 2,
            "K5_new20": 3,
            "K10_new5": 2,
            "K10_new10": 2,
            "K10_new20": 3,
        },
    }
    from cvsrffi.stage2_d92_be_hard12 import HARD12_ROWS as V1_ROWS

    assert {
        row["outer_key"] for row in HARD12_ROWS
    }.isdisjoint(row["outer_key"] for row in V1_ROWS)
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["schema"] == "cvs.phase2.d92_e0d.method_lock.v1"
    assert lock["selection_sha256"] == CANONICAL_SELECTION_SHA256
    assert lock["only_promotion_candidate"] == "E0_FULL_ONLY"
    assert lock["strict_geometry_gate"]["mean_delta_h_vs_d92_full_min"] == 0.005
    assert lock["strict_geometry_gate"]["nonnegative_delta_h_vs_d92_full_outer_min"] == 8
