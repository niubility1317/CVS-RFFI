from __future__ import annotations

from collections import Counter
from pathlib import Path

from cvsrffi.stage2_d92_e0ocf_hard12 import (
    ARM_ORDER,
    HARD12_ROWS,
    build_hard12v3_manifest,
)


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_e0ocf_5arm_hard12v3_v1.json")


def test_hard12v3_freezes_five_arms_and_twelve_rows() -> None:
    assert ARM_ORDER == (
        "D92_FULL",
        "E0_FULL_ONLY",
        "E0_FIXED50",
        "E0_OCF25",
        "E0_OCF50",
    )
    assert len(HARD12_ROWS) == 12
    assert sum(row["role"] == "performance" for row in HARD12_ROWS) == 10
    assert sum(row["role"] == "liveness" for row in HARD12_ROWS) == 2
    assert {row["role"] for row in HARD12_ROWS} == {"performance", "liveness"}


def test_hard12v3_manifest_expands_sixty_jobs_and_marks_primary(tmp_path: Path) -> None:
    manifest = build_hard12v3_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["job_count"] == 60
    assert manifest["scene_arm_count"] == 180
    assert manifest["shard_count"] == 8
    assert manifest["primary_arm"] == "E0_OCF25"
    assert manifest["arms"] == list(ARM_ORDER)
    assert {job["arm_id"] for job in manifest["jobs"]} == set(ARM_ORDER)
    assert Counter(job["outer_role"] for job in manifest["jobs"]) == {
        "performance": 50,
        "liveness": 10,
    }
    assert sum(job["role"] == "primary" for job in manifest["jobs"]) == 12
    assert sum(job["role"] == "diagnostic_only" for job in manifest["jobs"]) == 12
