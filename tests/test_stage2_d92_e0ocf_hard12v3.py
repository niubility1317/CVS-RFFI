from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_e0ocf_hard12 import (
    ARM_ORDER,
    D92E0OCFHard12V3Error,
    HARD12_ROWS,
    SMOKE_OUTER_KEY,
    build_hard12v3_manifest,
    validate_hard12v3_manifest,
    validate_method_lock,
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
    assert manifest["smoke_outer_key"] == SMOKE_OUTER_KEY
    assert manifest["arms"] == list(ARM_ORDER)
    assert {job["arm_id"] for job in manifest["jobs"]} == set(ARM_ORDER)
    assert Counter(job["outer_role"] for job in manifest["jobs"]) == {
        "performance": 50,
        "liveness": 10,
    }
    assert sum(job["role"] == "primary" for job in manifest["jobs"]) == 12
    assert sum(job["role"] == "diagnostic_only" for job in manifest["jobs"]) == 12


def test_hard12v3_smoke_outer_is_the_first_frozen_liveness_row() -> None:
    assert SMOKE_OUTER_KEY == "rx_20_1__seed_713106__k_1__new_20"
    assert json.loads(METHOD_LOCK.read_text(encoding="utf-8"))["smoke_outer_key"] == SMOKE_OUTER_KEY
    matches = [row for row in HARD12_ROWS if row["outer_key"] == SMOKE_OUTER_KEY]
    assert len(matches) == 1
    assert matches[0]["role"] == "liveness"


def test_hard12v3_builder_rejects_missing_or_tampered_smoke_outer_key(tmp_path: Path) -> None:
    for value in (None, "rx_7_14__seed_713105__k_1__new_20"):
        lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
        if value is None:
            lock.pop("smoke_outer_key", None)
        else:
            lock["smoke_outer_key"] = value
        lock_path = tmp_path / ("lock_missing.json" if value is None else "lock_tampered.json")
        lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
        try:
            build_hard12v3_manifest(
                context_path=CONTEXT,
                method_lock_path=lock_path,
                output_root=tmp_path / ("matrix_missing" if value is None else "matrix_tampered"),
                require_package_files=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("builder accepted invalid smoke_outer_key")


def test_method_lock_validator_rejects_ocf_lambda_drift() -> None:
    """Would fail if a changed OCF scientific identity reused the frozen lock."""

    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    lock["arms"]["E0_OCF25"]["lambda"] = 0.50
    with pytest.raises(D92E0OCFHard12V3Error, match="method lock"):
        validate_method_lock(lock)


@pytest.mark.parametrize("tamper", ("outer", "role"))
def test_manifest_validator_rejects_tampered_outer_or_role(
    tmp_path: Path, tamper: str
) -> None:
    """Would fail if a noncanonical outer or arm role reached runner/analysis."""

    manifest = build_hard12v3_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    if tamper == "outer":
        manifest["selected_rows"][0]["receiver"] = "3-19"
    else:
        manifest["jobs"][0]["role"] = "primary"
    with pytest.raises(D92E0OCFHard12V3Error, match="manifest"):
        validate_hard12v3_manifest(
            manifest,
            expected_method_lock_sha256=manifest["method_lock_sha256"],
        )
