from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cvsrffi.stage2_d92_be_hard12 import (
    CANONICAL_SELECTION_SHA256,
    HARD12_ROWS,
    build_hard12_manifest,
    canonical_selection_sha256,
)


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_qtie_target125_20260804_r3\artifacts\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_be_2x2_hard12_v1.json")


def test_canonical_selection_payload_has_one_reproducible_digest():
    assert canonical_selection_sha256() == (
        "95d94d586f5084d4982d67ec6402c4244f80e818ef3f95a5a03771085a6885a4"
    )
    assert CANONICAL_SELECTION_SHA256 == canonical_selection_sha256()


def test_hard12_manifest_joins_real_context_and_expands_four_arms(tmp_path: Path):
    manifest = build_hard12_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["job_count"] == 48
    assert manifest["scene_arm_count"] == 144
    assert manifest["outer_count"] == 12
    assert manifest["performance_outer_count"] == 10
    assert manifest["liveness_outer_count"] == 2
    assert manifest["selection_sha256"] == CANONICAL_SELECTION_SHA256
    assert {job["arm_id"] for job in manifest["jobs"]} == {
        "FULL",
        "B0",
        "E0",
        "B0E0",
    }
    counts = Counter(job["outer_role"] for job in manifest["jobs"])
    assert counts == {"performance": 40, "liveness": 8}
    for outer in HARD12_ROWS:
        selected = [
            job for job in manifest["jobs"] if job["outer_key"] == outer["outer_key"]
        ]
        assert len(selected) == 4
        assert len({job["planned_shard_index"] for job in selected}) == 1
        assert len({job["source_job_root"] for job in selected}) == 1


def test_k5_jobs_use_the_original_d92_k5_packages_not_context_k10_pool(tmp_path: Path):
    manifest = build_hard12_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    k5 = [job for job in manifest["jobs"] if job["k_shot"] == 5]
    assert len(k5) == 12
    assert all("__k_5__new_20" in job["source_job_root"] for job in k5)
    assert all(
        "__k_5__new_20" in job["packages"]["after_enrollment"]["package_root"]
        for job in k5
    )


def test_hard12_coverage_and_method_lock_are_exact(tmp_path: Path):
    manifest = build_hard12_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["coverage"] == {
        "receiver_counts": {"20-1": 3, "3-19": 3, "7-14": 2, "7-7": 2, "8-8": 2},
        "seed_counts": {
            "713102": 2,
            "713103": 2,
            "713104": 3,
            "713105": 3,
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
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["protocol_schema"] == "p2_min_v1"
    assert lock["selection_sha256"] == CANONICAL_SELECTION_SHA256
    assert lock["only_promotion_candidate"] == "B0E0"
    assert lock["strict_pareto_gate"]["mean_delta_h_min"] == 0.005
    assert lock["strict_pareto_gate"]["nonnegative_delta_h_outer_min"] == 8
    assert lock["strict_pareto_gate"]["median_wall_reduction_min"] == 0.40
    assert lock["strict_pareto_gate"]["median_incremental_peak_reduction_min"] == 0.40
