from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cvsrffi.full_ablation_spec import build_phase1_t1_rows
from scripts.run_full_ablation_phase1_t1 import (
    Phase1RunnerError,
    build_phase1_command,
    normalize_exception_fingerprint,
    validate_phase1_row_completion,
    validate_phase1_release_plan,
)


def _plan() -> dict:
    plan = {
        "schema": "cvs.full_ablation.plan.v1",
        "design_id": "cvs_full_ablation_phase1_phase2_20260728",
        "phase": "phase1",
        "stage": "t1",
        "git_commit": "a" * 40,
        "formal_launch_authority": False,
        "seed_registry_sha256": "c" * 64,
        "rows": build_phase1_t1_rows(
            [7281101, 7281102, 7281103, 7281104, 7281105],
            git_commit="a" * 40,
        ),
    }
    return plan


def _seal_for_test(plan: dict) -> None:
    content = {
        key: value
        for key, value in plan.items()
        if key != "sealed_content_sha256"
    }
    plan["sealed_content_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_unsealed_plan_is_dry_run_valid_but_not_launchable() -> None:
    plan = _plan()
    validate_phase1_release_plan(plan, require_launch_authority=False)
    with pytest.raises(Phase1RunnerError, match="launch authority"):
        validate_phase1_release_plan(plan, require_launch_authority=True)


def test_sealed_plan_requires_review_and_local_verified_executors() -> None:
    plan = _plan()
    plan["formal_launch_authority"] = True
    plan["run_id"] = "cvs_full_ablation_phase1_t1_20260728_v1"
    plan["independent_review"] = {"p0_count": 0, "p1_count": 0}
    _seal_for_test(plan)
    with pytest.raises(Phase1RunnerError, match="LOCAL_VERIFIED"):
        validate_phase1_release_plan(plan, require_launch_authority=True)
    for row in plan["rows"]:
        row["executor_status"] = "LOCAL_VERIFIED"
    _seal_for_test(plan)
    validate_phase1_release_plan(plan, require_launch_authority=True)


def test_command_binds_exact_row_identity(tmp_path) -> None:
    row = _plan()["rows"][0]
    command = build_phase1_command(
        row,
        run_id="run-v1",
        python_executable="/envs/ssr-gpu/bin/python",
        train_script=tmp_path / "train_ssdg.py",
        wisig_pkl=tmp_path / "wisig.pkl",
        output_dir=tmp_path / row["row_key"],
        sealed_plan_sha256="d" * 64,
        seed_registry_sha256="c" * 64,
    )
    joined = " ".join(command)
    assert "--formal_ablation true" in joined
    assert f"--ablation_id {row['ablation_id']}" in joined
    assert f"--seed {row['train_seed']}" in joined
    assert f"--git_commit {row['git_commit']}" in joined
    assert f"--row_key {row['row_key']}" in joined
    assert f"--sealed_plan_sha256 {'d' * 64}" in joined
    assert f"--seed_registry_sha256 {'c' * 64}" in joined
    assert "--device cuda:0" in joined


def test_plan_rejects_row_commit_or_arm_drift() -> None:
    plan = _plan()
    drift = copy.deepcopy(plan)
    drift["rows"][0]["git_commit"] = "b" * 40
    with pytest.raises(Phase1RunnerError, match="Git commit"):
        validate_phase1_release_plan(drift, require_launch_authority=False)
    drift = copy.deepcopy(plan)
    drift["rows"][0]["ablation_id"] = "P1-UNKNOWN"
    with pytest.raises(Phase1RunnerError, match="arm set"):
        validate_phase1_release_plan(drift, require_launch_authority=False)


def test_exception_fingerprint_ignores_paths_addresses_and_numbers() -> None:
    first = "RuntimeError at C:\\run\\row1.py:123 address 0xABC value 9"
    second = "RuntimeError at D:\\other\\row2.py:987 address 0xDEF value 42"
    assert normalize_exception_fingerprint(first) == normalize_exception_fingerprint(
        second
    )


def test_completion_receipt_binds_row_plan_split_and_terminal(tmp_path) -> None:
    plan = _plan()
    plan["formal_launch_authority"] = True
    plan["run_id"] = "phase1-v1"
    plan["independent_review"] = {"p0_count": 0, "p1_count": 0}
    for row in plan["rows"]:
        row["executor_status"] = "LOCAL_VERIFIED"
        row["config_hash"] = "d" * 64
    _seal_for_test(plan)
    row = plan["rows"][0]
    output = tmp_path / row["row_key"]
    output.mkdir()
    terminal = {"status": "COMPLETE"}
    terminal_path = output / "phase1_terminal_status.json"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    resource_path = output / "phase1_resource_summary.json"
    resource_path.write_text(
        json.dumps({"wall_time_seconds": 1.0}),
        encoding="utf-8",
    )
    receipt = {
        "run_id": plan["run_id"],
        "row_key": row["row_key"],
        "ablation_id": row["ablation_id"],
        "train_seed": row["train_seed"],
        "git_commit": plan["git_commit"],
        "sealed_plan_sha256": plan["sealed_content_sha256"],
        "seed_registry_sha256": plan["seed_registry_sha256"],
        "resolved_config_hash": row["config_hash"],
        "method_config_hash": row["method_config_hash"],
        "terminal_manifest_sha256": hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest(),
        "resource_summary_sha256": hashlib.sha256(
            resource_path.read_bytes()
        ).hexdigest(),
        "source_split_receipt": {
            "split_manifest_sha256": "e" * 64,
            "source_target_receiver_overlap_count": 0,
        },
        "terminal_status": "COMPLETE",
        "exit_code": 0,
    }
    (output / "phase1_training_completion_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    assert (
        validate_phase1_row_completion(
            row=row,
            plan=plan,
            output_dir=output,
            return_code=0,
        )["row_key"]
        == row["row_key"]
    )
