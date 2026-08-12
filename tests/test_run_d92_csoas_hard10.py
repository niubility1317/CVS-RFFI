from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from scripts import run_d92_csoas_hard10 as runner  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(k_shot: int = 10, *, fallback: bool = False, codec_retry: int = 0, class_count: int = 11) -> dict[str, object]:
    active = k_shot > 2 and not fallback
    prefix = "d92_csoas_"
    row: dict[str, object] = {
        "scenario": runner.SCENES[0],
        "arm_id": runner.ARM_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "after_registered_d_mode_effective": "csoas_full" if k_shot > 2 else "d92_full_alias",
        "after_total_component_fit_count": 2 if k_shot > 2 else 3,
        "after_actual_component_inventory": {
            "actual_component_fit_count": 1 if k_shot > 2 else 3,
            "full_component_fit_count": 1 if k_shot > 2 else 3,
            "block3_component_fit_count": 0,
        },
        prefix + "active": active,
        prefix + "fallback_active": fallback,
        prefix + "fallback_reason": "NUMERIC_FALLBACK_EXACT_E0" if fallback else (None if k_shot > 2 else "K1_K2_EXACT_D92_FULL_ALIAS"),
        prefix + "candidate_attempt_fit_count": 1 if k_shot > 2 else 0,
        prefix + "fallback_reference_fit_count": 1 if fallback else 0,
        prefix + "candidate_statistic_receipt_available": active,
        prefix + "fallback_reference_full_head_byte_exact": True if fallback else None,
        prefix + "paired_e0_codec_state_equal": None,
        prefix + "g0_eligible": False,
        prefix + "g0_block_reason": "PENDING_DEPLOYED_CODEC_PAIRED_E0" if k_shot > 2 and not fallback else ("NUMERIC_FALLBACK_EXACT_E0" if fallback else "K1_K2_EXACT_D92_FULL_ALIAS"),
        prefix + "codec_retry_count": codec_retry,
        prefix + "query_rows_used": 0,
        "query_macs": class_count * 288,
        "after_state_bytes": 8583,
        "registered_class_count": class_count,
    }
    row["d92_e0d_csoas_g0_eligible"] = row.pop(prefix + "g0_eligible")
    row["d92_e0d_csoas_g0_block_reason"] = row.pop(prefix + "g0_block_reason")
    for field in runner.QUERY_ZERO_FIELDS:
        row[field] = False
    for field in runner.CSOAS_QUERY_ZERO_FIELDS:
        row[field] = False
    return row


@pytest.mark.parametrize("k_shot", [10, 1])
def test_fit_audit_accepts_real_csoas_active_and_exact_k1_alias(tmp_path: Path, k_shot: int) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(k_shot), "scenario": scene} for scene in runner.SCENES])
    runner._validate_fit_audit(path, k_shot=k_shot)


def test_fit_audit_rejects_numeric_fallback_for_formal_k_gt_2(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(10, fallback=True), "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="fallback"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_codec_retry_for_formal_k_gt_2(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(10, codec_retry=1), "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="retry"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_non_full1_two_state_inventory(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    rows = []
    for scene in runner.SCENES:
        row = _row(10)
        row["scenario"] = scene
        row["after_actual_component_inventory"] = {
            **row["after_actual_component_inventory"],
            "full_component_fit_count": 2,
        }
        rows.append(row)
    _write(path, rows)
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="FULL1"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_any_query_access(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row(10)
    row[runner.QUERY_ZERO_FIELDS[0]] = True
    _write(path, [{**row, "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="query access"):
        runner._validate_fit_audit(path, k_shot=10)


@pytest.mark.parametrize("class_count", [16, 26])
def test_fit_audit_accepts_hard9_new_class_counts(tmp_path: Path, class_count: int) -> None:
    path = tmp_path / f"fit_audit_{class_count}.json"
    _write(path, [{**_row(10, class_count=class_count), "scenario": scene} for scene in runner.SCENES])
    runner._validate_fit_audit(path, k_shot=10)


def test_runner_parser_exposes_prepare_smoke_and_eight_shard_commands() -> None:
    parser = runner.parser()
    assert set(parser._subparsers._group_actions[0].choices) >= {"prepare", "truth-free-smoke", "run-shard"}
    assert runner.SHARD_COUNT == 8
    assert runner._is_full_matrix({"job_count": 10, "jobs": [{}] * 10})
