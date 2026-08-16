from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from scripts import run_d92_ccoc_hard9_k1 as runner  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _reference_resources(*, peak: int = 10) -> dict[str, dict[str, int]]:
    return {
        scene: {
            "registration_wall_time_ns": 100_000_000,
            "registration_incremental_peak_working_set_bytes": peak,
            "query_macs": 11 * 288,
            "state_bytes": 11 * 289 * 4,
        }
        for scene in runner.SCENES
    }


def _row(
    scene: str,
    *,
    k_shot: int = 10,
    candidate_peak: int = 729_088,
    candidate_wall: int = 120_000_000,
    candidate_state_bytes: int = 11 * 289 * 4,
    candidate_query_macs: int = 11 * 288,
    postprocess_mode: object = None,
) -> dict[str, object]:
    active = k_shot > 2
    prefix = "d92_e0d_ccoc_"
    row: dict[str, object] = {
        "scenario": scene,
        "arm_id": runner.ARM_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "after_registered_d_mode_effective": "ccoc_full" if active else "d92_full_alias",
        "after_state_postprocess_mode": postprocess_mode,
        "after_total_component_fit_count": 2 if active else 3,
        "after_actual_component_inventory": {
            "actual_component_fit_count": 1 if active else 3,
            "full_component_fit_count": 1 if active else 3,
        },
        "registered_class_count": 11,
        "query_macs": candidate_query_macs,
        "after_state_bytes": candidate_state_bytes,
        "after_registration_resource": {
            "registration_wall_time_ns": candidate_wall,
            "registration_incremental_peak_working_set_bytes": candidate_peak,
        },
        prefix + "active": active,
        prefix + "fallback_active": False,
        prefix + "fallback_reason": None if active else "K1_K2_EXACT_D92_FULL_ALIAS",
        prefix + "candidate_attempt_fit_count": 1 if active else 0,
        prefix + "fallback_reference_fit_count": 0,
        prefix + "candidate_statistic_receipt_available": active,
        prefix + "paired_e0_codec_state_equal": None,
        prefix + "g0_eligible": active,
        prefix + "g0_block_reason": None if active else "K1_K2_EXACT_D92_FULL_ALIAS",
        prefix + "query_rows_used": 0,
    }
    for field in runner.QUERY_ZERO_FIELDS:
        row[field] = False
        row["d92_e0d_" + field] = False
        row[prefix + field] = False
    return row


def _fit_audit_rows(**kwargs: object) -> list[dict[str, object]]:
    return [_row(scene, **kwargs) for scene in runner.SCENES]


def test_fit_audit_accepts_ccoc_k_gt_2_and_k1_exact_alias(tmp_path: Path) -> None:
    for k_shot in (10, 1):
        path = tmp_path / f"fit_audit_k{k_shot}.json"
        _write(path, _fit_audit_rows(k_shot=k_shot))
        result = runner._validate_fit_audit(
            path,
            k_shot=k_shot,
            reference_resources=_reference_resources(),
        )
        assert result["scene_count"] == 3
        assert result["candidate_peak_hard_pass"] is True
        assert result["candidate_peak_target_pass"] is False


def test_candidate_peak_is_absolute_not_offset_by_e0_peak(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, _fit_audit_rows(candidate_peak=729_088))

    low_reference = runner._validate_fit_audit(
        path,
        k_shot=10,
        reference_resources=_reference_resources(peak=1),
    )
    high_reference = runner._validate_fit_audit(
        path,
        k_shot=10,
        reference_resources=_reference_resources(peak=9_999_999),
    )

    assert low_reference["candidate_peak_hard_pass"] is True
    assert high_reference["candidate_peak_hard_pass"] is True
    assert low_reference["candidate_peak_target_pass"] is False
    assert high_reference["candidate_peak_target_pass"] is False
    assert low_reference["candidate_peak_max_bytes"] == high_reference[
        "candidate_peak_max_bytes"
    ] == 729_088


@pytest.mark.parametrize(
    ("field", "row_kwargs", "reference"),
    (
        (
            "wall",
            {"candidate_wall": 150_000_001},
            _reference_resources(),
        ),
        (
            "ratio",
            {"candidate_wall": 140_000_000},
            {
                scene: {
                    **resource,
                    "registration_wall_time_ns": 90_000_000,
                }
                for scene, resource in _reference_resources().items()
            },
        ),
        (
            "peak",
            {"candidate_peak": 1_048_577},
            _reference_resources(),
        ),
        (
            "query MAC",
            {"candidate_query_macs": 11 * 288 + 1},
            _reference_resources(),
        ),
        (
            "state",
            {"candidate_state_bytes": 11 * 289 * 4 + 1},
            _reference_resources(),
        ),
        (
            "postprocess",
            {"postprocess_mode": "unexpected_postprocess"},
            _reference_resources(),
        ),
    ),
)
def test_fit_audit_rejects_each_single_scene_resource_or_integrity_drift(
    tmp_path: Path,
    field: str,
    row_kwargs: dict[str, int],
    reference: dict[str, dict[str, int]],
) -> None:
    rows = _fit_audit_rows()
    rows[1] = _row(runner.SCENES[1], **row_kwargs)
    path = tmp_path / f"fit_audit_{field.replace(' ', '_')}.json"
    _write(path, rows)

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match=field):
        runner._validate_fit_audit(
            path,
            k_shot=10,
            reference_resources=reference,
        )


def test_fit_audit_rejects_any_query_access(tmp_path: Path) -> None:
    rows = _fit_audit_rows()
    rows[0][runner.QUERY_ZERO_FIELDS[0]] = True
    path = tmp_path / "fit_audit_query.json"
    _write(path, rows)

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="query access"):
        runner._validate_fit_audit(
            path,
            k_shot=10,
            reference_resources=_reference_resources(),
        )


def test_parser_has_only_prepare_smoke_and_shard_execution_boundaries() -> None:
    parser = runner.parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert {"prepare", "smoke", "run-shard"} <= commands
    assert "truth" not in parser.format_help().lower()
    assert runner.SHARD_COUNT == 8


def test_systemic_failure_needs_two_distinct_pre_prediction_outers(
    tmp_path: Path,
) -> None:
    first = {
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "job_id": "first",
        "arm_id": runner.ARM_ID,
    }
    second = {
        "outer_key": "rx_7_7__seed_713103__k_10__new_5",
        "job_id": "second",
        "arm_id": runner.ARM_ID,
    }

    assert runner._record_pre_prediction_failure(tmp_path, first, "same-error") is False
    assert runner._record_pre_prediction_failure(tmp_path, first, "same-error") is False
    assert runner._record_pre_prediction_failure(tmp_path, second, "same-error") is True

    stop = json.loads(
        (tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").read_text(
            encoding="utf-8"
        )
    )
    assert stop["schema"] == "cvs.phase2.d92_ccoc_hard9_k1.systemic_failure.v1"
    assert stop["distinct_outer_count"] == 2
