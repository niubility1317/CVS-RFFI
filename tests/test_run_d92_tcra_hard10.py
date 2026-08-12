from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from scripts import run_d92_tcra_hard10 as runner  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(k_shot: int = 10) -> dict[str, object]:
    active = k_shot > 2
    prefix = "d92_e0d_tcra_"
    row: dict[str, object] = {
        "scenario": runner.SCENES[0],
        "arm_id": runner.ARM_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "after_registered_d_mode_effective": "full_only" if active else "d92_full_alias",
        "after_state_postprocess_mode": "d42_tcra" if active else None,
        "after_total_component_fit_count": 2 if active else 3,
        "after_actual_component_inventory": {"actual_component_fit_count": 1 if active else 3},
        prefix + "active": active,
        prefix + "fallback_active": False,
        prefix + "fallback_reason": None if active else "K1_K2_EXACT_D92_FULL_ALIAS",
        prefix + "final_gate_revision": "safe_directional_v2" if active else None,
        prefix + "state_postprocess_mode": "d42_tcra" if active else None,
        prefix + "direct_state_publish": True if active else None,
        prefix + "requantize_call_count": 0 if active else None,
        prefix + "e0_state_sha256": "a" * 64,
        prefix + "final_state_sha256": ("b" if active else "a") * 64,
        prefix + "changed_code2_count": 1 if active else 0,
        prefix + "requested_atomic_ascent_count": 1 if active else 0,
        prefix + "applied_atomic_ascent_count": 1 if active else 0,
        prefix + "generated_atomic_ascent_count": 3 if active else 0,
        prefix + "selected_atomic_ascent_count": 1 if active else 0,
        prefix + "rejected_atomic_ascent_count": 2 if active else 0,
        prefix + "prefix_guard_rejected_count": 1 if active else 0,
        prefix + "greedy_step_count": 2 if active else 0,
        prefix + "aggregate_saturation_count": 0,
        prefix + "code1_byte_exact": True if active else None,
        prefix + "scale1_byte_exact": True if active else None,
        prefix + "scale2_byte_exact": True if active else None,
        prefix + "intercept_byte_exact": True if active else None,
        prefix + "log_diag_byte_exact": True if active else None,
        prefix + "coef2_byte_exact": False if active else None,
        prefix + "modified_state_field_names": ["coef2_qint8"] if active else None,
        prefix + "old_tail_count_by_class": [1] * 6 if active else None,
        prefix + "old_tail_gain_by_class": [0.1] * 6 if active else None,
        prefix + "old_tail_min_gain": 0.1 if active else None,
        prefix + "old_tail_gain_sum": 0.6 if active else None,
        prefix + "old_tail_strict_positive_count": 6 if active else None,
        prefix + "pooled_new_cross_tail_gain": 0.1 if active else None,
        prefix + "pooled_new_allclass_tail_gain": 0.1 if active else None,
        prefix + "old_to_new_hinge_delta": 0.0 if active else None,
        prefix + "new_to_old_hinge_delta": 0.0 if active else None,
        prefix + "guard_tolerance": 1e-6 if active else None,
        prefix + "support_guard_pass": True if active else None,
        prefix + "safe_directional_pass": True if active else None,
        prefix + "class_permutation_equivariant": True if active else None,
        prefix + "row_permutation_invariant": True if active else None,
        prefix + "true_class_row_only": True if active else None,
        prefix + "competitor_code_decrement_count": 0 if active else None,
        prefix + "persistent_state_bytes_delta": 0 if active else None,
        prefix + "component_fit_count": 0,
        prefix + "support_score_macs_upper_bound": 10.0 if active else None,
        prefix + "support_coordinate_comparisons_upper_bound": 10.0 if active else None,
        prefix + "support_macs_upper_bound": 10.0 if active else None,
        prefix + "support_transient_bytes_upper_bound": 10.0 if active else None,
    }
    row.update({field: False for field in runner.QUERY_ZERO_FIELDS})
    return row


@pytest.mark.parametrize("k_shot", [10, 1])
def test_fit_audit_accepts_active_and_k1_alias(tmp_path: Path, k_shot: int) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(k_shot), "scenario": scene} for scene in runner.SCENES])
    runner._validate_fit_audit(path, k_shot=k_shot)


def test_fit_audit_accepts_skipped_saturated_atom_for_active_candidate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row(10)
    row["d92_e0d_tcra_aggregate_saturation_count"] = 1
    _write(path, [{**row, "scenario": scene} for scene in runner.SCENES])
    runner._validate_fit_audit(path, k_shot=10)


@pytest.mark.parametrize("saturation", [-1, 3])
def test_fit_audit_rejects_impossible_saturation_count(
    tmp_path: Path, saturation: int
) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row(10)
    row["d92_e0d_tcra_aggregate_saturation_count"] = saturation
    _write(path, [{**row, "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92TCRAHard10RunnerError):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_fallback_for_k_gt_2(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row(10)
    row["d92_e0d_tcra_active"] = False
    row["d92_e0d_tcra_fallback_active"] = True
    row["d92_e0d_tcra_fallback_reason"] = "support_guard_failed"
    row["d92_e0d_tcra_final_state_sha256"] = row["d92_e0d_tcra_e0_state_sha256"]
    _write(path, [{**row, "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92TCRAHard10RunnerError, match="did not activate"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_nonzero_query_access(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row(10)
    row["query_truth_access"] = True
    _write(path, [{**row, "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92TCRAHard10RunnerError, match="query access"):
        runner._validate_fit_audit(path, k_shot=10)


def test_runner_context_binds_real_executor_and_tcra_exception() -> None:
    base = runner._base_runner
    original = base.D92ParetoDistillHard11RunnerError
    with runner._runner_context():
        assert base.build_hard11_manifest is runner.build_hard10_manifest
        assert base._validate_shared_smoke is runner._validate_shared_smoke
        assert base.D92ParetoDistillHard11RunnerError is runner.D92TCRAHard10RunnerError
        assert base.D92ParetoDistillHard11Error is runner.D92TCRAHard10RunnerError
    assert base.D92ParetoDistillHard11RunnerError is original


def test_shared_failure_receipts_are_rewritten_to_tcra(tmp_path: Path) -> None:
    stop = tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"
    record = (
        tmp_path
        / "systemic_pre_prediction_failures"
        / ("f" * 64)
        / "outer"
        / "job.json"
    )
    _write(stop, {"schema": "cvs.phase2.d92_pareto_distill_hard11.systemic_stop.v1"})
    _write(record, {"schema": "cvs.phase2.d92_pareto_distill_hard11.pre_prediction_failure.v1"})
    runner._rewrite_shared_failure_evidence(tmp_path)
    assert json.loads(stop.read_text())["schema"] == "cvs.phase2.d92_tcra_hard10.systemic_stop.v1"
    assert json.loads(record.read_text())["schema"] == "cvs.phase2.d92_tcra_hard10.pre_prediction_failure.v1"
