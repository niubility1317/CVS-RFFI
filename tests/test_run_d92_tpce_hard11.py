from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_d92_tpce_hard11 as runner


SCENES = runner.SCENES


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(*, k_shot: int = 10) -> dict[str, object]:
    active = k_shot > 2
    prefix = "d92_e0d_tpce_"
    row: dict[str, object] = {
        "scenario": SCENES[0],
        "arm_id": runner.ARM_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "after_registered_d_mode_effective": "full_only",
        "after_state_postprocess_mode": "d42_tpce" if active else None,
        "after_total_component_fit_count": 2 if active else 3,
        "after_actual_component_inventory": {"actual_component_fit_count": 1 if active else 3},
        prefix + "active": active,
        prefix + "fallback_active": False,
        prefix + "fallback_reason": None if active else "K1_K2_EXACT_D92_FULL_ALIAS",
        prefix + "quantile": 0.2 if active else None,
        prefix + "quantile_method": "lower" if active else None,
        prefix + "state_postprocess_mode": "d42_tpce" if active else None,
        prefix + "direct_state_publish": True if active else None,
        prefix + "requantize_call_count": 0 if active else None,
        prefix + "e0_state_sha256": "a" * 64,
        prefix + "final_state_sha256": ("b" if active else "a") * 64,
        prefix + "changed_code2_count": 1 if active else 0,
        prefix + "requested_atomic_exchange_count": 1 if active else 0,
        prefix + "applied_atomic_exchange_count": 1 if active else 0,
        prefix + "aggregate_saturation_count": 0,
        prefix + "code1_byte_exact": True,
        prefix + "scale1_byte_exact": True,
        prefix + "scale2_byte_exact": True,
        prefix + "intercept_byte_exact": True,
        prefix + "log_diag_byte_exact": True,
        prefix + "old_tail_count_by_class": [1] * 6 if active else None,
        prefix + "pooled_new_tail_count": 1 if active else None,
        prefix + "tied_competitor_relation_count": 1 if active else None,
        prefix + "guard_tolerance": 1e-6 if active else None,
        prefix + "old_tail_gain_by_class": [0.1] * 6 if active else None,
        prefix + "old_tail_min_gain": 0.1 if active else None,
        prefix + "pooled_new_cross_tail_gain": 0.1 if active else None,
        prefix + "pooled_new_allclass_tail_gain": 0.1 if active else None,
        prefix + "old_to_new_hinge_delta": 0.0 if active else None,
        prefix + "new_to_old_hinge_delta": 0.0 if active else None,
        prefix + "support_guard_pass": True if active else None,
        prefix + "class_permutation_equivariant": True if active else None,
        prefix + "old_group_uniform_shift": False if active else None,
        prefix + "support_score_macs_upper_bound": 10.0 if active else None,
        prefix + "support_coordinate_comparisons_upper_bound": 10.0 if active else None,
        prefix + "support_macs_upper_bound": 10.0 if active else None,
        prefix + "support_transient_bytes_upper_bound": 10.0 if active else None,
        prefix + "persistent_state_bytes_delta": 0 if active else None,
        prefix + "component_fit_count": 0,
    }
    row.update({field: False for field in runner.QUERY_ZERO_FIELDS})
    return row


@pytest.mark.parametrize("k_shot", [10, 1])
def test_fit_audit_accepts_active_and_exact_alias(tmp_path: Path, k_shot: int) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(k_shot=k_shot), "scenario": scene} for scene in SCENES])
    runner._validate_fit_audit(path, k_shot=k_shot)


def test_fit_audit_rejects_tpce_code2_guard_drift(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row()
    row["d92_e0d_tpce_code1_byte_exact"] = False
    _write(path, [{**row, "scenario": scene} for scene in SCENES])
    with pytest.raises(runner.D92D92TPCEHard11RunnerError, match="fit audit"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_allows_numeric_fallback_diagnostics_but_no_applied_update(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row()
    prefix = "d92_e0d_tpce_"
    row[prefix + "active"] = False
    row[prefix + "fallback_active"] = True
    row[prefix + "fallback_reason"] = "aggregate_saturation"
    row[prefix + "final_state_sha256"] = row[prefix + "e0_state_sha256"]
    row[prefix + "changed_code2_count"] = 0
    row[prefix + "applied_atomic_exchange_count"] = 0
    row[prefix + "aggregate_saturation_count"] = 1
    _write(path, [{**row, "scenario": scene} for scene in SCENES])
    runner._validate_fit_audit(path, k_shot=10)
