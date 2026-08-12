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
        "after_registered_d_mode_effective": (
            "full_only" if active else "d92_full_alias"
        ),
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
        prefix + "generated_atomic_exchange_count": 3 if active else 0,
        prefix + "selected_atomic_exchange_count": 1 if active else 0,
        prefix + "rejected_atomic_exchange_count": 2 if active else 0,
        prefix + "greedy_step_count": 1 if active else 0,
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


def test_fit_audit_rejects_tpce_greedy_subset_count_drift(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row()
    row["d92_e0d_tpce_rejected_atomic_exchange_count"] = 3
    _write(path, [{**row, "scenario": scene} for scene in SCENES])
    with pytest.raises(runner.D92D92TPCEHard11RunnerError, match="atomic"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_k_greater_than_two_numeric_fallback(tmp_path: Path) -> None:
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
    with pytest.raises(
        runner.D92D92TPCEHard11RunnerError,
        match="did not activate",
    ):
        runner._validate_fit_audit(path, k_shot=10)


def test_shared_smoke_schema_translation_is_in_memory_only() -> None:
    tpce = {
        "schema": "cvs.phase2.d92_tpce_hard11.smoke_receipt.v1",
        "status": "D92_TPCE_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "marker": 92,
    }
    translated = runner._base_smoke_receipt_view(tpce)
    assert translated == {
        "schema": "cvs.phase2.d92_pareto_distill_hard11.smoke_receipt.v1",
        "status": (
            "D92_PARETO_DISTILL_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
        ),
        "marker": 92,
    }
    assert tpce["schema"] == "cvs.phase2.d92_tpce_hard11.smoke_receipt.v1"


def test_shard_schema_rewrite_touches_only_its_owned_outputs(tmp_path: Path) -> None:
    output = tmp_path / "run"
    owned = output / "jobs" / "owned"
    other = output / "jobs" / "other"
    summary = output / "summaries" / "shard_0.json"
    smoke = output / "smoke" / "smoke_receipt.json"
    base_receipt = {
        "schema": "cvs.phase2.d92_pareto_distill_hard11.job_receipt.v1",
        "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
    }
    _write(owned / "job_receipt.json", base_receipt)
    _write(other / "job_receipt.json", base_receipt)
    _write(
        summary,
        {
            "schema": "cvs.phase2.d92_pareto_distill_hard11.shard_summary.v1",
            "status": "PASS",
        },
    )
    _write(
        smoke,
        {
            "schema": "cvs.phase2.d92_tpce_hard11.smoke_receipt.v1",
            "status": "D92_TPCE_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        },
    )
    manifest = {
        "output_root": str(output),
        "jobs": [
            {"planned_shard_index": 0, "output_root": str(owned)},
            {"planned_shard_index": 1, "output_root": str(other)},
        ],
    }
    runner._rewrite_shard_output(manifest, shard_index=0)
    assert "d92_tpce_hard11" in json.loads(
        (owned / "job_receipt.json").read_text(encoding="utf-8")
    )["schema"]
    assert "d92_pareto_distill_hard11" in json.loads(
        (other / "job_receipt.json").read_text(encoding="utf-8")
    )["schema"]
    assert "d92_tpce_hard11" in json.loads(
        summary.read_text(encoding="utf-8")
    )["schema"]
    assert "d92_tpce_hard11" in json.loads(
        smoke.read_text(encoding="utf-8")
    )["schema"]


def test_runner_context_delegates_manifest_artifact_check_without_recursion(
    monkeypatch,
) -> None:
    observed: list[object] = []
    sentinel = {"jobs": []}

    monkeypatch.setattr(
        runner,
        "_BASE_VERIFY_MANIFEST_ARTIFACTS",
        lambda manifest: observed.append(manifest),
    )
    with runner._runner_context():
        runner._base_runner._verify_manifest_artifacts(sentinel)
    assert observed == [sentinel]
