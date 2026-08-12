from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_d92_csoas_hard10_analysis as analysis  # noqa: E402


def _score(delta: float = 0.02) -> dict[str, object]:
    old = {f"tx_{i}": {"role": "target_old", "accuracy": 0.70 + delta} for i in range(6)}
    before_scene = {scene: {"query_count": 60} for scene in analysis.SCENES}
    after_scene = {
        scene: {
            "query_count": 80,
            "h_old_new": 0.70 + delta,
            "old_acc": 0.70 + delta,
            "seen_new_acc": 0.70 + delta,
            "new_to_old_rate": 0.10 - delta,
            "old_to_new_rate": 0.10 - delta,
        }
        for scene in analysis.SCENES
    }
    return {
        "candidate": analysis.CANDIDATE_ID,
        "truth_sidecar_sha256": "a" * 64,
        "before": {"old_acc": 0.70, "by_scenario": before_scene, "by_tx": old},
        "after": {
            "h_old_new": 0.70 + delta,
            "old_acc": 0.70 + delta,
            "seen_new_acc": 0.70 + delta,
            "by_tx": old,
            "by_scenario": after_scene,
            "query_macs": 3168,
            "after_state_bytes": 8583,
            "new_to_old_rate": 0.10 - delta,
            "old_to_new_rate": 0.10 - delta,
        },
        "old_forgetting_pp": -delta * 100.0,
    }


def test_score_metrics_and_strict_direction_are_eight_metric_and_k1_independent() -> None:
    metrics = analysis.compute_score_metrics(_score())
    assert analysis.EIGHT_PARETO_METRICS == (
        "h_old_new", "old_balanced_accuracy", "c_old_acc", "old_floor", "seen_new_acc",
        "average_forgetting", "new_to_old_rate", "old_to_new_rate",
    )
    assert metrics["old_class_count"] == 6
    assert metrics["query_macs"] == 3168
    baseline = analysis.compute_score_metrics(_score(0.0))
    deltas = analysis.strict_pareto_deltas(metrics, baseline)
    assert all(deltas[name] > 0 for name in analysis.EIGHT_PARETO_METRICS[:5])
    assert deltas["average_forgetting"] < 0
    assert deltas["new_to_old_rate"] < 0
    assert deltas["old_to_new_rate"] < 0


def test_resource_gate_uses_hard_limits_and_query_state_exact() -> None:
    baseline = [{"registration_wall_time_ns": 100_000_000, "registration_incremental_peak_working_set_bytes": 1_000_000}]
    candidate = [{"registration_wall_time_ns": 140_000_000, "registration_incremental_peak_working_set_bytes": 1_400_000}]
    result = analysis.evaluate_resource_gate(candidate, baseline, query_state_exact=True)
    assert result["hard_passed"] is True
    assert result["target_passed"] is False
    assert analysis.decide_verdict({
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": True,
        "stability": True,
        "resource_integrity": True,
        "resource_hard": True,
        "resource_target": False,
        "compute_reduction": True,
    }) == "REVISE_ONCE"


def test_component_fit_gate_uses_actual_receipt_not_mac_estimate() -> None:
    rows = [
        {"outer_role": "performance", "outer_key": "rx_x", "k_shot": 10, "fit_count": 2, "actual_component_inventory": {"actual_component_fit_count": 1}},
    ]
    result = analysis.evaluate_component_fit_reduction_gate(rows)
    assert result["passed"] is True
    assert result["rows"][0]["original_d92_total_component_fit_count"] == 88
    rows[0]["fit_count"] = 80
    assert analysis.evaluate_component_fit_reduction_gate(rows)["passed"] is False


def test_truth_binding_checks_actual_sidecar_hash(tmp_path: Path) -> None:
    truth = tmp_path / "jobs" / "outer" / "offline" / "scorer" / "truth_sidecar.json"
    truth.parent.mkdir(parents=True)
    truth.write_text("truth", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(truth.read_bytes()).hexdigest()
    job = {"outer_key": "outer", "truth_sidecar": str(truth), "truth_sidecar_sha256": digest}
    receipt = {"truth_sidecar_sha256": digest}
    score = {"truth_sidecar_sha256": digest}
    assert analysis.validate_truth_binding(score, receipt, job, truth) == digest
    with pytest.raises(analysis.D92CSOASHard10AnalysisError):
        analysis.validate_truth_binding(score, receipt, job, truth.with_name("other.json"))
