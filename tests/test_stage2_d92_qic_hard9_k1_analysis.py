from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_d92_qic_hard9_k1_analysis as analysis  # noqa: E402


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
        "per_old_class_floor_before": 0.70,
    }


def test_qic_exports_truth_last_eight_metric_contract() -> None:
    assert analysis.EIGHT_PARETO_METRICS == (
        "h_old_new",
        "old_balanced_accuracy",
        "c_old_acc",
        "old_floor",
        "seen_new_acc",
        "average_forgetting",
        "new_to_old_rate",
        "old_to_new_rate",
    )
    assert analysis.VERDICTS == (
        "REJECT_ROUTE",
        "REVISE_ONCE",
        "ADVANCE_TO_TARGET125_CANDIDATE",
    )
    assert analysis.CANDIDATE_ID
    assert analysis.ARM_ID


def test_qic_score_metrics_use_strict_eight_directions() -> None:
    candidate = analysis.compute_score_metrics(_score())
    baseline = analysis.compute_score_metrics(_score(0.0))
    deltas = analysis.strict_pareto_deltas(candidate, baseline)
    assert candidate["old_class_count"] == 6
    assert all(deltas[name] > 0 for name in analysis.EIGHT_PARETO_METRICS[:5])
    assert deltas["average_forgetting"] < 0
    assert deltas["new_to_old_rate"] < 0
    assert deltas["old_to_new_rate"] < 0


def test_qic_hard_resource_gate_keeps_absolute_peak_and_target_separate() -> None:
    rows = [
        {
            "outer_key": "outer",
            "scenario": scene,
            "candidate_wall_ns": 100_000_000,
            "wall_ratio": 1.0,
            "candidate_peak_bytes": 729_088,
            "wall_hard_pass": True,
            "ratio_hard_pass": True,
            "peak_hard_pass": True,
            "wall_target_pass": True,
            "ratio_target_pass": True,
            "peak_target_pass": False,
            "query_state_exact": True,
        }
        for scene in analysis.SCENES
    ]
    result = analysis.evaluate_resource_gate(rows)
    assert result["hard_passed"] is True
    assert result["target_passed"] is False
    assert result["candidate_peak_max_bytes"] == 729_088


def test_qic_fit_audit_namespace_is_selected_from_receipt_shape() -> None:
    assert analysis._fit_audit_prefix({"d92_e0d_qic_active": True}) == "d92_e0d_qic_"
    assert analysis._fit_audit_prefix({"d92_e0d_ccoc_active": True}) == "d92_e0d_ccoc_"


def test_qic_k1_is_liveness_and_hard_failure_precedes_revision() -> None:
    gates = {
        "complete_artifact_closure": True,
        "performance_outer_closure": True,
        "all_strict_pareto": True,
        "all_magnitude": False,
        "stability": True,
        "resource_integrity": True,
        "resource_hard": False,
        "resource_target": False,
    }
    assert analysis.decide_verdict(gates) == "REJECT_ROUTE"
    gates["resource_hard"] = True
    assert analysis.decide_verdict(gates) == "REVISE_ONCE"
    gates["all_magnitude"] = True
    gates["resource_target"] = True
    assert analysis.decide_verdict(gates) == "ADVANCE_TO_TARGET125_CANDIDATE"


def test_qic_output_is_exclusive_seven_file_package(tmp_path: Path) -> None:
    result = {
        "schema": "cvs.phase2.d92_qic_hard9_k1.analysis.v1",
        "verdict": "REVISE_ONCE",
        "status": "ANALYZED",
        "gate_state": {},
        "gates": {},
        "aggregate": {},
        "paired_rows": [],
        "per_old_class_rows": [],
        "scenario_rows": [],
        "liveness_rows": [],
    }
    output_root = tmp_path / "analysis"
    paths = analysis.write_analysis_outputs(result, output_root)
    assert set(paths) == {
        "summary.json",
        "gates.json",
        "paired_rows.csv",
        "per_old_class_rows.csv",
        "scenario_rows.csv",
        "liveness_rows.csv",
        "analysis.md",
    }
    with pytest.raises(analysis.D92QICHard9K1AnalysisError, match="overwrite"):
        analysis.write_analysis_outputs(result, output_root)


def test_qic_score_binding_accepts_retrieved_remote_path_suffix(tmp_path: Path) -> None:
    outer = "rx_7_7__seed_713103__k_10__new_5"
    job_root = tmp_path / "jobs" / outer / analysis.ARM_ID
    job_root.mkdir(parents=True)
    local_binding = job_root / "score_binding.json"
    local_binding.write_text("{}", encoding="utf-8")
    remote_binding = (
        "/home/user/project/runs/qic/jobs/"
        f"{outer}/{analysis.ARM_ID}/score_binding.json"
    )

    assert analysis._resolve_score_binding_path(remote_binding, job_root) == local_binding.resolve()
    with pytest.raises(analysis.D92QICHard9K1AnalysisError, match="path drift"):
        analysis._resolve_score_binding_path(
            f"/home/user/project/runs/qic/jobs/other/{analysis.ARM_ID}/score_binding.json",
            job_root,
        )


def test_qic_markdown_exposes_all_four_da_registration_states() -> None:
    result = {
        "verdict": "REJECT_ROUTE",
        "status": "REJECTED_EVIDENCE_CLOSURE",
        "aggregate": {},
        "gates": {},
    }
    markdown = analysis.render_analysis_markdown(result)
    for state in ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"):
        assert state in markdown
    assert set(analysis._four_state_metrics({})) == {
        "DA0_REG0",
        "DA1_REG0",
        "DA0_REG1",
        "DA1_REG1",
    }
