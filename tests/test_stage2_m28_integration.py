from __future__ import annotations

import importlib.util

from cvsrffi.stage2_m24_row_executor import d1_overlay_from_base_cache, execute_m24_row
from cvsrffi.stage2_m28_local_flip_risk import C1, C2, m28_arm_config_hash
from test_stage2_m24_integration import _caches


def test_m28_lifecycle_scripts_are_available() -> None:
    for name in (
        "scripts.run_m28_local_flip_risk_matrix",
        "scripts.score_m28_local_flip_risk_matrix",
        "scripts.summarize_m28_local_flip_risk_matrix",
    ):
        assert importlib.util.find_spec(name) is not None


def test_m28_runner_freezes_four_arm_screen_and_full125_sizes() -> None:
    from scripts import run_m28_local_flip_risk_matrix as runner

    assert runner.EVIDENCE_ARMS == (runner.D1, runner.B3, runner.C1, runner.C2)
    screen = runner.matrix_spec("screen")
    full = runner.matrix_spec("full125")
    assert screen["paired_input_identity_count"] == 4
    assert screen["expected_method_rows"] == 16
    assert full["paired_input_identity_count"] == 125
    assert full["expected_method_rows"] == 500


def test_m28_summary_gate_requires_gain_over_b0_and_b3() -> None:
    from scripts import summarize_m28_local_flip_risk_matrix as summarizer

    def arm_row(arm, h, min_old=0.30, min_new=0.25):
        return {
            "arm": arm,
            "metrics": {
                "H": {"pooled_query_weighted_mean": h},
                "min_old": {"pooled_query_weighted_mean": min_old},
                "min_new": {"pooled_query_weighted_mean": min_new},
            },
        }

    result = {
        "arm_summary": [
            arm_row("M24-D1-COMPILE-PARITY", 0.50),
            arm_row("M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL", 0.5018),
            arm_row(C1, 0.5021),
            arm_row(C2, 0.5019),
        ],
        "help_harm": {
            "overall": [
                {"candidate_arm": C1, "N_help": 12, "N_harm": 4},
                {"candidate_arm": C2, "N_help": 8, "N_harm": 8},
            ]
        },
    }
    gate = summarizer._gate(result)
    assert gate["decision"] == "PROMOTE_TO_FULL125"
    assert gate["passed_arms"] == [C1]


def test_m28_row_is_truth_unopened_and_publishes_local_risk_diagnostics(tmp_path) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {"package_root_sha256": "3" * 64, "package_seal_sha256": "4" * 64}
    )
    compact = d1_overlay_from_base_cache(base)
    receipt = execute_m24_row(
        arm=C2,
        row_id="synthetic_m28_c2",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        output_root=tmp_path / "c2",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["query_truth_opened"] is False
    assert receipt["fit_query_rows_used"] == 0
    assert receipt["candidate_lock_sha256"] == m28_arm_config_hash(C2)
    assert receipt["resource"]["registration_timing_scope"] == "b3_conditioned_support_only_local_flip_risk"
    assert all(
        audit["selection_policy"] == "QUERY_LOCAL_EXACT_B0_OR_B3"
        and audit["query_application"]["row_source_allowlist"] == ["B0", "B3"]
        and audit["query_application"]["query_state_update"] is False
        for audit in receipt["scenario_audit"].values()
    )
