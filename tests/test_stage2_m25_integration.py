from __future__ import annotations

import importlib.util
from pathlib import Path

import json
import pytest

from cvsrffi.stage2_m24_row_executor import execute_m24_row
from cvsrffi.stage2_m25_anchored_residual import B1, B2, B3
from test_stage2_m24_integration import _caches


def test_m25_full125_lifecycle_scripts_are_available() -> None:
    for name in (
        "scripts.run_m25_anchored_residual_full125",
        "scripts.score_m25_anchored_residual_full125",
        "scripts.summarize_m25_anchored_residual_full125",
    ):
        assert importlib.util.find_spec(name) is not None


def test_b3_k2_row_is_truth_unopened_and_exact_g0_fallback(tmp_path: Path) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {"package_root_sha256": "3" * 64, "package_seal_sha256": "4" * 64}
    )
    receipt = execute_m24_row(
        arm=B3,
        row_id="synthetic_m25_b3_k2",
        receiver="3-19",
        base_cache=base,
        overlay_cache=__import__(
            "cvsrffi.stage2_m24_row_executor", fromlist=["d1_overlay_from_base_cache"]
        ).d1_overlay_from_base_cache(base),
        output_root=tmp_path / "b3",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["query_truth_opened"] is False
    assert receipt["fit_query_rows_used"] == 0
    assert receipt["anchored_base_parity"]["prediction_disagreements"] == 0
    assert receipt["anchored_base_parity"]["before_prediction_disagreements"] == 0
    assert receipt["resource"]["registration_timing_scope"] == "g0_anchored_support_only_residual"
    assert receipt["resource"]["persistent_update_state_bytes"] == 0
    assert all(
        audit["selected_strength"] == 0.0
        and audit["fallback_reason"] == "K_LT_5_EXACT_G0"
        for audit in receipt["scenario_audit"].values()
    )


def test_b1_b3_rows_report_real_prototype_mac(tmp_path: Path) -> None:
    base, _overlay = _caches()
    base["manifest"].update(
        {"package_root_sha256": "3" * 64, "package_seal_sha256": "4" * 64}
    )
    executor = __import__(
        "cvsrffi.stage2_m24_row_executor", fromlist=["d1_overlay_from_base_cache"]
    )
    compact = executor.d1_overlay_from_base_cache(base)
    receipts = {}
    for arm in (B1, B2, B3):
        receipts[arm] = execute_m24_row(
            arm=arm,
            row_id=f"synthetic_{arm}",
            receiver="3-19",
            base_cache=base,
            overlay_cache=compact,
            output_root=tmp_path / arm,
            seed=7282101,
        )
    for arm, receipt in receipts.items():
        expected = max(
            (
                sum(audit["prototype_count_by_class"]) * 256
                if audit["selected_strength"] > 0.0
                else 0
            )
            for audit in receipt["scenario_audit"].values()
        )
        assert receipt["resource"]["local_evidence_prototype_mac"] == expected
        assert receipt["resource"]["query_head_mac"] >= expected


def test_active_local_prototype_mac_uses_row_feature_dim() -> None:
    from cvsrffi.stage2_m24_row_executor import _local_prototype_mac

    assert _local_prototype_mac(feature_dim=256, prototype_counts=(1, 2), active=True) == 768
    assert _local_prototype_mac(feature_dim=256, prototype_counts=(1, 2), active=False) == 0


def test_full125_runner_freezes_four_125_row_arms() -> None:
    from scripts import run_m25_anchored_residual_full125 as runner

    assert runner.EVIDENCE_ARMS == (runner.D1, B1, B2, B3)
    assert runner.EXPECTED_INPUT_IDENTITIES == 125
    assert runner.EXPECTED_METHOD_ROWS == 500
    assert len(runner.DEFAULT_RECEIVERS) == 5
    assert len(runner.DEFAULT_SEEDS) == 5
    assert len(runner.DEFAULT_CONDITIONS) == 5


def test_summarizer_disables_d1_only_parity_requirement() -> None:
    from scripts import summarize_m25_anchored_residual_full125 as summarizer

    assert summarizer.ARMS == (summarizer.D1, B1, B2, B3)
    assert summarizer.PARITY_ARM is None


def test_runner_rejects_b0_parity_drift() -> None:
    from scripts import run_m25_anchored_residual_full125 as runner

    task = {
        "arm": runner.D1,
        "receiver": "3-19",
        "k_shot": 5,
        "new_class_count": 20,
    }
    receipt = {
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "arm": runner.D1,
        "receiver": "3-19",
        "k_shot": 5,
        "new_class_count": 20,
        "d1_historical_parity": {
            "prediction_disagreements": 1,
            "before_prediction_disagreements": 0,
        },
    }
    with pytest.raises(ValueError, match="B0 parity"):
        runner._validate_receipt(task, receipt)


def test_runner_rejects_task_receipt_identity_drift() -> None:
    from scripts import run_m25_anchored_residual_full125 as runner

    task = {
        "arm": B2,
        "receiver": "3-19",
        "k_shot": 10,
        "new_class_count": 5,
    }
    receipt = {
        "status": "PREDICTIONS_COMPLETE_TRUTH_UNOPENED",
        "arm": B2,
        "receiver": "3-19",
        "k_shot": 10,
        "new_class_count": 20,
    }
    with pytest.raises(ValueError, match="identity drift"):
        runner._validate_receipt(task, receipt)


def test_summarizer_exclusive_writer_refuses_overwrite(tmp_path: Path) -> None:
    from scripts import summarize_m25_anchored_residual_full125 as summarizer

    output = tmp_path / "summary.json"
    summarizer._write_summary_exclusive(output, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        summarizer._write_summary_exclusive(output, {"status": "PASS"})
