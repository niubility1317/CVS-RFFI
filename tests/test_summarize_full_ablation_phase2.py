from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from cvsrffi.stage2_ablation_truth_scorer import (
    FAILED_ROW_SCHEMA,
    SAME_ROW_SCORE_SCHEMA,
)
from cvsrffi.stage2_metric_scorer import canonical_json_bytes


SCRIPT = Path("code/scripts/summarize_full_ablation_phase2.py")
SPEC = importlib.util.spec_from_file_location("summarize_full_ablation_phase2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def _scenario(value: float) -> dict:
    return {
        "scenario": "leo_clear_weak",
        "A_o_pre": value,
        "A_o_post": value,
        "A_n": value,
        "H": value,
        "F": 0.0,
        "min_old": value,
        "min_new": value,
    }


def _pass(
    logical: str,
    *,
    ablation: str = "P2-FULL",
    physical: str = "physical-1",
    alias_of: str | None = None,
    value: float = 0.8,
) -> dict:
    scenario_rows = [_scenario(value)]
    behavior = {"schema": "behavior"}
    quantization = {"schema": "quantization"}
    resource = {"schema": "resource"}
    digest = lambda value: hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    receipt = {
        "scorer_output_must_not_feed_predictor": True,
        "truth_opened_after_prediction_commit": True,
        "logical_row_key": logical,
        "ablation_id": ablation,
        "physical_execution_id": physical,
        "effective_config_hash": "a" * 64,
        "alias_of": alias_of,
        "independent_observation": alias_of is None,
        "stage": "stage2c",
        "receiver": "20-1",
        "k_shot": 5,
        "before_prediction_hash": "b" * 64,
        "after_prediction_hash": "c" * 64,
        "behavior_receipt_sha256": digest(behavior),
        "quantization_receipt_sha256": digest(quantization),
        "resource_receipt_sha256": digest(resource),
        "same_row_metrics_sha256": digest(scenario_rows),
    }
    return {
        "schema": SAME_ROW_SCORE_SCHEMA,
        "status": "PASS",
        "logical_row_key": logical,
        "ablation_id": ablation,
        "physical_execution_id": physical,
        "effective_config_hash": "a" * 64,
        "alias_of": alias_of,
        "independent_observation": alias_of is None,
        "stage": "stage2c",
        "receiver": "20-1",
        "k_shot": 5,
        "scenario_rows": scenario_rows,
        "before_prediction_hash": "b" * 64,
        "after_prediction_hash": "c" * 64,
        "behavior": behavior,
        "quantization": quantization,
        "resource": resource,
        "behavior_receipt_sha256": digest(behavior),
        "quantization_receipt_sha256": digest(quantization),
        "resource_receipt_sha256": digest(resource),
        "same_row_metrics_sha256": digest(scenario_rows),
        "truth_opened_after_prediction_commit": True,
        "scorer_receipt": receipt,
        "scorer_receipt_sha256": digest(receipt),
    }


def _failed(logical: str) -> dict:
    receipt = {
        "schema": "cvs.full_ablation.phase2.failure_receipt.v1",
        "logical_row_key": logical,
        "ablation_id": "P2-FULL",
        "physical_execution_id": "physical-failed",
        "effective_config_hash": "b" * 64,
        "alias_of": None,
        "independent_observation": True,
        "stage": "stage2c",
        "receiver": "20-1",
        "k_shot": 1,
        "failure_code": "ZERO_PREDICTION",
        "failure_fingerprint": "RuntimeError:fixed",
        "zero_prediction": True,
    }
    return {
        "schema": FAILED_ROW_SCHEMA,
        "status": "FAILED",
        "logical_row_key": logical,
        "ablation_id": "P2-FULL",
        "physical_execution_id": "physical-failed",
        "effective_config_hash": "b" * 64,
        "alias_of": None,
        "independent_observation": True,
        "stage": "stage2c",
        "receiver": "20-1",
        "k_shot": 1,
        "scenario_rows": [],
        "failure_code": "ZERO_PREDICTION",
        "failure_fingerprint": "RuntimeError:fixed",
        "zero_prediction": True,
        "failure_receipt": receipt,
        "failure_receipt_sha256": hashlib.sha256(
            canonical_json_bytes(receipt)
        ).hexdigest(),
    }


def test_failed_and_alias_rows_are_retained_but_never_double_counted() -> None:
    canonical = _pass("canonical", value=0.8)
    alias = _pass(
        "alias",
        ablation="P2-F3",
        physical="physical-1",
        alias_of="canonical",
        value=0.1,
    )
    failed = _failed("failed")
    summary = summary_module.summarize_rows([canonical, alias, failed])
    assert summary["logical_row_count"] == 3
    assert summary["independent_physical_execution_count"] == 2
    assert summary["failed_row_count"] == 1
    assert summary["alias_row_count"] == 1
    assert [item["logical_row_key"] for item in summary["excluded_failed_rows"]] == [
        "failed"
    ]
    assert [item["logical_row_key"] for item in summary["excluded_alias_rows"]] == [
        "alias"
    ]
    assert {row["logical_row_key"] for row in summary["all_rows"]} == {
        "canonical",
        "alias",
        "failed",
    }
    full = next(
        row for row in summary["arm_summaries"] if row["ablation_id"] == "P2-FULL"
    )
    assert full["independent_pass_row_count"] == 1
    assert full["metric_mean"]["H"] == 0.8
    alias_arm = next(
        row for row in summary["arm_summaries"] if row["ablation_id"] == "P2-F3"
    )
    assert alias_arm["independent_pass_row_count"] == 0
    assert alias_arm["metric_mean"]["H"] is None


def test_rejects_alias_without_canonical_or_with_binding_drift() -> None:
    alias = _pass(
        "alias",
        ablation="P2-F3",
        alias_of="missing",
    )
    with pytest.raises(summary_module.Phase2SummaryError, match="canonical"):
        summary_module.summarize_rows([alias])

    canonical = _pass("canonical")
    alias["alias_of"] = "canonical"
    alias["effective_config_hash"] = "c" * 64
    with pytest.raises(summary_module.Phase2SummaryError, match="does not bind"):
        summary_module.summarize_rows([canonical, alias])


def test_rejects_two_independent_rows_for_one_physical_execution() -> None:
    one = _pass("one")
    two = _pass("two")
    with pytest.raises(summary_module.Phase2SummaryError, match="reuse"):
        summary_module.summarize_rows([one, two])


def test_rejects_alias_flag_forgery() -> None:
    forged = _pass("alias", alias_of="canonical")
    forged["independent_observation"] = True
    with pytest.raises(
        summary_module.Phase2SummaryError,
        match="independent_observation|contradiction",
    ):
        summary_module.summarize_rows([forged])


def test_rejects_metric_payload_tamper() -> None:
    tampered = _pass("canonical")
    tampered["scenario_rows"][0]["H"] = 0.99
    with pytest.raises(summary_module.Phase2SummaryError, match="payload hash"):
        summary_module.summarize_rows([tampered])


def test_summary_write_and_cli_are_no_overwrite(tmp_path: Path) -> None:
    row_path = tmp_path / "row.json"
    output = tmp_path / "summary.json"
    row_path.write_text(
        json.dumps(_pass("canonical"), ensure_ascii=False),
        encoding="utf-8",
    )
    assert (
        summary_module.main(
            ["--row-record", str(row_path), "--output", str(output)]
        )
        == 0
    )
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        summary_module.main(
            ["--row-record", str(row_path), "--output", str(output)]
        )
    assert output.read_bytes() == original
