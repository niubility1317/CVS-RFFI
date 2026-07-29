from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_ablation_truth_scorer import (
    BEHAVIOR_RECEIPT_SCHEMA,
    QUANTIZATION_RECEIPT_SCHEMA,
    RESOURCE_RECEIPT_SCHEMA,
    FullAblationScoringError,
    build_failed_row_record,
    score_full_ablation_row,
    write_row_record_exclusive,
)
from cvsrffi.stage2_metric_scorer import Stage2ScoringError


SHA = "a" * 64
OLD1 = "cls_" + "1" * 32
OLD2 = "cls_" + "2" * 32
NEW1 = "cls_" + "3" * 32
NEW2 = "cls_" + "4" * 32
TOKENS = ["qid_" + str(index) * 32 for index in range(1, 5)]
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
TOKENS_BY_SCENARIO = {
    scenario: [
        "qid_" + f"{scenario_index + 1}{index}" * 16
        for index in range(1, 5)
    ]
    for scenario_index, scenario in enumerate(SCENARIOS)
}


def _identity(*, alias_of: str | None = None) -> dict:
    return {
        "logical_row_key": "P2-FULL::logical"
        if alias_of is None
        else "P2-F3::alias",
        "ablation_id": "P2-FULL" if alias_of is None else "P2-F3",
        "physical_execution_id": "physical-001",
        "effective_config_hash": "b" * 64,
        "alias_of": alias_of,
    }


def _behavior() -> dict:
    return {
        "schema": BEHAVIOR_RECEIPT_SCHEMA,
        "fallback_counts": {"k_le_2": 0},
        "full_block_weights": {"full": 0.6, "block3": 0.4},
        "fisher_gate_accept_counts": {"attempted": 4, "accepted": 3},
        "atomic_rollback_counts": {"attempted": 1, "rolled_back": 1},
        "failure_closure_count": 0,
    }


def _quantization() -> dict:
    return {
        "schema": QUANTIZATION_RECEIPT_SCHEMA,
        "max_logit_abs_error": 0.01,
        "mean_logit_abs_error": 0.002,
        "argmax_flip_rate": 0.25,
        "prediction_agreement_rate": 0.75,
    }


def _resource() -> dict:
    return {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "feature_cache_bytes": 1024,
        "deployment_state_bytes": 768,
        "state_bytes": 512,
        "registration_time_ms": 3.5,
        "row_peak_rss_bytes": 4096,
        "row_peak_vram_bytes": 2048,
        "candidate_peak_memory_isolated": False,
        "closed_form_fit_count": 2,
        "mac_equivalent_upper_bound": 1000,
        "query_head_mac": 50,
        "candidate_head_batch_query_latency_ms_per_row": 0.5,
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": None,
        "row_orchestration_time_ms": 8.0,
        "auxiliary_state_cost_in_candidate_resource": False,
        "auxiliary_prediction_cost_in_candidate_latency": False,
    }


def _prediction_binding_and_arrays(*, stage: str = "stage2c"):
    scenarios = [scenario for scenario in SCENARIOS for _ in TOKENS]
    tokens = [
        token
        for scenario in SCENARIOS
        for token in TOKENS_BY_SCENARIO[scenario]
    ]
    before = [OLD1, OLD2, OLD1, OLD1] * len(SCENARIOS)
    after = [OLD1, NEW1, NEW1, OLD1] * len(SCENARIOS)
    if stage == "stage2b":
        before = [OLD1, OLD2, OLD1, OLD1] * len(SCENARIOS)
        after = [OLD1, OLD1, OLD1, OLD1] * len(SCENARIOS)
    arrays = {
        "query_tokens": np.asarray(tokens),
        "scenarios": np.asarray(scenarios),
        "candidate_after": np.asarray(after),
        "candidate_before": np.asarray(before),
        "identity_after": np.asarray(after),
        "identity_before": np.asarray(before),
        "direct": np.asarray(before),
        "shared_view_counts": np.asarray([1, 3, 5, 1] * len(SCENARIOS)),
    }
    binding = {
        "stage": stage,
        "row_id": "physical-001",
        "receiver": "20-1",
        "k_shot": 5,
        "candidate_lock_sha256": "c" * 64,
        "predictor_package_root_sha256": "d" * 64,
        "predictor_package_seal_sha256": "e" * 64,
        "scenarios": list(SCENARIOS),
        "resource_receipt": {},
        "adapter_resource_verification": {},
    }
    audit = {
        "prediction_artifact_sha256": "f" * 64,
        "prediction_seal_sha256": "0" * 64,
    }
    return binding, arrays, audit


def _truth_and_manifest(*, stage: str = "stage2c"):
    rows = []
    for scenario in SCENARIOS:
        rows.extend(
            [
                {
                    "scenario": scenario,
                    "query_token": TOKENS_BY_SCENARIO[scenario][0],
                    "true_class_index": 0,
                    "true_class_handle": OLD1,
                    "transmitter_label": "old-1",
                    "evaluation_role": "target_old",
                    "receiver_label": "20-1",
                },
                {
                    "scenario": scenario,
                    "query_token": TOKENS_BY_SCENARIO[scenario][1],
                    "true_class_index": 1,
                    "true_class_handle": OLD2,
                    "transmitter_label": "old-2",
                    "evaluation_role": "target_old",
                    "receiver_label": "20-1",
                },
                {
                    "scenario": scenario,
                    "query_token": TOKENS_BY_SCENARIO[scenario][2],
                    "true_class_index": (
                        2 if stage == "stage2c" else None
                    ),
                    "true_class_handle": (
                        NEW1 if stage == "stage2c" else None
                    ),
                    "transmitter_label": "new-1",
                    "evaluation_role": "target_new",
                    "receiver_label": "20-1",
                },
                {
                    "scenario": scenario,
                    "query_token": TOKENS_BY_SCENARIO[scenario][3],
                    "true_class_index": (
                        3 if stage == "stage2c" else None
                    ),
                    "true_class_handle": (
                        NEW2 if stage == "stage2c" else None
                    ),
                    "transmitter_label": "new-2",
                    "evaluation_role": "target_new",
                    "receiver_label": "20-1",
                },
            ]
        )
    truth = {
        "schema": "cvs.phase2.query_truth_sidecar.v2",
        "stage": stage,
        "receiver": "20-1",
        "seed": 123,
        "rows": rows,
    }
    manifest = {
        "predictor_package_root_sha256": "d" * 64,
        "predictor_package_seal_sha256": "e" * 64,
    }
    audit = {
        "scoring_manifest_sha256": "6" * 64,
        "truth_sidecar_sha256": "7" * 64,
    }
    return truth, manifest, audit


def _patch_verified_inputs(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    stage: str = "stage2c",
) -> None:
    binding, arrays, audit = _prediction_binding_and_arrays(stage=stage)
    truth, manifest, truth_audit = _truth_and_manifest(stage=stage)

    def prediction(*_args, **_kwargs):
        events.append("prediction_verified")
        return binding, arrays, audit

    def truth_side(*_args, **_kwargs):
        assert events == ["prediction_verified"]
        events.append("truth_opened")
        return truth, manifest, truth_audit

    monkeypatch.setattr(
        "cvsrffi.stage2_ablation_truth_scorer.load_verified_sealed_prediction",
        prediction,
    )
    monkeypatch.setattr(
        "cvsrffi.stage2_ablation_truth_scorer.load_verified_scoring_sidecar",
        truth_side,
    )


def _score(
    monkeypatch: pytest.MonkeyPatch,
    *,
    alias_of: str | None = None,
    stage: str = "stage2c",
) -> dict:
    events: list[str] = []
    _patch_verified_inputs(monkeypatch, events, stage=stage)
    result = score_full_ablation_row(
        "unused.cvspred",
        "unused.truth.manifest.json",
        expected_prediction_artifact_sha256=SHA,
        expected_prediction_seal_sha256=SHA,
        expected_scoring_manifest_sha256=SHA,
        row_identity=_identity(alias_of=alias_of),
        behavior_receipt=_behavior(),
        quantization_receipt=_quantization(),
        resource_receipt=_resource(),
    )
    assert events == ["prediction_verified", "truth_opened"]
    return result


def test_truth_opens_only_after_prediction_verification_and_scores_same_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _score(monkeypatch)
    assert result["truth_opened_after_prediction_commit"] is True
    assert result["independent_observation"] is True
    assert len(result["scenario_rows"]) == 3
    row = result["scenario_rows"][0]
    assert {
        key: row[key]
        for key in ("A_o_pre", "A_o_post", "A_n", "H", "F", "min_old", "min_new")
    } == {
        "A_o_pre": 1.0,
        "A_o_post": 0.5,
        "A_n": 0.5,
        "H": 0.5,
        "F": 0.5,
        "min_old": 0.0,
        "min_new": 0.0,
    }
    assert row["old_to_new_count"] == 1
    assert row["new_to_old_count"] == 1
    assert row["old_to_new_rate"] == 0.5
    assert row["new_to_old_rate"] == 0.5
    assert row["per_class_confusion"]["old-2"] == {"new-1": 1}
    assert result["before_prediction_hash"] != result["after_prediction_hash"]
    assert result["scorer_receipt"]["truth_opened_after_prediction_commit"] is True
    assert len(result["scorer_receipt_sha256"]) == 64


def test_prediction_verification_failure_never_opens_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def fail_prediction(*_args, **_kwargs):
        raise Stage2ScoringError("tampered prediction")

    def truth_side(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("truth must stay closed")

    monkeypatch.setattr(
        "cvsrffi.stage2_ablation_truth_scorer.load_verified_sealed_prediction",
        fail_prediction,
    )
    monkeypatch.setattr(
        "cvsrffi.stage2_ablation_truth_scorer.load_verified_scoring_sidecar",
        truth_side,
    )
    with pytest.raises(Stage2ScoringError, match="tampered prediction"):
        score_full_ablation_row(
            "unused.cvspred",
            "unused.truth.manifest.json",
            expected_prediction_artifact_sha256=SHA,
            expected_prediction_seal_sha256=SHA,
            expected_scoring_manifest_sha256=SHA,
            row_identity=_identity(),
            behavior_receipt=_behavior(),
            quantization_receipt=_quantization(),
            resource_receipt=_resource(),
        )
    assert opened is False


def test_alias_is_never_an_independent_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _score(monkeypatch, alias_of="P2-FULL::logical")
    assert result["alias_of"] == "P2-FULL::logical"
    assert result["independent_observation"] is False
    assert result["physical_execution_id"] == "physical-001"


def test_stage2b_reports_only_preregistration_old_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _score(monkeypatch, stage="stage2b")
    row = result["scenario_rows"][0]
    assert row["A_o_pre"] == 0.5
    assert row["min_old"] == 0.0
    assert row["A_o_post"] is None
    assert row["A_n"] is None
    assert row["H"] is None
    assert row["F"] is None
    assert row["min_new"] is None


@pytest.mark.parametrize(
    ("receipt_name", "mutate", "message"),
    [
        (
            "behavior",
            lambda value: value["full_block_weights"].update({"full": 0.9}),
            "sum to one",
        ),
        (
            "behavior",
            lambda value: value["fisher_gate_accept_counts"].update(
                {"accepted": 5}
            ),
            "cannot exceed",
        ),
        (
            "quantization",
            lambda value: value.update({"prediction_agreement_rate": 0.9}),
            "must sum to one",
        ),
        (
            "resource",
            lambda value: value.update({"state_bytes": -1}),
            "nonnegative integer",
        ),
    ],
)
def test_invalid_behavior_quantization_or_resource_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    receipt_name: str,
    mutate,
    message: str,
) -> None:
    events: list[str] = []
    _patch_verified_inputs(monkeypatch, events)
    receipts = {
        "behavior": _behavior(),
        "quantization": _quantization(),
        "resource": _resource(),
    }
    mutate(receipts[receipt_name])
    with pytest.raises(FullAblationScoringError, match=message):
        score_full_ablation_row(
            "unused.cvspred",
            "unused.truth.manifest.json",
            expected_prediction_artifact_sha256=SHA,
            expected_prediction_seal_sha256=SHA,
            expected_scoring_manifest_sha256=SHA,
            row_identity=_identity(),
            behavior_receipt=receipts["behavior"],
            quantization_receipt=receipts["quantization"],
            resource_receipt=receipts["resource"],
        )
    assert events == ["prediction_verified"]


def test_failed_row_retains_failure_without_performance_values() -> None:
    failed = build_failed_row_record(
        row_identity=_identity(),
        stage="stage2c",
        receiver="20-1",
        k_shot=1,
        failure_code="ZERO_PREDICTION",
        failure_fingerprint="RuntimeError:fixed",
        zero_prediction=True,
    )
    assert failed["status"] == "FAILED"
    assert failed["scenario_rows"] == []
    assert failed["A_o_pre"] is None
    assert failed["H"] is None


def test_row_record_is_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "row.json"
    payload = build_failed_row_record(
        row_identity=_identity(),
        stage="stage2c",
        receiver="20-1",
        k_shot=1,
        failure_code="FAILED",
        failure_fingerprint="fingerprint",
        zero_prediction=False,
    )
    write_row_record_exclusive(path, payload)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_row_record_exclusive(path, {"changed": True})
    assert path.read_bytes() == original


def test_truth_scorer_imports_no_predictor_training_dataset_or_scheduler() -> None:
    source = Path(
        "code/cvsrffi/stage2_ablation_truth_scorer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("predictor", "dataset", "train", "scheduler", "torch")
    assert not any(token in name.lower() for name in imported for token in forbidden)


def test_predictor_side_does_not_import_truth_scorer() -> None:
    for name in (
        "stage2_predictor_entry.py",
        "stage2_predictor_runtime.py",
        "stage2_prediction_artifact.py",
    ):
        source = Path("code/cvsrffi", name).read_text(encoding="utf-8")
        assert "stage2_ablation_truth_scorer" not in source
