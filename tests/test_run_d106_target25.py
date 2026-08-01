from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping

import pytest

import test_stage2_d106_target25_inputs as input_fixture
from cvsrffi.stage2_d106_k_conditioned_router import ROUTE_BY_K, TARGET25_ROW_SCHEMA
from cvsrffi.stage2_d106_matrix_protocol import canonical_sha256
from cvsrffi.stage2_d106_target25_inputs import prepare_d106_target25_inputs
from cvsrffi.stage2_d106_target25_runner import (
    D106Target25RunnerError,
    TRUTH_CATALOG_SCHEMA,
    predict_d106_target25,
    score_d106_target25,
    smoke_d106_target25_prepared_state,
    smoke_d106_target25_state,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_d106_target25.py"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_write(path: Path, value: Mapping[str, Any]) -> str:
    path.write_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return _sha_file(path)


def _query_ids_by_root(split_path: Path) -> dict[str, list[str]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for row in split["rows"]:
        for scenario in row["scenarios"]:
            for state_name in ("before", "after"):
                query = scenario[state_name]["query_physical_ids"]
                result[canonical_sha256(sorted(query))] = query
    return result


def _synthetic_evaluator(**kwargs: Any) -> dict[str, Any]:
    registry = list(kwargs["registered_classes"])
    query = list(kwargs["query_physical_ids"])
    if len(registry) < 2 or len(query) != 3:
        raise AssertionError("synthetic evaluator fixture drift")
    base = [registry[0], registry[1], registry[-1]]
    arms = {
        "M0": base,
        "M_DA": [registry[0], registry[0], registry[-1]],
        "M_HEAD": [registry[1], registry[1], registry[-1]],
        "M_JOINT": [registry[-1], registry[0], registry[1]],
    }
    row: dict[str, Any] = {
        "schema": TARGET25_ROW_SCHEMA,
        "row_id": kwargs["row_id"],
        "receiver": kwargs["receiver"],
        "scene": kwargs["scene"],
        "K": kwargs["active_k"],
        "registered_classes": registry,
        "query_physical_ids": query,
        "arm_predictions": arms,
        "shared_component_receipts": {
            "synthetic_feature_tap_sha256": "b" * 64,
            "synthetic_method_lock_sha256": "c" * 64,
        },
        "query_truth_access": False,
        "query_role_access": False,
        "query_selection": False,
        "query_state_updates": 0,
    }
    row["prediction_receipt_sha256"] = canonical_sha256(row)
    return row


def _prepared(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    input_kwargs = input_fixture._inputs(tmp_path)
    query_ids = _query_ids_by_root(input_kwargs["split_locator_path"])
    receipt = prepare_d106_target25_inputs(**input_kwargs)
    run_kwargs = {
        "plan_manifest_path": Path(receipt["plan_manifest"]),
        "expected_plan_file_sha256": receipt["plan_file_sha256"],
        "context_manifest_path": Path(receipt["context_manifest"]),
        "expected_context_file_sha256": receipt["context_file_sha256"],
    }
    return receipt, run_kwargs, query_ids


def _factory(query_ids: Mapping[str, list[str]]):
    def materialize(request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "row_id": request["evaluation_row_id"],
            "receiver": request["receiver"],
            "scene": request["scenario"],
            "active_k": request["k_shot"],
            "registered_classes": request["registered_classes"],
            "query_physical_ids": query_ids[request["query_physical_root_sha256"]],
        }

    return materialize


def _predicted(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _receipt, run_kwargs, query_ids = _prepared(tmp_path)
    prediction = predict_d106_target25(
        **run_kwargs,
        output_dir=tmp_path / "predictions",
        state_input_factory=_factory(query_ids),
        state_evaluator=_synthetic_evaluator,
    )
    return prediction, run_kwargs


def _truth_from_prediction(prediction_path: Path, plan_path: Path) -> dict[str, Any]:
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_by_job = {row["job_id"]: row for row in plan["rows"]}
    rows = []
    for row in prediction["rows"]:
        plan_row = plan_by_job[row["job_id"]]
        scenes = []
        for scene, plan_scene in zip(row["scenarios"], plan_row["scenarios"], strict=True):
            states = []
            for state, plan_state in zip(scene["states"], plan_scene["states"], strict=True):
                query = state["prediction_row"]["query_physical_ids"]
                old = plan_state["old_classes"]
                new = plan_state["new_classes"]
                labels = [old[0], old[1], new[0] if new else old[0]]
                states.append(
                    {
                        "state": state["state"],
                        "query_physical_ids": query,
                        "labels": labels,
                    }
                )
            scenes.append({"scenario": scene["scenario"], "states": states})
        rows.append({"job_id": row["job_id"], "scenarios": scenes})
    truth: dict[str, Any] = {
        "schema": TRUTH_CATALOG_SCHEMA,
        "matrix_receipt_sha256": prediction["matrix_receipt_sha256"],
        "rows": rows,
    }
    truth["truth_catalog_receipt_sha256"] = canonical_sha256(truth)
    return truth


def test_cli_exposes_only_prepare_predict_score() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "{prepare,predict,score}" in result.stdout
    predict = subprocess.run(
        [sys.executable, str(SCRIPT), "predict", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert predict.returncode == 0
    assert "--checkpoint" in predict.stdout
    assert "--rdce-wire" in predict.stdout
    assert "--rcmr-lock" in predict.stdout
    assert "--feature-batch-size" in predict.stdout
    assert "state-input-factory" not in predict.stdout


def test_cli_prepare_forwards_new_route_lock_file_interface(tmp_path: Path) -> None:
    values = input_fixture._inputs(tmp_path)
    command = [
        sys.executable,
        str(SCRIPT),
        "prepare",
        "--matrix-index",
        str(values["matrix_index_path"]),
        "--matrix-index-sha256",
        values["expected_matrix_index_sha256"],
        "--split-locator",
        str(values["split_locator_path"]),
        "--split-locator-sha256",
        values["expected_split_locator_sha256"],
        "--checkpoint",
        str(values["checkpoint_path"]),
        "--checkpoint-sha256",
        values["expected_checkpoint_sha256"],
        "--rdce-wire",
        str(values["rdce_wire_path"]),
        "--rdce-wire-sha256",
        values["expected_rdce_wire_sha256"],
        "--rdce-lock",
        str(values["rdce_lock_path"]),
        "--rdce-lock-sha256",
        values["expected_rdce_lock_sha256"],
        "--rcmr-lock",
        str(values["rcmr_lock_path"]),
        "--rcmr-lock-sha256",
        values["expected_rcmr_lock_sha256"],
        "--kcr-route-lock",
        str(values["kcr_route_lock_path"]),
        "--kcr-route-lock-sha256",
        values["expected_kcr_route_lock_sha256"],
        "--output-dir",
        str(values["output_dir"]),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["outer_job_count"] == 25


def test_predict_closes_matrix_and_routes_only_after_four_arms(tmp_path: Path) -> None:
    prediction, _run_kwargs = _predicted(tmp_path)
    manifest = json.loads(
        Path(prediction["prediction_manifest"]).read_text(encoding="utf-8")
    )
    assert (
        manifest["outer_job_count"],
        manifest["scenario_row_count"],
        manifest["matched_arm_pair_count"],
        manifest["state_surface_count"],
    ) == (25, 75, 300, 600)
    assert manifest["query_fit_count"] == 0
    assert manifest["query_update_count"] == 0
    assert manifest["query_selection_count"] == 0
    for row in manifest["rows"]:
        for scene in row["scenarios"]:
            for state in scene["states"]:
                assert set(state["prediction_row"]["arm_predictions"]) == {
                    "M0",
                    "M_DA",
                    "M_HEAD",
                    "M_JOINT",
                }
                assert state["routed_prediction"]["selected_arm"] == ROUTE_BY_K[
                    row["k_shot"]
                ]


def test_single_state_smoke_has_no_truth_surface() -> None:
    request = {
        "evaluation_row_id": "one-state",
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "k_shot": 1,
        "registered_classes": ["old-0", "old-1"],
        "query_physical_root_sha256": canonical_sha256(["q0", "q1", "q2"]),
    }

    def factory(value: Mapping[str, Any]) -> dict[str, Any]:
        assert not any("truth" in key for key in value)
        return {
            "row_id": value["evaluation_row_id"],
            "receiver": value["receiver"],
            "scene": value["scenario"],
            "active_k": value["k_shot"],
            "registered_classes": value["registered_classes"],
            "query_physical_ids": ["q0", "q1", "q2"],
        }

    result = smoke_d106_target25_state(
        state_request=request,
        state_input_factory=factory,
        state_evaluator=_synthetic_evaluator,
    )
    assert result["prediction_row"]["query_truth_access"] is False
    assert result["routed_prediction"]["selected_arm"] == "M_DA"


def test_real_checkpoint_smoke_interface_has_no_truth_or_dynamic_factory() -> None:
    parameters = inspect.signature(smoke_d106_target25_prepared_state).parameters
    assert "checkpoint_path" in parameters
    assert "rdce_wire_path" in parameters
    assert "rcmr_lock_path" in parameters
    assert "truth" not in " ".join(parameters)
    assert "factory" not in " ".join(parameters)


def test_missing_surface_is_rejected_before_truth_open(tmp_path: Path) -> None:
    prediction, run_kwargs = _predicted(tmp_path)
    path = Path(prediction["prediction_manifest"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["rows"][0]["scenarios"][0]["states"].pop()
    document["prediction_manifest_receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "prediction_manifest_receipt_sha256"
        }
    )
    os.chmod(path, stat.S_IWRITE)
    prediction_sha = _canonical_write(path, document)
    score_dir = tmp_path / "scores"
    with pytest.raises(D106Target25RunnerError, match="prediction states"):
        score_d106_target25(
            **run_kwargs,
            prediction_manifest_path=path,
            expected_prediction_file_sha256=prediction_sha,
            truth_catalog_path=tmp_path / "must-not-open.json",
            expected_truth_catalog_file_sha256="d" * 64,
            output_dir=score_dir,
        )
    assert not score_dir.exists()


def test_route_tamper_is_rejected_before_truth_open(tmp_path: Path) -> None:
    prediction, run_kwargs = _predicted(tmp_path)
    path = Path(prediction["prediction_manifest"])
    document = json.loads(path.read_text(encoding="utf-8"))
    routed = document["rows"][0]["scenarios"][0]["states"][0][
        "routed_prediction"
    ]
    routed["selected_arm"] = "M_JOINT"
    routed["route_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in routed.items() if key != "route_receipt_sha256"}
    )
    document["prediction_manifest_receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "prediction_manifest_receipt_sha256"
        }
    )
    os.chmod(path, stat.S_IWRITE)
    prediction_sha = _canonical_write(path, document)
    with pytest.raises(D106Target25RunnerError, match="K-route"):
        score_d106_target25(
            **run_kwargs,
            prediction_manifest_path=path,
            expected_prediction_file_sha256=prediction_sha,
            truth_catalog_path=tmp_path / "must-not-open.json",
            expected_truth_catalog_file_sha256="e" * 64,
            output_dir=tmp_path / "scores",
        )


def test_score_writes_truth_event_then_125_same_row_metrics(tmp_path: Path) -> None:
    prediction, run_kwargs = _predicted(tmp_path)
    prediction_path = Path(prediction["prediction_manifest"])
    truth_path = tmp_path / "truth.json"
    truth_sha = _canonical_write(
        truth_path,
        _truth_from_prediction(prediction_path, run_kwargs["plan_manifest_path"]),
    )
    result = score_d106_target25(
        **run_kwargs,
        prediction_manifest_path=prediction_path,
        expected_prediction_file_sha256=prediction[
            "prediction_manifest_file_sha256"
        ],
        truth_catalog_path=truth_path,
        expected_truth_catalog_file_sha256=truth_sha,
        output_dir=tmp_path / "scores",
    )
    assert Path(result["truth_open_event"]).is_file()
    score = json.loads(Path(result["score_manifest"]).read_text(encoding="utf-8"))
    assert score["metric_row_count"] == 125
    assert len(score["rows"]) == 125
    assert set(row["method"] for row in score["rows"]) == {
        "M0",
        "M_DA",
        "M_HEAD",
        "M_JOINT",
        "ROUTED",
    }
    assert all(len(row["metric_row_receipt_sha256"]) == 64 for row in score["rows"])
