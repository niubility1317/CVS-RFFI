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
import numpy as np

import test_stage2_d106_target25_inputs as input_fixture
import cvsrffi.stage2_d106_target25_runner as runner_module
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


def _expanded_rows(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    plan_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    query_ids: dict[str, list[str]] = {}
    for row in raw_rows:
        old = [f"old-{index}" for index in range(2)]
        new = [f"new-{index}" for index in range(row["new_count"])]
        plan_scenes = []
        context_scenes = []
        for scene in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
            plan_states = []
            context_states = []
            for state_name in ("before", "after"):
                registry = old if state_name == "before" else old + new
                support = [
                    f"{row['job_id']}/{scene}/{state_name}/{class_id}/{shot}"
                    for class_id in registry
                    for shot in range(row["k_shot"])
                ]
                query = [f"{row['job_id']}/{scene}/{state_name}/q{index}" for index in range(3)]
                query_root = canonical_sha256(sorted(query))
                query_ids[query_root] = query
                state = {
                    "state": state_name,
                    "registration_state": "BEFORE_REGISTRATION" if state_name == "before" else "AFTER_REGISTRATION",
                    "registered_classes": registry,
                    "old_classes": old,
                    "new_classes": [] if state_name == "before" else new,
                    "capsule_id": canonical_sha256({"row": row["job_id"], "state": state_name}),
                    "split_id": canonical_sha256({"row": row["job_id"], "scene": scene, "state": state_name}),
                    "authority_receipt_sha256": canonical_sha256({"row": row["job_id"], "validator": state_name}),
                    "support_physical_root_sha256": canonical_sha256(sorted(support)),
                    "query_physical_root_sha256": query_root,
                }
                state["state_input_receipt_sha256"] = canonical_sha256(state)
                support_key = f"{state_name}_enrollment"
                query_key = f"{state_name}_apply"
                plan_states.append(state)
                context_states.append(
                    {
                        **state,
                        "support_received_iq_ref": row["packages"][support_key],
                        "query_received_iq_ref": row["packages"][query_key],
                    }
                )
            scenario_row_id = f"{row['job_id']}::{scene}"
            plan_scenes.append({"scenario_row_id": scenario_row_id, "scenario": scene, "states": plan_states})
            context_scenes.append({"scenario_row_id": scenario_row_id, "scenario": scene, "states": context_states})
        base = {
            name: row[name]
            for name in (
                "job_id",
                "receiver",
                "seed",
                "k_shot",
                "source_pool_k",
                "new_count",
            )
        }
        plan_rows.append({**base, "scenarios": plan_scenes})
        context_rows.append({**base, "scenarios": context_scenes})
    return plan_rows, context_rows, query_ids


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
    receipt = prepare_d106_target25_inputs(**input_kwargs)
    plan_path = Path(receipt["plan_manifest"])
    context_path = Path(receipt["context_manifest"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    plan_rows, context_rows, query_ids = _expanded_rows(plan["rows"])
    plan["rows"] = plan_rows
    plan["plan_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_receipt_sha256"}
    )
    context["rows"] = context_rows
    context["plan_receipt_sha256"] = plan["plan_receipt_sha256"]
    context["context_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in context.items() if key != "context_receipt_sha256"}
    )
    os.chmod(plan_path, stat.S_IWRITE)
    os.chmod(context_path, stat.S_IWRITE)
    plan_sha = _canonical_write(plan_path, plan)
    context_sha = _canonical_write(context_path, context)
    run_kwargs = {
        "plan_manifest_path": plan_path,
        "expected_plan_file_sha256": plan_sha,
        "context_manifest_path": context_path,
        "expected_context_file_sha256": context_sha,
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
        "--d92-matrix-manifest",
        str(values["d92_matrix_manifest_path"]),
        "--d92-matrix-manifest-sha256",
        values["expected_d92_matrix_manifest_sha256"],
        "--d92-output-root",
        str(values["d92_output_root"]),
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


def test_raw_plan_expands_and_rejects_cross_scene_physical_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = input_fixture._inputs(tmp_path)
    receipt = prepare_d106_target25_inputs(**values)
    monkeypatch.setattr(
        runner_module,
        "_load_raw_package",
        lambda _value: ({}, {}, {}),
    )

    def derived(**kwargs: Any):
        row = kwargs["row"]
        scene = kwargs["scene"]
        state_name = kwargs["state_name"]
        old = ("old-0", "old-1")
        new = tuple(f"new-{index}" for index in range(row["new_count"]))
        registry = old if state_name == "before" else old + new
        support = tuple(
            f"{row['receiver']}/new{row['new_count']}/{scene}/{state_name}/s{index}"
            for index in range(row["k_shot"])
        )
        if row["new_count"] == 20 and row["k_shot"] in (5, 10):
            query = tuple(f"{row['receiver']}/{scene}/{state_name}/matched-q{index}" for index in range(3))
        else:
            query = tuple(f"{row['job_id']}/{scene}/{state_name}/q{index}" for index in range(3))
        if row["k_shot"] == 10 and row["new_count"] == 5 and scene == "leo_rain_weak" and state_name == "before":
            support = (f"{row['receiver']}/new5/leo_clear_weak/before/s0", *support[1:])
        state = {
            "state": state_name,
            "registration_state": "BEFORE_REGISTRATION" if state_name == "before" else "AFTER_REGISTRATION",
            "registered_classes": list(registry),
            "old_classes": list(old),
            "new_classes": [] if state_name == "before" else list(new),
            "capsule_id": canonical_sha256({"row": row["job_id"], "state": state_name}),
            "split_id": canonical_sha256({"row": row["job_id"], "scene": scene, "state": state_name}),
            "authority_receipt_sha256": "a" * 64,
            "support_physical_root_sha256": canonical_sha256(sorted(support)),
            "query_physical_root_sha256": canonical_sha256(sorted(query)),
        }
        state["state_input_receipt_sha256"] = canonical_sha256(state)
        return (
            state,
            {
                **state,
                "support_received_iq_ref": dict(kwargs["support_ref"]),
                "query_received_iq_ref": dict(kwargs["query_ref"]),
            },
            support,
            query,
        )

    monkeypatch.setattr(runner_module, "_derived_state", derived)
    with pytest.raises(D106Target25RunnerError, match="across target scenarios"):
        runner_module._prepared_inputs(
            plan_manifest_path=Path(receipt["plan_manifest"]),
            expected_plan_file_sha256=receipt["plan_file_sha256"],
            context_manifest_path=Path(receipt["context_manifest"]),
            expected_context_file_sha256=receipt["context_file_sha256"],
        )


def test_raw_plan_rejects_source_pool_k_tamper(tmp_path: Path) -> None:
    values = input_fixture._inputs(tmp_path)
    receipt = prepare_d106_target25_inputs(**values)
    plan_path = Path(receipt["plan_manifest"])
    context_path = Path(receipt["context_manifest"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    for document in (plan, context):
        row = next(
            item
            for item in document["rows"]
            if item["receiver"] == "20-1"
            and item["k_shot"] == 5
            and item["new_count"] == 20
        )
        row["source_pool_k"] = 5
    plan["plan_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_receipt_sha256"}
    )
    context["plan_receipt_sha256"] = plan["plan_receipt_sha256"]
    context["context_receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in context.items()
            if key != "context_receipt_sha256"
        }
    )
    os.chmod(plan_path, stat.S_IWRITE)
    os.chmod(context_path, stat.S_IWRITE)
    plan_sha = _canonical_write(plan_path, plan)
    context_sha = _canonical_write(context_path, context)
    with pytest.raises(D106Target25RunnerError, match="source-pool K binding"):
        runner_module._prepared_inputs(
            plan_manifest_path=plan_path,
            expected_plan_file_sha256=plan_sha,
            context_manifest_path=context_path,
            expected_context_file_sha256=context_sha,
        )


def test_k5_materialization_requires_k10_manifest_but_keeps_active_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "source_d92_job_id": "rx_20_1__seed_713102__k_10__new_20",
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 5,
        "source_pool_k": 10,
        "new_count": 20,
    }
    registry = ("old-0", "old-1")
    manifest = {
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 10,
        "registered_classes": [
            {"class_handle": value, "class_index": index}
            for index, value in enumerate(registry)
        ],
        "package_root_sha256": "a" * 64,
    }
    captured: dict[str, int] = {}

    def support_rows(
        _payload: Mapping[str, np.ndarray],
        *,
        registered_classes: tuple[str, ...],
        active_k: int,
    ):
        captured["active_k"] = active_k
        ids = tuple(
            f"{class_id}/rank-{rank}"
            for class_id in registered_classes
            for rank in range(active_k)
        )
        return (
            np.zeros((len(ids), 2, 8), dtype=np.float32),
            tuple(class_id for class_id in registered_classes for _ in range(active_k)),
            ids,
        )

    monkeypatch.setattr(runner_module, "_validate_matched_packages", lambda *_: None)
    monkeypatch.setattr(runner_module, "_d106_support_rows", support_rows)
    monkeypatch.setattr(
        runner_module,
        "_d106_query_rows",
        lambda _payload: (
            np.zeros((2, 2, 8), dtype=np.float32),
            ("query-0", "query-1"),
        ),
    )
    ref = {
        "package_root": "C:/sealed/package",
        "detached_seal_path": "C:/sealed/seal.json",
        "expected_seal_sha256": "b" * 64,
    }
    plan_state, _context_state, support_ids, _query_ids = runner_module._derived_state(
        row=row,
        scene="leo_clear_weak",
        state_name="before",
        support_ref=ref,
        query_ref=ref,
        support_loaded=({"leo_clear_weak": {}}, manifest, {}),
        query_loaded=({"leo_clear_weak": {}}, manifest, {}),
        old_registry=None,
    )
    assert captured["active_k"] == 5
    assert len(support_ids) == 5 * len(registry)
    assert plan_state["registered_classes"] == list(registry)

    bad_manifest = {**manifest, "k_shot": 5}
    with pytest.raises(D106Target25RunnerError, match="package row binding"):
        runner_module._derived_state(
            row=row,
            scene="leo_clear_weak",
            state_name="before",
            support_ref=ref,
            query_ref=ref,
            support_loaded=({"leo_clear_weak": {}}, bad_manifest, {}),
            query_loaded=({"leo_clear_weak": {}}, bad_manifest, {}),
            old_registry=None,
        )


def test_d92_post_materialization_payload_contract_excludes_embedded_manifest() -> None:
    support = {
        name: np.asarray([0])
        for name in set(runner_module.SUPPORT_NPZ_MEMBERS) - {"manifest_json"}
    }
    support.update(
        {
            "support_leo_weak_iq": np.zeros((2, 2, 8), dtype=np.float32),
            "support_class_indices": np.asarray([0, 1], dtype=np.int64),
            "support_rank_within_class": np.asarray([0, 0], dtype=np.int64),
            "support_tokens": np.asarray(["s0", "s1"]),
        }
    )
    iq, labels, tokens = runner_module._d106_support_rows(
        support,
        registered_classes=("old-0", "old-1"),
        active_k=1,
    )
    assert iq.shape == (2, 2, 8)
    assert labels == ("old-0", "old-1")
    assert tokens == ("s0", "s1")

    query = {
        name: np.asarray([0])
        for name in set(runner_module.QUERY_NPZ_MEMBERS) - {"manifest_json"}
    }
    query.update(
        {
            "query_leo_weak_iq": np.zeros((2, 2, 8), dtype=np.float32),
            "query_tokens": np.asarray(["q0", "q1"]),
        }
    )
    query_iq, query_tokens = runner_module._d106_query_rows(query)
    assert query_iq.shape == (2, 2, 8)
    assert query_tokens == ("q0", "q1")
    with pytest.raises(D106Target25RunnerError, match="truth/role"):
        runner_module._d106_query_rows({**query, "query_truth": np.asarray([0, 1])})


def test_package_and_d106_runtime_namespaces_are_distinct_and_both_bound() -> None:
    source_runtime = "f" * 64
    d106_runtime = "b" * 64
    binding = runner_module._runtime_identity_binding(
        {"feature_runtime_sha256": source_runtime},
        {"feature_runtime_sha256": source_runtime},
        d106_runtime,
    )
    assert binding == {
        "source_package_feature_runtime_sha256": source_runtime,
        "d106_runtime_sha256": d106_runtime,
    }
    materializer_source = inspect.getsource(
        runner_module._D106RealStateMaterializer.__call__
    )
    assert "**runtime_identity" in materializer_source
    assert (
        'runtime_sha256=runtime_identity["d106_runtime_sha256"]'
        in materializer_source
    )


def test_support_query_runtime_mismatch_fails_closed() -> None:
    with pytest.raises(D106Target25RunnerError, match="support/query package"):
        runner_module._runtime_identity_binding(
            {"feature_runtime_sha256": "a" * 64},
            {"feature_runtime_sha256": "b" * 64},
            "c" * 64,
        )


@pytest.mark.parametrize(
    ("support_runtime", "query_runtime", "d106_runtime"),
    [
        ("not-a-sha", "not-a-sha", "c" * 64),
        ("a" * 64, "not-a-sha", "c" * 64),
        ("a" * 64, "a" * 64, "not-a-sha"),
    ],
)
def test_illegal_runtime_sha_fails_closed(
    support_runtime: str, query_runtime: str, d106_runtime: str
) -> None:
    with pytest.raises(D106Target25RunnerError, match="must be a lowercase SHA256"):
        runner_module._runtime_identity_binding(
            {"feature_runtime_sha256": support_runtime},
            {"feature_runtime_sha256": query_runtime},
            d106_runtime,
        )


def test_relu_plus_totalization_replaces_only_same_physical_zero_row() -> None:
    signed = np.asarray(
        [[-3.0, -4.0, -12.0], [1.25, -2.5, 3.75]], dtype=np.float32
    )
    plus = np.maximum(signed, np.float32(0.0))
    original_nonzero_bytes = plus[1].tobytes(order="C")

    output, receipt = runner_module._totalize_relu_plus_rows(
        plus, signed, ("physical-0", "physical-1"), scope="support"
    )

    assert np.array_equal(output[0], signed[0])
    assert output[1].tobytes(order="C") == original_nonzero_bytes
    assert output.dtype == np.float32
    assert output.shape == plus.shape
    assert receipt["schema"] == runner_module.PLUS_TOTALIZATION_SCHEMA
    assert receipt["revision"] == runner_module.PLUS_TOTALIZATION_REVISION
    assert receipt["scope"] == "support"
    assert receipt["row_count"] == 2
    assert receipt["replaced_count"] == 1
    assert receipt["replaced_physical_id_root_sha256"] == canonical_sha256(
        ["physical-0"]
    )
    assert receipt["ordered_physical_ids_sha256"] == canonical_sha256(
        ["physical-0", "physical-1"]
    )
    assert receipt["query_truth_access"] is False
    assert receipt["state_updated"] is False
    assert receipt["totalization_receipt_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "totalization_receipt_sha256"
        }
    )


def test_relu_plus_query_totalization_is_isolated_and_noop_bytes_hold() -> None:
    signed = np.asarray([[2.0, -1.0], [0.25, 7.0]], dtype=np.float32)
    plus = np.maximum(signed, np.float32(0.0))

    output, receipt = runner_module._totalize_relu_plus_rows(
        plus, signed, ("query-0", "query-1"), scope="query"
    )

    assert output.tobytes(order="C") == plus.tobytes(order="C")
    assert receipt["scope"] == "query"
    assert receipt["replaced_count"] == 0
    assert receipt["replaced_physical_id_root_sha256"] == canonical_sha256([])
    assert receipt["query_truth_access"] is False
    assert receipt["state_updated"] is False


@pytest.mark.parametrize(
    "signed",
    [
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[np.nan, 1.0]], dtype=np.float32),
    ],
)
def test_relu_plus_totalization_rejects_invalid_signed_rows(
    signed: np.ndarray,
) -> None:
    plus = np.zeros_like(signed)
    with pytest.raises(D106Target25RunnerError):
        runner_module._totalize_relu_plus_rows(
            plus, signed, ("physical-0",), scope="support"
        )


@pytest.mark.parametrize(
    ("plus", "signed", "physical_ids"),
    [
        (
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
            ("physical-0",),
        ),
        (
            np.ones((1, 2), dtype=np.float64),
            np.ones((1, 2), dtype=np.float32),
            ("physical-0",),
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            ("physical-0",),
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            ("physical-0", "physical-0"),
        ),
    ],
)
def test_relu_plus_totalization_rejects_shape_dtype_or_id_drift(
    plus: np.ndarray,
    signed: np.ndarray,
    physical_ids: tuple[str, ...],
) -> None:
    with pytest.raises(D106Target25RunnerError):
        runner_module._totalize_relu_plus_rows(
            plus, signed, physical_ids, scope="support"
        )


def test_real_materializer_calls_totalization_for_support_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support_ids = ("support-0", "support-1")
    query_ids = ("query-0", "query-1")
    registry = ("old-0", "old-1")
    source_runtime = "a" * 64
    manifest = {
        "receiver": "20-1",
        "seed": 713102,
        "k_shot": 1,
        "registered_classes": [
            {"class_handle": value, "class_index": index}
            for index, value in enumerate(registry)
        ],
        "feature_runtime_sha256": source_runtime,
    }
    packages = iter(
        [
            ({"leo_clear_weak": {}}, manifest, {}),
            ({"leo_clear_weak": {}}, manifest, {}),
        ]
    )
    support_iq = np.zeros((2, 2, 8), dtype=np.float32)
    query_iq = np.zeros((2, 2, 8), dtype=np.float32)
    signed = np.asarray(
        [
            [-3.0, -4.0, -12.0],
            [1.0, -2.0, 3.0],
            [-1.0, -2.0, -2.0],
            [0.5, 0.25, -1.0],
        ],
        dtype=np.float32,
    )
    scopes: list[str] = []
    receipts: dict[str, Mapping[str, Any]] = {}
    real_totalize = runner_module._totalize_relu_plus_rows

    def tracked_totalize(
        plus: np.ndarray,
        signed_rows: np.ndarray,
        physical_ids: tuple[str, ...],
        *,
        scope: str,
    ):
        scopes.append(scope)
        output, receipt = real_totalize(
            plus, signed_rows, physical_ids, scope=scope
        )
        receipts[scope] = receipt
        return output, receipt

    class StopAfterPublish(RuntimeError):
        pass

    published: dict[str, Any] = {}

    def capture_publish(
        _feature_path: Path, _receipt_path: Path, **kwargs: Any
    ) -> None:
        published.update(kwargs)
        raise StopAfterPublish

    monkeypatch.setattr(runner_module, "_validate_matched_packages", lambda *_: None)
    monkeypatch.setattr(
        runner_module._D106RealStateMaterializer,
        "_package",
        lambda _self, _value: next(packages),
    )
    monkeypatch.setattr(
        runner_module._D106RealStateMaterializer,
        "_model_for",
        lambda _self, _input_len: object(),
    )
    monkeypatch.setattr(
        runner_module,
        "_d106_support_rows",
        lambda *_args, **_kwargs: (
            support_iq,
            registry,
            support_ids,
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_d106_query_rows",
        lambda *_args, **_kwargs: (query_iq, query_ids),
    )
    monkeypatch.setattr(
        runner_module,
        "_tap_rows",
        lambda *_args, **_kwargs: (signed, np.zeros_like(signed), "d" * 64),
    )
    monkeypatch.setattr(runner_module, "_totalize_relu_plus_rows", tracked_totalize)
    monkeypatch.setattr(runner_module, "publish_d106_paired_features", capture_publish)

    artifact_root = tmp_path / "states"
    artifact_root.mkdir()
    materializer = object.__new__(runner_module._D106RealStateMaterializer)
    materializer.artifact_root = artifact_root
    materializer.checkpoint_sha256 = "b" * 64
    materializer.model_load_receipt_sha256 = "e" * 64
    materializer.rdce_asset = type(
        "Asset", (), {"runtime_sha256": "c" * 64}
    )()
    materializer.device = "cpu"
    materializer.feature_batch_size = 4
    support_ref = {"expected_seal_sha256": "f" * 64}
    query_ref = {"expected_seal_sha256": "9" * 64}
    request = {
        "artifact_index": 0,
        "support_received_iq_ref": support_ref,
        "query_received_iq_ref": query_ref,
        "scenario": "leo_clear_weak",
        "registered_classes": registry,
        "receiver": "20-1",
        "seed": 713102,
        "source_pool_k": 1,
        "k_shot": 1,
        "support_physical_root_sha256": canonical_sha256(sorted(support_ids)),
        "query_physical_root_sha256": canonical_sha256(sorted(query_ids)),
        "evaluation_row_id": "row-0",
    }

    with pytest.raises(StopAfterPublish):
        materializer(request)

    assert scopes == ["support", "query"]
    assert np.array_equal(published["support_plus"][0], signed[0])
    assert np.array_equal(published["query_plus"][0], signed[2])
    assert published["support_plus"][1].tobytes() == np.maximum(
        signed[1], np.float32(0.0)
    ).tobytes()
    assert published["query_plus"][1].tobytes() == np.maximum(
        signed[3], np.float32(0.0)
    ).tobytes()
    expected_forward = canonical_sha256(
        {
            "schema": "cvs.phase2.d106.target25.same_model_forward.v1",
            "row_id": "row-0",
            "received_iq_pair_sha256": canonical_sha256(
                {"support": "f" * 64, "query": "9" * 64}
            ),
            "checkpoint_sha256": "b" * 64,
            "model_load_receipt_sha256": "e" * 64,
            "tap_receipt_sha256": "d" * 64,
            "source_package_feature_runtime_sha256": source_runtime,
            "d106_runtime_sha256": "c" * 64,
            "support_rows": 2,
            "query_rows": 2,
            "support_plus_totalization_receipt_sha256": receipts["support"][
                "totalization_receipt_sha256"
            ],
            "query_plus_totalization_receipt_sha256": receipts["query"][
                "totalization_receipt_sha256"
            ],
            "query_fit_count": 0,
            "query_update_count": 0,
        }
    )
    assert published["forward_receipt_sha256"] == expected_forward
    for scope in ("support", "query"):
        path = artifact_root / "state-000" / f"{scope}_plus_totalization.receipt.json"
        assert json.loads(path.read_text(encoding="utf-8")) == receipts[scope]


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
