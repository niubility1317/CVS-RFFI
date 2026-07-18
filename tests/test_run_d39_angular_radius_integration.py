from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SCRIPTS = CODE / "scripts"
for value in (CODE, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_d25_support_only_concat as runner  # noqa: E402


def _cell_manifest(
    *, receiver: str = "20-1", seed: int = 713101, new_count: int = 5
) -> tuple[dict[str, object], dict[str, object]]:
    old = [
        {"class_index": index, "class_handle": f"old-{index}"}
        for index in range(6)
    ]
    new = [
        {"class_index": 6 + index, "class_handle": f"new-{index}"}
        for index in range(new_count)
    ]
    common: dict[str, object] = {
        "receiver": receiver,
        "seed": seed,
        "k_shot": 10,
    }
    return (
        {**common, "registered_classes": old},
        {**common, "registered_classes": old + new},
    )


def _support_blocks() -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(390718)
    classes = ("old_left", "old_right", "new_left", "new_right")
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        z_center = np.zeros(160, dtype=np.float32)
        fft_center = np.zeros(96, dtype=np.float32)
        rf_center = np.zeros(32, dtype=np.float32)
        z_center[class_index] = 1.0
        fft_center[class_index] = 1.0
        rf_center[class_index] = 1.0
        for rank in range(10):
            z_rows.append(z_center + 0.004 * rng.normal(size=160).astype(np.float32))
            fft_rows.append(
                fft_center + 0.004 * rng.normal(size=96).astype(np.float32)
            )
            rf_rows.append(rf_center + 0.004 * rng.normal(size=32).astype(np.float32))
            labels.append(label)
            ranks.append(rank)
    return (
        {
            "labels": np.asarray(labels),
            "ranks": np.asarray(ranks, dtype=np.int64),
            "tokens": np.asarray(
                [f"physical-{index:03d}" for index in range(len(labels))]
            ),
        },
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def test_d39_preopen_cell_is_fixed_before_support_open() -> None:
    runner._require_d39_development_cell(*_cell_manifest())
    for manifests in (
        _cell_manifest(receiver="3-19"),
        _cell_manifest(seed=713102),
        _cell_manifest(new_count=2),
        _cell_manifest(new_count=10),
    ):
        with pytest.raises(runner.D25RunnerError):
            runner._require_d39_development_cell(*manifests)


def test_d39_candidate_lock_v17_is_exact_six_by_three_by_five() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D39_V1)
    assert tuple(candidates) == runner.D39_CANDIDATES
    assert runner.D39_CANDIDATES == (
        runner.IDENTITY_CANDIDATE,
        runner.D39_PROTONET_CDA,
        runner.DIAG_CANDIDATE,
        runner.D39_D38_B_INT8,
        runner.D39_INT8,
        runner.D39_FP32,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D39_V1) == (
        runner.D39_INT8,
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 90
    assert candidates[runner.D39_D38_B_INT8].core.arm == "B"
    assert candidates[runner.D39_INT8].deploy_precision == "int8"
    assert candidates[runner.D39_FP32].deploy_precision == "fp32"
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D39_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v17"
    assert {
        "d38_strong_b3_quantized_core_sha256",
        "d39_angular_radius_core_sha256",
    } <= set(lock["source_closure"])
    assert lock["d39_formula_lock"] == {
        "radius_formula": "(nu*r0^2+(K-1)*m2)/(nu+K-1)",
        "nu": 4.0,
        "epsilon": 0.001,
        "r0_floor": 0.05,
        "k1_policy": "all_radius_equals_frozen_r0",
        "score": "-0.5*(theta/(radius+epsilon))^2-log(radius+epsilon)",
        "training_trajectory": "exact_D38_B_20_plus_10",
    }
    int8_row = next(
        row for row in lock["candidates"] if row["candidate_id"] == runner.D39_INT8
    )
    assert int8_row["config"]["k1_radius_rule"] == (
        "K_minus_1_zero_radius_equals_r0_bitwise"
    )
    assert int8_row["config"]["label_permutation_equivariant"] is True
    assert sum(row["eligible_positive_route"] for row in lock["candidates"]) == 1


def test_d39_cli_and_artifact_schemas_have_no_query_truth_oracle_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D39_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    surfaces = (
        {item.dest.lower() for item in parser._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    )
    for names in surfaces:
        assert not any(token in name for name in names for token in forbidden)
    for artifact in (
        "support_fold",
        "support_audit",
        "selection",
        "resource_matrix",
        "geometry_matrix",
        "receipt",
    ):
        assert runner._artifact_schema(runner.CANDIDATE_SET_D39_V1, artifact) == (
            f"cvs.phase2.d39.{artifact}.v1"
        )


def test_d39_outer_fold_closes_real_radius_source_prefix_pairwise_and_precision() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D39_V1)[
        runner.D39_INT8
    ]
    result = runner._evaluate_d39_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        candidate_id=runner.D39_INT8,
        config=config,
        seed=713101,
        device="cpu",
        scenario="leo_clear_weak",
        outer_fold=0,
    )
    source = result["radius_source_audit"]
    assert source["old_source_row_count"] == 16
    assert source["new_source_row_count"] == 16
    assert source["old_source_held_intersection_count"] == 0
    assert source["new_source_held_intersection_count"] == 0
    assert source["old_source_new_class_row_count"] == 0
    assert source["new_source_old_class_row_count"] == 0
    assert len(source["old_source_physical_token_sha256"]) == 64
    assert len(source["new_source_physical_token_sha256"]) == 64
    assert result["old_prototype_prefix_bitwise_unchanged"] is True
    assert result["old_radius_prefix_bitwise_unchanged"] is True
    assert result["r0_bitwise_unchanged"] is True
    assert result["radius_positive_finite"] is True
    assert result["radius_fp16_shared_between_int8_fp32"] is True
    assert len(result["outer_prediction_sha256"]) == 64
    assert len(result["radius_fp16_sha256"]) == 64
    assert len(result["r0_fp16_sha256"]) == 64
    assert len(result["pairwise_support_diagnostics"]) == 4
    assert result["resource"]["radius_fit_acos_scalar_operations"] == 32
    assert result["resource"]["per_query_acos_scalar_operations"] == 4
    assert result["resource"]["per_query_log_scalar_operations"] == 4
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["resource"]["clean_sample_access"] is False
    assert result["resource"]["source_sample_access"] is False
    assert len(result["training_trace"]) == 30


def _metric(value: float, names: tuple[str, ...]) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {name: value for name in names},
    }


def _selector_rows(candidate_id: str) -> list[dict[str, object]]:
    old_names = ("old_alpha", "old_beta")
    new_names = ("new_alpha", "new_beta")
    is_diag = candidate_id == runner.DIAG_CANDIDATE
    is_d38 = candidate_id == runner.D39_D38_B_INT8
    is_d39 = candidate_id in (runner.D39_INT8, runner.D39_FP32)
    value = 0.70 if is_diag else 0.80 if is_d38 else 0.90 if is_d39 else 0.60
    intrusion = 2 if is_diag else 3 if is_d38 else 1 if is_d39 else 4
    confusion = 1 if is_d38 else 0 if is_d39 else 2
    margin = 0.10 if is_d38 else 0.50 if is_d39 else 0.0
    trace = [{"phase": "stage2b", "step": 1}, {"phase": "stage2c", "step": 1}]
    rows: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for fold_index, _ in enumerate(runner.HELD_RANKS):
            held_identity = hashlib.sha256(
                f"{scenario}:{fold_index}".encode("utf-8")
            ).hexdigest()
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "scenario": scenario,
                "fold_index": fold_index,
                "held_ranks": list(runner.HELD_RANKS[fold_index]),
                "held_physical_token_count": 8,
                "held_physical_token_sha256": held_identity,
                "before_old": _metric(0.80 if is_d38 or is_d39 else value, old_names),
                "after_old": _metric(value, old_names),
                "after_new": _metric(value, new_names),
                "H_old_new": value,
                "forgetting": 0.0,
                "joint_floor": value,
                "outer_held_new_intrusion_count": intrusion,
                "new_new_confusion_count": confusion,
                "new_new_margin_min": margin,
                "new_new_margin_mean": margin,
                "matched_fp32_outer_argmax_change_count": 0,
                "deployment_precision": (
                    "int8"
                    if candidate_id == runner.D39_INT8
                    else "fp32"
                    if candidate_id == runner.D39_FP32
                    else "control"
                ),
                "outer_prediction_sha256": (
                    "same-d39-outer" if is_d39 else f"outer-{candidate_id}"
                ),
                "radius_fp16_sha256": (
                    "same-d39-radius" if is_d39 else f"radius-{candidate_id}"
                ),
                "r0_fp16_sha256": (
                    "same-d39-r0" if is_d39 else f"r0-{candidate_id}"
                ),
                "registration_before_prediction_sha256": (
                    "same-d38-trajectory" if is_d38 or is_d39 else candidate_id
                ),
                "training_trace": trace,
                "old_score_columns_bitwise_unchanged": True,
                "old_prototype_prefix_bitwise_unchanged": True,
                "old_radius_prefix_bitwise_unchanged": True,
                "r0_bitwise_unchanged": True,
                "radius_positive_finite": True,
                "radius_fp16_shared_between_int8_fp32": True,
                "radius_source_audit": {
                    "old_source_held_intersection_count": 0,
                    "new_source_held_intersection_count": 0,
                    "old_source_new_class_row_count": 0,
                    "new_source_old_class_row_count": 0,
                    "query_rows_used": 0,
                },
                "target_old_int8_prototypes_used_for_prediction": (
                    candidate_id == runner.D39_INT8
                ),
                "target_new_int8_prototypes_used_for_prediction": (
                    candidate_id == runner.D39_INT8
                ),
                "geometry_summary": {
                    "old_radius_source_state": "registration_preceding_int8_before_state",
                    "old_radius_materialized_before_stage2c": True,
                    "old_radius_materialization_hook_call_count": 1,
                    "old_radius_materialization_stage2b_trace_length": 20,
                    "new_radius_source_state": "final_int8_append_state",
                    "old_radius_new_support_row_count": 0,
                    "held_radius_fit_row_count": 0,
                    "query_rows_used": 0,
                    "radius_nu": 4.0,
                    "radius_epsilon": 0.001,
                    "label_permutation_equivariant": True,
                },
                "resource": {
                    "peak_trainable_parameters": 1_000,
                    "adaptation_epochs": 30,
                    "total_optimizer_steps": 30,
                    "persistent_state_cap_pass": True,
                    "radius_storage_dtype": "float16",
                    "r0_storage_dtype": "float16",
                    "r0_fp16": 0.1,
                    "resident_fp32_target_prototype_count": 0,
                    "dense_query_graph_bytes": 0,
                    "query_rows_used_for_fit": 0,
                    "query_labels_used_for_fit": False,
                    "query_role_oracle_access": False,
                    "query_true_batch_class_count_access": False,
                    "query_class_quota_access": False,
                    "query_batch_global_assignment": False,
                    "clean_sample_access": False,
                    "source_sample_access": False,
                    "class_id_specific_branch": False,
                },
            }
            rows.append(row)
    return rows


def test_d39_matrix_row_identity_and_15_key_selector_are_fail_closed() -> None:
    folds = {
        candidate_id: _selector_rows(candidate_id)
        for candidate_id in runner.D39_CANDIDATES
    }
    assert len(runner._validate_d39_matrix_rows(folds)) == 15
    assert sum(len(rows) for rows in folds.values()) == 90
    assert all(
        row["candidate_id"] == candidate_id
        for candidate_id, rows in folds.items()
        for row in rows
    )
    selected, decisions = runner._select_d39_candidate(folds)
    assert selected == runner.D39_INT8
    assert [
        row["candidate_id"] for row in decisions if row["eligible_positive_route"]
    ] == [runner.D39_INT8]
    decision = next(row for row in decisions if row["candidate_id"] == runner.D39_INT8)
    assert decision["d39_registration_before_and_d38_trace_identity_gate_pass"]
    assert decision["d39_intrusion_strong_b3_and_d38_gate_pass"]
    assert decision["d39_joint_15_key_aggregate_strict_gate_pass"]
    assert decision["strong_b3_outer_held_new_intrusion_count"] == 30

    bad_identity = copy.deepcopy(folds)
    bad_identity[runner.D39_PROTONET_CDA][0] = dict(
        bad_identity[runner.D39_PROTONET_CDA][0], candidate_id="wrong"
    )
    with pytest.raises(runner.D25RunnerError):
        runner._validate_d39_matrix_rows(bad_identity)

    bad_physical = copy.deepcopy(folds)
    bad_physical[runner.D39_FP32][0]["held_physical_token_sha256"] = "f" * 64
    with pytest.raises(runner.D25RunnerError):
        runner._validate_d39_matrix_rows(bad_physical)


@pytest.mark.parametrize(
    "broken_gate",
    (
        "d38_seen_new_and_confusion",
        "strong_b3_old",
        "strong_b3_h_joint",
        "strong_b3_intrusion",
        "radius_source",
        "old_radius_prefix",
        "r0_prefix",
        "fp32_argmax",
        "explicit_fp32_candidate",
        "resource_query_flag",
    ),
)
def test_d39_selector_independent_counterexamples_fall_back_to_identity(
    broken_gate: str,
) -> None:
    folds = copy.deepcopy(
        {
            candidate_id: _selector_rows(candidate_id)
            for candidate_id in runner.D39_CANDIDATES
        }
    )
    d39_rows = folds[runner.D39_INT8]
    if broken_gate == "d38_seen_new_and_confusion":
        for row in d39_rows:
            row["after_new"] = _metric(0.70, ("new_alpha", "new_beta"))
            row["new_new_confusion_count"] = 2
    elif broken_gate == "strong_b3_old":
        d39_rows[0]["after_old"] = _metric(0.60, ("old_alpha", "old_beta"))
    elif broken_gate == "strong_b3_h_joint":
        d39_rows[0]["H_old_new"] = 0.60
        d39_rows[0]["joint_floor"] = 0.60
    elif broken_gate == "strong_b3_intrusion":
        for row in d39_rows:
            row["outer_held_new_intrusion_count"] = 4
    elif broken_gate == "radius_source":
        d39_rows[0]["radius_source_audit"][
            "old_source_held_intersection_count"
        ] = 1
    elif broken_gate == "old_radius_prefix":
        d39_rows[0]["old_radius_prefix_bitwise_unchanged"] = False
    elif broken_gate == "r0_prefix":
        d39_rows[0]["r0_bitwise_unchanged"] = False
    elif broken_gate == "fp32_argmax":
        d39_rows[0]["matched_fp32_outer_argmax_change_count"] = 1
    elif broken_gate == "explicit_fp32_candidate":
        folds[runner.D39_FP32][0]["outer_prediction_sha256"] = "drift"
    elif broken_gate == "resource_query_flag":
        d39_rows[0]["resource"]["query_role_oracle_access"] = True
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(broken_gate)
    selected, decisions = runner._select_d39_candidate(folds)
    assert selected == runner.IDENTITY_CANDIDATE
    assert not any(row["eligible_positive_route"] for row in decisions)


def test_d39_full_k10_selected_only_state_resource_geometry_and_gate() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D39_V1)[
        runner.D39_INT8
    ]
    resource, geometry = runner._full_d39_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        config=config,
        seed=713101,
        device="cpu",
        scenario="leo_clear_weak",
    )
    assert resource["deployment_k_shot"] == 10
    assert resource["pairwise_support_diagnostic_row_count"] == 20
    assert resource["resident_fp32_target_prototype_count"] == 0
    assert resource["formal_state_int8_only"] is True
    assert resource["old_prototype_prefix_bitwise_unchanged"] is True
    assert resource["old_radius_prefix_bitwise_unchanged"] is True
    assert resource["r0_bitwise_unchanged"] is True
    assert resource["radius_positive_finite"] is True
    assert resource["radius_fp16_shared_between_int8_fp32"] is True
    assert resource["radius_source_audit"]["old_source_new_class_row_count"] == 0
    assert resource["radius_source_audit"]["new_source_old_class_row_count"] == 0
    assert np.isfinite(resource["batch1_head_latency_mean_ms"])
    assert np.isfinite(resource["batch1_head_latency_p95_ms"])
    assert resource["latency_includes_argmax"] is True
    assert geometry["query_rows_used"] == 0

    for candidate_id in runner.D39_CANDIDATES:
        assert runner._full_state_refit_required(
            runner.CANDIDATE_SET_D39_V1, candidate_id, runner.D39_INT8
        ) is (candidate_id == runner.D39_INT8)

    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": candidate_id == runner.D39_INT8,
        }
        for candidate_id in runner.D39_CANDIDATES
    ]
    resource["matched_fp32_full_k10_argmax_change_count"] = 0
    resources = {
        runner.D39_INT8: {
            scenario: dict(resource)
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
    }
    selected, reason = runner._apply_full_k10_d39_gate(
        runner.D39_INT8, decisions, resources
    )
    assert selected == runner.D39_INT8
    assert reason is None
    for field, value in (
        ("query_role_oracle_access", True),
        ("matched_fp32_full_k10_argmax_change_count", 1),
        ("radius_positive_finite", False),
    ):
        broken_resources = copy.deepcopy(resources)
        broken_decisions = copy.deepcopy(decisions)
        broken_resources[runner.D39_INT8]["leo_rain_weak"][field] = value
        selected, reason = runner._apply_full_k10_d39_gate(
            runner.D39_INT8, broken_decisions, broken_resources
        )
        assert selected == runner.IDENTITY_CANDIDATE
        assert reason == (
            "FULL_K10_D39_RADIUS_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"
        )
