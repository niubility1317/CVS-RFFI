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
    common: dict[str, object] = {"receiver": receiver, "seed": seed, "k_shot": 10}
    return (
        {**common, "registered_classes": old},
        {**common, "registered_classes": old + new},
    )


def _support_blocks() -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(400718)
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


def test_d40_preopen_cell_and_exact_candidate_lock_are_closed() -> None:
    runner._require_d40_development_cell(*_cell_manifest())
    for manifests in (
        _cell_manifest(receiver="3-19"),
        _cell_manifest(seed=713102),
        _cell_manifest(new_count=2),
        _cell_manifest(new_count=10),
    ):
        with pytest.raises(runner.D25RunnerError):
            runner._require_d40_development_cell(*manifests)

    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D40_V1)
    assert tuple(candidates) == runner.D40_CANDIDATES
    assert runner.D40_CANDIDATES == (
        runner.IDENTITY_CANDIDATE,
        runner.D40_PROTONET_CDA,
        runner.DIAG_CANDIDATE,
        runner.D40_D38_B_INT8,
        runner.D40_INT8,
        runner.D40_FP32,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D40_V1) == (
        runner.D40_INT8,
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 90
    assert candidates[runner.D40_D38_B_INT8].core.arm == "B"
    assert candidates[runner.D40_INT8].deploy_precision == "int8"
    assert candidates[runner.D40_FP32].deploy_precision == "fp32"
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D40_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v18"
    assert {
        "d38_strong_b3_quantized_core_sha256",
        "d40_hnbr_core_sha256",
        "d25_runner_sha256",
    } <= set(lock["source_closure"])
    assert lock["d40_formula_lock"]["temperature"] == 18.0
    assert lock["d40_formula_lock"]["stage2b"].startswith(
        "D38_arm_A_stage2B_"
    )
    assert lock["d40_formula_lock"]["stage2c"].startswith("zero_step")
    assert sum(row["eligible_positive_route"] for row in lock["candidates"]) == 1


def test_d40_cli_artifacts_and_run_surface_keep_query_sealed() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D40_V1 in action.choices
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
        assert runner._artifact_schema(runner.CANDIDATE_SET_D40_V1, artifact) == (
            f"cvs.phase2.d40.{artifact}.v1"
        )


def test_d40_outer_fold_closes_physical_source_pairwise_precision_and_resources() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D40_V1)[
        runner.D40_INT8
    ]
    result = runner._evaluate_d40_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        candidate_id=runner.D40_INT8,
        config=config,
        seed=713101,
        device="cpu",
        scenario="leo_clear_weak",
        outer_fold=0,
    )
    source = result["direction_source_audit"]
    assert source["old_source_row_count"] == 16
    assert source["new_source_row_count"] == 16
    assert source["old_source_held_intersection_count"] == 0
    assert source["new_source_held_intersection_count"] == 0
    assert source["held_direction_fit_row_count"] == 0
    assert len(source["old_source_physical_token_sha256"]) == 64
    assert len(source["new_source_physical_token_sha256"]) == 64
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["old_base_prefix_bitwise_unchanged"] is True
    assert len(result["outer_prediction_sha256"]) == 64
    assert len(result["pairwise_support_diagnostics"]) == 4
    assert result["resource"]["peak_trainable_parameters"] <= 2016
    assert result["resource"]["adaptation_epochs"] == 20
    assert result["resource"]["total_optimizer_steps"] == 20
    assert result["resource"]["stage2c_optimizer_steps"] == 0
    assert result["resource"]["persistent_state_cap_pass"] is True
    assert result["resource"]["resident_fp32_target_prototype_count"] == 0
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["resource"]["query_role_oracle_access"] is False
    assert result["resource"]["clean_sample_access"] is False
    assert result["resource"]["source_sample_access"] is False
    assert len(result["training_trace"]) == 20
    assert result["geometry_summary"]["stage2c_solver"] == (
        "zero_step_synchronous_new_hnbr"
    )
    assert result["geometry_summary"]["old_hnbr_synchronous"] is True
    assert result["geometry_summary"]["new_hnbr_synchronous"] is True


def test_d40_exact_strong_b3_pairwise_enrichment_uses_matched_held_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, _, _, _ = _support_blocks()
    diag_features = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    new_scores = np.asarray(
        [
            [0.1, 0.2, 0.8, 0.3],
            [0.6, 0.2, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.9],
            [0.1, 0.7, 0.8, 0.6],
        ],
        dtype=np.float32,
    )
    old_scores = np.asarray(
        [
            [0.9, 0.1, 0.2, 0.0],
            [0.1, 0.0, 0.9, 0.2],
            [0.2, 0.8, 0.1, 0.0],
            [0.1, 0.7, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    calls: list[np.ndarray] = []

    def fake_diag_scores(state, features, *, include_new):
        assert state == {"sealed": "diag"}
        assert include_new is True
        calls.append(np.array(features, copy=True))
        return new_scores if len(calls) == 1 else old_scores

    monkeypatch.setattr(runner.legacy, "_diag_scores", fake_diag_scores)
    row: dict[str, object] = {"geometry_summary": {}, "resource": {}}
    runner._enrich_d40_strong_b3_pairwise(
        row,
        rows,
        diag_features,
        {"sealed": "diag"},
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        scenario="leo_clear_weak",
        outer_fold=0,
    )

    assert len(calls) == 2
    assert np.array_equal(calls[0], diag_features[[20, 21, 30, 31]])
    assert np.array_equal(calls[1], diag_features[[0, 1, 10, 11]])
    pairwise = row["pairwise_support_diagnostics"]
    assert [item["physical_sample_id"] for item in pairwise] == [
        "physical-020",
        "physical-021",
        "physical-030",
        "physical-031",
    ]
    assert [item["true_new_handle"] for item in pairwise] == [
        "new_left",
        "new_left",
        "new_right",
        "new_right",
    ]
    assert np.allclose(
        [item["new_new_margin"] for item in pairwise],
        [0.5, -0.1, 0.6, -0.2],
    )
    assert row["new_new_confusion_count"] == 2
    assert row["new_new_margin_min"] == pytest.approx(-0.2)
    assert row["new_old_margin_min"] == pytest.approx(-0.2)
    assert row["outer_held_new_intrusion_count"] == 1
    assert row["resource"]["pairwise_support_diagnostic_row_count"] == 4
    assert row["geometry_summary"]["query_rows_used"] == 0


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
    is_d38 = candidate_id == runner.D40_D38_B_INT8
    is_d40 = candidate_id in (runner.D40_INT8, runner.D40_FP32)
    value = 0.70 if is_diag else 0.80 if is_d38 else 0.90 if is_d40 else 0.60
    intrusion = 2 if is_diag else 3 if is_d38 else 1 if is_d40 else 4
    confusion = 2 if is_diag else 3 if is_d38 else 1 if is_d40 else 4
    margin = 0.10 if is_diag else 0.0 if is_d38 else 0.50 if is_d40 else -0.1
    trace = [{"phase": "stage2b", "step": index} for index in range(20)]
    rows: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for fold_index, _ in enumerate(runner.HELD_RANKS):
            held_identity = hashlib.sha256(
                f"{scenario}:{fold_index}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "fold_index": fold_index,
                    "held_ranks": list(runner.HELD_RANKS[fold_index]),
                    "held_physical_token_count": 8,
                    "held_physical_token_sha256": held_identity,
                    "before_old": _metric(value, old_names),
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
                        if candidate_id == runner.D40_INT8
                        else "fp32"
                        if candidate_id == runner.D40_FP32
                        else "control"
                    ),
                    "outer_prediction_sha256": (
                        "same-d40-outer" if is_d40 else f"outer-{candidate_id}"
                    ),
                    "registration_before_prediction_sha256": (
                        "same-d40-before" if is_d40 else f"before-{candidate_id}"
                    ),
                    "training_trace": trace,
                    "old_score_columns_bitwise_unchanged": True,
                    "old_base_prefix_bitwise_unchanged": True,
                    "direction_source_audit": {
                        "old_source_held_intersection_count": 0,
                        "new_source_held_intersection_count": 0,
                        "old_source_new_class_row_count": 0,
                        "new_source_old_class_row_count": 0,
                        "held_direction_fit_row_count": 0,
                        "query_rows_used": 0,
                    },
                    "target_old_int8_prototypes_used_for_prediction": (
                        candidate_id == runner.D40_INT8
                    ),
                    "target_new_int8_prototypes_used_for_prediction": (
                        candidate_id == runner.D40_INT8
                    ),
                    "geometry_summary": {
                        "stage2c_solver": "zero_step_synchronous_new_hnbr",
                        "hnbr_temperature": 18.0,
                        "stable_softmax_subtracts_row_max": True,
                        "positive_projection_only": True,
                        "old_hnbr_synchronous": True,
                        "new_hnbr_synchronous": True,
                        "new_hnbr_uses_residualized_new_direction_as_negative": False,
                        "new_hnbr_old_negative_precision": "int8_decoded",
                        "new_hnbr_old_negative_matches_before_int8_decode": True,
                        "old_fp32_reference_used_as_new_hnbr_negative": False,
                        "old_prefix_bitwise_unchanged": True,
                        "label_permutation_equivariant": True,
                        "class_id_specific_branch": False,
                        "fp32_target_direction_stored_in_formal_state": False,
                        "query_rows_used": 0,
                    },
                    "resource": {
                        "peak_trainable_parameters": 2016,
                        "adaptation_epochs": 20,
                        "total_optimizer_steps": 20,
                        "stage2c_optimizer_steps": 0,
                        "persistent_state_cap_pass": True,
                        "estimated_hnbr_support_macs": 10_000,
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
                    },
                }
            )
    return rows


def _positive_folds() -> dict[str, list[dict[str, object]]]:
    return {
        candidate_id: _selector_rows(candidate_id)
        for candidate_id in runner.D40_CANDIDATES
    }


def test_d40_matrix_is_90_rows_and_physical_mismatch_fails_closed() -> None:
    folds = _positive_folds()
    assert len(runner._validate_d40_matrix_rows(folds)) == 15
    assert sum(len(rows) for rows in folds.values()) == 90
    selected, decisions = runner._select_d40_candidate(folds)
    assert selected == runner.D40_INT8
    assert [
        row["candidate_id"] for row in decisions if row["eligible_positive_route"]
    ] == [runner.D40_INT8]

    bad_physical = copy.deepcopy(folds)
    bad_physical[runner.D40_FP32][0]["held_physical_token_sha256"] = "f" * 64
    with pytest.raises(runner.D25RunnerError):
        runner._validate_d40_matrix_rows(bad_physical)

    bad_identity = copy.deepcopy(folds)
    bad_identity[runner.D40_PROTONET_CDA][0]["candidate_id"] = "wrong"
    with pytest.raises(runner.D25RunnerError):
        runner._validate_d40_matrix_rows(bad_identity)


@pytest.mark.parametrize(
    "broken_gate",
    (
        "before_classwise",
        "before_aggregate",
        "after_old",
        "intrusion",
        "seen_new",
        "confusion_cap",
        "new_floor",
        "new_margin",
        "joint_per_key",
        "joint_aggregate",
        "internal_fp32",
        "explicit_fp32",
        "old_prefix",
        "source_protocol",
        "old_fp32_negative",
        "stage2b_step_count",
        "zero_hnbr_macs",
        "resource_query_flag",
    ),
)
def test_d40_selector_each_preregistered_gate_has_a_counterexample(
    broken_gate: str,
) -> None:
    folds = copy.deepcopy(_positive_folds())
    rows = folds[runner.D40_INT8]
    if broken_gate == "before_classwise":
        rows[0]["before_old"] = _metric(0.60, ("old_alpha", "old_beta"))
    elif broken_gate == "before_aggregate":
        for row in rows:
            row["before_old"]["overall_accuracy"] = 0.70
    elif broken_gate == "after_old":
        rows[0]["after_old"] = _metric(0.60, ("old_alpha", "old_beta"))
    elif broken_gate == "intrusion":
        for row in rows:
            row["outer_held_new_intrusion_count"] = 3
    elif broken_gate == "seen_new":
        rows[0]["after_new"]["overall_accuracy"] = 0.60
    elif broken_gate == "confusion_cap":
        for row in rows:
            row["new_new_confusion_count"] = 3
    elif broken_gate == "new_floor":
        for row in rows:
            row["after_new"]["per_class_accuracy"]["new_alpha"] = 0.60
    elif broken_gate == "new_margin":
        rows[0]["new_new_margin_min"] = 0.0
    elif broken_gate == "joint_per_key":
        rows[0]["H_old_new"] = 0.60
    elif broken_gate == "joint_aggregate":
        for row in rows:
            row["H_old_new"] = 0.70
            row["joint_floor"] = 0.70
    elif broken_gate == "internal_fp32":
        rows[0]["matched_fp32_outer_argmax_change_count"] = 1
    elif broken_gate == "explicit_fp32":
        folds[runner.D40_FP32][0]["outer_prediction_sha256"] = "drift"
    elif broken_gate == "old_prefix":
        rows[0]["old_base_prefix_bitwise_unchanged"] = False
    elif broken_gate == "source_protocol":
        rows[0]["direction_source_audit"]["old_source_held_intersection_count"] = 1
    elif broken_gate == "old_fp32_negative":
        rows[0]["geometry_summary"][
            "old_fp32_reference_used_as_new_hnbr_negative"
        ] = True
    elif broken_gate == "stage2b_step_count":
        rows[0]["resource"]["total_optimizer_steps"] = 19
    elif broken_gate == "zero_hnbr_macs":
        rows[0]["resource"]["estimated_hnbr_support_macs"] = 0
    elif broken_gate == "resource_query_flag":
        rows[0]["resource"]["query_role_oracle_access"] = True
    else:  # pragma: no cover
        raise AssertionError(broken_gate)
    selected, decisions = runner._select_d40_candidate(folds)
    assert selected == runner.IDENTITY_CANDIDATE
    assert not any(row["eligible_positive_route"] for row in decisions)


def test_d40_full_k10_is_selected_only_and_full_gate_is_fail_closed() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D40_V1)[
        runner.D40_INT8
    ]
    resource, geometry = runner._full_d40_state_audit(
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
    assert resource["formal_state_int8_only"] is True
    assert resource["resident_fp32_target_prototype_count"] == 0
    assert resource["old_prefix_bitwise_unchanged"] is True
    assert resource["old_base_prefix_bitwise_unchanged"] is True
    assert resource["stage2c_optimizer_steps"] == 0
    assert resource["direction_source_audit"]["query_rows_used"] == 0
    assert np.isfinite(resource["batch1_head_latency_mean_ms"])
    assert geometry["query_rows_used"] == 0

    for candidate_id in runner.D40_CANDIDATES:
        assert runner._full_state_refit_required(
            runner.CANDIDATE_SET_D40_V1, candidate_id, runner.D40_INT8
        ) is (candidate_id == runner.D40_INT8)

    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": candidate_id == runner.D40_INT8,
        }
        for candidate_id in runner.D40_CANDIDATES
    ]
    resources = {
        runner.D40_INT8: {
            scenario: dict(resource)
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
    }
    selected, reason = runner._apply_full_k10_d40_gate(
        runner.D40_INT8, copy.deepcopy(decisions), resources
    )
    assert selected == runner.D40_INT8
    assert reason is None

    for field, value in (
        ("query_role_oracle_access", True),
        ("matched_fp32_full_k10_argmax_change_count", 1),
        ("stage2c_optimizer_steps", 1),
        ("adaptation_epochs", 19),
        ("estimated_hnbr_support_macs", 0),
    ):
        broken_resources = copy.deepcopy(resources)
        broken_resources[runner.D40_INT8]["leo_rain_weak"][field] = value
        selected, reason = runner._apply_full_k10_d40_gate(
            runner.D40_INT8, copy.deepcopy(decisions), broken_resources
        )
        assert selected == runner.IDENTITY_CANDIDATE
        assert reason == (
            "FULL_K10_D40_HNBR_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"
        )
