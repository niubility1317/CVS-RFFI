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


def _cell(receiver: str = "20-1", seed: int = 713101, new_count: int = 5):
    old = [{"class_index": i, "class_handle": f"old-{i}"} for i in range(6)]
    new = [{"class_index": 6 + i, "class_handle": f"new-{i}"} for i in range(new_count)]
    common = {"receiver": receiver, "seed": seed, "k_shot": 10}
    return ({**common, "registered_classes": old}, {**common, "registered_classes": old + new})


def _support_blocks():
    rng = np.random.default_rng(420718)
    classes = ("old_left", "old_right", "new_left", "new_right")
    labels, ranks, tokens, z_rows, fft_rows, rf_rows = [], [], [], [], [], []
    for class_index, label in enumerate(classes):
        for rank in range(10):
            z = np.zeros(160, np.float32); z[class_index] = 2.0
            fft = np.zeros(96, np.float32); fft[class_index] = 1.5
            rf = np.zeros(32, np.float32); rf[class_index] = 1.0
            z_rows.append(z + 0.08 * rng.normal(size=160).astype(np.float32))
            fft_rows.append(fft + 0.08 * rng.normal(size=96).astype(np.float32))
            rf_rows.append(rf + 0.08 * rng.normal(size=32).astype(np.float32))
            labels.append(label); ranks.append(rank); tokens.append(f"{label}-physical-{rank}")
    rows = {
        "labels": np.asarray(labels),
        "ranks": np.asarray(ranks, dtype=np.int64),
        "tokens": np.asarray(tokens),
    }
    return rows, np.asarray(z_rows), np.asarray(fft_rows), np.asarray(rf_rows)


def test_d42_cell_candidate_lock_cli_and_artifacts_are_frozen() -> None:
    runner._require_d42_development_cell(*_cell())
    for invalid in (_cell(receiver="3-19"), _cell(seed=713102), _cell(new_count=4)):
        with pytest.raises(runner.D25RunnerError):
            runner._require_d42_development_cell(*invalid)
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D42_V1)
    assert tuple(candidates) == runner.D42_CANDIDATES == (
        runner.IDENTITY_CANDIDATE, runner.D42_PROTONET_CDA, runner.DIAG_CANDIDATE,
        runner.D42_D40_INT8, runner.D42_D41_INT8, runner.D42_INT8, runner.D42_FP32,
    )
    assert len(candidates) * 3 * 5 == 105
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D42_V1) == (runner.D42_INT8,)
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D42_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v20"
    assert {
        "diag_cosine_feature_operator_sha256", "d38_strong_b3_quantized_core_sha256",
        "d40_hnbr_core_sha256", "d41_bec_core_sha256",
        "d42_unified_shrinkage_lda_core_sha256", "d25_runner_sha256",
    } <= set(lock["source_closure"])
    assert lock["d42_formula_lock"]["query_view"] == "full_288d_only"
    assert lock["d42_formula_lock"]["lda_solver"] == "lsqr"
    assert lock["d42_formula_lock"]["sklearn_locked_version"] == "1.7.2"
    assert runner.D42_SKLEARN_LOCKED_VERSION == runner.SKLEARN_RUNTIME_VERSION == "1.7.2"
    assert sum(row["eligible_positive_route"] for row in lock["candidates"]) == 1
    action = next(item for item in runner.build_parser()._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D42_V1 in action.choices
    for artifact in ("support_fold", "selection", "resource_matrix", "geometry_matrix", "receipt"):
        assert runner._artifact_schema(runner.CANDIDATE_SET_D42_V1, artifact) == f"cvs.phase2.d42.{artifact}.v1"
    run_source = inspect.getsource(runner.run)
    assert '"sklearn_locked_version": D42_SKLEARN_LOCKED_VERSION' in run_source
    assert '"sklearn_version_lock_pass": str(sklearn.__version__)' in run_source


def test_d42_outer_and_full_k10_close_state_source_ground_and_precision(tmp_path: Path) -> None:
    rows, z, fft, rf = _support_blocks()
    ground_path = tmp_path / "phase1_int8_component.npz"
    ground_path.write_bytes(b"sealed-phase1-int8-component")
    ground_sha256 = runner._sha256_file(ground_path)
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D42_V1)[runner.D42_INT8]
    result = runner._evaluate_d42_fold(
        rows, z, fft, rf,
        old_classes=("old_left", "old_right"), new_classes=("new_left", "new_right"),
        held_ranks=(0, 1), candidate_id=runner.D42_INT8, config=config,
        seed=713101, ground_int8_path=ground_path,
        expected_ground_int8_sha256=ground_sha256, device="cpu",
        scenario="leo_clear_weak", outer_fold=0,
    )
    assert result["source_audit"]["old_source_row_count"] == 16
    assert result["source_audit"]["new_source_row_count"] == 16
    assert result["source_audit"]["old_source_held_intersection_count"] == 0
    assert result["source_audit"]["new_source_held_intersection_count"] == 0
    assert result["ground_int8_audit"]["entry_sha256"] == ground_sha256
    assert result["ground_int8_audit"]["exit_sha256"] == ground_sha256
    assert result["ground_int8_audit"]["bitwise_unchanged"] is True
    assert len(result["pairwise_support_diagnostics"]) == 8
    assert {item["true_role"] for item in result["pairwise_support_diagnostics"]} == {"old", "new"}
    assert result["final_argmax_new_to_old_count"] + result["final_argmax_new_to_new_count"] == int(
        round(4 - sum(result["after_new"]["per_class_accuracy"].values()) * 2)
    )
    assert result["int8_fp32_margin_sign_flip_count"] == 0
    assert np.isfinite(result["int8_fp32_max_score_abs_error"])
    assert result["geometry_summary"]["query_view"] == "full_288d_only"
    assert result["geometry_summary"]["old_log_diag_bitwise_unchanged"] is True
    assert result["geometry_summary"]["old_log_diag_before_sha256"] == result["geometry_summary"]["old_log_diag_final_sha256"]
    resource = result["resource"]
    assert resource["coefficient_shape"] == [4, 288]
    assert resource["coefficient_dtype"] == "int8"
    assert resource["intercept_dtype"] == "float16"
    assert resource["formal_target_vectors_int8_no_fp32_sidecar"] is True
    assert "formal_state_int8_only" not in resource
    assert resource["adaptation_epochs"] == resource["total_optimizer_steps"] == 20
    assert resource["lda_optimizer_steps"] == 0
    assert resource["lda_solver"] == "lsqr" and resource["lda_shrinkage"] == "auto"
    assert resource["lda_priors"] == "equal_1_over_registered_class_count"
    assert result["source_audit"]["sklearn_version"] == "1.7.2"
    assert result["source_audit"]["sklearn_locked_version"] == "1.7.2"
    assert result["source_audit"]["sklearn_version_lock_pass"] is True
    assert result["geometry_summary"]["sklearn_version"] == "1.7.2"
    assert resource["sklearn_version"] == "1.7.2"
    assert resource["sklearn_locked_version"] == "1.7.2"
    assert resource["sklearn_version_lock_pass"] is True

    full_resource, full_geometry = runner._full_d42_state_audit(
        rows, z, fft, rf,
        old_classes=("old_left", "old_right"), new_classes=("new_left", "new_right"),
        config=config, seed=713101, ground_int8_path=ground_path,
        expected_ground_int8_sha256=ground_sha256, device="cpu", scenario="leo_clear_weak",
    )
    assert full_resource["deployment_k_shot"] == 10
    assert full_resource["coefficient_shape"] == [4, 288]
    assert full_resource["formal_target_vectors_int8_no_fp32_sidecar"] is True
    assert "formal_state_int8_only" not in full_resource
    assert full_resource["ground_int8_audit"]["entry_sha256"] == ground_sha256
    assert full_resource["ground_int8_audit"]["exit_sha256"] == ground_sha256
    assert full_geometry["query_rows_used"] == 0


def test_d42_ground_real_file_entry_and_exit_rehash_fail_closed(tmp_path: Path) -> None:
    ground_path = tmp_path / "phase1_int8_component.npz"
    ground_path.write_bytes(b"sealed-ground-v1")
    expected = runner._sha256_file(ground_path)
    entry = runner._ground_int8_entry_audit(ground_path, expected)
    ground_path.write_bytes(b"tampered-ground-v2")
    with pytest.raises(runner.D25RunnerError, match="changed during fit"):
        runner._ground_int8_exit_audit(ground_path, entry)


def _metric(value: float, names: tuple[str, ...]):
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {name: value for name in names},
    }


def _selector_rows(candidate_id: str):
    old_names = tuple(f"old-{i}" for i in range(6)); new_names = tuple(f"new-{i}" for i in range(5))
    is_b3 = candidate_id == runner.DIAG_CANDIDATE
    is_d42 = candidate_id in (runner.D42_INT8, runner.D42_FP32)
    before = .70 if is_b3 else .80 if is_d42 else .60
    after = .65 if is_b3 else .78 if is_d42 else .55
    new = .65 if is_b3 else .76 if is_d42 else .55
    count = 2 if is_b3 else 1 if is_d42 else 3
    margin = .10 if is_b3 else .50 if is_d42 else -.10
    rows = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for fold_index, held in enumerate(runner.HELD_RANKS):
            identity = hashlib.sha256(f"{scenario}:{fold_index}".encode()).hexdigest()
            rows.append({
                "candidate_id": candidate_id, "scenario": scenario, "fold_index": fold_index,
                "held_ranks": list(held), "held_physical_token_count": 22,
                "held_physical_token_sha256": identity,
                "before_old": _metric(before, old_names), "after_old": _metric(after, old_names),
                "after_new": _metric(new, new_names), "H_old_new": min(after, new),
                "forgetting": before - after, "joint_floor": min(after, new),
                "final_argmax_old_to_new_count": count,
                "final_argmax_new_to_old_count": count,
                "final_argmax_new_to_new_count": count,
                "pairwise_old_to_new_misorder_count": count,
                "pairwise_new_to_old_misorder_count": count,
                "pairwise_new_to_new_misorder_count": count,
                "old_new_margin_min": margin, "new_old_margin_min": margin,
                "new_new_margin_min": margin,
                "matched_fp32_before_argmax_change_count": 0,
                "matched_fp32_outer_argmax_change_count": 0,
                "int8_fp32_margin_sign_flip_count": 0,
                "int8_fp32_max_score_abs_error": 0.01,
                "deployment_precision": "int8" if candidate_id == runner.D42_INT8 else "fp32" if candidate_id == runner.D42_FP32 else "control",
                "registration_before_prediction_sha256": "d42-before" if is_d42 else candidate_id,
                "outer_prediction_sha256": "d42-final" if is_d42 else candidate_id,
                "training_trace": [{"phase": "B", "step": i} for i in range(20)],
                "ground_int8_audit": {"entry_sha256": "g", "exit_sha256": "g", "bitwise_unchanged": True, "update_access": False},
                "source_audit": {"old_source_row_count": 48, "new_source_row_count": 40,
                    "old_source_held_intersection_count": 0, "new_source_held_intersection_count": 0,
                    "old_source_new_class_row_count": 0, "new_source_old_class_row_count": 0,
                    "held_fit_row_count": 0, "query_rows_used": 0,
                    "sklearn_version": "1.7.2", "sklearn_locked_version": "1.7.2",
                    "sklearn_version_lock_pass": True},
                "geometry_summary": {"old_only_metric_helper_called_once": True,
                    "old_only_metric_new_support_argument_count": 0,
                    "before_materialized_pre_stage2c": True,
                    "before_materialization_optimizer_step": 20,
                    "stage2c_log_diag_frozen": True,
                    "old_log_diag_bitwise_unchanged": True, "stage2b_lda_scope": "old_only",
                    "stage2c_lda_scope": "all_registered", "lda_solver": "lsqr",
                    "lda_shrinkage": "auto", "lda_priors": "equal_1_over_registered_class_count",
                    "lda_coefficient_semantics": "precision_weighted_target_prototype_w_c_equals_sigma_inverse_mu_c",
                    "formal_old_target_vectors_residual_int8": True,
                    "formal_new_target_vectors_residual_int8": True,
                    "formal_target_vector_semantics": "precision_weighted_target_prototype",
                    "class_means_persisted_in_formal_state": False,
                    "shared_covariance_persisted_in_formal_state": False,
                    "label_permutation_equivariant": True, "class_id_specific_branch": False,
                    "query_view": "full_288d_only", "ground_int8_update_access": False,
                    "sklearn_version": "1.7.2", "sklearn_locked_version": "1.7.2",
                    "sklearn_version_lock_pass": True},
                "resource": {"peak_trainable_parameters": 3456, "adaptation_epochs": 20,
                    "total_optimizer_steps": 20, "lda_optimizer_steps": 0,
                    "persistent_state_bytes": 5000, "persistent_state_cap_pass": True,
                    "resident_fp32_target_coefficient_count": 0,
                    "formal_coefficients_residual_int8": True,
                    "formal_target_vectors_int8_no_fp32_sidecar": True,
                    "coefficient_dtype": "int8", "intercept_dtype": "float16",
                    "coefficient_row_count": 11, "coefficient_dimension": 288,
                    "estimated_lda_fit_macs": 100., "estimated_macs_per_query": 288.,
                    "dense_query_graph_bytes": 0, "query_rows_used_for_fit": 0,
                    "query_labels_used_for_fit": False, "query_role_oracle_access": False,
                    "query_true_batch_class_count_access": False, "query_class_quota_access": False,
                    "query_batch_global_assignment": False, "clean_sample_access": False,
                    "source_sample_access": False, "sklearn_version": "1.7.2",
                    "sklearn_locked_version": "1.7.2", "sklearn_version_lock_pass": True},
            })
    return rows


def _positive_matrix():
    return {candidate: _selector_rows(candidate) for candidate in runner.D42_CANDIDATES}


def test_d42_selector_positive_and_105_row_physical_sha_closure() -> None:
    matrix = _positive_matrix()
    assert sum(len(rows) for rows in matrix.values()) == 105
    selected, decisions = runner._select_d42_candidate(matrix)
    assert selected == runner.D42_INT8
    decision = next(row for row in decisions if row["candidate_id"] == runner.D42_INT8)
    assert decision["eligible_positive_route"] is True
    assert all(decision[name] for name in (
        "before_gate", "after_old_gate", "new_gate", "joint_gate", "confusion_gate",
        "pairwise_gate", "precision_gate", "lifecycle_gate", "ground_gate",
        "source_gate", "state_gate", "resource_gate",
    ))
    broken = copy.deepcopy(matrix)
    broken[runner.D42_FP32][0]["held_physical_token_sha256"] = "f" * 64
    with pytest.raises(runner.D25RunnerError, match="physical-token closure"):
        runner._select_d42_candidate(broken)


@pytest.mark.parametrize(
    ("gate", "mutate"),
    [
        ("before_gate", lambda r: r["before_old"]["per_class_accuracy"].__setitem__("old-0", 0.0)),
        ("after_old_gate", lambda r: r["after_old"]["per_class_accuracy"].__setitem__("old-0", 0.0)),
        ("new_gate", lambda r: r["after_new"]["per_class_accuracy"].__setitem__("new-0", 0.0)),
        ("joint_gate", lambda r: r.__setitem__("H_old_new", 0.0)),
        ("confusion_gate", lambda r: r.__setitem__("final_argmax_old_to_new_count", 33)),
        ("pairwise_gate", lambda r: r.__setitem__("old_new_margin_min", -2.0)),
        ("precision_gate", lambda r: r.__setitem__("int8_fp32_margin_sign_flip_count", 1)),
        ("lifecycle_gate", lambda r: r["geometry_summary"].__setitem__("old_log_diag_bitwise_unchanged", False)),
        ("ground_gate", lambda r: r["ground_int8_audit"].__setitem__("exit_sha256", "changed")),
        ("source_gate", lambda r: r["source_audit"].__setitem__("old_source_held_intersection_count", 1)),
        ("state_gate", lambda r: r["resource"].__setitem__("coefficient_dtype", "float32")),
        ("resource_gate", lambda r: r["resource"].__setitem__("total_optimizer_steps", 21)),
    ],
)
def test_d42_each_selector_gate_has_an_independent_counterexample(gate, mutate) -> None:
    matrix = _positive_matrix(); mutate(matrix[runner.D42_INT8][0])
    selected, decisions = runner._select_d42_candidate(matrix)
    decision = next(row for row in decisions if row["candidate_id"] == runner.D42_INT8)
    assert selected == runner.IDENTITY_CANDIDATE
    assert decision[gate] is False
    assert decision["eligible_positive_route"] is False


@pytest.mark.parametrize(
    ("gate", "scope"),
    [
        ("lifecycle_gate", "geometry_summary"),
        ("source_gate", "source_audit"),
        ("resource_gate", "resource"),
    ],
)
def test_d42_sklearn_version_drift_fails_each_independent_lock(gate, scope) -> None:
    matrix = _positive_matrix()
    matrix[runner.D42_INT8][0][scope]["sklearn_version"] = "1.7.3"
    selected, decisions = runner._select_d42_candidate(matrix)
    decision = next(row for row in decisions if row["candidate_id"] == runner.D42_INT8)
    assert selected == runner.IDENTITY_CANDIDATE
    assert decision[gate] is False


def test_d42_selected_only_full_k10_gate_requires_exact_closure() -> None:
    resource = copy.deepcopy(_selector_rows(runner.D42_INT8)[0]["resource"])
    resource.update({
        "deployment_precision": "int8", "matched_fp32_before_argmax_change_count": 0,
        "matched_fp32_full_k10_argmax_change_count": 0, "int8_fp32_margin_sign_flip_count": 0,
        "int8_fp32_max_score_abs_error": .01, "lda_solver": "lsqr", "lda_shrinkage": "auto",
        "lda_priors": "equal_1_over_registered_class_count", "query_view": "full_288d_only",
        "source_audit": {"old_source_row_count": 60, "new_source_row_count": 50,
            "old_source_new_class_row_count": 0, "new_source_old_class_row_count": 0,
            "query_rows_used": 0, "sklearn_version": "1.7.2",
            "sklearn_locked_version": "1.7.2", "sklearn_version_lock_pass": True},
        "ground_int8_audit": {"entry_sha256": "g", "exit_sha256": "g", "bitwise_unchanged": True, "update_access": False},
        "batch1_head_latency_mean_ms": 1.0, "batch1_head_latency_p95_ms": 2.0,
        "batch1_head_latency_sample_count": 110, "latency_includes_argmax": True,
        "full_k10_refit_only_no_candidate_change": True,
    })
    resources = {runner.D42_INT8: {scenario: copy.deepcopy(resource) for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS}}
    decisions = [{"candidate_id": runner.D42_INT8, "eligible_positive_route": True}]
    selected, reason = runner._apply_full_k10_d42_gate(runner.D42_INT8, decisions, resources)
    assert selected == runner.D42_INT8 and reason is None
    resources[runner.D42_INT8][runner.legacy.FORMAL_LEO_WEAK_SCENARIOS[0]]["query_view"] = "masked"
    selected, reason = runner._apply_full_k10_d42_gate(runner.D42_INT8, decisions, resources)
    assert selected == runner.IDENTITY_CANDIDATE
    assert reason == "FULL_K10_D42_STATE_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"
