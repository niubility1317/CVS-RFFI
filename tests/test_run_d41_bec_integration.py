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
    new = [
        {"class_index": 6 + i, "class_handle": f"new-{i}"}
        for i in range(new_count)
    ]
    common = {"receiver": receiver, "seed": seed, "k_shot": 10}
    return ({**common, "registered_classes": old}, {**common, "registered_classes": old + new})


def _support_blocks():
    rng = np.random.default_rng(410718)
    classes = ("old_left", "old_right", "new_left", "new_right")
    labels, ranks, z_rows, fft_rows, rf_rows = [], [], [], [], []
    for class_index, label in enumerate(classes):
        for rank in range(10):
            z = np.zeros(160, np.float32); z[class_index] = 1
            fft = np.zeros(96, np.float32); fft[class_index] = 1
            rf = np.zeros(32, np.float32); rf[class_index] = 1
            z_rows.append(z + 0.003 * rng.normal(size=160).astype(np.float32))
            fft_rows.append(fft + 0.003 * rng.normal(size=96).astype(np.float32))
            rf_rows.append(rf + 0.003 * rng.normal(size=32).astype(np.float32))
            labels.append(label); ranks.append(rank)
    rows = {
        "labels": np.asarray(labels),
        "ranks": np.asarray(ranks, dtype=np.int64),
        "tokens": np.asarray([f"physical-{i:03d}" for i in range(len(labels))]),
    }
    return rows, np.asarray(z_rows), np.asarray(fft_rows), np.asarray(rf_rows)


def test_d41_cell_candidate_lock_cli_and_artifacts_are_frozen() -> None:
    runner._require_d41_development_cell(*_cell())
    for invalid in (_cell(receiver="3-19"), _cell(seed=713102), _cell(new_count=4)):
        with pytest.raises(runner.D25RunnerError):
            runner._require_d41_development_cell(*invalid)
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D41_V1)
    assert tuple(candidates) == runner.D41_CANDIDATES == (
        runner.IDENTITY_CANDIDATE, runner.D41_PROTONET_CDA, runner.DIAG_CANDIDATE,
        runner.D41_D40_INT8, runner.D41_INT8, runner.D41_FP32,
    )
    assert len(candidates) * 3 * 5 == 90
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D41_V1) == (runner.D41_INT8,)
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D41_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v19"
    assert {
        "d38_strong_b3_quantized_core_sha256",
        "d40_hnbr_core_sha256",
        "d41_bec_core_sha256",
        "d25_runner_sha256",
    } <= set(lock["source_closure"])
    assert lock["d41_formula_lock"]["views"] == ["full", "minus_z", "minus_fft", "minus_rf"]
    assert sum(row["eligible_positive_route"] for row in lock["candidates"]) == 1
    action = next(item for item in runner.build_parser()._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D41_V1 in action.choices
    forbidden = ("query", "truth", "role", "quota", "source", "clean")
    for names in (
        {item.dest.lower() for item in runner.build_parser()._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    ):
        assert not any(token in name for name in names for token in forbidden)
    for artifact in ("support_fold", "selection", "resource_matrix", "geometry_matrix", "receipt"):
        assert runner._artifact_schema(runner.CANDIDATE_SET_D41_V1, artifact) == f"cvs.phase2.d41.{artifact}.v1"


def test_d41_outer_fold_closes_views_source_ground_precision_and_pairwise(tmp_path: Path) -> None:
    rows, z, fft, rf = _support_blocks()
    ground_path = tmp_path / "phase1_int8_component.npz"
    ground_path.write_bytes(b"sealed-phase1-int8-component")
    ground_sha256 = runner._sha256_file(ground_path)
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D41_V1)[runner.D41_INT8]
    result = runner._evaluate_d41_fold(
        rows, z, fft, rf,
        old_classes=("old_left", "old_right"), new_classes=("new_left", "new_right"),
        held_ranks=(0, 1), candidate_id=runner.D41_INT8, config=config,
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
    assert result["ground_int8_audit"]["update_access"] is False
    assert Path(result["ground_int8_audit"]["path"]) == ground_path.resolve()
    assert len(result["pairwise_support_diagnostics"]) == 4
    assert result["actual_new_to_old_count"] == sum(
        item["new_old_margin"] <= 0 for item in result["pairwise_support_diagnostics"]
    )
    assert result["geometry_summary"]["support_view_names"] == ["full", "minus_z", "minus_fft", "minus_rf"]
    assert result["geometry_summary"]["physical_support_observation_multiplicity"] == 1
    assert result["geometry_summary"]["query_view"] == "full_only"
    assert result["resource"]["total_optimizer_steps"] == 30
    assert result["resource"]["stage2b_optimizer_steps"] == 20
    assert result["resource"]["stage2c_optimizer_steps"] == 10
    assert result["resource"]["estimated_bec_support_macs"] > result["resource"]["estimated_single_view_support_macs"]
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert len(result["training_trace"]) == 30
    full_resource, full_geometry = runner._full_d41_state_audit(
        rows, z, fft, rf,
        old_classes=("old_left", "old_right"), new_classes=("new_left", "new_right"),
        config=config, seed=713101, ground_int8_path=ground_path,
        expected_ground_int8_sha256=ground_sha256, device="cpu",
        scenario="leo_clear_weak",
    )
    assert full_resource["deployment_k_shot"] == 10
    assert full_resource["ground_int8_audit"]["entry_sha256"] == ground_sha256
    assert full_resource["ground_int8_audit"]["exit_sha256"] == ground_sha256
    assert full_geometry["query_rows_used"] == 0


def test_d41_ground_real_file_entry_and_exit_rehash_fail_closed(tmp_path: Path) -> None:
    ground_path = tmp_path / "phase1_int8_component.npz"
    ground_path.write_bytes(b"sealed-ground-v1")
    expected = runner._sha256_file(ground_path)
    entry = runner._ground_int8_entry_audit(ground_path, expected)
    ground_path.write_bytes(b"tampered-ground-v2")
    with pytest.raises(runner.D25RunnerError, match="changed during fit"):
        runner._ground_int8_exit_audit(ground_path, entry)
    with pytest.raises(runner.D25RunnerError, match="entry SHA256 drift"):
        runner._ground_int8_entry_audit(ground_path, expected)


def _metric(value: float, names: tuple[str, ...]):
    return {"overall_accuracy": value, "class_floor_accuracy": value, "per_class_accuracy": {name: value for name in names}}


def _selector_rows(candidate_id: str):
    old_names = tuple(f"old-{i}" for i in range(6)); new_names = tuple(f"new-{i}" for i in range(5))
    is_b3 = candidate_id == runner.DIAG_CANDIDATE
    is_d40 = candidate_id == runner.D41_D40_INT8
    is_bec = candidate_id in (runner.D41_INT8, runner.D41_FP32)
    before = .70 if is_b3 else .75 if (is_bec or is_d40) else .60
    after = .65 if is_b3 else .68 if is_d40 else .72 if is_bec else .55
    new = .65 if is_b3 else .72 if is_bec else .55
    count = 2 if is_b3 else 1 if is_bec else 3
    margin = .10 if is_b3 else .50 if is_bec else -.10
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
                "outer_held_new_intrusion_count": count, "actual_old_to_new_count": count,
                "actual_new_to_old_count": count, "new_new_confusion_count": count,
                "new_new_margin_min": margin, "new_old_margin_min": margin,
                "matched_fp32_before_argmax_change_count": 0,
                "matched_fp32_outer_argmax_change_count": 0,
                "deployment_precision": "int8" if candidate_id == runner.D41_INT8 else "fp32" if candidate_id == runner.D41_FP32 else "control",
                "registration_before_prediction_sha256": "bec-before" if is_bec else candidate_id,
                "outer_prediction_sha256": "bec-final" if is_bec else candidate_id,
                "training_trace": [{"phase": "B", "step": i} for i in range(30)],
                "ground_int8_audit": {"entry_sha256": "g", "exit_sha256": "g", "bitwise_unchanged": True, "update_access": False},
                "source_audit": {"old_source_row_count": 48, "new_source_row_count": 40,
                    "old_source_held_intersection_count": 0, "new_source_held_intersection_count": 0,
                    "old_source_new_class_row_count": 0, "new_source_old_class_row_count": 0,
                    "held_fit_row_count": 0, "query_rows_used": 0},
                "geometry_summary": {"support_view_names": ["full", "minus_z", "minus_fft", "minus_rf"],
                    "support_view_count": 4, "physical_support_observation_multiplicity": 1,
                    "loss_formula": "mean_four_class_macro_ce_plus_mean_three_js",
                    "macro_ce_coefficient": 1.0, "js_coefficient": 1.0, "query_view": "full_only",
                    "stage2b_before_artifact_immutable": True, "stage2c_continues_stage2b_fp32_state": True,
                    "stage2b_trainable_state": ["log_diag", "all_old_weights"],
                    "stage2c_trainable_state": ["log_diag", "all_old_weights", "all_new_weights"],
                    "new_weight_initialization": "stage2b_metric_full_view_transformed_centroid",
                    "final_all_registry_residual_int8": True, "ground_int8_update_access": False},
                "resource": {"peak_trainable_parameters": 3456, "adaptation_epochs": 30,
                    "total_optimizer_steps": 30, "stage2b_optimizer_steps": 20, "stage2c_optimizer_steps": 10,
                    "persistent_state_cap_pass": True, "resident_fp32_target_prototype_count": 0,
                    "formal_state_int8_only": True, "estimated_bec_support_macs": 100.,
                    "estimated_single_view_support_macs": 20., "query_rows_used_for_fit": 0,
                    "query_labels_used_for_fit": False, "query_role_oracle_access": False,
                    "query_true_batch_class_count_access": False, "query_class_quota_access": False,
                    "query_batch_global_assignment": False, "clean_sample_access": False, "source_sample_access": False},
            })
    return rows


def _positive_matrix():
    return {candidate: _selector_rows(candidate) for candidate in runner.D41_CANDIDATES}


def test_d41_selector_positive_and_physical_identity_closure() -> None:
    matrix = _positive_matrix()
    selected, decisions = runner._select_d41_candidate(matrix)
    assert selected == runner.D41_INT8
    decision = next(row for row in decisions if row["candidate_id"] == runner.D41_INT8)
    assert decision["eligible_positive_route"] is True
    assert all(decision[name] for name in (
        "before_gate", "after_old_gate", "new_gate", "confusion_gate", "margin_gate",
        "joint_gate", "precision_gate", "view_gate", "lifecycle_gate", "ground_gate",
        "source_gate", "resource_gate",
    ))
    broken = copy.deepcopy(matrix)
    broken[runner.D41_FP32][0]["held_physical_token_sha256"] = "f" * 64
    with pytest.raises(runner.D25RunnerError):
        runner._select_d41_candidate(broken)


@pytest.mark.parametrize(
    ("gate", "mutate"),
    [
        ("before_gate", lambda r: r["before_old"]["per_class_accuracy"].__setitem__("old-0", 0.0)),
        ("after_old_gate", lambda r: r["after_old"]["per_class_accuracy"].__setitem__("old-0", 0.0)),
        ("new_gate", lambda r: r["after_new"]["per_class_accuracy"].__setitem__("new-0", 0.0)),
        ("confusion_gate", lambda r: r.__setitem__("actual_new_to_old_count", 22)),
        ("margin_gate", lambda r: r.__setitem__("new_old_margin_min", -2.0)),
        ("joint_gate", lambda r: r.__setitem__("H_old_new", 0.0)),
        ("precision_gate", lambda r: r.__setitem__("matched_fp32_before_argmax_change_count", 1)),
        ("view_gate", lambda r: r["geometry_summary"].__setitem__("physical_support_observation_multiplicity", 4)),
        ("lifecycle_gate", lambda r: r["geometry_summary"].__setitem__("stage2b_before_artifact_immutable", False)),
        ("ground_gate", lambda r: r["ground_int8_audit"].__setitem__("exit_sha256", "changed")),
        ("source_gate", lambda r: r["source_audit"].__setitem__("old_source_held_intersection_count", 1)),
        ("resource_gate", lambda r: r["resource"].__setitem__("peak_trainable_parameters", 3455)),
    ],
)
def test_d41_each_selector_gate_has_an_independent_counterexample(gate, mutate) -> None:
    matrix = _positive_matrix(); mutate(matrix[runner.D41_INT8][0])
    selected, decisions = runner._select_d41_candidate(matrix)
    decision = next(row for row in decisions if row["candidate_id"] == runner.D41_INT8)
    assert selected == runner.IDENTITY_CANDIDATE
    assert decision[gate] is False
    assert decision["eligible_positive_route"] is False


def test_d41_selected_only_full_k10_gate_requires_exact_closure() -> None:
    resource = copy.deepcopy(_selector_rows(runner.D41_INT8)[0]["resource"])
    resource.update({
        "deployment_precision": "int8", "matched_fp32_full_k10_argmax_change_count": 0,
        "support_view_names": ["full", "minus_z", "minus_fft", "minus_rf"],
        "support_view_count": 4, "physical_support_observation_multiplicity": 1,
        "query_view": "full_only", "source_audit": {"old_source_row_count": 60,
            "new_source_row_count": 50, "old_source_new_class_row_count": 0,
            "new_source_old_class_row_count": 0, "query_rows_used": 0},
        "ground_int8_audit": {"entry_sha256": "g", "exit_sha256": "g", "bitwise_unchanged": True, "update_access": False},
        "batch1_head_latency_mean_ms": 1.0, "batch1_head_latency_p95_ms": 2.0,
        "batch1_head_latency_sample_count": 110, "latency_includes_argmax": True,
        "full_k10_refit_only_no_candidate_change": True,
    })
    resource["matched_fp32_before_argmax_change_count"] = 0
    resources = {runner.D41_INT8: {scenario: copy.deepcopy(resource) for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS}}
    decisions = [{"candidate_id": runner.D41_INT8, "eligible_positive_route": True}]
    selected, reason = runner._apply_full_k10_d41_gate(runner.D41_INT8, decisions, resources)
    assert selected == runner.D41_INT8 and reason is None
    resources[runner.D41_INT8][runner.legacy.FORMAL_LEO_WEAK_SCENARIOS[0]]["query_view"] = "masked"
    selected, reason = runner._apply_full_k10_d41_gate(runner.D41_INT8, [{"candidate_id": runner.D41_INT8, "eligible_positive_route": True}], resources)
    assert selected == runner.IDENTITY_CANDIDATE
    assert reason == "FULL_K10_D41_BEC_PRECISION_VIEW_GROUND_OR_RESOURCE_GATE_FAILED"
