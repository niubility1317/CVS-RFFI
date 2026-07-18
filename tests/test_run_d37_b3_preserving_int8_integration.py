from __future__ import annotations

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


def _support_blocks() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(370718)
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
            z_rows.append(z_center + 0.005 * rng.normal(size=160).astype(np.float32))
            fft_rows.append(fft_center + 0.005 * rng.normal(size=96).astype(np.float32))
            rf_rows.append(rf_center + 0.005 * rng.normal(size=32).astype(np.float32))
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


def test_d37_candidate_lock_is_exact_seven_by_three_by_five() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D37_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.D25_C0,
        runner.DIAG_CANDIDATE,
        runner.D33_B3_FAST,
        runner.D37_A,
        runner.D37_B,
        runner.D37_C,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D37_V1) == (
        runner.D37_CANDIDATES
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 105
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D37_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v15"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D37_V1
    assert "d37_b3_preserving_int8_core_sha256" in lock["source_closure"]


def test_d37_cli_has_no_query_truth_role_quota_or_clean_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D37_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    surfaces = (
        {item.dest.lower() for item in parser._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    )
    for names in surfaces:
        assert not any(token in name for name in names for token in forbidden)


def test_d37_outer_fold_emits_rank_pair_oof_and_append_only_int8_evidence() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D37_V1)[runner.D37_A]
    result = runner._evaluate_d37_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        candidate_id=runner.D37_A,
        config=config,
    )
    resource = result["resource"]
    geometry = result["geometry_summary"]
    assert geometry["inner_crossfit_rank_pairs"] == [[2, 3], [4, 5], [6, 7], [8, 9]]
    assert geometry["inner_crossfit_no_self_participation"] is True
    assert geometry["oof_uses_physical_labels_only"] is True
    assert resource["inner_crossfit_fold_count"] == 4
    assert resource["oof_calibration_row_count"] == 32
    assert resource["oof_calibration_old_row_count"] == 16
    assert resource["oof_calibration_new_row_count"] == 16
    assert result["oof_feasible_interval_pass"] is True
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert resource["old_prefix_bitwise_unchanged"] is True
    assert resource["resident_fp32_target_prototype_count"] == 0
    assert resource["peak_trainable_parameters"] == 0
    assert resource["total_optimizer_steps"] == 0
    assert resource["persistent_state_cap_pass"] is True
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_rows_used_for_fit"] == 0


@pytest.mark.parametrize(
    "message",
    (
        "empty OOF feasible interval: lower=1.0, upper=0.0",
        "OOF feasible interval contains no deployable FP16 offset",
    ),
)
def test_d37_empty_or_nonrepresentable_interval_fails_candidate_not_run(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D37_V1)[runner.D37_B]

    def _fail(*args: object, **kwargs: object) -> object:
        raise runner.D37B3PreservingInt8Error(message)

    monkeypatch.setattr(runner, "fit_oof_feasible_offset_d37", _fail)
    result = runner._evaluate_d37_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        candidate_id=runner.D37_B,
        config=config,
    )
    assert result["oof_feasible_interval_pass"] is False
    assert result["oof_failure_reason"] == message
    assert result["resource"]["oof_feasible_interval_pass"] is False
    assert result["resource"][
        "infeasible_state_scored_only_by_base_score_for_diagnostics"
    ] is True
    assert np.isfinite(float(result["after_old"]["overall_accuracy"]))


def test_d37_full_k10_audit_uses_five_physical_oof_folds() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D37_V1)[runner.D37_A]
    resource, geometry = runner._full_d37_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        config=config,
    )
    assert resource["full_k10_crossfit_fold_count"] == 5
    assert resource["oof_crossfit_fold_count"] == 5
    assert resource["oof_calibration_row_count"] == 40
    assert resource["oof_source"] == runner.D37_OOF_SOURCE
    assert resource["oof_feasible_interval_pass"] is True
    assert resource["old_prefix_bitwise_unchanged"] is True
    assert geometry["full_k10_crossfit_rank_pairs"] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]


def _metric(value: float, names: tuple[str, ...]) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {name: value for name in names},
    }


def _candidate_rows(
    candidate_id: str,
    *,
    value: float,
    safe: bool,
    feasible: bool,
    old_names: tuple[str, ...],
    new_names: tuple[str, ...],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for fold_index, _ in enumerate(runner.HELD_RANKS):
            result.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "fold_index": fold_index,
                    "before_old": _metric(value, old_names),
                    "after_old": _metric(value, old_names),
                    "after_new": _metric(value, new_names),
                    "b3_reference_old": _metric(min(value, 0.75), old_names),
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": safe,
                    "oof_feasible_interval_pass": feasible,
                    "outer_held_zero_new_intrusion_pass": safe,
                    "outer_held_new_intrusion_count": 0 if safe else 1,
                    "new_physical_loso_all_reachable": safe,
                    "unreachable_new_class_count": 0 if safe else 1,
                    "target_old_int8_prototypes_used_for_prediction": safe,
                    "target_new_int8_prototypes_used_for_prediction": safe,
                }
            )
    return result


def test_d37_selector_requires_feasible_interval_and_strict_comparators() -> None:
    old_names = ("old_alpha", "old_beta")
    new_names = ("new_alpha", "new_beta")
    values = {
        runner.IDENTITY_CANDIDATE: (0.50, False, False),
        runner.D25_C0: (0.55, False, False),
        runner.DIAG_CANDIDATE: (0.70, False, False),
        runner.D33_B3_FAST: (0.72, False, False),
        runner.D37_A: (0.78, True, True),
        runner.D37_B: (0.80, True, True),
        runner.D37_C: (0.82, True, True),
    }
    folds = {
        candidate_id: _candidate_rows(
            candidate_id,
            value=value,
            safe=safe,
            feasible=feasible,
            old_names=old_names,
            new_names=new_names,
        )
        for candidate_id, (value, safe, feasible) in values.items()
    }
    selected, decisions = runner._select_d37_candidate(folds)
    assert selected == runner.D37_C
    selected_decision = next(row for row in decisions if row["candidate_id"] == selected)
    assert selected_decision["oof_feasible_interval_all_folds"] is True
    assert selected_decision["quantized_old_head_classwise_noninferior_to_strong_b3"] is True
    assert selected_decision["d37_joint_comparator_gate_pass"] is True

    folds[runner.D37_C][0]["oof_feasible_interval_pass"] = False
    selected, _ = runner._select_d37_candidate(folds)
    assert selected == runner.D37_B
    for candidate_id in runner.D37_CANDIDATES:
        folds[candidate_id][0]["oof_feasible_interval_pass"] = False
    selected, _ = runner._select_d37_candidate(folds)
    assert selected == runner.D25_C0


def test_d37_selector_rejects_single_fold_b3_regression_and_stronger_identity() -> None:
    old_names = ("old_alpha", "old_beta")
    new_names = ("new_alpha", "new_beta")
    values = {
        runner.IDENTITY_CANDIDATE: (0.50, False, False),
        runner.D25_C0: (0.55, False, False),
        runner.DIAG_CANDIDATE: (0.70, False, False),
        runner.D33_B3_FAST: (0.72, False, False),
        runner.D37_A: (0.78, True, True),
        runner.D37_B: (0.80, True, True),
        runner.D37_C: (0.82, True, True),
    }
    folds = {
        candidate_id: _candidate_rows(
            candidate_id,
            value=value,
            safe=safe,
            feasible=feasible,
            old_names=old_names,
            new_names=new_names,
        )
        for candidate_id, (value, safe, feasible) in values.items()
    }
    folds[runner.D37_C][0]["before_old"]["per_class_accuracy"]["old_alpha"] = 0.0
    _, decisions = runner._select_d37_candidate(folds)
    d37_c = next(row for row in decisions if row["candidate_id"] == runner.D37_C)
    assert d37_c["quantized_old_head_classwise_noninferior_to_b3"] is False
    assert d37_c["eligible_positive_route"] is False

    folds[runner.D37_C][0]["before_old"]["per_class_accuracy"]["old_alpha"] = 0.82
    for row in folds[runner.IDENTITY_CANDIDATE]:
        row["before_old"] = _metric(0.95, old_names)
        row["after_old"] = _metric(0.95, old_names)
        row["after_new"] = _metric(0.95, new_names)
        row["H_old_new"] = 0.95
        row["joint_floor"] = 0.95
    selected, decisions = runner._select_d37_candidate(folds)
    assert selected == runner.D25_C0
    assert all(
        not row["eligible_positive_route"]
        for row in decisions
        if row["candidate_id"] in runner.D37_CANDIDATES
    )


def test_d37_intrusion_counts_all_old_rows_not_only_before_correct() -> None:
    predictions = np.asarray(["new-a", "old-b", "new-b", "new-a"])
    old_mask = np.asarray([True, True, True, False])
    assert runner._d37_old_to_new_intrusion_count(
        predictions, old_mask, ("new-a", "new-b")
    ) == 2


def test_d37_full_k10_gate_requires_feasible_offset_and_resource_closure() -> None:
    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": True,
            "d37_hard_gate_pass": True,
            "d37_classwise_comparator_gate_pass": True,
            "d37_joint_comparator_gate_pass": True,
        }
        for candidate_id in runner.D37_CANDIDATES
    ]
    resources = {
        candidate_id: {
            scenario: {
                "oof_feasible_interval_pass": True,
                "quantized_old_head_classwise_noninferior_to_b3": True,
                "old_support_non_degradation_pass": True,
                "full_support_old_to_new_intrusion_count": 0,
                "full_k10_crossfit_fold_count": 5,
                "full_k10_crossfit_no_self_participation": True,
                "old_prefix_bitwise_unchanged": True,
                "old_score_prefix_bitwise_unchanged": True,
                "unreachable_new_class_count": 0,
                "target_old_int8_prototypes_used_for_prediction": True,
                "target_new_int8_prototypes_used_for_prediction": True,
                "resident_fp32_target_prototype_count": 0,
                "peak_trainable_parameters": 0,
                "total_optimizer_steps": 0,
                "persistent_state_cap_pass": True,
                "dense_query_graph_bytes": 0,
                "latency_includes_argmax": True,
                "query_rows_used_for_fit": 0,
            }
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        for candidate_id in runner.D37_CANDIDATES
    }
    selected, reason = runner._apply_full_k10_d37_gate(
        runner.D37_C, decisions, resources
    )
    assert selected == runner.D37_C
    assert reason is None
    resources[runner.D37_C]["leo_rain_weak"]["oof_feasible_interval_pass"] = False
    selected, reason = runner._apply_full_k10_d37_gate(
        runner.D37_C, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_D37_OOF_B3_SAFETY_REACHABILITY_OR_RESOURCE_GATE_FAILED"
