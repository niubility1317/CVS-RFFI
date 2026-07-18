from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SCRIPTS = CODE / "scripts"
for value in (CODE, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_d25_support_only_concat as runner  # noqa: E402


def _support_blocks() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(360718)
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
            z_rows.append(z_center + 0.01 * rng.normal(size=160).astype(np.float32))
            fft_rows.append(fft_center + 0.01 * rng.normal(size=96).astype(np.float32))
            rf_rows.append(rf_center + 0.01 * rng.normal(size=32).astype(np.float32))
            labels.append(label)
            ranks.append(rank)
    return (
        {
            "labels": np.asarray(labels),
            "ranks": np.asarray(ranks, dtype=np.int64),
        },
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def test_d36_candidate_lock_is_exact_seven_by_three_by_five() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D36_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.D25_C0,
        runner.DIAG_CANDIDATE,
        runner.D33_B3_FAST,
        runner.D36_A,
        runner.D36_B,
        runner.D36_C,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D36_V1) == (
        runner.D36_CANDIDATES
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 105
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D36_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v14"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D36_V1
    assert "d36_compiled_joint_int8_core_sha256" in lock["source_closure"]


def test_d36_cli_is_callable_without_query_truth_role_or_quota_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D36_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    surfaces = (
        {item.dest.lower() for item in parser._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    )
    for names in surfaces:
        assert not any(token in name for name in names for token in forbidden)


def test_d36_outer_fold_emits_oof_int8_and_resource_evidence() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D36_V1)[runner.D36_A]
    result = runner._evaluate_d36_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        candidate_id=runner.D36_A,
        config=config,
        ground_anchor=None,
        ground_medoid_index=None,
        ground_anchor_sha256=None,
    )
    resource = result["resource"]
    geometry = result["geometry_summary"]
    assert geometry["inner_crossfit_no_self_participation"] is True
    assert geometry["inner_crossfit_rank_pairs"] == [[2, 3], [4, 5], [6, 7], [8, 9]]
    assert resource["inner_crossfit_fold_count"] == 4
    assert resource["oof_calibration_row_count"] == 32
    assert result["target_old_int8_prototypes_used_for_prediction"] is True
    assert result["target_new_int8_prototypes_used_for_prediction"] is True
    assert resource["resident_fp32_target_prototype_count"] == 0
    assert resource["peak_trainable_parameters"] <= 50_000
    assert resource["total_optimizer_steps"] <= 20
    assert resource["persistent_state_cap_pass"] is True
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert isinstance(result["old_score_columns_bitwise_unchanged"], bool)
    assert result["old_score_columns_bitwise_unchanged_semantics"] == (
        "measured_before_vs_after_compiled_target_old_scores"
    )
    # D36 intentionally recompiles the old head across Stage2-B/C.
    assert result["old_score_columns_bitwise_unchanged"] is False


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
    old_names: tuple[str, ...],
    new_names: tuple[str, ...],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for _ in runner.HELD_RANKS:
            result.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "before_old": _metric(value, old_names),
                    "after_old": _metric(value, old_names),
                    "after_new": _metric(value, new_names),
                    "b3_reference_old": _metric(min(value, 0.75), old_names),
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": safe,
                    "outer_held_zero_new_intrusion_pass": safe,
                    "outer_held_new_intrusion_count": 0 if safe else 1,
                    "new_physical_loso_all_reachable": safe,
                    "unreachable_new_class_count": 0 if safe else 1,
                    "target_old_int8_prototypes_used_for_prediction": safe,
                    "target_new_int8_prototypes_used_for_prediction": safe,
                }
            )
    return result


def test_d36_selector_uses_all_registered_classes_without_named_ids() -> None:
    old_names = ("old_alpha", "old_beta")
    new_names = ("new_alpha", "new_beta")
    values = {
        runner.IDENTITY_CANDIDATE: (0.50, False),
        runner.D25_C0: (0.55, False),
        runner.DIAG_CANDIDATE: (0.70, False),
        runner.D33_B3_FAST: (0.72, False),
        runner.D36_A: (0.78, True),
        runner.D36_B: (0.80, True),
        runner.D36_C: (0.82, True),
    }
    folds = {
        candidate_id: _candidate_rows(
            candidate_id,
            value=value,
            safe=safe,
            old_names=old_names,
            new_names=new_names,
        )
        for candidate_id, (value, safe) in values.items()
    }
    selected, decisions = runner._select_d36_candidate(folds)
    assert selected == runner.D36_C
    selected_decision = next(
        row for row in decisions if row["candidate_id"] == runner.D36_C
    )
    assert selected_decision["d36_all_old_class_floor_gate_pass"] is True
    assert selected_decision["d36_all_new_class_floor_gate_pass"] is True
    assert selected_decision["d36_generic_floor_gate_pass"] is True
    assert not any(
        token in key.lower()
        for key in selected_decision
        for token in ("14-7", "09f8", "f608")
    )


def test_d36_full_k10_gate_reuses_generic_floor_and_resource_closure() -> None:
    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": True,
            "d36_hard_gate_pass": True,
            "d36_classwise_comparator_gate_pass": True,
            "d36_generic_floor_gate_pass": True,
            "d36_joint_comparator_gate_pass": True,
        }
        for candidate_id in runner.D36_CANDIDATES
    ]
    resources = {
        candidate_id: {
            scenario: {
                "quantized_old_head_classwise_noninferior_to_b3": True,
                "old_support_non_degradation_pass": True,
                "full_support_old_to_new_intrusion_count": 0,
                "full_k10_crossfit_fold_count": 5,
                "full_k10_crossfit_no_self_participation": True,
                "target_old_int8_prototypes_used_for_prediction": True,
                "target_new_int8_prototypes_used_for_prediction": True,
                "resident_fp32_target_prototype_count": 0,
                "peak_trainable_parameters": 1_440,
                "total_optimizer_steps": 12,
                "persistent_state_cap_pass": True,
                "dense_query_graph_bytes": 0,
                "latency_includes_argmax": True,
                "query_rows_used_for_fit": 0,
            }
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        for candidate_id in runner.D36_CANDIDATES
    }
    selected, reason = runner._apply_full_k10_d36_gate(
        runner.D36_C, decisions, resources
    )
    assert selected == runner.D36_C
    assert reason is None
    resources[runner.D36_C]["leo_rain_weak"]["persistent_state_cap_pass"] = False
    selected, reason = runner._apply_full_k10_d36_gate(
        runner.D36_C, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_D36_OOF_SAFETY_GENERIC_FLOOR_OR_RESOURCE_GATE_FAILED"


def test_d36_launcher_is_unique_and_sha_closed() -> None:
    launcher = (CODE / "scripts" / "launch_d36_compiled_joint_int8_20260718.sh").read_text(
        encoding="utf-8"
    )
    assert 'OUTPUT="$RUN/output/support_screen_v1"' in launcher
    assert "--candidate-set d36_v1" in launcher
    assert "EXPECTED_D36_CORE_SHA256=" in launcher
    assert "EXPECTED_D36_FISHER_SHA256=" in launcher
    assert "EXPECTED_RUNNER_SHA256=" in launcher
    assert "query" not in launcher.lower()

    retry = (
        CODE / "scripts" / "launch_d36_compiled_joint_int8_retry1_20260718.sh"
    ).read_text(encoding="utf-8")
    assert 'OUTPUT="$RUN/output/support_screen_retry1"' in retry
    assert "--candidate-set d36_v1" in retry
    assert (
        "EXPECTED_D36_CORE_SHA256="
        "32d8d5364c363513d9d9f54ed49575999df9a80bbc96edb06f3829ffc7f5198a"
    ) in retry
    assert "query" not in retry.lower()

    retry2 = (
        CODE / "scripts" / "launch_d36_compiled_joint_int8_retry2_20260718.sh"
    ).read_text(encoding="utf-8")
    assert 'OUTPUT="$RUN/output/support_screen_retry2"' in retry2
    assert "PYTHON=/home/szu2070436088/.conda/envs/SDG-SEI/bin/python" in retry2
    assert (
        "EXPECTED_D36_CORE_SHA256="
        "e53b164b17da0ffcdf62b2f1024c931917d6d590fc5938b6f77a388270c3e09e"
    ) in retry2
    assert "query" not in retry2.lower()

    retry3 = (
        CODE / "scripts" / "launch_d36_compiled_joint_int8_retry3_20260718.sh"
    ).read_text(encoding="utf-8")
    assert 'OUTPUT="$RUN/output/support_screen_retry3"' in retry3
    assert "PYTHON=/home/szu2070436088/.conda/envs/SDG-SEI/bin/python" in retry3
    assert (
        "EXPECTED_PREDICTOR_BUNDLE_SHA256="
        "0b17420162b3c9698e9e8c2fc5c5edcb374719d10c3bfcc9a8ffc20e00a63383"
    ) in retry3
    assert "query" not in retry3.lower()
