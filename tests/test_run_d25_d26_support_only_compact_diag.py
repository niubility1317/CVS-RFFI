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
from cvsrffi.stage2_multimodal_compact_diag import (  # noqa: E402
    D26CompactDiagConfig,
)


def test_d26_candidate_lock_matrix_and_historical_sets_are_stable() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D26_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D26_A,
        runner.D26_B,
        runner.D26_C,
    )
    assert [
        (candidates[name].stage2b_steps, candidates[name].stage2c_steps)
        for name in runner.D26_CANDIDATES
    ] == [(15, 0), (15, 10), (15, 15)]
    assert 6 * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 90
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D26_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v3"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D26_V1
    assert lock["selection_baseline"] == runner.D25_C0
    assert lock["d26_core_git_commit"] == runner.D26_CORE_GIT_COMMIT
    assert "d26_compact_diag_core_sha256" in lock["source_closure"]
    assert "diag_cosine_feature_operator_sha256" in lock["source_closure"]
    assert "d25_c3_core_sha256" not in lock["source_closure"]
    assert all(
        candidates[name].bias_guard_mode == "joint_bias0"
        for name in runner.D26_CANDIDATES
    )

    strict_candidates = runner.preregistered_candidates(
        runner.CANDIDATE_SET_D26_V2
    )
    assert tuple(strict_candidates) == tuple(candidates)
    assert all(
        strict_candidates[name].bias_guard_mode
        == "pre_registration_old_only"
        for name in runner.D26_CANDIDATES
    )
    assert all(
        min(strict_candidates[name].new_group_bias_grid) == -12.0
        for name in runner.D26_CANDIDATES
    )
    strict_lock = runner._candidate_lock(
        strict_candidates, runner.CANDIDATE_SET_D26_V2
    )
    assert strict_lock["schema"] == "cvs.phase2.d25.candidate_lock.v4"
    assert strict_lock["candidate_set"] == runner.CANDIDATE_SET_D26_V2

    assert tuple(runner.preregistered_candidates()) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D25_C1,
        runner.D25_C2,
    )
    assert tuple(runner.preregistered_candidates(runner.CANDIDATE_SET_C3_V1)) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.C3_A,
        runner.C3_B,
        runner.C3_C,
    )
    assert runner._candidate_lock(runner.preregistered_candidates())["schema"] == (
        "cvs.phase2.d25.candidate_lock.v1"
    )
    assert runner._candidate_lock(
        runner.preregistered_candidates(runner.CANDIDATE_SET_C3_V1),
        runner.CANDIDATE_SET_C3_V1,
    )["schema"] == "cvs.phase2.d25.candidate_lock.v2"


def test_d26_cli_has_no_query_source_or_clean_surface() -> None:
    parser = runner.build_parser()
    candidate_action = next(
        action for action in parser._actions if action.dest == "candidate_set"
    )
    assert candidate_action.default == runner.CANDIDATE_SET_D25_V4
    assert tuple(candidate_action.choices) == (
        runner.CANDIDATE_SET_D25_V4,
        runner.CANDIDATE_SET_C3_V1,
        runner.CANDIDATE_SET_D26_V1,
        runner.CANDIDATE_SET_D26_V2,
    )
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    destinations = {action.dest.lower() for action in parser._actions}
    parameters = {name.lower() for name in inspect.signature(runner.run).parameters}
    for names in (destinations, parameters):
        assert not any(token in name for name in names for token in forbidden)


def _synthetic_blocks() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(260718)
    classes = tuple(f"old_{index}" for index in range(3)) + tuple(
        f"new_{index}" for index in range(2)
    )
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        z_center = rng.normal(size=160)
        fft_center = rng.normal(size=96)
        rf_center = rng.normal(size=32)
        for rank in range(10):
            labels.append(label)
            ranks.append(rank)
            z_rows.append(z_center + 0.02 * rng.normal(size=160))
            fft_rows.append(fft_center + 0.02 * rng.normal(size=96))
            rf_rows.append(rf_center + 0.02 * rng.normal(size=32))
    return (
        {"labels": np.asarray(labels), "ranks": np.asarray(ranks, dtype=np.int64)},
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def test_real_d26_fold_freezes_old_scores_and_logs_support_only_bias() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_blocks()
    result = runner._evaluate_d26_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        held_ranks=(0, 1),
        candidate_id=runner.D26_A,
        config=D26CompactDiagConfig(stage2b_steps=1, stage2c_steps=0),
    )
    assert result["old_prefix_sha256_before"] == result["old_prefix_sha256_after"]
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["fit_k_shot"] == 8
    assert result["resource"]["total_optimizer_steps"] == 1
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["new_group_bias_support_only_audit"]["query_rows_used"] == 0
    assert result["geometry_summary"]["bias_applied_to_new_suffix_only"] is True
    assert len(result["training_trace"]) == 3  # B step0+1, C closed-form step0.


def test_full_d26_resource_and_geometry_are_deployment_bounded() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_blocks()
    resource, geometry = runner._full_d26_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        config=D26CompactDiagConfig(stage2b_steps=1, stage2c_steps=0),
    )

    assert resource["total_optimizer_steps"] == 1
    assert resource["total_adaptation_epochs"] == 1
    assert resource["peak_trainable_parameters"] == 4 * 288
    assert resource["persistent_state_cap_pass"] is True
    assert resource["trainable_parameter_cap_pass"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["estimated_score_mac_ratio_vs_identity_single_qknn"] < 1.0
    assert resource["batch1_head_latency_sample_count"] == len(rows["labels"])
    assert len(resource["complete_loss_trace"]) == 3
    assert geometry["bias_applied_to_new_suffix_only"] is True
    assert "weights" not in geometry
    assert "features" not in geometry
    assert "prototypes" not in geometry


def _metric(value: float, labels: tuple[str, ...]) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {label: value for label in labels},
    }


def _candidate_rows(
    candidate_id: str, value: float, *, support_pass: bool, steps: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for _ in runner.HELD_RANKS:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "before_old": _metric(value, ("old-a", "old-b")),
                    "after_old": _metric(value, ("old-a", "old-b")),
                    "after_new": _metric(value, ("new-a", "new-b")),
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": True,
                    "old_support_non_degradation_pass": support_pass,
                    "resource": {"total_optimizer_steps": steps},
                }
            )
    return rows


def test_d26_selector_uses_c0_gates_and_b3_as_reference_only() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(
            runner.IDENTITY_CANDIDATE, 0.40, support_pass=True, steps=0
        ),
        runner.DIAG_CANDIDATE: _candidate_rows(
            runner.DIAG_CANDIDATE, 0.70, support_pass=True, steps=60
        ),
        runner.D25_C0: _candidate_rows(
            runner.D25_C0, 0.50, support_pass=True, steps=0
        ),
        runner.D26_A: _candidate_rows(runner.D26_A, 0.65, support_pass=True, steps=15),
        runner.D26_B: _candidate_rows(runner.D26_B, 0.65, support_pass=True, steps=25),
        runner.D26_C: _candidate_rows(runner.D26_C, 0.65, support_pass=True, steps=30),
    }
    selected, decisions = runner._select_d26_candidate(folds)
    assert selected == runner.D26_A
    d26 = next(row for row in decisions if row["candidate_id"] == runner.D26_A)
    b3 = next(row for row in decisions if row["candidate_id"] == runner.DIAG_CANDIDATE)
    assert d26["eligible_positive_route"] is True
    assert d26["B3_performance_reference_only"] is True
    assert d26["mean_H_delta_vs_B3"] < 0.0
    assert b3["diagnostic_only"] is True
    assert b3["eligible_positive_route"] is False


def test_d26_full_k10_failure_revokes_selected_route() -> None:
    decisions = [{"candidate_id": runner.D26_A, "eligible_positive_route": True}]
    resources = {
        runner.D26_A: {
            scenario: {"old_support_non_degradation_pass": scenario != "leo_rain_weak"}
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
    }
    selected, reason = runner._apply_full_k10_d26_old_support_gate(
        runner.D26_A, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"
    assert decisions[0]["eligible_positive_route"] is False
