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
    rng = np.random.default_rng(350718)
    classes = ("old_0", "old_1", "old_2", "new_0", "new_1")
    labels: list[str] = []
    ranks: list[int] = []
    features: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        center = np.zeros(288, dtype=np.float32)
        center[class_index * 16 : (class_index + 1) * 16] = 4.0
        for rank in range(10):
            features.append(center + 0.02 * rng.normal(size=288))
            labels.append(label)
            ranks.append(rank)
    value = np.asarray(features, dtype=np.float32)
    return (
        {"labels": np.asarray(labels), "ranks": np.asarray(ranks, dtype=np.int64)},
        value[:, :160],
        value[:, 160:256],
        value[:, 256:],
    )


def test_d35_candidate_lock_is_exact_seven_by_three_by_five() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D35_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.D25_C0,
        runner.DIAG_CANDIDATE,
        runner.D33_B3_FAST,
        runner.D35_A,
        runner.D35_B,
        runner.D35_C,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D35_V1) == (
        runner.D35_CANDIDATES
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 105
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D35_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v13"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D35_V1
    assert "d35_dense_safe_registration_core_sha256" in lock["source_closure"]
    d35_rows = [
        row for row in lock["candidates"] if row["candidate_id"] in runner.D35_CANDIDATES
    ]
    assert [row["config"]["registration"]["arm"] for row in d35_rows] == [
        "A",
        "B",
        "C",
    ]
    assert all(row["config"]["all_new_classes_globally_visible"] for row in d35_rows)


def test_d35_cli_has_no_query_truth_role_quota_source_or_clean_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D35_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    surfaces = (
        {item.dest.lower() for item in parser._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    )
    for names in surfaces:
        assert not any(token in name for name in names for token in forbidden)


def test_d35_fold_emits_frozen_prefix_safety_reachability_and_resources() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D35_V1)[runner.D35_B]
    result = runner._evaluate_d35_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        held_ranks=(0, 1),
        candidate_id=runner.D35_B,
        config=config,
    )
    resource = result["resource"]
    assert result["old_score_prefix_bitwise_unchanged"] is True
    assert result["fit_old_support_non_degradation_pass"] is True
    assert isinstance(result["outer_held_zero_new_intrusion_pass"], bool)
    assert set(result["new_class_reachability"]) == {"new_0", "new_1"}
    assert resource["peak_trainable_parameters"] == 0
    assert resource["total_optimizer_steps"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_role_oracle_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False
    assert resource["all_new_classes_globally_visible"] is True
    assert resource["total_post_backbone_macs_per_query"] == (
        resource["old_prefix_macs_per_query"]
        + resource["dense_safe_extra_macs_per_query"]
    )
    assert resource["dense_safe_extra_macs_per_query"] == (
        resource["query_prototype_dot_macs"]
    )
    assert resource["total_post_backbone_macs_per_query"] < (
        resource["identity_single_qknn_macs_same_registered_count"]
    )


def _metric(value: float, prefix: str) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {f"{prefix}_0": value, f"{prefix}_1": value},
    }


def _candidate_rows(candidate_id: str, *, value: float, safe: bool) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for _ in runner.HELD_RANKS:
            result.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "before_old": _metric(value, "old"),
                    "after_old": _metric(value, "old"),
                    "after_new": _metric(value, "new"),
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": safe,
                    "old_score_prefix_bitwise_unchanged": safe,
                    "fit_old_support_non_degradation_pass": safe,
                    "outer_held_zero_new_intrusion_pass": safe,
                    "outer_held_new_intrusion_count": 0 if safe else 1,
                    "new_physical_loso_all_reachable": safe,
                    "unreachable_new_class_count": 0 if safe else 1,
                    "resource": {"active_closed_form_scalars": 3000},
                }
            )
    return result


def test_d35_selector_uses_hard_gates_and_joint_comparators() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(runner.IDENTITY_CANDIDATE, value=0.5, safe=False),
        runner.D25_C0: _candidate_rows(runner.D25_C0, value=0.55, safe=False),
        runner.DIAG_CANDIDATE: _candidate_rows(runner.DIAG_CANDIDATE, value=0.70, safe=False),
        runner.D33_B3_FAST: _candidate_rows(runner.D33_B3_FAST, value=0.72, safe=False),
        runner.D35_A: _candidate_rows(runner.D35_A, value=0.95, safe=False),
        runner.D35_B: _candidate_rows(runner.D35_B, value=0.80, safe=True),
        runner.D35_C: _candidate_rows(runner.D35_C, value=0.82, safe=True),
    }
    selected, decisions = runner._select_d35_candidate(folds)
    assert selected == runner.D35_C
    a = next(row for row in decisions if row["candidate_id"] == runner.D35_A)
    assert a["d35_hard_gate_pass"] is False
    assert a["eligible_positive_route"] is False
    c = next(row for row in decisions if row["candidate_id"] == runner.D35_C)
    assert c["d35_classwise_comparator_gate_pass"] is True
    assert set(c["classwise_comparator_thresholds"]) == {"old", "new"}


def test_d35_full_k10_gate_reuses_outer_intrusion_and_reachability() -> None:
    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": True,
            "outer_held_zero_new_intrusion_all_folds": True,
        }
        for candidate_id in runner.D35_CANDIDATES
    ]
    resources = {
        candidate_id: {
            scenario: {
                "old_support_non_degradation_pass": True,
                "old_score_prefix_bitwise_unchanged": True,
                "unreachable_new_class_count": 0,
                "all_new_classes_globally_visible": True,
                "peak_trainable_parameters": 0,
                "total_optimizer_steps": 0,
                "persistent_state_cap_pass": True,
                "dense_query_graph_bytes": 0,
                "latency_includes_argmax": True,
                "query_rows_used_for_fit": 0,
            }
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        for candidate_id in runner.D35_CANDIDATES
    }
    selected, reason = runner._apply_full_k10_d35_gate(
        runner.D35_C, decisions, resources
    )
    assert selected == runner.D35_C
    assert reason is None
    resources[runner.D35_C]["leo_rain_weak"]["unreachable_new_class_count"] = 1
    decisions[-1]["eligible_positive_route"] = True
    selected, reason = runner._apply_full_k10_d35_gate(
        runner.D35_C, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_D35_SAFETY_REACHABILITY_OR_RESOURCE_GATE_FAILED"


def test_d35_launcher_is_unique_and_sha_closed() -> None:
    launcher = (CODE / "scripts" / "launch_d35_dense_safe_20260718.sh").read_text(
        encoding="utf-8"
    )
    assert 'OUTPUT="$RUN/output/support_screen_v1"' in launcher
    assert "--candidate-set d35_v1" in launcher
    assert "EXPECTED_D35_CORE_SHA256=" in launcher
    assert "EXPECTED_RUNNER_SHA256=" in launcher
    assert "__D35_" not in launcher
    assert "query" not in launcher.lower()
