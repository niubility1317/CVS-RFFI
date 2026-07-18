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
    rng = np.random.default_rng(340718)
    classes = ("old_0", "old_1", "old_2", "new_0", "new_1")
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        center = np.zeros(288, dtype=np.float32)
        center[class_index * 12 : (class_index + 1) * 12] = 3.0
        for rank in range(10):
            feature = center + 0.04 * rng.normal(size=288)
            labels.append(label)
            ranks.append(rank)
            z_rows.append(feature[:160])
            fft_rows.append(feature[160:256])
            rf_rows.append(feature[256:])
    return (
        {"labels": np.asarray(labels), "ranks": np.asarray(ranks, dtype=np.int64)},
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def test_d34_candidate_lock_is_exact_seven_by_three_by_five() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D34_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.D25_C0,
        runner.DIAG_CANDIDATE,
        runner.D33_B3_FAST,
        runner.D34_A,
        runner.D34_B,
        runner.D34_C,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D34_V1) == (
        runner.D34_CANDIDATES
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 105
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D34_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v12"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D34_V1
    closure = lock["source_closure"]
    assert "d34_collision_local_registration_core_sha256" in closure
    assert "d34_b3_fisher_closed_form_core_sha256" in closure
    assert "d33_spherical_registration_core_sha256" in closure
    d34_rows = [
        row for row in lock["candidates"] if row["candidate_id"] in runner.D34_CANDIDATES
    ]
    assert [row["config"]["registration"]["arm"] for row in d34_rows] == [
        "A",
        "B",
        "C",
    ]
    assert all(row["eligible_positive_route"] for row in d34_rows)


def test_d34_cli_has_no_query_truth_role_quota_source_or_clean_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D34_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    surfaces = (
        {item.dest.lower() for item in parser._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    )
    for names in surfaces:
        assert not any(token in name for name in names for token in forbidden)


def test_d34_fold_emits_frozen_prefix_collision_trace_and_resources() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D34_V1)[runner.D34_B]
    result = runner._evaluate_d34_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        held_ranks=(0, 1),
        candidate_id=runner.D34_B,
        config=config,
    )
    resource = result["resource"]
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["old_score_prefix_bitwise_unchanged"] is True
    assert result["old_support_non_degradation_pass"] is True
    assert isinstance(result["old_loso_zero_intrusion_pass"], bool)
    assert result["collision_edge_count"] >= 2
    assert result["unreachable_edge_count"] >= 0
    assert result["training_trace"]
    assert resource["peak_trainable_parameters"] == 0
    assert resource["total_optimizer_steps"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["query_role_oracle_access"] is False
    assert resource["query_class_quota_access"] is False
    assert resource["clean_sample_access"] is False
    assert resource["source_sample_access"] is False
    assert resource["target_new_int8_prototypes_used_for_prediction"] is True
    assert resource["total_post_backbone_macs_per_query"] < (
        resource["identity_single_qknn_macs_same_registered_count"]
    )
    geometry = result["geometry_summary"]
    assert geometry["old_score_prefix_bitwise_unchanged"] is True


def test_full_d34_k10_audit_includes_latency_and_old_safety() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D34_V1)[runner.D34_C]
    resource, geometry = runner._full_d34_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        config=config,
    )
    assert resource["deployment_k_shot"] == 10
    assert resource["old_score_prefix_bitwise_unchanged"] is True
    assert resource["old_support_non_degradation_pass"] is True
    assert resource["persistent_state_cap_pass"] is True
    assert resource["latency_includes_argmax"] is True
    assert resource["batch1_head_latency_mean_ms"] > 0.0
    assert geometry["old_score_prefix_bitwise_unchanged"] is True


def _metric(value: float, prefix: str) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {f"{prefix}_0": value, f"{prefix}_1": value},
    }


def _candidate_rows(
    candidate_id: str,
    *,
    value: float,
    safe: bool,
    edge_count: int,
) -> list[dict[str, object]]:
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
                    "old_support_non_degradation_pass": safe,
                    "old_loso_zero_intrusion_pass": safe,
                    "collision_edge_count": edge_count,
                    "unreachable_edge_count": 0,
                    "resource": {"total_optimizer_steps": 0},
                }
            )
    return result


def test_d34_selector_hard_rejects_old_intrusion_then_ranks_joint_floor() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(
            runner.IDENTITY_CANDIDATE, value=0.5, safe=False, edge_count=0
        ),
        runner.D25_C0: _candidate_rows(
            runner.D25_C0, value=0.55, safe=False, edge_count=0
        ),
        runner.DIAG_CANDIDATE: _candidate_rows(
            runner.DIAG_CANDIDATE, value=0.7, safe=False, edge_count=0
        ),
        runner.D33_B3_FAST: _candidate_rows(
            runner.D33_B3_FAST, value=0.72, safe=False, edge_count=0
        ),
        runner.D34_A: _candidate_rows(
            runner.D34_A, value=0.95, safe=False, edge_count=15
        ),
        runner.D34_B: _candidate_rows(
            runner.D34_B, value=0.78, safe=True, edge_count=30
        ),
        runner.D34_C: _candidate_rows(
            runner.D34_C, value=0.82, safe=True, edge_count=45
        ),
    }
    selected, decisions = runner._select_d34_candidate(folds)
    assert selected == runner.D34_C
    a = next(row for row in decisions if row["candidate_id"] == runner.D34_A)
    assert a["d34_old_safety_hard_gate_pass"] is False
    assert a["eligible_positive_route"] is False


def test_d34_full_k10_gate_reuses_fold_loso_and_checks_deployment_closure() -> None:
    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": True,
            "old_loso_zero_intrusion_all_folds": True,
        }
        for candidate_id in runner.D34_CANDIDATES
    ]
    resources = {
        candidate_id: {
            scenario: {
                    "old_support_non_degradation_pass": True,
                    "old_score_prefix_bitwise_unchanged": True,
                    "unreachable_edge_count": 0,
                "peak_trainable_parameters": 0,
                "total_optimizer_steps": 0,
                "persistent_state_cap_pass": True,
                "dense_query_graph_bytes": 0,
                "latency_includes_argmax": True,
                "query_rows_used_for_fit": 0,
            }
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        for candidate_id in runner.D34_CANDIDATES
    }
    selected, reason = runner._apply_full_k10_d34_gate(
        runner.D34_C, decisions, resources
    )
    assert selected == runner.D34_C
    assert reason is None
    resources[runner.D34_C]["leo_rain_weak"][
        "old_score_prefix_bitwise_unchanged"
    ] = False
    selected, reason = runner._apply_full_k10_d34_gate(
        runner.D34_C, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_D34_OLD_SAFETY_OR_RESOURCE_GATE_FAILED"

    resources[runner.D34_C]["leo_rain_weak"][
        "old_score_prefix_bitwise_unchanged"
    ] = True
    resources[runner.D34_C]["leo_rain_weak"]["unreachable_edge_count"] = 1
    decisions[-1]["eligible_positive_route"] = True
    selected, reason = runner._apply_full_k10_d34_gate(
        runner.D34_C, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_D34_OLD_SAFETY_OR_RESOURCE_GATE_FAILED"


def test_d34_launcher_is_unique_v1_and_sha_closed() -> None:
    launcher = (CODE / "scripts" / "launch_d34_collision_local_20260718.sh").read_text(
        encoding="utf-8"
    )
    assert 'OUTPUT="$RUN/output/support_screen_v1"' in launcher
    assert "--candidate-set d34_v1" in launcher
    assert "EXPECTED_D34_CORE_SHA256=" in launcher
    assert "EXPECTED_RUNNER_SHA256=" in launcher
    assert "__D34_" not in launcher
    assert "query" not in launcher.lower()
