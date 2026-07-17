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
    rng = np.random.default_rng(330718)
    classes = ("old_0", "old_1", "old_2", "new_0", "new_1")
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
            z_rows.append(z_center + 0.08 * rng.normal(size=160))
            fft_rows.append(fft_center + 0.08 * rng.normal(size=96))
            rf_rows.append(rf_center + 0.08 * rng.normal(size=32))
    rows = {
        "labels": np.asarray(labels),
        "ranks": np.asarray(ranks, dtype=np.int64),
    }
    return (
        rows,
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def test_d33_candidate_lock_has_four_routes_and_105_rows() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D33_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D33_A,
        runner.D33_B,
        runner.D33_C,
        runner.D33_B3_FAST,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D33_V1) == (
        runner.D33_CANDIDATES
    )
    assert 7 * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 105
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D33_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v11"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D33_V1
    assert "d33_spherical_registration_core_sha256" in lock["source_closure"]
    assert "d33_b3_fisher_closed_form_core_sha256" in lock["source_closure"]
    assert "d20_dali_core_sha256" not in lock["source_closure"]
    route_rows = [
        row for row in lock["candidates"] if row["candidate_id"] in runner.D33_CANDIDATES
    ]
    assert len(route_rows) == 4
    assert all(row["eligible_positive_route"] for row in route_rows)
    assert all(row["config"]["int8_predictor_dependency"] is False for row in route_rows)


def test_d33_cli_keeps_query_source_and_clean_out_of_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D33_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    destinations = {item.dest.lower() for item in parser._actions}
    parameters = {name.lower() for name in inspect.signature(runner.run).parameters}
    for names in (destinations, parameters):
        assert not any(token in name for name in names for token in forbidden)


def test_d33_fold_covers_adam_and_fisher_and_alias_semantics() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D33_V1)
    for candidate_id in (runner.D33_B, runner.D33_B3_FAST):
        result = runner._evaluate_d33_fold(
            rows,
            z_rows,
            fft_rows,
            rf_rows,
            old_classes=("old_0", "old_1", "old_2"),
            new_classes=("new_0", "new_1"),
            held_ranks=(0, 1),
            candidate_id=candidate_id,
            config=candidates[candidate_id],
        )
        resource = result["resource"]
        assert result["old_score_columns_bitwise_unchanged"] is result[
            "raw_old_score_columns_bitwise_unchanged_after_registration"
        ]
        assert result["final_old_score_columns_bitwise_unchanged_after_registration"] is result[
            "raw_old_score_columns_bitwise_unchanged_after_registration"
        ]
        assert result["old_score_columns_bitwise_unchanged_semantics"].startswith("raw_")
        assert resource["total_optimizer_steps"] <= 15
        assert resource["peak_trainable_parameters"] <= 80_000
        assert resource["dense_query_graph_bytes"] == 0
        assert resource["query_rows_used_for_fit"] == 0
        assert resource["actual_int8_component_used_for_prediction"] is False
        assert resource["total_post_backbone_macs_per_query"] > 0
        # Regression for the D32-v1 aggregation KeyError.
        rows15 = []
        for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for _ in runner.HELD_RANKS:
                clone = dict(result)
                clone["scenario"] = scenario
                rows15.append(clone)
        aggregate = runner.legacy._aggregate_candidate(rows15)
        assert aggregate["all_old_columns_bitwise_unchanged"] is result[
            "old_score_columns_bitwise_unchanged"
        ]


def test_full_d33_resource_has_argmax_latency_and_no_int8_predictor() -> None:
    rows, z_rows, fft_rows, rf_rows = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D33_V1)[
        runner.D33_B3_FAST
    ]
    resource, geometry = runner._full_d33_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        config=config,
    )
    assert resource["total_optimizer_steps"] == 0
    assert resource["latency_includes_argmax"] is True
    assert resource["batch1_head_latency_mean_ms"] > 0.0
    assert resource["persistent_state_cap_pass"] is True
    assert resource["actual_int8_component_used_for_prediction"] is False
    assert resource["total_post_backbone_macs_per_query"] < (
        resource["identity_single_qknn_macs_same_registered_count"]
    )
    assert geometry["final_old_score_transform_policy"] == "none_after_spherical_score"


def _metric(value: float) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {"old_0": value, "old_1": value},
    }


def _candidate_rows(candidate_id: str, value: float, steps: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for _ in runner.HELD_RANKS:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "before_old": _metric(value),
                    "after_old": _metric(value),
                    "after_new": {
                        "overall_accuracy": value,
                        "class_floor_accuracy": value,
                        "per_class_accuracy": {"new_0": value, "new_1": value},
                    },
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": False,
                    "old_support_non_degradation_pass": True,
                    "resource": {"total_optimizer_steps": steps},
                }
            )
    return rows


def test_d33_selector_and_full_resource_gate_use_all_four_routes() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(runner.IDENTITY_CANDIDATE, 0.4, 0),
        runner.DIAG_CANDIDATE: _candidate_rows(runner.DIAG_CANDIDATE, 0.7, 60),
        runner.D25_C0: _candidate_rows(runner.D25_C0, 0.5, 0),
        **{
            candidate_id: _candidate_rows(candidate_id, 0.65, 15)
            for candidate_id in runner.D33_CANDIDATES
        },
    }
    selected, decisions = runner._select_d26_candidate(folds, runner.D33_CANDIDATES)
    assert selected in runner.D33_CANDIDATES
    decision = next(row for row in decisions if row["candidate_id"] == selected)
    assert decision["family"] == "d33_spherical_registration"
    resources = {
        candidate_id: {
            scenario: {
                "old_support_non_degradation_pass": True,
                "peak_trainable_parameters": 2_016,
                "total_optimizer_steps": 15,
                "persistent_state_cap_pass": True,
                "dense_query_graph_bytes": 0,
                "total_post_backbone_macs_per_query": 1_500,
                "latency_includes_argmax": scenario != "leo_rain_weak",
            }
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        for candidate_id in runner.D33_CANDIDATES
    }
    gated, reason = runner._apply_full_k10_d26_old_support_gate(
        selected, decisions, resources, runner.D33_CANDIDATES
    )
    assert gated == runner.D25_C0
    assert reason == "FULL_K10_OLD_SUPPORT_OR_RESOURCE_PROTOCOL_GATE_FAILED"

