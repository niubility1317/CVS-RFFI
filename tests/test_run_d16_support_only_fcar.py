from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d16_support_only_fcar.py"
)
SPEC = importlib.util.spec_from_file_location("run_d16_support_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _metric(old0: float, old1: float) -> dict:
    return {
        "overall_accuracy": (old0 + old1) / 2.0,
        "min_class_accuracy": min(old0, old1),
        "per_class_accuracy": {"old0": old0, "old1": old1},
    }


def _new_metric(value: float) -> dict:
    return {
        "overall_accuracy": value,
        "min_class_accuracy": value,
        "per_class_accuracy": {"new0": value},
    }


def _fold(index: int = 0) -> dict:
    before = _metric(0.5, 1.0)
    base_before = _metric(0.4, 1.0)
    after = _metric(0.6, 1.0)
    base_after = _metric(0.5, 1.0)
    new = _new_metric(0.8)
    base_new = _new_metric(0.8)
    return {
        "fold": index,
        "before_old": before,
        "base_before_old": base_before,
        "after_old": after,
        "base_after_old": base_after,
        "after_new": new,
        "base_after_new": base_new,
        "joint": {
            "overall_accuracy": 0.8,
            "min_class_accuracy": 0.6,
            "per_class_accuracy": {
                "old0": 0.6,
                "old1": 1.0,
                "new0": 0.8,
            },
        },
        "base_joint": {
            "overall_accuracy": 0.7666666667,
            "min_class_accuracy": 0.5,
            "per_class_accuracy": {
                "old0": 0.5,
                "old1": 1.0,
                "new0": 0.8,
            },
        },
        "H_old_new": 0.8,
        "base_H_old_new": 0.76,
        "old_forgetting": -0.05,
        "candidate_vs_z0_per_class_non_degraded": {
            "before_old": {"old0": True, "old1": True},
            "after_old": {"old0": True, "old1": True},
            "after_new": {"new0": True},
        },
        "old_score_bitwise_locked": True,
        "held_disjoint_from_selection": True,
        "enabled": [True, False, False],
        "floor_handles": ["old0"],
    }


def test_candidate_grid_is_locked_to_z0_and_two_margin_bands() -> None:
    candidates = runner._candidates()
    assert [value.candidate_id for value in candidates] == [
        "d16_z0_true_zero_base",
        "d16_fcar_mb002",
        "d16_fcar_mb004",
    ]
    assert [value.margin_band for value in candidates] == [0.0, 0.02, 0.04]
    assert candidates[0].force_zero is True
    assert candidates[0].rank == 0
    assert all(value.rank == 8 for value in candidates[1:])
    assert all(value.shrink == 0.5 for value in candidates[1:])
    assert all(value.ridge == 0.01 for value in candidates)
    lock = runner._candidate_lock(candidates)
    assert lock == runner._candidate_lock(candidates)
    assert len(lock["lock_sha256"]) == 64


def test_gate_checks_every_fold_and_requires_strict_floor_gain() -> None:
    result = {"folds": [_fold(index) for index in range(5)]}
    passed = runner._scenario_gate(result, force_zero=False)
    assert passed["all_folds_gate_pass"] is True
    assert passed["strict_floor_gain_in_every_fold"] is True

    forgotten = copy.deepcopy(result)
    forgotten["folds"][3]["after_old"]["per_class_accuracy"]["old0"] = 0.4
    failed = runner._scenario_gate(forgotten, force_zero=False)
    assert failed["all_folds_gate_pass"] is False
    assert failed["folds"][3][
        "after_old_per_class_non_degraded_vs_same_fold_before"
    ] is False

    neutral_floor = copy.deepcopy(result)
    for fold in neutral_floor["folds"]:
        fold["after_old"]["per_class_accuracy"]["old0"] = 0.5
    failed = runner._scenario_gate(neutral_floor, force_zero=False)
    assert failed["all_folds_gate_pass"] is False
    assert failed["strict_floor_gain_in_every_fold"] is False


def test_all_positive_fail_selects_true_zero() -> None:
    candidates = runner._candidates()
    rows = [
        {
            "candidate_id": value.candidate_id,
            "force_zero": value.force_zero,
            "all_scenario_all_fold_gate_pass": False,
            "margin_band": value.margin_band,
            "worst_old_floor": 0.0,
            "worst_new_floor": 0.0,
            "mean_H_old_new": 0.0,
            "mean_joint_accuracy": 0.0,
        }
        for value in candidates
    ]
    selected, passed = runner._select_candidate(rows, candidates)
    assert selected == "d16_z0_true_zero_base"
    assert passed is False


def _rows(prefix: str) -> dict[str, np.ndarray]:
    labels = np.repeat(np.asarray(["old0", "old1"]), 10)
    ranks = np.tile(np.arange(10, dtype=np.int64), 2)
    return {
        "labels": labels,
        "ranks": ranks,
        "tokens": np.asarray(
            [f"{prefix}_token_{index}" for index in range(20)]
        ),
        "hashes": np.asarray(
            [f"{index:064x}" for index in range(20)]
        ),
    }


def test_cross_scenario_physical_and_parent_hash_disjointness() -> None:
    rows = {}
    for scenario_index, scenario in enumerate(
        runner.FORMAL_LEO_WEAK_SCENARIOS
    ):
        value = _rows(f"s{scenario_index}")
        value["hashes"] = np.asarray(
            [f"{scenario_index * 100 + index:064x}" for index in range(20)]
        )
        rows[scenario] = value
    audit = runner._cross_scenario_disjointness(rows)
    assert audit["all_pairwise_disjoint"] is True
    rows[runner.FORMAL_LEO_WEAK_SCENARIOS[1]]["tokens"][0] = rows[
        runner.FORMAL_LEO_WEAK_SCENARIOS[0]
    ]["tokens"][0]
    try:
        runner._cross_scenario_disjointness(rows)
    except runner.D16RunnerError as exc:
        assert "reused across LEO scenarios" in str(exc)
    else:
        raise AssertionError("cross-scenario physical reuse was accepted")


def test_runner_reuses_d14_preopen_and_exposes_no_query_scorer_cli() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from run_d14_support_only_pairwise_fisher_guard import" in source
    assert "_load_enrollment" in source
    assert "_payload_rows" in source
    assert "_build_feature_artifact" in source
    assert "load_verified_somph_predictor_bundle" not in source
    assert "--before-seal-sha256" in source
    assert "--after-seal-sha256" in source
    assert "--query" not in source
    assert "--truth" not in source
    assert "--scorer" not in source
    assert "authority_evidence" not in source
    assert "development_select only" in source
