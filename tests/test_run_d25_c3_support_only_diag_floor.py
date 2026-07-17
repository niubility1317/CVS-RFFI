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
from cvsrffi.stage2_multimodal_diag_floor_adapter import (  # noqa: E402
    D25C3Config,
    D25C3LossWeights,
)


def test_c3_candidate_lock_is_exactly_six_routes_and_formal_epoch_bounded() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_C3_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.C3_A,
        runner.C3_B,
        runner.C3_C,
    )
    assert [
        (
            candidates[name].stage2b_steps,
            candidates[name].stage2c_steps,
        )
        for name in runner.C3_CANDIDATES
    ] == [(20, 0), (20, 10), (15, 15)]
    assert all(
        candidates[name].stage2b_steps + candidates[name].stage2c_steps <= 30
        for name in runner.C3_CANDIDATES
    )
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_C3_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v2"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_C3_V1
    assert lock["selection_baseline"] == runner.D25_C0
    assert "d25_c3_core_sha256" in lock["source_closure"]
    historical_lock = runner._candidate_lock(runner.preregistered_candidates())
    assert historical_lock["selection_baseline"] == runner.IDENTITY_CANDIDATE
    assert "candidate_set" not in historical_lock


def test_c3_matrix_and_cli_surface_are_locked_without_query_inputs() -> None:
    assert len(runner.preregistered_candidates(runner.CANDIDATE_SET_C3_V1)) == 6
    assert 6 * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(runner.HELD_RANKS) == 90
    source = inspect.getsource(runner.run)
    assert "expected_rows != 90" in source
    parser = runner.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "candidate_set" in destinations
    forbidden = {
        "query",
        "query_root",
        "truth",
        "role",
        "quota",
        "global_assignment",
        "source_root",
        "clean_root",
        "scorer",
    }
    assert not destinations.intersection(forbidden)


def _synthetic_blocks() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(713101)
    classes = tuple(f"old_{index}" for index in range(6)) + tuple(
        f"new_{index}" for index in range(5)
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
            z_rows.append(z_center + 0.03 * rng.normal(size=160))
            fft_rows.append(fft_center + 0.03 * rng.normal(size=96))
            rf_rows.append(rf_center + 0.03 * rng.normal(size=32))
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


def test_real_c3_fold_logs_complete_steps_and_old_support_guard() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_blocks()
    old_classes = tuple(f"old_{index}" for index in range(6))
    new_classes = tuple(f"new_{index}" for index in range(5))
    config = D25C3Config(
        loss_weights=D25C3LossWeights(
            equal_class_ce=1.0,
            tail_cvar=0.20,
            hard_negative_margin=0.10,
            proximity=0.01,
        ),
        stage2b_steps=2,
        stage2c_steps=1,
    )
    result = runner._evaluate_c3_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=old_classes,
        new_classes=new_classes,
        held_ranks=(0, 1),
        candidate_id=runner.C3_C,
        config=config,
    )
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["old_prefix_sha256_before"] == result["old_prefix_sha256_after"]
    assert result["shared_sha256_before"] == result["shared_sha256_after"]
    assert len(result["training_trace"]) == 3
    assert result["resource"]["total_optimizer_steps"] == 3
    assert result["resource"]["formal_adaptation_epoch_limit_pass"] is True
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert isinstance(result["old_support_non_degradation_pass"], bool)
    assert set(result["fit_old_before_registration"]["per_class_accuracy"]) == set(
        old_classes
    )


def test_full_k10_old_support_failure_revokes_fold_selected_route() -> None:
    decisions = [
        {
            "candidate_id": runner.C3_C,
            "eligible_positive_route": True,
        }
    ]
    resources = {
        runner.C3_C: {
            scenario: {"old_support_non_degradation_pass": scenario != "leo_rain_weak"}
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
    }
    selected, reason = runner._apply_full_k10_c3_old_support_gate(
        runner.C3_C, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"
    assert decisions[0]["eligible_positive_route"] is False
    assert decisions[0]["full_k10_old_support_non_degradation_pass"] is False
