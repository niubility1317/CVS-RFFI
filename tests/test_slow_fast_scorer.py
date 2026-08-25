from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.slow_fast_scorer import (
    _score_transition_diagnostics,
    summarize_candidate_scores,
    summarize_shadow_candidate_scores,
)
from cvsrffi.stage2_meta_adapter_scorer import PairedStage2BScore, StateScore


def _state(name: str, correct: tuple[int, int]) -> StateScore:
    total = (100, 100)
    accuracy = {str(i): correct[i] / total[i] for i in range(2)}
    return StateScore(
        state=name,
        registration_state="REG0",
        query_ids=("q0",),
        mean_old_acc=sum(accuracy.values()) / 2,
        old_class_floor=min(accuracy.values()),
        per_class_accuracy=accuracy,
        per_class_correct={str(i): correct[i] for i in range(2)},
        per_class_total={str(i): total[i] for i in range(2)},
        micro_old_acc=sum(correct) / sum(total),
    )


def _pair(scenario: str, da0: tuple[int, int], da1: tuple[int, int]) -> PairedStage2BScore:
    return PairedStage2BScore(
        da0=_state("DA0_REG0", da0),
        da1=_state("DA1_REG0", da1),
        mean_delta_pp=0.0,
        floor_delta_pp=0.0,
        candidate_id="FAST_FILM_R8",
        bundle_id="bundle",
        row_id=scenario,
        row={"scenario": scenario},
        registered_class_ids=(0, 1),
    )


def test_candidate_summary_uses_aggregate_class_counts_and_promotion_limits() -> None:
    rows = [
        _pair("leo_clear_weak", (70, 60), (73, 62)),
        _pair("leo_low_elev_weak", (60, 50), (63, 52)),
        _pair("leo_rain_weak", (50, 40), (53, 42)),
    ]
    summary = summarize_candidate_scores(rows)

    assert summary["mean_delta_pp"] == pytest.approx(2.5)
    assert summary["floor_delta_pp"] == pytest.approx(2.0)
    assert summary["max_class_drop_pp"] == pytest.approx(2.0)
    assert summary["verdict"] == "PROMOTE_TO_TARGET25"


def test_shadow_summary_keeps_each_state_and_selects_best_only_after_truth() -> None:
    rows = []
    for scenario, da0, fixed, gate in (
        ("leo_clear_weak", (70, 60), (74, 62), (69, 61)),
        ("leo_low_elev_weak", (60, 50), (64, 52), (59, 51)),
        ("leo_rain_weak", (50, 40), (54, 42), (49, 41)),
    ):
        rows.append(
            {
                "candidate_id": "COMMON_SHIFT_R4",
                "scenario": scenario,
                "states": {
                    "DA0_REG0": _state("DA0_REG0", da0).to_dict(),
                    "DA1_L0250_REG0": _state("DA1_L0250_REG0", fixed).to_dict(),
                    "DA1_GATE_CF_REG0": _state("DA1_GATE_CF_REG0", gate).to_dict(),
                },
            }
        )

    summary = summarize_shadow_candidate_scores(rows)

    assert summary["best_truth_last_shadow_state"] == "DA1_L0250_REG0"
    assert summary["states"]["DA1_L0250_REG0"]["mean_delta_pp"] == pytest.approx(3.0)
    assert summary["states"]["DA1_GATE_CF_REG0"]["verdict"] == "SCIENTIFIC_FAILURE_NO_PROMOTION"
    assert summary["truth_last_selection_reused_for_adaptation"] is False


def test_transition_diagnostics_separate_old_new_flips_and_raw_cosine_geometry() -> None:
    baseline_scores = np.asarray(
        [[0.60, 0.40], [0.70, 0.30], [0.55, 0.45]], dtype=np.float32
    )
    adapted_scores = np.asarray(
        [[0.40, 0.60], [0.20, 0.80], [0.30, 0.70]], dtype=np.float32
    )

    diagnostics = _score_transition_diagnostics(
        registered_class_ids=(10, 20),
        baseline_predictions=np.asarray([10, 10, 10]),
        adapted_predictions=np.asarray([20, 20, 20]),
        baseline_scores=baseline_scores,
        adapted_scores=adapted_scores,
        old_positions=np.asarray([0, 1]),
        old_true_class_ids=np.asarray([10, 20]),
        new_positions=np.asarray([2]),
    )

    assert diagnostics["old_decision_change_count"] == 2
    assert diagnostics["old_positive_flip_count"] == 1
    assert diagnostics["old_negative_flip_count"] == 1
    assert diagnostics["new_decision_change_count"] == 1
    assert diagnostics["old_true_class_raw_cosine_delta_mean"] == pytest.approx(0.15)
    assert diagnostics["old_top1_top2_margin_delta_mean"] == pytest.approx(0.10)
    assert diagnostics["old_score_l2_change_mean"] == pytest.approx(
        np.mean([np.sqrt(0.08), np.sqrt(0.50)])
    )
    assert diagnostics["per_class_true_margin_delta_mean"] == pytest.approx(
        {"10": -0.40, "20": 1.00}
    )
    assert diagnostics["new_intrusion_delta_mean"] == pytest.approx(0.15)
