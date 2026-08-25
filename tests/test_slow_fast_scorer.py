from __future__ import annotations

import pytest

from cvsrffi.slow_fast_scorer import summarize_candidate_scores
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
