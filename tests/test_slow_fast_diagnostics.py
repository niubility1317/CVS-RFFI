from __future__ import annotations

import pytest

from cvsrffi.slow_fast_diagnostics import build_shadow_response_surface


def _state(mean_old_acc: float) -> dict[str, object]:
    return {
        "old_class_metrics": {
            "mean_old_acc": mean_old_acc,
            "old_class_floor": mean_old_acc - 0.1,
        }
    }


def test_response_surface_joins_full_state_axis_and_computes_tied_rank_spearman() -> None:
    row_scores = [
        {
            "candidate_id": "FAST_FILM_R8",
            "scenario": "leo_clear_weak",
            "states": {
                "DA0_REG0": _state(0.50),
                "DA1_A_REG0": _state(0.53),
                "DA1_B_REG0": _state(0.52),
                "DA1_C_REG0": _state(0.51),
            },
        }
    ]
    support_receipts = [
        {
            "candidate_id": "FAST_FILM_R8",
            "scenario": "leo_clear_weak",
            "states": ["DA0_REG0", "DA1_A_REG0", "DA1_B_REG0", "DA1_C_REG0"],
            "shadow_support_diagnostics": {
                "DA0_REG0": {"risk_gain": 0.0, "q90_feature_move": 0.0},
                "DA1_A_REG0": {"risk_gain": 0.01, "q90_feature_move": 0.05},
                "DA1_B_REG0": {"risk_gain": 0.02, "q90_feature_move": 0.10},
                "DA1_C_REG0": {"risk_gain": 0.03, "q90_feature_move": 0.15},
            },
        }
    ]

    summary = build_shadow_response_surface(row_scores, support_receipts)

    assert summary["state_count"] == 4
    assert summary["spearman_support_query"] == pytest.approx(-1.0)
    assert summary["spearman_move_query"] == pytest.approx(-1.0)
    assert summary["p0_stop_signal"] is True
    assert [row["state"] for row in summary["rows"]] == [
        "DA0_REG0",
        "DA1_A_REG0",
        "DA1_B_REG0",
        "DA1_C_REG0",
    ]


def test_response_surface_uses_average_ranks_for_ties() -> None:
    row_scores = [
        {
            "candidate_id": "COMMON_SHIFT_R4",
            "scenario": "leo_rain_weak",
            "states": {
                "DA0_REG0": _state(0.50),
                "DA1_A_REG0": _state(0.51),
                "DA1_B_REG0": _state(0.51),
                "DA1_C_REG0": _state(0.53),
            },
        }
    ]
    support_receipts = [
        {
            "candidate_id": "COMMON_SHIFT_R4",
            "scenario": "leo_rain_weak",
            "states": ["DA0_REG0", "DA1_A_REG0", "DA1_B_REG0", "DA1_C_REG0"],
            "shadow_support_diagnostics": {
                "DA0_REG0": {"risk_gain": 0.0, "q90_feature_move": 0.0},
                "DA1_A_REG0": {"risk_gain": 0.01, "q90_feature_move": 0.05},
                "DA1_B_REG0": {"risk_gain": 0.01, "q90_feature_move": 0.05},
                "DA1_C_REG0": {"risk_gain": 0.03, "q90_feature_move": 0.15},
            },
        }
    ]

    summary = build_shadow_response_surface(row_scores, support_receipts)

    assert summary["spearman_support_query"] == pytest.approx(1.0)
    assert summary["p0_stop_signal"] is False


def test_response_surface_stops_when_positive_rank_signal_is_below_point_two() -> None:
    states = {"DA0_REG0": _state(0.50)}
    support_states = {
        "DA0_REG0": {"risk_gain": 0.0, "q90_feature_move": 0.0}
    }
    query_rank_by_state = [2, 5, 1, 4, 3]
    for index, query_rank in enumerate(query_rank_by_state, start=1):
        state = f"DA1_{index}_REG0"
        states[state] = _state(0.50 + query_rank / 100.0)
        support_states[state] = {
            "risk_gain": index / 100.0,
            "q90_feature_move": index / 20.0,
        }

    summary = build_shadow_response_surface(
        [{"candidate_id": "FAST_FILM_R8", "scenario": "leo_clear_weak", "states": states}],
        [
            {
                "candidate_id": "FAST_FILM_R8",
                "scenario": "leo_clear_weak",
                "shadow_support_diagnostics": support_states,
            }
        ],
    )

    assert summary["spearman_support_query"] == pytest.approx(0.1)
    assert summary["p0_stop_signal"] is True
