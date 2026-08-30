from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_wiser_pilot import (
    ARMS,
    formal_promotion_decision,
    load_query_package,
)


def test_wiser_matrix_contains_baseline_and_all_abc_arms() -> None:
    assert ARMS == ("B0", "A", "B", "C", "ABC")


def test_query_package_rejects_label_or_truth_members(tmp_path: Path) -> None:
    path = tmp_path / "query.npz"
    np.savez_compressed(
        path,
        query_leo_weak_iq=np.zeros((2, 2, 256), np.float32),
        query_tokens=np.asarray(["q0", "q1"]),
        query_labels=np.asarray([0, 1]),
    )

    with pytest.raises(ValueError, match="forbidden"):
        load_query_package(path)


def test_formal_promotion_ignores_c_and_requires_all_preregistered_gates() -> None:
    rows = []
    for scenario in ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        baseline = {
            "P1_SOURCE_HEAD": {"balanced_accuracy": 0.50, "floor": 0.30},
            "P2_SOURCE_PROTOTYPE": {"balanced_accuracy": 0.50, "floor": 0.30},
            "P3_OLD_D92": {"balanced_accuracy": 0.50, "floor": 0.30},
        }
        promoted = {
            "P1_SOURCE_HEAD": {"balanced_accuracy": 0.54, "floor": 0.31},
            "P2_SOURCE_PROTOTYPE": {"balanced_accuracy": 0.52, "floor": 0.31},
            "P3_OLD_D92": {"balanced_accuracy": 0.54, "floor": 0.31},
        }
        rows.extend(
            [
                {"arm": "B0", "scenario": scenario, "probes": baseline, "geometry": {"within_trace": 2.0, "between_within_ratio": 1.0}},
                {"arm": "B", "scenario": scenario, "probes": promoted, "geometry": {"within_trace": 1.8, "between_within_ratio": 1.1}},
                {"arm": "C", "scenario": scenario, "probes": promoted, "geometry": {"within_trace": 1.0, "between_within_ratio": 2.0}},
            ]
        )

    decision = formal_promotion_decision(rows, arm="B")

    assert decision["passed"] is True
    assert decision["formal_arm"] == "B"
    assert decision["scenario_count"] == 3
    assert decision["c_diagnostic_rows_used"] == 0
