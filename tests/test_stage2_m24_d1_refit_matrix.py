from __future__ import annotations

import pytest

from scripts import run_m24_d1_refit_matrix as runner
from scripts.extend_m24_full125_inputs import patched_constants
from scripts.score_m24_d1_refit_matrix import standardized_forgetting
from cvsrffi.stage2_m24_safe_residual import D0, D1, D1_REFIT


def test_refit_matrix_is_three_arms_over_complete_125_inputs() -> None:
    assert runner.EVIDENCE_ARMS == (D0, D1, D1_REFIT)
    assert runner.DEFAULT_RECEIVERS == ("20-1", "3-19", "7-14", "7-7", "8-8")
    assert runner.DEFAULT_SEEDS == (7282101, 7282102, 7282103, 7282104, 7282105)
    assert runner.DEFAULT_CONDITIONS == (
        (1, 20),
        (2, 20),
        (5, 20),
        (10, 20),
        (10, 5),
    )
    assert runner.EXPECTED_INPUT_IDENTITIES == 125
    assert runner.EXPECTED_METHOD_ROWS == 375
    assert (
        len(runner.EVIDENCE_ARMS)
        * len(runner.DEFAULT_RECEIVERS)
        * len(runner.DEFAULT_SEEDS)
        * len(runner.DEFAULT_CONDITIONS)
        == 375
    )


def test_standardized_forgetting_uses_r0_candidate_pre_for_every_arm() -> None:
    r0 = {
        "scenario_rows": [
            {
                "scenario": "leo_clear_weak",
                "states": {
                    "DA1_REG0": {"old_accuracy": 0.80},
                    "DA1_REG1": {"old_accuracy": 0.62},
                },
            }
        ]
    }
    candidate = {
        "scenario_rows": [
            {
                "scenario": "leo_clear_weak",
                "states": {
                    "DA1_REG0": {"old_accuracy": 0.75},
                    "DA1_REG1": {"old_accuracy": 0.61},
                },
            }
        ]
    }
    result = standardized_forgetting(r0, candidate)
    assert result[0]["scenario"] == "leo_clear_weak"
    assert result[0]["A_o_pre_within"] == pytest.approx(0.75)
    assert result[0]["A_o_pre_reference_r0"] == pytest.approx(0.80)
    assert result[0]["A_o_post"] == pytest.approx(0.61)
    assert result[0]["F_within"] == pytest.approx(0.14)
    assert result[0]["F_std"] == pytest.approx(0.19)


def test_full125_input_extension_adds_exactly_two_seeds() -> None:
    package = patched_constants("package")
    feature = patched_constants("feature")
    assert package == {
        "METHOD_SEEDS": (7282104, 7282105),
        "EXPECTED_TASKS": 30,
    }
    assert feature == {
        "METHOD_SEEDS": (7282104, 7282105),
        "EXPECTED_TASKS": 50,
        "EXPECTED_SCOPE_CACHES": 150,
    }
