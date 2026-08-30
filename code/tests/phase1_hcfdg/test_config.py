import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from cvsrffi.phase1_hcfdg.config import (
    SELECTION_SEEDS,
    StageBudget,
    candidate_config,
    deep_screen_rows,
    quick_screen_rows,
    residual_rows,
)


def test_quick_screen_uses_frozen_three_seeds_and_36_rows():
    rows = quick_screen_rows((1, 8))
    assert {row.seed for row in rows} == {392001, 392002, 392003}
    assert {row.candidate_id for row in rows} == {f"A{i}" for i in range(6)}
    assert len(rows) == 36
    assert all(row.optimizer_updates == 4000 for row in rows)


def test_candidate_activation_is_cumulative_and_report_ordered():
    assert candidate_config("A4").use_lodo is True
    assert candidate_config("A4").use_csd is False
    assert candidate_config("A9").use_content_conditioning is True
    assert candidate_config("A9").use_hdro is True
    assert candidate_config("A12").residual_mode == "phasedelta_dsq"


def test_candidate_definitions_keep_every_transition_explicit():
    expected = {
        "A0": (True, False, False, False, False, "off", False, False, "off", 4000),
        "A1": (False, False, False, False, False, "off", False, False, "off", 4000),
        "A2": (False, True, False, False, False, "off", False, False, "off", 4000),
        "A3": (False, True, True, False, False, "off", False, False, "off", 4000),
        "A4": (False, True, True, True, False, "off", False, False, "off", 4000),
        "A5": (False, True, True, True, True, "off", False, False, "off", 4000),
        "A6": (False, True, True, True, True, "receiver_swap", False, False, "off", 6300),
        "A7": (
            False,
            True,
            True,
            True,
            True,
            "receiver_day_channel_joint_curriculum",
            False,
            False,
            "off",
            6300,
        ),
        "A8": (
            False,
            True,
            True,
            True,
            True,
            "receiver_day_channel_joint_curriculum",
            True,
            False,
            "off",
            6300,
        ),
        "A9": (
            False,
            True,
            True,
            True,
            True,
            "receiver_day_channel_joint_curriculum",
            True,
            True,
            "off",
            6300,
        ),
        "A10": (
            False,
            True,
            True,
            True,
            True,
            "receiver_day_channel_joint_curriculum",
            True,
            True,
            "phasedelta",
            6300,
        ),
        "A11": (
            False,
            True,
            True,
            True,
            True,
            "receiver_day_channel_joint_curriculum",
            True,
            True,
            "dsq",
            6300,
        ),
        "A12": (
            False,
            True,
            True,
            True,
            True,
            "receiver_day_channel_joint_curriculum",
            True,
            True,
            "phasedelta_dsq",
            6300,
        ),
    }
    for candidate_id, values in expected.items():
        config = candidate_config(candidate_id, v2_passed=candidate_id in {"A10", "A11", "A12"})
        assert (
            config.use_dual_control,
            config.use_environment_encoder,
            config.use_rectangular_batch,
            config.use_lodo,
            config.use_csd,
            config.counterfactual_mode,
            config.use_hdro,
            config.use_content_conditioning,
            config.residual_mode,
            config.optimizer_updates,
        ) == values


def test_stage_budget_freezes_v2_schedule_and_is_json_serializable():
    budget = StageBudget()
    assert (budget.stage0, budget.stage1, budget.stage2, budget.stage3, budget.stage4) == (
        700,
        1200,
        2100,
        1700,
        600,
    )
    assert budget.total_updates == 6300
    assert budget.freeze_progress == 0.50
    assert budget.environment_updates_per_four_main_updates == 1
    json.dumps(asdict(budget), sort_keys=True)


def test_config_and_matrix_rows_are_immutable_and_json_serializable():
    config = candidate_config("A5")
    row = quick_screen_rows((1, 8))[0]

    with pytest.raises(FrozenInstanceError):
        config.candidate_id = "A0"
    with pytest.raises(FrozenInstanceError):
        row.gpu = 0

    json.dumps(asdict(config), sort_keys=True)
    json.dumps(asdict(row), sort_keys=True)


def test_deep_screen_preserves_candidate_fold_seed_report_order():
    rows = deep_screen_rows((8, 1))
    assert [(row.candidate_id, row.heldout_receiver, row.seed) for row in rows[:6]] == [
        ("A6", 8, 392001),
        ("A6", 8, 392002),
        ("A6", 8, 392003),
        ("A6", 1, 392001),
        ("A6", 1, 392002),
        ("A6", 1, 392003),
    ]
    assert [row.candidate_id for row in rows[-6:]] == ["A9"] * 6
    assert len(rows) == 24
    assert {row.optimizer_updates for row in rows} == {6300}


def test_residual_rows_require_a_passed_v2_parent_and_keep_parent_binding():
    with pytest.raises(ValueError, match="v2_passed"):
        residual_rows((1, 8))
    with pytest.raises(ValueError, match="A8 or A9"):
        residual_rows((1, 8), v2_passed=True, v2_parent_candidate_id="A5")

    rows = residual_rows((1, 8), v2_passed=True, v2_parent_candidate_id="A8")
    assert len(rows) == 18
    assert {row.candidate_id for row in rows} == {"A10", "A11", "A12"}
    assert {row.optimizer_updates for row in rows} == {6300}
    assert all(row.v2_parent_candidate_id == "A8" for row in rows)


def test_candidate_config_rejects_an_explicit_failed_v2_residual_definition():
    with pytest.raises(ValueError, match="v2_passed"):
        candidate_config("A10", v2_passed=False)


def test_unknown_candidates_and_invalid_fold_shapes_fail_closed():
    with pytest.raises(ValueError, match="unknown candidate"):
        candidate_config("A13")
    with pytest.raises(ValueError, match="exactly two folds"):
        quick_screen_rows((1,))
    with pytest.raises(ValueError, match="duplicate"):
        deep_screen_rows((1, 1))


def test_seed_registry_is_exact_and_cannot_be_mutated():
    assert SELECTION_SEEDS == (392001, 392002, 392003)
    with pytest.raises(TypeError):
        SELECTION_SEEDS[0] = 0
