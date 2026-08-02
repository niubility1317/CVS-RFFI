from __future__ import annotations

import inspect

from scripts import run_d112_g1_sourceheld_one_shot as runner


def test_three_arm_design_is_identifiable_and_has_no_da_only_arm() -> None:
    assert runner.ARMS == ("M0", "M_HEAD_GROUND", "M_JOINT_SEAM")
    assert "M_DA" not in runner.ARMS
    assert runner.EFFECT_PAIRS == {
        "HEAD_GROUND_VS_M0": ("M_HEAD_GROUND", "M0"),
        "SEAM_MOTION_AT_HEAD": ("M_JOINT_SEAM", "M_HEAD_GROUND"),
        "JOINT_VS_M0": ("M_JOINT_SEAM", "M0"),
    }


def test_predict_cli_has_no_truth_surface() -> None:
    parser_source = inspect.getsource(runner.parse_args)
    predict_block = parser_source.split('commands.add_parser("predict")', 1)[1].split(
        'commands.add_parser("score")', 1
    )[0]
    assert "truth" not in predict_block.lower()
    assert "target" not in predict_block.lower()


def test_score_is_the_only_truth_open_surface() -> None:
    args = runner.parse_args(
        [
            "score",
            "--prediction-root",
            "predictions",
            "--truth-json",
            "truth.json",
            "--truth-input-seal-json",
            "truth_input_seal.json",
            "--truth-open-event-json",
            "truth_open_event.json",
            "--output-json",
            "scores.json",
        ]
    )
    assert args.command == "score"


def test_core_predict_path_has_no_truth_argument() -> None:
    assert set(inspect.signature(runner.predict).parameters) == {"args"}
    source = inspect.getsource(runner.predict)
    assert "args.truth" not in source
    assert "truth_json" not in source
    assert 'audit["query_state_updates"]' not in source
