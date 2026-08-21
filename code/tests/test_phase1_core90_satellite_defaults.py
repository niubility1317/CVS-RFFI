import argparse
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import _resolve_sat_training_mode, build_arg_parser  # noqa: E402
from training_controls import (  # noqa: E402
    LEO_WEAK_SCENARIOS,
    LEO_WEAK_SCENARIOS_CSV,
    PHASE1_CORE90_SAT_EFFECTIVE_CE_WEIGHT,
    PHASE1_CORE90_SAT_DEFAULTS,
    PHASE1_CORE90_SAT_SUPERVISION_START_EPOCH,
    PHASE1_CORE90_SAT_VIEW_SCHEDULE,
    apply_phase1_core90_satellite_defaults,
    resolve_phase1_sat_training_scenarios,
)


def test_core90_satellite_default_policy_is_frozen():
    assert PHASE1_CORE90_SAT_DEFAULTS == {
        "use_concat_sat_channel_aug": True,
        "concat_sat_ce_only": True,
        "concat_sat_ce_weight": 1.0,
        "concat_sat_start_epoch": 1,
        "use_sat_consistency": True,
        "lambda_sat_cls": 0.68,
        "lambda_sat_cons": 0.0,
        "sat_cons_start_epoch": 80,
        "sat_train_scenario": "",
        "sat_train_scenarios": "",
        "sat_view_schedule": PHASE1_CORE90_SAT_VIEW_SCHEDULE,
        "sat_view_prob": 1.0,
        "sat_view_seed": 2027,
        "eval_sat_channel": True,
        "eval_sat_scenarios": LEO_WEAK_SCENARIOS_CSV,
    }
    assert PHASE1_CORE90_SAT_VIEW_SCHEDULE == (
        "1@0.30:leo_clear_weak;"
        "41@0.60:leo_low_elev_weak,leo_rain_weak;"
        "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    )
    assert "mixed_orbit" not in PHASE1_CORE90_SAT_VIEW_SCHEDULE
    assert PHASE1_CORE90_SAT_EFFECTIVE_CE_WEIGHT == pytest.approx(0.68)
    assert PHASE1_CORE90_SAT_SUPERVISION_START_EPOCH == 80


def test_ssdg_parser_defaults_to_core90_concat_and_leo_weak_testing():
    args = build_arg_parser().parse_args(["--output_dir", "x"])

    for key, expected in PHASE1_CORE90_SAT_DEFAULTS.items():
        if hasattr(args, key):
            assert getattr(args, key) == expected

    assert args.use_concat_sat_channel_aug is True
    assert args.concat_sat_ce_only is True
    assert args.sat_training_mode == ""
    assert _resolve_sat_training_mode(args) == "concat_ce_only"
    assert resolve_phase1_sat_training_scenarios(
        args.sat_train_scenario,
        args.sat_train_scenarios,
    ) == list(LEO_WEAK_SCENARIOS)
    assert args.lambda_sat_cls == pytest.approx(0.68)
    assert args.lambda_sat_cons == pytest.approx(0.0)
    assert args.eval_sat_channel is True
    assert args.eval_sat_scenarios == LEO_WEAK_SCENARIOS_CSV


def test_centralized_parser_receives_the_same_core90_defaults(monkeypatch):
    import train

    class ParserDefaultsCaptured(RuntimeError):
        pass

    def capture_defaults(parser, *args, **kwargs):
        for key, expected in PHASE1_CORE90_SAT_DEFAULTS.items():
            if any(action.dest == key for action in parser._actions):
                if key == "concat_sat_ce_weight":
                    expected = PHASE1_CORE90_SAT_EFFECTIVE_CE_WEIGHT
                elif key == "concat_sat_start_epoch":
                    expected = PHASE1_CORE90_SAT_SUPERVISION_START_EPOCH
                assert parser.get_default(key) == expected
        raise ParserDefaultsCaptured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_defaults)
    with pytest.raises(ParserDefaultsCaptured):
        train.main()


def test_explicit_diagnostic_overrides_remain_available_but_are_not_defaults():
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "x",
            "--sat_training_mode",
            "disabled",
            "--no_use_concat_sat_channel_aug",
            "--no_concat_sat_ce_only",
            "--no_use_sat_consistency",
            "--sat_train_scenario",
            "mixed_orbit",
            "--sat_train_scenarios",
            "mixed_orbit",
            "--sat_view_schedule",
            "",
            "--no_eval_sat_channel",
            "--eval_sat_scenarios",
            "mixed_orbit",
        ]
    )

    assert args.sat_training_mode == "disabled"
    assert args.use_concat_sat_channel_aug is False
    assert args.concat_sat_ce_only is False
    assert args.use_sat_consistency is False
    assert args.sat_train_scenarios == "mixed_orbit"
    assert args.eval_sat_channel is False
    assert args.eval_sat_scenarios == "mixed_orbit"

    boolean_only = build_arg_parser().parse_args(
        [
            "--output_dir",
            "x",
            "--no_use_concat_sat_channel_aug",
            "--no_use_sat_consistency",
        ]
    )
    assert _resolve_sat_training_mode(boolean_only) == "disabled"
    assert boolean_only.use_concat_sat_channel_aug is False
    assert boolean_only.concat_sat_ce_only is False

    single_clear = build_arg_parser().parse_args(
        ["--output_dir", "x", "--sat_train_scenario", "leo_clear_weak"]
    )
    assert resolve_phase1_sat_training_scenarios(
        single_clear.sat_train_scenario,
        single_clear.sat_train_scenarios,
    ) == ["leo_clear_weak"]


def test_common_default_applier_only_sets_arguments_present_on_the_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_sat_channel", action="store_true")
    parser.add_argument("--eval_sat_scenarios", default="legacy")

    apply_phase1_core90_satellite_defaults(parser)
    args = parser.parse_args([])

    assert args.eval_sat_channel is True
    assert args.eval_sat_scenarios == LEO_WEAK_SCENARIOS_CSV
    assert not hasattr(args, "sat_training_mode")
