from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from train import add_fcr_training_args, resolve_fcr_training_options  # noqa: E402


LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_adv3b02_fcr_20260901.sh"


def _resolve(row: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_mode", default="centralized")
    parser.add_argument("--use_concat_sat_channel_aug", action="store_true")
    add_fcr_training_args(parser)
    return resolve_fcr_training_options(
        parser.parse_args(
            ["--phase1_method", "adv3b02_fcr", "--use_fcr", "--fcr_ablation_row", row]
        )
    )


def test_r0_r8_are_explicit_validated_rows_with_monotone_capabilities() -> None:
    rows = [_resolve(f"R{index}") for index in range(9)]
    assert [args.fcr_ablation_row for args in rows] == [f"R{index}" for index in range(9)]
    assert rows[0].effective_fcr_lambdas == {name: 0.0 for name in rows[0].effective_fcr_lambdas}
    assert rows[1].effective_fcr_lambdas["self"] == 1.0
    assert rows[2].effective_fcr_lambdas["swap"] == 1.0
    assert rows[3].effective_fcr_lambdas["shared"] == 1.0
    assert rows[4].effective_fcr_lambdas["latent_cycle"] == 1.0
    assert rows[5].fcr_basic_need_diagnostic is True
    assert rows[6].fcr_targeted_transplant is True
    assert rows[6].effective_fcr_lambdas["need"] == 1.0
    assert rows[7].fcr_physics_ordered_decoder is True
    assert rows[7].effective_fcr_lambdas["phys"] == 1.0
    assert rows[8].fcr_three_axis_intervention is True

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_mode", default="centralized")
    parser.add_argument("--use_concat_sat_channel_aug", action="store_true")
    add_fcr_training_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--phase1_method", "adv3b02_fcr", "--use_fcr", "--fcr_ablation_row", "R9"]
        )


def test_train_config_dry_run_validates_row_without_opening_data() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "train.py"),
            "--phase1_method",
            "adv3b02_fcr",
            "--use_fcr",
            "--fcr_ablation_row",
            "R6",
            "--epochs",
            "200",
            "--fcr_config_dry_run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    marker = "[FCR-DRY-RUN] "
    line = next(line for line in result.stdout.splitlines() if line.startswith(marker))
    payload = json.loads(line[len(marker) :])
    assert payload["fcr_ablation_row"] == "R6"
    assert payload["epochs"] == 200
    assert payload["targeted_transplant"] is True
    assert payload["final_evaluation"] == [
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ]


def test_launcher_freezes_defaults_four_evaluations_and_no_query_paths() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required_fragments = (
        "--phase1_method adv3b02_fcr",
        "--use_fcr",
        "--epochs 200",
        "--fcr_ablation_row",
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--lambda_sat_cls 0.68",
        "--sat_cons_start_epoch 80",
        "clean leo_clear_weak leo_low_elev_weak leo_rain_weak",
        "refusing to overwrite existing output root",
        "RUN_ID",
        "OUTPUT_ROOT",
        "--dry-run",
    )
    for fragment in required_fragments:
        assert fragment in text
    lowered = text.lower()
    for forbidden in ("phase2", "query", "truth", "scorer"):
        assert forbidden not in lowered
    for row in range(9):
        assert f"R{row}" in text


def test_launcher_requires_caller_run_id_and_output_root_without_defaults() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'RUN_ID="${RUN_ID:-}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-}"' in text
    assert '[[ -n "${RUN_ID}" ]]' in text
    assert '[[ -n "${OUTPUT_ROOT}" ]]' in text
