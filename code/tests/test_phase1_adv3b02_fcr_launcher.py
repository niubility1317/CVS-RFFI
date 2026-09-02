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
    args = parser.parse_args(
        ["--phase1_method", "adv3b02_fcr", "--use_fcr", "--fcr_ablation_row", row]
    )
    args.use_meta_ssl_cvs = True
    args.ssl_labeled_ratio = 0.07
    args.ssl_unlabeled_ratio = 0.63
    args.ssl_val_ratio = 0.30
    return resolve_fcr_training_options(args)


def test_r0_r8_are_explicit_validated_rows_with_monotone_capabilities() -> None:
    rows = [_resolve(f"R{index}") for index in range(9)]
    assert [args.fcr_ablation_row for args in rows] == [f"R{index}" for index in range(9)]
    expected_active = (
        set(),
        {"self", "eta"},
        {"self", "eta", "swap"},
        {"self", "eta", "swap", "shared"},
        {"self", "eta", "swap", "shared", "latent_cycle"},
        {"self", "eta", "swap", "shared", "latent_cycle", "need"},
        {"self", "eta", "swap", "shared", "latent_cycle", "need"},
        {"self", "eta", "swap", "shared", "latent_cycle", "need", "phys"},
        {"self", "eta", "swap", "shared", "latent_cycle", "need", "phys", "factor"},
    )
    for args, expected in zip(rows, expected_active):
        active = {name for name, value in args.effective_fcr_lambdas.items() if value > 0.0}
        assert active == expected
    assert rows[5].fcr_basic_need_diagnostic is True
    assert rows[5].fcr_targeted_transplant is False
    assert rows[6].fcr_targeted_transplant is True
    assert rows[6].effective_fcr_lambdas["need"] == 1.0
    assert rows[6].fcr_physics_ordered_decoder is False
    assert rows[6].fcr_decoder_mode == "control"
    assert rows[7].fcr_physics_ordered_decoder is True
    assert rows[7].fcr_decoder_mode == "full_physics"
    assert rows[7].effective_fcr_lambdas["phys"] == 1.0
    assert rows[7].fcr_three_axis_intervention is False
    assert rows[8].fcr_three_axis_intervention is True
    signatures = [args.fcr_execution_signature for args in rows]
    assert len(signatures) == len(set(signatures)) == 9
    assert signatures[7] != signatures[8]
    assert "decoder=control" in signatures[6]
    assert "decoder=full_physics" in signatures[7]

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
            "--use_meta_ssl_cvs",
            "--ssl_labeled_ratio",
            "0.07",
            "--ssl_unlabeled_ratio",
            "0.63",
            "--ssl_val_ratio",
            "0.30",
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
    assert payload["meta_ssl_roles"] == {
        "labeled": 0.07,
        "unlabeled": 0.63,
        "validation": 0.30,
    }
    assert payload["identity_logit_route"] == "fcr_tx_logits"
    assert payload["feature_schema"] == "ADV3B02:FCR:z_f_id:unit_l2:160:v1"
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
        "--model_variant lite_d",
        "--epochs 200",
        "--fcr_ablation_row",
        "--use_meta_ssl_cvs",
        "--ssl_labeled_ratio 0.07",
        "--ssl_unlabeled_ratio 0.63",
        "--ssl_val_ratio 0.30",
        "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--lambda_sat_cls 0.68",
        "--sat_cons_start_epoch 80",
        "clean leo_clear_weak leo_low_elev_weak leo_rain_weak",
        "refusing to overwrite existing output root",
        "RUN_ID",
        "OUTPUT_ROOT",
        "INIT_CHECKPOINT",
        "--init_checkpoint",
        "--init_checkpoint_expected_seed 392002",
        "--init_checkpoint_expected_epoch 200",
        "--init_checkpoint_expected_candidate S392002_ADV3B03_MU10_ALPHA20_E200",
        "--init_checkpoint_require_mature_identity_complete",
        "--branch_ablation no_dac",
        "--domain_branch_ablation no_stats",
        "--dry-run",
    )
    for fragment in required_fragments:
        assert fragment in text
    lowered = text.lower()
    for forbidden in ("phase2", "query", "truth", "scorer"):
        assert forbidden not in lowered
    for row in range(1, 9):
        assert f"R{row}" in text
    assert "--row=R0" not in text
    assert "ROWS=(R1 R2 R3 R4 R5 R6 R7 R8)" in text
    assert '[[ "${SEED}" == "392002" ]]' in text


def test_launcher_requires_caller_run_id_and_output_root_without_defaults() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'RUN_ID="${RUN_ID:-}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-}"' in text
    assert '[[ -n "${RUN_ID}" ]]' in text
    assert '[[ -n "${OUTPUT_ROOT}" ]]' in text
    assert 'INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"' in text
    assert '[[ -n "${INIT_CHECKPOINT}" ]]' in text
