from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_adv3b02_ecrs_v1_20260901.sh"
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from train import effective_concat_sat_ce_weight, validate_ecrs_v1_hyperparameters  # noqa: E402


def _bash_executable() -> str:
    configured = os.environ.get("HERMES_GIT_BASH_PATH", "").strip()
    if configured:
        return configured
    if os.name == "nt":
        return r"C:\Program Files\Git\bin\bash.exe"
    return "bash"


def test_launcher_dry_run_freezes_report_v1_contract() -> None:
    result = subprocess.run(
        [_bash_executable(), str(SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    output = result.stdout
    assert output.count("[ECRS-V1-CANDIDATE]") == 8
    assert output.count("[ECRS-V1-BASELINE]") == 1
    assert "rung=R0" in output and "rung=R8" in output
    assert "rung=R0 mode=train_shared_baseline" in output
    baseline_cmd = output.split("[ECRS-V1-BASELINE]", 1)[1].split("[ECRS-V1-CANDIDATE]", 1)[0]
    assert "--init_checkpoint" not in baseline_cmd
    assert output.count("--init_checkpoint") == 8
    assert "ADV3B02_ECRS_R0/best.pth" in output
    for token in (
        "--model_variant lite_d",
        "--branch_ablation no_dac",
        "--domain_branch_ablation no_stats",
        "--ssl_labeled_ratio 0.07",
        "--ssl_unlabeled_ratio 0.63",
        "--ssl_val_ratio 0.30",
        "--seed 392005",
        "--wisig_equalized 1",
        "--wisig_target_receiver_only_eval",
        "source_days=1,2,3",
        "target_days=0,1,2,3",
        "source_rxs=1,3,4,6,8",
        "target_rxs=0,2,5,7,9,10,11",
        "source_pool=90000 L_s=6300 U_s=56700 V=27000",
        "target_per_scenario=168000",
        "--concat_sat_ce_only",
        "--concat_sat_ce_weight 0.68",
        "--concat_sat_start_epoch 1",
        "--concat_sat_ce_start_epoch 80",
        "--lambda_sat_cons 0",
        "1@0.30:leo_clear_weak",
        "41@0.60:leo_low_elev_weak,leo_rain_weak",
        "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--epochs 200",
        "--test_eval_policy interval_final",
        "--test_eval_start_epoch 200",
        "--test_eval_interval 200",
        "--test_eval_final_window 0",
        "--test_eval_final_interval 0",
        "K=28 anchors=8 response_dim=64 rho_max=0.25",
        "--no_ecrs_enable_learnable_basis",
        "--no_ecrs_enable_fasttrust",
        "--lambda_ecrs_canonical 0.10",
        "--lambda_ecrs_split_fit 0.10",
        "--lambda_ecrs_pair_cross 0.10",
        "--lambda_ecrs_pair_surface 0.03",
        "--lambda_ecrs_same_tx 0.05",
        "--lambda_ecrs_diff_tx 0.03",
        "--ecrs_alpha_resp 0.15",
        "--ecrs_ridge_alpha 0.01",
    ):
        assert token in output


def test_core90_sat_ce_starts_e80_without_suppressing_ecrs_pair_view() -> None:
    args = SimpleNamespace(concat_sat_ce_weight=0.68, concat_sat_ce_start_epoch=80)
    assert effective_concat_sat_ce_weight(args, 79) == 0.0
    assert effective_concat_sat_ce_weight(args, 80) == 0.68


def test_ecrs_v1_rejects_out_of_report_weight_ranges() -> None:
    valid = SimpleNamespace(
        lambda_ecrs_canonical=0.10,
        lambda_ecrs_split_fit=0.10,
        lambda_ecrs_pair_cross=0.10,
        lambda_ecrs_pair_surface=0.03,
        lambda_ecrs_same_tx=0.05,
        lambda_ecrs_diff_tx=0.03,
        ecrs_alpha_resp=0.15,
        ecrs_ridge_alpha=0.01,
    )
    validate_ecrs_v1_hyperparameters(valid)
    invalid = SimpleNamespace(**vars(valid))
    invalid.lambda_ecrs_split_fit = 1.0
    try:
        validate_ecrs_v1_hyperparameters(invalid)
    except ValueError as error:
        assert "lambda_ecrs_split_fit" in str(error)
    else:
        raise AssertionError("out-of-report ECRS weight must be rejected")


def test_launcher_rejects_phase2_or_target_data() -> None:
    env = dict(os.environ)
    env["WISIG_PKL"] = "/tmp/ManyTx.pkl"
    result = subprocess.run(
        [_bash_executable(), str(SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
