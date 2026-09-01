from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_adv3b02_ecrs_v1_20260901.sh"


def _bash_executable() -> str:
    configured = os.environ.get("HERMES_GIT_BASH_PATH", "").strip()
    return configured if configured else "bash"


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
    assert output.count("[ECRS-V1-CANDIDATE]") == 9
    assert "rung=R0" in output and "rung=R8" in output
    for token in (
        "--model_variant lite_d",
        "--branch_ablation no_dac",
        "--domain_branch_ablation no_stats",
        "--ssl_labeled_ratio 0.07",
        "--ssl_unlabeled_ratio 0.63",
        "--ssl_val_ratio 0.30",
        "--concat_sat_ce_only",
        "--concat_sat_ce_weight 0.68",
        "--lambda_sat_cons 0",
        "1@0.30:leo_clear_weak",
        "41@0.60:leo_low_elev_weak,leo_rain_weak",
        "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        "--epochs 200",
        "K=28 response_dim=64 rho_max=0.25",
        "--no_ecrs_enable_learnable_basis",
        "--no_ecrs_enable_fasttrust",
    ):
        assert token in output


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
