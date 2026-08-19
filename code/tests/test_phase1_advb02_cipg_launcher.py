from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_adv3_mechanism32_queue_20260701.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")


def test_cipg_screen_locks_historical_mixed_orbit_and_only_changes_pair_loss():
    result = subprocess.run(
        [str(GIT_BASH), str(SCRIPT), "--cipg-screen", "--dry-run"],
        cwd=str(PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
    output = result.stdout.replace(r"\;", ";")
    assert "ADV3B02_MIXED_ORBIT_E200" in output
    assert "ADV3B02_CIPG_MIXED_E200" in output
    assert "--labeled_ratio 0.07" in output
    assert "--unlabeled_ratio 0.63" in output
    assert "--source_val_ratio 0.30" in output
    assert "--best_metric source_val_sat_hmean" in output
    assert "--enable_joint_safe_guard false" in output
    assert "--sat_train_scenario mixed_orbit" in output
    assert "--sat_train_scenarios mixed_orbit" in output
    assert "--sat_view_schedule 1@0.30:mixed_orbit;41@0.60:mixed_orbit;91@0.80:mixed_orbit" in output
    assert "--eval_sat_scenarios mixed_orbit" in output
    assert "--lambda_zid_channel_invariance 0" in output
    assert "--lambda_zid_channel_invariance 0.18" in output
    assert "--zid_channel_pair_weight 1.0" in output
    assert "leo_clear_weak" not in output
    assert "leo_low_elev_weak" not in output
    assert "leo_rain_weak" not in output
