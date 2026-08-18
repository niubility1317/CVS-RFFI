from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "launch_adv3b02_riei_drift_same_protocol_20260818.sh"
)


def test_direct_matrix_launcher_uses_leo_weak_for_train_and_eval_and_declares_48_jobs():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in text
    assert "--use_sat_channel_view_aug" in text
    assert "--eval_sat_scenarios \"${LEO_SCENARIOS}\"" in text or "--eval_sat_scenarios \"${LEO_SCENARIOS}" in text
    assert 'SEEDS_CSV="${SEEDS_CSV:-713101,713102}"' in text
    assert "rx5_d0|0|2,3|0,1,2,3,4|7,8,9,10,11" in text
    assert "rx5_d012|0,1,2|3|0,1,2,3,4|7,8,9,10,11" in text
    assert "TOTAL_JOBS=48" in text


def test_direct_matrix_launcher_has_dynamic_worker_queue():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "claim_next_job" in text
    assert "run_worker" in text
    assert "WAVE_START" not in text
