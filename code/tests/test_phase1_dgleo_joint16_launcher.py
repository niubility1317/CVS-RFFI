import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_dgleo_joint16_20260706.sh"


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        ["bash", SCRIPT, "--dry-run", *extra],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_dgleo_joint16_declares_protocol_and_non_ce_satellite_training():
    out = _dry_run("--only=DGLEO_J10_BALANCED_A")

    assert "DGLEO_J10_BALANCED_A" in out
    assert "algorithm=DGLEO_JOINT16" in out
    assert "group=J10" in out
    assert "dg_primary=1" in out
    assert "leo_primary=1" in out
    assert "domain_loss_on=1" in out
    assert "sat_consistency_on=1" in out
    assert "base=EPOC_CONCAT_SAT" in out
    assert "concat_sat_mode=full_2b_core_domain" in out
    assert "concat_sat_full_loss=1" in out
    assert "concat_sat_ce_only=0" in out
    assert "--use_concat_sat_channel_aug" in out
    assert "--no_concat_sat_ce_only" in out
    assert "--concat_sat_start_epoch 1" in out
    assert "--sat_view_prob 1.0" in out
    assert "--sat_view_seed 707061" in out
    assert "--lambda_domain 1.35" in out
    assert "--lambda_adv 0.22" in out
    assert "--lambda_orth 0.070" in out
    assert "--lambda_cons 0.105" in out
    assert "--lambda_fishr 0.055" in out
    assert "--lambda_sat_cls 0.82" in out
    assert "--lambda_sat_cons 0.055" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_dgleo_joint16_is_source_only_and_rejects_target_inputs():
    out = _dry_run("--only=DGLEO_J10_BALANCED_A")

    assert "phase1_dataset=ManySig_only" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "manytx_in_training=0" in out
    assert "proxy_unknown_real_tx_calibration=0" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "--proxy_unknown_tx_ids" not in out

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"WISIG_PKL=/tmp/ManyTx.pkl bash {SCRIPT} --dry-run --only=DGLEO_J10_BALANCED_A",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr


def test_dgleo_joint16_assigns_two_candidates_per_gpu():
    out = _dry_run()

    assert "candidates=16" in out
    assert out.count("[DGLEO-CANDIDATE]") == 16
    gpu_ids = re.findall(r"CUDA_VISIBLE_DEVICES=([0-7])", out)
    assert sorted(gpu_ids) == sorted(str(gpu) for gpu in range(8) for _ in range(2))
    for group in ("J1", "J2", "J3", "J4", "J5", "J7", "J10", "J11"):
        assert f"group={group}" in out
