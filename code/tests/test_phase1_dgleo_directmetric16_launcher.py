import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_dgleo_directmetric16_20260706.sh"


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        ["bash", SCRIPT, "--dry-run", *extra],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_directmetric16_declares_protocol_and_full_concat_sat_training():
    out = _dry_run("--only=DGLEO_DM_P0C_BAL_A")

    assert "DGLEO_DM_P0C_BAL_A" in out
    assert "algorithm=DGLEO_DIRECTMETRIC16" in out
    assert "direct_metric_validation=1" in out
    assert "direct_metric_loss_on=1" in out
    assert "base=EPOC_CONCAT_SAT_DIRECT_METRIC" in out
    assert "concat_sa=1" in out
    assert "concat_sat_mode=full_2b_core_domain" in out
    assert "concat_sat_full_loss=1" in out
    assert "concat_sat_ce_only=0" in out
    assert "--use_concat_sat_channel_aug" in out
    assert "--no_concat_sat_ce_only" in out
    assert "--use_sat_consistency" in out
    assert "--lambda_sat_cls 0.84" in out
    assert "--lambda_sat_cons 0.060" in out
    assert "--lambda_direct_metric_accept 0.0080" in out
    assert "--direct_metric_source_overflow_weight 1.20" in out
    assert "--direct_metric_proxy_vaccept_weight 1.10" in out
    assert "--direct_metric_bridge_accept_weight 1.20" in out
    assert "--direct_metric_sat_pair_weight 0.50" in out
    assert "--direct_metric_zid_p95_target_deg 52" in out
    assert "--phase1_v2_hard_gates true" in out
    assert "--endpoint_accept_policy_id endpoint_accept_v1" in out
    assert "--tail_safety_state_machine true" in out
    assert "--tail_safety_p99_expansion_block_final_delta 2.0" in out
    assert "--tail_safety_p99_expansion_block_best_delta 3.5" in out
    assert "--tail_safety_cvar_expansion_block_final_delta 4.0" in out
    assert "--tail_safety_cvar_expansion_block_best_delta 6.0" in out
    assert "--os_eff_min_budget 0.15" in out
    assert "--phase1_v2_os_eff_all_phases true" in out
    assert "--phase1_v2_guard_blocks_final true" in out
    assert "--source_episode_density_gate true" in out
    assert "--source_episode_overflow_warn 0.90" in out
    assert "--source_episode_min_local_components 4" in out
    assert "--u_tri_state_required true" in out
    assert "--feasibility_gate true" in out
    assert "--feasibility_stage full" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out


def test_directmetric16_is_phase1_source_only_and_rejects_target_inputs():
    out = _dry_run("--only=DGLEO_DM_P0C_BAL_A")

    assert "phase1_dataset=ManySig_only" in out
    assert "source_only=1" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "target_unknown_training_count=0" in out
    assert "manytx_in_training=0" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "deployment_success_claim=0" in out
    assert "ManySig.pkl" in out
    assert "ManyTx.pkl" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "--k_shot" not in out

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"WISIG_PKL=/tmp/ManyTx.pkl bash {SCRIPT} --dry-run --only=DGLEO_DM_P0C_BAL_A",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr


def test_directmetric16_assigns_two_candidates_per_gpu():
    out = _dry_run()

    assert "candidates=16" in out
    assert out.count("[DM16-CANDIDATE]") == 16
    gpu_ids = re.findall(r"CUDA_VISIBLE_DEVICES=([0-7])", out)
    assert sorted(gpu_ids) == sorted(str(gpu) for gpu in range(8) for _ in range(2))
    for group in ("P0A", "P0B", "P0C", "P0D", "P0E", "P1A", "P1B", "P1C"):
        assert f"group={group}" in out
