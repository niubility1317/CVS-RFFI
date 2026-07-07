import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_dgleo_osfix16_20260707.sh"


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        ["bash", SCRIPT, "--dry-run", *extra],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_osfix16_declares_direct_open_set_and_unlabeled_quarantine_protocol():
    out = _dry_run("--only=DGLEO_OSFIX_JOINT_A")

    assert "DGLEO_OSFIX_JOINT_A" in out
    assert "algorithm=DGLEO_OSFIX16" in out
    assert "base=EPOC_CONCAT_SAT_DIRECT_METRIC_UOPT" in out
    assert "phase1_dataset=ManySig_only" in out
    assert "source_only=1" in out
    assert "concat_sa=1" in out
    assert "concat_sat_mode=full_2b_core_domain" in out
    assert "concat_sat_ce_only=0" in out
    assert "direct_open_set_metric_loss=1" in out
    assert "unlabeled_quarantine_accept=1" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "--use_concat_sat_channel_aug" in out
    assert "--no_concat_sat_ce_only" in out
    assert "--lambda_u_domain" in out
    assert "--lambda_u_adv" in out
    assert "--lambda_u_sat_cons" in out
    assert "--lambda_u_direct_metric_accept" in out
    assert "--lambda_u_quarantine_accept" in out
    assert "--direct_metric_proxy_vaccept_weight" in out
    assert "--direct_metric_source_overflow_weight" in out
    assert "--direct_metric_bridge_accept_weight" in out
    assert "--direct_metric_low_density_accept_weight" in out
    assert "--direct_metric_tail_accept_weight" in out
    assert "--direct_metric_overflow_accept_weight" in out
    assert "--direct_metric_radius_inter_ratio_weight" in out
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
    assert "--u_direct_idle_blocks_promotion true" in out
    assert "--feasibility_gate true" in out
    assert "--feasibility_stage audit" in out
    assert "phase1_source_zid_prototypes.pt" in out
    assert "phase2_zid_prototypes.pt" not in out
    assert "ManyTx.pkl" not in out


def test_osfix16_assigns_two_candidates_per_gpu():
    out = _dry_run()

    assert "candidates=16" in out
    assert out.count("[OSFIX16-CANDIDATE]") == 16
    gpu_ids = re.findall(r"CUDA_VISIBLE_DEVICES=([0-7])", out)
    assert sorted(gpu_ids) == sorted(str(gpu) for gpu in range(8) for _ in range(2))
    for group in (
        "P0_CORE",
        "P0_DENSITY",
        "P0_PROXY",
        "P0_BRIDGE",
        "P0_TAIL",
        "P0_SATOPEN",
        "P1_UQ",
        "P1_JOINT",
    ):
        assert f"group={group}" in out


def test_osfix16_rejects_non_source_phase1_inputs():
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"WISIG_PKL=/tmp/ManyTx.pkl bash {SCRIPT} --dry-run --only=DGLEO_OSFIX_JOINT_A",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
