import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_dgleo_v2full32_20260707.sh"


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        ["bash", SCRIPT, "--dry-run", *extra],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_v2full32_declares_full_mechanism_source_only_protocol():
    out = _dry_run("--only=DGLEO_V2FULL32_FULL_STABLE")

    assert "DGLEO_V2FULL32_FULL_STABLE" in out
    assert "algorithm=DGLEO_V2FULL32" in out
    assert "base=EPOC_CONCAT_SAT_OSFIX_V2" in out
    assert "phase1_dataset=ManySig_only" in out
    assert "source_only=1" in out
    assert "concat_sa=1" in out
    assert "concat_sat_mode=full_2b_core_domain" in out
    assert "concat_sat_ce_only=0" in out
    assert "direct_open_set_metric_loss=1" in out
    assert "unlabeled_domain_supervision=1" in out
    assert "unlabeled_satellite_consistency=1" in out
    assert "unlabeled_direct_metric_accept=1" in out
    assert "unlabeled_quarantine_accept=1" in out
    assert "trusted_core_ambiguous_tail_outside_reject=1" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "stage2_unknown_query_eval_only=1" in out
    assert "stage2_success_claim=0" in out
    assert "--use_concat_sat_channel_aug" in out
    assert "--no_concat_sat_ce_only" in out
    assert "--lambda_domain" in out
    assert "--lambda_adv" in out
    assert "--lambda_u_domain" in out
    assert "--lambda_u_adv" in out
    assert "--lambda_u_sat_cons" in out
    assert "--lambda_u_direct_metric_accept" in out
    assert "--lambda_u_quarantine_accept" in out
    assert "--lambda_direct_metric_accept" in out
    assert "--direct_metric_virtual_detach false" in out
    assert "--direct_metric_proxy_vaccept_weight" in out
    assert "--direct_metric_source_overflow_weight" in out
    assert "--direct_metric_bridge_accept_weight" in out
    assert "--direct_metric_low_density_accept_weight" in out
    assert "--direct_metric_tail_accept_weight" in out
    assert "--direct_metric_overflow_accept_weight" in out
    assert "--direct_metric_radius_inter_ratio_weight" in out
    assert "--phase1_v2_hard_gates true" in out
    assert "--endpoint_accept_policy_id endpoint_accept_v1" in out
    assert "--loss_gate_exported false" in out
    assert "--tail_safety_state_machine true" in out
    assert "--tail_safety_p99_expansion_block_final_delta 2.0" in out
    assert "--tail_safety_p99_expansion_block_best_delta 3.5" in out
    assert "--os_eff_min_budget 0.15" in out
    assert "--u_tri_state_required true" in out
    assert "--u_direct_idle_blocks_promotion true" in out
    assert "--source_episode_density_gate true" in out
    assert "--source_episode_min_local_components 4" in out
    assert "--feasibility_gate true" in out
    assert "--feasibility_stage audit" in out
    assert "phase1_source_zid_prototypes.pt" in out
    assert "phase2_zid_prototypes.pt" not in out
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "ManyTx.pkl" not in out


def test_v2full32_assigns_four_candidates_per_gpu_and_contains_ladder_plus_ablations():
    out = _dry_run()

    assert "candidates=32" in out
    assert "four_per_gpu=1" in out
    assert out.count("[V2FULL32-CANDIDATE]") == 32
    gpu_ids = re.findall(r"CUDA_VISIBLE_DEVICES=([0-7])", out)
    assert sorted(gpu_ids) == sorted(str(gpu) for gpu in range(8) for _ in range(4))
    for strength in ("weak", "stable", "aggressive", "aggressive_safe"):
        assert f"strength={strength}" in out
    for group in (
        "G0_FULL_LADDER",
        "G1_DIRECT_LOSS_ABLATION",
        "G2_KNOWN_GEOMETRY",
        "G3_PROXY_BRIDGE",
        "G4_U_TRISTATE",
        "G5_SAT_DG_STRESS",
        "G6_GRADIENT_BUDGET",
        "G7_EXPORT_GATE",
    ):
        assert f"group={group}" in out
    for route in (
        "direct_metric_off_endpoint_eval",
        "source_episode_off_density_eval",
        "proxy_unknown_off_endpoint_eval",
        "u_branch_off_tri_state_eval",
        "u_domain_sat_only",
        "sat_domain_adv_strong",
        "os_budget_high",
        "local_component_export_strict",
    ):
        assert f"route={route}" in out


def test_v2full32_mechanism_ablations_change_the_actual_loss_weights():
    dm_off = _dry_run("--only=DGLEO_V2FULL32_DM_OFF")
    assert "route=direct_metric_off_endpoint_eval" in dm_off
    assert "direct_open_set_metric_loss=0" in dm_off
    assert "--lambda_direct_metric_accept 0.0000" in dm_off

    source_off = _dry_run("--only=DGLEO_V2FULL32_SOURCE_OFF")
    assert "route=source_episode_off_density_eval" in source_off
    assert "--lambda_source_episode 0.0000" in source_off

    proxy_off = _dry_run("--only=DGLEO_V2FULL32_PROXY_OFF")
    assert "route=proxy_unknown_off_endpoint_eval" in proxy_off
    assert "--lambda_proxy_unknown 0.0000" in proxy_off

    u_off = _dry_run("--only=DGLEO_V2FULL32_U_OFF")
    assert "route=u_branch_off_tri_state_eval" in u_off
    assert "unlabeled_domain_supervision=0" in u_off
    assert "unlabeled_satellite_consistency=0" in u_off
    assert "unlabeled_direct_metric_accept=0" in u_off
    assert "unlabeled_quarantine_accept=0" in u_off
    assert "--u_tri_state_required true" in u_off


def test_v2full32_rejects_non_source_phase1_inputs():
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"WISIG_PKL=/tmp/ManyTx.pkl bash {SCRIPT} --dry-run --only=DGLEO_V2FULL32_FULL_STABLE",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
