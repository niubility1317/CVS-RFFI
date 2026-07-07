import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
SCRIPT = "code/scripts/launch_phase1_dgleo_uopt24_20260707.sh"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import build_arg_parser  # noqa: E402


def _dry_run(*extra: str) -> str:
    result = subprocess.run(
        ["bash", SCRIPT, "--dry-run", *extra],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_train_parser_exposes_unlabeled_direct_optimization_args():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            "runs/tmp",
            "--lambda_u_domain",
            "0.25",
            "--lambda_u_adv",
            "0.12",
            "--lambda_u_sat_cons",
            "0.30",
            "--lambda_u_direct_metric_accept",
            "0.006",
            "--u_direct_metric_min_selected",
            "24",
        ]
    )

    assert args.lambda_u_domain == 0.25
    assert args.lambda_u_adv == 0.12
    assert args.lambda_u_sat_cons == 0.30
    assert args.lambda_u_direct_metric_accept == 0.006
    assert args.u_direct_metric_min_selected == 24
    assert args.u_sat_cons_start_epoch >= 1


def test_uopt24_declares_source_only_unlabeled_concat_sat_direct_training():
    out = _dry_run("--only=DGLEO_UOPT_P0_CORE_A")

    assert "DGLEO_UOPT_P0_CORE_A" in out
    assert "algorithm=DGLEO_UOPT24" in out
    assert "phase1_dataset=ManySig_only" in out
    assert "source_only=1" in out
    assert "real_unknown_classes_in_training=0" in out
    assert "target_receiver_samples_in_training=0" in out
    assert "manytx_in_training=0" in out
    assert "unlabeled_direct_optimization=1" in out
    assert "unlabeled_domain_supervision=1" in out
    assert "unlabeled_satellite_consistency=1" in out
    assert "unlabeled_direct_metric_accept=1" in out
    assert "concat_sa=1" in out
    assert "--use_concat_sat_channel_aug" in out
    assert "--no_concat_sat_ce_only" in out
    assert "--lambda_u_domain 0.18" in out
    assert "--lambda_u_adv 0.08" in out
    assert "--lambda_u_sat_cons 0.26" in out
    assert "--lambda_u_direct_metric_accept 0.0045" in out
    assert "--lambda_u_quarantine_accept" in out
    assert "--u_direct_metric_min_selected 20" in out
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
    assert "--new_wisig_pkl" not in out
    assert "--target_unknown" not in out
    assert "ManyTx.pkl" not in out


def test_uopt24_assigns_three_additional_candidates_per_gpu():
    out = _dry_run()

    assert "candidates=24" in out
    assert out.count("[UOPT24-CANDIDATE]") == 24
    gpu_ids = re.findall(r"CUDA_VISIBLE_DEVICES=([0-7])", out)
    assert sorted(gpu_ids) == sorted(str(gpu) for gpu in range(8) for _ in range(3))
    for group in ("P0_CORE", "P0_GATE", "P0_SAT", "P0_BAL", "P1_QUOTA", "P1_LATE", "P1_ADV", "P1_STRONG"):
        assert f"group={group}" in out


def test_uopt24_rejects_non_source_phase1_inputs():
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"WISIG_PKL=/tmp/ManyTx.pkl bash {SCRIPT} --dry-run --only=DGLEO_UOPT_P0_CORE_A",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing non-source Phase1 WISIG_PKL" in result.stderr
