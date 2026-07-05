import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_ssdg_parser_accepts_fixed_adv3b02_teacher_distillation_args():
    import sys

    code_root = PROJECT_ROOT / "code"
    ssdg_root = code_root / "SSDG"
    for path in (str(code_root), str(ssdg_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from train_ssdg import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "out",
            "--baseline_ckpt",
            "/runs/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth",
            "--from_scratch",
            "false",
            "--teacher_ckpt",
            "/runs/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth",
            "--lambda_teacher_clean_kl",
            "0.35",
            "--lambda_teacher_sat_kl",
            "0.20",
            "--lambda_teacher_zid_mse",
            "0.04",
            "--teacher_distill_temperature",
            "2.5",
            "--teacher_distill_start_epoch",
            "1",
            "--teacher_distill_warmup_epochs",
            "20",
        ]
    )

    assert args.teacher_ckpt.endswith("best_joint_safe_ssdg.pth")
    assert args.lambda_teacher_clean_kl == 0.35
    assert args.lambda_teacher_sat_kl == 0.20
    assert args.lambda_teacher_zid_mse == 0.04
    assert args.teacher_distill_temperature == 2.5
    assert args.teacher_distill_start_epoch == 1
    assert args.teacher_distill_warmup_epochs == 20


def test_epoc_adv3b02_launcher_dry_run_declares_teacher_distilled_open_set_route():
    result = subprocess.run(
        ["bash", "code/scripts/launch_phase1_epoc_adv3b02_distill_20260705.sh", "--dry-run", "--only=EPOC_DISTILL_A_MILD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "EPOC_DISTILL_A_MILD" in out
    assert "target_visibility=source_only_ground_training_no_target_receiver" in out
    assert "--baseline_ckpt" in out
    assert "ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth" in out
    assert "--teacher_ckpt" in out
    assert "--lambda_teacher_clean_kl" in out
    assert "--lambda_teacher_sat_kl" in out
    assert "--lambda_teacher_zid_mse" in out
    assert "--lambda_proxy_unknown" in out
    assert "--lambda_open_world_feat" in out
    assert "--lambda_source_episode" in out
    assert "--lambda_soft_unknown_mixup" in out
    assert "--new_wisig_pkl" not in out
    assert "ManyTx.pkl" not in out
    assert "proxy_unknown_tx_ids" not in out
    assert "--sat_train_scenarios" in out
    assert "leo_clear_weak" in out
    assert "leo_low_elev_weak" in out
    assert "leo_rain_weak" in out
    assert "--phase2_export_prototypes" in out
    assert "--best_metric joint_safe" in out
    assert "--enable_joint_safe_guard true" in out
    assert "target_unknown" not in out
