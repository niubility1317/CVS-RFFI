from __future__ import annotations

"""Behavior contracts for the frozen six-fold ADV3B02 CLIC-equivalent entry.

Each assertion protects a launch-time behavior, not a source-text convention:
wrong profile values, TX roles, checkpoint policy, parser syntax, root reuse, or
accidental target-side input must make the real launcher fail this suite.
"""

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_adv3b02_clic6_v1_20260813.sh"
SMOKE_LAUNCHER = CODE_ROOT / "scripts" / "smoke_phase1_adv3b02_clic_f1_v1_20260813.sh"
SMOKE_ROOT_NAME = ".smoke_phase1_adv3b02_clic6_20260813_v1_F1"
V2_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_adv3b02_clic6_v2_20260816.sh"
V2_SMOKE_LAUNCHER = CODE_ROOT / "scripts" / "smoke_phase1_adv3b02_clic_f1_v2_20260816.sh"
V2_RUN_ID = "phase1_adv3b02_clic6_20260816_v2"
V2_SMOKE_ROOT_NAME = ".smoke_phase1_adv3b02_clic6_20260816_v2_F1"
FORMAL_WISIG_PKL = "/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def _launcher_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PROJECT_ROOT": _wsl_path(tmp_path / "project"),
        "CODE_ROOT": _wsl_path(CODE_ROOT),
        "PYTHON": "python",
        "RUN_ID": "phase1_adv3b02_clic6_20260813_v1",
        "RUN_ROOT": _wsl_path(tmp_path / "runs"),
        "LOG_ROOT": _wsl_path(tmp_path / "logs"),
        "WISIG_PKL": _wsl_path(tmp_path / "ManySig.pkl"),
    }


def _smoke_env(tmp_path: Path) -> dict[str, str]:
    """Only the project root is configurable; formal roots stay unreachable."""

    return {
        "PROJECT_ROOT": _wsl_path(tmp_path / "project"),
        "CODE_ROOT": _wsl_path(CODE_ROOT),
        "PYTHON": "python",
        "WISIG_PKL": _wsl_path(tmp_path / "ManySig.pkl"),
    }


def _v2_launcher_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PROJECT_ROOT": _wsl_path(tmp_path / "project"),
        "CODE_ROOT": _wsl_path(CODE_ROOT),
        "PYTHON": "python",
        "RUN_ID": V2_RUN_ID,
        "RUN_ROOT": _wsl_path(tmp_path / "runs"),
        "LOG_ROOT": _wsl_path(tmp_path / "logs"),
        "WISIG_PKL": _wsl_path(tmp_path / "ManySig.pkl"),
    }


def _v2_smoke_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PROJECT_ROOT": _wsl_path(tmp_path / "project"),
        "CODE_ROOT": _wsl_path(CODE_ROOT),
        "PYTHON": "python",
        "WISIG_PKL": _wsl_path(tmp_path / "ManySig.pkl"),
    }


def _require_launcher() -> None:
    assert LAUNCHER.is_file(), f"missing launcher under test: {LAUNCHER}"


def _launcher_argv(
    tmp_path: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
    # WSL's bash.exe does not inherit arbitrary Windows subprocess env values.
    # Inject only this test's explicit contract inputs inside the WSL shell.
    env = _launcher_env(tmp_path)
    if env_overrides:
        env.update(env_overrides)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = " ".join(
        [
            assignments,
            "bash",
            shlex.quote(_wsl_path(LAUNCHER)),
            *(shlex.quote(value) for value in args),
        ]
    )
    return ["bash", "-lc", command]


def _run_launcher(
    tmp_path: Path,
    *args: str,
    check: bool = True,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _require_launcher()
    return subprocess.run(
        _launcher_argv(tmp_path, *args, env_overrides=env_overrides),
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _smoke_argv(
    tmp_path: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
    assert SMOKE_LAUNCHER.is_file(), f"missing smoke launcher: {SMOKE_LAUNCHER}"
    env = _smoke_env(tmp_path)
    if env_overrides:
        env.update(env_overrides)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = " ".join(
        [
            assignments,
            "bash",
            shlex.quote(_wsl_path(SMOKE_LAUNCHER)),
            *(shlex.quote(value) for value in args),
        ]
    )
    return ["bash", "-lc", command]


def _run_smoke(
    tmp_path: Path,
    *args: str,
    check: bool = True,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _smoke_argv(tmp_path, *args, env_overrides=env_overrides),
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _v2_launcher_argv(
    tmp_path: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
    assert V2_LAUNCHER.is_file(), f"missing v2 launcher under test: {V2_LAUNCHER}"
    env = _v2_launcher_env(tmp_path)
    if env_overrides:
        env.update(env_overrides)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = " ".join(
        [
            assignments,
            "bash",
            shlex.quote(_wsl_path(V2_LAUNCHER)),
            *(shlex.quote(value) for value in args),
        ]
    )
    return ["bash", "-lc", command]


def _run_v2_launcher(
    tmp_path: Path,
    *args: str,
    check: bool = True,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _v2_launcher_argv(tmp_path, *args, env_overrides=env_overrides),
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _v2_smoke_argv(
    tmp_path: Path,
    *args: str,
    env_overrides: dict[str, str] | None = None,
) -> list[str]:
    assert V2_SMOKE_LAUNCHER.is_file(), f"missing v2 smoke launcher: {V2_SMOKE_LAUNCHER}"
    env = _v2_smoke_env(tmp_path)
    if env_overrides:
        env.update(env_overrides)
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    command = " ".join(
        [
            assignments,
            "bash",
            shlex.quote(_wsl_path(V2_SMOKE_LAUNCHER)),
            *(shlex.quote(value) for value in args),
        ]
    )
    return ["bash", "-lc", command]


def _run_v2_smoke(
    tmp_path: Path,
    *args: str,
    check: bool = True,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _v2_smoke_argv(tmp_path, *args, env_overrides=env_overrides),
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _smoke_dry_run_line(tmp_path: Path) -> str:
    result = _run_smoke(tmp_path, "--dry-run")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("[DRY-RUN] ")
    return lines[0]


def _v2_smoke_dry_run_line(tmp_path: Path) -> str:
    result = _run_v2_smoke(tmp_path, "--dry-run")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("[DRY-RUN] ")
    return lines[0]


def _dry_run_lines(tmp_path: Path) -> list[str]:
    result = _run_launcher(tmp_path, "--dry-run")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 6
    assert all(line.startswith("[DRY-RUN] ") for line in lines)
    return lines


def _command_tokens(line: str) -> list[str]:
    return shlex.split(line.removeprefix("[DRY-RUN] "))


def _trainer_args(tokens: list[str]) -> list[str]:
    train_index = next(
        index
        for index, token in enumerate(tokens)
        if token.endswith("/SSDG/train_ssdg.py")
    )
    return tokens[train_index + 1 :]


def _arg_map(args: list[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            index += 1
            continue
        assert token not in values, f"duplicate flag in dry-run command: {token}"
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            values[token] = args[index + 1]
            index += 2
        else:
            values[token] = None
            index += 1
    return values


FOLDS = (
    ("F1_ADV3B02_CLIC", "20-15,20-19,6-15,8-20", "14-7", "14-10", "0"),
    ("F2_ADV3B02_CLIC", "14-10,20-19,6-15,8-20", "20-15", "14-7", "1"),
    ("F3_ADV3B02_CLIC", "14-10,14-7,6-15,8-20", "20-19", "20-15", "2"),
    ("F4_ADV3B02_CLIC", "14-10,14-7,20-15,8-20", "6-15", "20-19", "3"),
    ("F5_ADV3B02_CLIC", "14-10,14-7,20-15,20-19", "8-20", "6-15", "4"),
    ("F6_ADV3B02_CLIC", "14-7,20-15,20-19,6-15", "14-10", "8-20", "5"),
)


# Literal profile values resolved by historical set_candidate_defaults plus the
# ADV3B02_CORE90_SOFT_E200 branch.  Dynamic output/candidate fields are checked
# separately below.
EXPECTED_PROFILE = {
    "--from_scratch": "true",
    "--split_mode": "tx_rx_day_1_6_3",
    "--labeled_ratio": "0.07",
    "--unlabeled_ratio": "0.63",
    "--source_val_ratio": "0.30",
    "--base_candidate": "ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL",
    "--epochs": "200",
    "--label_epochs": "130",
    "--pseudo_epochs": "70",
    "--best_metric": "source_val_sat_hmean",
    "--checkpoint_selection": "final_only",
    "--phase1_source_val_selection_only": "true",
    "--enable_joint_safe_guard": "false",
    "--one_epoch_drop_guard_pp": "2.0",
    "--paic_guard_enabled": "true",
    "--paic_guard_sat_ce_delta": "0.12",
    "--paic_guard_grad_delta": "3.0",
    "--paic_guard_reliable_drop": "0.01",
    "--paic_guard_cooldown_epochs": "1",
    "--paic_guard_sat_scale": "0.75",
    "--use_phase2_ground_prototypes": "true",
    "--use_feature_masks": "true",
    "--use_txrx_geometry_losses": "true",
    "--use_tx_rx_balanced_sampler": "false",
    "--phase1_distribution_audit_only": "true",
    "--lambda_tx_proto": "0",
    "--lambda_rx_proto": "0",
    "--lambda_mask_aux": "0",
    "--lambda_tx_supcon_masked": "0",
    "--lambda_rx_supcon_masked": "0",
    "--lambda_txrx_rect": "0",
    "--use_proto_memory": "true",
    "--lambda_proto": "0.0032",
    "--proto_domain_align_weight": "0.10",
    "--proto_margin": "0.15",
    "--proto_push_weight": "0.10",
    "--proto_min_count": "2",
    "--lambda_open_world_feat": "0.0024",
    "--ow_feat_start_epoch": "12",
    "--ow_feat_warmup_epochs": "25",
    "--ow_feat_radius_deg": "12",
    "--ow_feat_inter_margin_deg": "55",
    "--ow_feat_sample_margin_deg": "5",
    "--ow_feat_domain_align_weight": "0",
    "--ow_feat_min_classes": "2",
    "--ow_feat_min_samples_per_class": "1",
    "--ow_feat_tail_mode": "robust_3sigma",
    "--ow_feat_tail_weight": "0.14",
    "--ow_feat_cvar_alpha": "0.95",
    "--ow_feat_vacuum_weight": "0.40",
    "--ow_feat_vacuum_width_deg": "6",
    "--ow_feat_vacuum_hard_k": "3",
    "--lambda_zid_compact": "0.032",
    "--zid_compact_start_epoch": "8",
    "--zid_compact_warmup_epochs": "25",
    "--zid_compact_supcon_weight": "0.30",
    "--zid_compact_radius_weight": "0.35",
    "--zid_compact_cvar_weight": "0.35",
    "--zid_compact_cvar_alpha": "0.95",
    "--zid_compact_radius_deg": "40",
    "--zid_compact_domain_aware": "true",
    "--lambda_proxy_unknown": "0.0045",
    "--proxy_unknown_start_epoch": "45",
    "--proxy_unknown_warmup_epochs": "25",
    "--proxy_unknown_holdout_tx_per_batch": "1",
    "--proxy_unknown_virtual_count": "48",
    "--proxy_unknown_virtual_mode": "hard",
    "--proxy_unknown_energy_margin": "0.0",
    "--proxy_unknown_energy_temperature": "1.0",
    "--proxy_unknown_placeholder_weight": "0.0",
    "--proxy_unknown_virtual_detach": "false",
    "--proxy_unknown_vacuum_weight": "0.55",
    "--proxy_unknown_vacuum_width_deg": "5",
    "--proxy_unknown_vacuum_hard_k": "3",
    "--proxy_unknown_vacuum_radius_deg": "40",
    "--proxy_unknown_core_quantile": "0.90",
    "--proxy_unknown_accept_quantile": "0.85",
    "--proxy_unknown_tail_quantile": "0.92",
    "--proxy_unknown_overflow_quantile": "0.97",
    "--proxy_unknown_vaccept_weight": "1.00",
    "--proxy_unknown_core_accept_weight": "0.45",
    "--proxy_unknown_component_gate_weight": "0.65",
    "--proxy_unknown_tail_quarantine_weight": "0.20",
    "--proxy_unknown_source_safe_weight": "0.20",
    "--proxy_unknown_vaccept_cvar_alpha": "0.30",
    "--proxy_unknown_unknown_margin": "0.08",
    "--proxy_unknown_known_margin": "0.05",
    "--proxy_unknown_energy_softplus_temperature": "0.04",
    "--proxy_unknown_component_temperature_deg": "3.0",
    "--proxy_unknown_component_margin_deg": "4.0",
    "--proxy_unknown_component_margin_temperature_deg": "3.0",
    "--proxy_unknown_shell_width_deg": "4.0",
    "--lambda_soft_unknown_mixup": "0.0045",
    "--soft_unknown_mixup_start_epoch": "25",
    "--soft_unknown_mixup_warmup_epochs": "25",
    "--soft_unknown_mixup_count": "24",
    "--soft_unknown_mixup_order": "3",
    "--soft_unknown_mixup_alpha": "0.5",
    "--soft_unknown_mixup_energy_margin": "1.0",
    "--soft_unknown_mixup_ce_weight": "0.60",
    "--soft_unknown_mixup_energy_weight": "1.0",
    "--soft_unknown_mixup_vacuum_weight": "0.35",
    "--soft_unknown_mixup_vacuum_width_deg": "6",
    "--soft_unknown_mixup_vacuum_hard_k": "3",
    "--soft_unknown_mixup_detach": "false",
    "--lambda_source_episode": "0.0035",
    "--source_episode_start_epoch": "20",
    "--source_episode_warmup_epochs": "25",
    "--source_episode_min_domains": "2",
    "--source_episode_radius_cap_deg": "33",
    "--source_episode_mixup_weight": "0.75",
    "--source_episode_mixup_hard_k": "3",
    "--phase2_export_prototypes": "true",
    "--phase2_export_feature_key": "z_id",
    "--phase2_export_split": "train",
    "--phase2_fuse_prototypes": "true",
    "--phase2_fuse_max_components": "6",
    "--phase2_fuse_merge_angle_deg": "2.5",
    "--phase2_fuse_radius_cap_deg": "15.0",
    "--phase2_fuse_tail_abs_deg": "24",
    "--phase2_fuse_accept_policy": "local_component",
    "--phase2_fuse_accept_radius_key": "p95",
    "--phase2_fuse_max_p95_increase_deg": "2.0",
    "--phase2_fuse_keep_tail_sentinel": "true",
    "--phase2_fuse_global_ball_accept": "false",
    "--test_eval_policy": "interval_final",
    "--test_eval_start_epoch": "1",
    "--test_eval_interval": "10",
    "--test_eval_final_window": "20",
    "--test_eval_final_interval": "2",
    "--use_sat_consistency": None,
    "--use_concat_sat_channel_aug": None,
    "--concat_sat_ce_only": None,
    "--sat_train_scenario": "leo_clear_weak",
    "--sat_train_scenarios": "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    "--sat_view_schedule": "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    "--sat_cons_start_epoch": "80",
    "--lambda_sat_cls": "0.68",
    "--lambda_sat_cons": "0",
    "--lambda_u": "0.16",
    "--lambda_ent": "0.01",
    "--lambda_domain": "1",
    "--lambda_adv": "0.35",
    "--lambda_group_ce": "0.16",
    "--lambda_fishr": "0.04",
    "--tau_min": "0.92",
    "--tau_max": "0.97",
    "--pseudo_quantile": "0.86",
    "--use_ema_teacher": "true",
    "--eval_sat_channel": "true",
    "--eval_sat_scenarios": "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    "--sat_eval_max_batches": "-1",
    "--device": "cuda:0",
    "--seed": "392002",
}


def test_dry_run_emits_exact_six_source_only_fold_commands(tmp_path: Path) -> None:
    """Break caught: a wrong role, profile value, fold order, or GPU mapping."""

    lines = _dry_run_lines(tmp_path)
    for line, (candidate, train_tx, known_tx, proxy_tx, gpu) in zip(lines, FOLDS):
        tokens = _command_tokens(line)
        assert tokens[0] == f"CUDA_VISIBLE_DEVICES={gpu}"
        values = _arg_map(_trainer_args(tokens))
        assert values["--candidate_id"] == candidate
        assert values["--run_id"] == "phase1_adv3b02_clic6_20260813_v1"
        assert values["--phase1_source_train_tx_ids"] == train_tx
        assert values["--phase1_source_known_validation_tx_ids"] == known_tx
        assert values["--phase1_source_proxy_unknown_tx_ids"] == proxy_tx
        assert values["--output_dir"].endswith(f"/{candidate}")
        assert values["--phase2_export_path"].endswith(
            f"/{candidate}/phase2_zid_prototypes.pt"
        )
        for flag, expected in EXPECTED_PROFILE.items():
            assert values.get(flag) == expected, f"{candidate}: {flag}"


def test_dry_run_command_is_accepted_by_the_real_train_parser(tmp_path: Path) -> None:
    """Break caught: boolean or final-only syntax rejected by train_ssdg.py."""

    from SSDG import train_ssdg

    parser = train_ssdg.build_arg_parser()
    for line in _dry_run_lines(tmp_path):
        parsed = parser.parse_args(_trainer_args(_command_tokens(line)))
        assert parsed.from_scratch is True
        assert parsed.checkpoint_selection == "final_only"
        assert parsed.split_mode == "tx_rx_day_1_6_3"
        assert parsed.labeled_ratio == 0.07
        assert parsed.unlabeled_ratio == 0.63
        assert parsed.source_val_ratio == 0.30


def test_all_six_commands_pass_the_real_train_runtime_dry_run_guard(tmp_path: Path) -> None:
    """Break caught: a parsed ADV command violates train_ssdg's runtime guards."""

    from SSDG import train_ssdg

    parser = train_ssdg.build_arg_parser()
    for line in _dry_run_lines(tmp_path):
        args = parser.parse_args(_trainer_args(_command_tokens(line)) + ["--dry_run"])
        assert train_ssdg.train(args) == 0


def test_adv3b02_smoke_launcher_emits_one_isolated_f1_command(tmp_path: Path) -> None:
    """Break caught: a smoke run can drift from F1 or write under the formal roots."""

    assert SMOKE_LAUNCHER.is_file()
    syntax = subprocess.run(
        ["bash", "-n", _wsl_path(SMOKE_LAUNCHER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    line = _smoke_dry_run_line(tmp_path)
    tokens = _command_tokens(line)
    values = _arg_map(_trainer_args(tokens))
    assert tokens[0] == "CUDA_VISIBLE_DEVICES=0"
    assert values["--candidate_id"] == "F1_ADV3B02_CLIC"
    assert values["--run_id"] == "phase1_adv3b02_clic6_20260813_v1"
    assert values["--phase1_adv3b02_technical_smoke_batches"] == "3"
    assert values["--output_dir"].endswith(f"/{SMOKE_ROOT_NAME}/F1_ADV3B02_CLIC")
    for flag, expected in EXPECTED_PROFILE.items():
        assert values.get(flag) == expected, f"smoke F1: {flag}"

    project = tmp_path / "project"
    assert not (project / "runs" / "phase1_adv3b02_clic6_20260813_v1").exists()
    assert not (project / "logs" / "phase1_adv3b02_clic6_20260813_v1").exists()
    assert not (project / "runs" / SMOKE_ROOT_NAME).exists()
    assert not (project / "logs" / SMOKE_ROOT_NAME).exists()


def test_adv3b02_v2_launchers_preserve_f1_profile_and_use_new_paths(tmp_path: Path) -> None:
    """Break caught: v2 drifts a frozen method field or reuses a v1 run/smoke root."""

    assert V2_LAUNCHER.is_file()
    assert V2_SMOKE_LAUNCHER.is_file()
    for launcher in (V2_LAUNCHER, V2_SMOKE_LAUNCHER):
        syntax = subprocess.run(
            ["bash", "-n", _wsl_path(launcher)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr

    formal = _run_v2_launcher(tmp_path, "--dry-run")
    formal_lines = [line for line in formal.stdout.splitlines() if line.strip()]
    assert len(formal_lines) == 6
    for line, (candidate, train_tx, known_tx, proxy_tx, gpu) in zip(formal_lines, FOLDS):
        tokens = _command_tokens(line)
        values = _arg_map(_trainer_args(tokens))
        assert tokens[0] == f"CUDA_VISIBLE_DEVICES={gpu}"
        assert values["--run_id"] == V2_RUN_ID
        assert values["--candidate_id"] == candidate
        assert values["--phase1_source_train_tx_ids"] == train_tx
        assert values["--phase1_source_known_validation_tx_ids"] == known_tx
        assert values["--phase1_source_proxy_unknown_tx_ids"] == proxy_tx
        for flag, expected in EXPECTED_PROFILE.items():
            assert values.get(flag) == expected, f"v2 {candidate}: {flag}"

    smoke_tokens = _command_tokens(_v2_smoke_dry_run_line(tmp_path))
    smoke_values = _arg_map(_trainer_args(smoke_tokens))
    assert smoke_tokens[0] == "CUDA_VISIBLE_DEVICES=0"
    assert smoke_values["--run_id"] == V2_RUN_ID
    assert smoke_values["--candidate_id"] == "F1_ADV3B02_CLIC"
    assert smoke_values["--phase1_adv3b02_technical_smoke_v2_max_batches"] == "4"
    assert "--phase1_adv3b02_technical_smoke_batches" not in smoke_values
    assert smoke_values["--output_dir"].endswith(
        f"/{V2_SMOKE_ROOT_NAME}/F1_ADV3B02_CLIC"
    )


def _write_v2_smoke_receipt(project_root: Path) -> Path:
    receipt_path = (
        project_root
        / "runs"
        / V2_SMOKE_ROOT_NAME
        / "F1_ADV3B02_CLIC"
        / "phase1_adv3b02_technical_smoke_v2_receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "cvs.phase1.adv3b02_technical_smoke.v2",
                "completed": True,
                "claim": "NO_PERFORMANCE_RESULT",
                "base_candidate": "ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL",
                "run_id": V2_RUN_ID,
                "candidate_id": "F1_ADV3B02_CLIC",
                "fold": "F1",
                "raw_batch_cap": 4,
                "raw_batches_observed": 4,
                "target_effective_steps": 3,
                "effective_forward_steps": 3,
                "effective_backward_steps": 3,
                "optimizer_attempts": 3,
                "optimizer_effective_steps": 3,
                "skipped_nonfinite_loss_batches": 0,
                "skipped_nonfinite_grad_batches": 1,
                "handled_grad_skip_count": 1,
                "source_val_rows_opened": 0,
                "query_rows_opened": 0,
                "target_rows_opened": 0,
                "test_rows_opened": 0,
                "selection_feedback_count": 0,
                "raw_batch_records": [
                    {
                        "raw_batch_index": index,
                        "loss_finite": True,
                        "grad_finite": index != 1,
                        "optimizer_attempted": index != 1,
                        "optimizer_effective": index != 1,
                        "amp_scale_before": 1.0,
                        "amp_scale_after": 1.0,
                    }
                    for index in range(1, 5)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return receipt_path


def test_adv3b02_v2_formal_rejects_forged_receipt_guard_fields_before_roots(
    tmp_path: Path,
) -> None:
    """Break caught: forged method/access receipts can pass v2's formal gate."""

    cases = (
        (
            "review_attack",
            {
                "base_candidate": "FORGED_METHOD",
                "source_val_rows_opened": 1,
                "query_rows_opened": 1,
                "target_rows_opened": 1,
                "test_rows_opened": 1,
                "selection_feedback_count": 1,
            },
            None,
        ),
        ("wrong_fold", {"fold": "F2"}, None),
        ("bool_access", {"source_val_rows_opened": False}, None),
        ("missing_access", {}, "selection_feedback_count"),
    )
    for case_name, replacements, removed_field in cases:
        case_root = tmp_path / case_name
        receipt_path = _write_v2_smoke_receipt(case_root / "project")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update(replacements)
        if removed_field is not None:
            receipt.pop(removed_field)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

        blocked = _run_v2_launcher(case_root, check=False)

        assert blocked.returncode != 0, case_name
        assert "requires a complete v2 technical smoke receipt" in blocked.stderr
        assert not (case_root / "runs").exists(), case_name
        assert not (case_root / "logs").exists(), case_name


def test_adv3b02_v2_formal_requires_new_smoke_receipt_and_preserves_roots(
    tmp_path: Path,
) -> None:
    """Break caught: formal v2 launches without its new smoke receipt or overwrites a root."""

    missing = _run_v2_launcher(tmp_path, check=False)
    assert missing.returncode != 0
    assert "requires a complete v2 technical smoke receipt" in missing.stderr
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "logs").exists()

    _write_v2_smoke_receipt(tmp_path / "project")
    collision = tmp_path / "runs"
    collision.mkdir()
    marker = collision / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    blocked = _run_v2_launcher(tmp_path, check=False, env_overrides={"PYTHON": "/bin/false"})
    assert blocked.returncode != 0
    assert "refusing to overwrite run/log root" in blocked.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "logs").exists()


def test_adv3b02_smoke_launcher_rejects_formal_root_overrides_and_existing_smoke_root(
    tmp_path: Path,
) -> None:
    """Break caught: smoke can be redirected to a formal root or overwrite its own receipt root."""

    formal_override = _run_smoke(
        tmp_path,
        check=False,
        env_overrides={"RUN_ROOT": _wsl_path(tmp_path / "project" / "runs" / "formal")},
    )
    assert formal_override.returncode != 0
    assert "RUN_ROOT/LOG_ROOT overrides are forbidden" in formal_override.stderr

    smoke_root = tmp_path / "project" / "runs" / SMOKE_ROOT_NAME
    smoke_root.mkdir(parents=True)
    marker = smoke_root / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")
    collision = _run_smoke(tmp_path, check=False)
    assert collision.returncode != 0
    assert "refusing to overwrite smoke run/log root" in collision.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "project" / "logs" / SMOKE_ROOT_NAME).exists()
    assert not (tmp_path / "project" / "runs" / "phase1_adv3b02_clic6_20260813_v1").exists()


def _adv3b02_smoke_args(tmp_path: Path):
    from SSDG import train_ssdg

    parser = train_ssdg.build_arg_parser()
    line = _smoke_dry_run_line(tmp_path)
    args = parser.parse_args(_trainer_args(_command_tokens(line)))
    # The test wrapper's temporary PROJECT_ROOT intentionally changes its
    # dry-run dataset path.  Runtime profile validation is stricter: it must
    # bind back to the formal F1 ManySig path.
    args.wisig_pkl = FORMAL_WISIG_PKL
    return args


def _adv3b02_smoke_v2_args(tmp_path: Path):
    from SSDG import train_ssdg

    parser = train_ssdg.build_arg_parser()
    line = _v2_smoke_dry_run_line(tmp_path)
    args = parser.parse_args(_trainer_args(_command_tokens(line)))
    args.wisig_pkl = FORMAL_WISIG_PKL
    return args


def test_adv3b02_smoke_parser_accepts_zero_or_three_batches(tmp_path: Path) -> None:
    """Break caught: the dedicated technical-control flag cannot represent its only legal states."""

    from SSDG import train_ssdg

    parser = train_ssdg.build_arg_parser()
    assert parser.parse_args(["--output_dir", "unused"]).phase1_adv3b02_technical_smoke_batches == 0
    assert _adv3b02_smoke_args(tmp_path).phase1_adv3b02_technical_smoke_batches == 3


def test_adv3b02_smoke_exact_f1_profile_validates_before_data(tmp_path: Path) -> None:
    """Break caught: the isolated F1 wrapper no longer matches its formal command source."""

    from SSDG import train_ssdg

    assert train_ssdg._validate_phase1_adv3b02_technical_smoke_args(
        _adv3b02_smoke_args(tmp_path)
    ) == 3


@pytest.mark.parametrize("batches", (1, 2, 4))
def test_adv3b02_smoke_rejects_partial_batch_counts_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batches: int,
) -> None:
    """Break caught: a partial smoke invocation reaches data construction."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    args.phase1_adv3b02_technical_smoke_batches = batches
    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: pytest.fail("partial ADV smoke reached data construction"),
    )
    with pytest.raises(ValueError, match="zero or three"):
        train_ssdg.train(args)


def test_adv3b02_smoke_rejects_wrong_method_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: the technical smoke accepts a non-ADV3B02 method identity."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    args.base_candidate = "NOT_ADV3B02"
    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: pytest.fail("wrong-method ADV smoke reached data construction"),
    )
    with pytest.raises(ValueError, match="base_candidate"):
        train_ssdg.train(args)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lambda_proto", 9.99),
        ("lambda_tx_proto", 0.01),
        ("use_feature_masks", False),
        ("wisig_pkl", "/foreign/ManySig.pkl"),
        (
            "sat_view_schedule",
            "1@0.30:leo_clear_weak;41@0.60:leo_clear_weak,leo_rain_weak;"
            "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
        ),
    ),
)
def test_adv3b02_smoke_rejects_any_frozen_method_profile_drift_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Break caught: a smoke invocation can change a frozen F1 mechanism or loss."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    setattr(args, field, value)
    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: pytest.fail(
            f"ADV smoke profile drift reached data construction: {field}"
        ),
    )
    with pytest.raises(ValueError, match=field):
        train_ssdg.train(args)


def test_adv3b02_smoke_flag_zero_does_not_freeze_the_f1_profile(tmp_path: Path) -> None:
    """Break caught: ordinary flag=0 training is constrained by smoke-only F1 values."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    args.phase1_adv3b02_technical_smoke_batches = 0
    args.base_candidate = "ordinary_source_only_candidate"
    args.lambda_proto = 9.99
    args.use_feature_masks = False
    args.dry_run = True
    assert train_ssdg.train(args) == 0


def test_adv3b02_smoke_rejects_bash_env_forged_formal_profile_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: inherited BASH_ENV can forge the recovered formal F1 row."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    args.lambda_proto = 9.99
    bash_env = tmp_path / "forge-formal-profile.sh"
    bash_env.write_text(
        """printf() {
  if [[ \"$1\" == \" %q\" ]]; then
    shift
    for item in \"$@\"; do
      [[ \"$item\" == \"0.0032\" ]] && item='9.99'
      builtin printf ' %q' \"$item\"
    done
  else
    builtin printf \"$@\"
  fi
}
""",
        encoding="utf-8",
        newline="\n",
    )
    if os.name == "nt":
        # WSL imports only variables named through WSLENV.  Preserve the
        # explicit inherited-environment attack the production helper used to
        # permit.  Pass an already WSL-addressable path verbatim: the legacy
        # bash.exe launcher does not convert BASH_ENV reliably with ``/u``.
        wsl_entries = [
            item
            for item in os.environ.get("WSLENV", "").split(":")
            if item and not item.startswith("BASH_ENV/") and item != "BASH_ENV"
        ]
        monkeypatch.setenv("WSLENV", ":".join([*wsl_entries, "BASH_ENV"]))
        monkeypatch.setenv("BASH_ENV", _wsl_path(bash_env))
    else:
        monkeypatch.setenv("BASH_ENV", str(bash_env))

    attack = subprocess.run(
        ["bash", _wsl_path(LAUNCHER) if os.name == "nt" else str(LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    forged_values = _arg_map(_trainer_args(_command_tokens(attack.stdout.splitlines()[0])))
    assert forged_values["--lambda_proto"] == "9.99"
    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: pytest.fail(
            "forged formal profile reached ADV smoke data construction"
        ),
    )
    with pytest.raises(ValueError, match="lambda_proto"):
        train_ssdg.train(args)


def test_adv3b02_smoke_v2_reuses_profile_and_bash_env_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: v2 trusts a BASH_ENV-forged F1 profile instead of the frozen one."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_v2_args(tmp_path)
    assert train_ssdg._validate_phase1_adv3b02_technical_smoke_v2_args(args) == 4
    args.lambda_proto = 9.99
    bash_env = tmp_path / "forge-v2-formal-profile.sh"
    bash_env.write_text(
        """printf() {
  if [[ \"$1\" == \" %q\" ]]; then
    shift
    for item in \"$@\"; do
      [[ \"$item\" == \"0.0032\" ]] && item='9.99'
      builtin printf ' %q' \"$item\"
    done
  else
    builtin printf \"$@\"
  fi
}
""",
        encoding="utf-8",
        newline="\n",
    )
    if os.name == "nt":
        wsl_entries = [
            item
            for item in os.environ.get("WSLENV", "").split(":")
            if item and not item.startswith("BASH_ENV/") and item != "BASH_ENV"
        ]
        monkeypatch.setenv("WSLENV", ":".join([*wsl_entries, "BASH_ENV"]))
        monkeypatch.setenv("BASH_ENV", _wsl_path(bash_env))
    else:
        monkeypatch.setenv("BASH_ENV", str(bash_env))

    attack = subprocess.run(
        ["bash", _wsl_path(V2_LAUNCHER) if os.name == "nt" else str(V2_LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    forged_values = _arg_map(_trainer_args(_command_tokens(attack.stdout.splitlines()[0])))
    assert forged_values["--lambda_proto"] == "9.99"
    with pytest.raises(ValueError, match="lambda_proto"):
        train_ssdg._validate_phase1_adv3b02_technical_smoke_v2_args(args)


def test_adv3b02_smoke_rejects_bash_resolved_from_an_untrusted_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a caller PATH can replace the formal-profile shell."""

    from SSDG import train_ssdg

    actual_bash = shutil.which("bash")
    if actual_bash is None:
        pytest.skip("no local bash is available to exercise PATH rejection")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    if os.name == "nt":
        fake_bash = fake_bin / "bash.exe"
        shutil.copy2(actual_bash, fake_bash)
    else:
        fake_bash = fake_bin / "bash"
        fake_bash.write_text("#!/bin/sh\nexec /bin/bash \"$@\"\n", encoding="utf-8")
        fake_bash.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))

    args = _adv3b02_smoke_args(tmp_path)
    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: pytest.fail(
            "untrusted PATH bash reached ADV smoke data construction"
        ),
    )
    with pytest.raises(ValueError, match="untrusted bash"):
        train_ssdg.train(args)


def _smoke_source_evidence() -> dict[str, object]:
    return {
        "source_roles": {
            "train_tx_ids": ["20-15", "20-19", "6-15", "8-20"],
            "known_validation_tx_ids": ["14-7"],
            "proxy_unknown_tx_ids": ["14-10"],
            "partition_sha256": "a" * 64,
            "held_tx_loaded_by_training": False,
        },
        "source_split": {
            "mode": "tx_rx_day_1_6_3",
            "labeled_size": 3920,
            "unlabeled_size": 35280,
            "source_val_size": 16800,
            "split_manifest_sha256": "b" * 64,
        },
        "source_dataset": {
            "wisig_pkl_path": FORMAL_WISIG_PKL,
            "wisig_pkl_sha256": "e" * 64,
        },
    }


def _prepare_v2_real_training_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    grad_nonfinite_raw_indices: set[int],
    loss_nonfinite_raw_indices: set[int],
):
    """Build a real tiny train loop; only the source data adapter is synthetic."""

    torch = pytest.importorskip("torch")
    from SSDG import train_ssdg

    args = train_ssdg.build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / "v2-smoke-output"),
            "--from_scratch",
            "true",
            "--epochs",
            "2",
            "--label_epochs",
            "2",
            "--pseudo_epochs",
            "0",
            "--batch_size",
            "2",
            "--eval_batch_size",
            "2",
            "--device",
            "cpu",
            "--phase1_adv3b02_technical_smoke_v2_max_batches",
            "4",
        ]
    )
    for field in (
        "lambda_u",
        "lambda_ent",
        "lambda_domain",
        "lambda_adv",
        "lambda_orth",
        "lambda_cons",
        "lambda_group_ce",
        "lambda_fishr",
        "lambda_sat_cls",
        "lambda_sat_cons",
    ):
        setattr(args, field, 0.0)
    args.base_candidate = "ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL"
    args.run_id = V2_RUN_ID
    args.candidate_id = "F1_ADV3B02_CLIC"
    args.wisig_pkl = FORMAL_WISIG_PKL
    args.use_unlabeled = False
    args.use_sat_consistency = False
    args.use_mixstyle = False
    args.use_aug = False
    args.amp = False

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.forward_count = 0
            self.last_raw_index = -1

        def forward(self, x, **_kwargs):
            raw_index = self.forward_count
            self.forward_count += 1
            self.last_raw_index = raw_index
            logits = self.linear(x)
            if raw_index in grad_nonfinite_raw_indices:
                logits.register_hook(lambda grad: torch.full_like(grad, float("inf")))
            return {"tx_logits": logits, "z_id": logits}

    model = TinyModel()
    batches = [
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
            {},
        )
        for _ in range(4)
    ]
    source_roles = {
        "enabled": True,
        "held_tx_loaded_by_training": False,
        "source_known_train_tx": ["20-15", "20-19", "6-15", "8-20"],
        "source_known_validation_tx": ["14-7"],
        "source_proxy_unknown_tx": ["14-10"],
        "partition_sha256": "c" * 64,
    }
    data_ctx = {
        "train_loader": batches,
        "balanced_train_sampler": None,
        "unlabeled_loader": [],
        "val_loader": [object()],
        "named_test_loaders": {},
        "domain_label_map": {},
        "num_domains": 1,
        "input_len": 2,
        "num_classes": 2,
        "class_id_to_tx": ["20-15", "20-19"],
        "split_info": {
            "mode": "tx_rx_day_1_6_3",
            "labeled_size": 8,
            "unlabeled_size": 0,
            "source_val_size": 1,
            "source_split_receipt": {
                "split_manifest_sha256": "d" * 64,
                "wisig_pkl_sha256": "e" * 64,
            },
            "tx_partition_receipt": source_roles,
        },
    }
    source_val_attempts: list[str] = []
    monkeypatch.setattr(
        train_ssdg,
        "_validate_phase1_adv3b02_technical_smoke_v2_args",
        lambda _args: 4,
    )
    monkeypatch.setattr(train_ssdg, "resolve_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(train_ssdg, "_prepare_cuda_memory_audit", lambda _device: None)
    monkeypatch.setattr(train_ssdg, "set_seed", lambda _seed: None)
    monkeypatch.setattr(train_ssdg, "_build_ssdg_wisig_data", lambda *_args: data_ctx)
    monkeypatch.setattr(
        train_ssdg,
        "_build_manytx_real_oe_data",
        lambda *_args, **_kwargs: {"loader": None, "sampler": None, "receipt": {}},
    )
    monkeypatch.setattr(train_ssdg, "merge_checkpoint_args", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(train_ssdg, "_apply_model_cli_args", lambda model_args, _args: model_args)
    monkeypatch.setattr(train_ssdg, "build_baseline_model", lambda *_args: model)
    monkeypatch.setattr(train_ssdg, "move_batch", lambda batch, _device: batch)

    def core_losses(out_l, y_l, *_args, **_kwargs):
        zero = out_l["tx_logits"].sum() * 0.0
        loss_cls = torch.nn.functional.cross_entropy(out_l["tx_logits"], y_l)
        if model.last_raw_index in loss_nonfinite_raw_indices:
            loss_cls = loss_cls * float("nan")
        return {
            "loss_cls": loss_cls,
            "loss_dom": zero,
            "loss_adv": zero,
            "loss_cons": zero,
            "loss_orth": zero,
            "loss_group_ce": zero,
            "cons_cos": zero,
            "dom_acc": zero,
        }

    monkeypatch.setattr(train_ssdg, "compute_core_losses", core_losses)

    def no_source_validation(*_args, **_kwargs):
        source_val_attempts.append("opened")
        pytest.fail("v2 technical smoke opened source validation")

    monkeypatch.setattr(train_ssdg, "evaluate_loader", no_source_validation)
    return train_ssdg, args, model, source_val_attempts


def test_adv3b02_smoke_v2_real_grad_skip_then_three_effective_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: one actual post-backward inf-gradient cannot recover to three steps."""

    train_ssdg, args, model, source_val_attempts = _prepare_v2_real_training_sequence(
        tmp_path,
        monkeypatch,
        grad_nonfinite_raw_indices={0},
        loss_nonfinite_raw_indices=set(),
    )

    assert train_ssdg.train(args) == 0
    receipt = json.loads(
        (Path(args.output_dir) / "phase1_adv3b02_technical_smoke_v2_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema"] == "cvs.phase1.adv3b02_technical_smoke.v2"
    assert receipt["claim"] == "NO_PERFORMANCE_RESULT"
    assert receipt["raw_batch_cap"] == 4
    assert receipt["raw_batches_observed"] == 4
    assert receipt["target_effective_steps"] == 3
    assert receipt["effective_forward_steps"] == 3
    assert receipt["effective_backward_steps"] == 3
    assert receipt["optimizer_attempts"] == 3
    assert receipt["optimizer_effective_steps"] == 3
    assert receipt["skipped_nonfinite_loss_batches"] == 0
    assert receipt["skipped_nonfinite_grad_batches"] == 1
    assert receipt["handled_grad_skip_count"] == 1
    records = receipt["raw_batch_records"]
    assert [record["raw_batch_index"] for record in records] == [1, 2, 3, 4]
    assert [record["loss_finite"] for record in records] == [True, True, True, True]
    assert [record["grad_finite"] for record in records] == [False, True, True, True]
    assert [record["optimizer_attempted"] for record in records] == [False, True, True, True]
    assert [record["optimizer_effective"] for record in records] == [False, True, True, True]
    assert all(math.isfinite(float(record["amp_scale_before"])) for record in records)
    assert all(math.isfinite(float(record["amp_scale_after"])) for record in records)
    assert model.forward_count == 4
    assert source_val_attempts == []
    assert receipt["source_val_rows_opened"] == 0
    assert receipt["target_rows_opened"] == 0
    assert receipt["query_rows_opened"] == 0
    assert receipt["test_rows_opened"] == 0
    assert receipt["selection_feedback_count"] == 0


def test_adv3b02_smoke_v2_rejects_second_grad_skip_before_source_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a second nonfinite-gradient skip is silently tolerated."""

    train_ssdg, args, model, source_val_attempts = _prepare_v2_real_training_sequence(
        tmp_path,
        monkeypatch,
        grad_nonfinite_raw_indices={0, 1},
        loss_nonfinite_raw_indices=set(),
    )
    with pytest.raises(RuntimeError, match="at most one.*nonfinite gradient"):
        train_ssdg.train(args)
    assert model.forward_count == 2
    assert source_val_attempts == []
    assert not (Path(args.output_dir) / "phase1_adv3b02_technical_smoke_v2_receipt.json").exists()


def test_adv3b02_smoke_v2_rejects_nonfinite_loss_before_source_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a nonfinite loss is misclassified as a recoverable v2 grad skip."""

    train_ssdg, args, model, source_val_attempts = _prepare_v2_real_training_sequence(
        tmp_path,
        monkeypatch,
        grad_nonfinite_raw_indices=set(),
        loss_nonfinite_raw_indices={0},
    )
    with pytest.raises(RuntimeError, match="nonfinite loss"):
        train_ssdg.train(args)
    assert model.forward_count == 1
    assert source_val_attempts == []
    assert not (Path(args.output_dir) / "phase1_adv3b02_technical_smoke_v2_receipt.json").exists()


def test_adv3b02_smoke_v2_finalizer_rejects_raw_cap_exhaustion(
    tmp_path: Path,
) -> None:
    """Break caught: a four-raw-batch window can claim pass with fewer than three steps."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_v2_args(tmp_path)
    out_dir = tmp_path / "v2-cap-output"
    out_dir.mkdir()
    counters = {
        "raw_batch_cap": 4,
        "raw_batches_observed": 4,
        "target_effective_steps": 3,
        "effective_forward_steps": 2,
        "effective_backward_steps": 2,
        "optimizer_attempts": 2,
        "optimizer_effective_steps": 2,
        "skipped_nonfinite_loss_batches": 0,
        "skipped_nonfinite_grad_batches": 1,
        "handled_grad_skip_count": 1,
        "raw_batch_records": [
            {
                "raw_batch_index": index,
                "loss_finite": True,
                "grad_finite": index != 1,
                "optimizer_attempted": index in {2, 3},
                "optimizer_effective": index in {2, 3},
                "amp_scale_before": 1.0,
                "amp_scale_after": 1.0,
            }
            for index in range(1, 5)
        ],
    }
    with pytest.raises(RuntimeError, match="raw-batch cap"):
        train_ssdg._finalize_phase1_adv3b02_technical_smoke_v2(
            out_dir=out_dir,
            args=args,
            counters=counters,
            source_evidence=_smoke_source_evidence(),
        )
    assert not (out_dir / "phase1_adv3b02_technical_smoke_v2_receipt.json").exists()


def test_adv3b02_smoke_v2_control_zero_leaves_formal_rows_unrestricted(
    tmp_path: Path,
) -> None:
    """Break caught: adding v2 changes ordinary formal parser/runtime validation."""

    from SSDG import train_ssdg

    formal_line = _dry_run_lines(tmp_path)[1]
    args = train_ssdg.build_arg_parser().parse_args(_trainer_args(_command_tokens(formal_line)))
    assert args.phase1_adv3b02_technical_smoke_batches == 0
    assert args.phase1_adv3b02_technical_smoke_v2_max_batches == 0
    assert train_ssdg._validate_phase1_adv3b02_technical_smoke_v2_args(args) == 0


def test_adv3b02_smoke_finalizer_seals_three_finite_effective_batches(
    tmp_path: Path,
) -> None:
    """Break caught: an incomplete or non-effective probe can be called complete."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    out_dir = tmp_path / "smoke-output"
    out_dir.mkdir()
    counters = {
        "batches": 3,
        "forward_batches": 3,
        "backward_batches": 3,
        "optimizer_attempts": 3,
        "optimizer_effective_steps": 3,
        "optimizer_nonfinite_batches": 0,
    }

    receipt_path = train_ssdg._finalize_phase1_adv3b02_technical_smoke(
        out_dir=out_dir,
        args=args,
        counters=counters,
        source_evidence=_smoke_source_evidence(),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "cvs.phase1.adv3b02_technical_smoke.v1"
    assert receipt["claim"] == "NO_PERFORMANCE_RESULT"
    assert receipt["batches"] == 3
    assert receipt["forward_batches"] == 3
    assert receipt["backward_batches"] == 3
    assert receipt["optimizer_attempts"] == 3
    assert receipt["optimizer_effective_steps"] == 3
    assert receipt["optimizer_nonfinite_batches"] == 0
    assert receipt["source_val_rows_opened"] == 0
    assert receipt["query_rows_opened"] == 0
    assert receipt["target_rows_opened"] == 0
    assert receipt["test_rows_opened"] == 0
    assert receipt["selection_feedback_count"] == 0
    assert receipt["source_dataset"] == _smoke_source_evidence()["source_dataset"]


def test_adv3b02_smoke_source_evidence_binds_dataset_path_and_split_hash(
    tmp_path: Path,
) -> None:
    """Break caught: the receipt loses the exact source dataset binding."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    data_ctx = {
        "named_test_loaders": {},
        "split_info": {
            "mode": "tx_rx_day_1_6_3",
            "labeled_size": 3920,
            "unlabeled_size": 35280,
            "source_val_size": 16800,
            "source_split_receipt": {
                "split_manifest_sha256": "b" * 64,
                "wisig_pkl_sha256": "e" * 64,
            },
            "tx_partition_receipt": {
                "enabled": True,
                "held_tx_loaded_by_training": False,
                "source_known_train_tx": ["20-15", "20-19", "6-15", "8-20"],
                "source_known_validation_tx": ["14-7"],
                "source_proxy_unknown_tx": ["14-10"],
                "partition_sha256": "a" * 64,
            },
        },
    }
    evidence = train_ssdg._adv3b02_technical_smoke_source_evidence(data_ctx, args)
    assert evidence["source_dataset"] == {
        "wisig_pkl_path": FORMAL_WISIG_PKL,
        "wisig_pkl_sha256": "e" * 64,
    }


def test_adv3b02_smoke_finalizer_rejects_nonfinite_or_existing_receipt(
    tmp_path: Path,
) -> None:
    """Break caught: failed batches or an existing immutable receipt are overwritten."""

    from SSDG import train_ssdg

    args = _adv3b02_smoke_args(tmp_path)
    out_dir = tmp_path / "smoke-output"
    out_dir.mkdir()
    failed = {
        "batches": 3,
        "forward_batches": 3,
        "backward_batches": 2,
        "optimizer_attempts": 2,
        "optimizer_effective_steps": 2,
        "optimizer_nonfinite_batches": 1,
    }
    receipt_path = out_dir / "phase1_adv3b02_technical_smoke_receipt.json"
    with pytest.raises(RuntimeError, match="finite, effective optimizer steps"):
        train_ssdg._finalize_phase1_adv3b02_technical_smoke(
            out_dir=out_dir,
            args=args,
            counters=failed,
            source_evidence=_smoke_source_evidence(),
        )
    assert not receipt_path.exists()

    sentinel = b"foreign-immutable-receipt\n"
    receipt_path.write_bytes(sentinel)
    complete = {
        "batches": 3,
        "forward_batches": 3,
        "backward_batches": 3,
        "optimizer_attempts": 3,
        "optimizer_effective_steps": 3,
        "optimizer_nonfinite_batches": 0,
    }
    with pytest.raises(FileExistsError):
        train_ssdg._finalize_phase1_adv3b02_technical_smoke(
            out_dir=out_dir,
            args=args,
            counters=complete,
            source_evidence=_smoke_source_evidence(),
        )
    assert receipt_path.read_bytes() == sentinel


def test_adv3b02_smoke_flag_zero_does_not_restrict_formal_rows(tmp_path: Path) -> None:
    """Break caught: adding the probe changes ordinary F2--F6 validation."""

    from SSDG import train_ssdg

    formal_line = _dry_run_lines(tmp_path)[1]
    args = train_ssdg.build_arg_parser().parse_args(
        _trainer_args(_command_tokens(formal_line))
    )
    assert args.phase1_adv3b02_technical_smoke_batches == 0
    assert args.candidate_id == "F2_ADV3B02_CLIC"
    assert train_ssdg._validate_phase1_adv3b02_technical_smoke_args(args) == 0


@pytest.mark.parametrize("batch_count", (1, 2))
def test_adv3b02_smoke_fails_short_first_epoch_before_any_source_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_count: int,
) -> None:
    """Break caught: a short smoke loader enters source validation or a later epoch."""

    torch = pytest.importorskip("torch")
    from SSDG import train_ssdg

    args = train_ssdg.build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / "short-smoke"),
            "--from_scratch",
            "true",
            "--epochs",
            "2",
            "--label_epochs",
            "2",
            "--pseudo_epochs",
            "0",
            "--batch_size",
            "2",
            "--eval_batch_size",
            "2",
            "--device",
            "cpu",
            "--phase1_adv3b02_technical_smoke_batches",
            "3",
        ]
    )
    for field in (
        "lambda_u",
        "lambda_ent",
        "lambda_domain",
        "lambda_adv",
        "lambda_orth",
        "lambda_cons",
        "lambda_group_ce",
        "lambda_fishr",
        "lambda_sat_cls",
        "lambda_sat_cons",
    ):
        setattr(args, field, 0.0)
    args.use_unlabeled = False
    args.use_sat_consistency = False
    args.use_mixstyle = False
    args.use_aug = False
    args.amp = False

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.forward_count = 0

        def forward(self, x, **_kwargs):
            self.forward_count += 1
            logits = self.linear(x)
            return {"tx_logits": logits, "z_id": logits}

    model = TinyModel()
    batches = [
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
            {},
        )
        for _ in range(batch_count)
    ]
    source_roles = {
        "enabled": True,
        "held_tx_loaded_by_training": False,
        "source_known_train_tx": ["20-15", "20-19", "6-15", "8-20"],
        "source_known_validation_tx": ["14-7"],
        "source_proxy_unknown_tx": ["14-10"],
        "partition_sha256": "c" * 64,
    }
    data_ctx = {
        "train_loader": batches,
        "balanced_train_sampler": None,
        "unlabeled_loader": [],
        "val_loader": [object()],
        "named_test_loaders": {},
        "domain_label_map": {},
        "num_domains": 1,
        "input_len": 2,
        "num_classes": 2,
        "class_id_to_tx": ["20-15", "20-19"],
        "split_info": {
            "mode": "tx_rx_day_1_6_3",
            "labeled_size": 2 * batch_count,
            "unlabeled_size": 0,
            "source_val_size": 1,
            "source_split_receipt": {"split_manifest_sha256": "d" * 64},
            "tx_partition_receipt": source_roles,
        },
    }
    monkeypatch.setattr(
        train_ssdg,
        "_validate_phase1_adv3b02_technical_smoke_args",
        lambda _args: 3,
    )
    monkeypatch.setattr(train_ssdg, "resolve_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(train_ssdg, "_prepare_cuda_memory_audit", lambda _device: None)
    monkeypatch.setattr(train_ssdg, "set_seed", lambda _seed: None)
    monkeypatch.setattr(train_ssdg, "_build_ssdg_wisig_data", lambda *_args: data_ctx)
    monkeypatch.setattr(
        train_ssdg,
        "_build_manytx_real_oe_data",
        lambda *_args, **_kwargs: {"loader": None, "sampler": None, "receipt": {}},
    )
    monkeypatch.setattr(train_ssdg, "merge_checkpoint_args", lambda *_args, **_kwargs: args)
    monkeypatch.setattr(train_ssdg, "_apply_model_cli_args", lambda model_args, _args: model_args)
    monkeypatch.setattr(train_ssdg, "build_baseline_model", lambda *_args: model)
    monkeypatch.setattr(train_ssdg, "move_batch", lambda batch, _device: batch)

    def core_losses(out_l, y_l, *_args, **_kwargs):
        zero = out_l["tx_logits"].sum() * 0.0
        return {
            "loss_cls": torch.nn.functional.cross_entropy(out_l["tx_logits"], y_l),
            "loss_dom": zero,
            "loss_adv": zero,
            "loss_cons": zero,
            "loss_orth": zero,
            "loss_group_ce": zero,
            "cons_cos": zero,
            "dom_acc": zero,
        }

    monkeypatch.setattr(train_ssdg, "compute_core_losses", core_losses)
    monkeypatch.setattr(
        train_ssdg,
        "evaluate_loader",
        lambda *_args, **_kwargs: pytest.fail("short ADV smoke opened source validation"),
    )

    with pytest.raises(RuntimeError, match="first epoch.*observed"):
        train_ssdg.train(args)
    assert model.forward_count == batch_count
    assert not (Path(args.output_dir) / "phase1_adv3b02_technical_smoke_receipt.json").exists()


def test_dry_run_has_no_target_side_input_and_creates_no_roots(tmp_path: Path) -> None:
    """Break caught: source-only dry-run touches roots or emits target-side input."""

    lines = _dry_run_lines(tmp_path)
    # pytest's temporary directory embeds this test's own word ``target``;
    # inspect the emitted trainer interface rather than that unrelated path.
    emitted_interface = []
    for line in lines:
        values = _arg_map(_trainer_args(_command_tokens(line)))
        for flag, value in values.items():
            if flag in {"--wisig_pkl", "--output_dir", "--phase2_export_path"}:
                continue
            emitted_interface.extend((flag, value or ""))
    joined = " ".join(emitted_interface).lower()
    for forbidden in ("target", "package", "truth", "query", "scorer", "target-metrics"):
        assert forbidden not in joined
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "logs").exists()


def test_existing_root_fails_before_any_new_output_mutation(tmp_path: Path) -> None:
    """Break caught: a formal invocation overwrites a pre-existing run/log root."""

    _require_launcher()
    existing_run_root = tmp_path / "runs"
    existing_run_root.mkdir()
    marker = existing_run_root / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    result = subprocess.run(
        _launcher_argv(tmp_path),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing to overwrite run/log root" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "logs").exists()


def test_later_fold_collision_is_rejected_before_any_child_or_log_mutation(tmp_path: Path) -> None:
    """Break caught: F1--F5 can launch before a pre-existing later-fold collision is seen."""

    collision = tmp_path / "runs" / "F6_ADV3B02_CLIC"
    collision.mkdir(parents=True)
    marker = collision / "preserve.txt"
    marker.write_text("preserve", encoding="utf-8")

    result = _run_launcher(
        tmp_path,
        check=False,
        env_overrides={"PYTHON": "/bin/false"},
    )

    assert result.returncode != 0
    assert "refusing to overwrite planned fold output: F6_ADV3B02_CLIC" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "logs").exists()
    assert not any((tmp_path / "runs").glob("F[1-5]_ADV3B02_CLIC"))
    assert not list(tmp_path.rglob("*.pid"))


def test_contract_validator_rejects_profile_role_split_and_checkpoint_drift(tmp_path: Path) -> None:
    """Break caught: frozen ADV profile/roles/split/final policy can silently drift."""

    expected_contract = _run_launcher(tmp_path, "--print-contract").stdout
    contract_path = tmp_path / "contract.txt"
    contract_path.write_bytes(expected_contract.encode("utf-8"))
    accepted = _run_launcher(tmp_path, "--validate-contract-file", _wsl_path(contract_path), check=False)
    assert accepted.returncode == 0, accepted.stderr

    mutations = (
        ("profile.proxy_unknown_core_quantile=0.90", "profile.proxy_unknown_core_quantile=0.80"),
        ("fold.1.train=20-15,20-19,6-15,8-20", "fold.1.train=20-15,20-19,6-15,14-7"),
        ("split_mode=tx_rx_day_1_6_3", "split_mode=tx_rx_day_1_7_2"),
        ("checkpoint_selection=final_only", "checkpoint_selection=source_validation_only"),
    )
    for before, after in mutations:
        mutated = tmp_path / f"mutated-{len(before)}.txt"
        assert before in expected_contract
        mutated.write_bytes(expected_contract.replace(before, after, 1).encode("utf-8"))
        rejected = _run_launcher(
            tmp_path,
            "--validate-contract-file",
            _wsl_path(mutated),
            check=False,
        )
        assert rejected.returncode != 0
        assert "frozen contract mismatch" in rejected.stderr
