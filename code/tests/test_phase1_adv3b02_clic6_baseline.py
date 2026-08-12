from __future__ import annotations

"""Behavior contracts for the frozen six-fold ADV3B02 CLIC-equivalent entry.

Each assertion protects a launch-time behavior, not a source-text convention:
wrong profile values, TX roles, checkpoint policy, parser syntax, root reuse, or
accidental target-side input must make the real launcher fail this suite.
"""

import shlex
import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_adv3b02_clic6_v1_20260813.sh"

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
