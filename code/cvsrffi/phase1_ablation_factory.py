"""Frozen Phase1 T1 ablation configurations.

The factory owns every method-defining switch used by the six manuscript T1
arms.  Callers provide only run identity, paths, and paired seeds.  This keeps
the arm comparison fail-closed against launcher-default drift.
"""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from copy import deepcopy
from typing import Any, Mapping


PHASE1_ABLATION_SCHEMA = "cvs.phase1_ablation.config.v1"
PHASE1_ABLATION_IDS = (
    "P1-FULL",
    "P1-SUP",
    "P1-A0",
    "P1-B0",
    "P1-C0",
    "P1-D0",
)
PHASE1_LABEL_RHOS_BY_ID = {
    "P1-LABEL-RHO005": 0.005,
    "P1-LABEL-RHO010": 0.010,
    "P1-LABEL-RHO020": 0.020,
    "P1-LABEL-RHO050": 0.050,
}
PHASE1_LABEL_ABLATION_IDS = tuple(PHASE1_LABEL_RHOS_BY_ID)


class Phase1AblationConfigError(ValueError):
    """Raised when a Phase1 ablation identity or configuration is invalid."""


_FULL_CONFIG: dict[str, Any] = {
    # Frozen protocol and optimization schedule.
    "dataset": "wisig",
    "wisig_equalized": "1",
    "wisig_domain": "rx_day",
    "wisig_out_len": 256,
    "wisig_train_ratio": 0.2,
    "wisig_val_ratio": -1.0,
    "wisig_guard_gap": 8,
    "wisig_train_days": "0,1",
    "wisig_test_days": "2,3",
    "wisig_train_rxs": "0,1,2,3,4,5,6",
    "wisig_test_rxs": "7,8,9,10,11",
    "wisig_split_strategy": "random",
    "wisig_cap_strategy": "random",
    "wisig_max_day123_per_combo": 0,
    "wisig_max_train_per_combo": 0,
    "wisig_max_val_per_combo": 0,
    "wisig_max_test_per_combo": 0,
    "split_mode": "tx_rx_day_1_7_2",
    "labeled_ratio": 0.07,
    "unlabeled_ratio": 0.63,
    "source_val_ratio": 0.30,
    "epochs": 200,
    "label_epochs": 130,
    "pseudo_epochs": 70,
    "from_scratch": True,
    "lr": 0.0002,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "eval_batch_size": 256,
    "num_workers": 4,
    "prefetch_factor": 2,
    "eval_max_batches": 0,
    "checkpoint_selection": "source_validation_only",
    "phase1_source_val_selection_only": True,
    "best_metric": "source_val_sat_hmean",
    # Physical dual-representation backbone.
    "representation_mode": "dual",
    "model_size": "M",
    "model_variant": "lite_d",
    "id_feature_key": "feat_joint",
    "branch_ablation": "no_dac",
    "domain_branch_ablation": "none",
    "domain_enhancer": "rcn_stats",
    "domain_enhancer_strength": 0.35,
    # Supervised/domain/disentanglement objectives.
    "lambda_domain": 1.0,
    "lambda_adv": 0.35,
    "lambda_orth": 0.05,
    "lambda_cons": 0.08,
    "lambda_group_ce": 0.16,
    "lambda_fishr": 0.04,
    # Receiver-day pseudo-label loop.
    "use_unlabeled": True,
    "lambda_u": 0.16,
    "lambda_ent": 0.01,
    "pseudo_threshold_mode": "rx_day_quantile",
    "tau_min": 0.92,
    "tau_max": 0.97,
    "pseudo_quantile": 0.86,
    "pseudo_domain_gate": True,
    "use_ema_teacher": True,
    # Angular prototype/tail-risk geometry.
    "use_proto_memory": True,
    "lambda_proto": 0.0032,
    "proto_domain_align_weight": 0.10,
    "proto_margin": 0.15,
    "proto_push_weight": 0.10,
    "proto_min_count": 2,
    "lambda_open_world_feat": 0.0024,
    "ow_feat_start_epoch": 12,
    "ow_feat_warmup_epochs": 25,
    "ow_feat_tail_weight": 0.14,
    "ow_feat_vacuum_weight": 0.40,
    "lambda_zid_compact": 0.032,
    "zid_compact_start_epoch": 8,
    "zid_compact_warmup_epochs": 25,
    "lambda_proxy_unknown": 0.0045,
    "proxy_unknown_start_epoch": 45,
    "proxy_unknown_warmup_epochs": 25,
    "proxy_unknown_core_quantile": 0.90,
    "proxy_unknown_accept_quantile": 0.85,
    "proxy_unknown_vaccept_weight": 1.0,
    "proxy_unknown_core_accept_weight": 0.45,
    "proxy_unknown_vaccept_cvar_alpha": 0.30,
    "lambda_soft_unknown_mixup": 0.0045,
    "soft_unknown_mixup_start_epoch": 25,
    "soft_unknown_mixup_warmup_epochs": 25,
    # Identity-preserving source extrapolation.
    "use_mixstyle": True,
    "mixstyle_p": 0.18,
    "mixstyle_alpha": 0.10,
    "mixstyle_layers": "time_down,t1",
    "mixstyle_use_domain_label": True,
    "mixstyle_mix": "same_tx_crossdomain",
    "mixstyle_strength": 0.70,
    "mixstyle_fallback": "skip",
    "lambda_source_episode": 0.0035,
    "source_episode_start_epoch": 20,
    "source_episode_warmup_epochs": 25,
    "source_episode_radius_cap_deg": 33.0,
    "source_episode_mixup_weight": 0.75,
    "use_sat_consistency": True,
    "use_concat_sat_channel_aug": True,
    "concat_sat_ce_only": True,
    "sat_train_scenario": "leo_clear_weak",
    "sat_train_scenarios": "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    "sat_view_schedule": (
        "1@0.30:leo_clear_weak;"
        "41@0.60:leo_low_elev_weak,leo_rain_weak;"
        "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    ),
    "sat_cons_start_epoch": 80,
    "lambda_sat_cls": 0.68,
    "lambda_sat_cons": 0.0,
    # Immutable Phase2 bundle source.
    "phase2_export_prototypes": True,
    "phase2_export_feature_key": "z_id",
    "phase2_export_split": "train",
    "phase2_fuse_prototypes": True,
    "phase2_fuse_max_components": 6,
    "phase2_fuse_merge_angle_deg": 2.5,
    "phase2_fuse_radius_cap_deg": 15.0,
    "phase2_fuse_tail_abs_deg": 24.0,
    "phase2_fuse_accept_policy": "local_component",
    "phase2_fuse_accept_radius_key": "p95",
    "phase2_fuse_max_p95_increase_deg": 2.0,
    "phase2_fuse_keep_tail_sentinel": True,
    "phase2_fuse_global_ball_accept": False,
    # Source-validation telemetry; held-out target never selects a checkpoint.
    "eval_sat_channel": True,
    "eval_sat_scenarios": "leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    "test_eval_policy": "interval_final",
    "test_eval_start_epoch": 1,
    "test_eval_interval": 10,
    "test_eval_final_window": 20,
    "test_eval_final_interval": 2,
}

_RESOLVED_HASH_EXCLUDED_FIELDS = frozenset(
    {
        "ablation_config_hash",
        "ablation_enabled_objectives",
        "ablation_id",
        "ablation_method_config_hash",
        "ablation_schema",
        "baseline_ckpt",
        "candidate_id",
        "device",
        "dataset_receipt_path",
        "dataset_receipt_sha256",
        "dry_run",
        "expected_config_hash",
        "environment_receipt_path",
        "environment_receipt_sha256",
        "formal_ablation",
        "git_commit",
        "metrics_csv",
        "metrics_jsonl",
        "output_dir",
        "phase2_export_checkpoint",
        "phase2_export_path",
        "python_environment_id",
        "run_id",
        "row_key",
        "safe_best_path",
        "safe_latest_path",
        "sealed_plan_sha256",
        "seed",
        "seed_registry_sha256",
        "wisig_pkl",
    }
)


_ARM_OVERRIDES: dict[str, dict[str, Any]] = {
    "P1-FULL": {},
    "P1-SUP": {
        "label_epochs": 200,
        "pseudo_epochs": 0,
        "representation_mode": "dual",
        "lambda_domain": 0.0,
        "lambda_adv": 0.0,
        "lambda_orth": 0.0,
        "lambda_cons": 0.0,
        "lambda_group_ce": 0.0,
        "lambda_fishr": 0.0,
        "use_unlabeled": False,
        "lambda_u": 0.0,
        "lambda_ent": 0.0,
        "pseudo_domain_gate": False,
        "use_ema_teacher": False,
        "use_proto_memory": False,
        "lambda_proto": 0.0,
        "lambda_open_world_feat": 0.0,
        "lambda_zid_compact": 0.0,
        "lambda_proxy_unknown": 0.0,
        "lambda_soft_unknown_mixup": 0.0,
        "use_mixstyle": False,
        "lambda_source_episode": 0.0,
        "use_sat_consistency": False,
        "use_concat_sat_channel_aug": False,
        "concat_sat_ce_only": False,
        "lambda_sat_cls": 0.0,
    },
    "P1-A0": {
        "representation_mode": "single_parameter_matched",
        "lambda_domain": 0.0,
        "lambda_adv": 0.0,
        "lambda_orth": 0.0,
        "lambda_cons": 0.0,
    },
    "P1-B0": {
        "lambda_u": 0.0,
        "lambda_ent": 0.0,
    },
    "P1-C0": {
        "use_proto_memory": False,
        "lambda_proto": 0.0,
        "lambda_open_world_feat": 0.0,
        "lambda_zid_compact": 0.0,
        "lambda_proxy_unknown": 0.0,
        "lambda_soft_unknown_mixup": 0.0,
    },
    "P1-D0": {
        "use_mixstyle": False,
        "lambda_source_episode": 0.0,
        "use_sat_consistency": False,
        "use_concat_sat_channel_aug": False,
        "concat_sat_ce_only": False,
        "lambda_sat_cls": 0.0,
    },
    **{
        ablation_id: {
            "labeled_ratio": 0.70 * rho,
            "unlabeled_ratio": 0.70 * (1.0 - rho),
            "source_val_ratio": 0.30,
        }
        for ablation_id, rho in PHASE1_LABEL_RHOS_BY_ID.items()
    },
}


def phase1_ablation_config(ablation_id: str) -> dict[str, Any]:
    arm_id = str(ablation_id).strip().upper()
    if arm_id not in _ARM_OVERRIDES:
        raise Phase1AblationConfigError(f"unknown Phase1 ablation_id: {arm_id}")
    config = deepcopy(_FULL_CONFIG)
    config.update(deepcopy(_ARM_OVERRIDES[arm_id]))
    return config


def phase1_ablation_config_hash(
    ablation_id: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> str:
    payload = dict(config or phase1_ablation_config(ablation_id))
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def phase1_ablation_diff(ablation_id: str) -> dict[str, tuple[Any, Any]]:
    arm = phase1_ablation_config(ablation_id)
    full = phase1_ablation_config("P1-FULL")
    return {
        key: (full[key], arm[key])
        for key in full
        if full[key] != arm[key]
    }


def enabled_objectives(ablation_id: str) -> tuple[str, ...]:
    config = phase1_ablation_config(ablation_id)
    objectives = ["tx_cosface"]
    fields = (
        ("domain_supervision", "lambda_domain"),
        ("grl", "lambda_adv"),
        ("orthogonality", "lambda_orth"),
        ("center_consistency", "lambda_cons"),
        ("group_ce", "lambda_group_ce"),
        ("fishr", "lambda_fishr"),
        ("pseudo_ce", "lambda_u"),
        ("pseudo_entropy", "lambda_ent"),
        ("prototype", "lambda_proto"),
        ("open_world_geometry", "lambda_open_world_feat"),
        ("zid_geometry", "lambda_zid_compact"),
        ("coretail_proxy", "lambda_proxy_unknown"),
        ("soft_mix_boundary", "lambda_soft_unknown_mixup"),
        ("source_episode", "lambda_source_episode"),
        ("leo_stress_ce", "lambda_sat_cls"),
    )
    objectives.extend(
        name for name, field in fields if float(config[field]) > 0.0
    )
    return tuple(objectives)


def resolved_phase1_ablation_config(args: Namespace) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in sorted(vars(args).items())
        if key not in _RESOLVED_HASH_EXCLUDED_FIELDS
    }


def resolved_phase1_ablation_config_hash(args: Namespace) -> str:
    payload = resolved_phase1_ablation_config(args)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_phase1_ablation(args: Namespace) -> dict[str, Any]:
    arm_id = str(getattr(args, "ablation_id", "")).strip().upper()
    config = phase1_ablation_config(arm_id)
    git_commit = str(getattr(args, "git_commit", "")).strip().lower()
    if len(git_commit) != 40 or any(ch not in "0123456789abcdef" for ch in git_commit):
        raise Phase1AblationConfigError(
            "formal Phase1 ablation requires a full 40-character Git commit"
        )
    for key, value in config.items():
        if not hasattr(args, key):
            raise Phase1AblationConfigError(
                f"training parser lacks frozen Phase1 field: {key}"
            )
        setattr(args, key, deepcopy(value))
    method_config_hash = phase1_ablation_config_hash(arm_id, config=config)
    config_hash = resolved_phase1_ablation_config_hash(args)
    args.ablation_id = arm_id
    args.ablation_schema = PHASE1_ABLATION_SCHEMA
    args.ablation_method_config_hash = method_config_hash
    args.ablation_config_hash = config_hash
    args.ablation_enabled_objectives = list(enabled_objectives(arm_id))
    expected_config_hash = str(
        getattr(args, "expected_config_hash", "")
    ).strip().lower()
    if expected_config_hash and expected_config_hash != config_hash:
        raise Phase1AblationConfigError(
            "resolved Phase1 config hash differs from sealed plan"
        )
    return {
        "schema": PHASE1_ABLATION_SCHEMA,
        "ablation_id": arm_id,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "method_config_hash": method_config_hash,
        "enabled_objectives": list(enabled_objectives(arm_id)),
        "config": config,
        "resolved_config": resolved_phase1_ablation_config(args),
        "diff_from_p1_full": {
            key: {"p1_full": before, "arm": after}
            for key, (before, after) in phase1_ablation_diff(arm_id).items()
        },
    }


__all__ = [
    "PHASE1_ABLATION_IDS",
    "PHASE1_ABLATION_SCHEMA",
    "PHASE1_LABEL_ABLATION_IDS",
    "PHASE1_LABEL_RHOS_BY_ID",
    "Phase1AblationConfigError",
    "apply_phase1_ablation",
    "enabled_objectives",
    "phase1_ablation_config",
    "phase1_ablation_config_hash",
    "phase1_ablation_diff",
    "resolved_phase1_ablation_config",
    "resolved_phase1_ablation_config_hash",
]
