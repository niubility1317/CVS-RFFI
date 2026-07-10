from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_RUN_ID = "phase1_dgleo_p1verify128_20260710"
DEFAULT_ROOT = Path("/home/szu2070436088/2510044040/CV-SincNet")
DEFAULT_PYTHON = Path("/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python")
PAIRED_SEEDS = (710101, 710211, 710307, 710403)


BASE: Dict[str, Any] = {
    "lr": 0.000043,
    "u_dom": 0.18,
    "u_adv": 0.08,
    "u_sat": 0.30,
    "u_dm": 0.0050,
    "u_q": 0.0030,
    "u_start": 90,
    "u_min": 24,
    "l_rx_inv": 0.12,
    "l_day_inv": 0.08,
    "l_channel_inv": 0.12,
    "u_rx_inv": 0.06,
    "u_day_inv": 0.04,
    "u_channel_inv": 0.08,
    "channel_pair": 1.0,
    "dm_lambda": 0.0080,
    "dm_quantile": 1.10,
    "dm_source": 1.55,
    "dm_proxy": 1.45,
    "dm_bridge": 1.45,
    "dm_lowden": 1.10,
    "dm_tail": 1.55,
    "dm_overflow": 1.65,
    "dm_ratio": 1.45,
    "dm_satpair": 0.85,
    "dm_p95": 15.0,
    "dm_p99": 40.0,
    "dm_cvar": 26.0,
    "dm_domain_local": True,
    "dm_require_local": True,
    "dm_min_component": 2,
    "source_local_compact": 0.40,
    "source_local_invariant": 0.25,
    "source_local_inter": 0.20,
    "source_local_accept": 0.35,
    "source_local_density": 0.25,
    "source_w": 0.034,
    "proxy_w": 0.0050,
    "q_target": 0.14,
    "q_quantile": 0.76,
    "sat_cls": 0.88,
    "sat_cons": 0.074,
    "teacher_clean": 2.70,
    "teacher_sat": 1.50,
    "teacher_zid": 0.58,
    "domain_w": 1.40,
    "adv_w": 0.20,
    "os_min_budget": 0.15,
    "os_surgery": True,
    "os_min_closed_scale": 0.40,
}


CELLS: Sequence[Mapping[str, Any]] = (
    {"cell": "G0_FULL_WEAK", "group": "G0_JOINT", "role": "joint_weak", "overrides": {
        "l_rx_inv": 0.08, "l_day_inv": 0.05, "l_channel_inv": 0.08,
        "u_rx_inv": 0.04, "u_day_inv": 0.025, "u_channel_inv": 0.05,
        "dm_lambda": 0.0060, "source_w": 0.026, "proxy_w": 0.0038, "os_min_budget": 0.10,
    }},
    {"cell": "G0_FULL_BALANCED", "group": "G0_JOINT", "role": "joint_balanced", "overrides": {}},
    {"cell": "G0_FULL_GEOM_HIGH", "group": "G0_JOINT", "role": "joint_geometry_high", "overrides": {
        "dm_lambda": 0.0110, "dm_quantile": 1.55, "dm_source": 2.10, "dm_proxy": 2.00,
        "dm_bridge": 2.10, "dm_lowden": 1.70, "dm_tail": 2.20, "dm_overflow": 2.30,
        "dm_ratio": 2.00, "source_w": 0.052, "proxy_w": 0.0075, "os_min_budget": 0.22,
    }},
    {"cell": "G0_FULL_DG_PROTECT", "group": "G0_JOINT", "role": "joint_dg_protected", "overrides": {
        "teacher_clean": 3.00, "teacher_sat": 1.75, "sat_cls": 1.00, "sat_cons": 0.10,
        "os_min_closed_scale": 0.55, "dm_lambda": 0.0090, "source_w": 0.040, "proxy_w": 0.0060,
    }},

    {"cell": "G1_SAT_CE_CONTROL", "group": "G1_SATELLITE", "role": "sat_ce_control", "overrides": {
        "sat_cls": 1.10, "sat_cons": 0.0, "l_channel_inv": 0.0, "u_channel_inv": 0.0,
        "channel_pair": 0.0, "dm_satpair": 0.0,
    }},
    {"cell": "G1_SAT_PAIR_FOCUS", "group": "G1_SATELLITE", "role": "sat_pair_focus", "overrides": {
        "sat_cls": 0.72, "sat_cons": 0.025, "l_channel_inv": 0.16, "u_channel_inv": 0.10,
        "channel_pair": 2.0, "dm_satpair": 1.70,
    }},
    {"cell": "G1_SAT_INV_FOCUS", "group": "G1_SATELLITE", "role": "sat_invariance_focus", "overrides": {
        "sat_cls": 0.82, "sat_cons": 0.11, "l_channel_inv": 0.20, "u_channel_inv": 0.14,
        "channel_pair": 1.40, "dm_satpair": 1.10,
    }},
    {"cell": "G1_SAT_BALANCED", "group": "G1_SATELLITE", "role": "sat_balanced", "overrides": {
        "sat_cls": 1.00, "sat_cons": 0.10, "l_channel_inv": 0.15, "u_channel_inv": 0.10,
        "channel_pair": 1.20, "dm_satpair": 1.10,
    }},

    {"cell": "G2_LINV_OFF", "group": "G2_LABELED_INVARIANCE", "role": "labeled_invariance_off", "overrides": {
        "l_rx_inv": 0.0, "l_day_inv": 0.0, "l_channel_inv": 0.0, "channel_pair": 0.0,
    }},
    {"cell": "G2_LINV_RX", "group": "G2_LABELED_INVARIANCE", "role": "labeled_receiver_only", "overrides": {
        "l_rx_inv": 0.18, "l_day_inv": 0.0, "l_channel_inv": 0.0, "channel_pair": 0.0,
    }},
    {"cell": "G2_LINV_RXDAY", "group": "G2_LABELED_INVARIANCE", "role": "labeled_receiver_day", "overrides": {
        "l_rx_inv": 0.15, "l_day_inv": 0.12, "l_channel_inv": 0.0, "channel_pair": 0.0,
    }},
    {"cell": "G2_LINV_FULL", "group": "G2_LABELED_INVARIANCE", "role": "labeled_receiver_day_channel", "overrides": {
        "l_rx_inv": 0.15, "l_day_inv": 0.10, "l_channel_inv": 0.16, "channel_pair": 1.25,
    }},

    {"cell": "G3_UINV_OFF", "group": "G3_UNLABELED_INVARIANCE", "role": "unlabeled_invariance_off", "overrides": {
        "u_rx_inv": 0.0, "u_day_inv": 0.0, "u_channel_inv": 0.0,
    }},
    {"cell": "G3_UINV_RXDAY", "group": "G3_UNLABELED_INVARIANCE", "role": "unlabeled_receiver_day", "overrides": {
        "u_rx_inv": 0.09, "u_day_inv": 0.07, "u_channel_inv": 0.0,
    }},
    {"cell": "G3_UINV_CHANNEL", "group": "G3_UNLABELED_INVARIANCE", "role": "unlabeled_channel", "overrides": {
        "u_rx_inv": 0.0, "u_day_inv": 0.0, "u_channel_inv": 0.14, "channel_pair": 1.40,
    }},
    {"cell": "G3_UINV_FULL", "group": "G3_UNLABELED_INVARIANCE", "role": "unlabeled_receiver_day_channel", "overrides": {
        "u_rx_inv": 0.09, "u_day_inv": 0.07, "u_channel_inv": 0.13, "channel_pair": 1.30,
    }},

    {"cell": "G4_LOCAL_OFF", "group": "G4_LOCAL_COMPONENT", "role": "global_ball_diagnostic", "overrides": {
        "dm_domain_local": False, "dm_require_local": False,
        "source_local_compact": 0.0, "source_local_invariant": 0.0, "source_local_inter": 0.0,
        "source_local_accept": 0.0, "source_local_density": 0.0,
    }},
    {"cell": "G4_LOCAL_DM_ONLY", "group": "G4_LOCAL_COMPONENT", "role": "direct_metric_local_only", "overrides": {
        "source_local_compact": 0.0, "source_local_invariant": 0.0, "source_local_inter": 0.0,
        "source_local_accept": 0.0, "source_local_density": 0.0,
    }},
    {"cell": "G4_LOCAL_SOURCE_ONLY", "group": "G4_LOCAL_COMPONENT", "role": "source_episode_local_only", "overrides": {
        "dm_lambda": 0.0, "u_dm": 0.0, "source_w": 0.055,
        "source_local_compact": 0.55, "source_local_invariant": 0.35, "source_local_inter": 0.30,
        "source_local_accept": 0.50, "source_local_density": 0.40,
    }},
    {"cell": "G4_LOCAL_FULL_STRICT", "group": "G4_LOCAL_COMPONENT", "role": "strict_full_local", "overrides": {
        "dm_min_component": 4, "source_w": 0.055,
        "source_local_compact": 0.55, "source_local_invariant": 0.35, "source_local_inter": 0.30,
        "source_local_accept": 0.50, "source_local_density": 0.40,
    }},

    {"cell": "G5_U_OFF", "group": "G5_U_TRISTATE", "role": "unlabeled_branch_off", "overrides": {
        "u_dom": 0.0, "u_adv": 0.0, "u_sat": 0.0, "u_dm": 0.0, "u_q": 0.0,
        "u_rx_inv": 0.0, "u_day_inv": 0.0, "u_channel_inv": 0.0,
    }},
    {"cell": "G5_U_DOMAIN_SAT", "group": "G5_U_TRISTATE", "role": "unlabeled_domain_sat_only", "overrides": {
        "u_dom": 0.24, "u_adv": 0.12, "u_sat": 0.44, "u_dm": 0.0, "u_q": 0.0,
    }},
    {"cell": "G5_U_QUAR_ONLY", "group": "G5_U_TRISTATE", "role": "unlabeled_quarantine_only", "overrides": {
        "u_dm": 0.0, "u_q": 0.0070, "u_start": 65, "u_min": 16,
    }},
    {"cell": "G5_U_FULL", "group": "G5_U_TRISTATE", "role": "unlabeled_direct_quarantine_full", "overrides": {
        "u_dom": 0.24, "u_adv": 0.12, "u_sat": 0.44, "u_dm": 0.0080, "u_q": 0.0065,
        "u_start": 65, "u_min": 16, "u_rx_inv": 0.09, "u_day_inv": 0.07, "u_channel_inv": 0.13,
    }},

    {"cell": "G6_DM_PROXY", "group": "G6_DIRECT_OPEN_SET", "role": "proxy_vaccept_focus", "overrides": {
        "dm_proxy": 2.70, "dm_source": 1.40, "dm_bridge": 1.30, "dm_lowden": 1.10,
        "proxy_w": 0.0090,
    }},
    {"cell": "G6_DM_BRIDGE", "group": "G6_DIRECT_OPEN_SET", "role": "bridge_low_density_focus", "overrides": {
        "dm_bridge": 2.80, "dm_lowden": 2.50, "dm_proxy": 1.55, "dm_source": 1.60,
    }},
    {"cell": "G6_DM_TAIL", "group": "G6_DIRECT_OPEN_SET", "role": "tail_overflow_focus", "overrides": {
        "dm_quantile": 1.80, "dm_tail": 2.70, "dm_overflow": 2.85, "dm_ratio": 2.20,
        "dm_p95": 14.0, "dm_p99": 38.0, "dm_cvar": 24.0,
    }},
    {"cell": "G6_DM_BALANCED", "group": "G6_DIRECT_OPEN_SET", "role": "direct_metric_balanced", "overrides": {
        "dm_quantile": 1.50, "dm_source": 2.00, "dm_proxy": 2.00, "dm_bridge": 2.00,
        "dm_lowden": 1.80, "dm_tail": 2.10, "dm_overflow": 2.20, "dm_ratio": 2.00,
    }},

    {"cell": "G7_KD_HEAVY", "group": "G7_GRADIENT_CONFLICT", "role": "closed_kd_dominant", "overrides": {
        "teacher_clean": 3.30, "teacher_sat": 1.95, "teacher_zid": 0.72,
        "dm_lambda": 0.0050, "source_w": 0.022, "proxy_w": 0.0030,
        "os_min_budget": 0.08, "os_surgery": False, "os_min_closed_scale": 0.65,
    }},
    {"cell": "G7_GRAD_BALANCED", "group": "G7_GRADIENT_CONFLICT", "role": "gradient_balanced", "overrides": {
        "teacher_clean": 2.40, "teacher_sat": 1.30, "dm_lambda": 0.0090,
        "source_w": 0.042, "proxy_w": 0.0060, "os_min_budget": 0.18,
    }},
    {"cell": "G7_OS_HIGH", "group": "G7_GRADIENT_CONFLICT", "role": "open_set_high", "overrides": {
        "teacher_clean": 2.10, "teacher_sat": 1.15, "dm_lambda": 0.0130,
        "source_w": 0.065, "proxy_w": 0.0090, "os_min_budget": 0.27,
        "os_min_closed_scale": 0.32,
    }},
    {"cell": "G7_OS_HIGH_PROTECT", "group": "G7_GRADIENT_CONFLICT", "role": "open_set_high_dg_protected", "overrides": {
        "teacher_clean": 2.80, "teacher_sat": 1.60, "sat_cls": 1.00, "sat_cons": 0.10,
        "dm_lambda": 0.0120, "source_w": 0.060, "proxy_w": 0.0085,
        "os_min_budget": 0.24, "os_min_closed_scale": 0.55, "os_surgery": True,
    }},
)


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_source_wisig_pkl(path: Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    forbidden = ("manytx.pkl", "manyrx.pkl", "singleday.pkl", "target", "unknown")
    if any(token in normalized for token in forbidden) or not normalized.endswith("/manysig.pkl"):
        raise ValueError(f"Phase1 P1 verification requires source-only ManySig.pkl, got: {path}")


def build_matrix() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cell_index, cell in enumerate(CELLS):
        for replicate, seed in enumerate(PAIRED_SEEDS, start=1):
            config = deepcopy(BASE)
            config.update(dict(cell.get("overrides", {})))
            gpu = (cell_index + replicate - 1) % 8
            rows.append(
                {
                    "candidate_id": f"P1V128_{cell['cell']}_S{replicate}",
                    "cell": str(cell["cell"]),
                    "group": str(cell["group"]),
                    "role": str(cell["role"]),
                    "replicate": replicate,
                    "seed": int(seed),
                    "gpu": gpu,
                    "source_only": True,
                    "phase1_proxy_only": True,
                    "checkpoint_selection": "final_only",
                    "sat_train_family": "simplified_leo_residual_weak_v1",
                    "sat_eval_family": "legacy_satellite_physics_holdout_v1",
                    "config": config,
                }
            )
    validate_matrix(rows)
    return rows


def validate_matrix(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 128:
        raise ValueError(f"expected 128 candidates, got {len(rows)}")
    ids = [str(row["candidate_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs are not unique")
    cells = Counter(str(row["cell"]) for row in rows)
    if len(cells) != 32 or set(cells.values()) != {4}:
        raise ValueError(f"expected 32 cells x 4 seeds, got {dict(cells)}")
    gpu_counts = Counter(int(row["gpu"]) for row in rows)
    if gpu_counts != Counter({gpu: 16 for gpu in range(8)}):
        raise ValueError(f"GPU allocation is not balanced: {dict(gpu_counts)}")
    for row in rows:
        if row.get("checkpoint_selection") != "final_only":
            raise ValueError(f"non-final checkpoint policy: {row['candidate_id']}")
        if not bool(row.get("source_only")) or not bool(row.get("phase1_proxy_only")):
            raise ValueError(f"non-source Phase1 row: {row['candidate_id']}")
        if row.get("sat_train_family") == row.get("sat_eval_family"):
            raise ValueError(f"satellite train/eval family overlap: {row['candidate_id']}")


def _append(args: List[str], name: str, value: Any) -> None:
    args.extend([name, str(value)])


def build_command(
    row: Mapping[str, Any],
    *,
    root: Path,
    python: Path,
    run_id: str,
    wisig_pkl: Path,
    teacher_ckpt: Path,
) -> List[str]:
    c = dict(row["config"])
    proxy_vaccept_weight = 0.0 if float(c["proxy_w"]) <= 0.0 else 0.035 + float(c["proxy_w"]) * 8.5
    out_dir = root / "runs" / run_id / str(row["candidate_id"])
    cmd = [str(python), "-u", str(root / "code" / "SSDG" / "train_ssdg.py")]
    fixed: Sequence[tuple[str, Any]] = (
        ("--wisig_pkl", wisig_pkl), ("--split_mode", "tx_rx_day_1_7_2"),
        ("--labeled_ratio", 0.08), ("--unlabeled_ratio", 0.72), ("--source_val_ratio", 0.20),
        ("--baseline_ckpt", teacher_ckpt), ("--from_scratch", "false"), ("--teacher_ckpt", teacher_ckpt),
        ("--output_dir", out_dir), ("--run_id", run_id), ("--candidate_id", row["candidate_id"]),
        ("--base_candidate", "ADV3B02_CORE90_SOFT_E200_P1_FINAL_ONLY"),
        ("--epochs", 200), ("--label_epochs", 145), ("--pseudo_epochs", 55),
        ("--lr", c["lr"]), ("--weight_decay", 0.00008), ("--batch_size", 112), ("--eval_batch_size", 256),
        ("--best_metric", "source_val_sat_hmean"), ("--phase1_source_val_selection_only", "true"),
        ("--enable_joint_safe_guard", "false"), ("--test_eval_policy", "interval_final"),
        ("--test_eval_start_epoch", 999999), ("--test_eval_interval", 0),
        ("--test_eval_final_window", 0), ("--test_eval_final_interval", 0),
        ("--phase1_v2_hard_gates", "true"), ("--endpoint_accept_policy_id", "endpoint_accept_v1"),
        ("--endpoint_threshold_source", "source_val_only"), ("--endpoint_calibration_split", "source_val"),
        ("--loss_gate_exported", "false"), ("--tail_safety_state_machine", "true"),
        ("--tail_stop_blocks_final", "true"), ("--tail_safety_warning_patience", 2),
        ("--tail_safety_rollback_patience", 1), ("--tail_safety_max_rollbacks", 1),
        ("--tail_safety_p95_target_deg", 16), ("--tail_safety_p99_target_deg", 42),
        ("--tail_safety_cvar_target_deg", 28), ("--tail_safety_proxy_vaccept_target", 0.50),
        ("--tail_safety_p99_expansion_block_final_delta", 2.0),
        ("--tail_safety_p99_expansion_block_best_delta", 3.5),
        ("--tail_safety_cvar_expansion_block_final_delta", 4.0),
        ("--tail_safety_cvar_expansion_block_best_delta", 6.0),
        ("--tail_safety_reference_window", 5), ("--tail_rollback_enabled", "false"),
        ("--os_eff_min_budget", c["os_min_budget"]), ("--os_budget_controller", "true"),
        ("--os_budget_max_scale", 4.0), ("--os_budget_min_closed_scale", c["os_min_closed_scale"]),
        ("--os_gradient_surgery", _bool(c["os_surgery"])), ("--os_gradient_surgery_interval", 1),
        ("--phase1_v2_os_eff_all_phases", "true"), ("--phase1_v2_guard_blocks_final", "true"),
        ("--u_tri_state_required", "true"), ("--u_direct_idle_blocks_promotion", "true"),
        ("--u_tri_min_core_rate", 0.05), ("--u_tri_max_core_rate", 0.95),
        ("--u_tri_min_ambiguous_rate", 0.01), ("--u_tri_max_outside_rate", 0.80),
        ("--u_tri_min_class_coverage", 2), ("--u_tri_min_domain_coverage", 2),
        ("--u_tri_max_pair_disagreement_rate", 0.25), ("--u_tri_min_pseudo_component_agreement", 0.80),
        ("--source_episode_density_gate", "true"), ("--source_episode_overflow_warn", 0.90),
        ("--source_episode_min_local_components", 4),
        ("--source_episode_local_compact_weight", c["source_local_compact"]),
        ("--source_episode_local_invariant_weight", c["source_local_invariant"]),
        ("--source_episode_local_inter_weight", c["source_local_inter"]),
        ("--source_episode_local_inter_margin_deg", 35),
        ("--source_episode_local_accept_weight", c["source_local_accept"]),
        ("--source_episode_local_density_weight", c["source_local_density"]),
        ("--direct_metric_multiview_separate", "true"),
        ("--direct_metric_domain_local_components", _bool(c["dm_domain_local"])),
        ("--direct_metric_require_domain_local_components", _bool(c["dm_require_local"])),
        ("--direct_metric_min_samples_per_component", c["dm_min_component"]),
        ("--direct_metric_clean_weight", 1.0), ("--direct_metric_sat_weight", 1.0),
        ("--u_geometry_all_valid_queries", "true"), ("--u_unlabeled_shuffle", "true"),
        ("--u_quarantine_core_accept_target", 0.82),
        ("--endpoint_require_artifact_on_export", "true"),
        ("--endpoint_calibration_min_component_samples", 4),
        ("--endpoint_calibration_min_class_samples", 4),
        ("--endpoint_calibration_core_quantile", 0.80),
        ("--endpoint_calibration_accept_quantile", 0.95),
        ("--endpoint_calibration_tail_quantile", 0.99),
        ("--checkpoint_selection", "final_only"), ("--sat_protocol_disjoint_required", "true"),
        ("--zid_leakage_probe_required", "true"), ("--zid_leakage_probe_max_batches", 24),
        ("--zid_leakage_probe_ridge", 0.01), ("--zid_receiver_probe_max_excess", 0.20),
        ("--zid_day_probe_max_excess", 0.15), ("--zid_channel_probe_max_excess", 0.15),
        ("--feasibility_gate", "true"), ("--feasibility_stage", "audit"),
        ("--feasibility_relaxed_pass", "false"), ("--feasibility_local_pass", "false"),
        ("--teacher_distill_start_epoch", 1), ("--teacher_distill_warmup_epochs", 30),
        ("--teacher_distill_temperature", 2.5),
        ("--lambda_teacher_clean_kl", c["teacher_clean"]),
        ("--lambda_teacher_sat_kl", c["teacher_sat"]),
        ("--lambda_teacher_zid_mse", c["teacher_zid"]),
        ("--lambda_domain", c["domain_w"]), ("--lambda_adv", c["adv_w"]),
        ("--lambda_orth", 0.060), ("--lambda_cons", 0.100), ("--lambda_group_ce", 0.250),
        ("--lambda_fishr", 0.055),
        ("--lambda_zid_receiver_invariance", c["l_rx_inv"]),
        ("--lambda_zid_day_invariance", c["l_day_inv"]),
        ("--lambda_zid_channel_invariance", c["l_channel_inv"]),
        ("--lambda_u_zid_receiver_invariance", c["u_rx_inv"]),
        ("--lambda_u_zid_day_invariance", c["u_day_inv"]),
        ("--lambda_u_zid_channel_invariance", c["u_channel_inv"]),
        ("--zid_invariance_min_groups", 2), ("--zid_invariance_min_samples_per_group", 2),
        ("--zid_channel_pair_weight", c["channel_pair"]),
        ("--lambda_open_world_feat", 0.0065), ("--ow_feat_start_epoch", 1),
        ("--ow_feat_warmup_epochs", 40), ("--ow_feat_radius_deg", 19),
        ("--ow_feat_inter_margin_deg", 60), ("--ow_feat_sample_margin_deg", 5),
        ("--ow_feat_tail_mode", "robust_3sigma"), ("--ow_feat_tail_weight", 0.10),
        ("--ow_feat_cvar_alpha", 0.95), ("--ow_feat_vacuum_weight", 0.024),
        ("--ow_feat_vacuum_width_deg", 4),
        ("--lambda_zid_compact", 0.070), ("--zid_compact_start_epoch", 1),
        ("--zid_compact_warmup_epochs", 40), ("--zid_compact_radius_deg", 34),
        ("--zid_compact_cvar_alpha", 0.88), ("--zid_compact_supcon_weight", 0.30),
        ("--zid_compact_radius_weight", 0.34), ("--zid_compact_cvar_weight", 0.36),
        ("--lambda_source_episode", c["source_w"]), ("--source_episode_start_epoch", 18),
        ("--source_episode_warmup_epochs", 45), ("--source_episode_radius_cap_deg", 18),
        ("--source_episode_radius_mode", "min_three_sigma_core"),
        ("--source_episode_core_quantile", 0.72), ("--source_episode_min_sigma_deg", 2),
        ("--source_episode_mixup_weight", 0.020), ("--source_episode_mixup_hard_k", 2),
        ("--lambda_proxy_unknown", c["proxy_w"]), ("--proxy_unknown_start_epoch", 56),
        ("--proxy_unknown_warmup_epochs", 45), ("--proxy_unknown_holdout_tx_per_batch", 1),
        ("--proxy_unknown_virtual_count", 80), ("--proxy_unknown_virtual_mode", "hard"),
        ("--proxy_unknown_energy_margin", 0.35), ("--proxy_unknown_energy_temperature", 1.0),
        ("--proxy_unknown_placeholder_weight", 0.05), ("--proxy_unknown_virtual_detach", "true"),
        ("--proxy_unknown_vacuum_weight", 0.034), ("--proxy_unknown_vacuum_width_deg", 5),
        ("--proxy_unknown_vacuum_hard_k", 2), ("--proxy_unknown_vacuum_radius_deg", 32),
        ("--proxy_unknown_core_quantile", 0.70), ("--proxy_unknown_accept_quantile", 0.80),
        ("--proxy_unknown_tail_quantile", 0.90), ("--proxy_unknown_overflow_quantile", 0.97),
        ("--proxy_unknown_component_radius_mode", "core_quantile"),
        ("--proxy_unknown_component_radius_quantile", 0.70),
        ("--proxy_unknown_vaccept_weight", round(proxy_vaccept_weight, 6)),
        ("--proxy_unknown_core_accept_weight", 0.040),
        ("--proxy_unknown_component_gate_weight", 0.070),
        ("--proxy_unknown_tail_quarantine_weight", 0.180),
        ("--proxy_unknown_source_safe_weight", 0.360),
        ("--proxy_unknown_bridge_accept_weight", 0.140),
        ("--proxy_unknown_shell_outward_accept_weight", 0.100),
        ("--proxy_unknown_low_density_accept_weight", 0.110),
        ("--proxy_unknown_energy_margin_quantile_weight", 0.090),
        ("--proxy_unknown_radius_budget_weight", 0.120),
        ("--proxy_unknown_radius_inter_ratio_weight", 0.145),
        ("--proxy_unknown_vaccept_cvar_alpha", 0.16),
        ("--proxy_unknown_unknown_margin", 0.10), ("--proxy_unknown_known_margin", 0.04),
        ("--proxy_unknown_energy_softplus_temperature", 0.04),
        ("--proxy_unknown_accept_softplus_temperature", 0.035),
        ("--proxy_unknown_bridge_accept_target", 0.14),
        ("--proxy_unknown_shell_outward_accept_target", 0.18),
        ("--proxy_unknown_tail_accept_target", 0.28),
        ("--proxy_unknown_overflow_accept_target", 0.14),
        ("--proxy_unknown_energy_margin_q", 0.08), ("--proxy_unknown_energy_margin_target", 0.10),
        ("--proxy_unknown_radius_budget_deg", 14), ("--proxy_unknown_radius_max_budget_deg", 22),
        ("--proxy_unknown_radius_inter_ratio_target", 0.74),
        ("--proxy_unknown_density_temperature_deg", 3),
        ("--proxy_unknown_component_temperature_deg", 3),
        ("--proxy_unknown_component_margin_deg", 4),
        ("--proxy_unknown_component_margin_temperature_deg", 3),
        ("--proxy_unknown_shell_width_deg", 4),
        ("--lambda_direct_metric_accept", c["dm_lambda"]), ("--direct_metric_start_epoch", 28),
        ("--direct_metric_warmup_epochs", 32), ("--direct_metric_virtual_count", 88),
        ("--direct_metric_virtual_mode", "hard"), ("--direct_metric_virtual_detach", "false"),
        ("--direct_metric_core_quantile", 0.70), ("--direct_metric_accept_quantile", 0.80),
        ("--direct_metric_tail_quantile", 0.90), ("--direct_metric_overflow_quantile", 0.97),
        ("--direct_metric_zid_p50_target_deg", 8), ("--direct_metric_zid_p95_target_deg", c["dm_p95"]),
        ("--direct_metric_zid_p99_target_deg", c["dm_p99"]),
        ("--direct_metric_zid_tail_cvar_target_deg", c["dm_cvar"]),
        ("--direct_metric_source_overflow_target", 0.40),
        ("--direct_metric_proxy_vaccept_target", 0.28),
        ("--direct_metric_bridge_accept_target", 0.18),
        ("--direct_metric_low_density_accept_target", 0.08),
        ("--direct_metric_tail_accept_target", 0.28),
        ("--direct_metric_overflow_accept_target", 0.14),
        ("--direct_metric_radius_inter_ratio_target", 0.76),
        ("--direct_metric_core_accept_target", 0.82), ("--direct_metric_sat_pair_target_deg", 9),
        ("--direct_metric_zid_quantile_weight", c["dm_quantile"]),
        ("--direct_metric_source_overflow_weight", c["dm_source"]),
        ("--direct_metric_proxy_vaccept_weight", c["dm_proxy"]),
        ("--direct_metric_bridge_accept_weight", c["dm_bridge"]),
        ("--direct_metric_low_density_accept_weight", c["dm_lowden"]),
        ("--direct_metric_tail_accept_weight", c["dm_tail"]),
        ("--direct_metric_overflow_accept_weight", c["dm_overflow"]),
        ("--direct_metric_radius_inter_ratio_weight", c["dm_ratio"]),
        ("--direct_metric_core_accept_weight", 0.34),
        ("--direct_metric_sat_pair_weight", c["dm_satpair"]),
        ("--direct_metric_quantile_temperature_deg", 3), ("--direct_metric_accept_temperature", 0.04),
        ("--direct_metric_component_temperature_deg", 3),
        ("--direct_metric_density_temperature_deg", 3),
        ("--direct_metric_component_margin_deg", 4), ("--direct_metric_source_margin_deg", 2),
        ("--direct_metric_shell_width_deg", 4), ("--direct_metric_accept_cvar_alpha", 0.16),
        ("--lambda_u", 0.12), ("--lambda_ent", 0.008),
        ("--lambda_u_domain", c["u_dom"]), ("--lambda_u_adv", c["u_adv"]),
        ("--lambda_u_sat_cons", c["u_sat"]), ("--lambda_u_direct_metric_accept", c["u_dm"]),
        ("--lambda_u_quarantine_accept", c["u_q"]), ("--u_domain_start_epoch", 1),
        ("--u_sat_cons_start_epoch", 1), ("--u_direct_metric_start_epoch", c["u_start"]),
        ("--u_direct_metric_min_selected", c["u_min"]), ("--u_direct_metric_use_sat_pair", "true"),
        ("--u_direct_metric_valid_domain_only", "true"), ("--u_quarantine_start_epoch", c["u_start"]),
        ("--u_quarantine_valid_domain_only", "true"), ("--u_quarantine_include_sat_view", "true"),
        ("--u_quarantine_min_count", 4), ("--u_quarantine_core_quantile", 0.70),
        ("--u_quarantine_accept_quantile", c["q_quantile"]),
        ("--u_quarantine_accept_target", c["q_target"]), ("--u_quarantine_cvar_alpha", 0.20),
        ("--u_quarantine_accept_temperature", 0.04), ("--u_sat_zid_cons_weight", 0.30),
        ("--tau_min", 0.93), ("--tau_max", 0.985), ("--pseudo_quantile", 0.88),
        ("--use_ema_teacher", "true"), ("--ema_decay", 0.999),
        ("--concat_sat_ce_weight", 1.0),
        ("--concat_sat_start_epoch", 1), ("--sat_view_prob", 1.0),
        ("--sat_view_seed", int(row["seed"]) + 5000), ("--sat_train_scenario", "leo_clear_weak"),
        ("--sat_train_scenarios", "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"),
        ("--sat_view_schedule", "1@0.45:leo_clear_weak;31@0.72:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;91@0.90:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"),
        ("--sat_cons_start_epoch", 12), ("--lambda_sat_cls", c["sat_cls"]),
        ("--lambda_sat_cons", c["sat_cons"]), ("--phase2_export_prototypes", "true"),
        ("--phase2_export_path", out_dir / "phase1_source_zid_prototypes.pt"),
        ("--phase2_export_feature_key", "z_id"), ("--phase2_export_split", "train"),
        ("--phase2_fuse_prototypes", "true"), ("--phase2_fuse_max_components", 6),
        ("--phase2_fuse_merge_angle_deg", 1.7), ("--phase2_fuse_radius_cap_deg", 12),
        ("--phase2_fuse_tail_abs_deg", 15), ("--phase2_fuse_accept_policy", "local_component"),
        ("--phase2_fuse_accept_radius_key", "p95"), ("--phase2_fuse_max_p95_increase_deg", 0.6),
        ("--phase2_fuse_keep_tail_sentinel", "true"), ("--phase2_fuse_tail_auto_accept", "false"),
        ("--phase2_fuse_global_ball_accept", "false"), ("--eval_sat_channel", "true"),
        ("--eval_sat_scenarios", "clear_leo,low_elev_leo,rain_leo,storm_mp,geo_clear,mixed_orbit"),
        ("--sat_eval_max_batches", -1), ("--device", "cuda:0"), ("--seed", row["seed"]),
    )
    for name, value in fixed:
        _append(cmd, name, value)
    cmd.extend(["--use_sat_consistency", "--use_concat_sat_channel_aug", "--no_concat_sat_ce_only"])
    return cmd


def matrix_payload(rows: Sequence[Mapping[str, Any]], run_id: str, max_active: int) -> Dict[str, Any]:
    return {
        "schema": "phase1_dgleo_p1verify128_matrix_v1",
        "run_id": run_id,
        "candidate_count": len(rows),
        "cell_count": len({row["cell"] for row in rows}),
        "paired_seeds": list(PAIRED_SEEDS),
        "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())),
        "max_active_per_gpu": int(max_active),
        "backfill_policy": "same_gpu_launch_next_after_any_terminal_exit",
        "claim_boundary": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_SUCCESS_CLAIM",
        "candidates": list(rows),
    }


def write_matrix(path: Path, rows: Sequence[Mapping[str, Any]], run_id: str, max_active: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix_payload(rows, run_id, max_active), indent=2, sort_keys=True), encoding="utf-8")


def _pmon_pids() -> Dict[int, set[int]]:
    proc = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    result: Dict[int, set[int]] = defaultdict(set)
    for raw in proc.stdout.splitlines():
        parts = raw.split()
        if len(parts) < 4 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        if parts[2].startswith("C"):
            result[int(parts[0])].add(int(parts[1]))
    return result


def _terminal_status(out_dir: Path, returncode: int) -> str:
    path = out_dir / "phase1_terminal_status.json"
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("status")
            if value:
                return str(value)
        except Exception:
            pass
    return "PROCESS_COMPLETE" if returncode == 0 else "PROCESS_FAILED_NO_TERMINAL"


def _write_event(writer: csv.writer, handle, event: str, row: Mapping[str, Any], **extra: Any) -> None:
    writer.writerow(
        [
            time.strftime("%Y-%m-%dT%H:%M:%S%z"), event, row["candidate_id"], row["gpu"],
            row["seed"], extra.get("pid", ""), extra.get("returncode", ""), extra.get("status", ""),
            extra.get("log", ""),
        ]
    )
    handle.flush()


def run_scheduler(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> int:
    root = Path(args.root)
    python = Path(args.python)
    wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
    validate_source_wisig_pkl(wisig)
    teacher = Path(
        args.teacher_ckpt
        or root / "runs" / "phase1_adv3_mechanism32_queue_20260701" / "ADV3B02_CORE90_SOFT_E200" / "best_joint_safe_ssdg.pth"
    )
    for required in (python, wisig, teacher, root / "code" / "SSDG" / "train_ssdg.py"):
        if not required.is_file():
            raise FileNotFoundError(required)
    log_root = root / "logs" / args.run_id
    run_root = root / "runs" / args.run_id
    log_root.mkdir(parents=True, exist_ok=False)
    run_root.mkdir(parents=True, exist_ok=False)
    write_matrix(log_root / "candidate_matrix.json", rows, args.run_id, args.max_active_per_gpu)
    queues: Dict[int, deque] = {gpu: deque() for gpu in range(8)}
    for row in rows:
        queues[int(row["gpu"])].append(row)
    active: Dict[int, Dict[str, Any]] = {}
    events_path = log_root / "scheduler_events.tsv"
    with events_path.open("w", encoding="utf-8", newline="") as events:
        writer = csv.writer(events, delimiter="\t")
        writer.writerow(["timestamp", "event", "candidate_id", "gpu", "seed", "pid", "returncode", "status", "log"])
        events.flush()
        while any(queues[gpu] for gpu in queues) or active:
            for pid, state in list(active.items()):
                code = state["process"].poll()
                if code is None:
                    continue
                state["log_handle"].close()
                status = _terminal_status(state["out_dir"], int(code))
                _write_event(
                    writer, events, "TERMINAL", state["row"], pid=pid, returncode=code,
                    status=status, log=state["log_path"],
                )
                del active[pid]
            try:
                observed = _pmon_pids()
            except Exception as exc:
                print(f"[P1V128-PMON-RETRY] error={type(exc).__name__}:{exc}", flush=True)
                time.sleep(float(args.poll_seconds))
                continue
            own_pids = set(active)
            for gpu in range(8):
                own_on_gpu = sum(1 for state in active.values() if int(state["row"]["gpu"]) == gpu)
                external = len(observed.get(gpu, set()) - own_pids)
                while queues[gpu] and own_on_gpu + external < int(args.max_active_per_gpu):
                    row = queues[gpu].popleft()
                    candidate_id = str(row["candidate_id"])
                    out_dir = run_root / candidate_id
                    log_path = log_root / f"{candidate_id}.out"
                    if out_dir.exists() or log_path.exists():
                        raise FileExistsError(f"stale candidate artifact: {candidate_id}")
                    out_dir.mkdir(parents=True)
                    command = build_command(
                        row, root=root, python=python, run_id=args.run_id,
                        wisig_pkl=wisig, teacher_ckpt=teacher,
                    )
                    log_handle = log_path.open("w", encoding="utf-8")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    env["PYTHONPATH"] = f"{root / 'code'}:{root}:{env.get('PYTHONPATH', '')}"
                    process = subprocess.Popen(
                        command,
                        cwd=str(root),
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[process.pid] = {
                        "process": process, "row": row, "log_handle": log_handle,
                        "log_path": log_path, "out_dir": out_dir,
                    }
                    _write_event(
                        writer, events, "LAUNCHED", row, pid=process.pid,
                        status="RUNNING", log=log_path,
                    )
                    print(
                        f"[P1V128-LAUNCHED] candidate={candidate_id} gpu={gpu} pid={process.pid} "
                        f"active_on_gpu={own_on_gpu + 1}/{args.max_active_per_gpu}",
                        flush=True,
                    )
                    own_on_gpu += 1
                    time.sleep(float(args.launch_settle_seconds))
            if active:
                time.sleep(float(args.poll_seconds))
        print(f"[P1V128-SCHEDULER-COMPLETE] run_id={args.run_id} candidates={len(rows)}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the 128-row Phase1 P1 validation matrix with same-GPU backfill.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--wisig-pkl", default="")
    parser.add_argument("--teacher-ckpt", default="")
    parser.add_argument("--max-active-per-gpu", type=int, default=2, choices=[2])
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--launch-settle-seconds", type=float, default=4.0)
    parser.add_argument("--emit-matrix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_matrix()
    if args.emit_matrix:
        write_matrix(Path(args.emit_matrix), rows, args.run_id, args.max_active_per_gpu)
    if args.dry_run:
        root = Path(args.root)
        python = Path(args.python)
        wisig = Path(args.wisig_pkl or root / "Dataset_WigSig" / "ManySig.pkl")
        validate_source_wisig_pkl(wisig)
        teacher = Path(
            args.teacher_ckpt
            or root / "runs" / "phase1_adv3_mechanism32_queue_20260701" / "ADV3B02_CORE90_SOFT_E200" / "best_joint_safe_ssdg.pth"
        )
        commands = [build_command(row, root=root, python=python, run_id=args.run_id, wisig_pkl=wisig, teacher_ckpt=teacher) for row in rows]
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "candidate_count": len(rows),
                    "cell_count": len(CELLS),
                    "gpu_total_counts": dict(sorted(Counter(int(row["gpu"]) for row in rows).items())),
                    "max_active_per_gpu": args.max_active_per_gpu,
                    "unique_command_count": len({tuple(command) for command in commands}),
                    "first_command": shlex.join(commands[0]),
                    "last_command": shlex.join(commands[-1]),
                    "claim": "PHASE1_SOURCE_ONLY_PROXY_DIAGNOSTIC_NO_TRUE_UNKNOWN_SUCCESS_CLAIM",
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    return run_scheduler(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
