from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import time
from collections import defaultdict
from copy import deepcopy
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from cvsrffi.phase1_ablation_factory import (
    apply_phase1_ablation,
    phase1_ablation_config,
)
from cvsrffi.crra_training import (
    crra_gate_scale,
    validate_crra_phase1_config,
    validate_crra_phase1_scenarios,
)

try:
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    from dataset_wisig import (
        WiSigCompactDataset,
        WiSigSubsetDataset,
        _resolve_days,
        _resolve_rxs,
        load_wisig_compact_pkl,
        make_wisig_trainval_test_by_day_rx,
    )
    from post_stage_common import (
        build_baseline_model,
        domain_from_extra,
        ensure_dir,
        load_checkpoint,
        mean_logs,
        merge_checkpoint_args,
        move_batch,
        resolve_device,
        save_payload,
        set_seed,
    )
    from training_controls import parse_sat_scenarios, satellite_protocol_manifest
    from baseline_origin_sat_view import parse_sat_view_schedule, normalize_crra_nuisance_meta
    from concat_sat_channel_aug import ConcatSatChannelAugment
    from cvsrffi.tensors import build_domain_label_map
    from cvsrffi.balanced_tx_rx_sampler import BalancedTxDomainBatchSampler
    from cvsrffi.bounded_domain_confusion import bounded_domain_objectives
    from cvsrffi.muse_ssdg import (
        MUSEClassificationPrototypeBank,
        MUSEConfig,
        MUSERoute,
        MUSEScheduleState,
        MUSETemporalMemory,
        MUSETrainingHeads,
        adv3b02_core90_u_satellite_policy,
        align_source_domain_prior,
        candidate_set_cross_entropy,
        candidate_set_mask,
        compute_muse_reliability,
        build_rc4_calibration,
        calibrate_sat_anchor_thresholds,
        rc4_identity_losses,
        rc4_active_identity_components,
        rc4_identity_tail_scales,
        rc4_tail_transition_scale,
        route_fasttrust_rc4,
        route_sat_anchor_trusted,
        sat_anchor_clean_kl,
        trusted_satellite_cross_entropy,
        geometric_fuse_probabilities,
        js_head_disagreement,
        muse_schedule_for_epoch,
        route_fasttrust,
        route_muse_reliability,
        select_satellite_student_mask,
        select_adv3b02_u_satellite_scenario,
        stable_sample_keys,
        weighted_soft_cross_entropy,
    )
    from cvsrffi.eval import (
        aggregate_named_stats,
        apply_sat_channel_for_scenario,
        evaluate_loader,
        evaluate_named_loaders,
        evaluate_sat_scenarios,
        format_named_test_lines,
        format_sat_test_lines,
        make_loader,
    )
    from cvsrffi.losses import (
        PrototypeMemoryBank,
        compute_core_losses,
        direct_metric_acceptance_loss,
        multiview_direct_metric_acceptance_loss,
        multiview_source_episode_three_sigma_loss,
        fishr_logit_gradient_variance_loss,
        make_soft_unknown_mixup,
        open_world_feature_space_loss,
        proxy_unknown_energy_loss,
        one_way_kl_from_teacher,
        sanitize_loss,
        soft_unknown_mixup_loss,
        source_episode_three_sigma_loss,
        tx_conditional_domain_invariance_loss,
        crra_nuisance_huber_loss,
        crra_satellite_shell_loss,
        unlabeled_known_acceptance_quarantine_loss,
        zid_compactness_loss,
    )
    from cvsrffi.leakage_probe import frozen_ridge_linear_probe
    from cvsrffi.phase2_prototypes import (
        Phase1CalibrationError,
        PrototypeFusionConfig,
        audit_identity_feature_contract,
        attach_endpoint_accept_v1_manifest,
        calibrate_endpoint_accept_v1,
        export_phase2_prototypes,
        extract_endpoint_calibration_features,
        fuse_tx_domain_prototypes,
        save_phase2_prototype_export,
        verify_endpoint_accept_v1_manifest,
    )
    from cvsrffi.hard_gate import LocalComponentHardGate
    from cvsrffi.ssdg_guard import (
        detect_one_epoch_drop,
        detect_paic_variance_guard,
        guard_minimums_from_args,
        joint_safe_score,
        missing_joint_safe_metrics,
        protected_metric_snapshot,
        sat_protocol_requirement_satisfied,
    )
    from cvsrffi.phase1_v2_control import (
        TailSafetyConfig,
        TailSafetyStateMachine,
        assess_endpoint_contract,
        assess_feasibility_gate,
        assess_open_set_effective_budget,
        assess_phase1_v2_final_export_policy,
        assess_source_episode_density_gate,
        assess_unlabeled_tri_state,
        compute_open_set_budget_action,
        should_skip_phase1_v2_final_export,
    )
    from cvsrffi.schedule import (
        build_aug_base_cfg,
        build_stage_state,
        configure_augmentor_for_epoch,
        configure_mixstyle_for_epoch,
        format_stage_state,
        make_augmentor,
    )
    from cvsrffi.tensors import make_torch_generator, parse_csv_indices
except ModuleNotFoundError:
    torch = None
    F = None
    GradScaler = autocast = DataLoader = None
    BalancedTxDomainBatchSampler = None
    MUSEClassificationPrototypeBank = MUSEConfig = MUSERoute = MUSEScheduleState = MUSETemporalMemory = None
    MUSETrainingHeads = candidate_set_cross_entropy = candidate_set_mask = None
    adv3b02_core90_u_satellite_policy = select_adv3b02_u_satellite_scenario = None
    align_source_domain_prior = compute_muse_reliability = geometric_fuse_probabilities = None
    calibrate_sat_anchor_thresholds = route_sat_anchor_trusted = None
    sat_anchor_clean_kl = trusted_satellite_cross_entropy = None
    build_rc4_calibration = rc4_identity_losses = rc4_tail_transition_scale = route_fasttrust_rc4 = None
    bounded_domain_objectives = rc4_active_identity_components = rc4_identity_tail_scales = None
    js_head_disagreement = muse_schedule_for_epoch = route_fasttrust = route_muse_reliability = None
    select_satellite_student_mask = stable_sample_keys = None
    weighted_soft_cross_entropy = None
    WiSigCompactDataset = WiSigSubsetDataset = None
    PrototypeMemoryBank = None
    direct_metric_acceptance_loss = None
    multiview_direct_metric_acceptance_loss = None
    multiview_source_episode_three_sigma_loss = None
    make_soft_unknown_mixup = None
    open_world_feature_space_loss = None
    proxy_unknown_energy_loss = None
    sanitize_loss = None
    soft_unknown_mixup_loss = None
    source_episode_three_sigma_loss = None
    tx_conditional_domain_invariance_loss = None
    crra_nuisance_huber_loss = None
    crra_satellite_shell_loss = None
    unlabeled_known_acceptance_quarantine_loss = None
    zid_compactness_loss = None
    export_phase2_prototypes = None
    fuse_tx_domain_prototypes = None
    save_phase2_prototype_export = None
    PrototypeFusionConfig = None
    Phase1CalibrationError = audit_identity_feature_contract = None
    attach_endpoint_accept_v1_manifest = verify_endpoint_accept_v1_manifest = None
    LocalComponentHardGate = None
    calibrate_endpoint_accept_v1 = extract_endpoint_calibration_features = None
    _resolve_days = _resolve_rxs = load_wisig_compact_pkl = make_wisig_trainval_test_by_day_rx = None
    build_baseline_model = domain_from_extra = ensure_dir = load_checkpoint = None
    mean_logs = merge_checkpoint_args = move_batch = resolve_device = save_payload = set_seed = None
    parse_sat_scenarios = satellite_protocol_manifest = None
    parse_sat_view_schedule = normalize_crra_nuisance_meta = None
    ConcatSatChannelAugment = None
    build_domain_label_map = evaluate_loader = evaluate_named_loaders = make_loader = parse_csv_indices = None
    apply_sat_channel_for_scenario = fishr_logit_gradient_variance_loss = make_torch_generator = None
    aggregate_named_stats = compute_core_losses = evaluate_sat_scenarios = None
    detect_one_epoch_drop = detect_paic_variance_guard = guard_minimums_from_args = None
    joint_safe_score = missing_joint_safe_metrics = protected_metric_snapshot = None
    sat_protocol_requirement_satisfied = None
    TailSafetyConfig = TailSafetyStateMachine = None
    assess_endpoint_contract = assess_feasibility_gate = None
    assess_open_set_effective_budget = assess_unlabeled_tri_state = None
    assess_phase1_v2_final_export_policy = None
    assess_source_episode_density_gate = None
    compute_open_set_budget_action = None
    should_skip_phase1_v2_final_export = None
    one_way_kl_from_teacher = None
    frozen_ridge_linear_probe = None
    build_aug_base_cfg = build_stage_state = configure_augmentor_for_epoch = configure_mixstyle_for_epoch = None
    format_stage_state = make_augmentor = None
    format_named_test_lines = format_sat_test_lines = None


_BICAD_XR_LEO_WEAK = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
_BICAD_XR_LEO_WEAK_CSV = ",".join(_BICAD_XR_LEO_WEAK)
_BICAD_XR_SAT_VIEW_SCHEDULE = (
    "1@0.30:leo_clear_weak;"
    "41@0.60:leo_low_elev_weak,leo_rain_weak;"
    "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
)


class BiCADXRProtocol:
    """The closed, source-only protocol consumed by the BiCAD-XR entry."""

    def __init__(self, *, phase1_method: str, candidate_id: str) -> None:
        from cvsrffi.phase1_bicad_xr.config import candidate_config

        resolved_config = candidate_config(candidate_id)
        self.phase1_method = phase1_method
        self.candidate_id = resolved_config.candidate_id
        self.use_concat_sat_channel_aug = True
        self.concat_sat_ce_only = True
        self.concat_sat_ce_weight = float(resolved_config.lambda_sat_cls)
        self.concat_sat_start_epoch = int(resolved_config.concat_sat_start_epoch)
        self.sat_train_scenarios = _BICAD_XR_LEO_WEAK_CSV
        self.lambda_sat_cls = float(resolved_config.lambda_sat_cls)
        self.lambda_sat_cons = float(resolved_config.lambda_sat_cons)
        self.strict_pair_concat = bool(resolved_config.strict_pair_concat)
        self.pair_identity = bool(resolved_config.pair_identity)
        self.pair_vicreg = bool(resolved_config.pair_vicreg)
        self.pair_delta = bool(resolved_config.pair_delta)
        self.dynamic_adversarial_dose = bool(resolved_config.dynamic_adversarial_dose)
        self.satellite_supervision_mode = resolved_config.satellite_supervision_mode
        self.pair_projector_dim = int(resolved_config.pair_projector_dim)
        self.factor_interaction_dim = int(resolved_config.factor_interaction_dim)
        self.lambda_sat_cls_start = float(resolved_config.lambda_sat_cls_start)
        self.lambda_sat_cls_end = float(resolved_config.lambda_sat_cls_end)
        self.target_access = False


def _cli_option(token: str) -> Tuple[str, Optional[str]]:
    if "=" in token:
        name, value = token.split("=", 1)
        return name, value
    return token, None


def _cli_option_values(argv: Sequence[str], option: str) -> List[str]:
    values: List[str] = []
    for index, token in enumerate(argv):
        name, inline_value = _cli_option(str(token))
        if name != option:
            continue
        if inline_value is not None:
            values.append(inline_value)
        elif index + 1 < len(argv) and not str(argv[index + 1]).startswith("-"):
            values.append(str(argv[index + 1]))
        else:
            values.append("")
    return values


def _cli_phase1_method(argv: Sequence[str]) -> str:
    values = _cli_option_values(argv, "--phase1_method")
    return values[-1].strip().lower() if values else "legacy"


def _reject_bicad_conflicts(argv: Sequence[str]) -> None:
    if _cli_phase1_method(argv) != "bicad_xr":
        return
    for token in argv:
        name, _ = _cli_option(str(token))
        lowered = name.lower()
        if lowered == "--phase2_fuse_prototypes" or lowered.startswith("--phase2_export_") or any(
            marker in lowered for marker in ("target", "support", "query", "truth")
        ):
            raise ValueError(f"BiCAD-XR rejects deployment input option: {name}")
        if lowered in {"--use_fasttrust", "--use_mixstyle"} or lowered.startswith("--fasttrust"):
            raise ValueError(f"BiCAD-XR rejects incompatible legacy option: {name}")
    for value in _cli_option_values(argv, "--muse_lr_schedule"):
        if value.strip().lower() != "off":
            raise ValueError("BiCAD-XR rejects incompatible FastTrust schedule")
    for value in _cli_option_values(argv, "--sat_train_scenario"):
        if value.strip().lower() != "":
            raise ValueError(f"BiCAD-XR rejects incompatible satellite scenario: {value}")
    for value in _cli_option_values(argv, "--sat_train_scenarios"):
        normalized = ",".join(part.strip().lower() for part in value.split(",") if part.strip())
        if normalized != _BICAD_XR_LEO_WEAK_CSV:
            raise ValueError(f"BiCAD-XR rejects incompatible satellite scenario: {value}")


def parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the public Task7 entry without changing the legacy CLI parser."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_bicad_conflicts(raw_argv)
    parse_argv = list(raw_argv)
    if not _cli_option_values(parse_argv, "--output_dir"):
        parse_argv = ["--output_dir", "", *parse_argv]
    args = build_arg_parser().parse_args(parse_argv)
    setattr(args, "_task7_cli_args", tuple(str(item) for item in raw_argv))
    return args


def route_phase1_method(args: argparse.Namespace) -> str:
    method = str(getattr(args, "phase1_method", "legacy") or "legacy").strip().lower()
    return "bicad_xr" if method == "bicad_xr" else "legacy"


def resolve_bicad_protocol(args: argparse.Namespace) -> BiCADXRProtocol:
    if route_phase1_method(args) != "bicad_xr":
        return args  # type: ignore[return-value]
    raw_argv = tuple(getattr(args, "_task7_cli_args", ()))
    if raw_argv:
        _reject_bicad_conflicts(raw_argv)
    candidate_id = str(getattr(args, "candidate_id", "D5") or "D5").strip() or "D5"
    return BiCADXRProtocol(
        phase1_method="bicad_xr",
        candidate_id=candidate_id,
    )


def parse_and_resolve(argv: Sequence[str]) -> BiCADXRProtocol:
    raw_argv = tuple(str(item) for item in argv)
    _reject_bicad_conflicts(raw_argv)
    args = parse(raw_argv)
    resolved = resolve_bicad_protocol(args)
    if not isinstance(resolved, BiCADXRProtocol):
        raise ValueError("BiCAD-XR entry requires --phase1_method bicad_xr")
    return resolved


def bicad_xr_runtime(protocol: BiCADXRProtocol) -> Dict[str, Any]:
    if not isinstance(protocol, BiCADXRProtocol):
        raise TypeError("bicad_xr_runtime expects a resolved BiCADXRProtocol")
    scenarios = list(_BICAD_XR_LEO_WEAK)
    return {
        "runtime_version": 1,
        "phase1_method": protocol.phase1_method,
        "candidate_id": protocol.candidate_id,
        "source_only": True,
        "target_access": False,
        "protocol": {
            "use_concat_sat_channel_aug": protocol.use_concat_sat_channel_aug,
            "concat_sat_ce_only": protocol.concat_sat_ce_only,
            "concat_sat_ce_weight": protocol.concat_sat_ce_weight,
            "lambda_sat_cls": protocol.lambda_sat_cls,
            "lambda_sat_cons": protocol.lambda_sat_cons,
            "concat_sat_start_epoch": protocol.concat_sat_start_epoch,
            "sat_train_scenarios": scenarios,
            "strict_pair_concat": protocol.strict_pair_concat,
            "pair_identity": protocol.pair_identity,
            "pair_vicreg": protocol.pair_vicreg,
            "pair_delta": protocol.pair_delta,
            "dynamic_adversarial_dose": protocol.dynamic_adversarial_dose,
            "satellite_supervision_mode": protocol.satellite_supervision_mode,
            "pair_projector_dim": protocol.pair_projector_dim,
            "factor_interaction_dim": protocol.factor_interaction_dim,
            "lambda_sat_cls_start": protocol.lambda_sat_cls_start,
            "lambda_sat_cls_end": protocol.lambda_sat_cls_end,
        },
        "reconstruct": {
            "phase1_method": protocol.phase1_method,
            "candidate_id": protocol.candidate_id,
            "protocol_version": "bicad_xr_v1",
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train two-stage SSDG from a Stable-SAT baseline checkpoint.")
    parser.add_argument("--phase1_method", type=str, default="legacy")
    parser.add_argument("--baseline_ckpt", type=str, default="", help="Optional checkpoint. Empty means train SSDG from scratch.")
    parser.add_argument("--from_scratch", type=str2bool, default=True)
    parser.add_argument("--split_mode", type=str, default="tx_rx_day_1_6_3", choices=["tx_rx_day_1_6_3", "tx_rx_day_1_7_2"])
    parser.add_argument("--labeled_ratio", type=float, default=0.08)
    parser.add_argument("--unlabeled_ratio", type=float, default=0.72)
    parser.add_argument("--source_val_ratio", type=float, default=0.20)
    parser.add_argument("--source_cal_ratio", type=float, default=0.0)
    parser.add_argument("--source_select_ratio", type=float, default=0.0)
    parser.add_argument(
        "--phase1_source_role_protocol",
        type=str,
        default="legacy_l_u_v",
        choices=["l_s_u_s_v_cal_v_select", "legacy_l_u_v"],
    )
    parser.add_argument("--pseudo_threshold_mode", type=str, default="rx_day_quantile", choices=["global", "rx_day_quantile"])
    parser.add_argument("--pseudo_quantile", type=float, default=0.70)
    parser.add_argument("--tau_conf", type=float, default=0.0, help="Alias for --tau_min used by older launchers.")
    parser.add_argument("--tau_min", type=float, default=0.80)
    parser.add_argument("--tau_max", type=float, default=0.97)
    parser.add_argument("--label_epochs", type=int, default=170)
    parser.add_argument("--pseudo_epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="",
        help="Optional per-epoch telemetry CSV path. Defaults to output_dir/metrics_epoch.csv.",
    )
    parser.add_argument(
        "--metrics_jsonl",
        type=str,
        default="",
        help="Optional per-epoch telemetry JSONL path. Defaults to output_dir/metrics_epoch.jsonl.",
    )
    parser.add_argument("--epochs", type=int, default=0, help="Compatibility alias: when >0, sets total epochs.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_ent", type=float, default=0.01)
    parser.add_argument("--lambda_u_domain", type=float, default=0.0)
    parser.add_argument("--lambda_u_adv", type=float, default=0.0)
    parser.add_argument("--lambda_u_sat_cons", type=float, default=0.0)
    parser.add_argument("--lambda_u_direct_metric_accept", type=float, default=0.0)
    parser.add_argument("--lambda_u_quarantine_accept", type=float, default=0.0)
    parser.add_argument("--u_domain_start_epoch", type=int, default=1)
    parser.add_argument("--u_sat_cons_start_epoch", type=int, default=1)
    parser.add_argument("--u_direct_metric_start_epoch", type=int, default=1)
    parser.add_argument("--u_direct_metric_min_selected", type=int, default=16)
    parser.add_argument("--u_direct_metric_use_sat_pair", type=str2bool, default=True)
    parser.add_argument("--u_direct_metric_valid_domain_only", type=str2bool, default=True)
    parser.add_argument("--u_quarantine_start_epoch", type=int, default=1)
    parser.add_argument("--u_quarantine_valid_domain_only", type=str2bool, default=True)
    parser.add_argument("--u_quarantine_include_sat_view", type=str2bool, default=True)
    parser.add_argument("--u_quarantine_min_count", type=int, default=4)
    parser.add_argument("--u_quarantine_core_quantile", type=float, default=0.70)
    parser.add_argument("--u_quarantine_accept_quantile", type=float, default=0.80)
    parser.add_argument("--u_quarantine_accept_target", type=float, default=0.20)
    parser.add_argument("--u_quarantine_cvar_alpha", type=float, default=0.25)
    parser.add_argument("--u_quarantine_accept_temperature", type=float, default=0.04)
    parser.add_argument("--u_quarantine_core_accept_target", type=float, default=0.82)
    parser.add_argument("--u_geometry_all_valid_queries", type=str2bool, default=False)
    parser.add_argument("--u_direct_include_ambiguous", type=str2bool, default=False)
    parser.add_argument("--u_tri_quota_routing", type=str2bool, default=False)
    parser.add_argument("--u_tri_core_quota", type=float, default=0.20)
    parser.add_argument("--u_tri_ambiguous_quota", type=float, default=0.30)
    parser.add_argument("--u_tri_quota_require_pseudo_mask", type=str2bool, default=True)
    parser.add_argument("--u_direct_include_outside_known", type=str2bool, default=False)
    parser.add_argument("--u_outside_stop_gradient", type=str2bool, default=False)
    parser.add_argument("--u_route_use_teacher_weak", type=str2bool, default=False)
    parser.add_argument("--u_route_use_reference_bank", type=str2bool, default=False)
    parser.add_argument("--u_tri_tail_pair_weight", type=float, default=0.0)
    parser.add_argument("--u_tri_outside_pair_weight", type=float, default=0.0)
    parser.add_argument("--u_tri_tail_pair_target_deg", type=float, default=12.0)
    parser.add_argument("--u_tri_outside_pair_target_deg", type=float, default=20.0)
    parser.add_argument("--u_unlabeled_shuffle", type=str2bool, default=True)
    parser.add_argument("--u_sat_zid_cons_weight", type=float, default=0.25)
    parser.add_argument("--lambda_domain", "--lambda_dom", dest="lambda_domain", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.45)
    parser.add_argument("--lambda_orth", type=float, default=0.05)
    parser.add_argument("--lambda_cons", type=float, default=0.08)
    parser.add_argument("--lambda_group_ce", type=float, default=0.10)
    parser.add_argument("--group_ce_top_frac", type=float, default=0.35)
    parser.add_argument("--group_ce_min_domains", type=int, default=4)
    parser.add_argument("--group_ce_mode", type=str, default="hard")
    parser.add_argument("--lambda_fishr", type=float, default=0.02)
    parser.add_argument("--lambda_zid_receiver_invariance", type=float, default=0.0)
    parser.add_argument("--lambda_zid_day_invariance", type=float, default=0.0)
    parser.add_argument("--lambda_zid_channel_invariance", type=float, default=0.0)
    parser.add_argument("--lambda_u_zid_receiver_invariance", type=float, default=0.0)
    parser.add_argument("--lambda_u_zid_day_invariance", type=float, default=0.0)
    parser.add_argument("--lambda_u_zid_channel_invariance", type=float, default=0.0)
    parser.add_argument("--zid_invariance_min_groups", type=int, default=2)
    parser.add_argument("--zid_invariance_min_samples_per_group", type=int, default=2)
    parser.add_argument("--zid_channel_pair_weight", type=float, default=1.0)
    parser.add_argument("--fishr_min_domains", type=int, default=4)
    parser.add_argument("--strong_noise_std", type=float, default=0.015)
    parser.add_argument("--use_unlabeled", type=str2bool, default=True)
    parser.add_argument("--pseudo_domain_gate", type=str2bool, default=True)
    parser.add_argument("--pseudo_temporal_gate", type=str2bool, default=True)
    parser.add_argument(
        "--pseudo_temporal_mode",
        type=str,
        default="batch_neighbor",
        choices=["batch_neighbor", "epoch_bank"],
    )
    parser.add_argument("--pseudo_temporal_bank_min_streak", type=int, default=2)
    parser.add_argument("--pseudo_temporal_window", type=int, default=2)
    parser.add_argument("--pseudo_temporal_min_conf", type=float, default=0.80)
    parser.add_argument("--pseudo_strong_agreement", type=str2bool, default=True)
    parser.add_argument("--use_ema_teacher", type=str2bool, default=False)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_muse_ssdg", type=str2bool, default=False)
    parser.add_argument("--muse_level", type=str, default="M0", choices=["M0", "M1", "M2", "M3"])
    parser.add_argument("--muse_external_final_eval", type=str2bool, default=False)
    parser.add_argument("--phase1_external_final_eval", type=str2bool, default=False)
    parser.add_argument(
        "--muse_epoch_basis",
        type=str,
        default="unlabeled_loader",
        choices=["unlabeled_loader"],
    )
    parser.add_argument("--muse_unlabeled_batch_size", type=int, default=256)
    parser.add_argument("--muse_fused_student_forward", type=str2bool, default=True)
    parser.add_argument(
        "--muse_lr_schedule",
        type=str,
        default="fasttrust",
        choices=["off", "fasttrust"],
    )
    parser.add_argument("--muse_s2a_start", type=int, default=17)
    parser.add_argument("--muse_s2b_start", type=int, default=41)
    parser.add_argument("--muse_s3a_start", type=int, default=69)
    parser.add_argument("--muse_s3b_start", type=int, default=161)
    parser.add_argument("--muse_s3c_start", type=int, default=181)
    parser.add_argument("--muse_final_epoch", type=int, default=200)
    parser.add_argument("--muse_lambda_u_full", type=float, default=0.60)
    parser.add_argument("--muse_lambda_u_consolidate", type=float, default=0.25)
    parser.add_argument("--muse_p_sat_s2a_end", type=float, default=0.25)
    parser.add_argument("--muse_p_sat_full", type=float, default=0.50)
    parser.add_argument("--muse_grl_min", type=float, default=0.02)
    parser.add_argument("--muse_grl_max", type=float, default=0.10)
    parser.add_argument("--muse_high_threshold", type=float, default=0.80)
    parser.add_argument("--muse_low_threshold", type=float, default=0.30)
    parser.add_argument("--muse_hard_max_fraction", type=float, default=0.25)
    parser.add_argument("--muse_identity_max_fraction", type=float, default=0.50)
    parser.add_argument("--muse_class_balanced_cap", type=str2bool, default=True)
    parser.add_argument("--muse_fasttrust_hard_only_no_fill", type=str2bool, default=False)
    parser.add_argument("--muse_require_temporal_stability", type=str2bool, default=True)
    parser.add_argument("--muse_candidate_mass", type=float, default=0.75)
    parser.add_argument("--muse_candidate_max_classes", type=int, default=3)
    parser.add_argument("--muse_fusion_global_weight", type=float, default=0.50)
    parser.add_argument("--muse_fusion_local_weight", type=float, default=0.25)
    parser.add_argument("--muse_fusion_prototype_weight", type=float, default=0.25)
    parser.add_argument("--muse_prior_alignment_gamma", type=float, default=0.35)
    parser.add_argument("--muse_reliability_confidence_weight", type=float, default=0.25)
    parser.add_argument("--muse_reliability_margin_weight", type=float, default=0.20)
    parser.add_argument("--muse_reliability_js_weight", type=float, default=0.20)
    parser.add_argument("--muse_reliability_prototype_weight", type=float, default=0.20)
    parser.add_argument("--muse_reliability_stability_weight", type=float, default=0.15)
    parser.add_argument("--muse_unlabeled_prototype_weight", type=float, default=0.075)
    parser.add_argument("--muse_enable_u_prototype_update", type=str2bool, default=None)
    parser.add_argument("--muse_use_prototype_evidence", type=str2bool, default=True)
    parser.add_argument("--muse_temporal_stability_steps", type=int, default=3)
    parser.add_argument("--muse_nuisance_dim", type=int, default=6)
    parser.add_argument("--muse_lambda_domain", type=float, default=1.0)
    parser.add_argument("--muse_lambda_adv", type=float, default=1.0)
    parser.add_argument("--muse_lambda_self", type=float, default=0.10)
    parser.add_argument("--muse_lambda_nuisance", type=float, default=0.10)
    parser.add_argument("--muse_nuisance_detached", type=str2bool, default=False)
    parser.add_argument("--muse_lambda_satellite", type=float, default=0.68)
    parser.add_argument("--muse_enable_u_satellite_identity", type=str2bool, default=True)
    parser.add_argument("--muse_lambda_cross_receiver", type=float, default=0.10)
    parser.add_argument("--muse_lambda_high", type=float, default=1.0)
    parser.add_argument("--muse_lambda_mid", type=float, default=0.50)
    parser.add_argument("--muse_lambda_low_entropy", type=float, default=0.01)
    parser.add_argument("--sat_anchor_ssl", type=str2bool, default=False)
    parser.add_argument("--sat_anchor_beta", type=float, default=0.50)
    parser.add_argument("--sat_anchor_calibration_epsilon", type=float, default=0.02)
    parser.add_argument("--sat_anchor_hard_max_fraction", type=float, default=0.25)
    parser.add_argument("--sat_anchor_fill_to_fraction", type=float, default=0.0)
    parser.add_argument("--sat_anchor_receiver_balanced_cap", type=str2bool, default=False)
    parser.add_argument("--sat_anchor_lambda_pair", type=float, default=0.02)
    parser.add_argument("--sat_anchor_pair_interval", type=int, default=1)
    parser.add_argument("--sat_anchor_lambda_satellite", type=float, default=0.10)
    parser.add_argument("--sat_anchor_lambda_clean_kl", type=float, default=0.10)
    parser.add_argument("--sat_anchor_temperature", type=float, default=2.0)
    parser.add_argument("--sat_anchor_identity_start_epoch", type=int, default=80)
    parser.add_argument(
        "--sat_anchor_u_gradient_scope",
        type=str,
        default="full",
        choices=["full", "adapter_tail"],
    )
    parser.add_argument("--sat_anchor_adapter", type=str2bool, default=False)
    parser.add_argument("--sat_anchor_adapter_rank", type=int, default=8)
    parser.add_argument("--fasttrust_rc4", type=str2bool, default=False)
    parser.add_argument("--rc4_use_anchor", type=str2bool, default=True)
    parser.add_argument("--rc4_cache_anchor_logits", type=str2bool, default=False)
    parser.add_argument("--rc4_use_correctness_calibration", type=str2bool, default=True)
    parser.add_argument("--rc4_enable_hard", type=str2bool, default=True)
    parser.add_argument("--rc4_enable_partial", type=str2bool, default=True)
    parser.add_argument("--rc4_enable_partial_set", type=str2bool, default=True)
    parser.add_argument("--rc4_enable_partial_conditional", type=str2bool, default=True)
    parser.add_argument("--rc4_enable_negative", type=str2bool, default=True)
    parser.add_argument("--rc4_decouple_partial_negative_aps", type=str2bool, default=False)
    parser.add_argument(
        "--rc4_partial_threshold_scope",
        type=str,
        default="predicted_class",
        choices=["global", "predicted_class"],
    )
    parser.add_argument("--rc4_hard_effective_budget", type=float, default=0.0)
    parser.add_argument("--rc4_partial_effective_budget", type=float, default=0.10)
    parser.add_argument("--rc4_negative_effective_budget", type=float, default=0.10)
    parser.add_argument("--rc4_total_identity_effective_budget", type=float, default=0.0)
    parser.add_argument("--rc4_use_calibrated_partial_threshold", type=str2bool, default=False)
    parser.add_argument("--rc4_class_receiver_cap", type=str2bool, default=True)
    parser.add_argument("--rc4_class_receiver_effective_budget", type=float, default=0.0)
    parser.add_argument("--rc4_satellite_hard_only", type=str2bool, default=False)
    parser.add_argument("--rc4_identity_start_epoch", type=int, default=11)
    parser.add_argument("--rc4_consolidation_start_epoch", type=int, default=181)
    parser.add_argument("--rc4_calibration_update_epochs", type=str, default="1,41,91,161")
    parser.add_argument("--rc4_lambda_hard", type=float, default=0.60)
    parser.add_argument("--rc4_lambda_partial", type=float, default=0.40)
    parser.add_argument("--rc4_lambda_partial_set", type=float, default=-1.0)
    parser.add_argument("--rc4_lambda_partial_conditional", type=float, default=-1.0)
    parser.add_argument("--rc4_lambda_negative", type=float, default=0.20)
    parser.add_argument("--rc4_lambda_domain", type=float, default=0.16)
    parser.add_argument("--rc4_lambda_zdom", type=float, default=-1.0)
    parser.add_argument("--rc4_lambda_discriminator", type=float, default=-1.0)
    parser.add_argument("--rc4_lambda_confusion", type=float, default=-1.0)
    parser.add_argument("--rc4_lambda_feature_anchor", type=float, default=0.0)
    parser.add_argument("--rc4_lambda_self", type=float, default=0.10)
    parser.add_argument("--rc4_lambda_satellite", type=float, default=0.10)
    parser.add_argument("--rc4_tail_transition_start_epoch", type=int, default=91)
    parser.add_argument("--rc4_tail_transition_epochs", type=int, default=20)
    parser.add_argument("--rc4_tail_transition_floor", type=float, default=0.25)
    parser.add_argument("--rc4_identity_tail_hard_final", type=float, default=0.60)
    parser.add_argument("--rc4_identity_tail_partial_set_final", type=float, default=0.20)
    parser.add_argument("--rc4_identity_tail_partial_conditional_final", type=float, default=0.0)
    parser.add_argument("--rc4_gradient_telemetry_epochs", type=str, default="1,41,91,161,181,200")
    parser.add_argument("--rc4_recovery_checkpoint_interval", type=int, default=0)
    parser.add_argument(
        "--identity_domain_objective_mode",
        type=str,
        default="grl_ce",
        choices=["grl_ce", "bounded_confusion"],
    )
    parser.add_argument("--identity_domain_discriminator_scale", type=float, default=0.50)
    parser.add_argument("--identity_domain_confusion_scale", type=float, default=0.50)
    parser.add_argument("--rc4_nonfinite_guard_min_count", type=int, default=8)
    parser.add_argument("--rc4_nonfinite_guard_fraction", type=float, default=0.05)
    parser.add_argument("--teacher_ckpt", type=str, default="", help="Optional frozen teacher checkpoint for ADV3B02 distillation.")
    parser.add_argument("--lambda_teacher_clean_kl", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_sat_kl", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_zid_mse", type=float, default=0.0)
    parser.add_argument("--teacher_distill_temperature", type=float, default=2.0)
    parser.add_argument("--teacher_distill_start_epoch", type=int, default=1)
    parser.add_argument("--teacher_distill_warmup_epochs", type=int, default=0)
    parser.add_argument("--use_sat_consistency", dest="use_sat_consistency", action="store_true", default=True)
    parser.add_argument("--no_use_sat_consistency", dest="use_sat_consistency", action="store_false")
    parser.add_argument("--sat_train_scenario", type=str, default="mixed_orbit")
    parser.add_argument("--sat_train_scenarios", type=str, default="")
    parser.add_argument("--sat_view_schedule", type=str, default="")
    parser.add_argument("--use_concat_sat_channel_aug", dest="use_concat_sat_channel_aug", action="store_true", default=False)
    parser.add_argument("--no_use_concat_sat_channel_aug", dest="use_concat_sat_channel_aug", action="store_false")
    parser.add_argument("--concat_sat_ce_only", dest="concat_sat_ce_only", action="store_true", default=False)
    parser.add_argument("--no_concat_sat_ce_only", dest="concat_sat_ce_only", action="store_false")
    parser.add_argument(
        "--sat_training_mode",
        type=str,
        default="",
        choices=["", "disabled", "view_aux", "concat_ce_only", "concat_full", "concat_masked", "concat_legacy"],
    )
    parser.add_argument("--concat_sat_ce_weight", type=float, default=1.0)
    parser.add_argument("--concat_sat_deduplicate_tx_ce", type=str2bool, default=False)
    parser.add_argument("--concat_sat_teacher_clean_only", type=str2bool, default=False)
    parser.add_argument("--concat_sat_start_epoch", type=int, default=1)
    parser.add_argument("--sat_view_prob", type=float, default=1.0)
    parser.add_argument("--sat_view_seed", type=int, default=2027)
    parser.add_argument("--use_crra", dest="use_crra", action="store_true", default=False)
    parser.add_argument("--no_use_crra", dest="use_crra", action="store_false")
    parser.add_argument("--crra_scenario", type=str, default="mixed_orbit")
    parser.add_argument("--crra_rank", type=int, default=8)
    parser.add_argument("--crra_alpha_max", type=float, default=0.25)
    parser.add_argument("--crra_shrinkage", type=float, default=0.10)
    parser.add_argument("--crra_condition_dim", type=int, default=32)
    parser.add_argument("--crra_nuisance_dim", type=int, default=9)
    parser.add_argument("--crra_start_epoch", type=int, default=17)
    parser.add_argument("--crra_ramp_epochs", type=int, default=30)
    parser.add_argument("--crra_target_adapter", type=str2bool, default=False)
    parser.add_argument("--lambda_crra_pair", type=float, default=0.05)
    parser.add_argument("--lambda_crra_sat_kl", type=float, default=0.05)
    parser.add_argument("--lambda_crra_sat_shell", type=float, default=0.0)
    parser.add_argument("--crra_sat_shell_width_deg", type=float, default=12.0)
    parser.add_argument("--lambda_crra_energy", type=float, default=0.001)
    parser.add_argument("--lambda_crra_gate_l1", type=float, default=0.001)
    parser.add_argument("--lambda_crra_nuisance", type=float, default=0.02)
    parser.add_argument("--lambda_crra_condition_tx_adv", type=float, default=0.02)
    parser.add_argument("--sat_protocol_disjoint_required", type=str2bool, default=False)
    parser.add_argument("--lambda_sat_cls", type=float, default=0.10)
    parser.add_argument("--lambda_sat_cons", type=float, default=0.0)
    parser.add_argument("--sat_cons_start_epoch", type=int, default=20)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="clean_val_tx",
        choices=["clean_val_tx", "source_val_sat_hmean", "test_overall_tx", "sat_mean_tx", "sat_worst_tx", "joint_safe"],
        help="Source-val telemetry metric only; Phase1 checkpoint selection is final-only.",
    )
    parser.add_argument(
        "--checkpoint_selection",
        type=str,
        default="final_only",
        choices=["final_only"],
    )
    parser.add_argument("--safe_best_path", type=str, default="", help="Optional path for the guarded best checkpoint.")
    parser.add_argument("--safe_latest_path", type=str, default="", help="Optional path for the latest guarded checkpoint.")
    parser.add_argument("--phase1_source_val_selection_only", type=str2bool, default=True)
    parser.add_argument(
        "--test_eval_policy",
        type=str,
        default="every_epoch",
        choices=["every_epoch", "val_improved_final", "interval_final"],
        help="Legacy compatibility option; Phase1 source-only training never evaluates held-out tests by epoch.",
    )
    parser.add_argument(
        "--test_eval_start_epoch",
        type=int,
        default=1,
        help="First epoch allowed to run named test-set and satellite evaluation during SSDG training.",
    )
    parser.add_argument(
        "--test_eval_interval",
        type=int,
        default=0,
        help="For --test_eval_policy interval_final, run named test-set and satellite evaluation every N epochs plus final.",
    )
    parser.add_argument(
        "--test_eval_final_window",
        type=int,
        default=0,
        help="For --test_eval_policy interval_final, use a denser interval inside the final N epochs; 0 disables.",
    )
    parser.add_argument(
        "--test_eval_final_interval",
        type=int,
        default=0,
        help="For --test_eval_policy interval_final, run named test-set and satellite evaluation every N epochs inside --test_eval_final_window; final epoch still runs.",
    )
    parser.add_argument(
        "--source_val_heavy_eval_start_epoch",
        type=int,
        default=1,
        help="First epoch eligible for heavy source-val tail and satellite evaluation.",
    )
    parser.add_argument(
        "--source_val_heavy_eval_interval",
        type=int,
        default=1,
        help="Run heavy source-val tail and satellite evaluation every N epochs before the final window.",
    )
    parser.add_argument(
        "--source_val_heavy_eval_final_window",
        type=int,
        default=0,
        help="Use a denser heavy source-val evaluation interval inside the final N epochs; 0 disables.",
    )
    parser.add_argument(
        "--source_val_heavy_eval_final_interval",
        type=int,
        default=1,
        help="Heavy source-val evaluation interval inside the final window. The final epoch always runs.",
    )
    parser.add_argument("--enable_joint_safe_guard", type=str2bool, default=False)
    parser.add_argument("--joint_guard_require_satellite", type=str2bool, default=True)
    parser.add_argument("--joint_guard_min_strict_udu", type=float, default=0.0)
    parser.add_argument("--joint_guard_min_receiver_floor", type=float, default=0.0)
    parser.add_argument("--joint_guard_min_sat_mean", type=float, default=0.0)
    parser.add_argument("--joint_guard_min_sat_floor", type=float, default=0.0)
    parser.add_argument("--joint_guard_min_sat_strict_mean", type=float, default=0.0)
    parser.add_argument("--joint_guard_min_sat_strict_floor", type=float, default=0.0)
    parser.add_argument("--one_epoch_drop_guard_pp", type=float, default=2.0)
    parser.add_argument("--paic_guard_enabled", type=str2bool, default=False)
    parser.add_argument("--paic_guard_sat_ce_delta", type=float, default=0.12)
    parser.add_argument("--paic_guard_grad_delta", type=float, default=3.0)
    parser.add_argument("--paic_guard_reliable_drop", type=float, default=0.01)
    parser.add_argument("--paic_guard_domain_delta", type=float, default=0.0)
    parser.add_argument("--paic_guard_sat_cons_delta", type=float, default=0.0)
    parser.add_argument("--paic_guard_block_best", type=str2bool, default=True)
    parser.add_argument("--paic_guard_cooldown_epochs", type=int, default=1)
    parser.add_argument("--paic_guard_sat_scale", type=float, default=0.75)
    parser.add_argument("--phase1_v2_hard_gates", type=str2bool, default=False)
    parser.add_argument("--endpoint_accept_policy_id", type=str, default="endpoint_accept_v1")
    parser.add_argument("--endpoint_threshold_source", type=str, default="source_val_only")
    parser.add_argument("--endpoint_calibration_split", type=str, default="source_val")
    parser.add_argument("--loss_gate_exported", type=str2bool, default=False)
    parser.add_argument("--tail_safety_state_machine", type=str2bool, default=False)
    parser.add_argument("--tail_stop_blocks_final", type=str2bool, default=True)
    parser.add_argument("--tail_safety_warning_patience", type=int, default=2)
    parser.add_argument("--tail_safety_rollback_patience", type=int, default=1)
    parser.add_argument("--tail_safety_max_rollbacks", type=int, default=1)
    parser.add_argument("--tail_safety_p95_target_deg", type=float, default=54.0)
    parser.add_argument("--tail_safety_p99_target_deg", type=float, default=70.0)
    parser.add_argument("--tail_safety_cvar_target_deg", type=float, default=56.0)
    parser.add_argument("--tail_safety_proxy_vaccept_target", type=float, default=0.35)
    parser.add_argument("--tail_safety_p99_expansion_block_final_delta", type=float, default=2.0)
    parser.add_argument("--tail_safety_p99_expansion_block_best_delta", type=float, default=3.5)
    parser.add_argument("--tail_safety_cvar_expansion_block_final_delta", type=float, default=4.0)
    parser.add_argument("--tail_safety_cvar_expansion_block_best_delta", type=float, default=6.0)
    parser.add_argument("--tail_safety_reference_window", type=int, default=5)
    parser.add_argument("--tail_safety_absolute_violation_drives_state", type=str2bool, default=True)
    parser.add_argument("--tail_safety_training_stop_enabled", type=str2bool, default=True)
    parser.add_argument("--tail_safety_reference_requires_absolute_safe", type=str2bool, default=True)
    parser.add_argument("--tail_rollback_enabled", type=str2bool, default=False)
    parser.add_argument("--tail_rollback_cooldown_epochs", type=int, default=2)
    parser.add_argument("--tail_rollback_closed_scale", type=float, default=0.60)
    parser.add_argument("--os_eff_min_budget", type=float, default=0.0)
    parser.add_argument("--os_eff_max_budget", type=float, default=0.0)
    parser.add_argument("--os_budget_controller", type=str2bool, default=False)
    parser.add_argument("--os_budget_max_scale", type=float, default=4.0)
    parser.add_argument("--os_budget_min_closed_scale", type=float, default=0.35)
    parser.add_argument("--os_budget_target_reserve", type=float, default=0.0)
    parser.add_argument("--os_gradient_surgery", type=str2bool, default=False)
    parser.add_argument("--os_gradient_surgery_interval", type=int, default=1)
    parser.add_argument("--os_gradient_protect_closed", type=str2bool, default=False)
    parser.add_argument(
        "--os_budget_scope",
        type=str,
        default="all_shared",
        choices=["all_shared", "zid_path"],
    )
    parser.add_argument("--os_objective_budget_controller", type=str2bool, default=False)
    parser.add_argument("--os_objective_boundary_share", type=float, default=0.40)
    parser.add_argument("--os_objective_source_share", type=float, default=0.25)
    parser.add_argument("--os_objective_invariant_share", type=float, default=0.20)
    parser.add_argument("--os_objective_u_share", type=float, default=0.15)
    parser.add_argument("--os_objective_min_scale", type=float, default=0.25)
    parser.add_argument("--os_objective_max_scale", type=float, default=8.0)
    parser.add_argument("--phase1_v2_os_eff_all_phases", type=str2bool, default=True)
    parser.add_argument("--phase1_v2_guard_blocks_final", type=str2bool, default=True)
    parser.add_argument("--source_val_dg_health_guard", type=str2bool, default=False)
    parser.add_argument("--source_val_dg_health_start_epoch", type=int, default=10)
    parser.add_argument("--source_val_dg_health_warning_drop_pp", type=float, default=3.0)
    parser.add_argument("--source_val_dg_health_stop_drop_pp", type=float, default=8.0)
    parser.add_argument("--source_val_dg_health_floor", type=float, default=60.0)
    parser.add_argument("--source_val_dg_health_min_open_scale", type=float, default=0.20)
    parser.add_argument("--source_val_dg_health_stop_patience", type=int, default=1)
    parser.add_argument("--u_tri_state_required", type=str2bool, default=False)
    parser.add_argument("--u_direct_idle_blocks_promotion", type=str2bool, default=True)
    parser.add_argument("--u_tri_min_core_rate", type=float, default=0.05)
    parser.add_argument("--u_tri_max_core_rate", type=float, default=0.95)
    parser.add_argument("--u_tri_min_ambiguous_rate", type=float, default=0.01)
    parser.add_argument("--u_tri_max_outside_rate", type=float, default=0.80)
    parser.add_argument("--u_tri_min_class_coverage", type=int, default=2)
    parser.add_argument("--u_tri_min_domain_coverage", type=int, default=2)
    parser.add_argument("--u_tri_max_pair_disagreement_rate", type=float, default=0.25)
    parser.add_argument("--u_tri_min_pseudo_component_agreement", type=float, default=0.80)
    parser.add_argument("--source_episode_density_gate", type=str2bool, default=False)
    parser.add_argument("--source_episode_overflow_warn", type=float, default=0.90)
    parser.add_argument("--source_episode_min_local_components", type=int, default=1)
    parser.add_argument("--source_episode_local_compact_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_local_invariant_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_local_inter_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_local_inter_margin_deg", type=float, default=20.0)
    parser.add_argument("--source_episode_local_center_target_deg", type=float, default=0.0)
    parser.add_argument("--source_episode_local_overlap_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_local_overlap_margin_deg", type=float, default=4.0)
    parser.add_argument("--source_episode_local_accept_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_local_density_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_local_min_samples", type=int, default=2)
    parser.add_argument("--source_episode_local_radius_floor_deg", type=float, default=3.0)
    parser.add_argument("--source_episode_local_density_beta", type=float, default=0.20)
    parser.add_argument("--source_episode_local_density_cap", type=float, default=2.0)
    parser.add_argument("--source_episode_local_term_cap", type=float, default=4.0)
    parser.add_argument("--source_episode_leave_domain_target_deg", type=float, default=40.0)
    parser.add_argument("--source_episode_leave_domain_target_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_structural_cvar_alpha", type=float, default=0.20)
    parser.add_argument("--source_episode_structural_start_epoch", type=int, default=-1)
    parser.add_argument("--source_episode_structural_warmup_epochs", type=int, default=-1)
    parser.add_argument("--source_episode_clean_weight", type=float, default=1.0)
    parser.add_argument("--source_episode_sat_weight", type=float, default=1.0)
    parser.add_argument("--source_episode_multiview_normalize", type=str2bool, default=True)
    parser.add_argument("--direct_metric_multiview_separate", type=str2bool, default=False)
    parser.add_argument("--direct_metric_domain_local_components", type=str2bool, default=False)
    parser.add_argument("--direct_metric_require_domain_local_components", type=str2bool, default=False)
    parser.add_argument("--direct_metric_min_samples_per_component", type=int, default=2)
    parser.add_argument("--direct_metric_clean_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_sat_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_hierarchical_class_gate", type=str2bool, default=False)
    parser.add_argument(
        "--direct_metric_hierarchical_combine",
        type=str,
        default="product",
        choices=["product", "smooth_min"],
    )
    parser.add_argument("--endpoint_require_artifact_on_export", type=str2bool, default=True)
    parser.add_argument("--endpoint_calibration_min_component_samples", type=int, default=4)
    parser.add_argument("--endpoint_calibration_min_class_samples", type=int, default=4)
    parser.add_argument("--endpoint_calibration_core_quantile", type=float, default=0.80)
    parser.add_argument("--endpoint_calibration_accept_quantile", type=float, default=0.95)
    parser.add_argument("--endpoint_calibration_tail_quantile", type=float, default=0.99)
    parser.add_argument("--phase1_export_diagnostic_on_block", type=str2bool, default=False)
    parser.add_argument("--zid_leakage_probe_required", type=str2bool, default=False)
    parser.add_argument("--zid_leakage_probe_max_batches", type=int, default=0)
    parser.add_argument("--zid_leakage_probe_ridge", type=float, default=0.01)
    parser.add_argument("--zid_receiver_probe_max_excess", type=float, default=0.20)
    parser.add_argument("--zid_day_probe_max_excess", type=float, default=0.15)
    parser.add_argument("--zid_channel_probe_max_excess", type=float, default=0.15)
    parser.add_argument("--feasibility_gate", type=str2bool, default=False)
    parser.add_argument(
        "--feasibility_stage",
        type=str,
        default="audit",
        choices=["audit", "relaxed", "local", "full"],
    )
    parser.add_argument("--feasibility_relaxed_pass", type=str2bool, default=False)
    parser.add_argument("--feasibility_local_pass", type=str2bool, default=False)
    parser.add_argument("--feasibility_loss_response_slope", type=float, default=float("nan"))
    parser.add_argument("--feasibility_overflow_excess_cvar95_delta", type=float, default=float("nan"))
    parser.add_argument("--use_phase2_ground_prototypes", type=str2bool, default=False)
    parser.add_argument("--use_feature_masks", type=str2bool, default=False)
    parser.add_argument("--use_txrx_geometry_losses", type=str2bool, default=False)
    parser.add_argument("--use_tx_rx_balanced_sampler", type=str2bool, default=False)
    parser.add_argument("--balanced_sampler_tx_per_batch", type=int, default=6)
    parser.add_argument("--balanced_sampler_domain_per_batch", type=int, default=6)
    parser.add_argument("--balanced_sampler_samples_per_cell", type=int, default=3)
    parser.add_argument("--balanced_sampler_replacement", type=str2bool, default=True)
    parser.add_argument("--phase1_distribution_audit_only", type=str2bool, default=True)
    parser.add_argument("--lambda_tx_proto", type=float, default=0.0)
    parser.add_argument("--lambda_rx_proto", type=float, default=0.0)
    parser.add_argument("--lambda_mask_aux", type=float, default=0.0)
    parser.add_argument("--lambda_tx_supcon_masked", type=float, default=0.0)
    parser.add_argument("--lambda_rx_supcon_masked", type=float, default=0.0)
    parser.add_argument("--lambda_txrx_rect", type=float, default=0.0)
    parser.add_argument("--use_proto_memory", type=str2bool, default=False)
    parser.add_argument("--lambda_proto", type=float, default=0.0)
    parser.add_argument("--proto_momentum", type=float, default=0.95)
    parser.add_argument("--proto_margin", type=float, default=0.15)
    parser.add_argument("--proto_domain_align_weight", type=float, default=0.5)
    parser.add_argument("--proto_push_weight", type=float, default=0.1)
    parser.add_argument("--proto_min_count", type=int, default=2)
    parser.add_argument("--lambda_open_world_feat", type=float, default=0.0)
    parser.add_argument("--ow_feat_start_epoch", type=int, default=1)
    parser.add_argument("--ow_feat_warmup_epochs", type=int, default=0)
    parser.add_argument("--ow_feat_radius_deg", type=float, default=12.0)
    parser.add_argument("--ow_feat_inter_margin_deg", type=float, default=55.0)
    parser.add_argument("--ow_feat_sample_margin_deg", type=float, default=5.0)
    parser.add_argument("--ow_feat_domain_align_weight", type=float, default=0.0)
    parser.add_argument("--ow_feat_min_classes", type=int, default=2)
    parser.add_argument("--ow_feat_min_samples_per_class", type=int, default=1)
    parser.add_argument("--ow_feat_tail_mode", type=str, default="none", choices=["none", "robust_3sigma"])
    parser.add_argument("--ow_feat_tail_weight", type=float, default=0.0)
    parser.add_argument("--ow_feat_cvar_alpha", type=float, default=0.95)
    parser.add_argument("--ow_feat_vacuum_weight", type=float, default=0.0)
    parser.add_argument("--ow_feat_vacuum_width_deg", type=float, default=4.0)
    parser.add_argument("--ow_feat_vacuum_hard_k", type=int, default=2)
    parser.add_argument("--ow_feat_soft_gate", type=str2bool, default=False)
    parser.add_argument("--ow_feat_gate_floor", type=float, default=0.25)
    parser.add_argument("--lambda_zid_compact", type=float, default=0.0)
    parser.add_argument("--zid_compact_start_epoch", type=int, default=1)
    parser.add_argument("--zid_compact_supcon_weight", type=float, default=0.35)
    parser.add_argument("--zid_compact_radius_weight", type=float, default=0.35)
    parser.add_argument("--zid_compact_cvar_weight", type=float, default=0.30)
    parser.add_argument("--zid_compact_cvar_alpha", type=float, default=0.90)
    parser.add_argument("--zid_compact_radius_deg", type=float, default=40.0)
    parser.add_argument("--zid_compact_warmup_epochs", type=int, default=30)
    parser.add_argument("--zid_compact_domain_aware", type=str2bool, default=True)
    parser.add_argument("--lambda_proxy_unknown", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_start_epoch", type=int, default=40)
    parser.add_argument("--proxy_unknown_warmup_epochs", type=int, default=0)
    parser.add_argument("--proxy_unknown_holdout_tx_per_batch", type=int, default=1)
    parser.add_argument("--proxy_unknown_virtual_count", type=int, default=16)
    parser.add_argument("--proxy_unknown_virtual_mode", type=str, default="legacy", choices=["legacy", "hard", "mixed", "legacy_hard"])
    parser.add_argument("--proxy_unknown_energy_margin", type=float, default=1.0)
    parser.add_argument("--proxy_unknown_energy_temperature", type=float, default=1.0)
    parser.add_argument("--proxy_unknown_placeholder_weight", type=float, default=0.5)
    parser.add_argument("--proxy_unknown_virtual_detach", type=str2bool, default=True)
    parser.add_argument("--proxy_unknown_vacuum_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_vacuum_width_deg", type=float, default=4.0)
    parser.add_argument("--proxy_unknown_vacuum_hard_k", type=int, default=2)
    parser.add_argument("--proxy_unknown_vacuum_radius_deg", type=float, default=40.0)
    parser.add_argument("--proxy_unknown_core_quantile", type=float, default=0.90)
    parser.add_argument("--proxy_unknown_accept_quantile", type=float, default=0.95)
    parser.add_argument("--proxy_unknown_tail_quantile", type=float, default=0.95)
    parser.add_argument("--proxy_unknown_overflow_quantile", type=float, default=0.99)
    parser.add_argument(
        "--proxy_unknown_component_radius_mode",
        type=str,
        default="core_quantile",
        choices=["three_sigma", "core_quantile", "accept_quantile", "min_three_sigma_core", "min_three_sigma_quantile"],
    )
    parser.add_argument("--proxy_unknown_component_radius_quantile", type=float, default=0.80)
    parser.add_argument("--proxy_unknown_vaccept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_core_accept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_component_gate_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_tail_quarantine_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_source_safe_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_bridge_accept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_shell_outward_accept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_low_density_accept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_energy_margin_quantile_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_radius_budget_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_radius_inter_ratio_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_vaccept_cvar_alpha", type=float, default=0.25)
    parser.add_argument("--proxy_unknown_unknown_margin", type=float, default=0.08)
    parser.add_argument("--proxy_unknown_known_margin", type=float, default=0.05)
    parser.add_argument("--proxy_unknown_energy_softplus_temperature", type=float, default=0.04)
    parser.add_argument("--proxy_unknown_accept_softplus_temperature", type=float, default=0.04)
    parser.add_argument("--proxy_unknown_bridge_accept_target", type=float, default=0.20)
    parser.add_argument("--proxy_unknown_shell_outward_accept_target", type=float, default=0.25)
    parser.add_argument("--proxy_unknown_tail_accept_target", type=float, default=0.45)
    parser.add_argument("--proxy_unknown_overflow_accept_target", type=float, default=0.25)
    parser.add_argument("--proxy_unknown_energy_margin_q", type=float, default=0.10)
    parser.add_argument("--proxy_unknown_energy_margin_target", type=float, default=0.08)
    parser.add_argument("--proxy_unknown_radius_budget_deg", type=float, default=10.0)
    parser.add_argument("--proxy_unknown_radius_max_budget_deg", type=float, default=15.0)
    parser.add_argument("--proxy_unknown_radius_inter_ratio_target", type=float, default=0.25)
    parser.add_argument("--proxy_unknown_density_temperature_deg", type=float, default=3.0)
    parser.add_argument("--proxy_unknown_component_temperature_deg", type=float, default=3.0)
    parser.add_argument("--proxy_unknown_component_margin_deg", type=float, default=4.0)
    parser.add_argument("--proxy_unknown_component_margin_temperature_deg", type=float, default=3.0)
    parser.add_argument("--proxy_unknown_shell_width_deg", type=float, default=4.0)
    parser.add_argument("--lambda_soft_unknown_mixup", type=float, default=0.0)
    parser.add_argument("--soft_unknown_mixup_start_epoch", type=int, default=-1)
    parser.add_argument("--soft_unknown_mixup_warmup_epochs", type=int, default=-1)
    parser.add_argument("--soft_unknown_mixup_count", type=int, default=16)
    parser.add_argument("--soft_unknown_mixup_order", type=int, default=3)
    parser.add_argument("--soft_unknown_mixup_alpha", type=float, default=0.5)
    parser.add_argument("--soft_unknown_mixup_energy_margin", type=float, default=1.0)
    parser.add_argument("--soft_unknown_mixup_ce_weight", type=float, default=1.0)
    parser.add_argument("--soft_unknown_mixup_energy_weight", type=float, default=1.0)
    parser.add_argument("--soft_unknown_mixup_vacuum_weight", type=float, default=0.0)
    parser.add_argument("--soft_unknown_mixup_vacuum_width_deg", type=float, default=6.0)
    parser.add_argument("--soft_unknown_mixup_vacuum_hard_k", type=int, default=2)
    parser.add_argument("--soft_unknown_mixup_detach", type=str2bool, default=False)
    parser.add_argument("--lambda_source_episode", type=float, default=0.0)
    parser.add_argument("--source_episode_start_epoch", type=int, default=1)
    parser.add_argument("--source_episode_warmup_epochs", type=int, default=0)
    parser.add_argument("--source_episode_min_domains", type=int, default=2)
    parser.add_argument("--source_episode_radius_cap_deg", type=float, default=30.0)
    parser.add_argument(
        "--source_episode_radius_mode",
        type=str,
        default="min_three_sigma_core",
        choices=["three_sigma", "core_quantile", "min_three_sigma_core"],
    )
    parser.add_argument("--source_episode_core_quantile", type=float, default=0.80)
    parser.add_argument("--source_episode_min_sigma_deg", type=float, default=3.0)
    parser.add_argument("--source_episode_mixup_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_mixup_hard_k", type=int, default=2)
    parser.add_argument("--lambda_direct_metric_accept", type=float, default=0.0)
    parser.add_argument("--direct_metric_start_epoch", type=int, default=20)
    parser.add_argument("--direct_metric_warmup_epochs", type=int, default=20)
    parser.add_argument("--direct_metric_virtual_count", type=int, default=32)
    parser.add_argument("--direct_metric_virtual_mode", type=str, default="hard", choices=["legacy", "hard", "mixed", "legacy_hard"])
    parser.add_argument("--direct_metric_virtual_detach", type=str2bool, default=True)
    parser.add_argument("--direct_metric_gate_reference_detach", type=str2bool, default=True)
    parser.add_argument("--direct_metric_reference_bank", type=str2bool, default=False)
    parser.add_argument("--direct_metric_reference_refresh_epochs", type=int, default=10)
    parser.add_argument("--direct_metric_reference_per_component", type=int, default=4)
    parser.add_argument("--direct_metric_core_quantile", type=float, default=0.70)
    parser.add_argument("--direct_metric_accept_quantile", type=float, default=0.80)
    parser.add_argument("--direct_metric_tail_quantile", type=float, default=0.90)
    parser.add_argument("--direct_metric_overflow_quantile", type=float, default=0.97)
    parser.add_argument("--direct_metric_zid_p50_target_deg", type=float, default=28.0)
    parser.add_argument("--direct_metric_zid_p95_target_deg", type=float, default=54.0)
    parser.add_argument("--direct_metric_zid_p99_target_deg", type=float, default=70.0)
    parser.add_argument("--direct_metric_zid_tail_cvar_target_deg", type=float, default=56.0)
    parser.add_argument("--direct_metric_source_overflow_target", type=float, default=0.45)
    parser.add_argument("--direct_metric_proxy_vaccept_target", type=float, default=0.35)
    parser.add_argument("--direct_metric_bridge_accept_target", type=float, default=0.25)
    parser.add_argument("--direct_metric_low_density_accept_target", type=float, default=0.12)
    parser.add_argument("--direct_metric_tail_accept_target", type=float, default=0.35)
    parser.add_argument("--direct_metric_overflow_accept_target", type=float, default=0.20)
    parser.add_argument("--direct_metric_radius_inter_ratio_target", type=float, default=0.85)
    parser.add_argument("--direct_metric_core_accept_target", type=float, default=0.82)
    parser.add_argument("--direct_metric_core_tpr_target", type=float, default=0.85)
    parser.add_argument("--direct_metric_known_accept_target", type=float, default=0.65)
    parser.add_argument("--direct_metric_known_tpr_target", type=float, default=0.85)
    parser.add_argument("--direct_metric_sat_pair_target_deg", type=float, default=10.0)
    parser.add_argument("--direct_metric_zid_quantile_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_source_overflow_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_proxy_vaccept_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_bridge_accept_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_low_density_accept_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_tail_accept_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_overflow_accept_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_radius_inter_ratio_weight", type=float, default=1.0)
    parser.add_argument("--direct_metric_global_quantile_weight", type=float, default=0.0)
    parser.add_argument("--direct_metric_component_inter_margin_weight", type=float, default=0.0)
    parser.add_argument("--direct_metric_component_overlap_weight", type=float, default=0.0)
    parser.add_argument("--direct_metric_core_accept_weight", type=float, default=0.25)
    parser.add_argument("--direct_metric_core_tpr_weight", type=float, default=0.0)
    parser.add_argument("--direct_metric_known_coverage_weight", type=float, default=0.0)
    parser.add_argument("--direct_metric_sat_pair_weight", type=float, default=0.0)
    parser.add_argument("--direct_metric_quantile_temperature_deg", type=float, default=3.0)
    parser.add_argument("--direct_metric_accept_temperature", type=float, default=0.04)
    parser.add_argument("--direct_metric_component_temperature_deg", type=float, default=3.0)
    parser.add_argument("--direct_metric_density_temperature_deg", type=float, default=3.0)
    parser.add_argument("--direct_metric_component_margin_deg", type=float, default=4.0)
    parser.add_argument("--direct_metric_component_inter_margin_deg", type=float, default=55.0)
    parser.add_argument("--direct_metric_component_overlap_margin_deg", type=float, default=4.0)
    parser.add_argument("--direct_metric_source_margin_deg", type=float, default=2.0)
    parser.add_argument("--direct_metric_source_radius_cap_deg", type=float, default=0.0)
    parser.add_argument("--direct_metric_shell_width_deg", type=float, default=4.0)
    parser.add_argument("--direct_metric_accept_cvar_alpha", type=float, default=0.25)
    parser.add_argument("--direct_metric_positive_first", type=str2bool, default=False)
    parser.add_argument("--direct_metric_negative_start_tpr", type=float, default=0.75)
    parser.add_argument("--direct_metric_negative_full_tpr", type=float, default=0.85)
    parser.add_argument("--direct_metric_require_effective_negative_grad", type=str2bool, default=False)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--candidate_id", type=str, default="")
    parser.add_argument("--base_candidate", type=str, default="")
    parser.add_argument("--formal_ablation", type=str2bool, default=False)
    parser.add_argument("--ablation_id", type=str, default="")
    parser.add_argument("--git_commit", type=str, default="")
    parser.add_argument("--expected_config_hash", type=str, default="")
    parser.add_argument("--row_key", type=str, default="")
    parser.add_argument("--sealed_plan_sha256", type=str, default="")
    parser.add_argument("--seed_registry_sha256", type=str, default="")
    parser.add_argument("--wisig_pkl_sha256", type=str, default="")
    parser.add_argument("--dataset_receipt_path", type=str, default="")
    parser.add_argument("--dataset_receipt_sha256", type=str, default="")
    parser.add_argument("--environment_receipt_path", type=str, default="")
    parser.add_argument("--environment_receipt_sha256", type=str, default="")
    parser.add_argument("--python_environment_id", type=str, default="")
    parser.add_argument("--reject_head", type=str2bool, default=False)
    parser.add_argument("--reject_class_index", type=int, default=-1)
    parser.add_argument("--lambda_energy_in", type=float, default=0.0)
    parser.add_argument("--lambda_energy_out", type=float, default=0.0)
    parser.add_argument("--lambda_reject_neg", type=float, default=0.0)
    parser.add_argument("--lambda_inter_neg", type=float, default=0.0)
    parser.add_argument("--lambda_shell_neg", type=float, default=0.0)
    parser.add_argument("--lambda_tail_outward_neg", type=float, default=0.0)
    parser.add_argument("--lambda_bridge_neg", type=float, default=0.0)
    parser.add_argument("--neg_shell_ratio", type=float, default=0.0)
    parser.add_argument("--neg_inter_ratio", type=float, default=0.0)
    parser.add_argument("--neg_tail_outward_ratio", type=float, default=0.0)
    parser.add_argument("--neg_bridge_ratio", type=float, default=0.0)
    parser.add_argument("--energy_in_margin", type=float, default=-10.0)
    parser.add_argument("--energy_out_margin", type=float, default=10.0)
    parser.add_argument("--tail_quarantine", type=str2bool, default=False)
    parser.add_argument("--tail_core_quantile", type=float, default=0.80)
    parser.add_argument("--tail_accept_quantile", type=float, default=0.92)
    parser.add_argument("--tail_extreme_quantile", type=float, default=0.95)
    parser.add_argument("--tail_soft_ce_weight", type=float, default=0.25)
    parser.add_argument("--tail_extreme_ce_weight", type=float, default=0.05)
    parser.add_argument("--lambda_tail_cvar", type=float, default=0.0)
    parser.add_argument("--lambda_overflow_cap", type=float, default=0.0)
    parser.add_argument("--unlabeled_risk_buffer", type=str2bool, default=False)
    parser.add_argument("--pseudo_known_requires_density", type=str2bool, default=True)
    parser.add_argument("--pseudo_known_maxprob_min", type=float, default=0.90)
    parser.add_argument("--risk_maxprob_min", type=float, default=0.70)
    parser.add_argument("--risk_density_percentile", type=float, default=10.0)
    parser.add_argument("--risk_geo_margin_min_deg", type=float, default=2.0)
    parser.add_argument("--lambda_risk_energy_out", type=float, default=0.0)
    parser.add_argument("--phase2_export_prototypes", type=str2bool, default=False)
    parser.add_argument("--phase2_export_path", type=str, default="")
    parser.add_argument("--phase2_export_checkpoint", type=str, default="")
    parser.add_argument(
        "--phase2_export_feature_key",
        type=str,
        default="z_id",
        choices=["z_id", "id_feat_joint", "feat_joint", "id_feat_pa", "id_feat_dac"],
    )
    parser.add_argument("--phase2_export_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--phase2_export_max_batches", type=int, default=0)
    parser.add_argument("--phase2_fuse_prototypes", type=str2bool, default=False)
    parser.add_argument("--phase2_fuse_max_components", type=int, default=4)
    parser.add_argument("--phase2_fuse_merge_angle_deg", type=float, default=6.0)
    parser.add_argument("--phase2_fuse_radius_cap_deg", type=float, default=25.0)
    parser.add_argument("--phase2_fuse_tail_abs_deg", type=float, default=30.0)
    parser.add_argument("--phase2_fuse_accept_policy", type=str, default="local_component")
    parser.add_argument("--phase2_fuse_accept_radius_key", type=str, default="p95")
    parser.add_argument("--phase2_fuse_max_p95_increase_deg", type=float, default=2.0)
    parser.add_argument("--phase2_fuse_keep_tail_sentinel", type=str2bool, default=True)
    parser.add_argument("--phase2_fuse_tail_auto_accept", type=str2bool, default=False)
    parser.add_argument("--phase2_fuse_global_ball_accept", type=str2bool, default=False)
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--model_variant", type=str, default="lite_d")
    parser.add_argument(
        "--representation_mode",
        type=str,
        default="dual",
        choices=["dual", "single_parameter_matched"],
    )
    parser.add_argument(
        "--id_feature_key",
        type=str,
        default="feat_joint",
        choices=["feat_joint", "feat_cls", "feat_con", "base"],
    )
    parser.add_argument("--branch_ablation", type=str, default="no_dac")
    parser.add_argument("--domain_branch_ablation", type=str, default="no_stats")
    parser.add_argument("--domain_enhancer", type=str, default="rcn_stats")
    parser.add_argument("--domain_enhancer_strength", type=float, default=0.35)
    parser.add_argument("--use_mixstyle", type=str2bool, default=True)
    parser.add_argument("--mixstyle_p", type=float, default=0.18)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.10)
    parser.add_argument("--mixstyle_eps", type=float, default=1e-6)
    parser.add_argument("--mixstyle_layers", type=str, default="time_down,t1")
    parser.add_argument("--mixstyle_use_domain_label", type=str2bool, default=True)
    parser.add_argument("--mixstyle_mix", type=str, default="same_tx_crossdomain")
    parser.add_argument("--mixstyle_strength", type=float, default=0.70)
    parser.add_argument("--mixstyle_fallback", type=str, default="skip")
    parser.add_argument("--mixstyle_late_start", type=int, default=110)
    parser.add_argument("--mixstyle_late_ramp_epochs", type=int, default=40)
    parser.add_argument("--mixstyle_late_min_p", type=float, default=0.05)
    parser.add_argument("--mixstyle_late_min_strength", type=float, default=0.32)
    parser.add_argument("--mixstyle_stop_epoch", type=int, default=0)
    parser.add_argument("--stage1_epochs", type=int, default=16)
    parser.add_argument("--stage2_epochs", type=int, default=68)
    parser.add_argument("--stage3_ramp_epochs", type=int, default=17)
    parser.add_argument("--late_stable_start", type=int, default=0)
    parser.add_argument("--late_stable_ramp_epochs", type=int, default=12)
    parser.add_argument("--use_aug", type=str2bool, default=True)
    parser.add_argument("--aug_enable_class_signature", type=str2bool, default=False)
    parser.add_argument("--aug_enable_pa_normal", type=str2bool, default=True)
    parser.add_argument("--aug_dac_only_apply_anti_shortcut", type=str2bool, default=False)
    parser.add_argument("--aug_dac_only_apply_channel", type=str2bool, default=False)
    parser.add_argument("--aug_pa_only_apply_anti_shortcut", type=str2bool, default=False)
    parser.add_argument("--aug_pa_only_apply_channel", type=str2bool, default=False)
    parser.add_argument("--aug_dac_pa_apply_anti_shortcut", type=str2bool, default=True)
    parser.add_argument("--aug_dac_pa_apply_channel", type=str2bool, default=True)
    parser.add_argument("--aug_scale_min", type=float, default=0.10)
    parser.add_argument("--aug_scale_max", type=float, default=0.35)
    parser.add_argument("--aug_warmup_epochs", type=int, default=3)
    parser.add_argument("--aug_ramp_epochs", type=int, default=15)
    parser.add_argument("--aug_ramp_curve", type=float, default=1.25)
    parser.add_argument("--aug_p_dac", type=float, default=0.0)
    parser.add_argument("--aug_p_pa", type=float, default=0.14)
    parser.add_argument("--aug_class_sig_mix", type=float, default=0.1)
    parser.add_argument("--aug_p_time_shift", type=float, default=0.35)
    parser.add_argument("--aug_max_time_shift", type=int, default=32)
    parser.add_argument("--aug_p_amp_scale", type=float, default=0.45)
    parser.add_argument("--aug_amp_min", type=float, default=0.90)
    parser.add_argument("--aug_amp_max", type=float, default=1.10)
    parser.add_argument("--aug_p_phase_rot", type=float, default=0.45)
    parser.add_argument("--aug_p_cfo", type=float, default=0.35)
    parser.add_argument("--aug_cfo_max", type=float, default=4e-4)
    parser.add_argument("--aug_p_phase_noise", type=float, default=0.30)
    parser.add_argument("--aug_phase_noise_sigma_max", type=float, default=0.006)
    parser.add_argument("--aug_p_awgn", type=float, default=0.40)
    parser.add_argument("--aug_snr_min_db", type=float, default=20.0)
    parser.add_argument("--aug_snr_max_db", type=float, default=36.0)
    parser.add_argument("--aug_p_multipath", type=float, default=0.18)
    parser.add_argument("--aug_mp_taps_min", type=int, default=2)
    parser.add_argument("--aug_mp_taps_max", type=int, default=4)
    parser.add_argument("--aug_mp_delay_max", type=int, default=4)
    parser.add_argument("--aug_p_dc_offset", type=float, default=0.30)
    parser.add_argument("--aug_dc_offset_max", type=float, default=0.02)
    parser.add_argument("--aug_p_bandedge_taper", type=float, default=0.25)
    parser.add_argument("--aug_taper_alpha_min", type=float, default=0.02)
    parser.add_argument("--aug_taper_alpha_max", type=float, default=0.10)
    parser.add_argument("--aug_dac_jitter_max", type=float, default=0.002)
    parser.add_argument("--aug_dac_poly_a3", type=float, default=0.12)
    parser.add_argument("--aug_dac_poly_a5", type=float, default=0.03)
    parser.add_argument("--aug_dac_iq_img_max", type=float, default=0.04)
    parser.add_argument("--aug_dac_inter_gain_max", type=float, default=0.03)
    parser.add_argument("--aug_dac_inter_off_max", type=float, default=0.008)
    parser.add_argument("--aug_dac_inter_skew_max", type=float, default=0.05)
    parser.add_argument("--aug_dac_dither", type=float, default=0.002)
    parser.add_argument("--aug_dac_inl_warp", type=float, default=0.03)
    parser.add_argument("--aug_dac_spur_amp_max", type=float, default=0.012)
    parser.add_argument("--aug_dac_slew_max", type=float, default=0.18)
    parser.add_argument("--aug_pa_mp_sigma", type=float, default=0.05)
    parser.add_argument("--aug_pa_mem_sigma", type=float, default=0.04)
    parser.add_argument("--aug_pa_ampm_max", type=float, default=0.20)
    parser.add_argument("--aug_pa_iq_img_max", type=float, default=0.02)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    parser.add_argument("--phase1_source_only_eval", type=str2bool, default=False)
    parser.add_argument(
        "--wisig_allow_shared_days_if_receivers_disjoint",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--phase1_realized_rho_tolerance",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--phase1_realized_source_val_tolerance",
        type=float,
        default=0.002,
    )
    add_sat_eval_args(parser)
    return parser


def split_tx_rx_day_1_7_2(
    dataset,
    *,
    labeled_ratio: float = 0.10,
    unlabeled_ratio: float = 0.70,
    source_val_ratio: float = 0.20,
) -> Tuple[List[int], List[int], List[int]]:
    total = float(labeled_ratio) + float(unlabeled_ratio) + float(source_val_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    train_total = float(labeled_ratio) + float(unlabeled_ratio)
    rho_label = float(labeled_ratio) / train_total if train_total > 0.0 else float("inf")
    if not math.isfinite(rho_label) or rho_label > 0.1 + 1e-8:
        raise ValueError(
            "Phase1 weak-label protocol requires rho_label=|L_s|/(|L_s|+|U_s|)<=0.1, "
            f"got {rho_label:.6f}"
        )
    groups: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = defaultdict(list)
    for global_i, item in enumerate(dataset.index):
        key = (int(item.tx_i), int(item.rx_i), int(item.day_i), int(getattr(item, "eq_i", 0)))
        groups[key].append((int(getattr(item, "sig_i", global_i)), int(global_i)))

    labeled: List[int] = []
    unlabeled: List[int] = []
    val: List[int] = []
    for _, pairs in sorted(groups.items()):
        ordered = [idx for _, idx in sorted(pairs, key=lambda z: z[0])]
        n = len(ordered)
        if n == 0:
            continue
        n_l = int(round(n * float(labeled_ratio)))
        n_u = int(round(n * float(unlabeled_ratio)))
        if labeled_ratio > 0 and n_l == 0 and n >= 3:
            n_l = 1
        if unlabeled_ratio > 0 and n_u == 0 and n >= 3:
            n_u = 1
        if n_l + n_u > n:
            n_u = max(0, n - n_l)
        n_v = n - n_l - n_u
        if source_val_ratio > 0 and n_v == 0 and n >= 3:
            if n_u > 1:
                n_u -= 1
                n_v = 1
            elif n_l > 1:
                n_l -= 1
                n_v = 1
        labeled.extend(ordered[:n_l])
        unlabeled.extend(ordered[n_l : n_l + n_u])
        val.extend(ordered[n_l + n_u :])
    realized_train = len(labeled) + len(unlabeled)
    max_labeled = int(math.floor(0.1 * float(realized_train) + 1e-8))
    if len(labeled) > max_labeled:
        grouped_candidates: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
        for idx in sorted(labeled):
            item = dataset.index[idx]
            key = (int(item.tx_i), int(item.rx_i), int(item.day_i), int(getattr(item, "eq_i", 0)))
            grouped_candidates[key].append(int(idx))
        tx_ids = sorted({key[0] for key in grouped_candidates})
        if max_labeled < len(tx_ids):
            raise ValueError(
                "Phase1 rho_label budget is too small to retain one labeled sample per known TX: "
                f"budget={max_labeled} known_tx={len(tx_ids)}"
            )
        tx_queues: Dict[int, List[int]] = {}
        for tx_id in tx_ids:
            keys = sorted(key for key in grouped_candidates if key[0] == tx_id)
            queue: List[int] = []
            depth = 0
            while True:
                added = False
                for key in keys:
                    values = grouped_candidates[key]
                    if depth < len(values):
                        queue.append(values[depth])
                        added = True
                if not added:
                    break
                depth += 1
            tx_queues[tx_id] = queue
        selected: List[int] = []
        cursor = {tx_id: 0 for tx_id in tx_ids}
        while len(selected) < max_labeled:
            added = False
            for tx_id in tx_ids:
                pos = cursor[tx_id]
                if pos < len(tx_queues[tx_id]) and len(selected) < max_labeled:
                    selected.append(tx_queues[tx_id][pos])
                    cursor[tx_id] = pos + 1
                    added = True
            if not added:
                break
        selected_set = set(selected)
        demoted = [idx for idx in labeled if idx not in selected_set]
        labeled = sorted(selected)
        unlabeled.extend(demoted)
    if not labeled:
        raise ValueError("Phase1 weak-label split has no labeled samples after enforcing rho_label<=0.1")
    realized_rho = float(len(labeled)) / float(realized_train) if realized_train > 0 else float("inf")
    if not math.isfinite(realized_rho) or realized_rho > 0.1 + 1e-8:
        raise ValueError(
            "Phase1 realized weak-label split violates rho_label<=0.1 after group rounding: "
            f"L_s={len(labeled)} U_s={len(unlabeled)} rho_label={realized_rho:.6f}"
        )
    return sorted(labeled), sorted(unlabeled), sorted(val)


def _source_validation_record_key(dataset, index: int) -> Tuple[int, int, int, int, int, int]:
    record = dataset.index[int(index)]
    return (
        int(getattr(record, "tx_i", -1)),
        int(getattr(record, "rx_i", -1)),
        int(getattr(record, "day_i", -1)),
        int(getattr(record, "eq_i", -1)),
        int(getattr(record, "sig_i", index)),
        int(getattr(record, "base_index", index)),
    )


def _partition_source_validation_roles(
    dataset,
    validation: Sequence[int],
    *,
    cal_fraction: float,
    min_class_samples: int = 4,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Partition source validation within TX and receiver/day strata."""

    fraction = float(cal_fraction)
    minimum = max(1, int(min_class_samples))
    if not 0.0 < fraction < 1.0:
        raise ValueError("source validation cal_fraction must be in (0,1)")
    by_class: Dict[int, Dict[Tuple[int, int], List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw_index in validation:
        index = int(raw_index)
        tx, rx, day, *_ = _source_validation_record_key(dataset, index)
        by_class[tx][(rx, day)].append(index)
    if not by_class:
        raise ValueError("MISSING_CLASS_IN_V_CAL: source validation pool is empty")

    v_cal: List[int] = []
    v_select: List[int] = []
    for tx in sorted(by_class):
        strata = by_class[tx]
        class_total = sum(len(values) for values in strata.values())
        if class_total < 2 * minimum:
            raise ValueError(
                "INSUFFICIENT_CLASS_SAMPLES: "
                f"class={tx} validation_count={class_total} required={2 * minimum}"
            )
        class_cal: List[int] = []
        class_select: List[int] = []
        singletons: List[int] = []
        for stratum in sorted(strata):
            values = sorted(
                strata[stratum],
                key=lambda index: hashlib.sha256(
                    repr(_source_validation_record_key(dataset, index)).encode("utf-8")
                ).hexdigest(),
            )
            if len(values) == 1:
                singletons.extend(values)
                continue
            cal_count = int(round(len(values) * fraction))
            cal_count = min(len(values) - 1, max(1, cal_count))
            class_cal.extend(values[:cal_count])
            class_select.extend(values[cal_count:])
        for index in sorted(singletons, key=lambda item: _source_validation_record_key(dataset, item)):
            if len(class_cal) <= len(class_select):
                class_cal.append(index)
            else:
                class_select.append(index)
        while len(class_cal) < minimum and len(class_select) > minimum:
            class_cal.append(class_select.pop())
        while len(class_select) < minimum and len(class_cal) > minimum:
            class_select.append(class_cal.pop())
        if len(class_cal) < minimum or len(class_select) < minimum:
            raise ValueError(
                "INSUFFICIENT_CLASS_SAMPLES: "
                f"class={tx} V_cal={len(class_cal)} V_select={len(class_select)} "
                f"required_each={minimum}"
            )
        v_cal.extend(class_cal)
        v_select.extend(class_select)

    cal_set = set(v_cal)
    select_set = set(v_select)
    overlap = cal_set & select_set
    if overlap or cal_set | select_set != {int(index) for index in validation}:
        raise ValueError("Phase1 V_cal/V_select partition lost or duplicated physical indices")

    def per_class(indices: Sequence[int]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for index in indices:
            counts[str(_source_validation_record_key(dataset, int(index))[0])] += 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))

    receipt = {
        "schema": "phase1_source_validation_roles_v2",
        "physical_id_overlap_count": len(overlap),
        "per_role_per_class": {
            "V_cal": per_class(v_cal),
            "V_select": per_class(v_select),
        },
    }
    return sorted(v_cal), sorted(v_select), receipt


def _resolve_unlabeled_batch_size(args) -> int:
    if bool(getattr(args, "strict_pair_concat", False)):
        return 32
    if not bool(getattr(args, "use_muse_ssdg", False)):
        return int(args.batch_size)
    value = int(getattr(args, "muse_unlabeled_batch_size", 256))
    if value < 1:
        raise ValueError("muse_unlabeled_batch_size must be positive")
    return value


def _unlabeled_drop_last(args) -> bool:
    if bool(getattr(args, "strict_pair_concat", False)):
        return True
    return not bool(getattr(args, "use_muse_ssdg", False))


def split_tx_rx_day_1_7_2_roles(
    dataset,
    *,
    labeled_ratio: float = 0.07,
    unlabeled_ratio: float = 0.63,
    source_cal_ratio: float = 0.15,
    source_select_ratio: float = 0.15,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build the current Phase1 L_s/U_s/V_cal/V_select source split.

    The legacy grouped splitter remains the allocator for L_s/U_s and the
    combined validation pool. The combined pool is then deterministically
    partitioned into two disjoint validation roles; V_select is the only
    selection input and V_cal is reserved for calibration/export.
    """
    ratios = (
        float(labeled_ratio),
        float(unlabeled_ratio),
        float(source_cal_ratio),
        float(source_select_ratio),
    )
    if any(value <= 0.0 for value in ratios):
        raise ValueError("Phase1 source role ratios must all be positive")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Phase1 source role ratios must sum to 1.0, got {sum(ratios)}")
    train_total = ratios[0] + ratios[1]
    rho_label = ratios[0] / train_total if train_total > 0.0 else float("inf")
    if not math.isfinite(rho_label) or rho_label > 0.1 + 1e-8:
        raise ValueError(
            "Phase1 weak-label protocol requires rho_label=|L_s|/(|L_s|+|U_s|)<=0.1, "
            f"got {rho_label:.6f}"
        )
    labeled, unlabeled, validation = split_tx_rx_day_1_7_2(
        dataset,
        labeled_ratio=ratios[0],
        unlabeled_ratio=ratios[1],
        source_val_ratio=ratios[2] + ratios[3],
    )
    validation_total = len(validation)
    if validation_total < 2:
        raise ValueError("Phase1 current source-role split requires both V_cal and V_select")
    cal_fraction = ratios[2] / (ratios[2] + ratios[3])
    v_cal, v_select, _ = _partition_source_validation_roles(
        dataset,
        validation,
        cal_fraction=cal_fraction,
        min_class_samples=4,
    )
    if set(v_cal) & set(v_select):
        raise ValueError("Phase1 V_cal and V_select physical indices overlap")
    return sorted(labeled), sorted(unlabeled), v_cal, v_select


def split_tx_rx_day_1_6_3(
    dataset,
    *,
    labeled_ratio: float = 0.10,
    unlabeled_ratio: float = 0.60,
    source_val_ratio: float = 0.30,
) -> Tuple[List[int], List[int], List[int]]:
    return split_tx_rx_day_1_7_2(
        dataset,
        labeled_ratio=labeled_ratio,
        unlabeled_ratio=unlabeled_ratio,
        source_val_ratio=source_val_ratio,
    )


def _as_plain_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def temporal_neighbor_agreement_mask(
    pseudo: Sequence[Any],
    conf: Sequence[Any],
    meta: Mapping[str, Any] | None,
    *,
    window: int = 1,
    min_conf: float = 0.0,
) -> List[bool]:
    pseudo_l = [int(v) for v in _as_plain_list(pseudo)]
    conf_l = [float(v) for v in _as_plain_list(conf)]
    n = min(len(pseudo_l), len(conf_l))
    if meta is None or n == 0:
        return [False for _ in range(n)]

    rx_l = _as_plain_list(meta.get("rx_i"))
    day_l = _as_plain_list(meta.get("day_i"))
    eq_l = _as_plain_list(meta.get("eq_i"))
    sig_l = _as_plain_list(meta.get("sig_i"))
    order_l = _as_plain_list(meta.get("base_index"))
    if len(order_l) < n:
        order_l = sig_l
    if min(len(rx_l), len(day_l), len(eq_l), len(sig_l)) < n:
        return [False for _ in range(n)]

    streams: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(n):
        key = (int(rx_l[i]), int(day_l[i]), int(eq_l[i]))
        streams[key].append(i)

    mask = [False for _ in range(n)]
    max_gap = max(1, int(window))
    for indices in streams.values():
        for i in indices:
            if conf_l[i] < float(min_conf):
                continue
            sig_i = int(sig_l[i])
            for j in indices:
                if i == j or conf_l[j] < float(min_conf):
                    continue
                order_gap = abs(int(order_l[i]) - int(order_l[j]))
                sig_gap = abs(sig_i - int(sig_l[j]))
                if pseudo_l[i] == pseudo_l[j] and sig_gap <= max_gap and order_gap <= max_gap:
                    mask[i] = True
                    break
    return mask


def _meta_from_extra(extra) -> Mapping[str, Any] | None:
    if extra is None or len(extra) < 2:
        return None
    meta = extra[1]
    return meta if isinstance(meta, Mapping) else None


def _metadata_label_tensor(extra, key: str, device, expected_count: int) -> Optional[torch.Tensor]:
    meta = _meta_from_extra(extra)
    if meta is None or key not in meta:
        return None
    value = meta.get(key)
    if torch.is_tensor(value):
        out = value.detach().to(device=device).view(-1).long()
    else:
        try:
            out = torch.as_tensor(_as_plain_list(value), device=device, dtype=torch.long).view(-1)
        except Exception:
            return None
    if out.numel() != int(expected_count):
        return None
    return out


def _expand_view_metadata(
    labels: Optional[torch.Tensor],
    *,
    clean_count: int,
    total_count: int,
) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    if labels.numel() == total_count:
        return labels
    if labels.numel() == clean_count and total_count == 2 * clean_count:
        return torch.cat([labels, labels], dim=0)
    return None


def _channel_view_labels(total_count: int, clean_count: int, applied: bool, device) -> torch.Tensor:
    labels = torch.zeros(max(0, int(total_count)), device=device, dtype=torch.long)
    if bool(applied) and int(clean_count) > 0 and int(total_count) >= 2 * int(clean_count):
        labels[int(clean_count) : 2 * int(clean_count)] = 1
    return labels


def _labeled_channel_pair_invariance_loss(
    clean_z_id: torch.Tensor,
    satellite_z_id: torch.Tensor,
    tx_labels: torch.Tensor,
    *,
    channel_weight: float,
    channel_pair_weight: float,
    min_groups: int,
    min_samples_per_group: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align a labeled clean/satellite pair without widening the CE-only batch.

    The CE-only satellite path intentionally keeps the clean batch as the
    main forward pass. This helper builds the smallest temporary pair view
    only for the TX-conditioned ``z_id`` geometry objective, so satellite
    features cannot enter receiver/day/domain losses or be counted twice by
    TX cross-entropy.
    """

    if clean_z_id.dim() != 2 or satellite_z_id.dim() != 2:
        raise ValueError("clean and satellite z_id features must both be rank-2 tensors")
    if clean_z_id.size(0) != satellite_z_id.size(0):
        raise ValueError("clean and satellite z_id batches must have the same size")
    labels = tx_labels.view(-1).long()
    if labels.numel() != clean_z_id.size(0):
        raise ValueError("TX labels must align with the clean/satellite z_id pair")
    if float(channel_weight) <= 0.0 or float(channel_pair_weight) <= 0.0 or labels.numel() == 0:
        zero = clean_z_id.sum() * 0.0
        return zero, {
            "active": 0.0,
            "channel_active": 0.0,
            "channel_pair_active": 0.0,
            "channel_loss": 0.0,
            "channel_pair_loss": 0.0,
            "channel_pair_count": 0.0,
        }

    pair_count = int(labels.numel())
    pair_z_id = torch.cat([clean_z_id, satellite_z_id], dim=0)
    pair_labels = torch.cat([labels, labels], dim=0)
    pair_channels = torch.cat(
        [
            torch.zeros(pair_count, device=labels.device, dtype=torch.long),
            torch.ones(pair_count, device=labels.device, dtype=torch.long),
        ],
        dim=0,
    )
    loss, metrics = tx_conditional_domain_invariance_loss(
        pair_z_id,
        pair_labels,
        channel_labels=pair_channels,
        channel_weight=float(channel_weight),
        channel_pair_weight=float(channel_pair_weight),
        paired_view_count=pair_count,
        min_groups=int(min_groups),
        min_samples_per_group=int(min_samples_per_group),
    )
    metrics = dict(metrics)
    metrics["channel_pair_active"] = 1.0
    metrics["channel_active"] = max(float(metrics.get("channel_active", 0.0)), 1.0)
    metrics["active"] = max(float(metrics.get("active", 0.0)), 1.0)
    return loss, metrics


class FrozenDirectMetricReferenceBank:
    """Window-frozen TX/domain/view anchors for direct acceptance geometry."""

    def __init__(self, *, per_component: int = 4, refresh_epochs: int = 10):
        self.per_component = max(1, int(per_component))
        self.refresh_epochs = max(1, int(refresh_epochs))
        self.active: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.pending: Dict[Tuple[int, int, int], List[torch.Tensor]] = defaultdict(list)
        self.pending_seen: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.active_epoch = -1
        self.version = 0

    def maybe_promote(self, epoch: int) -> bool:
        should_promote = int(epoch) >= 2 and (int(epoch) - 2) % self.refresh_epochs == 0
        if not should_promote or not self.pending:
            return False
        self.active = {
            key: torch.stack(values, dim=0).detach()
            for key, values in self.pending.items()
            if values
        }
        self.pending = defaultdict(list)
        self.pending_seen = defaultdict(int)
        self.active_epoch = int(epoch)
        self.version += 1
        return True

    def observe(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        d: Optional[torch.Tensor],
        *,
        view: int,
    ) -> None:
        if z is None or not torch.is_tensor(z) or z.dim() != 2 or z.size(0) == 0:
            return
        labels = y.detach().view(-1).long()
        domains = (
            d.detach().view(-1).long()
            if d is not None and torch.is_tensor(d) and d.numel() == labels.numel()
            else torch.full_like(labels, -1)
        )
        features = F.normalize(
            torch.nan_to_num(z.detach().float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=1,
        )
        for index in range(features.size(0)):
            label = int(labels[index].item())
            if label < 0:
                continue
            key = (label, int(domains[index].item()), int(view))
            seen = int(self.pending_seen[key])
            slot = seen % self.per_component
            value = features[index].clone()
            if len(self.pending[key]) < self.per_component:
                self.pending[key].append(value)
            else:
                self.pending[key][slot] = value
            self.pending_seen[key] = seen + 1

    def tensors(self, *, view: Optional[int] = None):
        selected = [
            (key, values)
            for key, values in sorted(self.active.items())
            if view is None or int(key[2]) == int(view)
        ]
        if not selected:
            return None, None, None
        z_parts = []
        y_parts = []
        d_parts = []
        for (label, domain, _), values in selected:
            z_parts.append(values)
            y_parts.append(torch.full((values.size(0),), int(label), device=values.device, dtype=torch.long))
            d_parts.append(torch.full((values.size(0),), int(domain), device=values.device, dtype=torch.long))
        return torch.cat(z_parts, dim=0), torch.cat(y_parts, dim=0), torch.cat(d_parts, dim=0)

    def paired_tensors(self):
        """Return clean/satellite anchors with identical label/domain ordering."""

        clean_parts = []
        sat_parts = []
        y_parts = []
        d_parts = []
        base_keys = sorted({(key[0], key[1]) for key in self.active})
        for label, domain in base_keys:
            clean = self.active.get((label, domain, 0))
            sat = self.active.get((label, domain, 1))
            if clean is None or sat is None:
                continue
            count = min(int(clean.size(0)), int(sat.size(0)))
            if count <= 0:
                continue
            clean_parts.append(clean[:count])
            sat_parts.append(sat[:count])
            y_parts.append(torch.full((count,), int(label), device=clean.device, dtype=torch.long))
            d_parts.append(torch.full((count,), int(domain), device=clean.device, dtype=torch.long))
        if not clean_parts:
            return None, None, None, None
        return (
            torch.cat(clean_parts, dim=0),
            torch.cat(sat_parts, dim=0),
            torch.cat(y_parts, dim=0),
            torch.cat(d_parts, dim=0),
        )

    @property
    def anchor_count(self) -> int:
        return sum(int(values.size(0)) for values in self.active.values())


def _temporal_mask_tensor(pseudo, conf, extra, args, device):
    meta = _meta_from_extra(extra)
    mask = temporal_neighbor_agreement_mask(
        pseudo,
        conf,
        meta,
        window=int(args.pseudo_temporal_window),
        min_conf=float(args.pseudo_temporal_min_conf),
    )
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _temporal_bank_mask_tensor(
    pseudo,
    conf,
    extra,
    args,
    device,
    *,
    epoch: int,
    bank: Dict[Tuple[int, int, int, int, int], Tuple[int, int, int]],
):
    """Require cross-epoch prediction stability without relying on batch neighbors."""

    meta = _meta_from_extra(extra)
    pseudo_l = [int(v) for v in _as_plain_list(pseudo)]
    conf_l = [float(v) for v in _as_plain_list(conf)]
    n = min(len(pseudo_l), len(conf_l))
    if meta is None or n == 0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    fields = {
        key: _as_plain_list(meta.get(key))
        for key in ("rx_i", "day_i", "eq_i", "sig_i", "base_index")
    }
    if len(fields["base_index"]) < n:
        fields["base_index"] = fields["sig_i"]
    if any(len(values) < n for values in fields.values()):
        return torch.zeros(n, dtype=torch.bool, device=device)
    min_conf = float(args.pseudo_temporal_min_conf)
    min_streak = max(1, int(getattr(args, "pseudo_temporal_bank_min_streak", 2)))
    passed = []
    for index in range(n):
        key = tuple(int(fields[name][index]) for name in ("rx_i", "day_i", "eq_i", "sig_i", "base_index"))
        pred = int(pseudo_l[index])
        previous = bank.get(key)
        streak = 1
        if previous is not None:
            previous_pred, previous_epoch, previous_streak = previous
            if previous_pred == pred and int(previous_epoch) == int(epoch) - 1:
                streak = int(previous_streak) + 1
            elif previous_pred == pred and int(previous_epoch) == int(epoch):
                streak = int(previous_streak)
        if conf_l[index] >= min_conf:
            bank[key] = (pred, int(epoch), streak)
        passed.append(conf_l[index] >= min_conf and streak >= min_streak)
    return torch.as_tensor(passed, dtype=torch.bool, device=device)


def _update_ema_model(ema_model, model, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(float(decay)).add_(p.data, alpha=1.0 - float(decay))
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def _resolve_epoch_schedule(args) -> int:
    """Resolve label/pseudo epochs, treating --epochs as total epochs."""
    if int(args.epochs) > 0:
        total_epochs = max(0, int(args.epochs))
        label_epochs = min(max(0, int(args.label_epochs)), total_epochs)
        args.label_epochs = label_epochs
        args.pseudo_epochs = max(0, total_epochs - label_epochs)
        return total_epochs
    args.label_epochs = max(0, int(args.label_epochs))
    args.pseudo_epochs = max(0, int(args.pseudo_epochs))
    return int(args.label_epochs) + int(args.pseudo_epochs)


def _apply_model_cli_args(model_args, args):
    for key in (
        "model_size",
        "model_variant",
        "representation_mode",
        "id_feature_key",
        "branch_ablation",
        "domain_branch_ablation",
        "domain_enhancer",
        "domain_enhancer_strength",
        "use_mixstyle",
        "mixstyle_p",
        "mixstyle_alpha",
        "mixstyle_eps",
        "mixstyle_layers",
        "mixstyle_use_domain_label",
        "mixstyle_mix",
        "mixstyle_strength",
        "mixstyle_fallback",
        "use_crra",
        "crra_rank",
        "crra_alpha_max",
        "crra_shrinkage",
        "crra_condition_dim",
        "crra_nuisance_dim",
        "crra_start_epoch",
        "crra_ramp_epochs",
        "sat_anchor_adapter",
        "sat_anchor_adapter_rank",
    ):
        if hasattr(args, key):
            setattr(model_args, key, getattr(args, key))
    return model_args


def _build_source_split_receipt(
    *,
    seed: int,
    split_mode: str,
    source_days: Sequence[Any],
    target_days: Sequence[Any],
    source_receivers: Sequence[Any],
    target_receivers: Sequence[Any],
    labeled_indices: Sequence[Any],
    unlabeled_indices: Sequence[Any],
    source_validation_indices: Sequence[Any],
    wisig_pkl_sha256: str,
    source_calibration_indices: Sequence[Any] | None = None,
    source_selection_indices: Sequence[Any] | None = None,
    requested_labeled_ratio: float | None = None,
    requested_unlabeled_ratio: float | None = None,
    requested_source_val_ratio: float | None = None,
    requested_source_cal_ratio: float | None = None,
    requested_source_select_ratio: float | None = None,
    source_role_protocol: str = "legacy_l_u_v",
    realized_rho_tolerance: float = 0.002,
    realized_source_val_tolerance: float = 0.002,
) -> Dict[str, Any]:
    labeled_size = int(len(labeled_indices))
    unlabeled_size = int(len(unlabeled_indices))
    source_calibration_indices = (
        list(source_calibration_indices)
        if source_calibration_indices is not None
        else list(source_validation_indices)
    )
    source_selection_indices = (
        list(source_selection_indices)
        if source_selection_indices is not None
        else []
    )
    source_calibration_size = int(len(source_calibration_indices))
    source_selection_size = int(len(source_selection_indices))
    source_validation_size = int(len(source_validation_indices))
    train_size = labeled_size + unlabeled_size
    role_validation_size = (
        source_calibration_size + source_selection_size
        if str(source_role_protocol) == "l_s_u_s_v_cal_v_select"
        else source_validation_size
    )
    source_pool_size = train_size + role_validation_size
    realized_rho = float(labeled_size) / float(max(1, train_size))
    realized_source_val_fraction = float(role_validation_size) / float(
        max(1, source_pool_size)
    )
    if requested_labeled_ratio is None:
        requested_labeled_ratio = float(labeled_size) / float(
            max(1, source_pool_size)
        )
    if requested_unlabeled_ratio is None:
        requested_unlabeled_ratio = float(unlabeled_size) / float(
            max(1, source_pool_size)
        )
    if requested_source_val_ratio is None:
        requested_source_val_ratio = realized_source_val_fraction
    requested_train_fraction = float(requested_labeled_ratio) + float(
        requested_unlabeled_ratio
    )
    requested_rho = float(requested_labeled_ratio) / float(
        max(1e-12, requested_train_fraction)
    )
    receipt = {
        "schema": "cvs.phase1.source_split_receipt.v1",
        "seed": int(seed),
        "split_mode": str(split_mode),
        "wisig_pkl_sha256": str(wisig_pkl_sha256),
        "source_days": sorted(str(value) for value in source_days),
        "target_days": sorted(str(value) for value in target_days),
        "source_receivers": sorted(
            str(value) for value in source_receivers
        ),
        "target_receivers": sorted(
            str(value) for value in target_receivers
        ),
        "source_target_receiver_overlap_count": len(
            {str(value) for value in source_receivers}
            & {str(value) for value in target_receivers}
        ),
        "labeled_indices_sha256": _canonical_json_sha256(
            [int(value) for value in labeled_indices]
        ),
        "unlabeled_indices_sha256": _canonical_json_sha256(
            [int(value) for value in unlabeled_indices]
        ),
        "source_validation_indices_sha256": _canonical_json_sha256(
            [int(value) for value in source_validation_indices]
        ),
        "source_calibration_indices_sha256": _canonical_json_sha256(
            [int(value) for value in source_calibration_indices]
        ),
        "source_selection_indices_sha256": _canonical_json_sha256(
            [int(value) for value in source_selection_indices]
        ),
        "labeled_size": labeled_size,
        "unlabeled_size": unlabeled_size,
        "source_validation_size": source_validation_size,
        "source_role_validation_size": role_validation_size,
        "source_calibration_size": source_calibration_size,
        "source_selection_size": source_selection_size,
        "source_role_protocol": str(source_role_protocol),
        "source_pool_size": source_pool_size,
        "requested_labeled_ratio": float(requested_labeled_ratio),
        "requested_unlabeled_ratio": float(requested_unlabeled_ratio),
        "requested_source_val_ratio": float(requested_source_val_ratio),
        "requested_source_cal_ratio": (
            float(requested_source_cal_ratio)
            if requested_source_cal_ratio is not None
            else float(source_calibration_size) / float(max(1, source_pool_size))
        ),
        "requested_source_select_ratio": (
            float(requested_source_select_ratio)
            if requested_source_select_ratio is not None
            else float(source_selection_size) / float(max(1, source_pool_size))
        ),
        "requested_rho_label": requested_rho,
        "realized_rho_label": realized_rho,
        "realized_source_val_fraction": realized_source_val_fraction,
        "realized_rho_tolerance": float(realized_rho_tolerance),
        "realized_source_val_tolerance": float(
            realized_source_val_tolerance
        ),
        "realized_rho_within_tolerance": (
            abs(realized_rho - requested_rho)
            <= float(realized_rho_tolerance) + 1e-12
        ),
        "realized_source_val_within_tolerance": (
            abs(
                realized_source_val_fraction
                - float(requested_source_val_ratio)
            )
            <= float(realized_source_val_tolerance) + 1e-12
        ),
    }
    receipt["split_manifest_sha256"] = _canonical_json_sha256(
        receipt
    )
    return receipt


def _build_ssdg_wisig_data(args, device: torch.device):
    ds_w = load_wisig_compact_pkl(args.wisig_pkl)
    infer_nc = len(ds_w.get("tx_list", []))
    if infer_nc > 0:
        args.num_classes = infer_nc
    eq = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
    day_list = list(ds_w.get("capture_date_list", []))
    rx_list = list(ds_w.get("rx_list", []))
    source_only_eval = bool(getattr(args, "phase1_source_only_eval", False))
    train_days = _resolve_days(day_list, parse_csv_indices(args.wisig_train_days), list(range(min(3, len(day_list)))))
    train_rxs = _resolve_rxs(rx_list, parse_csv_indices(args.wisig_train_rxs), list(range(len(rx_list))))
    test_days = [] if source_only_eval else _resolve_days(
        day_list, parse_csv_indices(args.wisig_test_days), [len(day_list) - 1]
    )
    test_rxs = [] if source_only_eval else _resolve_rxs(
        rx_list, parse_csv_indices(args.wisig_test_rxs), []
    )
    shared_days = sorted(set(train_days).intersection(test_days))
    receiver_overlap = sorted(set(train_rxs).intersection(test_rxs))
    allow_shared_days = bool(
        getattr(args, "wisig_allow_shared_days_if_receivers_disjoint", False)
    )
    if allow_shared_days and shared_days:
        if receiver_overlap:
            raise ValueError(
                "shared train/test days require disjoint receiver sets; "
                f"overlap={receiver_overlap}"
            )
    else:
        train_days = [d for d in train_days if d not in test_days]
        train_rxs = [r for r in train_rxs if r not in test_rxs]

    source_base = WiSigCompactDataset(
        ds_w,
        out_len=int(args.wisig_out_len),
        crop_mode="center",
        normalize=True,
        equalized=eq,
        day_keep=train_days,
        rx_keep=train_rxs,
        domain=str(args.wisig_domain),
        max_samples_per_combo=None if int(args.wisig_max_day123_per_combo) <= 0 else int(args.wisig_max_day123_per_combo),
        seed=int(args.seed),
        build_index=True,
    )
    source_role_protocol = str(
        getattr(args, "phase1_source_role_protocol", "legacy_l_u_v")
    ).strip().lower()
    source_cal_idx: List[int] = []
    source_select_idx: List[int] = []
    if source_role_protocol == "l_s_u_s_v_cal_v_select":
        if str(args.split_mode) != "tx_rx_day_1_7_2":
            raise ValueError(
                "current Phase1 L_s/U_s/V_cal/V_select protocol requires split_mode=tx_rx_day_1_7_2"
            )
        labeled_idx, unlabeled_idx, source_cal_idx, source_select_idx = split_tx_rx_day_1_7_2_roles(
            source_base,
            labeled_ratio=float(args.labeled_ratio),
            unlabeled_ratio=float(args.unlabeled_ratio),
            source_cal_ratio=float(args.source_cal_ratio),
            source_select_ratio=float(args.source_select_ratio),
        )
        val_idx = list(source_select_idx)
    else:
        split_fn = split_tx_rx_day_1_6_3 if str(args.split_mode) == "tx_rx_day_1_6_3" else split_tx_rx_day_1_7_2
        labeled_idx, unlabeled_idx, val_idx = split_fn(
            source_base,
            labeled_ratio=float(args.labeled_ratio),
            unlabeled_ratio=float(args.unlabeled_ratio),
            source_val_ratio=float(args.source_val_ratio),
        )
        source_cal_idx = list(val_idx)
        source_select_idx = []
    labeled_ds = WiSigSubsetDataset(source_base, labeled_idx, split_source="ssdg_labeled_tx_visible")
    unlabeled_ds = WiSigSubsetDataset(source_base, unlabeled_idx, split_source="ssdg_unlabeled_tx_hidden")
    if bool(getattr(args, "use_muse_ssdg", False)) and _muse_level_capabilities(
        getattr(args, "muse_level", "M0")
    )["base"]:
        unlabeled_ds = _MUSEUnlabeledDatasetView(unlabeled_ds)
    cal_ds = WiSigSubsetDataset(source_base, source_cal_idx, split_source="ssdg_source_v_cal")
    val_ds = WiSigSubsetDataset(source_base, val_idx, split_source="ssdg_source_v_select")

    if source_only_eval:
        named_tests = {"source_v_select": val_ds}
        named_meta = {
            "source_v_select": {
                "days_idx": list(train_days),
                "days_label": [day_list[index] for index in train_days],
                "rxs_idx": list(train_rxs),
                "rxs_label": [rx_list[index] for index in train_rxs],
                "size": len(val_ds),
                "phase1_role": "V_select",
                "target_access": False,
            }
        }
        test_split_info = {
            "mode": "phase1_source_only_v_select",
            "train_days_idx": list(train_days),
            "train_days_label": [day_list[index] for index in train_days],
            "test_days_idx": [],
            "test_days_label": [],
            "train_rxs_idx": list(train_rxs),
            "train_rxs_label": [rx_list[index] for index in train_rxs],
            "test_rxs_idx": [],
            "test_rxs_label": [],
            "target_access": False,
            "named_test_sizes": {"source_v_select": len(val_ds)},
        }
    else:
        _, _, _, named_tests, named_meta, test_split_info = make_wisig_trainval_test_by_day_rx(
            ds_w,
            equalized=eq,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            train_ratio=0.5,
            guard_gap=int(args.wisig_guard_gap),
            train_days=train_days,
            test_days=test_days,
            train_rxs=train_rxs,
            test_rxs=test_rxs,
            allow_shared_days_if_receivers_disjoint=allow_shared_days,
            max_samples_per_combo_test=None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo),
            seed=int(args.seed),
        )
    balanced_sampler = None
    if bool(getattr(args, "use_tx_rx_balanced_sampler", False)):
        if BalancedTxDomainBatchSampler is None or DataLoader is None:
            raise ImportError("BalancedTxDomainBatchSampler and torch DataLoader are required")
        balanced_sampler = BalancedTxDomainBatchSampler(
            labeled_ds,
            tx_per_batch=int(args.balanced_sampler_tx_per_batch),
            domain_per_batch=int(args.balanced_sampler_domain_per_batch),
            samples_per_tx_domain=int(args.balanced_sampler_samples_per_cell),
            replacement=bool(args.balanced_sampler_replacement),
            seed=int(args.seed),
            domain_key="rx_day",
            drop_last=True,
        )
        loader_kwargs = {
            "batch_sampler": balanced_sampler,
            "num_workers": int(args.num_workers),
            "pin_memory": device.type == "cuda",
            "persistent_workers": int(args.num_workers) > 0,
        }
        if int(args.num_workers) > 0:
            loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        labeled_loader = DataLoader(labeled_ds, **loader_kwargs)
    else:
        labeled_loader = make_loader(
            labeled_ds,
            int(args.batch_size),
            True,
            int(args.num_workers),
            device,
            True,
            int(args.prefetch_factor),
        )
    probe_train_loader = make_loader(
        labeled_ds,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        device,
        False,
        int(args.prefetch_factor),
    )
    unlabeled_loader = make_loader(
        unlabeled_ds,
        _resolve_unlabeled_batch_size(args),
        bool(getattr(args, "u_unlabeled_shuffle", True)),
        int(args.num_workers),
        device,
        _unlabeled_drop_last(args),
        int(args.prefetch_factor),
    )
    source_cal_loader = make_loader(
        cal_ds,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        device,
        False,
        int(args.prefetch_factor),
    )
    val_loader = make_loader(
        val_ds,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        device,
        False,
        int(args.prefetch_factor),
    )
    named_test_loaders = {
        name: make_loader(ds, int(args.eval_batch_size), False, int(args.num_workers), device, False, int(args.prefetch_factor))
        for name, ds in named_tests.items()
    }
    domain_label_map = build_domain_label_map(source_base)
    split_receipt = _build_source_split_receipt(
        seed=int(args.seed),
        split_mode=str(args.split_mode),
        source_days=train_days,
        target_days=test_days,
        source_receivers=train_rxs,
        target_receivers=test_rxs,
        labeled_indices=labeled_idx,
        unlabeled_indices=unlabeled_idx,
        source_validation_indices=val_idx,
        wisig_pkl_sha256=str(args.wisig_pkl_sha256),
        source_calibration_indices=source_cal_idx,
        source_selection_indices=source_select_idx,
        requested_labeled_ratio=float(args.labeled_ratio),
        requested_unlabeled_ratio=float(args.unlabeled_ratio),
        requested_source_val_ratio=float(args.source_val_ratio),
        requested_source_cal_ratio=float(getattr(args, "source_cal_ratio", 0.0)),
        requested_source_select_ratio=float(getattr(args, "source_select_ratio", 0.0)),
        source_role_protocol=source_role_protocol,
        realized_rho_tolerance=float(
            args.phase1_realized_rho_tolerance
        ),
        realized_source_val_tolerance=float(
            args.phase1_realized_source_val_tolerance
        ),
    )
    return {
        "train_loader": labeled_loader,
        "balanced_train_sampler": balanced_sampler,
        "probe_train_loader": probe_train_loader,
        "unlabeled_loader": unlabeled_loader,
        "source_calibration_loader": source_cal_loader,
        "val_loader": val_loader,
        "named_test_loaders": named_test_loaders,
        "domain_label_map": domain_label_map,
        "num_domains": max(1, len(domain_label_map)),
        "input_len": int(args.wisig_out_len),
        "class_id_to_tx": [str(value) for value in list(getattr(source_base, "tx_list", []) or [])],
        "source_receiver_indices": [int(value) for value in train_rxs],
        "source_day_indices": [int(value) for value in train_days],
        "split_info": {
            "mode": str(args.split_mode),
            "labeled_size": len(labeled_ds),
            "unlabeled_size": len(unlabeled_ds),
            "source_val_size": len(val_ds),
            "source_calibration_size": len(cal_ds),
            "source_selection_size": len(val_ds),
            "source_role_protocol": source_role_protocol,
            "source_role_ratios": {
                "L_s": float(args.labeled_ratio),
                "U_s": float(args.unlabeled_ratio),
                "V_cal": float(getattr(args, "source_cal_ratio", 0.0)),
                "V_select": float(getattr(args, "source_select_ratio", 0.0)),
            },
            "rho_label": float(len(labeled_ds)) / float(max(1, len(labeled_ds) + len(unlabeled_ds))),
            "balanced_sampler_active": bool(balanced_sampler is not None),
            "balanced_sampler_batch_size": int(balanced_sampler.batch_size) if balanced_sampler is not None else int(args.batch_size),
            "target_access": not source_only_eval,
            "test": test_split_info,
            "named_test_meta": named_meta,
            "source_split_receipt": split_receipt,
        },
    }


def _strong_augment(x: torch.Tensor, std: float) -> torch.Tensor:
    if float(std) <= 0:
        return x
    noise = torch.randn_like(x) * float(std)
    return torch.nan_to_num(x + noise, nan=0.0, posinf=0.0, neginf=0.0)


def _should_run_source_val_heavy_eval(epoch: int, total_epochs: int, args: Any) -> bool:
    """Schedule expensive source-val geometry/satellite passes without weakening final evidence."""

    epoch = int(epoch)
    total_epochs = int(total_epochs)
    if epoch >= total_epochs:
        return True
    start_epoch = max(1, int(getattr(args, "source_val_heavy_eval_start_epoch", 1)))
    if epoch < start_epoch:
        return False
    final_window = max(0, int(getattr(args, "source_val_heavy_eval_final_window", 0)))
    if final_window > 0 and epoch > total_epochs - final_window:
        final_interval = int(getattr(args, "source_val_heavy_eval_final_interval", 1))
        return final_interval > 0 and epoch % final_interval == 0
    interval = int(getattr(args, "source_val_heavy_eval_interval", 1))
    return interval > 0 and epoch % interval == 0


def _threshold_mask(conf: torch.Tensor, domains: torch.Tensor | None, args) -> torch.Tensor:
    tau_min = float(args.tau_min)
    tau_max = float(args.tau_max)
    if str(args.pseudo_threshold_mode) == "global" or domains is None:
        return conf >= tau_min
    mask = torch.zeros_like(conf, dtype=torch.bool)
    for domain in domains.unique():
        idx = domains == domain
        if not bool(idx.any()):
            continue
        q = torch.quantile(conf[idx].float(), float(args.pseudo_quantile)).clamp(tau_min, tau_max)
        mask[idx] = conf[idx] >= q
    return mask


def _evaluate(model, data_ctx, device, max_batches: int):
    val = evaluate_loader(model, data_ctx["val_loader"], device, data_ctx["domain_label_map"], max_batches=max_batches)
    named = evaluate_named_loaders(model, data_ctx["named_test_loaders"], device, data_ctx["domain_label_map"], max_batches=max_batches)
    return val, named


def _aggregate_main_test(named_stats: Mapping[str, Mapping[str, Any]], dataset: str) -> Dict[str, float]:
    if str(dataset).lower() == "wisig":
        keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    else:
        keys = list(named_stats.keys())
    return aggregate_named_stats(dict(named_stats), keys)


def _satellite_tx_scores(sat_test_stats: Mapping[str, Mapping[str, Any]]) -> List[float]:
    scores: List[float] = []
    for stats in (sat_test_stats or {}).values():
        agg = stats.get("aggregate", {}) if isinstance(stats, Mapping) else {}
        try:
            scores.append(float(agg.get("tx_acc")))
        except Exception:
            continue
    return scores


def _best_score(
    val_stats: Mapping[str, Any],
    test_stats: Mapping[str, Any],
    sat_test_stats: Mapping[str, Mapping[str, Any]],
    metric: str,
    named_test_stats: Mapping[str, Mapping[str, Any]] | None = None,
    args: Any | None = None,
) -> float:
    metric = str(metric or "clean_val_tx")
    if metric == "clean_val_tx":
        return float(val_stats.get("tx_acc", float("-inf")))
    if metric == "test_overall_tx":
        return float(test_stats.get("tx_acc", float("-inf")))
    if metric == "joint_safe":
        if protected_metric_snapshot is None or joint_safe_score is None:
            raise ImportError("cvsrffi.ssdg_guard is required for --best_metric joint_safe.")
        protected = protected_metric_snapshot(
            val_stats=val_stats,
            test_stats=test_stats,
            named_test_stats=named_test_stats or {},
            sat_test_stats=sat_test_stats,
        )
        minimums = guard_minimums_from_args(args) if guard_minimums_from_args is not None and args is not None else {}
        return float(
            joint_safe_score(
                protected,
                minimums=minimums,
                require_satellite=bool(getattr(args, "joint_guard_require_satellite", True)) if args is not None else False,
            )
        )
    if metric == "source_val_sat_hmean":
        clean = float(val_stats.get("tx_acc", float("-inf")))
        sat_scores = _satellite_tx_scores(sat_test_stats)
        if not math.isfinite(clean) or not sat_scores:
            return float("-inf")
        sat_floor = min(sat_scores)
        return 2.0 * clean * sat_floor / max(1e-8, clean + sat_floor)
    sat_scores = _satellite_tx_scores(sat_test_stats)
    if not sat_scores:
        return float(val_stats.get("tx_acc", float("-inf")))
    if metric == "sat_mean_tx":
        return sum(sat_scores) / len(sat_scores)
    if metric == "sat_worst_tx":
        return min(sat_scores)
    raise ValueError(f"Unknown SSDG best_metric={metric}")


def _joint_safe_guard_enabled(args) -> bool:
    return bool(getattr(args, "enable_joint_safe_guard", False)) or str(getattr(args, "best_metric", "")) == "joint_safe"


def _paic_guard_enabled(args) -> bool:
    return bool(getattr(args, "paic_guard_enabled", False)) or _joint_safe_guard_enabled(args)


def _phase2_audit_requested(args) -> bool:
    flags = (
        "use_phase2_ground_prototypes",
        "use_feature_masks",
        "use_txrx_geometry_losses",
        "use_tx_rx_balanced_sampler",
        "use_proto_memory",
        "phase2_export_prototypes",
    )
    weights = (
        "lambda_tx_proto",
        "lambda_rx_proto",
        "lambda_mask_aux",
        "lambda_tx_supcon_masked",
        "lambda_rx_supcon_masked",
        "lambda_txrx_rect",
        "lambda_proto",
        "lambda_open_world_feat",
        "lambda_zid_compact",
        "lambda_proxy_unknown",
        "lambda_soft_unknown_mixup",
        "lambda_source_episode",
        "lambda_direct_metric_accept",
        "lambda_u_domain",
        "lambda_u_adv",
        "lambda_u_sat_cons",
        "lambda_u_direct_metric_accept",
        "lambda_u_quarantine_accept",
    )
    return any(bool(getattr(args, key, False)) for key in flags) or any(float(getattr(args, key, 0.0)) > 0.0 for key in weights)


def _phase2_audit_state(args) -> Dict[str, Any]:
    requested = _phase2_audit_requested(args)
    modules = {
        "phase2_prototypes": "cvsrffi.phase2_prototypes",
        "feature_masks": "cvsrffi.feature_masks",
        "tx_rx_geometry": "cvsrffi.tx_rx_geometry",
        "balanced_tx_rx_sampler": "cvsrffi.balanced_tx_rx_sampler",
        "open_world_head": "cvsrffi.open_world_head",
    }
    import_status: Dict[str, int] = {}
    missing: list[str] = []
    if requested:
        for name, module in modules.items():
            try:
                importlib.import_module(module)
                import_status[name] = 1
            except Exception:
                import_status[name] = 0
                missing.append(name)
        if missing:
            raise ImportError(f"Phase1 prototype/mask audit requested but modules are unavailable: {','.join(missing)}")
    weights = {
        "tx_proto": float(getattr(args, "lambda_tx_proto", 0.0)),
        "rx_proto": float(getattr(args, "lambda_rx_proto", 0.0)),
        "mask_aux": float(getattr(args, "lambda_mask_aux", 0.0)),
        "tx_supcon_masked": float(getattr(args, "lambda_tx_supcon_masked", 0.0)),
        "rx_supcon_masked": float(getattr(args, "lambda_rx_supcon_masked", 0.0)),
        "txrx_rect": float(getattr(args, "lambda_txrx_rect", 0.0)),
        "proto_memory": float(getattr(args, "lambda_proto", 0.0)),
        "open_world_feat": float(getattr(args, "lambda_open_world_feat", 0.0)),
        "zid_compact": float(getattr(args, "lambda_zid_compact", 0.0)),
        "proxy_unknown": float(getattr(args, "lambda_proxy_unknown", 0.0)),
        "soft_unknown_mixup": float(getattr(args, "lambda_soft_unknown_mixup", 0.0)),
        "source_episode": float(getattr(args, "lambda_source_episode", 0.0)),
        "direct_metric_accept": float(getattr(args, "lambda_direct_metric_accept", 0.0)),
        "u_domain": float(getattr(args, "lambda_u_domain", 0.0)),
        "u_adv": float(getattr(args, "lambda_u_adv", 0.0)),
        "u_sat_cons": float(getattr(args, "lambda_u_sat_cons", 0.0)),
        "u_direct_metric_accept": float(getattr(args, "lambda_u_direct_metric_accept", 0.0)),
        "u_quarantine_accept": float(getattr(args, "lambda_u_quarantine_accept", 0.0)),
    }
    legacy_unwired = {
        key: value
        for key, value in weights.items()
        if key in {"tx_proto", "rx_proto", "mask_aux", "tx_supcon_masked", "rx_supcon_masked", "txrx_rect"} and value > 0.0
    }
    if legacy_unwired:
        raise NotImplementedError(
            "Non-zero legacy Phase1 prototype/mask/geometry audit losses are not wired into SSDG training yet. "
            "Use zero weights for --lambda_tx_proto/--lambda_rx_proto/--lambda_mask_aux/"
            "--lambda_tx_supcon_masked/--lambda_rx_supcon_masked/--lambda_txrx_rect, and use "
            "--lambda_proto, --lambda_open_world_feat, --lambda_zid_compact, --lambda_proxy_unknown, "
            "--lambda_soft_unknown_mixup, or --lambda_source_episode for the default-off active z_id feature-space bridge."
        )
    active_loss = bool(getattr(args, "use_proto_memory", False)) or any(
        value > 0.0
        for key, value in weights.items()
        if key in {
            "proto_memory",
            "open_world_feat",
            "zid_compact",
            "proxy_unknown",
            "soft_unknown_mixup",
            "source_episode",
            "direct_metric_accept",
            "u_domain",
            "u_adv",
            "u_sat_cons",
            "u_direct_metric_accept",
            "u_quarantine_accept",
        }
    )
    return {
        "requested": bool(requested),
        "audit_only": bool(getattr(args, "phase1_distribution_audit_only", True)) and not active_loss,
        "active_loss": bool(active_loss),
        "use_phase2_ground_prototypes": bool(getattr(args, "use_phase2_ground_prototypes", False)),
        "use_feature_masks": bool(getattr(args, "use_feature_masks", False)),
        "use_txrx_geometry_losses": bool(getattr(args, "use_txrx_geometry_losses", False)),
        "use_tx_rx_balanced_sampler": bool(getattr(args, "use_tx_rx_balanced_sampler", False)),
        "use_proto_memory": bool(getattr(args, "use_proto_memory", False)) or float(getattr(args, "lambda_proto", 0.0)) > 0.0,
        "use_open_world_feature_loss": float(getattr(args, "lambda_open_world_feat", 0.0)) > 0.0,
        "use_zid_compactness_loss": float(getattr(args, "lambda_zid_compact", 0.0)) > 0.0,
        "use_proxy_unknown_loss": float(getattr(args, "lambda_proxy_unknown", 0.0)) > 0.0,
        "use_soft_unknown_mixup_loss": float(getattr(args, "lambda_soft_unknown_mixup", 0.0)) > 0.0,
        "use_source_episode_loss": float(getattr(args, "lambda_source_episode", 0.0)) > 0.0,
        "use_direct_metric_acceptance_loss": float(getattr(args, "lambda_direct_metric_accept", 0.0)) > 0.0,
        "use_unlabeled_domain_loss": float(getattr(args, "lambda_u_domain", 0.0)) > 0.0,
        "use_unlabeled_adv_loss": float(getattr(args, "lambda_u_adv", 0.0)) > 0.0,
        "use_unlabeled_sat_cons_loss": float(getattr(args, "lambda_u_sat_cons", 0.0)) > 0.0,
        "use_unlabeled_direct_metric_loss": float(getattr(args, "lambda_u_direct_metric_accept", 0.0)) > 0.0,
        "use_unlabeled_quarantine_accept_loss": float(getattr(args, "lambda_u_quarantine_accept", 0.0)) > 0.0,
        "phase2_export_prototypes": bool(getattr(args, "phase2_export_prototypes", False)),
        "imports": import_status,
        "weights": weights,
    }


def _resolve_sat_eval_max_batches(args) -> int:
    sat_eval_max_batches = int(getattr(args, "sat_eval_max_batches", -1))
    if sat_eval_max_batches < 0:
        sat_eval_max_batches = int(getattr(args, "eval_max_batches", 0))
    return sat_eval_max_batches


def _derive_phase2_export_path(checkpoint_path: str | Path) -> str:
    path = Path(str(checkpoint_path).strip() or "best_joint_safe_ssdg.pth")
    return str(path.with_name(f"{path.stem}_phase1_source_prototypes.pt"))


def _endpoint_entry_decision_signature(gate, features: torch.Tensor, logits: torch.Tensor) -> List[Dict[str, Any]]:
    outputs = gate.batch_decide(features, logits)
    signatures: List[Dict[str, Any]] = []
    for row in outputs:
        debug = row.get("debug", {}) or {}
        component_id = debug.get("component_id")
        signatures.append({
            "decision": str(row.get("decision", "")),
            "class_id": int(row.get("class_id", -1)),
            "component_id": int(component_id) if component_id is not None else -1,
            "radius_region": str(debug.get("radius_region", "")),
        })
    return signatures


def _attach_verified_endpoint_entry_parity(
    package: Mapping[str, Any], calibration: Mapping[str, torch.Tensor]
) -> Dict[str, Any]:
    if LocalComponentHardGate is None:
        raise ImportError("LocalComponentHardGate is required for endpoint entry parity")
    features = calibration["features"].detach().float().cpu()[:64]
    logits = calibration["logits"].detach().float().cpu()[:64]
    if features.numel() == 0 or logits.size(0) != features.size(0):
        raise ValueError("endpoint entry parity requires aligned source-val features/logits")
    provisional = dict(package)
    metadata = dict(provisional.get("metadata", {}) or {})
    metadata["endpoint_runtime_entry_parity_digest"] = "0" * 64
    metadata["endpoint_runtime_entry_parity_sample_count"] = int(features.size(0))
    provisional["metadata"] = metadata
    provisional = attach_endpoint_accept_v1_manifest(provisional)
    provisional_identity = provisional["endpoint_accept_v1"]["inference_identity"]
    signatures = [
        _endpoint_entry_decision_signature(factory(provisional), features, logits)
        for factory in (
            LocalComponentHardGate.from_train_export,
            lambda artifact: LocalComponentHardGate.from_offline_eval(
                artifact, runtime_identity=provisional_identity
            ),
            lambda artifact: LocalComponentHardGate.from_runtime_inference(
                artifact, runtime_identity=provisional_identity
            ),
        )
    ]
    if not (signatures[0] == signatures[1] == signatures[2]):
        raise ValueError("endpoint_accept_v1 train/offline/runtime decision parity failed")
    canonical = json.dumps(signatures[0], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    final_package = dict(provisional)
    final_metadata = dict(final_package.get("metadata", {}) or {})
    final_metadata["endpoint_runtime_entry_parity_digest"] = digest
    final_metadata["endpoint_runtime_entry_parity_sample_count"] = int(features.size(0))
    final_metadata["endpoint_runtime_entry_parity_verified"] = True
    final_package["metadata"] = final_metadata
    final_package = attach_endpoint_accept_v1_manifest(final_package)
    final_identity = final_package["endpoint_accept_v1"]["inference_identity"]
    final_signatures = [
        _endpoint_entry_decision_signature(factory(final_package), features, logits)
        for factory in (
            LocalComponentHardGate.from_train_export,
            lambda artifact: LocalComponentHardGate.from_offline_eval(
                artifact, runtime_identity=final_identity
            ),
            lambda artifact: LocalComponentHardGate.from_runtime_inference(
                artifact, runtime_identity=final_identity
            ),
        )
    ]
    if not (final_signatures[0] == final_signatures[1] == final_signatures[2] == signatures[0]):
        raise ValueError("endpoint_accept_v1 final artifact decision parity failed")
    return final_package


def _maybe_export_phase2_prototypes_ssdg(
    args,
    model,
    data_ctx: Mapping[str, Any],
    device,
    *,
    default_checkpoint: str | Path,
    diagnostic_only: bool = False,
) -> Optional[Dict[str, Any]]:
    if not bool(getattr(args, "phase2_export_prototypes", False)):
        return None
    if export_phase2_prototypes is None:
        raise ImportError("cvsrffi.phase2_prototypes export support is required for --phase2_export_prototypes")
    loader_name = str(getattr(args, "phase2_export_split", "train") or "train").strip().lower()
    if loader_name == "train":
        loader = data_ctx.get("probe_train_loader", data_ctx["train_loader"])
    elif loader_name == "val":
        loader = data_ctx["val_loader"]
    else:
        raise ValueError(f"Unsupported --phase2_export_split: {loader_name}")
    checkpoint_path = str(getattr(args, "phase2_export_checkpoint", "") or "").strip()
    if not checkpoint_path:
        checkpoint_path = str(default_checkpoint)
    if bool(getattr(args, "formal_ablation", False)):
        if Path(checkpoint_path).resolve() != Path(default_checkpoint).resolve():
            raise ValueError(
                "formal Phase1 export must use the final training checkpoint"
            )
    if str(getattr(args, "checkpoint_selection", "")) == "final_only":
        if Path(checkpoint_path).resolve() != Path(default_checkpoint).resolve():
            raise ValueError("Phase1 final-only export forbids a non-final --phase2_export_checkpoint")
    output_path = str(getattr(args, "phase2_export_path", "") or "").strip()
    if not output_path:
        output_path = _derive_phase2_export_path(checkpoint_path)
    if bool(diagnostic_only):
        path = Path(output_path)
        output_path = str(path.with_name(f"{path.stem}_diagnostic{path.suffix}"))

    restore_state = deepcopy(getattr(model, "_orig_mod", model).state_dict())
    try:
        checkpoint_sha256 = ""
        ckpt: Mapping[str, Any] = {}
        if checkpoint_path:
            ckpt = _validate_phase1_checkpoint_payload(
                load_checkpoint(checkpoint_path, device),
                args,
                checkpoint_path,
            )
            _load_phase1_checkpoint_strict(model, ckpt, checkpoint_path)
            checkpoint_sha256 = _sha256_file(checkpoint_path)
        package = export_phase2_prototypes(
            model,
            loader,
            output_path=None,
            device=device,
            feature_key=str(getattr(args, "phase2_export_feature_key", "z_id") or "z_id"),
            max_batches=int(getattr(args, "phase2_export_max_batches", 0) or 0),
            metadata={
                "checkpoint_path": checkpoint_path,
                "loader_split": loader_name,
                "dataset": str(getattr(args, "dataset", "")),
                "split_mode": str(getattr(args, "split_mode", "")),
                "split_info": data_ctx.get("split_info", {}),
                "source": "SSDG.train_ssdg default-off Phase2 export hook",
                "base_protocol": "Safe-SSDG-CVS-R01",
                "default_training_behavior_changed": False,
                "claim": "PHASE1_SOURCE_ONLY_EXPORT_ONLY",
                "phase1_source_only": True,
                "stage2_success_claim": False,
                "deployment_success_claim": False,
                "final_reject_boundary": False,
                "endpoint_accept_boundary_exported": False,
                "diagnostic_only": bool(diagnostic_only),
                "promotion_ready": False if bool(diagnostic_only) else None,
                "endpoint_policy_id": str(getattr(args, "endpoint_accept_policy_id", "endpoint_accept_v1")),
                "run_id": str(getattr(args, "run_id", "")),
                "candidate_id": str(getattr(args, "candidate_id", "")),
                "source_checkpoint_sha256": checkpoint_sha256,
                "class_id_to_tx": list(data_ctx.get("class_id_to_tx", [])),
                "logit_class_order": list(range(int(getattr(args, "num_classes", 0)))),
                "known_class_count": int(getattr(args, "num_classes", 0)),
                "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
                "checkpoint_load_strict": True,
                "checkpoint_selection": str(
                    getattr(args, "checkpoint_selection", "")
                ),
                "satellite_protocol": dict(getattr(args, "sat_protocol_manifest", {}) or {}),
                "zid_leakage_probe": ((ckpt.get("stats", {}) or {}).get("zid_leakage_probe", {})),
            },
        )
        if bool(getattr(args, "phase2_fuse_prototypes", False)):
            if fuse_tx_domain_prototypes is None or PrototypeFusionConfig is None or save_phase2_prototype_export is None:
                raise ImportError("cvsrffi.phase2_prototypes fusion support is required for --phase2_fuse_prototypes")
            package = fuse_tx_domain_prototypes(
                package,
                PrototypeFusionConfig(
                    max_components_per_tx=int(getattr(args, "phase2_fuse_max_components", 4)),
                    merge_angle_deg=float(getattr(args, "phase2_fuse_merge_angle_deg", 6.0)),
                    radius_cap_deg=float(getattr(args, "phase2_fuse_radius_cap_deg", 25.0)),
                    tail_abs_deg=float(getattr(args, "phase2_fuse_tail_abs_deg", 30.0)),
                    accept_policy=str(getattr(args, "phase2_fuse_accept_policy", "local_component")),
                    accept_radius_key=str(getattr(args, "phase2_fuse_accept_radius_key", "p95")),
                    max_p95_increase_deg=float(getattr(args, "phase2_fuse_max_p95_increase_deg", 2.0)),
                    keep_tail_sentinel=bool(getattr(args, "phase2_fuse_keep_tail_sentinel", True)),
                    global_ball_accept=bool(getattr(args, "phase2_fuse_global_ball_accept", False)),
                    tail_auto_accept=bool(getattr(args, "phase2_fuse_tail_auto_accept", False)),
                ),
            )
        endpoint_artifact_ready = False
        if bool(getattr(args, "endpoint_require_artifact_on_export", True)) and not bool(
            diagnostic_only
        ):
            if not bool(getattr(args, "phase2_fuse_prototypes", False)):
                raise ValueError("endpoint_accept_v1 export requires --phase2_fuse_prototypes true")
            if (
                extract_endpoint_calibration_features is None
                or calibrate_endpoint_accept_v1 is None
                or attach_endpoint_accept_v1_manifest is None
                or verify_endpoint_accept_v1_manifest is None
                or assess_endpoint_contract is None
            ):
                raise ImportError("endpoint_accept_v1 source-val calibration support is unavailable")
            calibration = extract_endpoint_calibration_features(
                model,
                data_ctx.get("source_calibration_loader", data_ctx["val_loader"]),
                device=device,
                feature_key=str(getattr(args, "phase2_export_feature_key", "z_id") or "z_id"),
                max_batches=int(getattr(args, "phase2_export_max_batches", 0) or 0),
                require_identity_contract=True,
            )
            identity_feature_audit = audit_identity_feature_contract(
                calibration["z_id_features"],
                calibration["feat_joint_features"],
                calibration["labels"],
                calibration["domains"],
                calibration["logits"],
                expected_classes=int(getattr(args, "num_classes", 0)),
                min_class_samples=int(
                    getattr(args, "endpoint_calibration_min_class_samples", 4)
                ),
            )
            package_metadata = dict(package.get("metadata", {}) or {})
            package_metadata["identity_feature_contract_audit"] = identity_feature_audit
            package["metadata"] = package_metadata
            package = calibrate_endpoint_accept_v1(
                package,
                calibration["features"],
                calibration["labels"],
                calibration["logits"],
                min_component_samples=int(getattr(args, "endpoint_calibration_min_component_samples", 4)),
                min_class_samples=int(getattr(args, "endpoint_calibration_min_class_samples", 4)),
                core_quantile=float(getattr(args, "endpoint_calibration_core_quantile", 0.80)),
                accept_quantile=float(getattr(args, "endpoint_calibration_accept_quantile", 0.95)),
                tail_quantile=float(getattr(args, "endpoint_calibration_tail_quantile", 0.99)),
            )
            package = _attach_verified_endpoint_entry_parity(package, calibration)
            endpoint_manifest = verify_endpoint_accept_v1_manifest(package)
            endpoint_decision = assess_endpoint_contract(
                {
                    "phase": "Phase1_source_only",
                    "source_only": True,
                    "endpoint_policy_id": str(getattr(args, "endpoint_accept_policy_id", "endpoint_accept_v1")),
                    "endpoint_accept_boundary_exported": True,
                    "endpoint_artifact_required": True,
                    "endpoint_artifact": package,
                    "endpoint_threshold_source": str(getattr(args, "endpoint_threshold_source", "source_val_only")),
                    "endpoint_calibration_split": str(getattr(args, "endpoint_calibration_split", "source_val")),
                    "loss_gate_exported": False,
                    "phase1_proxy_only": True,
                    "real_unknown_eval_available": False,
                    "stage2_success_claim": False,
                    "deployment_success_claim": False,
                }
            )
            if endpoint_decision.fired:
                raise ValueError(f"endpoint_accept_v1 artifact contract failed: {endpoint_decision.reason}")
            endpoint_artifact_ready = True
        if save_phase2_prototype_export is None:
            raise ImportError("cvsrffi.phase2_prototypes save support is required for Phase2 export")
        paths = save_phase2_prototype_export(package, output_path)
        package = dict(package)
        package["paths"] = paths
        paths = package.get("paths", {}) if isinstance(package, dict) else {}
        print(
            f"[PHASE1-EXPORT] wrote prototypes={paths.get('pt_path', output_path)} "
            f"json={paths.get('json_path', '')} feature={getattr(args, 'phase2_export_feature_key', 'z_id')} "
            f"split={loader_name} fused={int(bool(getattr(args, 'phase2_fuse_prototypes', False)))} "
            "claim=PHASE1_SOURCE_ONLY_EXPORT_ONLY stage2_success_claim=0 "
            "deployment_success_claim=0 true_unknown_validated=0 "
            f"endpoint_artifact_ready={int(endpoint_artifact_ready)} "
            f"diagnostic_only={int(bool(diagnostic_only))} "
            f"endpoint_policy_id={str(getattr(args, 'endpoint_accept_policy_id', 'endpoint_accept_v1'))}",
            flush=True,
        )
        return package
    finally:
        model.load_state_dict(restore_state, strict=True)


def _evaluate_sat_if_enabled(model, data_ctx, device, args) -> Dict[str, Dict[str, Any]]:
    if not bool(getattr(args, "eval_sat_channel", False)):
        return {}
    scenarios = list(getattr(args, "eval_sat_scenario_list", []))
    if not scenarios:
        return {}
    return evaluate_sat_scenarios(
        model,
        data_ctx["named_test_loaders"],
        device,
        data_ctx["domain_label_map"],
        scenario_names=scenarios,
        args=args,
        max_batches=_resolve_sat_eval_max_batches(args),
    )


def _evaluate_source_val_sat_if_enabled(model, data_ctx, device, args) -> Dict[str, Dict[str, Any]]:
    if not bool(getattr(args, "eval_sat_channel", False)):
        return {}
    scenarios = list(getattr(args, "eval_sat_scenario_list", []))
    if not scenarios:
        return {}
    source_val_args = deepcopy(args)
    source_val_args.eval_sat_on = "all"
    return evaluate_sat_scenarios(
        model,
        {"source_val": data_ctx["val_loader"]},
        device,
        data_ctx["domain_label_map"],
        scenario_names=scenarios,
        args=source_val_args,
        max_batches=_resolve_sat_eval_max_batches(args),
    )


def _evaluate_source_val_tail_geometry(model, data_ctx, device, args) -> Dict[str, Any]:
    """Evaluate tail/proxy geometry on one fixed source-val protocol."""

    if direct_metric_acceptance_loss is None:
        return {"status": "FAILED", "reason": "direct_metric_acceptance_loss_unavailable"}
    was_training = bool(model.training)
    features: List[torch.Tensor] = []
    sat_features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    domains: List[torch.Tensor] = []
    sat_scenarios = list(getattr(args, "sat_train_protocol_scenario_list", []))
    sat_generator = make_torch_generator(device, int(getattr(args, "seed", 0)) + 91079) if make_torch_generator is not None else None
    try:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_ctx["val_loader"], start=1):
                if int(args.eval_max_batches) > 0 and batch_idx > int(args.eval_max_batches):
                    break
                x, y, extra = move_batch(batch, device)
                d = domain_from_extra(extra, data_ctx["domain_label_map"], device)
                out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=d)
                features.append(out["z_id"].detach().float())
                if sat_scenarios and apply_sat_channel_for_scenario is not None:
                    scenario = sat_scenarios[(batch_idx - 1) % len(sat_scenarios)]
                    x_sat, _ = apply_sat_channel_for_scenario(
                        x,
                        scenario,
                        args,
                        gen=sat_generator,
                        return_meta=False,
                    )
                    sat_out = model(
                        _safe_iq_tensor(x_sat),
                        y_tx=None,
                        grl_lambda=1.0,
                        return_aux=True,
                        domain_labels=d,
                    )
                    sat_features.append(sat_out["z_id"].detach().float())
                labels.append(y.detach().long())
                if d is not None:
                    domains.append(d.detach().long())
        if not features:
            return {"status": "FAILED", "reason": "source_val_geometry_empty"}
        z = torch.cat(features, dim=0)
        y = torch.cat(labels, dim=0)
        d = torch.cat(domains, dim=0) if len(domains) == len(features) else None
        cuda_devices = [int(device.index or 0)] if getattr(device, "type", "cpu") == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(getattr(args, "seed", 0)) + 91073)
            if (
                bool(getattr(args, "direct_metric_multiview_separate", False))
                and len(sat_features) == len(features)
                and sat_features
                and multiview_direct_metric_acceptance_loss is not None
            ):
                z_sat = torch.cat(sat_features, dim=0)
                _loss, info = multiview_direct_metric_acceptance_loss(
                    z,
                    z_sat,
                    y,
                    d,
                    clean_weight=float(args.direct_metric_clean_weight),
                    sat_weight=float(args.direct_metric_sat_weight),
                    pair_weight=float(args.direct_metric_sat_pair_weight),
                    sat_pair_target_rad=math.radians(float(args.direct_metric_sat_pair_target_deg)),
                    **_direct_metric_kwargs(args),
                )
            else:
                _loss, info = direct_metric_acceptance_loss(z, y, d, **_direct_metric_kwargs(args))
        result = {str(key): float(value) if isinstance(value, (int, float)) else value for key, value in info.items()}
        result.update(
            {
                "status": "COMPLETE" if float(info.get("active", 0.0)) > 0.0 else "FAILED",
                "protocol": "fixed_source_val_multiview_local_component_v2",
                "sample_count": int(z.size(0)),
                "seed": int(getattr(args, "seed", 0)) + 91073,
                "satellite_scenarios": sat_scenarios,
                "multiview_sample_count": int(torch.cat(sat_features, dim=0).size(0)) if sat_features else 0,
            }
        )
        if result["status"] != "COMPLETE":
            result["reason"] = "source_val_direct_metric_inactive"
        return result
    finally:
        model.train(was_training)


def _evaluate_zid_leakage_probes(model, data_ctx, device, args) -> Dict[str, Any]:
    """Fit frozen source-train probes and evaluate only on source validation."""

    if frozen_ridge_linear_probe is None:
        return {"status": "FAILED", "reason": "frozen_ridge_linear_probe_unavailable"}
    train_scenarios = list(getattr(args, "sat_train_scenario_list", []))
    eval_scenarios = list(getattr(args, "eval_sat_scenario_list", []))
    max_batches = int(getattr(args, "zid_leakage_probe_max_batches", 0))
    was_training = bool(model.training)

    def _collect(loader, scenarios: Sequence[str], seed_offset: int) -> Dict[str, torch.Tensor]:
        clean_features: List[torch.Tensor] = []
        receiver_labels: List[torch.Tensor] = []
        day_labels: List[torch.Tensor] = []
        channel_features: List[torch.Tensor] = []
        channel_labels: List[torch.Tensor] = []
        generator = make_torch_generator(device, int(args.seed) + int(seed_offset)) if make_torch_generator is not None else None
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                if max_batches > 0 and batch_idx > max_batches:
                    break
                x, _y, extra = move_batch(batch, device)
                batch_size = int(x.size(0))
                receiver = _metadata_label_tensor(extra, "rx_i", device, batch_size)
                day = _metadata_label_tensor(extra, "day_i", device, batch_size)
                if receiver is None or day is None:
                    continue
                d = domain_from_extra(extra, data_ctx["domain_label_map"], device)
                clean_out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=d)
                z_clean = clean_out["z_id"].detach().float().cpu()
                clean_features.append(z_clean)
                receiver_labels.append(receiver.detach().cpu())
                day_labels.append(day.detach().cpu())
                if scenarios and apply_sat_channel_for_scenario is not None:
                    scenario = str(scenarios[(batch_idx - 1) % len(scenarios)])
                    x_sat, _ = apply_sat_channel_for_scenario(
                        x,
                        scenario,
                        args,
                        gen=generator,
                        return_meta=False,
                    )
                    sat_out = model(
                        _safe_iq_tensor(x_sat),
                        y_tx=None,
                        grl_lambda=1.0,
                        return_aux=True,
                        domain_labels=d,
                    )
                    z_sat = sat_out["z_id"].detach().float().cpu()
                    channel_features.extend([z_clean, z_sat])
                    channel_labels.extend(
                        [
                            torch.zeros(batch_size, dtype=torch.long),
                            torch.ones(batch_size, dtype=torch.long),
                        ]
                    )
        return {
            "clean_features": torch.cat(clean_features, dim=0) if clean_features else torch.empty((0, 0)),
            "receiver_labels": torch.cat(receiver_labels, dim=0) if receiver_labels else torch.empty((0,), dtype=torch.long),
            "day_labels": torch.cat(day_labels, dim=0) if day_labels else torch.empty((0,), dtype=torch.long),
            "channel_features": torch.cat(channel_features, dim=0) if channel_features else torch.empty((0, 0)),
            "channel_labels": torch.cat(channel_labels, dim=0) if channel_labels else torch.empty((0,), dtype=torch.long),
        }

    try:
        model.eval()
        train = _collect(data_ctx["probe_train_loader"], train_scenarios, 92011)
        evaluate = _collect(data_ctx["val_loader"], eval_scenarios, 92029)
        ridge = float(getattr(args, "zid_leakage_probe_ridge", 0.01))
        receiver = frozen_ridge_linear_probe(
            train["clean_features"],
            train["receiver_labels"],
            evaluate["clean_features"],
            evaluate["receiver_labels"],
            ridge=ridge,
        )
        day = frozen_ridge_linear_probe(
            train["clean_features"],
            train["day_labels"],
            evaluate["clean_features"],
            evaluate["day_labels"],
            ridge=ridge,
        )
        channel = frozen_ridge_linear_probe(
            train["channel_features"],
            train["channel_labels"],
            evaluate["channel_features"],
            evaluate["channel_labels"],
            ridge=ridge,
        )
        status = "COMPLETE" if all(obj.get("status") == "COMPLETE" for obj in (receiver, day, channel)) else "FAILED"
        result: Dict[str, Any] = {
            "status": status,
            "schema": "phase1_zid_source_train_to_val_leakage_probe_v1",
            "fit_split": "source_labeled_train",
            "eval_split": "source_val",
            "channel_fit_scenarios": train_scenarios,
            "channel_eval_scenarios": eval_scenarios,
            "receiver": receiver,
            "day": day,
            "channel": channel,
        }
        for name, obj in (("receiver", receiver), ("day", day), ("channel", channel)):
            result[f"zid_{name}_probe_acc"] = obj.get("accuracy", float("nan"))
            result[f"zid_{name}_probe_balanced_acc"] = obj.get("balanced_accuracy", float("nan"))
            result[f"zid_{name}_probe_chance"] = obj.get("chance_accuracy", float("nan"))
            result[f"zid_{name}_probe_raw_excess"] = obj.get("excess_accuracy", float("nan"))
            result[f"zid_{name}_probe_balanced_chance"] = obj.get(
                "balanced_chance_accuracy", float("nan")
            )
            result[f"zid_{name}_probe_excess"] = obj.get("balanced_excess_accuracy", float("nan"))
        if status != "COMPLETE":
            result["reason"] = "one_or_more_probes_incomplete"
        return result
    finally:
        model.train(was_training)


def _assess_zid_leakage_probe(probe: Mapping[str, Any], args) -> Dict[str, Any]:
    limits = {
        "receiver": float(args.zid_receiver_probe_max_excess),
        "day": float(args.zid_day_probe_max_excess),
        "channel": float(args.zid_channel_probe_max_excess),
    }
    reasons: List[str] = []
    details: Dict[str, Any] = {"required": bool(args.zid_leakage_probe_required)}
    if str(probe.get("status", "")) != "COMPLETE":
        reasons.append("ZID_LEAKAGE_PROBE_INCOMPLETE")
    for name, limit in limits.items():
        value = float(probe.get(f"zid_{name}_probe_excess", float("nan")))
        details[f"{name}_excess"] = value
        details[f"{name}_max_excess"] = limit
        if not math.isfinite(value):
            reasons.append(f"ZID_{name.upper()}_PROBE_MISSING")
        elif value > limit:
            reasons.append(f"ZID_{name.upper()}_LEAKAGE_EXCESS")
    fired = bool(args.zid_leakage_probe_required and reasons)
    return {
        "fired": fired,
        "reason": ";".join(dict.fromkeys(reasons)) if fired else "",
        "details": details,
    }


def _validate_phase1_checkpoint_payload(checkpoint: Any, args, path: str | Path) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError(f"Phase1 checkpoint is invalid: {path}")
    if str(checkpoint.get("checkpoint_schema", "")) != "ssdg_phase1_training_state_v2":
        raise ValueError(f"Phase1 checkpoint schema mismatch: {path}")
    saved_run = str(checkpoint.get("run_id", ""))
    saved_candidate = str(checkpoint.get("candidate_id", ""))
    expected_run = str(getattr(args, "run_id", ""))
    expected_candidate = str(getattr(args, "candidate_id", ""))
    if saved_run != expected_run or saved_candidate != expected_candidate:
        raise ValueError(
            "Phase1 checkpoint identity mismatch: "
            f"expected={expected_run}/{expected_candidate} saved={saved_run}/{saved_candidate} path={path}"
        )
    saved_args = checkpoint.get("args", {}) or {}
    if str(saved_args.get("best_metric", "")) != str(getattr(args, "best_metric", "")):
        raise ValueError(f"Phase1 checkpoint selection metric mismatch: {path}")
    expected_selection = str(getattr(args, "checkpoint_selection", ""))
    if str(
        checkpoint.get(
            "checkpoint_selection",
            saved_args.get("checkpoint_selection", ""),
        )
    ) != expected_selection:
        raise ValueError(
            f"Phase1 checkpoint selection mismatch: expected={expected_selection} path={path}"
        )
    return checkpoint


def _load_phase1_checkpoint_strict(model, checkpoint: Mapping[str, Any], path: str | Path) -> None:
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as exc:
        raise ValueError(f"Phase1 checkpoint/model strict-load mismatch: {path}: {exc}") from exc


def _external_final_eval_requested(args) -> bool:
    return bool(getattr(args, "phase1_external_final_eval", False)) or (
        bool(getattr(args, "use_muse_ssdg", False))
        and bool(getattr(args, "muse_external_final_eval", False))
    )


def _run_final_heldout_evaluation(
    args,
    model,
    data_ctx,
    device,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """Delegate MUSE target evaluation to its launcher while preserving legacy behavior."""

    if bool(getattr(args, "use_muse_ssdg", False)) and bool(
        getattr(args, "muse_external_final_eval", False)
    ):
        return {
            "status": "DELEGATED_TO_MUSE_LAUNCHER",
            "selection_source": str(getattr(args, "checkpoint_selection", "")),
            "checkpoint": str(Path(checkpoint_path)),
            "claim": "PHASE1_TARGET_EVAL_DELEGATED_NO_INTERNAL_METRICS",
        }
    if bool(getattr(args, "phase1_external_final_eval", False)):
        return {
            "status": "DELEGATED_TO_EXTERNAL_PHASE1_EVAL",
            "selection_source": str(getattr(args, "checkpoint_selection", "")),
            "checkpoint": str(Path(checkpoint_path)),
            "claim": "PHASE1_FINAL_EVAL_DELEGATED_NO_INTERNAL_METRICS",
        }
    return _evaluate_frozen_phase1_checkpoint(args, model, data_ctx, device, checkpoint_path)


def _evaluate_frozen_phase1_checkpoint(args, model, data_ctx, device, checkpoint_path: str | Path) -> Dict[str, Any]:
    """Evaluate held-out receiver/day and satellite views once after source-val selection is frozen."""

    path = Path(checkpoint_path)
    if not path.is_file():
        return {"status": "NOT_RUN", "reason": "frozen_source_val_checkpoint_missing", "checkpoint": str(path)}
    restore_state = deepcopy(getattr(model, "_orig_mod", model).state_dict())
    try:
        checkpoint = _validate_phase1_checkpoint_payload(
            load_checkpoint(str(path), device),
            args,
            path,
        )
        _load_phase1_checkpoint_strict(model, checkpoint, path)
        named = evaluate_named_loaders(
            model,
            data_ctx["named_test_loaders"],
            device,
            data_ctx["domain_label_map"],
            max_batches=int(args.eval_max_batches),
        )
        test_stats = _aggregate_main_test(named, str(args.dataset))
        sat_stats = _evaluate_sat_if_enabled(model, data_ctx, device, args)
        protected = protected_metric_snapshot(
            val_stats=(checkpoint.get("stats", {}) or {}).get("val", {}),
            test_stats=test_stats,
            named_test_stats=named,
            sat_test_stats=sat_stats,
        )
        return {
            "status": "COMPLETE",
            "selection_source": str(args.checkpoint_selection),
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256_file(path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "test": test_stats,
            "named_test": named,
            "sat_test_named": sat_stats,
            "protected_metrics": protected,
            "claim": "PHASE1_FROZEN_HELDOUT_EVAL_NOT_MODEL_SELECTION",
        }
    finally:
        model.load_state_dict(restore_state, strict=True)


def _evaluate_checkpoint_source_val_tail_geometry(args, model, data_ctx, device, checkpoint_path: str | Path) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        return {"status": "FAILED", "reason": "checkpoint_missing", "checkpoint": str(path)}
    restore_state = deepcopy(getattr(model, "_orig_mod", model).state_dict())
    try:
        checkpoint = _validate_phase1_checkpoint_payload(
            load_checkpoint(str(path), device),
            args,
            path,
        )
        _load_phase1_checkpoint_strict(model, checkpoint, path)
        result = _evaluate_source_val_tail_geometry(model, data_ctx, device, args)
        result.update(
            {
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256_file(path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            }
        )
        return result
    finally:
        model.load_state_dict(restore_state, strict=True)


def _resolve_phase1_terminal_status(
    *,
    tail_stopped: bool,
    export_failed: bool,
    final_blocked: bool,
    selected_checkpoint_exists: bool,
    heldout_eval_status: str,
    external_final_eval: bool = False,
    p0_mechanisms_ready: bool = True,
    p1_mechanisms_ready: bool = True,
    endpoint_export_ready: bool = True,
    mechanism_gates_required: bool = True,
    endpoint_export_required: bool = True,
) -> str:
    """Resolve a fail-closed Phase1 terminal state without overstating promotion readiness."""

    if tail_stopped:
        return "STOPPED_TAIL"
    if export_failed:
        return "FAILED_EXPORT"
    if final_blocked:
        return "NON_PROMOTABLE_GUARD_BLOCKED"
    if bool(mechanism_gates_required) and not p0_mechanisms_ready:
        return "NON_PROMOTABLE_P0_DISABLED"
    if bool(mechanism_gates_required) and not p1_mechanisms_ready:
        return "NON_PROMOTABLE_P1_DISABLED"
    if not selected_checkpoint_exists:
        return "NO_SAFE_CHECKPOINT"
    heldout_status = str(heldout_eval_status).upper()
    heldout_complete = heldout_status == "COMPLETE" or (
        bool(external_final_eval)
        and heldout_status
        in {"DELEGATED_TO_MUSE_LAUNCHER", "DELEGATED_TO_EXTERNAL_PHASE1_EVAL"}
    )
    if not heldout_complete:
        return "HELDOUT_EVAL_INCOMPLETE"
    if bool(endpoint_export_required) and not endpoint_export_ready:
        return "NON_PROMOTABLE_ENDPOINT_NOT_EXPORTED"
    return "COMPLETE"


def _formal_ablation_terminal_flags(
    args,
    *,
    selected_checkpoint: Path,
    selected_checkpoint_evidence: Mapping[str, Any],
    selected_checkpoint_sha256: str,
    export_status: Mapping[str, Any],
    source_split_receipt: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    checkpoint_args = dict(selected_checkpoint_evidence.get("args", {}) or {})
    split_receipt = dict(source_split_receipt or {})
    expected_ablation_config = phase1_ablation_config(
        str(getattr(args, "ablation_id", ""))
    )
    train_rx = {
        value.strip()
        for value in str(getattr(args, "wisig_train_rxs", "")).split(",")
        if value.strip()
    }
    test_rx = {
        value.strip()
        for value in str(getattr(args, "wisig_test_rxs", "")).split(",")
        if value.strip()
    }
    expected_representation = (
        "single_parameter_matched"
        if str(args.ablation_id) == "P1-A0"
        else "dual"
    )
    p0_flags = {
        "formal_ablation_identity": (
            str(getattr(args, "candidate_id", "")) == str(args.ablation_id)
            and str(getattr(args, "ablation_schema", ""))
            == "cvs.phase1_ablation.config.v1"
        ),
        "git_commit_bound": len(str(getattr(args, "git_commit", ""))) == 40,
        "method_config_hash_bound": len(
            str(getattr(args, "ablation_method_config_hash", ""))
        )
        == 64,
        "resolved_config_hash_bound": len(
            str(getattr(args, "ablation_config_hash", ""))
        )
        == 64,
        "source_receiver_set_locked": train_rx
        == {"0", "1", "2", "3", "4", "5", "6"},
        "target_receiver_set_locked": test_rx
        == {"7", "8", "9", "10", "11"},
        "source_target_receiver_disjoint": not bool(train_rx & test_rx),
        "current_split_locked": (
            abs(
                float(args.labeled_ratio)
                - float(expected_ablation_config["labeled_ratio"])
            )
            <= 1e-12
            and abs(
                float(args.unlabeled_ratio)
                - float(expected_ablation_config["unlabeled_ratio"])
            )
            <= 1e-12
            and abs(
                float(args.source_val_ratio)
                - float(expected_ablation_config["source_val_ratio"])
            )
            <= 1e-12
        ),
        "final_only_selection": (
            bool(getattr(args, "phase1_source_val_selection_only", False))
            and str(getattr(args, "checkpoint_selection", ""))
            == "final_only"
        ),
        "representation_mode_matches_arm": str(
            getattr(args, "representation_mode", "")
        )
        == expected_representation,
        "selected_checkpoint_evidence_bound": bool(
            selected_checkpoint_evidence
        ),
        "selected_checkpoint_identity_match": (
            str(checkpoint_args.get("ablation_id", ""))
            == str(args.ablation_id)
            and str(checkpoint_args.get("ablation_config_hash", ""))
            == str(args.ablation_config_hash)
            and str(checkpoint_args.get("git_commit", ""))
            == str(args.git_commit)
        ),
        "row_identity_bound": (
            str(checkpoint_args.get("row_key", ""))
            == str(getattr(args, "row_key", ""))
        ),
        "sealed_plan_bound": (
            str(checkpoint_args.get("sealed_plan_sha256", ""))
            == str(getattr(args, "sealed_plan_sha256", ""))
        ),
        "seed_registry_bound": (
            str(checkpoint_args.get("seed_registry_sha256", ""))
            == str(getattr(args, "seed_registry_sha256", ""))
        ),
        "source_split_receipt_bound": (
            len(str(split_receipt.get("split_manifest_sha256", "")))
            == 64
        ),
        "source_split_requested_ratios_match_arm": (
            abs(
                float(split_receipt.get("requested_labeled_ratio", -1.0))
                - float(expected_ablation_config["labeled_ratio"])
            )
            <= 1e-12
            and abs(
                float(split_receipt.get("requested_unlabeled_ratio", -1.0))
                - float(expected_ablation_config["unlabeled_ratio"])
            )
            <= 1e-12
            and abs(
                float(split_receipt.get("requested_source_val_ratio", -1.0))
                - float(expected_ablation_config["source_val_ratio"])
            )
            <= 1e-12
        ),
        "source_split_realized_rho_within_tolerance": (
            split_receipt.get("realized_rho_within_tolerance") is True
        ),
        "source_split_realized_source_val_within_tolerance": (
            split_receipt.get("realized_source_val_within_tolerance")
            is True
        ),
        "source_target_receivers_physically_disjoint": int(
            split_receipt.get(
                "source_target_receiver_overlap_count",
                -1,
            )
        )
        == 0,
    }
    p1_flags = {
        "selected_checkpoint_is_final": selected_checkpoint.name
        == "final_ssdg.pth",
        "selected_checkpoint_role_final": str(
            selected_checkpoint_evidence.get("checkpoint_role", "")
        )
        == "training_final_only",
        "phase2_export_prototypes": bool(
            getattr(args, "phase2_export_prototypes", False)
        ),
        "phase2_fuse_prototypes": bool(
            getattr(args, "phase2_fuse_prototypes", False)
        ),
        "endpoint_export_complete": str(export_status.get("status", ""))
        == "COMPLETE",
        "endpoint_checkpoint_identity_match": bool(
            selected_checkpoint_sha256
        )
        and str(export_status.get("source_checkpoint_sha256", ""))
        == selected_checkpoint_sha256,
    }
    return p0_flags, p1_flags


def _loss_weights(args, stage_state: Mapping[str, Any] | None) -> Dict[str, float]:
    stage_state = stage_state or {}
    return {
        "dom": float(getattr(args, "lambda_domain", 0.0)) * float(stage_state.get("dom_scale", 1.0)),
        "adv": float(getattr(args, "lambda_adv", 0.0)) * float(stage_state.get("adv_scale", 1.0)),
        "orth": float(getattr(args, "lambda_orth", 0.0)) * float(stage_state.get("orth_scale", 1.0)),
        "cons": float(getattr(args, "lambda_cons", 0.0)) * float(stage_state.get("cons_scale", 0.0)),
        "group_ce": float(getattr(args, "lambda_group_ce", 0.0)) * float(stage_state.get("group_ce_scale", 1.0)),
        "fishr": float(getattr(args, "lambda_fishr", 0.0)),
        "sat_cls": float(getattr(args, "lambda_sat_cls", 0.0)),
        "sat_cons": float(getattr(args, "lambda_sat_cons", 0.0)),
        "proto": float(getattr(args, "lambda_proto", 0.0)),
        "open_world_feat": float(getattr(args, "lambda_open_world_feat", 0.0)),
        "zid_compact": float(getattr(args, "lambda_zid_compact", 0.0)),
        "proxy_unknown": float(getattr(args, "lambda_proxy_unknown", 0.0)),
        "soft_unknown_mixup": float(getattr(args, "lambda_soft_unknown_mixup", 0.0)),
        "source_episode": float(getattr(args, "lambda_source_episode", 0.0)),
        "direct_metric_accept": float(getattr(args, "lambda_direct_metric_accept", 0.0)),
        "u_domain": float(getattr(args, "lambda_u_domain", 0.0)),
        "u_adv": float(getattr(args, "lambda_u_adv", 0.0)),
        "u_sat_cons": float(getattr(args, "lambda_u_sat_cons", 0.0)),
        "u_direct_metric_accept": float(getattr(args, "lambda_u_direct_metric_accept", 0.0)),
        "u_quarantine_accept": float(getattr(args, "lambda_u_quarantine_accept", 0.0)),
    }


def _teacher_distill_scale(args, epoch: int) -> float:
    return _stage_gate_scale(
        epoch,
        start_epoch=int(getattr(args, "teacher_distill_start_epoch", 1)),
        warmup_epochs=int(getattr(args, "teacher_distill_warmup_epochs", 0)),
    )


def _teacher_distill_requested(args) -> bool:
    return (
        float(getattr(args, "lambda_teacher_clean_kl", 0.0)) > 0.0
        or float(getattr(args, "lambda_teacher_sat_kl", 0.0)) > 0.0
        or float(getattr(args, "lambda_teacher_zid_mse", 0.0)) > 0.0
    )


def _sat_anchor_requested(args) -> bool:
    return bool(getattr(args, "use_muse_ssdg", False)) and bool(
        getattr(args, "sat_anchor_ssl", False)
    )


def _rc4_requested(args) -> bool:
    return bool(getattr(args, "use_muse_ssdg", False)) and bool(
        getattr(args, "fasttrust_rc4", False)
    )


def _frozen_teacher_requested(args) -> bool:
    return _teacher_distill_requested(args) or _sat_anchor_requested(args) or _rc4_requested(args)


def _stage_gate_scale(epoch: int, *, start_epoch: int = 1, warmup_epochs: int = 0) -> float:
    start = max(1, int(start_epoch))
    if int(epoch) < start:
        return 0.0
    warm = max(0, int(warmup_epochs))
    if warm <= 0:
        return 1.0
    return min(1.0, max(0.0, float(int(epoch) - start + 1) / float(warm)))


def _format_loss_top(values: Mapping[str, float], *, limit: int = 8) -> str:
    finite_values: List[Tuple[float, str, float]] = []
    for key, value in values.items():
        try:
            fv = float(value)
        except Exception:
            continue
        if fv == fv and abs(fv) > 0.0:
            finite_values.append((abs(fv), key, fv))
    finite_values.sort(reverse=True)
    if not finite_values:
        return "[LOSS-TOP] none"
    return "[LOSS-TOP] " + " | ".join(f"{key}={value:.4f}" for _, key, value in finite_values[: int(limit)])


def _fallback_stage_state(phase: str) -> Dict[str, float]:
    if str(phase) == "label":
        name = "S1_core"
    else:
        name = "SSDG_pseudo"
    return {
        "phase": name,
        "use_aux_views": 0.0,
        "dom_scale": 1.0,
        "adv_scale": 0.70,
        "orth_scale": 0.50,
        "cons_scale": 0.0,
        "cls_aux_scale": 0.0,
        "reg_aux_scale": 0.0,
        "joint_inv_scale": 0.0,
        "kl_scale": 0.0,
        "group_ce_scale": 0.50,
    }


def _stage_state_for_epoch(epoch: int, args, phase: str) -> Dict[str, float]:
    if build_stage_state is None:
        return _fallback_stage_state(phase)
    try:
        state = dict(build_stage_state(int(epoch), args))
    except Exception:
        state = _fallback_stage_state(phase)
    if str(phase) == "pseudo":
        state["phase"] = "SSDG_pseudo"
    return state


def _format_stage_line(stage_state: Mapping[str, Any], phase: str) -> str:
    if format_stage_state is not None:
        try:
            return f"phase={phase} train_phase={format_stage_state(dict(stage_state))}"
        except Exception:
            pass
    return (
        f"phase={phase} | use_aux={float(stage_state.get('use_aux_views', 0.0)):.1f} "
        f"cons={float(stage_state.get('cons_scale', 0.0)):.2f} "
        f"cls_aux={float(stage_state.get('cls_aux_scale', 0.0)):.2f} "
        f"reg={float(stage_state.get('reg_aux_scale', 0.0)):.2f} "
        f"joint_inv={float(stage_state.get('joint_inv_scale', 0.0)):.2f} "
        f"kl={float(stage_state.get('kl_scale', 0.0)):.2f} "
        f"group_ce={float(stage_state.get('group_ce_scale', 0.0)):.2f}"
    )


def _fallback_mixstyle_state(args) -> Dict[str, Any]:
    enabled = bool(getattr(args, "use_mixstyle", False))
    return {
        "phase": "base" if enabled else "disabled",
        "enabled": enabled,
        "p": float(getattr(args, "mixstyle_p", 0.0)) if enabled else 0.0,
        "strength": float(getattr(args, "mixstyle_strength", 0.0)) if enabled else 0.0,
        "anneal_t": 0.0,
    }


def _fallback_aug_state(args) -> Dict[str, Any] | None:
    if not bool(getattr(args, "use_aug", True)):
        return None
    scale = float(getattr(args, "aug_scale_min", 0.10))
    return {
        "scale": scale,
        "p_dac": float(getattr(args, "aug_p_dac", 0.0)) * scale,
        "p_pa": float(getattr(args, "aug_p_pa", 0.14)) * scale,
        "p_time_shift": float(getattr(args, "aug_p_time_shift", 0.35)) * scale,
        "p_cfo": float(getattr(args, "aug_p_cfo", 0.35)) * scale,
        "p_awgn": float(getattr(args, "aug_p_awgn", 0.40)) * scale,
        "p_multipath": float(getattr(args, "aug_p_multipath", 0.18)) * scale,
        "max_time_shift": int(round(float(getattr(args, "aug_max_time_shift", 32)) * scale)),
        "cfo_max": float(getattr(args, "aug_cfo_max", 4e-4)) * scale,
        "phase_noise_sigma_max": float(getattr(args, "aug_phase_noise_sigma_max", 0.006)) * scale,
    }


def _grad_norm(model, name_filter=None) -> float:
    if torch is None:
        return float("nan")
    total = 0.0
    seen = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name_filter is not None and not name_filter(name):
            continue
        value = float(param.grad.detach().float().norm(2).item())
        total += value * value
        seen += 1
    return total ** 0.5 if seen > 0 else float("nan")


def _grads_are_finite(model) -> bool:
    if torch is None:
        return True
    for param in model.parameters():
        if param.grad is None:
            continue
        if not bool(torch.isfinite(param.grad.detach()).all().item()):
            return False
    return True


def _first_nonfinite_gradient(model) -> Dict[str, Any] | None:
    if torch is None:
        return None
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gradient = param.grad.detach()
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        return {
            "parameter_name": str(name),
            "nonfinite_elements": int((~finite).sum().item()),
            "nan_elements": int(torch.isnan(gradient).sum().item()),
            "posinf_elements": int(torch.isposinf(gradient).sum().item()),
            "neginf_elements": int(torch.isneginf(gradient).sum().item()),
        }
    return None


def _direct_metric_kwargs(args) -> Dict[str, Any]:
    return {
        "virtual_count": int(args.direct_metric_virtual_count),
        "virtual_mode": str(args.direct_metric_virtual_mode),
        "core_quantile": float(args.direct_metric_core_quantile),
        "accept_quantile": float(args.direct_metric_accept_quantile),
        "tail_quantile": float(args.direct_metric_tail_quantile),
        "overflow_quantile": float(args.direct_metric_overflow_quantile),
        "zid_p50_target_rad": math.radians(float(args.direct_metric_zid_p50_target_deg)),
        "zid_p95_target_rad": math.radians(float(args.direct_metric_zid_p95_target_deg)),
        "zid_p99_target_rad": math.radians(float(args.direct_metric_zid_p99_target_deg)),
        "zid_tail_cvar_target_rad": math.radians(float(args.direct_metric_zid_tail_cvar_target_deg)),
        "source_overflow_target": float(args.direct_metric_source_overflow_target),
        "proxy_vaccept_target": float(args.direct_metric_proxy_vaccept_target),
        "bridge_accept_target": float(args.direct_metric_bridge_accept_target),
        "low_density_accept_target": float(args.direct_metric_low_density_accept_target),
        "tail_accept_target": float(args.direct_metric_tail_accept_target),
        "overflow_accept_target": float(args.direct_metric_overflow_accept_target),
        "radius_inter_ratio_target": float(args.direct_metric_radius_inter_ratio_target),
        "core_accept_target": float(args.direct_metric_core_accept_target),
        "core_tpr_target": float(args.direct_metric_core_tpr_target),
        "known_accept_target": float(args.direct_metric_known_accept_target),
        "known_tpr_target": float(args.direct_metric_known_tpr_target),
        "zid_quantile_weight": float(args.direct_metric_zid_quantile_weight),
        "source_overflow_weight": float(args.direct_metric_source_overflow_weight),
        "proxy_vaccept_weight": float(args.direct_metric_proxy_vaccept_weight),
        "bridge_accept_weight": float(args.direct_metric_bridge_accept_weight),
        "low_density_accept_weight": float(args.direct_metric_low_density_accept_weight),
        "tail_accept_weight": float(args.direct_metric_tail_accept_weight),
        "overflow_accept_weight": float(args.direct_metric_overflow_accept_weight),
        "radius_inter_ratio_weight": float(args.direct_metric_radius_inter_ratio_weight),
        "global_quantile_weight": float(args.direct_metric_global_quantile_weight),
        "component_inter_margin_weight": float(
            args.direct_metric_component_inter_margin_weight
        ),
        "component_overlap_weight": float(args.direct_metric_component_overlap_weight),
        "core_accept_weight": float(args.direct_metric_core_accept_weight),
        "core_tpr_weight": float(args.direct_metric_core_tpr_weight),
        "known_coverage_weight": float(args.direct_metric_known_coverage_weight),
        "quantile_temperature_rad": math.radians(float(args.direct_metric_quantile_temperature_deg)),
        "accept_temperature": float(args.direct_metric_accept_temperature),
        "component_temperature_rad": math.radians(float(args.direct_metric_component_temperature_deg)),
        "density_temperature_rad": math.radians(float(args.direct_metric_density_temperature_deg)),
        "component_margin_rad": math.radians(float(args.direct_metric_component_margin_deg)),
        "component_inter_margin_rad": math.radians(
            float(args.direct_metric_component_inter_margin_deg)
        ),
        "component_overlap_margin_rad": math.radians(
            float(args.direct_metric_component_overlap_margin_deg)
        ),
        "source_margin_rad": math.radians(float(args.direct_metric_source_margin_deg)),
        "source_radius_cap_rad": math.radians(float(args.direct_metric_source_radius_cap_deg)),
        "shell_width_rad": math.radians(float(args.direct_metric_shell_width_deg)),
        "accept_cvar_alpha": float(args.direct_metric_accept_cvar_alpha),
        "positive_first": bool(args.direct_metric_positive_first),
        "negative_start_tpr": float(args.direct_metric_negative_start_tpr),
        "negative_full_tpr": float(args.direct_metric_negative_full_tpr),
        "require_effective_negative_grad": bool(
            args.direct_metric_require_effective_negative_grad
        ),
        "virtual_detach": bool(args.direct_metric_virtual_detach),
        "gate_reference_detach": bool(args.direct_metric_gate_reference_detach),
        "use_domain_local_components": bool(args.direct_metric_domain_local_components),
        "require_domain_local_components": bool(args.direct_metric_require_domain_local_components),
        "min_samples_per_component": int(args.direct_metric_min_samples_per_component),
        "hierarchical_class_gate": bool(args.direct_metric_hierarchical_class_gate),
        "hierarchical_gate_combine": str(args.direct_metric_hierarchical_combine),
    }


def _route_unlabeled_known_geometry(
    *,
    args,
    z_id_l: torch.Tensor,
    y_l: torch.Tensor,
    d_l: Optional[torch.Tensor],
    out_s: Mapping[str, torch.Tensor],
    out_u_sat: Optional[Mapping[str, torch.Tensor]],
    pseudo: torch.Tensor,
    d_u: Optional[torch.Tensor],
    pseudo_mask: torch.Tensor,
    valid_u_mask: torch.Tensor,
    labeled_view_count: int = 0,
    labeled_sat_applied: bool = False,
    dataset_role: str = "legacy_unlabeled",
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor]:
    _assert_muse_open_geometry_role(dataset_role)
    zero = out_s["z_id"].sum() * 0.0
    geometry_core_mask = torch.zeros_like(pseudo_mask, dtype=torch.bool)
    geometry_direct_mask = torch.zeros_like(pseudo_mask, dtype=torch.bool)
    info: Dict[str, Any] = {"active": 0.0, "routing_precomputed": 0.0}
    if unlabeled_known_acceptance_quarantine_loss is None:
        return zero, info, geometry_core_mask, geometry_direct_mask
    if bool(getattr(args, "u_geometry_all_valid_queries", False)):
        quarantine_mask = valid_u_mask.clone() if bool(args.u_quarantine_valid_domain_only) else torch.ones_like(pseudo_mask)
    else:
        quarantine_mask = ~pseudo_mask
        if bool(args.u_quarantine_valid_domain_only):
            quarantine_mask = quarantine_mask & valid_u_mask
    quarantine_count = int(quarantine_mask.sum().detach().item())
    quarantine_pool = int(valid_u_mask.sum().detach().item()) if bool(args.u_quarantine_valid_domain_only) else int(pseudo.numel())
    if quarantine_count < int(args.u_quarantine_min_count):
        info.update(
            {
                "query_count": float(quarantine_count),
                "quarantine_rate": float(quarantine_count) / float(max(1, quarantine_pool)),
                "routing_precomputed": 1.0,
            }
        )
        return zero, info, geometry_core_mask, geometry_direct_mask
    anchor_count = int(labeled_view_count) if int(labeled_view_count) > 0 else int(y_l.numel())
    if anchor_count > int(z_id_l.size(0)) or anchor_count > int(y_l.numel()):
        return zero, {**info, "reason": "invalid_labeled_view_count"}, geometry_core_mask, geometry_direct_mask
    clean_anchor_z = z_id_l[:anchor_count].detach()
    clean_anchor_y = y_l[:anchor_count].detach().long()
    clean_anchor_d = d_l[:anchor_count].detach().long() if d_l is not None else None
    query_y = pseudo[quarantine_mask].detach().long()
    query_d = d_u[quarantine_mask].detach().long() if d_u is not None else None

    def _route_view(anchor_z, anchor_y, anchor_d, query_z) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return unlabeled_known_acceptance_quarantine_loss(
            anchor_z,
            anchor_y,
            query_z,
            anchor_d=anchor_d,
            query_y=query_y,
            query_d=query_d,
            paired_view_count=0,
            return_state_masks=True,
            core_quantile=float(args.u_quarantine_core_quantile),
            accept_quantile=float(args.u_quarantine_accept_quantile),
            accept_target=float(args.u_quarantine_accept_target),
            core_accept_target=float(args.u_quarantine_core_accept_target),
            cvar_alpha=float(args.u_quarantine_cvar_alpha),
            accept_temperature=float(args.u_quarantine_accept_temperature),
            component_temperature_rad=math.radians(float(args.direct_metric_component_temperature_deg)),
            density_temperature_rad=math.radians(float(args.direct_metric_density_temperature_deg)),
            component_margin_rad=math.radians(float(args.direct_metric_component_margin_deg)),
            min_samples_per_class=int(args.direct_metric_min_samples_per_component),
            require_domain_local_components=bool(args.direct_metric_require_domain_local_components),
        )

    clean_loss, clean_info = _route_view(
        clean_anchor_z,
        clean_anchor_y,
        clean_anchor_d,
        out_s["z_id"][quarantine_mask],
    )
    view_infos = [("clean", clean_info)]
    losses = [clean_loss]
    if (
        bool(args.u_quarantine_include_sat_view)
        and out_u_sat is not None
        and bool(labeled_sat_applied)
        and int(z_id_l.size(0)) >= 2 * anchor_count
        and int(y_l.numel()) >= 2 * anchor_count
    ):
        sat_anchor_d = d_l[anchor_count : 2 * anchor_count].detach().long() if d_l is not None else None
        sat_loss, sat_info = _route_view(
            z_id_l[anchor_count : 2 * anchor_count].detach(),
            y_l[anchor_count : 2 * anchor_count].detach().long(),
            sat_anchor_d,
            out_u_sat["z_id"][quarantine_mask],
        )
        losses.append(sat_loss)
        view_infos.append(("sat", sat_info))
    loss = torch.stack(losses).mean()

    state_rows = []
    for view_name, view_info in view_infos:
        core = view_info.pop("_tri_trusted_core_mask", None)
        ambiguous = view_info.pop("_tri_ambiguous_tail_mask", None)
        outside = view_info.pop("_tri_outside_reject_mask", None)
        accept_prob = view_info.pop("_tri_accept_prob", None)
        label_match = view_info.pop("_tri_label_match_mask", None)
        if not all(torch.is_tensor(value) and int(value.numel()) == quarantine_count for value in (core, ambiguous, outside)):
            return zero, {**info, "reason": f"{view_name}_tri_state_missing"}, geometry_core_mask, geometry_direct_mask
        if not all(
            torch.is_tensor(value) and int(value.numel()) == quarantine_count
            for value in (accept_prob, label_match)
        ):
            return zero, {**info, "reason": f"{view_name}_tri_score_missing"}, geometry_core_mask, geometry_direct_mask
        state_rows.append(
            (
                core.bool(),
                ambiguous.bool(),
                outside.bool(),
                accept_prob.float(),
                label_match.bool(),
            )
        )
    combined_core = torch.stack([row[0] for row in state_rows], dim=0).all(dim=0)
    combined_outside = torch.stack([row[2] for row in state_rows], dim=0).any(dim=0)
    combined_ambiguous = (~combined_core) & (~combined_outside)
    combined_accept_prob = torch.stack([row[3] for row in state_rows], dim=0).min(dim=0).values
    combined_label_match = torch.stack([row[4] for row in state_rows], dim=0).all(dim=0)
    geometry_core_mask[quarantine_mask] = combined_core.to(device=pseudo_mask.device)

    pseudo_eligible = pseudo_mask[quarantine_mask].bool()
    if not bool(getattr(args, "u_tri_quota_require_pseudo_mask", True)):
        pseudo_eligible = torch.ones_like(pseudo_eligible)
    direct_eligible = pseudo_eligible & combined_label_match
    route_core = combined_core.clone()
    route_ambiguous = combined_ambiguous.clone()
    route_outside = combined_outside.clone()
    quota_active = bool(getattr(args, "u_tri_quota_routing", False))
    if quota_active:
        route_core = torch.zeros_like(combined_core)
        route_ambiguous = torch.zeros_like(combined_ambiguous)
        route_outside = torch.ones_like(combined_outside)
        core_quota = max(0.0, min(1.0, float(getattr(args, "u_tri_core_quota", 0.20))))
        ambiguous_quota = max(
            0.0,
            min(1.0 - core_quota, float(getattr(args, "u_tri_ambiguous_quota", 0.30))),
        )
        for cls in torch.unique(query_y[direct_eligible]):
            class_idx = torch.nonzero(
                direct_eligible & query_y.eq(cls), as_tuple=False
            ).flatten()
            if class_idx.numel() == 0:
                continue
            ranked = class_idx[
                torch.argsort(combined_accept_prob[class_idx], descending=True)
            ]
            core_n = min(int(ranked.numel()), int(math.ceil(core_quota * int(ranked.numel()))))
            ambiguous_n = min(
                int(ranked.numel()) - core_n,
                int(math.ceil(ambiguous_quota * int(ranked.numel()))),
            )
            if core_n > 0:
                route_core[ranked[:core_n]] = True
                route_outside[ranked[:core_n]] = False
            if ambiguous_n > 0:
                tail_idx = ranked[core_n : core_n + ambiguous_n]
                route_ambiguous[tail_idx] = True
                route_outside[tail_idx] = False

    direct_local = route_core | (
        route_ambiguous & bool(getattr(args, "u_direct_include_ambiguous", False))
    )
    if bool(getattr(args, "u_direct_include_outside_known", False)):
        direct_local = direct_local | route_outside
    direct_local = direct_local & direct_eligible
    geometry_direct_mask[quarantine_mask] = direct_local.to(device=pseudo_mask.device)

    tail_pair_loss = zero
    outside_pair_loss = zero
    if out_u_sat is not None:
        clean_query = F.normalize(out_s["z_id"][quarantine_mask].float(), dim=1)
        sat_query = F.normalize(out_u_sat["z_id"][quarantine_mask].float(), dim=1)
        pair_angles = torch.acos(
            (clean_query * sat_query).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        )

        def _pair_state_loss(state_mask: torch.Tensor, target_deg: float) -> torch.Tensor:
            if not bool(state_mask.any()):
                return pair_angles.sum() * 0.0
            excess = F.relu(pair_angles[state_mask] - math.radians(float(target_deg))).pow(2)
            top_k = max(1, min(int(excess.numel()), int(math.ceil(0.25 * excess.numel()))))
            return excess.topk(k=top_k, largest=True).values.mean()

        tail_pair_loss = _pair_state_loss(
            combined_ambiguous,
            float(getattr(args, "u_tri_tail_pair_target_deg", 12.0)),
        )
        outside_pair_loss = pair_angles.sum() * 0.0
        if not bool(getattr(args, "u_outside_stop_gradient", False)):
            outside_pair_loss = _pair_state_loss(
                combined_outside,
                float(getattr(args, "u_tri_outside_pair_target_deg", 20.0)),
            )
        loss = (
            loss
            + max(0.0, float(getattr(args, "u_tri_tail_pair_weight", 0.0)))
            * tail_pair_loss
            + max(0.0, float(getattr(args, "u_tri_outside_pair_weight", 0.0)))
            * outside_pair_loss
        )

    numeric_keys = set.intersection(
        *[
            {key for key, value in view_info.items() if isinstance(value, (int, float))}
            for _, view_info in view_infos
        ]
    ) if view_infos else set()
    info = {}
    for key in numeric_keys:
        values = [float(view_info[key]) for _, view_info in view_infos if math.isfinite(float(view_info[key]))]
        info[key] = max(values) if values else float("nan")
    info.update(
        {
            "active": min(float(view_info.get("active", 0.0)) for _, view_info in view_infos),
            "query_count": float(quarantine_count),
            "tri_trusted_core_count": float(int(combined_core.sum().item())),
            "tri_ambiguous_tail_count": float(int(combined_ambiguous.sum().item())),
            "tri_outside_reject_count": float(int(combined_outside.sum().item())),
            "tri_trusted_core_rate": float(combined_core.float().mean().item()),
            "tri_ambiguous_tail_rate": float(combined_ambiguous.float().mean().item()),
            "tri_outside_reject_rate": float(combined_outside.float().mean().item()),
            "tri_direct_count": float(int(direct_local.sum().item())),
            "tri_direct_rate": float(direct_local.float().mean().item()),
            "tri_direct_eligible_count": float(int(direct_eligible.sum().item())),
            "tri_direct_eligible_rate": float(direct_eligible.float().mean().item()),
            "tri_quota_routing": 1.0 if quota_active else 0.0,
            "tri_route_core_count": float(int(route_core.sum().item())),
            "tri_route_ambiguous_count": float(int(route_ambiguous.sum().item())),
            "tri_route_outside_count": float(int(route_outside.sum().item())),
            "tri_route_core_rate": float(route_core.float().mean().item()),
            "tri_route_ambiguous_rate": float(route_ambiguous.float().mean().item()),
            "tri_route_outside_rate": float(route_outside.float().mean().item()),
            "tri_route_accept_score_mean": float(combined_accept_prob.mean().item()),
            "tri_route_label_match_rate": float(combined_label_match.float().mean().item()),
            "tri_direct_includes_outside_known": 1.0
            if bool(getattr(args, "u_direct_include_outside_known", False))
            else 0.0,
            "tri_tail_pair_loss": float(tail_pair_loss.detach().item()),
            "tri_outside_pair_loss": float(outside_pair_loss.detach().item()),
            "tri_pair_disagreement_rate": (
                float((state_rows[0][0] != state_rows[1][0]).float().mean().item())
                if len(state_rows) == 2
                else 0.0
            ),
            "multiview_local_components": 1.0 if len(state_rows) == 2 else 0.0,
            "global_component_fallback": max(
                float(view_info.get("global_component_fallback", 1.0)) for _, view_info in view_infos
            ),
        }
    )
    for view_name, view_info in view_infos:
        for key, value in view_info.items():
            if isinstance(value, (int, float)):
                info[f"{view_name}_{key}"] = value
    info["quarantine_rate"] = (
        float(info.get("tri_ambiguous_tail_count", 0.0)) + float(info.get("tri_outside_reject_count", 0.0))
    ) / float(max(1, quarantine_count))
    info["valid_domain_rate"] = float(int(valid_u_mask.sum().detach().item())) / float(max(1, int(pseudo.numel())))
    outside_full_mask = torch.zeros_like(pseudo_mask, dtype=torch.bool)
    outside_full_mask[quarantine_mask] = combined_outside.to(device=pseudo_mask.device)
    info["_tri_outside_full_mask"] = outside_full_mask.detach()
    info["routing_precomputed"] = 1.0
    return loss, info, geometry_core_mask, geometry_direct_mask


def _select_unlabeled_geometry_masks(
    pseudo_mask: torch.Tensor,
    geometry_core_mask: torch.Tensor,
    geometry_direct_mask: torch.Tensor,
    valid_domain_mask: torch.Tensor,
    *,
    all_valid_queries: bool,
    direct_valid_domain_only: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep CE confidence-gated while routing geometry and invariance independently."""

    if all_valid_queries:
        ce_mask = pseudo_mask & geometry_core_mask
        direct_mask = geometry_direct_mask.clone()
        invariance_mask = geometry_core_mask & valid_domain_mask
    else:
        ce_mask = pseudo_mask.clone()
        direct_mask = pseudo_mask.clone()
        invariance_mask = pseudo_mask & valid_domain_mask
    if direct_valid_domain_only:
        direct_mask = direct_mask & valid_domain_mask
    return ce_mask, direct_mask, invariance_mask


def _backward_with_open_set_projection(
    model,
    scaler,
    closed_loss,
    open_loss,
    *,
    project_conflicts: bool = True,
    budget_controller: bool = False,
    min_budget: float = 0.0,
    max_budget: float = 0.0,
    max_os_scale: float = 4.0,
    min_closed_scale: float = 0.35,
    protect_closed_on_conflict: bool = False,
    open_loss_groups: Optional[Mapping[str, torch.Tensor]] = None,
    open_group_shares: Optional[Mapping[str, float]] = None,
    open_group_min_scale: float = 0.25,
    open_group_max_scale: float = 8.0,
    budget_param_filter=None,
) -> Dict[str, float]:
    """Balance objective gradients, then protect closed gradients from open-set conflicts."""

    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    params = [param for _, param in named_params]
    budget_scope = [
        bool(budget_param_filter(name)) if budget_param_filter is not None else True
        for name, _ in named_params
    ]
    zid_scope_active = budget_param_filter is not None and any(budget_scope)
    closed_scaled = scaler.scale(closed_loss)
    closed_grads = torch.autograd.grad(closed_scaled, params, retain_graph=True, allow_unused=True)
    group_grads: Dict[str, Tuple[Optional[torch.Tensor], ...]] = {}
    group_norms_scaled: Dict[str, float] = {}
    group_scales: Dict[str, float] = {}
    configured_groups = [
        (str(name), loss)
        for name, loss in (open_loss_groups or {}).items()
        if torch.is_tensor(loss) and bool(loss.requires_grad)
    ]
    if configured_groups:
        for group_index, (name, group_loss) in enumerate(configured_groups):
            grads = torch.autograd.grad(
                scaler.scale(group_loss),
                params,
                retain_graph=(group_index + 1 < len(configured_groups)),
                allow_unused=True,
            )
            group_grads[name] = grads
            group_sq = group_loss.new_tensor(0.0)
            for in_scope, grad in zip(budget_scope, grads):
                if in_scope and grad is not None:
                    group_sq = group_sq + grad.detach().float().pow(2).sum()
            group_norms_scaled[name] = float(group_sq.sqrt().detach().cpu().item())

        positive_groups = [
            name
            for name, _ in configured_groups
            if math.isfinite(group_norms_scaled.get(name, 0.0))
            and group_norms_scaled.get(name, 0.0) > 1e-12
            and float((open_group_shares or {}).get(name, 0.0)) > 0.0
        ]
        share_total = sum(float((open_group_shares or {}).get(name, 0.0)) for name in positive_groups)
        norm_total = sum(group_norms_scaled[name] for name in positive_groups)
        for name, _ in configured_groups:
            norm = group_norms_scaled.get(name, 0.0)
            if name not in positive_groups or share_total <= 0.0 or norm_total <= 0.0:
                group_scales[name] = 0.0 if norm <= 1e-12 else 1.0
                continue
            target_norm = norm_total * float((open_group_shares or {}).get(name, 0.0)) / share_total
            group_scales[name] = max(
                float(open_group_min_scale),
                min(float(open_group_max_scale), target_norm / max(1e-12, norm)),
            )
        open_grads_list: List[Optional[torch.Tensor]] = []
        for param_index in range(len(params)):
            combined = None
            for name, _ in configured_groups:
                grad = group_grads[name][param_index]
                if grad is None:
                    continue
                scaled_grad = grad * float(group_scales.get(name, 1.0))
                combined = scaled_grad if combined is None else combined + scaled_grad
            open_grads_list.append(combined)
        open_grads = tuple(open_grads_list)
    else:
        open_scaled = scaler.scale(open_loss)
        open_grads = torch.autograd.grad(open_scaled, params, retain_graph=False, allow_unused=True)

    finite_gradient_bundle = all(
        grad is None or bool(torch.isfinite(grad.detach()).all().item())
        for gradients in (closed_grads, open_grads)
        for grad in gradients
    )
    if not finite_gradient_bundle:
        for param, grad_closed, grad_open in zip(params, closed_grads, open_grads):
            if grad_closed is None and grad_open is None:
                param.grad = None
            elif grad_closed is None:
                param.grad = grad_open.detach().clone()
            elif grad_open is None:
                param.grad = grad_closed.detach().clone()
            else:
                param.grad = (grad_closed + grad_open).detach().clone()
        result = {
            "active": 1.0,
            "conflict": 0.0,
            "pre_cosine": float("nan"),
            "post_cosine": float("nan"),
            "closed_grad_norm": float("nan"),
            "open_grad_norm": float("nan"),
            "total_closed_grad_norm": float("nan"),
            "total_open_grad_norm": float("nan"),
            "shared_param_count": 0.0,
            "budget_scope_shared_trainable_params": 0.0 if zid_scope_active else 1.0,
            "budget_scope_shared_zid_path": 1.0 if zid_scope_active else 0.0,
            "balanced_closed_grad_norm": float("nan"),
            "balanced_open_grad_norm": float("nan"),
            "effective_closed_grad_norm": float("nan"),
            "effective_open_grad_norm": float("nan"),
            "os_scale": 1.0,
            "closed_scale": 1.0,
            "pre_budget": float("nan"),
            "post_budget": float("nan"),
            "reason_code": 4.0,
            "conflict_projection_priority_code": 1.0 if protect_closed_on_conflict else 0.0,
            "nonfinite_gradient_bundle": 1.0,
        }
        for name, _ in configured_groups:
            result[f"objective_{name}_raw_norm"] = float("nan")
            result[f"objective_{name}_scale"] = float(group_scales.get(name, 1.0))
            result[f"objective_{name}_effective_norm"] = float("nan")
        return result
    raw_dot = closed_loss.new_tensor(0.0)
    closed_sq = closed_loss.new_tensor(0.0)
    open_sq = closed_loss.new_tensor(0.0)
    total_closed_sq = closed_loss.new_tensor(0.0)
    total_open_sq = closed_loss.new_tensor(0.0)
    shared_param_count = 0
    for in_scope, grad_closed, grad_open in zip(budget_scope, closed_grads, open_grads):
        if grad_closed is not None:
            total_closed_sq = total_closed_sq + grad_closed.detach().float().pow(2).sum()
        if grad_open is not None:
            total_open_sq = total_open_sq + grad_open.detach().float().pow(2).sum()
        if in_scope and grad_closed is not None and grad_open is not None:
            shared_param_count += 1
            closed_sq = closed_sq + grad_closed.detach().float().pow(2).sum()
            open_sq = open_sq + grad_open.detach().float().pow(2).sum()
            raw_dot = raw_dot + (grad_closed.detach().float() * grad_open.detach().float()).sum()
    raw_denom = (closed_sq.sqrt() * open_sq.sqrt()).clamp_min(1e-12)
    pre_cos = raw_dot / raw_denom if bool((raw_denom > 0).item()) else raw_dot.new_tensor(0.0)
    scale = max(1.0, float(scaler.get_scale()))
    closed_norm = float(closed_sq.sqrt().detach().cpu().item()) / scale
    open_norm = float(open_sq.sqrt().detach().cpu().item()) / scale
    total_closed_norm = float(total_closed_sq.sqrt().detach().cpu().item()) / scale
    total_open_norm = float(total_open_sq.sqrt().detach().cpu().item()) / scale
    pre_budget = open_norm / max(1e-12, open_norm + closed_norm)
    os_scale = 1.0
    closed_scale = 1.0
    reason_code = 0.0
    if bool(budget_controller) and compute_open_set_budget_action is not None:
        action = compute_open_set_budget_action(
            os_total=open_norm,
            closed_total=closed_norm,
            min_budget=float(min_budget),
            max_budget=float(max_budget),
            max_os_scale=float(max_os_scale),
            min_closed_scale=float(min_closed_scale),
        )
        os_scale = float(action.os_scale)
        closed_scale = float(action.closed_scale)
        reason_code = (
            3.0
            if action.reason == "B_os_eff_upper_controller_active"
            else 1.0
            if action.active
            else 2.0
            if action.reason == "OS_LOSS_IDLE"
            else 0.0
        )

    balanced_closed = [grad * closed_scale if grad is not None else None for grad in closed_grads]
    balanced_open = [grad * os_scale if grad is not None else None for grad in open_grads]
    dot = closed_loss.new_tensor(0.0)
    balanced_closed_sq = closed_loss.new_tensor(0.0)
    balanced_open_sq = closed_loss.new_tensor(0.0)
    for in_scope, grad_closed, grad_open in zip(budget_scope, balanced_closed, balanced_open):
        if in_scope and grad_closed is not None and grad_open is not None:
            balanced_closed_sq = balanced_closed_sq + grad_closed.detach().float().pow(2).sum()
            balanced_open_sq = balanced_open_sq + grad_open.detach().float().pow(2).sum()
            dot = dot + (grad_closed.detach().float() * grad_open.detach().float()).sum()
    conflict = bool(project_conflicts) and bool((dot < 0).item()) and bool((balanced_closed_sq > 0).item())
    preserve_closed_on_conflict = conflict and (bool(protect_closed_on_conflict) or reason_code == 3.0)
    coeff_denom = balanced_closed_sq if preserve_closed_on_conflict else balanced_open_sq
    coeff = dot / coeff_denom.clamp_min(1e-12) if conflict else dot.new_tensor(0.0)
    post_dot = dot.new_tensor(0.0)
    post_closed_sq = dot.new_tensor(0.0)
    post_open_sq = dot.new_tensor(0.0)
    for in_scope, param, grad_closed, grad_open in zip(budget_scope, params, balanced_closed, balanced_open):
        if grad_closed is None and grad_open is None:
            param.grad = None
            continue
        if grad_closed is None:
            combined = grad_open
            projected_closed = None
            projected_open = grad_open
        elif grad_open is None:
            combined = grad_closed
            projected_closed = grad_closed
            projected_open = None
        else:
            if not in_scope:
                projected_closed = grad_closed
                projected_open = grad_open
                combined = projected_closed + projected_open
                param.grad = combined.detach().clone()
                continue
            if conflict and preserve_closed_on_conflict:
                projected_closed = grad_closed
                projected_open = (
                    grad_open
                    - coeff.to(dtype=grad_open.dtype, device=grad_open.device) * grad_closed
                )
            elif conflict:
                projected_closed = (
                    grad_closed
                    - coeff.to(dtype=grad_closed.dtype, device=grad_closed.device) * grad_open
                )
                projected_open = grad_open
            else:
                projected_closed = grad_closed
                projected_open = grad_open
            combined = projected_closed + projected_open
        param.grad = combined.detach().clone()
        if projected_closed is not None and projected_open is not None:
            post_dot = post_dot + (projected_closed.detach().float() * projected_open.detach().float()).sum()
        if projected_closed is not None and projected_open is not None:
            post_closed_sq = post_closed_sq + projected_closed.detach().float().pow(2).sum()
        if projected_open is not None and projected_closed is not None:
            post_open_sq = post_open_sq + projected_open.detach().float().pow(2).sum()
    post_denom = (post_closed_sq.sqrt() * post_open_sq.sqrt()).clamp_min(1e-12)
    post_cos = post_dot / post_denom if bool((post_denom > 0).item()) else post_dot.new_tensor(0.0)
    balanced_closed_norm = float(balanced_closed_sq.sqrt().detach().cpu().item()) / scale
    balanced_open_norm = float(balanced_open_sq.sqrt().detach().cpu().item()) / scale
    projected_closed_norm = float(post_closed_sq.sqrt().detach().cpu().item()) / scale
    projected_open_norm = float(post_open_sq.sqrt().detach().cpu().item()) / scale
    post_budget = projected_open_norm / max(1e-12, projected_open_norm + projected_closed_norm)
    result = {
        "active": 1.0,
        "conflict": 1.0 if conflict else 0.0,
        "pre_cosine": float(pre_cos.detach().cpu().item()),
        "post_cosine": float(post_cos.detach().cpu().item()),
        "closed_grad_norm": closed_norm,
        "open_grad_norm": open_norm,
        "total_closed_grad_norm": total_closed_norm,
        "total_open_grad_norm": total_open_norm,
        "shared_param_count": float(shared_param_count),
        "budget_scope_shared_trainable_params": 0.0 if zid_scope_active else 1.0,
        "budget_scope_shared_zid_path": 1.0 if zid_scope_active else 0.0,
        "balanced_closed_grad_norm": balanced_closed_norm,
        "balanced_open_grad_norm": balanced_open_norm,
        "effective_closed_grad_norm": projected_closed_norm,
        "effective_open_grad_norm": projected_open_norm,
        "os_scale": os_scale,
        "closed_scale": closed_scale,
        "pre_budget": pre_budget,
        "post_budget": post_budget,
        "reason_code": reason_code,
        "conflict_projection_priority_code": 1.0 if preserve_closed_on_conflict else 0.0,
        "nonfinite_gradient_bundle": 0.0,
    }
    for name, _ in configured_groups:
        raw_norm = float(group_norms_scaled.get(name, 0.0)) / scale
        group_scale = float(group_scales.get(name, 1.0))
        result[f"objective_{name}_raw_norm"] = raw_norm
        result[f"objective_{name}_scale"] = group_scale
        result[f"objective_{name}_effective_norm"] = raw_norm * group_scale * os_scale
    return result


def _log_value(logs: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(logs.get(key, default))
    except Exception:
        return float(default)


def _sum_log_values(log_items: Sequence[Mapping[str, Any]], key: str) -> float:
    total = 0.0
    for logs in log_items:
        total += _log_value(logs, key)
    return total


def _max_log_value(log_items: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_log_value(logs, key, float("nan")) for logs in log_items]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def _capture_training_rng_state(sat_gen=None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sat_generator": sat_gen.get_state() if sat_gen is not None else None,
    }
    return state


def _restore_training_rng_state(state: Mapping[str, Any], sat_gen=None) -> None:
    if not isinstance(state, Mapping):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if sat_gen is not None and state.get("sat_generator") is not None:
        sat_gen.set_state(state["sat_generator"])


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_cuda_memory_audit(device) -> None:
    """Initialize the selected CUDA context before resetting peak counters."""

    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_formal_dataset_receipt(args) -> Dict[str, Any]:
    receipt_path = Path(
        str(getattr(args, "dataset_receipt_path", "")).strip()
    )
    expected_receipt_hash = str(
        getattr(args, "dataset_receipt_sha256", "")
    ).strip().lower()
    if (
        not receipt_path.is_file()
        or len(expected_receipt_hash) != 64
        or _sha256_file(receipt_path) != expected_receipt_hash
    ):
        raise ValueError("formal Phase1 dataset receipt is missing or invalid")
    receipt = json.loads(
        receipt_path.read_text(encoding="utf-8-sig")
    )
    dataset_path = Path(str(args.wisig_pkl)).resolve()
    stat = dataset_path.stat()
    expected = {
        "schema": "cvs.phase1.dataset_receipt.v1",
        "sealed_plan_sha256": str(args.sealed_plan_sha256),
        "wisig_pkl_sha256": str(args.wisig_pkl_sha256),
        "wisig_pkl_path": str(dataset_path),
        "wisig_pkl_size": int(stat.st_size),
        "wisig_pkl_mtime_ns": int(stat.st_mtime_ns),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"formal Phase1 dataset receipt drift: {key}"
            )
    return dict(receipt)


def _validate_formal_environment_receipt(args) -> Dict[str, Any]:
    receipt_path = Path(
        str(getattr(args, "environment_receipt_path", "")).strip()
    )
    expected_hash = str(
        getattr(args, "environment_receipt_sha256", "")
    ).strip().lower()
    if (
        not receipt_path.is_file()
        or len(expected_hash) != 64
        or _sha256_file(receipt_path) != expected_hash
    ):
        raise ValueError(
            "formal Phase1 environment receipt is missing or invalid"
        )
    receipt = json.loads(
        receipt_path.read_text(encoding="utf-8-sig")
    )
    expected_environment_id = str(
        getattr(args, "python_environment_id", "")
    ).strip()
    if (
        not expected_environment_id
        or Path(sys.prefix).name.lower()
        != expected_environment_id.lower()
    ):
        raise ValueError(
            "formal Phase1 Python environment differs from the sealed ID"
        )
    expected = {
        "schema": "cvs.phase1.python_environment_receipt.v1",
        "environment_id": expected_environment_id,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"formal Phase1 environment receipt drift: {key}"
            )
    return dict(receipt)


def _safe_percent(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "nan"


def _telemetry_field_name(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        else:
            safe.append("_")
    return "_".join(part for part in "".join(safe).split("_") if part)


def _telemetry_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu") and hasattr(value, "numel"):
        if int(value.numel()) == 1:
            value = value.cpu().item()
        else:
            return str(tuple(int(v) for v in value.shape))
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    return float(numeric) if math.isfinite(numeric) else None


def _flatten_telemetry(row: Dict[str, Any], prefix: str, values: Mapping[str, Any] | None) -> None:
    for key, value in (values or {}).items():
        if isinstance(value, Mapping):
            _flatten_telemetry(row, f"{prefix}_{key}", value)
            continue
        raw_key = str(key)
        field_key = raw_key if raw_key.startswith(f"{prefix}/") or raw_key.startswith(f"{prefix}_") else f"{prefix}_{raw_key}"
        row[_telemetry_field_name(field_key)] = _telemetry_scalar(value)


def _count_nonfinite(values: Mapping[str, Any] | None) -> int:
    count = 0
    for value in (values or {}).values():
        if hasattr(value, "detach"):
            value = value.detach()
        try:
            numeric = float(value)
        except Exception:
            continue
        if not math.isfinite(numeric):
            count += 1
    return count


def _build_ssdg_epoch_telemetry_row(
    *,
    args,
    epoch: int,
    epochs: int,
    lr: float,
    epoch_time_s: float,
    phase: str,
    train_logs: Mapping[str, Any],
    val_stats: Mapping[str, Any],
    test_stats: Mapping[str, Any],
    named_test_stats: Mapping[str, Mapping[str, Any]],
    sat_test_stats: Mapping[str, Mapping[str, Any]],
    stage_state: Mapping[str, Any],
    mixstyle_state: Mapping[str, Any],
    aug_state: Mapping[str, Any] | None,
    loss_weights: Mapping[str, float],
    best_score: float,
    best_val: float,
    best_test: float,
    best_epoch: int,
    latest_path: str,
    best_path: str,
    is_best: bool,
    protected_metrics: Mapping[str, Any] | None = None,
    guard_state: Mapping[str, Any] | None = None,
    phase2_audit_state: Mapping[str, Any] | None = None,
    safe_latest_path: str = "",
    safe_best_path: str = "",
    safe_checkpoint_saved: bool = False,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "schema": "ssdg_epoch_telemetry_v1",
        "epoch": int(epoch),
        "epochs": int(epochs),
        "phase": str(phase),
        "lr": float(lr),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "epoch_time_s": float(epoch_time_s),
        "seed": int(getattr(args, "seed", 0)),
        "output_dir": str(getattr(args, "output_dir", "")),
        "baseline_ckpt": str(getattr(args, "baseline_ckpt", "")),
        "from_scratch": bool(getattr(args, "from_scratch", False)),
        "dataset": str(getattr(args, "dataset", "wisig")),
        "split_mode": str(getattr(args, "split_mode", "")),
        "labeled_ratio": float(getattr(args, "labeled_ratio", 0.0)),
        "unlabeled_ratio": float(getattr(args, "unlabeled_ratio", 0.0)),
        "source_val_ratio": float(getattr(args, "source_val_ratio", 0.0)),
        "source_cal_ratio": float(getattr(args, "source_cal_ratio", 0.0)),
        "source_select_ratio": float(getattr(args, "source_select_ratio", 0.0)),
        "phase1_source_role_protocol": str(
            getattr(args, "phase1_source_role_protocol", "legacy_l_u_v")
        ),
        "label_epochs": int(getattr(args, "label_epochs", 0)),
        "pseudo_epochs": int(getattr(args, "pseudo_epochs", 0)),
        "optimizer": "AdamW",
        "amp": bool(getattr(args, "amp", False)),
        "best_metric": str(getattr(args, "best_metric", "")),
        "best_score": _telemetry_scalar(best_score),
        "best_val_tx": _telemetry_scalar(best_val),
        "best_test_tx": _telemetry_scalar(best_test),
        "best_epoch": int(best_epoch),
        "latest_path": str(latest_path),
        "best_path": str(best_path),
        "safe_latest_path": str(safe_latest_path),
        "safe_best_path": str(safe_best_path),
        "safe_checkpoint_saved": bool(safe_checkpoint_saved),
        "is_best": bool(is_best),
        "use_unlabeled": bool(getattr(args, "use_unlabeled", False)),
        "lambda_u": float(getattr(args, "lambda_u", 0.0)),
        "lambda_ent": float(getattr(args, "lambda_ent", 0.0)),
        "lambda_u_domain": float(getattr(args, "lambda_u_domain", 0.0)),
        "lambda_u_adv": float(getattr(args, "lambda_u_adv", 0.0)),
        "lambda_u_sat_cons": float(getattr(args, "lambda_u_sat_cons", 0.0)),
        "lambda_u_direct_metric_accept": float(getattr(args, "lambda_u_direct_metric_accept", 0.0)),
        "lambda_u_quarantine_accept": float(getattr(args, "lambda_u_quarantine_accept", 0.0)),
        "u_domain_start_epoch": int(getattr(args, "u_domain_start_epoch", 0)),
        "u_sat_cons_start_epoch": int(getattr(args, "u_sat_cons_start_epoch", 0)),
        "u_direct_metric_start_epoch": int(getattr(args, "u_direct_metric_start_epoch", 0)),
        "u_direct_metric_min_selected": int(getattr(args, "u_direct_metric_min_selected", 0)),
        "u_direct_metric_use_sat_pair": bool(getattr(args, "u_direct_metric_use_sat_pair", False)),
        "u_direct_metric_valid_domain_only": bool(getattr(args, "u_direct_metric_valid_domain_only", False)),
        "u_quarantine_start_epoch": int(getattr(args, "u_quarantine_start_epoch", 0)),
        "u_quarantine_valid_domain_only": bool(getattr(args, "u_quarantine_valid_domain_only", False)),
        "u_quarantine_accept_target": float(getattr(args, "u_quarantine_accept_target", 0.0)),
        "pseudo_threshold_mode": str(getattr(args, "pseudo_threshold_mode", "")),
        "tau_min": float(getattr(args, "tau_min", 0.0)),
        "tau_max": float(getattr(args, "tau_max", 0.0)),
        "pseudo_quantile": float(getattr(args, "pseudo_quantile", 0.0)),
        "pseudo_domain_gate": bool(getattr(args, "pseudo_domain_gate", False)),
        "pseudo_temporal_gate": bool(getattr(args, "pseudo_temporal_gate", False)),
        "pseudo_temporal_mode": str(getattr(args, "pseudo_temporal_mode", "batch_neighbor")),
        "pseudo_strong_agreement": bool(getattr(args, "pseudo_strong_agreement", False)),
        "use_ema_teacher": bool(getattr(args, "use_ema_teacher", False)),
        "use_sat_consistency": bool(getattr(args, "use_sat_consistency", False)),
        "sat_train_scenario": str(getattr(args, "sat_train_scenario", "")),
        "sat_train_scenarios": ",".join(getattr(args, "sat_train_scenario_list", []) or []),
        "sat_view_schedule": str(getattr(args, "sat_view_schedule", "") or ""),
        "sat_training_mode": str(getattr(args, "sat_training_mode", "") or ""),
        "use_concat_sat_channel_aug": bool(getattr(args, "use_concat_sat_channel_aug", False)),
        "concat_sat_ce_only": bool(getattr(args, "concat_sat_ce_only", False)),
        "concat_sat_deduplicate_tx_ce": bool(getattr(args, "concat_sat_deduplicate_tx_ce", False)),
        "concat_sat_teacher_clean_only": bool(getattr(args, "concat_sat_teacher_clean_only", False)),
        "id_feature_key": str(getattr(args, "id_feature_key", "feat_joint")),
        "os_budget_scope": str(getattr(args, "os_budget_scope", "all_shared")),
        "direct_metric_hierarchical_combine": str(
            getattr(args, "direct_metric_hierarchical_combine", "product")
        ),
        "eval_sat_channel": bool(getattr(args, "eval_sat_channel", False)),
        "eval_sat_scenarios": str(getattr(args, "eval_sat_scenarios", "")),
        "lambda_crra_pair": float(getattr(args, "lambda_crra_pair", 0.0)),
        "lambda_crra_sat_kl": float(getattr(args, "lambda_crra_sat_kl", 0.0)),
        "lambda_crra_sat_shell": float(getattr(args, "lambda_crra_sat_shell", 0.0)),
        "nonfinite_train_metric_count": _count_nonfinite(train_logs),
        "nonfinite_val_metric_count": _count_nonfinite(val_stats),
        "nonfinite_test_metric_count": _count_nonfinite(test_stats),
    }
    _flatten_telemetry(row, "protected", protected_metrics or {})
    _flatten_telemetry(row, "joint_guard", guard_state or {})
    _flatten_telemetry(row, "phase2_audit", phase2_audit_state or {})
    for name, default in (
        ("high_ratio", 0.0),
        ("mid_ratio", 0.0),
        ("low_ratio", 0.0),
        ("effective_weight", 0.0),
        ("head_js", 0.0),
        ("proto_update_weight", 0.0),
        ("pseudo_precision_diagnostic", "N/A"),
    ):
        row[f"train_muse_{name}"] = _telemetry_scalar(
            train_logs.get(f"muse/{name}", default)
        )
    for name in (
        "lambda_domain",
        "lambda_adv",
        "lambda_orth",
        "lambda_cons",
        "lambda_group_ce",
        "lambda_fishr",
        "lambda_sat_cls",
        "lambda_sat_cons",
        "concat_sat_start_epoch",
        "concat_sat_ce_weight",
        "sat_view_prob",
        "sat_view_seed",
        "lambda_proto",
        "lambda_open_world_feat",
        "lambda_zid_compact",
        "lambda_proxy_unknown",
        "lambda_soft_unknown_mixup",
        "lambda_direct_metric_accept",
        "lambda_u_domain",
        "lambda_u_adv",
        "lambda_u_sat_cons",
        "lambda_u_direct_metric_accept",
        "lambda_u_quarantine_accept",
        "u_domain_start_epoch",
        "u_sat_cons_start_epoch",
        "u_direct_metric_start_epoch",
        "u_direct_metric_min_selected",
        "u_sat_zid_cons_weight",
        "u_quarantine_start_epoch",
        "u_quarantine_accept_target",
        "u_quarantine_core_quantile",
        "u_quarantine_accept_quantile",
        "u_quarantine_cvar_alpha",
        "proto_domain_align_weight",
        "ow_feat_radius_deg",
        "ow_feat_inter_margin_deg",
        "ow_feat_sample_margin_deg",
        "ow_feat_domain_align_weight",
        "ow_feat_tail_weight",
        "ow_feat_cvar_alpha",
        "soft_unknown_mixup_count",
        "soft_unknown_mixup_order",
        "soft_unknown_mixup_alpha",
        "soft_unknown_mixup_energy_margin",
        "soft_unknown_mixup_ce_weight",
        "soft_unknown_mixup_energy_weight",
        "soft_unknown_mixup_vacuum_weight",
        "soft_unknown_mixup_vacuum_width_deg",
        "soft_unknown_mixup_vacuum_hard_k",
        "lambda_source_episode",
        "source_episode_mixup_weight",
        "source_episode_mixup_hard_k",
        "source_episode_radius_mode",
        "source_episode_core_quantile",
        "source_episode_radius_cap_deg",
        "source_episode_min_sigma_deg",
        "direct_metric_start_epoch",
        "direct_metric_warmup_epochs",
        "direct_metric_virtual_count",
        "direct_metric_virtual_mode",
        "direct_metric_core_quantile",
        "direct_metric_accept_quantile",
        "direct_metric_tail_quantile",
        "direct_metric_overflow_quantile",
        "direct_metric_zid_p50_target_deg",
        "direct_metric_zid_p95_target_deg",
        "direct_metric_zid_p99_target_deg",
        "direct_metric_zid_tail_cvar_target_deg",
        "direct_metric_source_overflow_target",
        "direct_metric_proxy_vaccept_target",
        "direct_metric_bridge_accept_target",
        "direct_metric_low_density_accept_target",
        "direct_metric_tail_accept_target",
        "direct_metric_overflow_accept_target",
        "direct_metric_radius_inter_ratio_target",
        "direct_metric_core_accept_target",
        "direct_metric_sat_pair_target_deg",
        "direct_metric_zid_quantile_weight",
        "direct_metric_source_overflow_weight",
        "direct_metric_proxy_vaccept_weight",
        "direct_metric_bridge_accept_weight",
        "direct_metric_low_density_accept_weight",
        "direct_metric_tail_accept_weight",
        "direct_metric_overflow_accept_weight",
        "direct_metric_radius_inter_ratio_weight",
        "direct_metric_core_accept_weight",
        "direct_metric_sat_pair_weight",
        "direct_metric_accept_cvar_alpha",
        "ow_feat_vacuum_weight",
        "ow_feat_vacuum_width_deg",
        "proxy_unknown_component_radius_mode",
        "proxy_unknown_component_radius_quantile",
        "proxy_unknown_vacuum_weight",
        "proxy_unknown_vacuum_width_deg",
        "proxy_unknown_bridge_accept_weight",
        "proxy_unknown_shell_outward_accept_weight",
        "proxy_unknown_low_density_accept_weight",
        "proxy_unknown_energy_margin_quantile_weight",
        "proxy_unknown_radius_budget_weight",
        "proxy_unknown_radius_inter_ratio_weight",
        "proxy_unknown_bridge_accept_target",
        "proxy_unknown_tail_accept_target",
        "proxy_unknown_overflow_accept_target",
        "proxy_unknown_energy_margin_target",
        "proxy_unknown_radius_budget_deg",
        "proxy_unknown_radius_inter_ratio_target",
        "label_smoothing",
        "group_ce_top_frac",
        "strong_noise_std",
    ):
        row[name] = _telemetry_scalar(getattr(args, name, None))
    _flatten_telemetry(row, "stage", stage_state)
    _flatten_telemetry(row, "mixstyle", mixstyle_state)
    _flatten_telemetry(row, "aug", aug_state or {"enabled": False})
    _flatten_telemetry(row, "loss_weight", loss_weights)
    _flatten_telemetry(row, "train", train_logs)
    _flatten_telemetry(row, "val", val_stats)
    _flatten_telemetry(row, "test", test_stats)
    for name, stats in (named_test_stats or {}).items():
        _flatten_telemetry(row, f"named_test_{name}", stats)
    for scenario, stats in (sat_test_stats or {}).items():
        _flatten_telemetry(row, f"sat_test_{scenario}", stats)
    return row


def _write_ssdg_epoch_telemetry(
    csv_path: Path | str | None,
    jsonl_path: Path | str | None,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if csv_path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
    if jsonl_path:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n")


def _format_named_test_lines(named_test_stats: Mapping[str, Mapping[str, Any]], named_test_meta: Mapping[str, Mapping[str, Any]]) -> List[str]:
    if format_named_test_lines is not None:
        return format_named_test_lines(dict(named_test_stats), dict(named_test_meta))
    lines = []
    for name, stats in named_test_stats.items():
        lines.append(
            f"          {name}: tx={_safe_percent(stats.get('tx_acc'))}% "
            f"({int(stats.get('tx_correct', 0))}/{int(stats.get('tx_total', 0))})"
        )
    return lines


def _format_sat_test_lines(sat_test_stats: Mapping[str, Mapping[str, Any]]) -> List[str]:
    if format_sat_test_lines is not None:
        return format_sat_test_lines(dict(sat_test_stats))
    lines = []
    for scenario, stats in sat_test_stats.items():
        agg = stats.get("aggregate", {})
        selected = ",".join(stats.get("selected_names", []))
        lines.append(
            f"[SAT-TEST] scenario={scenario} selected={selected} "
            f"overall_tx={_safe_percent(agg.get('tx_acc'))}% "
            f"strict_udu={_safe_percent(stats.get('strict_udu'))}% "
            f"({int(agg.get('tx_correct', 0))}/{int(agg.get('tx_total', 0))})"
        )
    return lines


def format_ssdg_epoch_block(
    *,
    epoch: int,
    epochs: int,
    lr: float,
    epoch_time_s: float,
    phase: str,
    train_logs: Mapping[str, Any],
    val_stats: Mapping[str, Any],
    test_stats: Mapping[str, Any] | None = None,
    named_test_stats: Mapping[str, Mapping[str, Any]] | None = None,
    named_test_meta: Mapping[str, Mapping[str, Any]] | None = None,
    sat_test_stats: Mapping[str, Mapping[str, Any]] | None = None,
    stage_state: Mapping[str, Any] | None = None,
    mixstyle_state: Mapping[str, Any] | None = None,
    aug_state: Mapping[str, Any] | None = None,
    loss_weights: Mapping[str, float] | None = None,
    best_val: float,
    best_test: float = float("nan"),
    best_epoch: int | None = None,
    latest_path: str,
    best_path: str,
    is_best: bool,
    protected_metrics: Mapping[str, Any] | None = None,
    guard_state: Mapping[str, Any] | None = None,
    phase2_audit_state: Mapping[str, Any] | None = None,
    safe_latest_path: str = "",
    safe_best_path: str = "",
    safe_checkpoint_saved: bool = False,
) -> str:
    sep = "=" * 132
    minor = "-" * 132
    loss_total = _log_value(train_logs, "train/loss")
    loss_cls = _log_value(train_logs, "train/loss_tx_labeled")
    loss_dom = _log_value(train_logs, "train/loss_domain_labeled")
    loss_adv = _log_value(train_logs, "train/loss_adv_labeled")
    loss_orth = _log_value(train_logs, "train/loss_orth_labeled")
    loss_group_ce = _log_value(train_logs, "train/loss_group_ce_labeled")
    loss_sat = _log_value(train_logs, "train/loss_sat_cls_labeled")
    loss_sat_cons = _log_value(train_logs, "train/loss_sat_cons_labeled")
    loss_fishr = _log_value(train_logs, "train/loss_fishr_labeled")
    loss_proto = _log_value(train_logs, "train/loss_proto_labeled")
    loss_ow_feat = _log_value(train_logs, "train/loss_open_world_feat")
    loss_soft_unknown_mixup = _log_value(train_logs, "train/loss_soft_unknown_mixup")
    loss_source_episode = _log_value(train_logs, "train/loss_source_episode")
    loss_direct_metric_accept = _log_value(train_logs, "train/loss_direct_metric_accept")
    loss_cons = _log_value(train_logs, "train/loss_cons_labeled")
    loss_u = _log_value(train_logs, "train/loss_unlabeled")
    w_cls = _log_value(train_logs, "train/w_loss_tx_labeled", loss_cls)
    w_dom = _log_value(train_logs, "train/w_loss_domain_labeled", loss_dom)
    w_adv = _log_value(train_logs, "train/w_loss_adv_labeled", loss_adv)
    w_orth = _log_value(train_logs, "train/w_loss_orth_labeled", loss_orth)
    w_cons = _log_value(train_logs, "train/w_loss_cons_labeled", 0.0)
    w_group_ce = _log_value(train_logs, "train/w_loss_group_ce_labeled", loss_group_ce)
    w_sat = _log_value(train_logs, "train/w_loss_sat_cls_labeled", loss_sat)
    w_sat_cons = _log_value(train_logs, "train/w_loss_sat_cons_labeled", loss_sat_cons)
    w_fishr = _log_value(train_logs, "train/w_loss_fishr_labeled", loss_fishr)
    w_proto = _log_value(train_logs, "train/w_loss_proto_labeled", loss_proto)
    w_ow_feat = _log_value(train_logs, "train/w_loss_open_world_feat", loss_ow_feat)
    w_soft_unknown_mixup = _log_value(train_logs, "train/w_loss_soft_unknown_mixup", loss_soft_unknown_mixup)
    w_source_episode = _log_value(train_logs, "train/w_loss_source_episode", loss_source_episode)
    w_direct_metric_accept = _log_value(train_logs, "train/w_loss_direct_metric_accept", loss_direct_metric_accept)
    w_u_domain = _log_value(train_logs, "train/w_loss_u_domain", 0.0)
    w_u_adv = _log_value(train_logs, "train/w_loss_u_adv", 0.0)
    w_u_sat_cons = _log_value(train_logs, "train/w_loss_u_sat_cons", 0.0)
    w_u_direct_metric = _log_value(train_logs, "train/w_loss_u_direct_metric_accept", 0.0)
    w_u_quarantine = _log_value(train_logs, "train/w_loss_u_quarantine_accept", 0.0)
    reliable = _log_value(train_logs, "train/reliable_ratio")
    pseudo_conf = _log_value(train_logs, "train/pseudo_conf")
    domain_pass = _log_value(train_logs, "train/domain_pass")
    temporal_pass = _log_value(train_logs, "train/temporal_pass")
    strong_pass = _log_value(train_logs, "train/strong_pass")
    pseudo_total = int(round(_log_value(train_logs, "train/pseudo_total")))
    pseudo_selected = int(round(_log_value(train_logs, "train/pseudo_selected")))
    pseudo_correct = int(round(_log_value(train_logs, "train/pseudo_correct")))
    pseudo_precision = 100.0 * pseudo_correct / max(1, pseudo_selected)
    pseudo_truth_available = _log_value(
        train_logs, "train/pseudo_truth_available", 1.0
    ) > 0.0
    test_stats = test_stats or {}
    named_test_stats = named_test_stats or {}
    named_test_meta = named_test_meta or {}
    sat_test_stats = sat_test_stats or {}
    protected_metrics = protected_metrics or {}
    guard_state = guard_state or {}
    phase2_audit_state = phase2_audit_state or {}
    if best_epoch is None:
        best_epoch = int(epoch)
    if stage_state is None:
        stage_state = _fallback_stage_state(phase)
    if mixstyle_state is None:
        mixstyle_state = _fallback_mixstyle_state(type("Args", (), {"use_mixstyle": True, "mixstyle_p": 0.18, "mixstyle_strength": 0.70})())
    if aug_state is None:
        aug_state = _fallback_aug_state(type("Args", (), {})())
    loss_weights = dict(loss_weights or {})
    lines = [sep]
    lines.append(f"[EPOCH-BEGIN] E{int(epoch):03d}/{int(epochs):03d} | time={float(epoch_time_s):.1f}s | lr={float(lr):.2e} | aux_scale=0.000")
    lines.append(f"[STAGE] {_format_stage_line(stage_state, phase)} label_stage={int(str(phase) == 'label')} pseudo_stage={int(str(phase) == 'pseudo')}")
    lines.append(
        "[MIXSTYLE-EPOCH] "
        f"phase={mixstyle_state.get('phase', 'unknown')} enabled={int(bool(mixstyle_state.get('enabled', False)))} "
        f"p={float(mixstyle_state.get('p', 0.0)):.3f} "
        f"strength={float(mixstyle_state.get('strength', 0.0)):.3f} "
        f"anneal_t={float(mixstyle_state.get('anneal_t', 0.0)):.3f}"
    )
    if aug_state is not None:
        lines.append(
            "[AUG] "
            f"scale={float(aug_state.get('scale', 0.0)):.3f} | "
            f"p_dac={float(aug_state.get('p_dac', 0.0)):.3f} "
            f"p_pa={float(aug_state.get('p_pa', 0.0)):.3f} "
            f"p_shift={float(aug_state.get('p_time_shift', 0.0)):.3f} "
            f"p_cfo={float(aug_state.get('p_cfo', 0.0)):.3f} "
            f"p_awgn={float(aug_state.get('p_awgn', 0.0)):.3f} "
            f"p_mp={float(aug_state.get('p_multipath', 0.0)):.3f} | "
            f"max_shift={int(aug_state.get('max_time_shift', 0))} "
            f"cfo_max={float(aug_state.get('cfo_max', 0.0)):.4g} "
            f"pn_max={float(aug_state.get('phase_noise_sigma_max', 0.0)):.4g}"
        )
    else:
        lines.append("[AUG] disabled")
    lines.append(minor)
    lines.append(
        "[LOSS-CORE-RAW] "
        f"total={loss_total:.4f} cls={loss_cls:.4f} dom={loss_dom:.4f} "
        f"adv={loss_adv:.4f} orth={loss_orth:.4f} cons={loss_cons:.4f} group_ce={loss_group_ce:.4f}"
    )
    lines.append(
        "[LOSS-CORE-W]   "
        f"cls={w_cls:.4f} dom={w_dom:.4f} adv={w_adv:.4f} "
        f"orth={w_orth:.4f} cons={w_cons:.4f} group_ce={w_group_ce:.4f}"
    )
    lines.append("[LOSS-AUX-RAW]  cls_pa=0.0000 cls_dac=0.0000 pa_joint_inv=0.0000 pa_kl=0.0000 dac_reg=0.0000 pa_reg=0.0000")
    lines.append("[LOSS-AUX-W]    cls_pa=0.0000 cls_dac=0.0000 pa_joint_inv=0.0000 pa_kl=0.0000 dac_reg=0.0000 pa_reg=0.0000")
    lines.append(f"[LOSS-SAT-RAW]  cls_sat={loss_sat:.4f} sat_cons={loss_sat_cons:.4f} sat_cos=nan")
    lines.append(f"[LOSS-SAT-W]    cls_sat={w_sat:.4f} sat_cons={w_sat_cons:.4f}")
    lines.append(
        f"[LOSS-DG-RAW]   proto={loss_proto:.4f} "
        f"proto_cos={_log_value(train_logs, 'train/proto_pull_cos'):.4f} "
        f"supcon=0.0000 fishr={loss_fishr:.4f} ow_feat={loss_ow_feat:.4f} "
        f"soft_unknown_mixup={loss_soft_unknown_mixup:.4f} source_episode={loss_source_episode:.4f} "
        f"direct_metric={loss_direct_metric_accept:.4f}"
    )
    lines.append(
        f"[LOSS-DG-W]     proto={w_proto:.4f} supcon=0.0000 fishr={w_fishr:.4f} "
        f"ow_feat={w_ow_feat:.4f} soft_unknown_mixup={w_soft_unknown_mixup:.4f} source_episode={w_source_episode:.4f} "
        f"direct_metric={w_direct_metric_accept:.4f}"
    )
    lines.append(
        "[OW-FEAT] "
        f"active_classes={_log_value(train_logs, 'train/ow_feat_active_classes'):.1f} "
        f"compact={_log_value(train_logs, 'train/ow_feat_compact'):.4f} "
        f"inter={_log_value(train_logs, 'train/ow_feat_inter'):.4f} "
        f"sample_margin={_log_value(train_logs, 'train/ow_feat_sample_margin'):.4f} "
        f"domain_align={_log_value(train_logs, 'train/ow_feat_domain_align'):.4f} "
        f"pos_angle={_log_value(train_logs, 'train/ow_feat_pos_angle_deg'):.2f}deg "
        f"p95={_log_value(train_logs, 'train/ow_feat_pos_angle_p95_deg'):.2f}deg "
        f"tail3s={_log_value(train_logs, 'train/ow_feat_tail_frac_gt_3sigma'):.4f} "
        f"min_inter={_log_value(train_logs, 'train/ow_feat_min_inter_deg'):.2f}deg "
        f"vac={_log_value(train_logs, 'train/ow_feat_vacuum_loss'):.4f} "
        f"vac_rate={_log_value(train_logs, 'train/ow_feat_vacuum_violation_rate'):.4f} "
        f"vac_gap={_log_value(train_logs, 'train/ow_feat_vacuum_margin_deg'):.2f}deg "
        f"proto_active={_log_value(train_logs, 'train/proto_active_classes'):.1f}"
    )
    lines.append(
        "[PROXY-UNK] "
        f"active={_log_value(train_logs, 'train/proxy_unknown_active'):.1f} "
        f"known={_log_value(train_logs, 'train/proxy_unknown_known_count'):.0f} "
        f"proxy={_log_value(train_logs, 'train/proxy_unknown_count'):.0f} "
        f"virtual={_log_value(train_logs, 'train/proxy_unknown_virtual_count'):.0f} "
        f"core={_log_value(train_logs, 'train/proxy_unknown_core_count'):.0f} "
        f"tail={_log_value(train_logs, 'train/proxy_unknown_tail_count'):.0f} "
        f"auc={_log_value(train_logs, 'train/proxy_unknown_auc_proxy'):.4f} "
        f"vaccept={_log_value(train_logs, 'train/proxy_unknown_virtual_accept_rate'):.4f} "
        f"reject_claim={_log_value(train_logs, 'train/proxy_unknown_proxy_reject_claim_allowed'):.0f} "
        f"core_accept={_log_value(train_logs, 'train/proxy_unknown_virtual_accept_rate_core'):.4f} "
        f"hard_accept={_log_value(train_logs, 'train/proxy_unknown_hard_proxy_accept_rate'):.4f} "
        f"vaccept_surr={_log_value(train_logs, 'train/proxy_unknown_vaccept_surrogate'):.4f} "
        f"gate={_log_value(train_logs, 'train/proxy_unknown_component_gate_unknown'):.4f} "
        f"shell={_log_value(train_logs, 'train/proxy_unknown_shell_accept_rate'):.4f} "
        f"bridge={_log_value(train_logs, 'train/proxy_unknown_bridge_accept_rate'):.4f} "
        f"outward={_log_value(train_logs, 'train/proxy_unknown_outward_accept_rate'):.4f} "
        f"vac={_log_value(train_logs, 'train/proxy_unknown_vacuum_loss'):.4f} "
        f"vac_rate={_log_value(train_logs, 'train/proxy_unknown_vacuum_violation_rate'):.4f} "
        f"vac_gap={_log_value(train_logs, 'train/proxy_unknown_vacuum_margin_deg'):.2f}deg"
    )
    lines.append(
        "[PROXY-ADG] "
        f"bridge_loss={_log_value(train_logs, 'train/proxy_unknown_bridge_governance_loss'):.4f} "
        f"shell_out={_log_value(train_logs, 'train/proxy_unknown_shell_outward_accept_loss'):.4f} "
        f"low_den={_log_value(train_logs, 'train/proxy_unknown_low_density_accept_loss'):.4f} "
        f"e_q={_log_value(train_logs, 'train/proxy_unknown_energy_margin_quantile_loss'):.4f} "
        f"e_q05={_log_value(train_logs, 'train/proxy_unknown_energy_margin_q05'):.4f} "
        f"e_q10={_log_value(train_logs, 'train/proxy_unknown_energy_margin_q10'):.4f} "
        f"r_p95={_log_value(train_logs, 'train/proxy_unknown_component_radius_p95_deg'):.2f}deg "
        f"r_max={_log_value(train_logs, 'train/proxy_unknown_component_radius_max_deg'):.2f}deg "
        f"gate_r95={_log_value(train_logs, 'train/proxy_unknown_component_gate_radius_p95_deg'):.2f}deg "
        f"gate_rmax={_log_value(train_logs, 'train/proxy_unknown_component_gate_radius_max_deg'):.2f}deg "
        f"low_den_rate={_log_value(train_logs, 'train/proxy_unknown_low_density_accept_rate'):.4f} "
        f"r_inter={_log_value(train_logs, 'train/proxy_unknown_radius_inter_ratio'):.4f}"
    )
    lines.append(
        "[SOFT-UNK-MIX] "
        f"count={_log_value(train_logs, 'train/soft_unknown_mixup_count'):.0f} "
        f"order={_log_value(train_logs, 'train/soft_unknown_mixup_order'):.0f} "
        f"ce={_log_value(train_logs, 'train/soft_unknown_mixup_ce'):.4f} "
        f"energy={_log_value(train_logs, 'train/soft_unknown_mixup_energy'):.4f} "
        f"vac={_log_value(train_logs, 'train/soft_unknown_mixup_vacuum'):.4f} "
        f"vaccept={_log_value(train_logs, 'train/soft_unknown_mixup_virtual_accept_rate'):.4f} "
        f"vac_rate={_log_value(train_logs, 'train/soft_unknown_mixup_vacuum_violation'):.4f}"
    )
    lines.append(
        "[SOURCE-EP] "
        f"classes={_log_value(train_logs, 'train/source_episode_classes'):.1f} "
        f"domains={_log_value(train_logs, 'train/source_episode_domains'):.1f} "
        f"overflow={_log_value(train_logs, 'train/source_episode_overflow_rate'):.4f} "
        f"r3s={_log_value(train_logs, 'train/source_episode_radius_3sigma_deg'):.2f}deg "
        f"r_core={_log_value(train_logs, 'train/source_episode_radius_core_deg'):.2f}deg "
        f"r_safe={_log_value(train_logs, 'train/source_episode_radius_safe_deg'):.2f}deg "
        f"tail_q={_log_value(train_logs, 'train/source_episode_tail_query_rate'):.4f} "
        f"val_angle={_log_value(train_logs, 'train/source_episode_val_angle_deg'):.2f}deg "
        f"mix_count={_log_value(train_logs, 'train/source_episode_mixup_count'):.0f} "
        f"mix_order={_log_value(train_logs, 'train/source_episode_mixup_order'):.0f} "
        f"mix_loss={_log_value(train_logs, 'train/source_episode_mixup_loss'):.4f} "
        f"mix_overflow={_log_value(train_logs, 'train/source_episode_mixup_overflow_rate'):.4f} "
        f"mix_gap={_log_value(train_logs, 'train/source_episode_mixup_margin_deg'):.2f}deg"
    )
    lines.append(
        "[DM-ACCEPT] "
        f"active={_log_value(train_logs, 'train/dm_accept_active'):.1f} "
        f"classes={_log_value(train_logs, 'train/dm_accept_active_classes'):.1f} "
        f"p50={_log_value(train_logs, 'train/dm_accept_zid_p50_deg'):.2f}deg "
        f"p95={_log_value(train_logs, 'train/dm_accept_zid_p95_deg'):.2f}deg "
        f"p99={_log_value(train_logs, 'train/dm_accept_zid_p99_deg'):.2f}deg "
        f"tail_cvar={_log_value(train_logs, 'train/dm_accept_zid_tail_cvar_deg'):.2f}deg "
        f"source_overflow={_log_value(train_logs, 'train/dm_accept_source_overflow'):.4f} "
        f"proxy_vaccept={_log_value(train_logs, 'train/dm_accept_proxy_vaccept'):.4f} "
        f"bridge={_log_value(train_logs, 'train/dm_accept_bridge_accept_rate'):.4f} "
        f"low_den={_log_value(train_logs, 'train/dm_accept_low_density_accept_rate'):.4f} "
        f"tail_accept={_log_value(train_logs, 'train/dm_accept_tail_accept_rate'):.4f} "
        f"overflow_accept={_log_value(train_logs, 'train/dm_accept_overflow_accept_rate'):.4f} "
        f"radius_inter={_log_value(train_logs, 'train/dm_accept_radius_to_inter_ratio'):.4f} "
        f"core_accept={_log_value(train_logs, 'train/dm_accept_core_accept_rate'):.4f} "
        f"known_tpr={_log_value(train_logs, 'train/dm_accept_known_hard_tpr'):.4f} "
        f"neg_scale={_log_value(train_logs, 'train/dm_accept_negative_risk_scale'):.3f} "
        f"proxy_grad={_log_value(train_logs, 'train/dm_accept_proxy_gradient_active'):.0f} "
        f"sat_pair_p95={_log_value(train_logs, 'train/dm_accept_sat_pair_angle_p95_deg'):.2f}deg"
    )
    lines.append(
        "[U-DIRECT] "
        f"loss_domain={_log_value(train_logs, 'train/loss_u_domain'):.4f} "
        f"loss_adv={_log_value(train_logs, 'train/loss_u_adv'):.4f} "
        f"loss_sat={_log_value(train_logs, 'train/loss_u_sat_cons'):.4f} "
        f"loss_dm={_log_value(train_logs, 'train/loss_u_direct_metric_accept'):.4f} "
        f"loss_quarantine={_log_value(train_logs, 'train/loss_u_quarantine_accept'):.4f} "
        f"selected={_log_value(train_logs, 'train/u_dm_accept_selected'):.0f} "
        f"q_active={_log_value(train_logs, 'train/u_quarantine_active'):.1f} "
        f"q_count={_log_value(train_logs, 'train/u_quarantine_query_count'):.0f} "
        f"q_accept={_log_value(train_logs, 'train/u_quarantine_accept_rate'):.4f} "
        f"q_low_den={_log_value(train_logs, 'train/u_quarantine_low_density_accept_rate'):.4f} "
        f"dm_active={_log_value(train_logs, 'train/u_dm_accept_active'):.1f} "
        f"sat_pairs={_log_value(train_logs, 'train/u_dm_accept_sat_pair_count'):.0f} "
        f"p50={_log_value(train_logs, 'train/u_dm_accept_zid_p50_deg'):.2f}deg "
        f"p95={_log_value(train_logs, 'train/u_dm_accept_zid_p95_deg'):.2f}deg "
        f"p99={_log_value(train_logs, 'train/u_dm_accept_zid_p99_deg'):.2f}deg "
        f"tail_cvar={_log_value(train_logs, 'train/u_dm_accept_zid_tail_cvar_deg'):.2f}deg "
        f"source_overflow={_log_value(train_logs, 'train/u_dm_accept_source_overflow'):.4f} "
        f"proxy_vaccept={_log_value(train_logs, 'train/u_dm_accept_proxy_vaccept'):.4f} "
        f"bridge={_log_value(train_logs, 'train/u_dm_accept_bridge_accept_rate'):.4f} "
        f"low_den={_log_value(train_logs, 'train/u_dm_accept_low_density_accept_rate'):.4f} "
        f"radius_inter={_log_value(train_logs, 'train/u_dm_accept_radius_to_inter_ratio'):.4f} "
        f"sat_pair_p95={_log_value(train_logs, 'train/u_dm_accept_sat_pair_angle_p95_deg'):.2f}deg"
    )
    lines.append(
        "[LOSS-WEIGHT] "
        f"dom={float(stage_state.get('dom_scale', loss_weights.get('dom', 0.0))):.3f} "
        f"adv={float(stage_state.get('adv_scale', loss_weights.get('adv', 0.0))):.3f} "
        f"orth={float(stage_state.get('orth_scale', loss_weights.get('orth', 0.0))):.3f} "
        f"cons={float(stage_state.get('cons_scale', loss_weights.get('cons', 0.0))):.3f} "
        f"group_ce={float(stage_state.get('group_ce_scale', loss_weights.get('group_ce', 0.0))):.3f} "
        f"proto={float(loss_weights.get('proto', 0.0)):.6g} "
        f"ow_feat={float(loss_weights.get('open_world_feat', 0.0)):.6g} "
        f"soft_unknown_mixup={float(loss_weights.get('soft_unknown_mixup', 0.0)):.6g} "
        f"source_episode={float(loss_weights.get('source_episode', 0.0)):.6g} "
        f"direct_metric={float(loss_weights.get('direct_metric_accept', 0.0)):.6g} "
        f"u_domain={float(loss_weights.get('u_domain', 0.0)):.6g} "
        f"u_adv={float(loss_weights.get('u_adv', 0.0)):.6g} "
        f"u_sat_cons={float(loss_weights.get('u_sat_cons', 0.0)):.6g} "
        f"u_dm={float(loss_weights.get('u_direct_metric_accept', 0.0)):.6g} "
        f"u_quarantine={float(loss_weights.get('u_quarantine_accept', 0.0)):.6g} "
        "aux_scale=0.000"
    )
    lines.append(
        _format_loss_top(
            {
                "cls": w_cls,
                "dom": w_dom,
                "adv": w_adv,
                "orth": w_orth,
                "cons": w_cons,
                "group_ce": w_group_ce,
                "cls_sat": w_sat,
                "sat_cons": w_sat_cons,
                "fishr": w_fishr,
                "proto": w_proto,
                "ow_feat": w_ow_feat,
                "soft_unknown_mixup": w_soft_unknown_mixup,
                "source_episode": w_source_episode,
                "direct_metric": w_direct_metric_accept,
                "u_domain": w_u_domain,
                "u_adv": w_u_adv,
                "u_sat_cons": w_u_sat_cons,
                "u_dm": w_u_direct_metric,
                "u_quarantine": w_u_quarantine,
            }
        )
    )
    lines.append(
        "[LOSS-PSEUDO]   "
        f"u={loss_u:.4f} reliable={reliable:.3f} conf={pseudo_conf:.3f} "
        f"domain_pass={domain_pass:.3f} temporal_pass={temporal_pass:.3f} strong_pass={strong_pass:.3f} "
        f"total={pseudo_total} selected={pseudo_selected}/{pseudo_total} "
        + (
            f"correct={pseudo_correct} precision={pseudo_precision:.3f}%"
            if pseudo_truth_available
            else "correct=N/A precision=N/A"
        )
    )
    lines.append(minor)
    lines.append(
        f"[TRAIN] tx={_safe_percent(train_logs.get('train/tx_acc'))}% "
        f"dom={_safe_percent(train_logs.get('train/dom_acc'))}% "
        f"cons_cos={_log_value(train_logs, 'train/cons_cos'):.4f}"
    )
    lines.append(
        f"[GRAD]  total={_log_value(train_logs, 'train/grad_total'):.3f} "
        f"backbone={_log_value(train_logs, 'train/grad_backbone'):.3f} "
        f"aux={_log_value(train_logs, 'train/grad_aux'):.3f} "
        f"domain={_log_value(train_logs, 'train/grad_domain'):.3f}"
    )
    lines.append(
        "[OS-GRAD] "
        f"pre_budget={_log_value(train_logs, 'train/os_budget_controller_pre'):.4f} "
        f"post_budget={_log_value(train_logs, 'train/os_budget_controller_post'):.4f} "
        f"os_scale={_log_value(train_logs, 'train/os_budget_controller_os_scale', 1.0):.3f} "
        f"closed_scale={_log_value(train_logs, 'train/os_budget_controller_closed_scale', 1.0):.3f} "
        f"conflict={int(_log_value(train_logs, 'train/os_gradient_conflict', 0.0) >= 0.5)} "
        f"step_rate={_log_value(train_logs, 'train/optimizer_step_applied'):.3f}"
    )
    lines.append(f"[VAL]   tx={_safe_percent(val_stats.get('tx_acc'))}% dom={_safe_percent(val_stats.get('dom_acc'))}%")
    if math.isfinite(_log_value(stage_state, "source_val_sat_mean_tx", float("nan"))):
        lines.append(
            "[SOURCE-VAL-SAT] "
            f"mean_tx={_safe_percent(stage_state.get('source_val_sat_mean_tx'))}% "
            f"floor_tx={_safe_percent(stage_state.get('source_val_sat_floor_tx'))}% "
            "selection_safe=1"
        )
    lines.append(
        f"[TEST]  overall_tx={_safe_percent(test_stats.get('tx_acc'))}% "
        f"({int(test_stats.get('tx_correct', 0))}/{int(test_stats.get('tx_total', 0))})"
    )
    lines.append("[TEST-SPLIT]")
    lines.extend(_format_named_test_lines(named_test_stats, named_test_meta))
    if sat_test_stats:
        lines.extend(_format_sat_test_lines(sat_test_stats))
    if protected_metrics:
        lines.append(
            "[JOINT-METRIC] "
            f"strict_udu={_safe_percent(protected_metrics.get('strict_udu'))}% "
            f"receiver_floor={_safe_percent(protected_metrics.get('receiver_floor'))}% "
            f"sat_mean={_safe_percent(protected_metrics.get('sat_mean_tx'))}% "
            f"sat_floor={_safe_percent(protected_metrics.get('sat_floor_tx'))}% "
            f"sat_strict_mean={_safe_percent(protected_metrics.get('sat_strict_mean'))}% "
            f"sat_strict_floor={_safe_percent(protected_metrics.get('sat_strict_floor'))}%"
        )
    if guard_state:
        lines.append(
            "[JOINT-GUARD] "
            f"enabled={int(bool(guard_state.get('enabled', False)))} "
            f"safe={int(bool(guard_state.get('checkpoint_safe', True)))} "
            f"missing_required={int(guard_state.get('missing_required_metric_count', 0))} "
            f"drop={int(bool(guard_state.get('drop_guard_fired', False)))} "
            f"paic={int(bool(guard_state.get('paic_guard_fired', False)))} "
            f"cooldown_active={int(bool(guard_state.get('paic_cooldown_active', False)))} "
            f"reason={guard_state.get('reason', '')}"
        )
    if phase2_audit_state and bool(phase2_audit_state.get("requested", False)):
        imports = phase2_audit_state.get("imports", {}) if isinstance(phase2_audit_state.get("imports"), Mapping) else {}
        weights = phase2_audit_state.get("weights", {}) if isinstance(phase2_audit_state.get("weights"), Mapping) else {}
        lines.append(
            "[PROTO-TX] "
            f"enabled={int(bool(phase2_audit_state.get('use_phase2_ground_prototypes', False)))} "
            f"audit_only={int(bool(phase2_audit_state.get('audit_only', True)))} "
            f"lambda_tx_proto={float(weights.get('tx_proto', 0.0)):.6g} "
            f"module_import={int(bool(imports.get('phase2_prototypes', 0)))}"
        )
        lines.append(
            "[PROTO-RX] "
            f"enabled={int(bool(phase2_audit_state.get('use_phase2_ground_prototypes', False)))} "
            f"lambda_rx_proto={float(weights.get('rx_proto', 0.0)):.6g} "
            f"module_import={int(bool(imports.get('phase2_prototypes', 0)))}"
        )
        lines.append(
            "[MASK] "
            f"enabled={int(bool(phase2_audit_state.get('use_feature_masks', False)))} "
            f"lambda_mask_aux={float(weights.get('mask_aux', 0.0)):.6g} "
            f"module_import={int(bool(imports.get('feature_masks', 0)))}"
        )
        lines.append(
            "[BATCH-GEOM] "
            f"balanced_sampler={int(bool(phase2_audit_state.get('use_tx_rx_balanced_sampler', False)))} "
            f"geometry_losses={int(bool(phase2_audit_state.get('use_txrx_geometry_losses', False)))} "
            f"lambda_txrx_rect={float(weights.get('txrx_rect', 0.0)):.6g} "
            f"sampler_import={int(bool(imports.get('balanced_tx_rx_sampler', 0)))} "
            f"geometry_import={int(bool(imports.get('tx_rx_geometry', 0)))}"
        )
        lines.append(
            "[TXRX-ANOVA] "
            "status=audit_marker_only "
            f"active_loss={int(bool(phase2_audit_state.get('active_loss', False)))}"
        )
        lines.append(
            "[ZID-FEATURE-SPACE] "
            f"proto_memory={int(bool(phase2_audit_state.get('use_proto_memory', False)))} "
            f"lambda_proto={float(weights.get('proto_memory', 0.0)):.6g} "
            f"open_world_feat={int(bool(phase2_audit_state.get('use_open_world_feature_loss', False)))} "
            f"lambda_open_world_feat={float(weights.get('open_world_feat', 0.0)):.6g} "
            f"zid_compact={int(bool(phase2_audit_state.get('use_zid_compactness_loss', False)))} "
            f"lambda_zid_compact={float(weights.get('zid_compact', 0.0)):.6g} "
            f"proxy_unknown={int(bool(phase2_audit_state.get('use_proxy_unknown_loss', False)))} "
            f"lambda_proxy_unknown={float(weights.get('proxy_unknown', 0.0)):.6g} "
            f"soft_unknown_mixup={int(bool(phase2_audit_state.get('use_soft_unknown_mixup_loss', False)))} "
            f"lambda_soft_unknown_mixup={float(weights.get('soft_unknown_mixup', 0.0)):.6g} "
            f"source_episode={int(bool(phase2_audit_state.get('use_source_episode_loss', False)))} "
            f"lambda_source_episode={float(weights.get('source_episode', 0.0)):.6g} "
            f"phase2_export={int(bool(phase2_audit_state.get('phase2_export_prototypes', False)))}"
        )
    if int(best_epoch) > 0 and math.isfinite(float(best_val)):
        test_note = f"{float(best_test):.2f}%" if math.isfinite(float(best_test)) else "not_evaluated_on_frozen_best"
        lines.append(
            f"[BEST-SOURCE-VAL] val_tx={float(best_val):.2f}% @ E{int(best_epoch):03d} "
            f"heldout_test={test_note}"
        )
        lines.append(
            "[TAIL-CONTROL] "
            f"state={int(guard_state.get('phase1_v2_tail_state_code', -1))} "
            f"action={int(guard_state.get('phase1_v2_tail_action_code', -1))} "
            f"reference_ready={int(bool(guard_state.get('phase1_v2_tail_reference_ready', 0)))} "
            f"reference_saved={int(bool(guard_state.get('phase1_v2_tail_reference_saved', False)))} "
            f"rollback_applied={int(bool(guard_state.get('phase1_v2_tail_rollback_applied', False)))} "
            f"final_blocked={int(bool(guard_state.get('phase1_v2_final_blocked', False)))}"
        )
    else:
        lines.append("[BEST-SOURCE-VAL] status=not_saved reason=no_guard_safe_source_val_checkpoint")
    if bool(guard_state.get("enabled", False)) and (safe_latest_path or safe_best_path):
        safe_latest_exists = bool(safe_latest_path and Path(safe_latest_path).is_file())
        safe_best_exists = bool(safe_best_path and Path(safe_best_path).is_file())
        lines.append(
            f"[SAFE-CKPT] latest_safe={safe_latest_path or '-'} "
            f"status={'saved' if safe_latest_exists else 'not_saved'} | "
            f"best_safe={safe_best_path or '-'} status={'saved' if safe_best_exists else 'not_saved'} "
            f"saved_this_epoch={int(bool(safe_checkpoint_saved))}"
        )
    lines.append(
        f"[CKPT]  latest -> {latest_path} (recovery) | final_only -> {best_path} "
        f"(written after training; telemetry_best={int(bool(is_best))})"
    )
    lines.append(f"[EPOCH-END] E{int(epoch):03d}/{int(epochs):03d}")
    lines.append(sep)
    return "\n".join(lines)


def _safe_iq_tensor(x):
    if torch is None:
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_concat_sat_batch_for_training(
    concat_sat_aug,
    x,
    y,
    d,
    *,
    args,
    epoch: int,
    batch_idx: int,
):
    """Apply EPOC-style clean+satellite expansion for the full SSDG objective."""
    bsz = int(x.size(0)) if hasattr(x, "size") else 0
    info: Dict[str, Any] = {
        "active": 0.0,
        "expanded": 0.0,
        "applied": 0.0,
        "clean_batch_size": float(bsz),
        "total_batch_size": float(bsz),
        "view_prob": float(getattr(args, "sat_view_prob", 1.0)),
        "stage_start_epoch": float("nan"),
        "stage_index": float("nan"),
        "scenario_code": float("nan"),
    }
    if concat_sat_aug is None or int(epoch) < int(getattr(args, "concat_sat_start_epoch", 1)):
        return x, y, d, None, info
    if str(getattr(args, "sat_training_mode", "")) in {"concat_ce_only", "concat_masked"} or bool(
        getattr(args, "concat_sat_ce_only", False)
    ):
        sat_view = concat_sat_aug.transform(x, args=args, epoch=epoch, batch_idx=batch_idx)
        info.update(
            {
                "active": 1.0,
                "applied": 1.0 if bool(sat_view.applied) else 0.0,
                "clean_batch_size": float(int(sat_view.clean_batch_size)),
                "total_batch_size": float(int(sat_view.clean_batch_size)),
                "view_prob": float(sat_view.view_prob),
                "stage_start_epoch": float(int(sat_view.stage_start_epoch)),
                "stage_index": float(int(sat_view.stage_index)),
                "scenario_code": float(abs(hash(str(sat_view.scenario))) % 1000000),
                "crra_nuisance": sat_view.nuisance,
                "crra_nuisance_valid": sat_view.nuisance_valid,
                "crra_nuisance_fields": sat_view.nuisance_fields,
                "crra_meta": sat_view.meta,
            }
        )
        return x, y, d, sat_view, info
    concat_batch = concat_sat_aug.expand(
        x,
        y,
        d,
        args=args,
        epoch=epoch,
        batch_idx=batch_idx,
    )
    info.update(
        {
            "active": 1.0,
            "expanded": 1.0,
            "applied": 1.0 if bool(concat_batch.applied) else 0.0,
            "clean_batch_size": float(int(concat_batch.clean_batch_size)),
            "total_batch_size": float(int(concat_batch.total_batch_size)),
            "view_prob": float(concat_batch.view_prob),
            "stage_start_epoch": float(int(concat_batch.stage_start_epoch)),
            "stage_index": float(int(concat_batch.stage_index)),
            "scenario_code": float(abs(hash(str(concat_batch.scenario))) % 1000000),
            "crra_nuisance": concat_batch.nuisance,
            "crra_nuisance_valid": concat_batch.nuisance_valid,
            "crra_nuisance_fields": concat_batch.nuisance_fields,
            "crra_meta": concat_batch.meta,
        }
    )
    return _safe_iq_tensor(concat_batch.x), concat_batch.y, concat_batch.d_raw, None, info


def _resolve_sat_training_mode(args) -> str:
    """Resolve the explicit satellite path while retaining legacy flags."""
    explicit = str(getattr(args, "sat_training_mode", "") or "").strip().lower()
    if not explicit:
        if not bool(getattr(args, "use_concat_sat_channel_aug", False)):
            explicit = "view_aux" if bool(getattr(args, "use_sat_consistency", True)) else "disabled"
        elif bool(getattr(args, "concat_sat_ce_only", False)):
            explicit = "concat_ce_only"
        else:
            explicit = "concat_legacy"
    if explicit in {"concat_ce_only", "concat_masked"}:
        args.use_concat_sat_channel_aug = True
        args.concat_sat_ce_only = True
    elif explicit == "concat_full":
        args.use_concat_sat_channel_aug = True
        args.concat_sat_ce_only = False
    elif explicit == "concat_legacy":
        args.use_concat_sat_channel_aug = True
    elif explicit in {"disabled", "view_aux"}:
        args.use_concat_sat_channel_aug = False
        args.concat_sat_ce_only = False
    else:
        raise ValueError(f"unsupported satellite training mode: {explicit}")
    args.sat_training_mode = explicit
    return explicit


def _crra_satellite_aux_regularizers_active(args) -> bool:
    """Keep concat_masked satellite supervision limited to its registered losses."""
    return str(getattr(args, "sat_training_mode", "") or "").strip().lower() != "concat_masked"


def _crra_satellite_kl_active(
    legacy_sat_cons_weight: float,
    *,
    use_crra: bool,
    crra_stage_scale: float,
    crra_sat_kl_weight: float,
) -> bool:
    """Whether clean-to-satellite KL must be materialized for this batch."""
    return bool(
        float(legacy_sat_cons_weight) > 0.0
        or (
            bool(use_crra)
            and float(crra_stage_scale) > 0.0
            and float(crra_sat_kl_weight) > 0.0
        )
    )


def _muse_level_capabilities(level: str) -> Dict[str, bool]:
    normalized = str(level).strip().upper()
    if normalized not in {"M0", "M1", "M2", "M3"}:
        raise ValueError(f"unsupported MUSE level: {level}")
    rank = int(normalized[1])
    return {
        "base": rank >= 1,
        "fusion": rank >= 2,
        "satellite": rank >= 3,
    }


def _muse_u_prototype_update_enabled(args, capabilities: Mapping[str, bool]) -> bool:
    override = getattr(args, "muse_enable_u_prototype_update", None)
    return bool(capabilities.get("satellite", False)) if override is None else bool(override)


def _enforce_muse_source_protocol(args) -> None:
    """Resolve legacy parser defaults to MUSE roles and reject other drift."""

    if not bool(getattr(args, "use_muse_ssdg", False)):
        return
    capabilities = _muse_level_capabilities(getattr(args, "muse_level", "M0"))
    if not capabilities["base"]:
        return
    legacy_defaults = {
        "split_mode": "tx_rx_day_1_6_3",
        "labeled_ratio": 0.08,
        "unlabeled_ratio": 0.72,
        "source_val_ratio": 0.20,
        "source_cal_ratio": 0.0,
        "source_select_ratio": 0.0,
        "phase1_source_role_protocol": "legacy_l_u_v",
    }
    required = {
        "split_mode": "tx_rx_day_1_7_2",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_val_ratio": 0.30,
        "source_cal_ratio": 0.15,
        "source_select_ratio": 0.15,
        "phase1_source_role_protocol": "l_s_u_s_v_cal_v_select",
    }
    mismatches = []
    for name, expected in required.items():
        actual = getattr(args, name, None)
        legacy = legacy_defaults[name]
        if isinstance(expected, float):
            matches_expected = actual is not None and abs(float(actual) - expected) <= 1e-12
            matches_legacy = actual is not None and abs(float(actual) - float(legacy)) <= 1e-12
        else:
            matches_expected = str(actual) == expected
            matches_legacy = str(actual) == legacy
        if matches_expected:
            continue
        if matches_legacy:
            setattr(args, name, expected)
            continue
        mismatches.append(f"{name}={actual!r} expected={expected!r}")
    if mismatches:
        raise ValueError("MUSE source protocol mismatch: " + "; ".join(mismatches))


class _MUSEUnlabeledDatasetView:
    """Expose U_s IQ/domain metadata while removing TX truth at the dataset edge."""

    _TX_KEYS = frozenset({"tx", "tx_i", "true_tx_i", "tx_label", "y_tx"})

    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 4:
            raise ValueError("MUSE U_s dataset samples must contain IQ, TX, domain, and metadata")
        x_u, _, domain_u, raw_metadata = sample[:4]
        metadata = dict(raw_metadata or {})
        for key in self._TX_KEYS:
            metadata.pop(key, None)
        metadata["tx_label_visible"] = False
        return x_u, domain_u, metadata


def _move_muse_unlabeled_batch(batch, device):
    """Move a label-free U_s batch without materializing a TX tensor."""

    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError("MUSE U_s batches must be label-free (IQ, domain, metadata)")
    x_u, domain_u, metadata_u = batch
    if not torch.is_tensor(x_u):
        raise TypeError("MUSE U_s IQ batch must be a tensor")
    x_u = x_u.to(device, non_blocking=True)
    if torch.is_tensor(domain_u):
        domain_u = domain_u.to(device, non_blocking=True)
    return x_u, (domain_u, metadata_u)


def _dense_anchor_logit_cache(base_indices, logits, *, device):
    """Build a GPU-resident, base-index keyed cache without sample truth."""

    indices = torch.as_tensor(base_indices, device=device, dtype=torch.long).reshape(-1)
    values = torch.as_tensor(logits, device=device)
    if values.dim() != 2 or values.size(0) != indices.numel() or indices.numel() == 0:
        raise ValueError("anchor cache indices/logits must be non-empty and row-aligned")
    if bool(indices.lt(0).any()):
        raise ValueError("anchor cache base_index must be non-negative")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError("anchor cache base_index values must be unique")
    dense = torch.zeros(
        int(indices.max().item()) + 1,
        int(values.size(1)),
        device=device,
        dtype=values.dtype,
    )
    valid = torch.zeros(int(dense.size(0)), device=device, dtype=torch.bool)
    dense.index_copy_(0, indices, values)
    valid[indices] = True
    return {"logits": dense, "valid": valid, "rows": int(indices.numel())}


def _lookup_anchor_logits(cache, metadata, *, expected_count: int, device):
    if isinstance(metadata, Mapping):
        raw_indices = metadata.get("base_index")
        indices = (
            torch.as_tensor(raw_indices, device=device, dtype=torch.long).reshape(-1)
            if raw_indices is not None
            else None
        )
    else:
        indices = _metadata_label_tensor(metadata, "base_index", device, expected_count)
    if indices is None or indices.numel() != int(expected_count):
        raise ValueError("anchor cache lookup requires one base_index per U row")
    valid = cache["valid"]
    if bool(indices.lt(0).any()) or bool(indices.ge(valid.numel()).any()):
        raise ValueError("anchor cache missing base_index")
    if not bool(valid.index_select(0, indices).all()):
        raise ValueError("anchor cache missing base_index")
    return cache["logits"].index_select(0, indices)


def _build_rc4_anchor_logit_cache(
    anchor_model,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    domain_label_map,
    device,
    amp_enabled: bool,
):
    loader = make_loader(
        dataset,
        int(batch_size),
        False,
        int(num_workers),
        device,
        False,
        int(prefetch_factor),
    )
    all_indices, all_logits = [], []
    anchor_model.eval()
    with torch.no_grad():
        for batch in loader:
            x_u, extra_u = _move_muse_unlabeled_batch(batch, device)
            count = int(x_u.size(0))
            domains = domain_from_extra(extra_u, domain_label_map, device)
            indices = _metadata_label_tensor(extra_u, "base_index", device, count)
            if domains is None or indices is None or bool(domains.lt(0).any()):
                raise ValueError("anchor cache requires mapped domains and stable base_index")
            with autocast(enabled=bool(amp_enabled)):
                output = anchor_model(
                    x_u,
                    y_tx=None,
                    grl_lambda=0.0,
                    return_aux=True,
                    domain_labels=domains,
                )
            all_indices.append(indices.detach())
            all_logits.append(output["tx_logits"].detach())
    if not all_indices:
        raise ValueError("anchor cache dataset is empty")
    return _dense_anchor_logit_cache(
        torch.cat(all_indices), torch.cat(all_logits), device=device
    )


def _muse_epoch_pairs(
    labeled_loader,
    unlabeled_loader,
    *,
    use_muse: bool,
    use_unlabeled_step_budget: bool = False,
):
    """Yield a legacy epoch or an unlabeled-length MUSE ablation epoch.

    MUSE owns the joint epoch length.  Each U_s batch is consumed exactly once,
    while L_s is restarted only when its iterator is exhausted.  M0 uses only
    ``len(unlabeled_loader)`` as its step budget and never fetches a U_s batch.
    """

    if not bool(use_muse):
        if not bool(use_unlabeled_step_budget):
            for labeled_batch in labeled_loader:
                yield labeled_batch, None
            return
        labeled_iter = iter(labeled_loader)
        for _ in range(len(unlabeled_loader)):
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration as exc:
                    raise RuntimeError("MUSE M0 requires a non-empty labeled_loader") from exc
            yield labeled_batch, None
        return
    labeled_iter = iter(labeled_loader)
    for unlabeled_batch in unlabeled_loader:
        try:
            labeled_batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration as exc:
                raise RuntimeError("MUSE requires a non-empty labeled_loader") from exc
        yield labeled_batch, unlabeled_batch


def _legacy_unlabeled_active(args, *, muse_state, phase: str) -> bool:
    """Keep the pre-MUSE pseudo-label path out of every MUSE ablation arm."""

    return bool(
        muse_state is None
        and not bool(getattr(args, "use_muse_ssdg", False))
        and str(phase) == "pseudo"
        and bool(args.use_unlabeled)
    )


def _muse_config_from_args(args):
    if MUSEConfig is None:
        raise ImportError("cvsrffi.muse_ssdg.MUSEConfig is required for --use_muse_ssdg")
    config = MUSEConfig(
        s2a_start=int(args.muse_s2a_start),
        s2b_start=int(args.muse_s2b_start),
        s3a_start=int(args.muse_s3a_start),
        s3b_start=int(args.muse_s3b_start),
        s3c_start=int(args.muse_s3c_start),
        final_epoch=int(args.muse_final_epoch),
        lambda_u_full=float(args.muse_lambda_u_full),
        lambda_u_consolidate=float(args.muse_lambda_u_consolidate),
        p_sat_s2a_end=float(args.muse_p_sat_s2a_end),
        p_sat_full=float(args.muse_p_sat_full),
        grl_min=float(args.muse_grl_min),
        grl_max=float(args.muse_grl_max),
    )
    boundaries = (
        1,
        config.s2a_start,
        config.s2b_start,
        config.s3a_start,
        config.s3b_start,
        config.s3c_start,
        config.final_epoch + 1,
    )
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("MUSE schedule boundaries must be strictly increasing")
    for name in ("lambda_u_full", "lambda_u_consolidate", "p_sat_s2a_end", "p_sat_full"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"MUSE {name} must be in [0, 1]")
    return config


def _initialize_muse_training_state(args, model, device):
    capabilities = _muse_level_capabilities(getattr(args, "muse_level", "M0"))
    if not bool(getattr(args, "use_muse_ssdg", False)) or not capabilities["base"]:
        return None
    if any(
        value is None
        for value in (
            MUSETrainingHeads,
            MUSETemporalMemory,
            MUSEClassificationPrototypeBank,
        )
    ):
        raise ImportError("cvsrffi.muse_ssdg training state is required for --use_muse_ssdg")
    z_dim = int(getattr(model, "emb_dim", 0))
    num_classes = int(getattr(model, "num_classes", getattr(args, "num_classes", 0)))
    num_domains = int(getattr(model, "num_domains", 0))
    nuisance_dim = int(getattr(args, "muse_nuisance_dim", 6))
    if min(z_dim, num_classes, num_domains, nuisance_dim) < 1:
        raise ValueError("MUSE requires positive feature, class, domain, and nuisance dimensions")
    config = _muse_config_from_args(args)
    heads = MUSETrainingHeads(
        z_dim,
        z_dim,
        num_classes,
        num_domains,
        nuisance_dim,
    ).to(device)
    return {
        "level": str(args.muse_level).upper(),
        "capabilities": capabilities,
        "config": config,
        "heads": heads,
        "temporal_memory": MUSETemporalMemory(
            stability_steps=int(args.muse_temporal_stability_steps)
        ),
        "classification_prototypes": MUSEClassificationPrototypeBank(
            feature_dim=z_dim,
            unlabeled_weight=float(args.muse_unlabeled_prototype_weight),
        ),
        "source_global_class_counts": torch.zeros(
            num_classes, device=device, dtype=torch.float32
        ),
        "source_domain_class_counts": torch.zeros(
            num_domains, num_classes, device=device, dtype=torch.float32
        ),
        "source_prior_frozen": False,
        "sat_anchor_ssl": bool(getattr(args, "sat_anchor_ssl", False)) or bool(getattr(args, "fasttrust_rc4", False)),
        "fasttrust_rc4": bool(getattr(args, "fasttrust_rc4", False)),
        "sat_anchor_thresholds": None,
        "rc4_calibration": None,
        "schedule_state": None,
        "args": args,
    }


def _configure_muse_epoch_state(muse_state, epoch: int):
    """Apply one schedule boundary, including irreversible S3C freezes."""

    if muse_state is None:
        return None
    schedule = muse_schedule_for_epoch(int(epoch), muse_state["config"])
    muse_state["schedule_state"] = schedule
    muse_state["heads"].train()
    if schedule.freeze_statistics:
        muse_state["temporal_memory"].freeze()
        muse_state["classification_prototypes"].freeze()
        muse_state["source_prior_frozen"] = True
        muse_state["heads"].freeze_local_teacher()
    return schedule


def _optimizer_parameters(model, muse_state) -> List[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if muse_state is not None:
        parameters.extend(
            parameter
            for parameter in muse_state["heads"].parameters()
            if parameter.requires_grad
        )
    deduplicated: List[torch.nn.Parameter] = []
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        deduplicated.append(parameter)
    return deduplicated


def _autograd_grad_or_none(
    objective: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
):
    """Return empty component gradients when telemetry receives a graphless zero."""

    if not torch.is_tensor(objective) or not objective.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _fasttrust_optimizer_groups(model, muse_state) -> List[Dict[str, Any]]:
    backbone: List[torch.nn.Parameter] = []
    other: List[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "backbone" in str(name).lower() else other).append(parameter)
    if muse_state is not None:
        other.extend(
            parameter
            for parameter in muse_state["heads"].parameters()
            if parameter.requires_grad
        )
    groups: List[Dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "fasttrust_role": "backbone"})
    if other:
        groups.append({"params": other, "fasttrust_role": "other"})
    if not groups:
        raise ValueError("optimizer has no trainable parameters")
    return groups


def _fasttrust_lr_enabled(args) -> bool:
    return bool(getattr(args, "use_muse_ssdg", False)) and str(
        getattr(args, "muse_lr_schedule", "off")
    ) == "fasttrust"


def _next_fasttrust_zero_step_streak(
    train_logs: Mapping[str, Any], previous_streak: int
) -> int:
    """Count consecutive full epochs with no optimizer update at all."""

    step_rate = float(train_logs.get("train/optimizer_step_applied", float("nan")))
    if math.isfinite(step_rate) and step_rate <= 0.0:
        return max(0, int(previous_streak)) + 1
    return 0


def _fasttrust_zero_step_abort_required(streak: int) -> bool:
    """Fail closed after the same zero-update fingerprint repeats twice."""

    return int(streak) >= 2


def _rc4_gradient_vector(loss, parameters) -> Optional[torch.Tensor]:
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not torch.is_tensor(loss) or not loss.requires_grad or not parameters:
        return None
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    if all(gradient is None for gradient in gradients):
        return None
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        flattened.append(
            torch.zeros_like(parameter, memory_format=torch.preserve_format).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    return torch.cat(flattened).detach().float()


def _rc4_gradient_relationships(losses, parameters) -> Dict[str, Dict[str, float]]:
    parameters = tuple(parameters)
    vectors = {
        str(name): _rc4_gradient_vector(loss, parameters)
        for name, loss in losses.items()
    }
    norms = {
        name: (
            float(torch.linalg.vector_norm(vector).item())
            if vector is not None
            else float("nan")
        )
        for name, vector in vectors.items()
    }
    labeled = vectors.get("labeled")
    cosines = {}
    for name, vector in vectors.items():
        if name == "labeled":
            continue
        if labeled is None or vector is None:
            cosines[name] = float("nan")
            continue
        denominator = torch.linalg.vector_norm(labeled) * torch.linalg.vector_norm(vector)
        cosines[name] = (
            float(torch.dot(labeled, vector).div(denominator).item())
            if float(denominator.item()) > 0.0
            else float("nan")
        )
    return {"norms": norms, "cosines_to_labeled": cosines}


def _rc4_gradient_probe_parameters(model) -> Tuple[torch.nn.Parameter, ...]:
    prefixes = (
        "id_backbone.cls_head.",
        "id_backbone.con_proj.",
        "id_backbone.fuse.",
        "id_backbone.t3.",
    )
    selected = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(prefixes)
    )
    if selected:
        return selected
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def _fasttrust_lr_scales(epoch: int) -> Tuple[float, float]:
    """Return global and backbone-tail LR multipliers for E1-E200."""

    epoch = int(epoch)
    if not 1 <= epoch <= 200:
        raise ValueError("FastTrust LR epoch must be in [1,200]")
    if epoch <= 5:
        global_scale = epoch / 5.0
    elif epoch <= 160:
        progress = (epoch - 6) / float(160 - 6)
        global_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        global_scale = 0.1
    backbone_tail = 1.0 if epoch <= 160 else (0.2 if epoch <= 180 else 0.05)
    return float(global_scale), float(backbone_tail)


def _apply_fasttrust_lr(optimizer, *, base_lr: float, epoch: int) -> None:
    global_scale, backbone_tail = _fasttrust_lr_scales(int(epoch))
    for group in optimizer.param_groups:
        role_scale = backbone_tail if group.get("fasttrust_role") == "backbone" else 1.0
        group["lr"] = float(base_lr) * global_scale * role_scale


def _fasttrust_epoch_resource_metrics(
    *,
    u_samples_per_step: float,
    u_forward_samples_per_step: float,
    steps: int,
    elapsed_s: float,
    peak_memory_bytes: int,
) -> Dict[str, float]:
    elapsed = max(float(elapsed_s), 1e-12)
    step_count = max(int(steps), 0)
    return {
        "muse/u_samples_per_s": float(u_samples_per_step) * step_count / elapsed,
        "muse/u_forward_samples_per_s": (
            float(u_forward_samples_per_step) * step_count / elapsed
        ),
        "muse/peak_cuda_memory_mb": float(peak_memory_bytes) / float(1024**2),
    }


def _fasttrust_epoch_stage_timing_metrics(
    *,
    train_batches_s: float,
    base_validation_s: float,
    heavy_source_validation_s: float,
    checkpoint_io_s: float,
    epoch_elapsed_s: float,
) -> Dict[str, float]:
    accounted = (
        float(train_batches_s)
        + float(base_validation_s)
        + float(heavy_source_validation_s)
        + float(checkpoint_io_s)
    )
    return {
        "muse/time_train_batches_s": float(train_batches_s),
        "muse/time_base_validation_s": float(base_validation_s),
        "muse/time_heavy_source_validation_s": float(heavy_source_validation_s),
        "muse/time_checkpoint_io_s": float(checkpoint_io_s),
        "muse/time_other_s": max(0.0, float(epoch_elapsed_s) - accounted),
    }


def _rc4_should_collect_gradient_telemetry(
    muse_state: Mapping[str, Any] | None,
    *,
    batch_idx: int,
    epoch: int,
    telemetry_epochs: set[int],
) -> bool:
    return bool(
        muse_state is not None
        and muse_state.get("fasttrust_rc4", False)
        and int(batch_idx) == 1
        and int(epoch) in telemetry_epochs
    )


def _detach_log_mapping(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Detach every tensor before an epoch log retains the batch snapshot."""

    def detach(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, Mapping):
            return {key: detach(nested) for key, nested in value.items()}
        if isinstance(value, tuple):
            return tuple(detach(nested) for nested in value)
        if isinstance(value, list):
            return [detach(nested) for nested in value]
        return value

    return {key: detach(value) for key, value in values.items()}


def _rc4_nonfinite_guard_triggered(
    nonfinite_batches: int,
    steps_seen: int,
    *,
    min_count: int,
    fraction: float,
) -> bool:
    """Stop a systemic RC4 numeric failure before skipped graphs exhaust memory."""

    count = max(0, int(nonfinite_batches))
    steps = max(1, int(steps_seen))
    required = max(1, int(min_count))
    ratio = float(fraction)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("RC4 nonfinite guard fraction must be finite and in [0,1]")
    return count >= required and count / float(steps) >= ratio


def _split_muse_output(value: Any, split_at: int, total: int) -> Tuple[Any, Any]:
    if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == total:
        return value[:split_at], value[split_at:]
    if isinstance(value, Mapping):
        left: Dict[str, Any] = {}
        right: Dict[str, Any] = {}
        for key, nested in value.items():
            left[key], right[key] = _split_muse_output(nested, split_at, total)
        return left, right
    return value, value


def _forward_muse_student_views(
    model,
    strong_x,
    nuisance_x,
    domains,
    *,
    grl_lambda: float,
    fused: bool,
) -> Dict[str, Any]:
    common = {
        "y_tx": None,
        "grl_lambda": float(grl_lambda),
        "return_aux": True,
    }
    if nuisance_x is None:
        return {
            "strong": model(strong_x, domain_labels=domains, **common),
            "nuisance": None,
        }
    if not bool(fused):
        return {
            "strong": model(strong_x, domain_labels=domains, **common),
            "nuisance": model(nuisance_x, domain_labels=domains, **common),
        }
    if strong_x.shape != nuisance_x.shape:
        raise ValueError("fused MUSE student views must have matching shapes")
    combined = torch.cat([strong_x, nuisance_x], dim=0)
    combined_domains = (
        torch.cat([domains, domains], dim=0) if torch.is_tensor(domains) else domains
    )
    output = model(combined, domain_labels=combined_domains, **common)
    strong_output, nuisance_output = _split_muse_output(
        output, int(strong_x.shape[0]), int(combined.shape[0])
    )
    return {"strong": strong_output, "nuisance": nuisance_output}


def _sat_anchor_pair_step(batch_index: int, interval: int) -> bool:
    """Return whether this optimizer step carries the all-U clean/satellite pair."""

    cadence = int(interval)
    if cadence < 1:
        raise ValueError("SAT-Anchor pair interval must be positive")
    return int(batch_index) % cadence == 0


def _forward_sat_anchor_student_views(
    model,
    clean_x,
    satellite_x,
    domains,
    *,
    satellite_indices: torch.Tensor,
    grl_lambda: float,
    detach_backbone: bool,
    include_clean: bool = True,
) -> Dict[str, Any]:
    """Fuse clean U and required satellite rows into exactly one model call."""

    indices = torch.as_tensor(
        satellite_indices, device=clean_x.device, dtype=torch.long
    ).reshape(-1)
    has_satellite = satellite_x is not None and indices.numel() > 0
    include_clean = bool(include_clean)
    if not include_clean and not has_satellite:
        return {
            "clean": None,
            "satellite": None,
            "satellite_indices": indices,
        }
    if include_clean and has_satellite:
        selected_satellite = satellite_x.index_select(0, indices)
        combined = torch.cat([clean_x, selected_satellite], dim=0)
        if torch.is_tensor(domains):
            combined_domains = torch.cat([domains, domains.index_select(0, indices)], dim=0)
        else:
            combined_domains = domains
    elif include_clean:
        combined = clean_x
        combined_domains = domains
    else:
        combined = satellite_x.index_select(0, indices)
        combined_domains = (
            domains.index_select(0, indices) if torch.is_tensor(domains) else domains
        )
    output = model(
        combined,
        y_tx=None,
        grl_lambda=float(grl_lambda),
        return_aux=True,
        domain_labels=combined_domains,
        sat_anchor_detach_backbone=bool(detach_backbone),
    )
    if include_clean:
        clean_output, satellite_output = _split_muse_output(
            output, int(clean_x.shape[0]), int(combined.shape[0])
        )
    else:
        clean_output, satellite_output = None, output
    return {
        "clean": clean_output,
        "satellite": satellite_output if has_satellite else None,
        "satellite_indices": indices,
    }


def _calibrate_sat_anchor_vcal(
    teacher_model,
    calibration_loader,
    *,
    domain_label_map,
    device,
    num_classes: int,
    epsilon: float,
):
    """Calibrate class-complete SAT-Anchor gates using source-only ``V_cal``."""

    probabilities = []
    labels = []
    teacher_model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            x_cal, y_cal, extra_cal = move_batch(batch, device)
            domains = domain_from_extra(extra_cal, domain_label_map, device)
            output = teacher_model(
                x_cal,
                y_tx=None,
                grl_lambda=0.0,
                return_aux=True,
                domain_labels=domains,
            )
            probabilities.append(output["tx_logits"].float().softmax(dim=-1))
            labels.append(y_cal.long())
    if not probabilities:
        raise RuntimeError("SAT_ANCHOR_VCAL_EMPTY")
    return calibrate_sat_anchor_thresholds(
        torch.cat(probabilities, dim=0),
        torch.cat(labels, dim=0),
        num_classes=int(num_classes),
        epsilon=float(epsilon),
    )


def _calibrate_rc4_vcal(
    anchor_model,
    ema_model,
    calibration_loader,
    *,
    args,
    domain_label_map,
    device,
    num_classes: int,
    num_domains: int,
):
    """Fit one stage-frozen RC4 package from source-only ``V_cal``."""

    anchor_logits, ema_1_logits, ema_2_logits = [], [], []
    labels, domains, receivers, z_norms = [], [], [], []
    anchor_model.eval()
    ema_model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            x_cal, y_cal, extra_cal = move_batch(batch, device)
            d_cal = domain_from_extra(extra_cal, domain_label_map, device)
            receiver_cal = _metadata_label_tensor(
                extra_cal, "rx_i", device, int(x_cal.shape[0])
            )
            weak_2 = _strong_augment(x_cal, max(1e-5, float(args.strong_noise_std) * 0.25))
            combined = torch.cat([x_cal, weak_2], dim=0)
            combined_domains = torch.cat([d_cal, d_cal], dim=0)
            ema_output = ema_model(
                combined, y_tx=None, grl_lambda=0.0, return_aux=True,
                domain_labels=combined_domains,
            )
            ema_1, ema_2 = _split_muse_output(
                ema_output, int(x_cal.shape[0]), int(combined.shape[0])
            )
            if bool(args.rc4_use_anchor):
                anchor_output = anchor_model(
                    x_cal, y_tx=None, grl_lambda=0.0, return_aux=True,
                    domain_labels=d_cal,
                )
            else:
                anchor_output = ema_1
            anchor_logits.append(anchor_output["tx_logits"].float())
            ema_1_logits.append(ema_1["tx_logits"].float())
            ema_2_logits.append(ema_2["tx_logits"].float())
            labels.append(y_cal.long())
            domains.append(d_cal.long())
            receivers.append(receiver_cal.long())
            z_norms.append(ema_1["z_id"].float().norm(dim=-1))
    if not labels:
        raise RuntimeError("RC4_VCAL_EMPTY")
    return build_rc4_calibration(
        torch.cat(anchor_logits), torch.cat(ema_1_logits), torch.cat(ema_2_logits),
        torch.cat(labels), torch.cat(domains), torch.cat(z_norms),
        receivers=torch.cat(receivers),
        num_classes=int(num_classes), num_domains=int(num_domains), folds=5,
        hard_precision_target=0.98, partial_coverage_target=0.95,
        partial_precision_target=0.98, partial_min_coverage=0.01,
        partial_candidate_max_classes=int(args.muse_candidate_max_classes),
        negative_false_exclusion_target=0.01,
        decouple_partial_negative_aps=bool(args.rc4_decouple_partial_negative_aps),
        partial_threshold_scope=str(args.rc4_partial_threshold_scope),
    )


def _compute_sat_anchor_unlabeled_losses(
    *,
    route,
    anchor_outputs,
    student_views,
    muse_state,
    epoch: int,
    pair_active: bool,
):
    """Build only the three SAT-Anchor U objectives and compatible telemetry."""

    args = muse_state["args"]
    heads = muse_state["heads"]
    clean = student_views["clean"]
    satellite = student_views["satellite"]
    indices = student_views["satellite_indices"]
    clean_logits = clean["tx_logits"]
    batch_size = int(clean_logits.shape[0])
    zero = clean_logits.sum() * 0.0

    loss_anchor = (
        sat_anchor_clean_kl(
            clean_logits,
            anchor_outputs["tx_logits"],
            temperature=float(args.sat_anchor_temperature),
        )
        if float(args.sat_anchor_lambda_clean_kl) > 0.0
        else zero
    )
    full_satellite_logits = torch.zeros_like(clean_logits)
    if satellite is not None and indices.numel() > 0:
        full_satellite_logits = full_satellite_logits.index_copy(
            0, indices, satellite["tx_logits"]
        )
    identity_enabled = int(epoch) >= int(args.sat_anchor_identity_start_epoch)
    trusted = route.trusted if identity_enabled else torch.zeros_like(route.trusted)
    loss_satellite = (
        trusted_satellite_cross_entropy(
            full_satellite_logits,
            route.pseudo,
            trusted,
            full_unlabeled_batch_size=batch_size,
        )
        if float(args.sat_anchor_lambda_satellite) > 0.0
        else zero
    )
    loss_satellite = loss_satellite + zero
    full_pair = (
        bool(pair_active)
        and satellite is not None
        and indices.numel() == batch_size
        and torch.equal(
            indices,
            torch.arange(batch_size, device=indices.device, dtype=indices.dtype),
        )
    )
    loss_pair = (
        heads.sat_anchor_pair_loss(clean["z_id"], satellite["z_id"])
        if full_pair and float(args.sat_anchor_lambda_pair) > 0.0
        else zero
    )
    pair_drift = (
        (
            1.0
            - F.cosine_similarity(
                clean["z_id"].float(),
                satellite["z_id"].float(),
                dim=-1,
            )
        ).mean()
        if full_pair
        else zero.detach()
    )
    total = (
        float(args.sat_anchor_lambda_clean_kl) * loss_anchor
        + float(args.sat_anchor_lambda_satellite) * loss_satellite
        + float(args.sat_anchor_lambda_pair) * loss_pair
    )
    route_compat = MUSERoute(
        high=trusted,
        mid=torch.zeros_like(trusted),
        low=~trusted,
    )
    scalar_zero = zero.detach()
    return {
        "total": total,
        "identity": zero,
        "hard": loss_satellite,
        "soft": zero,
        "candidate": zero,
        "domain": zero,
        "adv": zero,
        "self": loss_pair,
        "nuisance": zero,
        "satellite": loss_satellite,
        "clean_anchor": loss_anchor,
        "cross_receiver": zero,
        "route": route_compat,
        "fasttrust_route": None,
        "sat_anchor_route": route,
        "reliability": route.confidence,
        "stable": route.agreement,
        "pseudo": route.pseudo,
        "head_js": torch.zeros_like(route.confidence),
        "proto_update_weight": scalar_zero,
        "proto_momentum": scalar_zero,
        "prototype_update_count": scalar_zero,
        "u_identity_selected_count": trusted.sum().to(dtype=clean_logits.dtype),
        "u_satellite_identity_selected_count": trusted.sum().to(dtype=clean_logits.dtype),
        "three_head_agreement_rate": route.agreement.float().mean(),
        "evidence_probabilities": None,
        "fused_probability": None,
        "identity_student": full_satellite_logits,
        "identity_student_satellite_mask": trusted,
        "pair_active": clean_logits.new_tensor(float(full_pair)),
        "pair_drift": pair_drift,
        "filled_count": route.filled.sum().to(dtype=clean_logits.dtype),
    }


def _compute_rc4_unlabeled_losses(
    *,
    route,
    ema_outputs,
    anchor_outputs,
    student_views,
    domains,
    model,
    muse_state,
    epoch: int,
):
    """Compute RC4 H/P/N identity plus all-U domain/self objectives."""

    args = muse_state["args"]
    strong = student_views["clean"]
    satellite = student_views["satellite"]
    indices = student_views["satellite_indices"]
    logits = strong["tx_logits"]
    rows = int(logits.shape[0])
    identity_active = int(epoch) >= int(args.rc4_identity_start_epoch)
    hard = route.hard if identity_active else torch.zeros_like(route.hard)
    partial = route.partial if identity_active else torch.zeros_like(route.partial)
    negative = route.negative if identity_active else torch.zeros_like(route.negative)
    identity = rc4_identity_losses(
        logits,
        route.fused_probability,
        pseudo=route.pseudo,
        candidate_mask=route.candidate_mask,
        hard_mask=hard,
        partial_mask=partial,
        negative_mask=negative,
        weights=route.weights,
        full_unlabeled_batch_size=rows,
        enable_partial_set=bool(args.rc4_enable_partial_set),
        enable_partial_conditional=bool(args.rc4_enable_partial_conditional),
        enable_negative_set=bool(args.rc4_enable_negative),
    )
    active_identity = rc4_active_identity_components(
        identity,
        enable_partial_set=bool(args.rc4_enable_partial_set),
        enable_partial_conditional=bool(args.rc4_enable_partial_conditional),
        enable_negative=bool(args.rc4_enable_negative),
    )
    valid_domain = (domains >= 0) & (domains < int(muse_state["heads"].num_domains))
    zero = logits.sum() * 0.0
    loss_domain = (
        F.cross_entropy(strong["dom_logits"][valid_domain].float(), domains[valid_domain])
        if bool(valid_domain.any()) and "dom_logits" in strong else zero
    )
    loss_discriminator = zero
    loss_confusion = zero
    if bool(valid_domain.any()) and getattr(model, "adv_head", None) is not None:
        if str(args.identity_domain_objective_mode) == "bounded_confusion":
            bounded_u = bounded_domain_objectives(
                model.adv_head,
                strong["z_id"][valid_domain],
                domains[valid_domain],
            )
            loss_discriminator = bounded_u["discriminator"]
            loss_confusion = bounded_u["confusion"]
            loss_adv = (
                float(args.rc4_lambda_discriminator if args.rc4_lambda_discriminator >= 0.0 else args.rc4_lambda_domain)
                * loss_discriminator
                + float(args.rc4_lambda_confusion if args.rc4_lambda_confusion >= 0.0 else args.rc4_lambda_domain)
                * loss_confusion
            )
        else:
            loss_adv = (
                F.cross_entropy(strong["adv_dom_logits"][valid_domain].float(), domains[valid_domain])
                if "adv_dom_logits" in strong else zero
            )
    else:
        loss_adv = zero
    loss_self = muse_state["heads"].self_supervised_loss(
        ema_outputs["z_id"].detach(), strong["z_id"]
    )
    loss_feature_anchor = zero
    if float(args.rc4_lambda_feature_anchor) > 0.0 and anchor_outputs is not None:
        loss_feature_anchor = F.mse_loss(
            F.normalize(strong["z_id"].float(), dim=-1),
            F.normalize(anchor_outputs["z_id"].detach().float(), dim=-1),
        )
    full_satellite_logits = torch.zeros_like(logits)
    if satellite is not None and indices.numel() > 0:
        full_satellite_logits = full_satellite_logits.index_copy(0, indices, satellite["tx_logits"])
    loss_satellite = (
        trusted_satellite_cross_entropy(
            full_satellite_logits, route.pseudo, hard,
            full_unlabeled_batch_size=rows,
        )
        if bool(args.rc4_satellite_hard_only) and float(args.rc4_lambda_satellite) > 0.0
        else zero
    )
    hard_scale, partial_set_scale, partial_conditional_scale = rc4_identity_tail_scales(
        int(epoch),
        start_epoch=int(args.rc4_consolidation_start_epoch),
        end_epoch=200,
        hard_final=float(args.rc4_identity_tail_hard_final),
        partial_set_final=float(args.rc4_identity_tail_partial_set_final),
        partial_conditional_final=float(args.rc4_identity_tail_partial_conditional_final),
    )
    tail_scale = rc4_tail_transition_scale(
        int(epoch),
        start_epoch=int(args.rc4_tail_transition_start_epoch),
        ramp_epochs=int(args.rc4_tail_transition_epochs),
        floor=float(args.rc4_tail_transition_floor),
    )
    partial_set_weight = float(
        args.rc4_lambda_partial_set
        if args.rc4_lambda_partial_set >= 0.0
        else args.rc4_lambda_partial
    )
    partial_conditional_weight = float(
        args.rc4_lambda_partial_conditional
        if args.rc4_lambda_partial_conditional >= 0.0
        else args.rc4_lambda_partial
    )
    zdom_weight = float(
        args.rc4_lambda_zdom if args.rc4_lambda_zdom >= 0.0 else args.rc4_lambda_domain
    )
    adv_weighted = (
        loss_adv
        if str(args.identity_domain_objective_mode) == "bounded_confusion"
        else float(args.rc4_lambda_domain) * loss_adv
    )
    total = (
        hard_scale * float(args.rc4_lambda_hard) * active_identity["hard"]
        + partial_set_scale * partial_set_weight * active_identity["partial_set"]
        + partial_conditional_scale * partial_conditional_weight * active_identity["partial_conditional"]
        + float(args.rc4_lambda_negative) * active_identity["negative"]
        + tail_scale * (zdom_weight * loss_domain + adv_weighted)
        + float(args.rc4_lambda_self) * loss_self
        + float(args.rc4_lambda_satellite) * loss_satellite
        + float(args.rc4_lambda_feature_anchor) * loss_feature_anchor
    )
    route_compat = MUSERoute(high=hard, mid=partial | negative, low=~(hard | partial | negative))
    return {
        "total": total, "identity": identity["total"], "hard": identity["hard"],
        "soft": identity["partial"], "candidate": identity["negative"],
        "domain": loss_domain, "adv": loss_adv, "self": loss_self,
        "domain_discriminator": loss_discriminator,
        "domain_confusion": loss_confusion,
        "feature_anchor": loss_feature_anchor,
        "nuisance": zero, "satellite": loss_satellite, "clean_anchor": zero,
        "cross_receiver": zero, "route": route_compat, "fasttrust_route": None,
        "sat_anchor_route": None, "rc4_route": route, "reliability": route.risk,
        "stable": route.agreement, "pseudo": route.pseudo,
        "head_js": route.disagreement, "proto_update_weight": zero.detach(),
        "proto_momentum": zero.detach(), "prototype_update_count": zero.detach(),
        "u_identity_selected_count": (hard | partial | negative).sum().to(logits.dtype),
        "u_satellite_identity_selected_count": hard.sum().to(logits.dtype),
        "three_head_agreement_rate": route.agreement.float().mean(),
        "evidence_probabilities": None, "fused_probability": route.fused_probability,
        "identity_student": logits, "identity_student_satellite_mask": hard,
        "pair_active": logits.new_tensor(1.0), "pair_drift": zero.detach(),
        "filled_count": zero.detach(),
        "rc4_partial_set": identity["partial_set"],
        "rc4_partial_conditional": identity["partial_conditional"],
        "rc4_negative_set": identity["negative_set"],
        "rc4_tail_scale": logits.new_tensor(tail_scale),
        "rc4_hard_tail_scale": logits.new_tensor(hard_scale),
        "rc4_partial_set_tail_scale": logits.new_tensor(partial_set_scale),
        "rc4_partial_conditional_tail_scale": logits.new_tensor(partial_conditional_scale),
        "rc4_gradient_losses": {
            "hard": hard_scale * float(args.rc4_lambda_hard) * active_identity["hard"],
            "partial_set": partial_set_scale * partial_set_weight * active_identity["partial_set"],
            "partial_conditional": (
                partial_conditional_scale
                * partial_conditional_weight
                * active_identity["partial_conditional"]
            ),
            "adv": tail_scale * adv_weighted,
        },
        "rc4_components_finite": torch.stack([
            identity["hard"].detach().float(),
            identity["partial_set"].detach().float(),
            identity["partial_conditional"].detach().float(),
            identity["negative_set"].detach().float(),
        ]).isfinite().all().to(logits.dtype),
    }


def _assert_muse_open_geometry_role(dataset_role: str) -> None:
    if str(dataset_role).strip() == "U_s":
        raise RuntimeError("MUSE_PROTOCOL_U_S_OPEN_GEOMETRY_FORBIDDEN")


def _muse_prototype_distance(features, pseudo, prototype_bank) -> torch.Tensor:
    prototypes = prototype_bank.prototypes
    normalized = F.normalize(features.float(), dim=-1, eps=1e-8)
    distances = []
    for index, class_id in enumerate(pseudo.detach().cpu().tolist()):
        prototype = prototypes.get(int(class_id))
        if prototype is None:
            distances.append(normalized.new_full((), float("inf")))
            continue
        reference = F.normalize(
            prototype.to(device=normalized.device, dtype=normalized.dtype),
            dim=0,
            eps=1e-8,
        )
        distances.append((1.0 - (normalized[index] * reference).sum()).clamp_min(0.0))
    return torch.stack(distances) if distances else normalized.new_zeros((0,))


def _update_muse_source_class_priors(labels, domains, valid, muse_state) -> None:
    """Accumulate class priors only from legal labeled source observations."""

    if bool(muse_state.get("source_prior_frozen", False)):
        return
    global_counts = muse_state["source_global_class_counts"]
    domain_counts = muse_state["source_domain_class_counts"]
    selected_labels = labels[valid].detach().to(
        device=global_counts.device, dtype=torch.long
    )
    selected_domains = domains[valid].detach().to(
        device=domain_counts.device, dtype=torch.long
    )
    if selected_labels.numel() == 0:
        return
    global_counts.add_(
        torch.bincount(selected_labels, minlength=global_counts.numel()).to(
            dtype=global_counts.dtype
        )
    )
    flat = selected_domains * int(global_counts.numel()) + selected_labels
    domain_counts.add_(
        torch.bincount(
            flat,
            minlength=int(domain_counts.numel()),
        ).reshape_as(domain_counts).to(dtype=domain_counts.dtype)
    )


def _muse_source_class_priors(muse_state, domains, *, device, dtype):
    """Return smoothed per-row domain priors and the legal L_s global prior."""

    global_counts = muse_state["source_global_class_counts"].to(
        device=device, dtype=dtype
    )
    domain_counts = muse_state["source_domain_class_counts"].to(
        device=device, dtype=dtype
    )
    global_prior = (global_counts + 1.0) / (
        global_counts.sum() + float(global_counts.numel())
    )
    smoothed_domains = domain_counts + 1.0
    smoothed_domains = smoothed_domains / smoothed_domains.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    domain_indices = domains.to(device=device, dtype=torch.long).reshape(-1)
    valid = (domain_indices >= 0) & (domain_indices < smoothed_domains.shape[0])
    domain_prior = global_prior.expand(domain_indices.numel(), -1).clone()
    if bool(valid.any()):
        domain_prior[valid] = smoothed_domains[domain_indices[valid]]
    return domain_prior, global_prior


def _compute_muse_labeled_auxiliary_loss(z_id, labels, domains, muse_state):
    """Supervise the local head and seed classification prototypes from L_s."""

    zero = z_id.sum() * 0.0
    if muse_state is None:
        return zero
    if bool(muse_state.get("sat_anchor_ssl", False)):
        return zero
    heads = muse_state["heads"]
    if domains is None or not torch.is_tensor(domains):
        return zero
    labels = labels.to(device=z_id.device, dtype=torch.long).reshape(-1)
    domains = domains.to(device=z_id.device, dtype=torch.long).reshape(-1)
    if labels.numel() != z_id.shape[0] or domains.numel() != z_id.shape[0]:
        return zero
    valid = (
        (labels >= 0)
        & (labels < int(heads.num_classes))
        & (domains >= 0)
        & (domains < int(heads.num_domains))
    )
    if not bool(valid.any()):
        return zero
    _update_muse_source_class_priors(labels, domains, valid, muse_state)
    local_probability = heads.local_prob(z_id[valid], domains[valid])
    loss = F.nll_loss(local_probability.clamp_min(1e-8).log(), labels[valid])
    schedule = muse_state.get("schedule_state")
    if schedule is not None and schedule.freeze_statistics:
        muse_state["temporal_memory"].freeze()
        muse_state["classification_prototypes"].freeze()
        muse_state["source_prior_frozen"] = True
        heads.freeze_local_teacher()
    muse_state["classification_prototypes"].observe_labeled(
        z_id[valid],
        labels[valid],
        domains[valid],
        momentum=(float(schedule.proto_momentum) if schedule is not None else 0.95),
    )
    return loss


def _build_muse_nuisance_iq(
    x_u,
    args,
    *,
    generator,
    scenario: str | None = None,
):
    """Build one paired simulated U_s IQ view without a model forward."""

    if apply_sat_channel_for_scenario is None or normalize_crra_nuisance_meta is None:
        raise ImportError("MUSE nuisance regression requires satellite simulator metadata")
    with torch.no_grad():
        selected_scenario = str(scenario or args.sat_train_scenario)
        x_view, raw_metadata = apply_sat_channel_for_scenario(
            x_u,
            selected_scenario,
            args,
            gen=generator,
            return_meta=True,
        )
        _, nuisance, nuisance_valid, _ = normalize_crra_nuisance_meta(
            raw_metadata,
            scenario=selected_scenario,
            batch_size=int(x_u.size(0)),
            device=x_u.device,
        )
        if torch.is_tensor(nuisance):
            nuisance = nuisance[:, : int(args.muse_nuisance_dim)]
    return _safe_iq_tensor(x_view), {
        "nuisance": nuisance,
        "nuisance_valid": nuisance_valid,
        "scenario": selected_scenario,
    }


def _build_muse_nuisance_view(
    model,
    x_u,
    domains,
    args,
    *,
    grl_lambda: float,
    generator,
    scenario: str | None = None,
):
    """Compatibility wrapper for one paired U_s nuisance forward."""

    x_view, metadata = _build_muse_nuisance_iq(
        x_u,
        args,
        generator=generator,
        scenario=scenario,
    )
    outputs = model(
        x_view,
        y_tx=None,
        grl_lambda=float(grl_lambda),
        return_aux=True,
        domain_labels=domains,
    )
    return outputs, metadata


def _muse_cross_receiver_alignment(features, pseudo, receivers, mask) -> torch.Tensor:
    zero = features.sum() * 0.0
    if receivers is None or not torch.is_tensor(receivers):
        return zero
    receiver_values = receivers.to(device=features.device).reshape(-1)
    if receiver_values.numel() != features.shape[0]:
        return zero
    selected = mask.bool().reshape(-1)
    terms = []
    for class_id in torch.unique(pseudo[selected]):
        class_mask = selected & pseudo.eq(class_id)
        if int(torch.unique(receiver_values[class_mask]).numel()) < 2:
            continue
        class_features = F.normalize(features[class_mask].float(), dim=-1, eps=1e-8)
        center = F.normalize(class_features.detach().mean(dim=0), dim=0, eps=1e-8)
        terms.append((1.0 - class_features @ center).mean())
    return torch.stack(terms).mean() if terms else zero


def _compute_muse_unlabeled_losses(
    x_u,
    metadata,
    teacher_outputs,
    student_outputs,
    muse_state,
    simulator_metadata,
):
    """Compute MUSE U_s objectives without accepting true TX identities."""

    del x_u
    if muse_state is None:
        raise ValueError("muse_state is required")
    args = muse_state["args"]
    capabilities = muse_state["capabilities"]
    schedule = muse_state.get("schedule_state")
    if schedule is None:
        raise ValueError("MUSE schedule_state must be set before computing losses")
    if schedule.freeze_statistics:
        muse_state["temporal_memory"].freeze()
        muse_state["classification_prototypes"].freeze()
        muse_state["source_prior_frozen"] = True
        muse_state["heads"].freeze_local_teacher()
    heads = muse_state["heads"]
    strong = student_outputs["strong"]
    weak = student_outputs.get("weak", teacher_outputs)
    zero = strong["tx_logits"].sum() * 0.0
    sample_count = int(strong["tx_logits"].shape[0])
    identity_satellite_mask = torch.zeros(
        sample_count, device=zero.device, dtype=torch.bool
    )
    identity_logits = strong["tx_logits"]
    identity_z_id = strong["z_id"]
    if capabilities["satellite"] and schedule.pseudo_enabled:
        requested_mask = metadata.get("satellite_mask")
        if requested_mask is not None:
            identity_satellite_mask = torch.as_tensor(
                requested_mask, device=zero.device, dtype=torch.bool
            ).reshape(-1)
            if identity_satellite_mask.numel() != sample_count:
                raise ValueError("satellite_mask must contain one value per U_s sample")
    identity_student = {
        "tx_logits": identity_logits,
        "z_id": identity_z_id,
    }
    domains = metadata.get("domains")
    if torch.is_tensor(domains) and domains.numel() == sample_count:
        domains = domains.to(device=strong["tx_logits"].device, dtype=torch.long).reshape(-1)
        valid_domain = (domains >= 0) & (domains < int(heads.num_domains))
    else:
        domains = torch.zeros(sample_count, device=strong["tx_logits"].device, dtype=torch.long)
        valid_domain = torch.zeros(sample_count, device=domains.device, dtype=torch.bool)

    loss_domain = zero
    loss_adv = zero
    if bool(valid_domain.any()):
        if "dom_logits" in strong:
            loss_domain = F.cross_entropy(
                strong["dom_logits"][valid_domain].float(), domains[valid_domain]
            )
        if "adv_dom_logits" in strong:
            loss_adv = F.cross_entropy(
                strong["adv_dom_logits"][valid_domain].float(), domains[valid_domain]
            )
    loss_self = heads.self_supervised_loss(weak["z_id"], identity_z_id)
    nuisance = simulator_metadata.get("nuisance")
    nuisance_valid = simulator_metadata.get("nuisance_valid")
    nuisance_outputs = student_outputs.get("nuisance")
    if (
        torch.is_tensor(nuisance)
        and nuisance.shape == (sample_count, int(heads.nuisance_dim))
        and nuisance_valid is not None
        and isinstance(nuisance_outputs, Mapping)
        and torch.is_tensor(nuisance_outputs.get("z_dom"))
    ):
        nuisance_z_dom = nuisance_outputs["z_dom"]
        if bool(getattr(args, "muse_nuisance_detached", False)):
            nuisance_z_dom = nuisance_z_dom.detach()
        loss_nuisance = heads.nuisance_loss(
            nuisance_z_dom, nuisance, nuisance_valid
        )
    else:
        loss_nuisance = strong["z_dom"].sum() * 0.0
    loss_base = (
        float(args.muse_lambda_domain) * loss_domain
        + float(args.muse_lambda_adv) * loss_adv
        + float(args.muse_lambda_self) * loss_self
        + float(args.muse_lambda_nuisance) * loss_nuisance
    )

    loss_identity = zero
    hard_loss = zero
    soft_loss = zero
    candidate_loss = zero
    loss_satellite = zero
    loss_cross_receiver = zero
    head_js = zero
    proto_update_weight = zero
    prototype_update_count = zero
    satellite_identity_selected_count = zero
    identity_selected_count = zero
    three_head_agreement_rate = zero
    route = route_muse_reliability(
        torch.zeros(sample_count, device=zero.device),
        high_threshold=float(args.muse_high_threshold),
        low_threshold=float(args.muse_low_threshold),
    )
    reliability = torch.zeros(sample_count, device=zero.device)
    stable = torch.zeros(sample_count, dtype=torch.bool, device=zero.device)
    global_probability = F.softmax(
        teacher_outputs["tx_logits"].detach().float(), dim=-1
    )
    evidence_probabilities: Dict[str, torch.Tensor] = {}
    fused = global_probability
    pseudo = teacher_outputs["tx_logits"].argmax(dim=-1)
    if not bool(schedule.pseudo_enabled) or not bool(capabilities["fusion"]):
        total = loss_base
        return {
            "stage": str(schedule.stage),
            "total": total,
            "base": loss_base,
            "identity": loss_identity,
            "hard": hard_loss,
            "soft": soft_loss,
            "candidate": candidate_loss,
            "domain": loss_domain,
            "adv": loss_adv,
            "self": loss_self,
            "nuisance": loss_nuisance,
            "satellite": loss_satellite,
            "cross_receiver": loss_cross_receiver,
            "route": route,
            "fasttrust_route": None,
            "reliability": reliability,
            "stable": stable,
            "pseudo": pseudo,
            "head_js": head_js,
            "proto_update_weight": proto_update_weight,
            "proto_momentum": zero.new_tensor(float(schedule.proto_momentum)),
            "prototype_update_count": prototype_update_count,
            "u_identity_selected_count": identity_selected_count,
            "u_satellite_identity_selected_count": satellite_identity_selected_count,
            "three_head_agreement_rate": three_head_agreement_rate,
            "evidence_probabilities": evidence_probabilities,
            "fused_probability": fused,
            "identity_student": identity_student,
            "identity_student_satellite_mask": identity_satellite_mask,
        }
    if capabilities["fusion"]:
        local_probability = heads.local_prob(
            teacher_outputs["z_id"].detach(), domains.clamp(0, heads.num_domains - 1)
        )
        prototype_probability = muse_state[
            "classification_prototypes"
        ].class_probabilities(
            teacher_outputs["z_id"].detach(),
            num_classes=int(heads.num_classes),
        )
        if not bool(getattr(args, "muse_use_prototype_evidence", True)):
            prototype_probability = global_probability
        evidence_probabilities = {
            "global": global_probability,
            "local": local_probability,
            "prototype": prototype_probability,
        }
        consensus_pseudo = global_probability.argmax(dim=-1)
        probabilities = list(evidence_probabilities.values())
        fused = geometric_fuse_probabilities(
            probabilities,
            [
                float(args.muse_fusion_global_weight),
                float(args.muse_fusion_local_weight),
                float(args.muse_fusion_prototype_weight),
            ],
        )
        domain_prior, global_prior = _muse_source_class_priors(
            muse_state,
            domains,
            device=fused.device,
            dtype=fused.dtype,
        )
        fused = align_source_domain_prior(
            fused,
            domain_prior,
            global_prior,
            gamma=float(args.muse_prior_alignment_gamma),
        )
        top_values, top_indices = fused.topk(k=min(2, fused.shape[-1]), dim=-1)
        pseudo = top_indices[:, 0]
        confidence = top_values[:, 0]
        margin = (
            top_values[:, 0] - top_values[:, 1]
            if top_values.shape[-1] > 1
            else top_values[:, 0]
        )
        if schedule.pseudo_enabled:
            memory_keys = metadata.get("memory_keys")
            if memory_keys is None:
                memory_keys = [(0, 0, 0, index, index) for index in range(sample_count)]
            stable = muse_state["temporal_memory"].observe(
                memory_keys,
                consensus_pseudo,
                confidence,
                int(metadata.get("epoch", 1)),
            )
        prototype_distance = _muse_prototype_distance(
            teacher_outputs["z_id"].detach(),
            consensus_pseudo,
            muse_state["classification_prototypes"],
        )
        head_js = js_head_disagreement(probabilities)
        reliability = compute_muse_reliability(
            confidence,
            margin,
            head_js,
            prototype_distance,
            stable.float(),
            weights=[
                float(args.muse_reliability_confidence_weight),
                float(args.muse_reliability_margin_weight),
                float(args.muse_reliability_js_weight),
                float(args.muse_reliability_prototype_weight),
                float(args.muse_reliability_stability_weight),
            ],
        )
        fasttrust_route = route_fasttrust(
            reliability,
            stable
            if bool(getattr(args, "muse_require_temporal_stability", True))
            else torch.ones_like(stable),
            evidence_probabilities,
            high_threshold=float(args.muse_high_threshold),
            low_threshold=float(args.muse_low_threshold),
            hard_max_fraction=float(args.muse_hard_max_fraction),
            identity_max_fraction=float(args.muse_identity_max_fraction),
            class_balanced_cap=bool(getattr(args, "muse_class_balanced_cap", True)),
            hard_only_no_fill=bool(
                getattr(args, "muse_fasttrust_hard_only_no_fill", False)
            ),
        )
        route = MUSERoute(
            high=fasttrust_route.hard,
            mid=fasttrust_route.soft,
            low=fasttrust_route.candidate | fasttrust_route.no_identity,
        )
        pseudo = pseudo.clone()
        pseudo[fasttrust_route.hard] = consensus_pseudo[fasttrust_route.hard]
        identity_selected_count = (
            fasttrust_route.hard
            | fasttrust_route.soft
            | fasttrust_route.candidate
        ).sum().to(dtype=zero.dtype)
        three_head_agreement_rate = fasttrust_route.agreement.float().mean()
        if bool(route.high.any()):
            hard_loss = F.cross_entropy(
                identity_logits[route.high].float(),
                pseudo[route.high].detach(),
            )
        soft_loss = weighted_soft_cross_entropy(
            identity_logits, fused, reliability.detach(), route.mid
        )
        if bool(schedule.candidate_enabled):
            candidate = candidate_set_mask(
                fused,
                mass=float(args.muse_candidate_mass),
                max_classes=int(args.muse_candidate_max_classes),
            )
            candidate_loss = candidate_set_cross_entropy(
                identity_logits,
                candidate,
                reliability.detach(),
                fasttrust_route.candidate,
            )
        loss_identity = float(schedule.lambda_u) * (
            float(args.muse_lambda_high) * hard_loss
            + float(args.muse_lambda_mid) * soft_loss
            + float(args.muse_lambda_low_entropy) * candidate_loss
        )

        if capabilities["satellite"] and schedule.pseudo_enabled:
            accepted = fasttrust_route.hard & identity_satellite_mask
            if (
                bool(getattr(args, "muse_enable_u_satellite_identity", True))
                and float(args.muse_lambda_satellite) > 0.0
                and bool(accepted.any())
            ):
                satellite_identity = student_outputs.get("satellite")
                if not isinstance(satellite_identity, Mapping):
                    raise ValueError("selected U_H satellite rows require satellite outputs")
                satellite_logits = satellite_identity.get("tx_logits")
                if (
                    not torch.is_tensor(satellite_logits)
                    or satellite_logits.shape != strong["tx_logits"].shape
                ):
                    raise ValueError(
                        "satellite tx_logits must match the strong identity tensor shape"
                    )
                loss_satellite = (
                    float(schedule.lambda_u)
                    * float(args.muse_lambda_satellite)
                    * F.cross_entropy(
                        satellite_logits[accepted].float(),
                        pseudo[accepted].detach(),
                    )
                )
                satellite_identity_selected_count = accepted.sum().to(dtype=zero.dtype)
            loss_cross_receiver = _muse_cross_receiver_alignment(
                identity_z_id,
                pseudo,
                metadata.get("receivers"),
                fasttrust_route.hard,
            )
        if (
            schedule.pseudo_enabled
            and _muse_u_prototype_update_enabled(args, capabilities)
            and float(args.muse_unlabeled_prototype_weight) > 0.0
        ):
            proto_update_mask = fasttrust_route.hard
            if bool(proto_update_mask.any()):
                proto_update_weight = zero.new_tensor(
                    float(args.muse_unlabeled_prototype_weight)
                )
                prototype_update_count = proto_update_mask.sum().to(dtype=zero.dtype)
                muse_state["classification_prototypes"].observe(
                    identity_z_id,
                    pseudo,
                    domains,
                    proto_update_mask,
                    proto_update_mask,
                    float(args.muse_unlabeled_prototype_weight),
                    momentum=float(schedule.proto_momentum),
                )

    total = (
        loss_base
        + loss_identity
        + loss_satellite
        + float(args.muse_lambda_cross_receiver) * loss_cross_receiver
    )
    return {
        "stage": str(schedule.stage),
        "total": total,
        "base": loss_base,
        "identity": loss_identity,
        "hard": hard_loss,
        "soft": soft_loss,
        "candidate": candidate_loss,
        "domain": loss_domain,
        "adv": loss_adv,
        "self": loss_self,
        "nuisance": loss_nuisance,
        "satellite": loss_satellite,
        "cross_receiver": loss_cross_receiver,
        "route": route,
        "fasttrust_route": fasttrust_route,
        "reliability": reliability,
        "stable": stable,
        "pseudo": pseudo,
        "head_js": head_js,
        "proto_update_weight": proto_update_weight,
        "proto_momentum": zero.new_tensor(float(schedule.proto_momentum)),
        "prototype_update_count": prototype_update_count,
        "u_identity_selected_count": identity_selected_count,
        "u_satellite_identity_selected_count": satellite_identity_selected_count,
        "three_head_agreement_rate": three_head_agreement_rate,
        "evidence_probabilities": evidence_probabilities,
        "fused_probability": fused,
        "identity_student": identity_student,
        "identity_student_satellite_mask": identity_satellite_mask,
    }


def _muse_checkpoint_state(muse_state) -> Dict[str, Any]:
    if muse_state is None:
        return {
            "muse_training_heads": None,
            "muse_temporal_memory": None,
            "muse_classification_prototypes": None,
            "muse_source_global_class_counts": None,
            "muse_source_domain_class_counts": None,
            "muse_source_prior_frozen": None,
            "muse_schedule_state": None,
        }
    schedule = muse_state.get("schedule_state")
    return {
        "muse_training_heads": muse_state["heads"].training_state_dict(),
        "muse_temporal_memory": muse_state["temporal_memory"].state_dict(),
        "muse_classification_prototypes": muse_state[
            "classification_prototypes"
        ].state_dict(),
        "muse_source_global_class_counts": muse_state[
            "source_global_class_counts"
        ].detach().clone(),
        "muse_source_domain_class_counts": muse_state[
            "source_domain_class_counts"
        ].detach().clone(),
        "muse_source_prior_frozen": bool(muse_state["source_prior_frozen"]),
        "muse_schedule_state": dict(vars(schedule)) if schedule is not None else None,
        "sat_anchor_thresholds": (
            {
                "confidence": muse_state["sat_anchor_thresholds"].confidence.detach().clone(),
                "margin": muse_state["sat_anchor_thresholds"].margin.detach().clone(),
            }
            if muse_state.get("sat_anchor_thresholds") is not None
            else None
        ),
        "rc4_calibration": muse_state.get("rc4_calibration"),
    }


def _restore_muse_checkpoint_state(muse_state, checkpoint: Mapping[str, Any]) -> None:
    """Restore MUSE-only training state without modifying deployment model weights."""

    if muse_state is None:
        return
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MUSE checkpoint must be a mapping")
    heads = checkpoint.get("muse_training_heads")
    if heads is not None:
        muse_state["heads"].load_state_dict(heads, strict=True)
    memory = checkpoint.get("muse_temporal_memory")
    if memory is not None:
        muse_state["temporal_memory"].load_state_dict(memory)
    prototypes = checkpoint.get("muse_classification_prototypes")
    if prototypes is not None:
        muse_state["classification_prototypes"].load_state_dict(prototypes)
    for checkpoint_key, state_key in (
        ("muse_source_global_class_counts", "source_global_class_counts"),
        ("muse_source_domain_class_counts", "source_domain_class_counts"),
    ):
        value = checkpoint.get(checkpoint_key)
        if value is None:
            continue
        target = muse_state[state_key]
        restored = torch.as_tensor(value, device=target.device, dtype=target.dtype)
        if restored.shape != target.shape:
            raise ValueError(f"{checkpoint_key} shape does not match MUSE state")
        target.copy_(restored)
    muse_state["source_prior_frozen"] = bool(
        checkpoint.get("muse_source_prior_frozen", False)
    )
    schedule = checkpoint.get("muse_schedule_state")
    if schedule is not None:
        if not isinstance(schedule, Mapping):
            raise ValueError("MUSE schedule state must be a mapping")
        muse_state["schedule_state"] = MUSEScheduleState(**dict(schedule))
    sat_thresholds = checkpoint.get("sat_anchor_thresholds")
    if sat_thresholds is not None:
        from cvsrffi.muse_ssdg import SATAnchorThresholds

        muse_state["sat_anchor_thresholds"] = SATAnchorThresholds(
            confidence=torch.as_tensor(
                sat_thresholds["confidence"],
                device=muse_state["source_global_class_counts"].device,
            ),
            margin=torch.as_tensor(
                sat_thresholds["margin"],
                device=muse_state["source_global_class_counts"].device,
            ),
        )
    if checkpoint.get("rc4_calibration") is not None:
        muse_state["rc4_calibration"] = checkpoint["rc4_calibration"]


def _pseudo_gate_pass_rates(
    *,
    domain_mask: torch.Tensor,
    temporal_mask: torch.Tensor,
    strong_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        domain_mask.float().mean(),
        temporal_mask.float().mean(),
        strong_mask.float().mean(),
    )


def _compose_unlabeled_closed_loss(
    base,
    *,
    args,
    muse_state,
    muse_total,
    identity,
    entropy,
    domain,
    adv,
    satellite,
):
    if muse_state is not None:
        return base + muse_total
    return (
        base
        + float(args.lambda_u) * identity
        + float(args.lambda_ent) * entropy
        + float(args.lambda_u_domain) * domain
        + float(args.lambda_u_adv) * adv
        + float(args.lambda_u_sat_cons) * satellite
    )


def train(args) -> int:
    if route_phase1_method(args) == "bicad_xr":
        return _train_bicad_xr(args)
    training_wall_started = time.time()
    muse_capabilities = _muse_level_capabilities(getattr(args, "muse_level", "M0"))
    muse_active = bool(getattr(args, "use_muse_ssdg", False)) and muse_capabilities["base"]
    if _sat_anchor_requested(args):
        if str(getattr(args, "muse_level", "")).upper() != "M3":
            raise ValueError("SAT-Anchor-SSL requires --muse_level M3")
        adapter_scope = str(getattr(args, "sat_anchor_u_gradient_scope", "full"))
        if adapter_scope == "adapter_tail" and not bool(
            getattr(args, "sat_anchor_adapter", False)
        ):
            raise ValueError("adapter_tail gradient scope requires --sat_anchor_adapter true")
        if int(getattr(args, "sat_anchor_pair_interval", 1)) < 1:
            raise ValueError("SAT-Anchor pair interval must be positive")
    if _rc4_requested(args):
        if str(getattr(args, "muse_level", "")).upper() != "M3":
            raise ValueError("FastTrust-RC4 requires --muse_level M3")
        if int(getattr(args, "rc4_identity_start_epoch", 11)) < 1:
            raise ValueError("RC4 identity start epoch must be positive")
        total_identity_budget = float(
            getattr(args, "rc4_total_identity_effective_budget", 0.0)
        )
        if not 0.0 <= total_identity_budget <= 1.0:
            raise ValueError("RC4 total identity effective budget must be in [0,1]")
        if total_identity_budget > 0.0 and bool(getattr(args, "rc4_enable_negative", False)):
            raise ValueError("RC4 quality-budget mode requires negative routing disabled")
        if int(getattr(args, "rc4_nonfinite_guard_min_count", 8)) < 1:
            raise ValueError("RC4 nonfinite guard min count must be positive")
        guard_fraction = float(getattr(args, "rc4_nonfinite_guard_fraction", 0.05))
        if not 0.0 <= guard_fraction <= 1.0:
            raise ValueError("RC4 nonfinite guard fraction must be in [0,1]")
    _enforce_muse_source_protocol(args)
    if muse_active and int(getattr(args, "epochs", 0)) <= 0:
        args.epochs = int(args.muse_final_epoch)
        args.label_epochs = int(args.muse_final_epoch)
        args.pseudo_epochs = 0
    ablation_manifest = None
    if bool(getattr(args, "formal_ablation", False)):
        ablation_manifest = apply_phase1_ablation(args)
        if str(getattr(args, "checkpoint_selection", "")) != "final_only":
            raise ValueError(
                "formal Phase1 ablation configuration is incompatible with mandatory final-only checkpoint selection"
            )
        if not str(getattr(args, "run_id", "")).strip():
            raise ValueError("formal Phase1 ablation requires --run_id")
        if str(getattr(args, "candidate_id", "")).strip() != str(args.ablation_id):
            raise ValueError(
                "formal Phase1 candidate_id must exactly equal ablation_id"
            )
        expected_row_key = (
            f"{args.ablation_id}__train_seed_{int(args.seed)}"
        )
        if str(getattr(args, "row_key", "")).strip() != expected_row_key:
            raise ValueError(
                "formal Phase1 row_key does not match ablation_id and seed"
            )
        for field in (
            "sealed_plan_sha256",
            "seed_registry_sha256",
            "wisig_pkl_sha256",
        ):
            value = str(getattr(args, field, "")).strip().lower()
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(
                    f"formal Phase1 requires a 64-character {field}"
                )
    elif str(getattr(args, "ablation_id", "")).strip():
        raise ValueError(
            "--ablation_id is reserved for --formal_ablation true"
        )
    total_epochs = _resolve_epoch_schedule(args)
    if muse_active and int(total_epochs) != int(args.muse_final_epoch):
        raise ValueError("MUSE total epochs must equal --muse_final_epoch")
    args.epochs = total_epochs
    args.lambda_dom = float(args.lambda_domain)
    if float(args.tau_conf) > 0.0:
        args.tau_min = float(args.tau_conf)
    if not bool(args.use_unlabeled):
        args.lambda_u = 0.0
    args.sat_train_scenario = str(args.sat_train_scenario or "mixed_orbit").strip().lower().replace("-", "_")
    args.crra_scenario = str(getattr(args, "crra_scenario", "mixed_orbit") or "mixed_orbit").strip().lower().replace("-", "_")
    _resolve_sat_training_mode(args)
    if str(getattr(args, "phase1_source_role_protocol", "legacy_l_u_v")) == "l_s_u_s_v_cal_v_select":
        requested_validation = float(args.source_cal_ratio) + float(args.source_select_ratio)
        if abs(float(args.source_val_ratio) - requested_validation) > 1e-6:
            raise ValueError(
                "current Phase1 source_val_ratio must equal source_cal_ratio + source_select_ratio; "
                f"got {float(args.source_val_ratio):.6f} vs {requested_validation:.6f}"
            )
    validate_crra_phase1_config(args)
    sat_train_spec = str(getattr(args, "sat_train_scenarios", "") or "").strip()
    if sat_train_spec:
        if parse_sat_scenarios is not None:
            args.sat_train_scenario_list = list(parse_sat_scenarios(sat_train_spec))
        else:
            args.sat_train_scenario_list = [
                part.strip().lower().replace("-", "_") for part in sat_train_spec.split(",") if part.strip()
            ]
    else:
        args.sat_train_scenario_list = [args.sat_train_scenario]
    if not args.sat_train_scenario_list:
        args.sat_train_scenario_list = [args.sat_train_scenario]
    args.sat_train_scenario = args.sat_train_scenario_list[0]
    args.sat_view_schedule = str(getattr(args, "sat_view_schedule", "") or "").strip()
    if float(getattr(args, "sat_view_prob", 1.0)) < 0.0 or float(getattr(args, "sat_view_prob", 1.0)) > 1.0:
        raise ValueError("--sat_view_prob must be in [0, 1]")
    if args.sat_view_schedule and parse_sat_view_schedule is not None:
        args.sat_view_stages = tuple(
            parse_sat_view_schedule(args.sat_view_schedule, default_prob=float(getattr(args, "sat_view_prob", 1.0)))
        )
    else:
        args.sat_view_stages = tuple()
    scheduled_train_scenarios = list(args.sat_train_scenario_list)
    for stage in args.sat_view_stages:
        scheduled_train_scenarios.extend(str(value) for value in stage.scenarios)
    args.sat_train_protocol_scenario_list = list(
        dict.fromkeys(str(value).strip().lower().replace("-", "_") for value in scheduled_train_scenarios)
    )
    if bool(getattr(args, "use_crra", False)):
        validate_crra_phase1_scenarios(args.sat_train_protocol_scenario_list)
    args.eval_sat_scenario_list = (
        parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []
    )
    if satellite_protocol_manifest is None:
        raise ImportError("training_controls.satellite_protocol_manifest is required for Phase1 satellite protocol audit")
    args.sat_protocol_manifest = satellite_protocol_manifest(
        args.sat_train_protocol_scenario_list,
        args.eval_sat_scenario_list,
        require_disjoint=bool(args.sat_protocol_disjoint_required),
    )
    if float(getattr(args, "concat_sat_ce_weight", 1.0)) < 0.0:
        raise ValueError("--concat_sat_ce_weight must be >= 0")
    if bool(getattr(args, "concat_sat_ce_only", False)) and not bool(getattr(args, "use_concat_sat_channel_aug", False)):
        print("[WARN] --concat_sat_ce_only has no effect unless --use_concat_sat_channel_aug is enabled.", flush=True)
    if not bool(getattr(args, "phase1_source_val_selection_only", True)):
        raise ValueError(
            "Phase1 is source-only: --phase1_source_val_selection_only must remain true; "
            "held-out receiver/day/satellite test feedback is forbidden during training."
        )
    expected_checkpoint_selection = "final_only"
    if str(getattr(args, "checkpoint_selection", "")) != expected_checkpoint_selection:
        raise ValueError(
            "Phase1 checkpoint selection drift: "
            f"expected {expected_checkpoint_selection}"
        )
    if bool(getattr(args, "tail_rollback_enabled", False)):
        raise ValueError(
            "Phase1 final-only mode forbids tail checkpoint rollback; retain tail references as metrics only."
        )
    if str(args.best_metric) not in {"clean_val_tx", "source_val_sat_hmean"}:
        raise ValueError(
            "Phase1 source-only checkpoint selection forbids test/receiver/satellite-test best metrics; "
            "use --best_metric clean_val_tx or source_val_sat_hmean."
        )
    if str(args.best_metric) == "source_val_sat_hmean" and not bool(getattr(args, "eval_sat_channel", False)):
        raise ValueError("source_val_sat_hmean requires --eval_sat_channel true")
    if int(getattr(args, "source_val_heavy_eval_start_epoch", 1)) < 1:
        raise ValueError("--source_val_heavy_eval_start_epoch must be >= 1")
    if int(getattr(args, "source_val_heavy_eval_interval", 1)) < 1:
        raise ValueError("--source_val_heavy_eval_interval must be >= 1")
    if int(getattr(args, "source_val_heavy_eval_final_window", 0)) < 0:
        raise ValueError("--source_val_heavy_eval_final_window must be >= 0")
    if int(getattr(args, "source_val_heavy_eval_final_interval", 1)) < 1:
        raise ValueError("--source_val_heavy_eval_final_interval must be >= 1")
    if float(getattr(args, "max_grad_norm", 0.0)) < 0.0:
        raise ValueError("--max_grad_norm must be >= 0")
    if float(getattr(args, "os_eff_max_budget", 0.0)) > 0.0 and float(
        getattr(args, "os_eff_max_budget", 0.0)
    ) < float(getattr(args, "os_eff_min_budget", 0.0)):
        raise ValueError("--os_eff_max_budget must be zero/disabled or >= --os_eff_min_budget")
    if float(getattr(args, "os_budget_target_reserve", 0.0)) < 0.0:
        raise ValueError("--os_budget_target_reserve must be >= 0")
    objective_shares = [
        float(getattr(args, "os_objective_boundary_share", 0.40)),
        float(getattr(args, "os_objective_source_share", 0.25)),
        float(getattr(args, "os_objective_invariant_share", 0.20)),
        float(getattr(args, "os_objective_u_share", 0.15)),
    ]
    if any(value < 0.0 for value in objective_shares) or sum(objective_shares) <= 0.0:
        raise ValueError("open-set objective shares must be non-negative with a positive total")
    if float(getattr(args, "os_objective_min_scale", 0.25)) <= 0.0:
        raise ValueError("--os_objective_min_scale must be > 0")
    if float(getattr(args, "os_objective_max_scale", 8.0)) < float(
        getattr(args, "os_objective_min_scale", 0.25)
    ):
        raise ValueError("--os_objective_max_scale must be >= --os_objective_min_scale")
    if int(getattr(args, "source_episode_local_min_samples", 2)) < 2:
        raise ValueError("--source_episode_local_min_samples must be >= 2")
    if float(getattr(args, "source_episode_local_radius_floor_deg", 3.0)) <= 0.0:
        raise ValueError("--source_episode_local_radius_floor_deg must be > 0")
    if float(getattr(args, "source_episode_local_density_cap", 2.0)) <= 0.0:
        raise ValueError("--source_episode_local_density_cap must be > 0")
    if float(getattr(args, "source_episode_local_term_cap", 4.0)) <= 0.0:
        raise ValueError("--source_episode_local_term_cap must be > 0")
    if float(getattr(args, "source_episode_clean_weight", 1.0)) < 0.0 or float(
        getattr(args, "source_episode_sat_weight", 1.0)
    ) < 0.0:
        raise ValueError("source-episode clean/satellite weights must be >= 0")
    if float(getattr(args, "source_episode_clean_weight", 1.0)) + float(
        getattr(args, "source_episode_sat_weight", 1.0)
    ) <= 0.0:
        raise ValueError("at least one source-episode view weight must be positive")
    if float(getattr(args, "source_val_dg_health_stop_drop_pp", 8.0)) < float(
        getattr(args, "source_val_dg_health_warning_drop_pp", 3.0)
    ):
        raise ValueError("source-val DG stop drop must be >= warning drop")
    if not 0.0 <= float(getattr(args, "source_val_dg_health_min_open_scale", 0.20)) <= 1.0:
        raise ValueError("--source_val_dg_health_min_open_scale must be in [0, 1]")
    if int(getattr(args, "source_val_dg_health_stop_patience", 1)) < 1:
        raise ValueError("--source_val_dg_health_stop_patience must be >= 1")
    if bool(getattr(args, "enable_joint_safe_guard", False)):
        raise ValueError(
            "Phase1 source-only checkpoint selection forbids held-out test joint guards; "
            "use Phase1 V2 source geometry guards and freeze before held-out evaluation."
        )
    if bool(getattr(args, "u_tri_state_required", False)) and not bool(
        getattr(args, "u_geometry_all_valid_queries", False)
    ):
        raise ValueError(
            "--u_tri_state_required true requires --u_geometry_all_valid_queries true; "
            "selected U_s samples cannot bypass geometry-first routing."
        )
    if bool(getattr(args, "u_direct_include_outside_known", False)) and not bool(
        getattr(args, "direct_metric_positive_first", False)
    ):
        raise ValueError(
            "--u_direct_include_outside_known true requires "
            "--direct_metric_positive_first true so source-known U_s cannot be repelled as unknown"
        )
    if (
        bool(getattr(args, "direct_metric_require_effective_negative_grad", False))
        and bool(getattr(args, "direct_metric_virtual_detach", True))
        and bool(getattr(args, "direct_metric_gate_reference_detach", True))
    ):
        raise ValueError(
            "--direct_metric_require_effective_negative_grad true requires either "
            "--direct_metric_virtual_detach false or --direct_metric_gate_reference_detach false"
        )
    if args.dry_run:
        print(
            f"[DRY-RUN] Parsed arguments and skipped data/model construction. "
            f"label_epochs={args.label_epochs} pseudo_epochs={args.pseudo_epochs} total_epochs={total_epochs}",
            flush=True,
        )
        return 0
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run SSDG.train_ssdg training.")
    if str(args.dataset).lower() != "wisig":
        raise ValueError("SSDG.train_ssdg currently implements the WiSig tx_rx_day_1_7_2 protocol.")
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    _prepare_cuda_memory_audit(device)
    out_dir = ensure_dir(args.output_dir)
    stale_identity_paths = [
        out_dir / "phase1_terminal_status.json",
        out_dir / "phase1_training_completion_receipt.json",
        out_dir / "phase1_ablation_manifest.json",
        out_dir / f"best_{args.best_metric}_ssdg.pth",
        out_dir / "final_ssdg.pth",
        out_dir / "latest_ssdg.pth",
        out_dir / "tail_reference_ssdg.pth",
        out_dir / "latest_finite_ssdg.pth",
        out_dir / "recovery_e90_ssdg.pth",
        out_dir / "first_rc4_anomaly.pt",
    ]
    stale_identity_paths = [path for path in stale_identity_paths if path.exists()]
    if stale_identity_paths:
        raise FileExistsError(
            "Phase1 output directory contains stale run identity artifacts; use a new candidate directory: "
            + ", ".join(str(path) for path in stale_identity_paths)
        )
    if ablation_manifest is not None:
        if not str(getattr(args, "phase2_export_path", "")).strip():
            args.phase2_export_path = str(
                out_dir / "phase2_zid_prototypes.pt"
            )
        ablation_manifest = {
            **ablation_manifest,
            "run_id": str(args.run_id),
            "candidate_id": str(args.candidate_id),
            "train_seed": int(args.seed),
            "row_key": str(args.row_key),
            "sealed_plan_sha256": str(
                args.sealed_plan_sha256
            ),
            "seed_registry_sha256": str(
                args.seed_registry_sha256
            ),
            "wisig_pkl_sha256": str(
                args.wisig_pkl_sha256
            ),
            "python_environment_id": str(
                args.python_environment_id
            ),
            "output_dir": str(out_dir),
            "phase2_export_path": str(args.phase2_export_path),
        }
        (out_dir / "phase1_ablation_manifest.json").write_text(
            json.dumps(
                ablation_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    formal_dataset_receipt = (
        _validate_formal_dataset_receipt(args)
        if bool(getattr(args, "formal_ablation", False))
        else {}
    )
    formal_environment_receipt = (
        _validate_formal_environment_receipt(args)
        if bool(getattr(args, "formal_ablation", False))
        else {}
    )
    metrics_csv_path = Path(str(args.metrics_csv).strip()) if str(args.metrics_csv).strip() else out_dir / "metrics_epoch.csv"
    metrics_jsonl_path = Path(str(args.metrics_jsonl).strip()) if str(args.metrics_jsonl).strip() else out_dir / "metrics_epoch.jsonl"
    final_path = out_dir / "final_ssdg.pth"
    default_safe_best_name = f"best_{args.best_metric}_ssdg.pth" if _joint_safe_guard_enabled(args) else f"best_{args.best_metric}_safe_ssdg.pth"
    safe_best_path = Path(str(args.safe_best_path).strip()) if str(args.safe_best_path).strip() else out_dir / default_safe_best_name
    safe_latest_path = Path(str(args.safe_latest_path).strip()) if str(args.safe_latest_path).strip() else out_dir / "latest_safe_ssdg.pth"
    phase2_audit_state = _phase2_audit_state(args)
    data_ctx = _build_ssdg_wisig_data(args, device)
    use_ckpt = bool(str(args.baseline_ckpt).strip()) and not bool(args.from_scratch)
    ckpt = load_checkpoint(args.baseline_ckpt, device) if use_ckpt else {"model": None, "args": {}, "stats": {}, "split_info": None}
    model_args = merge_checkpoint_args(ckpt, args, input_len=int(data_ctx["input_len"]), num_domains=int(data_ctx["num_domains"]))
    model_args = _apply_model_cli_args(model_args, args)
    model = build_baseline_model(model_args, device)
    if use_ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if bool(args.freeze_backbone):
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name for key in ("cls_head", "dom_head", "adv_head"))
    muse_state = _initialize_muse_training_state(args, model, device)
    if muse_state is not None and use_ckpt:
        _restore_muse_checkpoint_state(muse_state, ckpt)
    trainable_params = int(sum(p.numel() for p in _optimizer_parameters(model, muse_state)))
    total_params = int(sum(p.numel() for p in model.parameters()))
    if muse_state is not None:
        total_params += int(sum(p.numel() for p in muse_state["heads"].parameters()))
    ema_model = None
    if bool(args.use_ema_teacher) or muse_state is not None:
        ema_model = deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    teacher_model = None
    if _frozen_teacher_requested(args):
        if not str(getattr(args, "teacher_ckpt", "")).strip():
            raise ValueError("--teacher_ckpt is required for frozen-teacher objectives")
        if _teacher_distill_requested(args) and one_way_kl_from_teacher is None:
            raise ImportError("cvsrffi.losses.one_way_kl_from_teacher is required for teacher distillation")
        teacher_ckpt = load_checkpoint(str(args.teacher_ckpt), device)
        teacher_model_args = merge_checkpoint_args(
            teacher_ckpt,
            argparse.Namespace(),
            input_len=int(data_ctx["input_len"]),
            num_domains=int(data_ctx["num_domains"]),
        )
        # Student and fixed teacher must expose the same z_id semantics.  This
        # prevents teacher_zid_mse from silently pulling an invariant core back
        # toward the legacy defect-gated joint feature.
        teacher_model_args.id_feature_key = str(getattr(args, "id_feature_key", "feat_joint"))
        teacher_model = build_baseline_model(teacher_model_args, device)
        teacher_model.load_state_dict(teacher_ckpt["model"], strict=False)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    if bool(getattr(args, "rc4_cache_anchor_logits", False)):
        if not (
            muse_state is not None
            and bool(muse_state.get("fasttrust_rc4", False))
            and bool(args.rc4_use_anchor)
            and teacher_model is not None
        ):
            raise ValueError(
                "--rc4_cache_anchor_logits requires FastTrust RC4 with a frozen anchor"
            )
        if float(args.rc4_lambda_feature_anchor) > 0.0:
            raise ValueError(
                "anchor-logit cache cannot serve a nonzero feature-anchor loss"
            )
        cache_started = time.time()
        pre_cache_rng = _capture_training_rng_state()
        try:
            muse_state["anchor_logit_cache"] = _build_rc4_anchor_logit_cache(
                teacher_model,
                data_ctx["unlabeled_loader"].dataset,
                batch_size=_resolve_unlabeled_batch_size(args),
                num_workers=int(args.num_workers),
                prefetch_factor=int(args.prefetch_factor),
                domain_label_map=data_ctx["domain_label_map"],
                device=device,
                amp_enabled=bool(args.amp and device.type == "cuda"),
            )
        finally:
            _restore_training_rng_state(pre_cache_rng)
        print(
            "[RC4-ANCHOR-CACHE] "
            f"rows={muse_state['anchor_logit_cache']['rows']} "
            f"build_s={time.time() - cache_started:.3f} "
            "storage=train_only truth_access=0"
        )
    if _sat_anchor_requested(args):
        muse_state["sat_anchor_thresholds"] = _calibrate_sat_anchor_vcal(
            teacher_model,
            data_ctx["source_calibration_loader"],
            domain_label_map=data_ctx["domain_label_map"],
            device=device,
            num_classes=int(getattr(model, "num_classes", args.num_classes)),
            epsilon=float(args.sat_anchor_calibration_epsilon),
        )
    rc4_anomaly_path = out_dir / "first_rc4_anomaly.pt"
    rc4_anomaly_written = False
    optimizer_parameters = (
        _fasttrust_optimizer_groups(model, muse_state)
        if _fasttrust_lr_enabled(args)
        else _optimizer_parameters(model, muse_state)
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scaler = GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    proto_bank = None
    if bool(getattr(args, "use_proto_memory", False)) or float(getattr(args, "lambda_proto", 0.0)) > 0.0:
        if PrototypeMemoryBank is None:
            raise ImportError("cvsrffi.losses.PrototypeMemoryBank is required for --use_proto_memory/--lambda_proto")
        proto_bank = PrototypeMemoryBank(
            int(getattr(args, "num_classes", 0)),
            int(data_ctx["num_domains"]),
            momentum=float(args.proto_momentum),
            margin=float(args.proto_margin),
            domain_align_weight=float(args.proto_domain_align_weight),
            push_weight=float(args.proto_push_weight),
            min_count=int(args.proto_min_count),
        )
    loss_warn_counts: Dict[str, int] = {}
    sat_gen = make_torch_generator(device, int(args.seed) + 991) if make_torch_generator is not None else None
    concat_sat_aug = None
    if bool(getattr(args, "use_concat_sat_channel_aug", False)):
        if ConcatSatChannelAugment is None or apply_sat_channel_for_scenario is None:
            raise ImportError("concat_sat_channel_aug.py and sat_channel.py support are required for --use_concat_sat_channel_aug.")
        concat_sat_aug = ConcatSatChannelAugment(
            scenarios=getattr(args, "sat_train_scenario_list", [args.sat_train_scenario]),
            schedule=str(getattr(args, "sat_view_schedule", "") or ""),
            p=float(getattr(args, "sat_view_prob", 1.0)),
            seed=int(getattr(args, "sat_view_seed", args.seed)),
            apply_fn=apply_sat_channel_for_scenario,
        )
    aug_base_cfg = build_aug_base_cfg(args) if bool(args.use_aug) and build_aug_base_cfg is not None else None
    augmentor = make_augmentor(aug_base_cfg) if aug_base_cfg is not None and make_augmentor is not None else None
    print(
        "\n".join(
            [
                "[CONFIG-RUN] schema=ssdg_config_v1 "
                f"seed={int(args.seed)} device={device} output_dir={out_dir} baseline_ckpt={args.baseline_ckpt or '<scratch>'} "
                f"from_scratch={int(bool(args.from_scratch))} freeze_backbone={int(bool(args.freeze_backbone))}",
                "[CONFIG-DATA] "
                f"dataset={getattr(args, 'dataset', 'wisig')} split_mode={args.split_mode} "
                f"L/U/Vcal/Vselect={data_ctx['split_info']['labeled_size']}/{data_ctx['split_info']['unlabeled_size']}/"
                f"{data_ctx['split_info'].get('source_calibration_size', data_ctx['split_info']['source_val_size'])}/"
                f"{data_ctx['split_info'].get('source_selection_size', 0)} "
                f"ratios={float(args.labeled_ratio):.3f}/{float(args.unlabeled_ratio):.3f}/"
                f"{float(getattr(args, 'source_cal_ratio', 0.0)):.3f}/{float(getattr(args, 'source_select_ratio', 0.0)):.3f} "
                f"role_protocol={getattr(args, 'phase1_source_role_protocol', 'legacy_l_u_v')}",
                "[CONFIG-OPT] "
                f"optimizer=AdamW lr={float(args.lr):.6g} weight_decay={float(args.weight_decay):.6g} amp={int(bool(args.amp))} "
                f"params_trainable={trainable_params} params_total={total_params} "
                f"label_epochs={int(args.label_epochs)} pseudo_epochs={int(args.pseudo_epochs)} total_epochs={int(total_epochs)} "
                f"best_metric={args.best_metric}",
                "[CONFIG-LOSS] "
                f"lambda_domain={float(args.lambda_domain):.6g} lambda_adv={float(args.lambda_adv):.6g} "
                f"lambda_orth={float(args.lambda_orth):.6g} lambda_cons={float(args.lambda_cons):.6g} "
                f"lambda_group_ce={float(args.lambda_group_ce):.6g} lambda_fishr={float(args.lambda_fishr):.6g} "
                f"lambda_sat_cls={float(args.lambda_sat_cls):.6g} lambda_sat_cons={float(args.lambda_sat_cons):.6g} "
                f"lambda_proto={float(args.lambda_proto):.6g} lambda_open_world_feat={float(args.lambda_open_world_feat):.6g} "
                f"lambda_zid_compact={float(args.lambda_zid_compact):.6g} lambda_proxy_unknown={float(args.lambda_proxy_unknown):.6g} "
                f"proxy_virtual_mode={args.proxy_unknown_virtual_mode} proxy_vaccept_w={float(args.proxy_unknown_vaccept_weight):.6g} "
                f"proxy_gate_w={float(args.proxy_unknown_component_gate_weight):.6g} "
                f"lambda_soft_unknown_mixup={float(args.lambda_soft_unknown_mixup):.6g} "
                f"lambda_source_episode={float(args.lambda_source_episode):.6g} "
                f"lambda_direct_metric_accept={float(args.lambda_direct_metric_accept):.6g} "
                f"lambda_u={float(args.lambda_u):.6g} lambda_ent={float(args.lambda_ent):.6g} "
                f"lambda_u_domain={float(args.lambda_u_domain):.6g} lambda_u_adv={float(args.lambda_u_adv):.6g} "
                f"lambda_u_sat_cons={float(args.lambda_u_sat_cons):.6g} "
                f"lambda_u_direct_metric_accept={float(args.lambda_u_direct_metric_accept):.6g} "
                f"lambda_u_quarantine_accept={float(args.lambda_u_quarantine_accept):.6g} "
                f"label_smoothing={float(args.label_smoothing):.6g}",
                "[CONFIG-P1-INVARIANCE] "
                f"checkpoint_selection={args.checkpoint_selection} "
                f"sat_disjoint_required={int(bool(args.sat_protocol_disjoint_required))} "
                f"sat_disjoint={int(bool((args.sat_protocol_manifest or {}).get('disjoint', False)))} "
                f"sat_train_families={','.join((args.sat_protocol_manifest or {}).get('train_families', [])) or '-'} "
                f"sat_eval_families={','.join((args.sat_protocol_manifest or {}).get('eval_families', [])) or '-'} "
                f"zid_inv_l={float(args.lambda_zid_receiver_invariance):.4g}/"
                f"{float(args.lambda_zid_day_invariance):.4g}/"
                f"{float(args.lambda_zid_channel_invariance):.4g} "
                f"zid_inv_u={float(args.lambda_u_zid_receiver_invariance):.4g}/"
                f"{float(args.lambda_u_zid_day_invariance):.4g}/"
                f"{float(args.lambda_u_zid_channel_invariance):.4g} "
                f"dm_local={int(bool(args.direct_metric_domain_local_components))} "
                f"dm_local_required={int(bool(args.direct_metric_require_domain_local_components))} "
                f"leakage_probe_required={int(bool(args.zid_leakage_probe_required))}",
                "[CONFIG-U-DIRECT] "
                f"domain_start={int(args.u_domain_start_epoch)} sat_start={int(args.u_sat_cons_start_epoch)} "
                f"dm_start={int(args.u_direct_metric_start_epoch)} dm_min_selected={int(args.u_direct_metric_min_selected)} "
                f"dm_sat_pair={int(bool(args.u_direct_metric_use_sat_pair))} dm_valid_domain_only={int(bool(args.u_direct_metric_valid_domain_only))} "
                f"quarantine_start={int(args.u_quarantine_start_epoch)} quarantine_valid_domain_only={int(bool(args.u_quarantine_valid_domain_only))} "
                f"quarantine_sat={int(bool(args.u_quarantine_include_sat_view))} quarantine_min={int(args.u_quarantine_min_count)} "
                f"quarantine_target={float(args.u_quarantine_accept_target):.3f} "
                f"geometry_all_valid={int(bool(args.u_geometry_all_valid_queries))} "
                f"unlabeled_shuffle={int(bool(args.u_unlabeled_shuffle))} "
                f"sat_zid_cons_w={float(args.u_sat_zid_cons_weight):.6g} "
                f"domain_loss={float(args.lambda_u_domain):.6g} adv_loss={float(args.lambda_u_adv):.6g} "
                f"sat_cons={float(args.lambda_u_sat_cons):.6g} dm_accept={float(args.lambda_u_direct_metric_accept):.6g} "
                f"quarantine_accept={float(args.lambda_u_quarantine_accept):.6g}",
                "[CONFIG-TEACHER] "
                f"teacher_ckpt={str(getattr(args, 'teacher_ckpt', '') or '<none>')} "
                f"lambda_teacher_clean_kl={float(args.lambda_teacher_clean_kl):.6g} "
                f"lambda_teacher_sat_kl={float(args.lambda_teacher_sat_kl):.6g} "
                f"lambda_teacher_zid_mse={float(args.lambda_teacher_zid_mse):.6g} "
                f"temperature={float(args.teacher_distill_temperature):.6g} "
                f"start_epoch={int(args.teacher_distill_start_epoch)} warmup_epochs={int(args.teacher_distill_warmup_epochs)}",
                "[CONFIG-ADG] "
                f"bridge_w={float(args.proxy_unknown_bridge_accept_weight):.6g} "
                f"shell_out_w={float(args.proxy_unknown_shell_outward_accept_weight):.6g} "
                f"low_density_w={float(args.proxy_unknown_low_density_accept_weight):.6g} "
                f"energy_q_w={float(args.proxy_unknown_energy_margin_quantile_weight):.6g} "
                f"radius_w={float(args.proxy_unknown_radius_budget_weight):.6g} "
                f"ratio_w={float(args.proxy_unknown_radius_inter_ratio_weight):.6g} "
                f"bridge_target={float(args.proxy_unknown_bridge_accept_target):.6g} "
                f"tail_target={float(args.proxy_unknown_tail_accept_target):.6g} "
                f"overflow_target={float(args.proxy_unknown_overflow_accept_target):.6g}",
                "[CONFIG-DM-ACCEPT] "
                f"start={int(args.direct_metric_start_epoch)} warmup={int(args.direct_metric_warmup_epochs)} "
                f"virtual={int(args.direct_metric_virtual_count)} mode={args.direct_metric_virtual_mode} "
                f"targets=p50:{float(args.direct_metric_zid_p50_target_deg):.2f},p95:{float(args.direct_metric_zid_p95_target_deg):.2f},"
                f"p99:{float(args.direct_metric_zid_p99_target_deg):.2f},tail:{float(args.direct_metric_zid_tail_cvar_target_deg):.2f},"
                f"source_overflow:{float(args.direct_metric_source_overflow_target):.3f},proxy_vaccept:{float(args.direct_metric_proxy_vaccept_target):.3f},"
                f"bridge:{float(args.direct_metric_bridge_accept_target):.3f},low_den:{float(args.direct_metric_low_density_accept_target):.3f},"
                f"tail_accept:{float(args.direct_metric_tail_accept_target):.3f},overflow_accept:{float(args.direct_metric_overflow_accept_target):.3f},"
                f"radius_inter:{float(args.direct_metric_radius_inter_ratio_target):.3f},sat_pair_deg:{float(args.direct_metric_sat_pair_target_deg):.2f} "
                f"weights=zid:{float(args.direct_metric_zid_quantile_weight):.3f},source:{float(args.direct_metric_source_overflow_weight):.3f},"
                f"proxy:{float(args.direct_metric_proxy_vaccept_weight):.3f},bridge:{float(args.direct_metric_bridge_accept_weight):.3f},"
                f"low_den:{float(args.direct_metric_low_density_accept_weight):.3f},tail:{float(args.direct_metric_tail_accept_weight):.3f},"
                f"overflow:{float(args.direct_metric_overflow_accept_weight):.3f},ratio:{float(args.direct_metric_radius_inter_ratio_weight):.3f},"
                f"core:{float(args.direct_metric_core_accept_weight):.3f},sat_pair:{float(args.direct_metric_sat_pair_weight):.3f}",
                "[CONFIG-PSEUDO] "
                f"use_unlabeled={int(bool(args.use_unlabeled))} threshold_mode={args.pseudo_threshold_mode} "
                f"tau_min={float(args.tau_min):.6g} tau_max={float(args.tau_max):.6g} quantile={float(args.pseudo_quantile):.6g} "
                f"domain_gate={int(bool(args.pseudo_domain_gate))} temporal_gate={int(bool(args.pseudo_temporal_gate))} "
                f"strong_agreement={int(bool(args.pseudo_strong_agreement))} ema={int(bool(args.use_ema_teacher))}",
                "[CONFIG-SAT] "
                f"use_sat_consistency={int(bool(args.use_sat_consistency))} train_scenario={args.sat_train_scenario} "
                f"train_scenarios={','.join(getattr(args, 'sat_train_scenario_list', [args.sat_train_scenario]))} "
                f"use_concat_sat_channel_aug={int(bool(getattr(args, 'use_concat_sat_channel_aug', False)))} "
                f"concat_sat_ce_only={int(bool(getattr(args, 'concat_sat_ce_only', False)))} "
                f"sat_training_mode={getattr(args, 'sat_training_mode', '') or '<legacy-inferred>'} "
                f"sat_view_schedule={getattr(args, 'sat_view_schedule', '') or '<none>'} "
                f"sat_cons_start_epoch={int(args.sat_cons_start_epoch)} eval_sat_channel={int(bool(args.eval_sat_channel))} "
                f"eval_sat_scenarios={args.eval_sat_scenarios}",
                "[CONFIG-CONCAT-SAT] "
                f"enabled={int(concat_sat_aug is not None)} "
                f"mode={getattr(args, 'sat_training_mode', '') or '<legacy-inferred>'} "
                f"forward_count={2 if getattr(args, 'sat_training_mode', '') in {'concat_ce_only', 'concat_masked'} else 1} "
                f"samples_per_step={('B+B' if getattr(args, 'sat_training_mode', '') in {'concat_ce_only', 'concat_masked'} else '2B' if getattr(args, 'sat_training_mode', '') in {'concat_full', 'concat_legacy'} else 'B')} "
                f"sat_losses=ce+nuisance+shell;kl={float(getattr(args, 'lambda_crra_sat_kl', 0.0)):.6g};pair={float(getattr(args, 'lambda_crra_pair', 0.0)):.6g} "
                f"scenario_cycle={','.join(concat_sat_aug.scenarios) if concat_sat_aug is not None else '<none>'} "
                f"start_epoch={int(getattr(args, 'concat_sat_start_epoch', 1))} "
                f"view_prob={float(getattr(args, 'sat_view_prob', 1.0)):.3f} "
                f"seed={int(getattr(args, 'sat_view_seed', args.seed))} "
                f"ce_weight={float(getattr(args, 'concat_sat_ce_weight', 1.0)):.3f}",
                "[CONFIG-TELEMETRY] "
                f"metrics_csv={metrics_csv_path} metrics_jsonl={metrics_jsonl_path} "
                "per_epoch_loss_terms=raw_and_weighted",
                "[CONFIG-JOINT-SAFE] "
                f"enabled={int(_joint_safe_guard_enabled(args))} best_metric={args.best_metric} "
                f"safe_best_path={safe_best_path} safe_latest_path={safe_latest_path} "
                f"drop_guard_pp={float(args.one_epoch_drop_guard_pp):.6g} "
                f"paic_guard={int(_paic_guard_enabled(args))} "
                f"paic_sat_ce_delta={float(args.paic_guard_sat_ce_delta):.6g} "
                f"paic_grad_delta={float(args.paic_guard_grad_delta):.6g} "
                f"paic_reliable_drop={float(args.paic_guard_reliable_drop):.6g} "
                f"paic_sat_scale={float(args.paic_guard_sat_scale):.6g}",
                "[CONFIG-PHASE1-V2] "
                f"hard_gates={int(bool(args.phase1_v2_hard_gates))} "
                f"endpoint={str(args.endpoint_accept_policy_id)} "
                f"loss_gate_exported={int(bool(args.loss_gate_exported))} "
                f"tail_sm={int(bool(args.tail_safety_state_machine))} "
                f"tail_targets=p95:{float(args.tail_safety_p95_target_deg):.2f},p99:{float(args.tail_safety_p99_target_deg):.2f},"
                f"cvar:{float(args.tail_safety_cvar_target_deg):.2f},proxy:{float(args.tail_safety_proxy_vaccept_target):.3f} "
                f"tail_expansion_delta=final:{float(args.tail_safety_p99_expansion_block_final_delta):.2f},"
                f"best:{float(args.tail_safety_p99_expansion_block_best_delta):.2f},"
                f"cvar_final:{float(args.tail_safety_cvar_expansion_block_final_delta):.2f},"
                f"cvar_best:{float(args.tail_safety_cvar_expansion_block_best_delta):.2f} "
                f"tail_reference_window={int(args.tail_safety_reference_window)} "
                f"tail_rollback={int(bool(args.tail_rollback_enabled))}:cooldown{int(args.tail_rollback_cooldown_epochs)} "
                f"os_eff_min={float(args.os_eff_min_budget):.3f} "
                f"os_eff_max={float(args.os_eff_max_budget):.3f} "
                f"os_budget_controller={int(bool(args.os_budget_controller))} "
                f"os_gradient_surgery={int(bool(args.os_gradient_surgery))} "
                f"os_eff_all_phases={int(bool(args.phase1_v2_os_eff_all_phases))} "
                f"dg_health_guard={int(bool(args.source_val_dg_health_guard))}:"
                f"warn{float(args.source_val_dg_health_warning_drop_pp):.1f}:"
                f"stop{float(args.source_val_dg_health_stop_drop_pp):.1f}:"
                f"min_open{float(args.source_val_dg_health_min_open_scale):.2f} "
                f"guard_blocks_final={int(bool(args.phase1_v2_guard_blocks_final))} "
                f"u_tri_state_required={int(bool(args.u_tri_state_required))} "
                f"source_episode_density_gate={int(bool(args.source_episode_density_gate))} "
                f"source_episode_overflow_warn={float(args.source_episode_overflow_warn):.3f} "
                f"local_component_weights=compact:{float(args.source_episode_local_compact_weight):.3f},"
                f"invariant:{float(args.source_episode_local_invariant_weight):.3f},"
                f"inter:{float(args.source_episode_local_inter_weight):.3f},"
                f"accept:{float(args.source_episode_local_accept_weight):.3f},"
                f"density:{float(args.source_episode_local_density_weight):.3f} "
                f"local_min_samples={int(args.source_episode_local_min_samples)} "
                f"local_radius_floor={float(args.source_episode_local_radius_floor_deg):.2f}deg "
                f"local_caps=density:{float(args.source_episode_local_density_cap):.2f},"
                f"term:{float(args.source_episode_local_term_cap):.2f} "
                f"structural_schedule=start:{int(args.source_episode_structural_start_epoch)},"
                f"warmup:{int(args.source_episode_structural_warmup_epochs)} "
                f"source_views=clean:{float(args.source_episode_clean_weight):.2f},"
                f"sat:{float(args.source_episode_sat_weight):.2f},"
                f"normalized:{int(bool(args.source_episode_multiview_normalize))} "
                f"max_grad_norm={float(args.max_grad_norm):.2f} "
                f"multiview_separate={int(bool(args.direct_metric_multiview_separate))} "
                f"endpoint_artifact_required={int(bool(args.endpoint_require_artifact_on_export))} "
                f"feasibility_gate={int(bool(args.feasibility_gate))}:{str(args.feasibility_stage)}",
                "[CONFIG-PROTO-MASK] "
                f"requested={int(bool(phase2_audit_state.get('requested', False)))} "
                f"audit_only={int(bool(phase2_audit_state.get('audit_only', True)))} "
                f"active_loss={int(bool(phase2_audit_state.get('active_loss', False)))} "
                f"use_prototypes={int(bool(args.use_phase2_ground_prototypes))} "
                f"use_masks={int(bool(args.use_feature_masks))} "
                f"use_geometry={int(bool(args.use_txrx_geometry_losses))} "
                f"use_balanced_sampler={int(bool(args.use_tx_rx_balanced_sampler))} "
                f"lambda_tx_proto={float(args.lambda_tx_proto):.6g} "
                f"lambda_rx_proto={float(args.lambda_rx_proto):.6g} "
                f"lambda_mask_aux={float(args.lambda_mask_aux):.6g} "
                f"lambda_txrx_rect={float(args.lambda_txrx_rect):.6g} "
                f"use_proto_memory={int(bool(proto_bank is not None))} "
                f"lambda_proto={float(args.lambda_proto):.6g} "
                f"lambda_open_world_feat={float(args.lambda_open_world_feat):.6g} "
                f"lambda_zid_compact={float(args.lambda_zid_compact):.6g} "
                f"lambda_proxy_unknown={float(args.lambda_proxy_unknown):.6g} "
                f"lambda_soft_unknown_mixup={float(args.lambda_soft_unknown_mixup):.6g} "
                f"lambda_source_episode={float(args.lambda_source_episode):.6g} "
                f"phase2_export={int(bool(args.phase2_export_prototypes))}",
            ]
        ),
        flush=True,
    )
    print(
        f"[SSDG-TRAIN] init={'scratch' if not use_ckpt else args.baseline_ckpt} split={data_ctx['split_info']['mode']} "
        f"L/U/Vcal/Vselect={data_ctx['split_info']['labeled_size']}/{data_ctx['split_info']['unlabeled_size']}/"
        f"{data_ctx['split_info'].get('source_calibration_size', data_ctx['split_info']['source_val_size'])}/"
        f"{data_ctx['split_info'].get('source_selection_size', 0)} "
        f"ratios={float(args.labeled_ratio):.3f}/{float(args.unlabeled_ratio):.3f}/"
        f"{float(getattr(args, 'source_cal_ratio', 0.0)):.3f}/{float(getattr(args, 'source_select_ratio', 0.0)):.3f} "
        f"role_protocol={getattr(args, 'phase1_source_role_protocol', 'legacy_l_u_v')} "
        f"label_epochs={args.label_epochs} pseudo_epochs={args.pseudo_epochs} "
        f"lambda_domain={float(args.lambda_domain):.3f} lambda_fishr={float(args.lambda_fishr):.3f} "
        f"threshold={args.pseudo_threshold_mode} domain_gate={int(args.pseudo_domain_gate)} "
        f"temporal_gate={int(args.pseudo_temporal_gate)} ema={int(args.use_ema_teacher)} "
        f"best_metric={args.best_metric} output={out_dir}",
        flush=True,
    )
    print(
        "[TELEMETRY] schema=ssdg_epoch_telemetry_v1 "
        f"metrics_csv={metrics_csv_path} metrics_jsonl={metrics_jsonl_path} "
        "loss_terms=loss,loss_labeled,loss_tx,loss_domain,loss_adv,loss_cons,"
        "loss_orth,loss_group_ce,loss_fishr,loss_sat_cls,loss_sat_cons,"
        "loss_direct_metric_accept,loss_unlabeled,weighted_losses,grad_norms,pseudo_stats,eval_stats",
        flush=True,
    )

    best_score = float("-inf")
    best_val = float("-inf")
    best_test = float("nan")
    best_epoch = 0
    telemetry_rows: List[Dict[str, Any]] = []
    previous_protected_metrics: Dict[str, float] | None = None
    previous_train_logs: Dict[str, Any] | None = None
    paic_cooldown_remaining = 0
    tail_rollback_cooldown_remaining = 0
    tail_early_stop_requested = False
    dg_health_early_stop_requested = False
    dg_health_best_val = float("-inf")
    dg_health_open_scale = 1.0
    dg_health_bad_epochs = 0
    dg_health_drop_pp = 0.0
    tail_rollback_events: List[Dict[str, Any]] = []
    tail_reference_geometry: Dict[str, Any] = {}
    tail_reference_epoch = -1
    pseudo_temporal_bank: Dict[Tuple[int, int, int, int, int], Tuple[int, int, int]] = {}
    direct_metric_reference_bank = (
        FrozenDirectMetricReferenceBank(
            per_component=int(args.direct_metric_reference_per_component),
            refresh_epochs=int(args.direct_metric_reference_refresh_epochs),
        )
        if bool(getattr(args, "direct_metric_reference_bank", False))
        else None
    )
    last_source_val_tail_geometry: Dict[str, Any] = {}
    last_source_val_sat_stats: Dict[str, Dict[str, Any]] = {}
    last_source_val_heavy_eval_epoch = 0
    phase1_v2_tail_machine = None
    phase1_v2_final_blocked = False
    phase1_v2_reasons: List[str] = []
    fasttrust_zero_step_streak = 0
    if bool(getattr(args, "tail_safety_state_machine", False)):
        if TailSafetyStateMachine is None or TailSafetyConfig is None:
            raise ImportError("cvsrffi.phase1_v2_control is required for --tail_safety_state_machine.")
        phase1_v2_tail_machine = TailSafetyStateMachine(
            TailSafetyConfig(
                p95_target_deg=float(args.tail_safety_p95_target_deg),
                p99_target_deg=float(args.tail_safety_p99_target_deg),
                tail_cvar_target_deg=float(args.tail_safety_cvar_target_deg),
                proxy_vaccept_target=float(args.tail_safety_proxy_vaccept_target),
                warning_patience=int(args.tail_safety_warning_patience),
                rollback_patience=int(args.tail_safety_rollback_patience),
                max_rollbacks=int(args.tail_safety_max_rollbacks),
                p99_expansion_block_final_delta=float(args.tail_safety_p99_expansion_block_final_delta),
                p99_expansion_block_best_delta=float(args.tail_safety_p99_expansion_block_best_delta),
                tail_cvar_expansion_block_final_delta=float(args.tail_safety_cvar_expansion_block_final_delta),
                tail_cvar_expansion_block_best_delta=float(args.tail_safety_cvar_expansion_block_best_delta),
                reference_window=int(args.tail_safety_reference_window),
                absolute_violation_drives_state=bool(args.tail_safety_absolute_violation_drives_state),
                training_stop_enabled=bool(args.tail_safety_training_stop_enabled),
                reference_requires_absolute_safe=bool(args.tail_safety_reference_requires_absolute_safe),
            )
        )
    for epoch in range(1, total_epochs + 1):
        if direct_metric_reference_bank is not None:
            direct_metric_reference_bank.maybe_promote(epoch)

        t0 = time.time()
        train_batches_seconds = 0.0
        base_validation_seconds = 0.0
        heavy_source_validation_seconds = 0.0
        checkpoint_io_seconds = 0.0
        if _fasttrust_lr_enabled(args) and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if _fasttrust_lr_enabled(args):
            _apply_fasttrust_lr(optimizer, base_lr=float(args.lr), epoch=int(epoch))
        if muse_state is not None:
            _configure_muse_epoch_state(muse_state, int(epoch))
            if bool(muse_state.get("fasttrust_rc4", False)):
                update_epochs = {
                    int(value.strip())
                    for value in str(args.rc4_calibration_update_epochs).split(",")
                    if value.strip()
                }
                if int(epoch) in update_epochs:
                    muse_state["rc4_calibration"] = _calibrate_rc4_vcal(
                        teacher_model, ema_model, data_ctx["source_calibration_loader"],
                        args=args, domain_label_map=data_ctx["domain_label_map"],
                        device=device, num_classes=int(model.num_classes),
                        num_domains=int(model.num_domains),
                    )
        phase = "label" if epoch <= int(args.label_epochs) else "pseudo"
        stage_state = _stage_state_for_epoch(epoch, args, phase)
        cur_w = _loss_weights(args, stage_state)
        tail_rollback_cooldown_active = bool(tail_rollback_cooldown_remaining > 0)
        tail_closed_scale = (
            max(0.0, min(1.0, float(args.tail_rollback_closed_scale)))
            if tail_rollback_cooldown_active
            else 1.0
        )
        stage_state["tail_rollback_cooldown_active"] = 1.0 if tail_rollback_cooldown_active else 0.0
        stage_state["tail_rollback_cooldown_remaining"] = float(tail_rollback_cooldown_remaining)
        stage_state["tail_rollback_closed_scale"] = float(tail_closed_scale)
        if tail_rollback_cooldown_active:
            tail_rollback_cooldown_remaining = max(0, tail_rollback_cooldown_remaining - 1)
        paic_cooldown_active = bool(phase == "pseudo" and paic_cooldown_remaining > 0)
        if paic_cooldown_active:
            scale = max(0.0, min(1.0, float(args.paic_guard_sat_scale)))
            cur_w["sat_cls"] *= scale
            cur_w["sat_cons"] *= scale
            stage_state["paic_guard_cooldown_active"] = 1.0
            stage_state["paic_guard_sat_scale"] = scale
            paic_cooldown_remaining = max(0, paic_cooldown_remaining - 1)
        else:
            stage_state["paic_guard_cooldown_active"] = 0.0
        if configure_mixstyle_for_epoch is not None:
            mixstyle_state = configure_mixstyle_for_epoch(model, args, epoch)
        else:
            mixstyle_state = _fallback_mixstyle_state(args)
        if augmentor is not None and configure_augmentor_for_epoch is not None:
            aug_state = configure_augmentor_for_epoch(augmentor, aug_base_cfg, min(epoch, int(args.label_epochs)), args)
        else:
            aug_state = _fallback_aug_state(args)
        if hasattr(model, "set_crra_epoch"):
            model.set_crra_epoch(int(epoch))
        if ema_model is not None and hasattr(ema_model, "set_crra_epoch"):
            ema_model.set_crra_epoch(int(epoch))
        if teacher_model is not None and hasattr(teacher_model, "set_crra_epoch"):
            teacher_model.set_crra_epoch(int(epoch))
        model.train()
        balanced_train_sampler = data_ctx.get("balanced_train_sampler")
        if balanced_train_sampler is not None and hasattr(balanced_train_sampler, "set_epoch"):
            balanced_train_sampler.set_epoch(epoch)
        epoch_logs = []
        rc4_nonfinite_batches = 0
        rc4_gradient_metrics = {
            "rc4/g_L": float("nan"),
            "rc4/g_H": float("nan"),
            "rc4/g_Pset": float("nan"),
            "rc4/g_Pcond": float("nan"),
            "rc4/g_adv": float("nan"),
            "rc4/g_identity_to_labeled": float("nan"),
            "rc4/g_adv_to_labeled": float("nan"),
            "rc4/cos_H_to_labeled": float("nan"),
            "rc4/cos_Pset_to_labeled": float("nan"),
            "rc4/cos_Pcond_to_labeled": float("nan"),
            "rc4/gradient_telemetry_active": 0.0,
        }
        legacy_unlabeled_active = _legacy_unlabeled_active(
            args,
            muse_state=muse_state,
            phase=phase,
        )
        unlabeled_iter = (
            iter(data_ctx["unlabeled_loader"])
            if legacy_unlabeled_active
            else None
        )
        epoch_pairs = _muse_epoch_pairs(
            data_ctx["train_loader"],
            data_ctx["unlabeled_loader"],
            use_muse=muse_state is not None,
            use_unlabeled_step_budget=bool(getattr(args, "use_muse_ssdg", False)),
        )
        for batch_idx, (labeled_batch, muse_unlabeled_batch) in enumerate(epoch_pairs, start=1):
            muse_identity_grad_norm = float("nan")
            sat_anchor_pair_grad_norm = float("nan")
            sat_anchor_sat_grad_norm = float("nan")
            sat_anchor_anchor_grad_norm = float("nan")
            sat_anchor_pair_sat_grad_cos = float("nan")
            rc4_route = None
            x_l, y_l, extra_l = move_batch(labeled_batch, device)
            if muse_state is not None:
                _assert_muse_open_geometry_role("L_s")
            labeled_clean_count = int(y_l.numel())
            receiver_l_base = _metadata_label_tensor(extra_l, "rx_i", device, labeled_clean_count)
            day_l_base = _metadata_label_tensor(extra_l, "day_i", device, labeled_clean_count)
            d_l = domain_from_extra(extra_l, data_ctx["domain_label_map"], device)
            concat_active = concat_sat_aug is not None and epoch >= int(getattr(args, "concat_sat_start_epoch", 1))
            if concat_active and augmentor is not None:
                x_l = torch.nan_to_num(
                    augmentor(x_l, labels=y_l, no_pa=(not bool(args.aug_enable_pa_normal))),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            x_l, y_l, d_l, concat_sat_ce_view, concat_sat_info = _prepare_concat_sat_batch_for_training(
                concat_sat_aug,
                x_l,
                y_l,
                d_l,
                args=args,
                epoch=epoch,
                batch_idx=batch_idx,
            )
            concat_sat_full_batch = bool(concat_sat_info.get("expanded", 0.0) > 0.0)
            concat_sat_clean_bsz = int(concat_sat_info.get("clean_batch_size", 0.0))
            receiver_l = _expand_view_metadata(
                receiver_l_base,
                clean_count=labeled_clean_count,
                total_count=int(y_l.numel()),
            )
            day_l = _expand_view_metadata(
                day_l_base,
                clean_count=labeled_clean_count,
                total_count=int(y_l.numel()),
            )
            channel_l = _channel_view_labels(
                int(y_l.numel()),
                concat_sat_clean_bsz,
                concat_sat_full_batch and float(concat_sat_info.get("applied", 0.0)) > 0.0,
                device,
            )
            if augmentor is not None and not concat_active:
                x_l_main = torch.nan_to_num(
                    augmentor(x_l, labels=y_l, no_pa=(not bool(args.aug_enable_pa_normal))),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            else:
                x_l_main = x_l
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp and device.type == "cuda")):
                out_l = model(
                    x_l_main,
                    y_tx=y_l,
                    grl_lambda=(
                        0.0
                        if str(args.identity_domain_objective_mode) == "bounded_confusion"
                        else 1.0
                    ),
                    return_aux=True,
                    domain_labels=d_l,
                    update_crra_support=bool(getattr(args, "use_crra", False)),
                    crra_support_mask=(
                        torch.arange(x_l_main.size(0), device=x_l_main.device) < int(concat_sat_clean_bsz)
                        if bool(getattr(args, "use_crra", False))
                        and concat_sat_full_batch
                        and concat_sat_clean_bsz > 0
                        else None
                    ),
                )
                domain_stats = {"valid": (d_l >= 0) if d_l is not None else None}
                domain_gates = {
                    "dom": d_l is not None and "dom_logits" in out_l and cur_w["dom"] > 0.0,
                    "adv": (
                        d_l is not None
                        and "adv_dom_logits" in out_l
                        and cur_w["adv"] > 0.0
                        and str(args.identity_domain_objective_mode) == "grl_ce"
                    ),
                    "cons": d_l is not None and cur_w["cons"] > 0.0,
                    "group_ce": d_l is not None and cur_w["group_ce"] > 0.0,
                }
                core_losses = compute_core_losses(
                    out_l,
                    y_l,
                    d_l,
                    domain_stats,
                    domain_gates,
                    lambda logits, target: F.cross_entropy(logits, target, label_smoothing=float(args.label_smoothing)),
                    lambda logits, target: F.cross_entropy(logits, target),
                    label_smoothing=float(args.label_smoothing),
                    group_top_frac=float(args.group_ce_top_frac),
                    group_min_domains=int(args.group_ce_min_domains),
                    group_ce_mode=str(args.group_ce_mode),
                )
                loss_tx_l = core_losses["loss_cls"]
                loss_dom_l = core_losses["loss_dom"]
                loss_adv_l = core_losses["loss_adv"]
                loss_adv_discriminator_l = loss_adv_l * 0.0
                loss_adv_confusion_l = loss_adv_l * 0.0
                if (
                    str(args.identity_domain_objective_mode) == "bounded_confusion"
                    and d_l is not None
                    and cur_w["adv"] > 0.0
                    and getattr(model, "adv_head", None) is not None
                ):
                    valid_l_domain = (d_l >= 0) & (d_l < int(data_ctx["num_domains"]))
                    if bool(valid_l_domain.any()):
                        bounded_l = bounded_domain_objectives(
                            model.adv_head,
                            out_l["z_id"][valid_l_domain],
                            d_l[valid_l_domain],
                        )
                        loss_adv_discriminator_l = bounded_l["discriminator"]
                        loss_adv_confusion_l = bounded_l["confusion"]
                        loss_adv_l = (
                            float(args.identity_domain_discriminator_scale)
                            * loss_adv_discriminator_l
                            + float(args.identity_domain_confusion_scale)
                            * loss_adv_confusion_l
                        )
                loss_cons_l = core_losses["loss_cons"]
                loss_orth_l = core_losses["loss_orth"] if cur_w["orth"] > 0.0 else out_l["tx_logits"].sum() * 0.0
                loss_group_ce_l = core_losses["loss_group_ce"]
                if (
                    concat_sat_full_batch
                    and concat_sat_clean_bsz > 0
                    and bool(getattr(args, "concat_sat_deduplicate_tx_ce", False))
                ):
                    # concat_sa keeps both views in the domain/invariance path,
                    # but TX CE is counted once per view: clean here and sat in
                    # loss_sat_cls_l below.
                    loss_tx_l = F.cross_entropy(
                        out_l["tx_logits"][:concat_sat_clean_bsz],
                        y_l[:concat_sat_clean_bsz],
                        label_smoothing=float(args.label_smoothing),
                    )
                if d_l is not None and cur_w["fishr"] > 0.0:
                    loss_fishr_l = fishr_logit_gradient_variance_loss(
                        out_l["tx_logits"],
                        y_l,
                        d_l,
                        min_domains=int(args.fishr_min_domains),
                    )
                else:
                    loss_fishr_l = out_l["tx_logits"].sum() * 0.0
                z_id_l = out_l["z_id"]
                loss_muse_local_l = _compute_muse_labeled_auxiliary_loss(
                    z_id_l,
                    y_l,
                    d_l,
                    muse_state,
                )
                loss_zid_invariance_l = z_id_l.sum() * 0.0
                zid_invariance_info: Dict[str, float] = {
                    "active": 0.0,
                    "receiver_active": 0.0,
                    "day_active": 0.0,
                    "channel_active": 0.0,
                    "channel_pair_active": 0.0,
                    "receiver_loss": 0.0,
                    "day_loss": 0.0,
                    "channel_loss": 0.0,
                    "channel_pair_loss": 0.0,
                    "channel_pair_count": 0.0,
                    "channel_pair_angle_deg": float("nan"),
                }
                if any(
                    float(value) > 0.0
                    for value in (
                        args.lambda_zid_receiver_invariance,
                        args.lambda_zid_day_invariance,
                        args.lambda_zid_channel_invariance,
                    )
                ):
                    if tx_conditional_domain_invariance_loss is None:
                        raise ImportError("cvsrffi.losses.tx_conditional_domain_invariance_loss is required")
                    loss_zid_invariance_l, zid_invariance_info = tx_conditional_domain_invariance_loss(
                        z_id_l,
                        y_l,
                        receiver_labels=receiver_l,
                        day_labels=day_l,
                        channel_labels=channel_l,
                        receiver_weight=float(args.lambda_zid_receiver_invariance),
                        day_weight=float(args.lambda_zid_day_invariance),
                        channel_weight=float(args.lambda_zid_channel_invariance),
                        channel_pair_weight=float(args.zid_channel_pair_weight),
                        paired_view_count=(
                            concat_sat_clean_bsz
                            if concat_sat_full_batch and float(concat_sat_info.get("applied", 0.0)) > 0.0
                            else 0
                        ),
                        min_groups=int(args.zid_invariance_min_groups),
                        min_samples_per_group=int(args.zid_invariance_min_samples_per_group),
                    )
                teacher_scale = _teacher_distill_scale(args, epoch)
                loss_teacher_clean_kl_l = z_id_l.sum() * 0.0
                loss_teacher_sat_kl_l = z_id_l.sum() * 0.0
                loss_teacher_zid_mse_l = z_id_l.sum() * 0.0
                teacher_clean_out = None
                if teacher_model is not None and teacher_scale > 0.0:
                    teacher_clean_only = bool(
                        concat_sat_full_batch
                        and concat_sat_clean_bsz > 0
                        and getattr(args, "concat_sat_teacher_clean_only", False)
                    )
                    teacher_clean_count = (
                        int(concat_sat_clean_bsz)
                        if teacher_clean_only
                        else int(y_l.numel())
                    )
                    teacher_x = x_l_main[:teacher_clean_count]
                    teacher_y = y_l[:teacher_clean_count]
                    teacher_d = d_l[:teacher_clean_count] if d_l is not None else None
                    with torch.no_grad():
                        teacher_clean_out = teacher_model(
                            teacher_x,
                            y_tx=teacher_y,
                            grl_lambda=1.0,
                            return_aux=True,
                            domain_labels=teacher_d,
                        )
                    if float(args.lambda_teacher_clean_kl) > 0.0:
                        loss_teacher_clean_kl_l = one_way_kl_from_teacher(
                            out_l["tx_logits"][:teacher_clean_count],
                            teacher_clean_out["tx_logits"],
                            temperature=float(args.teacher_distill_temperature),
                        )
                    if float(args.lambda_teacher_zid_mse) > 0.0:
                        loss_teacher_zid_mse_l = F.mse_loss(
                            F.normalize(z_id_l[:teacher_clean_count].float(), dim=1),
                            F.normalize(teacher_clean_out["z_id"].detach().float(), dim=1),
                        )
                loss_proto_l = z_id_l.sum() * 0.0
                proto_info: Dict[str, float] = {
                    "proto_pull_cos": float("nan"),
                    "proto_domain_align": 0.0,
                    "proto_push": 0.0,
                    "proto_active_classes": 0.0,
                }
                if proto_bank is not None:
                    loss_proto_l, proto_info = proto_bank.loss(z_id_l, y_l, d_l)
                    if proto_bank.class_count is not None:
                        active = proto_bank.class_count >= int(args.proto_min_count)
                        proto_info["proto_active_classes"] = float(int(active.sum().detach().item()))
                loss_open_world_feat_l = z_id_l.sum() * 0.0
                ow_feat_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=int(getattr(args, "ow_feat_start_epoch", 1)),
                    warmup_epochs=int(getattr(args, "ow_feat_warmup_epochs", 0)),
                )
                ow_feat_info: Dict[str, float] = {
                    "compact": 0.0,
                    "inter": 0.0,
                    "sample_margin": 0.0,
                    "domain_align": 0.0,
                    "active_classes": 0.0,
                    "pos_angle_deg": float("nan"),
                    "min_inter_angle_deg": float("nan"),
                    "pos_angle_p50_deg": float("nan"),
                    "pos_angle_p95_deg": float("nan"),
                    "pos_angle_p99_deg": float("nan"),
                    "pos_angle_max_deg": float("nan"),
                    "tail_loss": 0.0,
                    "tail_cvar_deg": float("nan"),
                    "tail_frac_gt_3sigma": 0.0,
                    "tail_radius_3sigma_deg": float("nan"),
                    "vacuum_loss": 0.0,
                    "vacuum_violation_rate": 0.0,
                    "vacuum_min_neg_angle_deg": float("nan"),
                    "vacuum_margin_deg": float("nan"),
                    "vacuum_boundary_deg": float("nan"),
                }
                if float(args.lambda_open_world_feat) > 0.0 and ow_feat_stage_scale > 0.0:
                    if open_world_feature_space_loss is None:
                        raise ImportError("cvsrffi.losses.open_world_feature_space_loss is required for --lambda_open_world_feat")
                    loss_open_world_feat_l, ow_feat_info = open_world_feature_space_loss(
                        z_id_l,
                        y_l,
                        d_l,
                        radius_rad=math.radians(float(args.ow_feat_radius_deg)),
                        inter_margin_rad=math.radians(float(args.ow_feat_inter_margin_deg)),
                        sample_margin_rad=math.radians(float(args.ow_feat_sample_margin_deg)),
                        domain_align_weight=float(args.ow_feat_domain_align_weight),
                        min_classes=int(args.ow_feat_min_classes),
                        min_samples_per_class=int(args.ow_feat_min_samples_per_class),
                        tail_mode=str(args.ow_feat_tail_mode),
                        tail_weight=float(args.ow_feat_tail_weight),
                        cvar_alpha=float(args.ow_feat_cvar_alpha),
                        vacuum_weight=float(args.ow_feat_vacuum_weight),
                        vacuum_width_rad=math.radians(float(args.ow_feat_vacuum_width_deg)),
                        vacuum_hard_k=int(args.ow_feat_vacuum_hard_k),
                    )
                loss_zid_compact_l = z_id_l.sum() * 0.0
                zid_compact_info: Dict[str, float] = {
                    "supcon": 0.0,
                    "radius": 0.0,
                    "tail_cvar": 0.0,
                    "active_classes": 0.0,
                    "pos_angle_p50_deg": float("nan"),
                    "pos_angle_p95_deg": float("nan"),
                    "pos_angle_p99_deg": float("nan"),
                    "tail_cvar_deg": float("nan"),
                }
                zid_warm = _stage_gate_scale(
                    epoch,
                    start_epoch=int(getattr(args, "zid_compact_start_epoch", 1)),
                    warmup_epochs=int(getattr(args, "zid_compact_warmup_epochs", 0)),
                )
                if float(args.lambda_zid_compact) > 0.0 and zid_warm > 0.0:
                    if zid_compactness_loss is None:
                        raise ImportError("cvsrffi.losses.zid_compactness_loss is required for --lambda_zid_compact")
                    loss_zid_compact_l, zid_compact_info = zid_compactness_loss(
                        z_id_l,
                        y_l,
                        d_l,
                        radius_rad=math.radians(float(args.zid_compact_radius_deg)),
                        cvar_alpha=float(args.zid_compact_cvar_alpha),
                        supcon_weight=float(args.zid_compact_supcon_weight),
                        radius_weight=float(args.zid_compact_radius_weight),
                        cvar_weight=float(args.zid_compact_cvar_weight),
                        domain_aware=bool(args.zid_compact_domain_aware),
                    )
                loss_proxy_unknown_l = z_id_l.sum() * 0.0
                proxy_unknown_info: Dict[str, float] = {
                    "active": 0.0,
                    "known_count": 0.0,
                    "proxy_unknown_count": 0.0,
                    "virtual_count": 0.0,
                    "core_count": 0.0,
                    "tail_count": 0.0,
                    "overflow_count": 0.0,
                    "energy_known": float("nan"),
                    "energy_proxy": float("nan"),
                    "energy_virtual": float("nan"),
                    "energy_margin": float("nan"),
                    "accept_energy_threshold": float("nan"),
                    "core_energy_threshold": float("nan"),
                    "vaccept_surrogate": 0.0,
                    "vaccept_surrogate_CVaR": 0.0,
                    "core_accept_loss": 0.0,
                    "component_gate_unknown": 0.0,
                    "component_gate_accept_prob": float("nan"),
                    "component_gate_accept_prob_max": float("nan"),
                    "tail_quarantine_loss": 0.0,
                    "source_safe_loss": 0.0,
                    "bridge_governance_loss": 0.0,
                    "shell_outward_accept_loss": 0.0,
                    "low_density_accept_loss": 0.0,
                    "energy_margin_quantile_loss": 0.0,
                    "radius_budget_loss": 0.0,
                    "radius_inter_ratio_loss": 0.0,
                    "tail_accept_loss": 0.0,
                    "overflow_accept_loss": 0.0,
                    "energy_margin_q05": float("nan"),
                    "energy_margin_q10": float("nan"),
                    "component_radius_p95_deg": float("nan"),
                    "component_radius_max_deg": float("nan"),
                    "component_radius_mode_code": 1.0,
                    "component_gate_radius_p95_deg": float("nan"),
                    "component_gate_radius_max_deg": float("nan"),
                    "radius_inter_ratio": float("nan"),
                    "radius_to_inter_ratio": float("nan"),
                    "low_density_accept_prob": float("nan"),
                    "low_density_accept_rate": float("nan"),
                    "proxy_unknown_auc": float("nan"),
                    "virtual_accept_rate": float("nan"),
                    "proxy_vaccept": float("nan"),
                    "proxy_vaccept_proxy_only": float("nan"),
                    "proxy_reject_claim_allowed": 0.0,
                    "virtual_accept_rate_core": float("nan"),
                    "proxy_accept_rate": float("nan"),
                    "hard_proxy_accept_rate": float("nan"),
                    "shell_accept_rate": float("nan"),
                    "bridge_accept_rate": float("nan"),
                    "outward_accept_rate": float("nan"),
                    "vacuum_loss": 0.0,
                    "vacuum_violation_rate": 0.0,
                    "vacuum_margin_deg": float("nan"),
                    "vacuum_min_angle_deg": float("nan"),
                }
                proxy_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=int(getattr(args, "proxy_unknown_start_epoch", 40)),
                    warmup_epochs=int(getattr(args, "proxy_unknown_warmup_epochs", 0)),
                )
                source_episode_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=int(getattr(args, "source_episode_start_epoch", 1)),
                    warmup_epochs=int(getattr(args, "source_episode_warmup_epochs", 0)),
                )
                source_structural_start_epoch = int(
                    getattr(args, "source_episode_structural_start_epoch", -1)
                )
                if source_structural_start_epoch <= 0:
                    source_structural_start_epoch = int(getattr(args, "source_episode_start_epoch", 1))
                source_structural_warmup_epochs = int(
                    getattr(args, "source_episode_structural_warmup_epochs", -1)
                )
                if source_structural_warmup_epochs < 0:
                    source_structural_warmup_epochs = int(getattr(args, "source_episode_warmup_epochs", 0))
                source_episode_structural_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=source_structural_start_epoch,
                    warmup_epochs=source_structural_warmup_epochs,
                )
                direct_metric_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=int(getattr(args, "direct_metric_start_epoch", 20)),
                    warmup_epochs=int(getattr(args, "direct_metric_warmup_epochs", 20)),
                )
                soft_mixup_start_epoch = int(getattr(args, "soft_unknown_mixup_start_epoch", -1))
                if soft_mixup_start_epoch <= 0:
                    soft_mixup_start_epoch = int(getattr(args, "proxy_unknown_start_epoch", 40))
                soft_mixup_warmup_epochs = int(getattr(args, "soft_unknown_mixup_warmup_epochs", -1))
                if soft_mixup_warmup_epochs < 0:
                    soft_mixup_warmup_epochs = int(getattr(args, "proxy_unknown_warmup_epochs", 0))
                loss_soft_unknown_mixup_l = z_id_l.sum() * 0.0
                soft_unknown_mixup_info: Dict[str, float] = {
                    "soft_unknown_mixup_count": 0.0,
                    "soft_unknown_mixup_order": float(max(2, int(getattr(args, "soft_unknown_mixup_order", 3)))),
                    "soft_unknown_mixup_ce": 0.0,
                    "soft_unknown_mixup_energy": 0.0,
                    "soft_unknown_mixup_vacuum": 0.0,
                    "soft_unknown_mixup_virtual_accept_rate": 0.0,
                    "soft_unknown_mixup_vacuum_violation": 0.0,
                }
                soft_unknown_mixup_batch = None
                soft_unknown_mixup_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=soft_mixup_start_epoch,
                    warmup_epochs=soft_mixup_warmup_epochs,
                )
                soft_mixup_needed = (
                    (float(getattr(args, "lambda_soft_unknown_mixup", 0.0)) > 0.0 and soft_unknown_mixup_stage_scale > 0.0)
                    or (
                        float(args.lambda_source_episode) > 0.0
                        and source_episode_stage_scale > 0.0
                        and float(getattr(args, "source_episode_mixup_weight", 0.0)) > 0.0
                    )
                )
                if soft_mixup_needed:
                    if make_soft_unknown_mixup is None:
                        raise ImportError("cvsrffi.losses.make_soft_unknown_mixup is required for soft unknown mixup")
                    soft_unknown_mixup_batch = make_soft_unknown_mixup(
                        z_id_l,
                        y_l,
                        mixup_count=int(args.soft_unknown_mixup_count),
                        mixup_order=int(args.soft_unknown_mixup_order),
                        alpha=float(args.soft_unknown_mixup_alpha),
                    )
                proxy_active = float(args.lambda_proxy_unknown) > 0.0 and proxy_stage_scale > 0.0
                if proxy_active:
                    if proxy_unknown_energy_loss is None:
                        raise ImportError("cvsrffi.losses.proxy_unknown_energy_loss is required for --lambda_proxy_unknown")
                    valid_y = torch.unique(y_l[y_l >= 0])
                    if valid_y.numel() > 0:
                        holdout_idx = (int(epoch) + int(batch_idx)) % int(valid_y.numel())
                        holdout_label = int(valid_y[holdout_idx].detach().item())
                    else:
                        holdout_label = None
                    loss_proxy_unknown_l, proxy_unknown_info = proxy_unknown_energy_loss(
                        z_id_l,
                        y_l,
                        holdout_label=holdout_label,
                        virtual_count=int(args.proxy_unknown_virtual_count),
                        virtual_mode=str(args.proxy_unknown_virtual_mode),
                        energy_margin=float(args.proxy_unknown_energy_margin),
                        energy_temperature=float(args.proxy_unknown_energy_temperature),
                        placeholder_weight=float(args.proxy_unknown_placeholder_weight),
                        virtual_detach=bool(args.proxy_unknown_virtual_detach),
                        vacuum_weight=float(args.proxy_unknown_vacuum_weight),
                        vacuum_width_rad=math.radians(float(args.proxy_unknown_vacuum_width_deg)),
                        vacuum_hard_k=int(args.proxy_unknown_vacuum_hard_k),
                        vacuum_radius_rad=math.radians(float(args.proxy_unknown_vacuum_radius_deg)),
                        core_quantile=float(args.proxy_unknown_core_quantile),
                        accept_quantile=float(args.proxy_unknown_accept_quantile),
                        tail_quantile=float(args.proxy_unknown_tail_quantile),
                        overflow_quantile=float(args.proxy_unknown_overflow_quantile),
                        component_radius_mode=str(args.proxy_unknown_component_radius_mode),
                        component_radius_quantile=float(args.proxy_unknown_component_radius_quantile),
                        vaccept_weight=float(args.proxy_unknown_vaccept_weight),
                        core_accept_weight=float(args.proxy_unknown_core_accept_weight),
                        component_gate_weight=float(args.proxy_unknown_component_gate_weight),
                        tail_quarantine_weight=float(args.proxy_unknown_tail_quarantine_weight),
                        source_safe_weight=float(args.proxy_unknown_source_safe_weight),
                        bridge_accept_weight=float(args.proxy_unknown_bridge_accept_weight),
                        shell_outward_accept_weight=float(args.proxy_unknown_shell_outward_accept_weight),
                        low_density_accept_weight=float(args.proxy_unknown_low_density_accept_weight),
                        energy_margin_quantile_weight=float(args.proxy_unknown_energy_margin_quantile_weight),
                        radius_budget_weight=float(args.proxy_unknown_radius_budget_weight),
                        radius_inter_ratio_weight=float(args.proxy_unknown_radius_inter_ratio_weight),
                        vaccept_cvar_alpha=float(args.proxy_unknown_vaccept_cvar_alpha),
                        unknown_margin=float(args.proxy_unknown_unknown_margin),
                        known_margin=float(args.proxy_unknown_known_margin),
                        energy_softplus_temperature=float(args.proxy_unknown_energy_softplus_temperature),
                        accept_softplus_temperature=float(args.proxy_unknown_accept_softplus_temperature),
                        bridge_accept_target=float(args.proxy_unknown_bridge_accept_target),
                        shell_outward_accept_target=float(args.proxy_unknown_shell_outward_accept_target),
                        tail_accept_target=float(args.proxy_unknown_tail_accept_target),
                        overflow_accept_target=float(args.proxy_unknown_overflow_accept_target),
                        energy_margin_q=float(args.proxy_unknown_energy_margin_q),
                        energy_margin_target=float(args.proxy_unknown_energy_margin_target),
                        radius_budget_rad=math.radians(float(args.proxy_unknown_radius_budget_deg)),
                        radius_max_budget_rad=math.radians(float(args.proxy_unknown_radius_max_budget_deg)),
                        radius_inter_ratio_target=float(args.proxy_unknown_radius_inter_ratio_target),
                        density_temperature_rad=math.radians(float(args.proxy_unknown_density_temperature_deg)),
                        component_temperature_rad=math.radians(float(args.proxy_unknown_component_temperature_deg)),
                        component_margin_rad=math.radians(float(args.proxy_unknown_component_margin_deg)),
                        component_margin_temperature_rad=math.radians(float(args.proxy_unknown_component_margin_temperature_deg)),
                        shell_width_rad=math.radians(float(args.proxy_unknown_shell_width_deg)),
                    )
                if float(getattr(args, "lambda_soft_unknown_mixup", 0.0)) > 0.0 and soft_unknown_mixup_stage_scale > 0.0:
                    if soft_unknown_mixup_loss is None:
                        raise ImportError("cvsrffi.losses.soft_unknown_mixup_loss is required for --lambda_soft_unknown_mixup")
                    loss_soft_unknown_mixup_l, soft_unknown_mixup_info = soft_unknown_mixup_loss(
                        z_id_l,
                        y_l,
                        logits=out_l["tx_logits"],
                        mixup=soft_unknown_mixup_batch,
                        mixup_count=int(args.soft_unknown_mixup_count),
                        mixup_order=int(args.soft_unknown_mixup_order),
                        alpha=float(args.soft_unknown_mixup_alpha),
                        energy_margin=float(args.soft_unknown_mixup_energy_margin),
                        ce_weight=float(args.soft_unknown_mixup_ce_weight),
                        energy_weight=float(args.soft_unknown_mixup_energy_weight),
                        vacuum_weight=float(args.soft_unknown_mixup_vacuum_weight),
                        vacuum_width_rad=math.radians(float(args.soft_unknown_mixup_vacuum_width_deg)),
                        vacuum_hard_k=int(args.soft_unknown_mixup_vacuum_hard_k),
                        detach_mixup=bool(args.soft_unknown_mixup_detach),
                    )
                loss_source_episode_l = z_id_l.sum() * 0.0
                source_episode_info: Dict[str, float] = {
                    "source_episode_loss": 0.0,
                    "source_episode_overflow_rate": 0.0,
                    "source_overflow": 0.0,
                    "source_episode_radius_3sigma_deg": float("nan"),
                    "source_episode_radius_core_deg": float("nan"),
                    "source_episode_radius_safe_deg": float("nan"),
                    "source_episode_radius_mode_code": 2.0,
                    "source_episode_val_angle_deg": float("nan"),
                    "source_episode_tail_query_rate": 0.0,
                    "source_episode_classes": 0.0,
                    "source_episode_domains": 0.0,
                    "source_episode_mixup_count": 0.0,
                    "source_episode_mixup_order": float(max(2, int(getattr(args, "soft_unknown_mixup_order", 3)))),
                    "source_episode_mixup_loss": 0.0,
                    "source_episode_mixup_overflow_rate": 0.0,
                    "source_episode_mixup_margin_deg": float("nan"),
                }
                if float(args.lambda_source_episode) > 0.0 and source_episode_stage_scale > 0.0:
                    if source_episode_three_sigma_loss is None:
                        raise ImportError("cvsrffi.losses.source_episode_three_sigma_loss is required for --lambda_source_episode")
                    source_episode_kwargs = dict(
                        min_domains=int(args.source_episode_min_domains),
                        radius_cap_rad=math.radians(float(args.source_episode_radius_cap_deg)),
                        min_sigma_rad=math.radians(float(args.source_episode_min_sigma_deg)),
                        radius_mode=str(args.source_episode_radius_mode),
                        core_quantile=float(args.source_episode_core_quantile),
                        mixup_features=soft_unknown_mixup_batch.features if soft_unknown_mixup_batch is not None else None,
                        mixup_weight=float(args.source_episode_mixup_weight),
                        mixup_order=int(args.soft_unknown_mixup_order),
                        mixup_hard_k=int(args.source_episode_mixup_hard_k),
                        local_component_compact_weight=(
                            float(args.source_episode_local_compact_weight)
                            * source_episode_structural_stage_scale
                        ),
                        local_component_invariant_weight=(
                            float(args.source_episode_local_invariant_weight)
                            * source_episode_structural_stage_scale
                        ),
                        local_component_inter_weight=(
                            float(args.source_episode_local_inter_weight)
                            * source_episode_structural_stage_scale
                        ),
                        local_component_inter_margin_rad=math.radians(float(args.source_episode_local_inter_margin_deg)),
                        local_component_center_target_rad=math.radians(
                            float(args.source_episode_local_center_target_deg)
                        ),
                        local_component_overlap_weight=(
                            float(args.source_episode_local_overlap_weight)
                            * source_episode_structural_stage_scale
                        ),
                        local_component_overlap_margin_rad=math.radians(
                            float(args.source_episode_local_overlap_margin_deg)
                        ),
                        local_component_accept_weight=(
                            float(args.source_episode_local_accept_weight)
                            * source_episode_structural_stage_scale
                        ),
                        local_component_density_weight=(
                            float(args.source_episode_local_density_weight)
                            * source_episode_structural_stage_scale
                        ),
                        local_component_min_samples=int(args.source_episode_local_min_samples),
                        local_component_radius_floor_rad=math.radians(
                            float(args.source_episode_local_radius_floor_deg)
                        ),
                        local_component_density_beta=float(args.source_episode_local_density_beta),
                        local_component_density_cap=float(args.source_episode_local_density_cap),
                        local_component_term_cap=float(args.source_episode_local_term_cap),
                        leave_domain_target_rad=math.radians(
                            float(args.source_episode_leave_domain_target_deg)
                        ),
                        leave_domain_target_weight=(
                            float(args.source_episode_leave_domain_target_weight)
                            * source_episode_structural_stage_scale
                        ),
                        structural_cvar_alpha=float(args.source_episode_structural_cvar_alpha),
                    )
                    if (
                        concat_sat_full_batch
                        and concat_sat_clean_bsz > 0
                        and z_id_l.size(0) >= 2 * concat_sat_clean_bsz
                        and multiview_source_episode_three_sigma_loss is not None
                    ):
                        clean_slice = slice(0, concat_sat_clean_bsz)
                        sat_slice = slice(concat_sat_clean_bsz, 2 * concat_sat_clean_bsz)
                        loss_source_episode_l, source_episode_info = multiview_source_episode_three_sigma_loss(
                            z_id_l[clean_slice],
                            z_id_l[sat_slice],
                            y_l[clean_slice],
                            d_l[clean_slice] if d_l is not None else None,
                            clean_weight=float(args.source_episode_clean_weight),
                            sat_weight=float(args.source_episode_sat_weight),
                            normalize_active_weights=bool(args.source_episode_multiview_normalize),
                            **source_episode_kwargs,
                        )
                    else:
                        loss_source_episode_l, source_episode_info = source_episode_three_sigma_loss(
                            z_id_l,
                            y_l,
                            d_l,
                            **source_episode_kwargs,
                        )
                loss_direct_metric_accept_l = z_id_l.sum() * 0.0
                direct_metric_info: Dict[str, float] = {
                    "active": 0.0,
                    "active_classes": 0.0,
                    "zid_p50_deg": float("nan"),
                    "zid_p95_deg": float("nan"),
                    "zid_p99_deg": float("nan"),
                    "zid_tail_cvar_deg": float("nan"),
                    "source_overflow": float("nan"),
                    "source_overflow_loss": 0.0,
                    "proxy_vaccept": float("nan"),
                    "proxy_vaccept_loss": 0.0,
                    "bridge_accept_rate": float("nan"),
                    "bridge_accept_loss": 0.0,
                    "shell_accept_rate": float("nan"),
                    "outward_accept_rate": float("nan"),
                    "low_density_accept_rate": float("nan"),
                    "low_density_accept_loss": 0.0,
                    "tail_accept_rate": float("nan"),
                    "tail_accept_loss": 0.0,
                    "overflow_accept_rate": float("nan"),
                    "overflow_accept_loss": 0.0,
                    "radius_to_inter_ratio": float("nan"),
                    "radius_inter_ratio_loss": 0.0,
                    "core_accept_rate": float("nan"),
                    "core_accept_loss": 0.0,
                    "sat_pair_angle_p95_deg": float("nan"),
                    "sat_pair_loss": 0.0,
                    "zid_quantile_loss": 0.0,
                    "virtual_count": 0.0,
                }
                dm_bank_clean = dm_bank_clean_y = dm_bank_clean_d = None
                dm_bank_sat = dm_bank_sat_y = dm_bank_sat_d = None
                if direct_metric_reference_bank is not None:
                    if concat_sat_full_batch:
                        (
                            dm_bank_clean,
                            dm_bank_sat,
                            dm_bank_clean_y,
                            dm_bank_clean_d,
                        ) = direct_metric_reference_bank.paired_tensors()
                        dm_bank_sat_y = dm_bank_clean_y
                        dm_bank_sat_d = dm_bank_clean_d
                    else:
                        dm_bank_clean, dm_bank_clean_y, dm_bank_clean_d = direct_metric_reference_bank.tensors(view=0)
                    if concat_sat_full_batch and concat_sat_clean_bsz > 0:
                        direct_metric_reference_bank.observe(
                            z_id_l[:concat_sat_clean_bsz],
                            y_l[:concat_sat_clean_bsz],
                            d_l[:concat_sat_clean_bsz] if d_l is not None else None,
                            view=0,
                        )
                        direct_metric_reference_bank.observe(
                            z_id_l[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz],
                            y_l[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz],
                            d_l[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz] if d_l is not None else None,
                            view=1,
                        )
                    else:
                        direct_metric_reference_bank.observe(z_id_l, y_l, d_l, view=0)
                if float(getattr(args, "lambda_direct_metric_accept", 0.0)) > 0.0 and direct_metric_stage_scale > 0.0:
                    if direct_metric_acceptance_loss is None:
                        raise ImportError("cvsrffi.losses.direct_metric_acceptance_loss is required for --lambda_direct_metric_accept")
                    paired_view_count = concat_sat_clean_bsz if concat_sat_full_batch else 0
                    dm_kwargs = _direct_metric_kwargs(args)
                    if (
                        bool(getattr(args, "direct_metric_multiview_separate", False))
                        and concat_sat_full_batch
                        and paired_view_count > 0
                        and multiview_direct_metric_acceptance_loss is not None
                    ):
                        clean_slice = slice(0, paired_view_count)
                        sat_slice = slice(paired_view_count, 2 * paired_view_count)
                        loss_direct_metric_accept_l, direct_metric_info = multiview_direct_metric_acceptance_loss(
                            z_id_l[clean_slice],
                            z_id_l[sat_slice],
                            y_l[clean_slice],
                            d_l[clean_slice] if d_l is not None else None,
                            clean_weight=float(args.direct_metric_clean_weight),
                            sat_weight=float(args.direct_metric_sat_weight),
                            pair_weight=float(args.direct_metric_sat_pair_weight),
                            sat_pair_target_rad=math.radians(float(args.direct_metric_sat_pair_target_deg)),
                            clean_reference_z=dm_bank_clean,
                            sat_reference_z=dm_bank_sat,
                            reference_y=dm_bank_clean_y,
                            reference_d=dm_bank_clean_d,
                            **dm_kwargs,
                        )
                    else:
                        dm_bank_z = dm_bank_clean
                        dm_bank_y = dm_bank_clean_y
                        dm_bank_d = dm_bank_clean_d
                        if dm_bank_clean is not None and dm_bank_sat is not None:
                            dm_bank_z = torch.cat([dm_bank_clean, dm_bank_sat], dim=0)
                            dm_bank_y = torch.cat([dm_bank_clean_y, dm_bank_sat_y], dim=0)
                            dm_bank_d = torch.cat([dm_bank_clean_d, dm_bank_sat_d], dim=0)
                        loss_direct_metric_accept_l, direct_metric_info = direct_metric_acceptance_loss(
                            z_id_l,
                            y_l,
                            d_l,
                            paired_view_count=paired_view_count,
                            sat_pair_target_rad=math.radians(float(args.direct_metric_sat_pair_target_deg)),
                            sat_pair_weight=float(args.direct_metric_sat_pair_weight),
                            reference_z=dm_bank_z,
                            reference_y=dm_bank_y,
                            reference_d=dm_bank_d,
                            **dm_kwargs,
                        )
                zero_sat = out_l["tx_logits"].sum() * 0.0
                loss_sat_cls_l = zero_sat
                loss_sat_cons_l = zero_sat
                loss_crra_pair_l = zero_sat
                loss_crra_energy_l = zero_sat
                loss_crra_gate_l1_l = zero_sat
                loss_crra_nuisance_l = zero_sat
                loss_crra_sat_shell_l = zero_sat
                loss_crra_condition_tx_adv_l = zero_sat
                crra_pair_cosine_l = zero_sat
                crra_condition_tx_adv_acc_l = zero_sat
                crra_sat_shell_info = {
                    "valid_count": 0.0,
                    "active_classes": 0.0,
                    "mean_excess_rad": 0.0,
                }
                out_sat = None
                sat_z_id_l = None
                sat_zid_pair_applied = False
                sat_nuisance = None
                sat_nuisance_valid = None
                sat_nuisance_fields = tuple()
                crra_stage_scale = (
                    crra_gate_scale(
                        int(epoch),
                        start_epoch=int(getattr(args, "crra_start_epoch", 17)),
                        ramp_epochs=int(getattr(args, "crra_ramp_epochs", 30)),
                    )
                    if bool(getattr(args, "use_crra", False))
                    else 0.0
                )
                crra_sat_kl_active = _crra_satellite_kl_active(
                    cur_w["sat_cons"],
                    use_crra=bool(getattr(args, "use_crra", False)),
                    crra_stage_scale=crra_stage_scale,
                    crra_sat_kl_weight=float(getattr(args, "lambda_crra_sat_kl", 0.0)),
                )
                crra_sat_objective_active = bool(getattr(args, "use_crra", False)) and any(
                    float(getattr(args, name, 0.0)) > 0.0
                    for name in (
                        "lambda_crra_pair",
                        "lambda_crra_sat_kl",
                        "lambda_crra_sat_shell",
                        "lambda_crra_nuisance",
                    )
                )
                sat_objective_active = (
                    cur_w["sat_cls"] > 0.0
                    or cur_w["sat_cons"] > 0.0
                    or crra_sat_objective_active
                )
                sat_objective_start_epoch = (
                    min(
                        int(args.sat_cons_start_epoch),
                        int(getattr(args, "crra_start_epoch", args.sat_cons_start_epoch)),
                    )
                    if crra_sat_objective_active
                    else int(args.sat_cons_start_epoch)
                )
                use_sat_train = (
                    bool(args.use_sat_consistency)
                    and (not concat_sat_full_batch)
                    and concat_sat_ce_view is None
                    and epoch >= sat_objective_start_epoch
                    and sat_objective_active
                )
                if (
                    concat_sat_full_batch
                    and concat_sat_clean_bsz > 0
                    and int(y_l.numel()) >= 2 * concat_sat_clean_bsz
                    and epoch >= sat_objective_start_epoch
                    and sat_objective_active
                ):
                    clean_slice = slice(0, concat_sat_clean_bsz)
                    sat_slice = slice(concat_sat_clean_bsz, 2 * concat_sat_clean_bsz)
                    sat_logits = out_l["tx_logits"][sat_slice]
                    sat_y = y_l[sat_slice]
                    sat_z_id_l = out_l["z_id"][sat_slice]
                    sat_nuisance = concat_sat_info.get("crra_nuisance")
                    sat_nuisance_valid = concat_sat_info.get("crra_nuisance_valid")
                    sat_nuisance_fields = tuple(concat_sat_info.get("crra_nuisance_fields") or ())
                    if cur_w["sat_cls"] > 0.0:
                        loss_sat_cls_l = F.cross_entropy(sat_logits, sat_y)
                    if crra_sat_kl_active:
                        clean_prob = out_l["tx_logits"][clean_slice].detach().softmax(dim=1)
                        loss_sat_cons_l = F.kl_div(
                            F.log_softmax(sat_logits, dim=1),
                            clean_prob,
                            reduction="batchmean",
                        )
                    if (
                        teacher_model is not None
                        and teacher_scale > 0.0
                        and float(args.lambda_teacher_sat_kl) > 0.0
                        and teacher_clean_out is not None
                    ):
                        loss_teacher_sat_kl_l = one_way_kl_from_teacher(
                            sat_logits,
                            teacher_clean_out["tx_logits"][:concat_sat_clean_bsz],
                            temperature=float(args.teacher_distill_temperature),
                        )
                elif use_sat_train:
                    if apply_sat_channel_for_scenario is None:
                        raise ImportError("sat_channel.py support is required when --use_sat_consistency is enabled.")
                    sat_train_scenarios = list(getattr(args, "sat_train_scenario_list", [args.sat_train_scenario]))
                    sat_view_stages = tuple(getattr(args, "sat_view_stages", tuple()))
                    if sat_view_stages:
                        active_stage = sat_view_stages[0]
                        for stage in sat_view_stages:
                            if epoch >= int(stage.start_epoch):
                                active_stage = stage
                            else:
                                break
                        sat_train_scenarios = list(active_stage.scenarios) or sat_train_scenarios
                    sat_train_scenario = sat_train_scenarios[
                        (int(epoch) + int(batch_idx) - 2) % max(1, len(sat_train_scenarios))
                    ]
                    with torch.no_grad():
                        x_sat, raw_sat_meta = apply_sat_channel_for_scenario(
                            x_l,
                            str(sat_train_scenario),
                            args,
                            gen=sat_gen,
                            return_meta=bool(getattr(args, "use_crra", False)),
                        )
                    if bool(getattr(args, "use_crra", False)) and normalize_crra_nuisance_meta is not None:
                        _, sat_nuisance, sat_nuisance_valid, sat_nuisance_fields = normalize_crra_nuisance_meta(
                            raw_sat_meta,
                            scenario=str(sat_train_scenario),
                            batch_size=int(x_l.size(0)),
                            device=x_l.device,
                        )
                    out_sat = model(x_sat, y_tx=y_l, grl_lambda=1.0, return_aux=True, domain_labels=d_l)
                    sat_z_id_l = out_sat["z_id"]
                    sat_zid_pair_applied = True
                    loss_sat_cls_l = (
                        F.cross_entropy(out_sat["tx_logits"], y_l)
                        if cur_w["sat_cls"] > 0.0
                        else out_l["tx_logits"].sum() * 0.0
                    )
                    clean_prob = out_l["tx_logits"].detach().softmax(dim=1)
                    loss_sat_cons_l = F.kl_div(
                        F.log_softmax(out_sat["tx_logits"], dim=1),
                        clean_prob,
                        reduction="batchmean",
                    )
                    if (
                        teacher_model is not None
                        and teacher_scale > 0.0
                        and float(args.lambda_teacher_sat_kl) > 0.0
                        and teacher_clean_out is not None
                    ):
                        loss_teacher_sat_kl_l = one_way_kl_from_teacher(
                            out_sat["tx_logits"],
                            teacher_clean_out["tx_logits"],
                            temperature=float(args.teacher_distill_temperature),
                        )
                elif concat_sat_ce_view is not None and epoch >= int(args.sat_cons_start_epoch):
                    x_sat = _safe_iq_tensor(concat_sat_ce_view.x)
                    out_sat = model(x_sat, y_tx=y_l, grl_lambda=1.0, return_aux=True, domain_labels=d_l)
                    sat_z_id_l = out_sat["z_id"]
                    sat_zid_pair_applied = bool(float(concat_sat_info.get("applied", 0.0)) > 0.0)
                    sat_nuisance = concat_sat_ce_view.nuisance
                    sat_nuisance_valid = concat_sat_ce_view.nuisance_valid
                    sat_nuisance_fields = tuple(concat_sat_ce_view.nuisance_fields or ())
                    if cur_w["sat_cls"] > 0.0:
                        loss_sat_cls_l = float(args.concat_sat_ce_weight) * F.cross_entropy(out_sat["tx_logits"], y_l)
                    if crra_sat_kl_active:
                        clean_prob = out_l["tx_logits"].detach().softmax(dim=1)
                        loss_sat_cons_l = F.kl_div(
                            F.log_softmax(out_sat["tx_logits"], dim=1),
                            clean_prob,
                            reduction="batchmean",
                        )
                    if (
                        teacher_model is not None
                        and teacher_scale > 0.0
                        and float(args.lambda_teacher_sat_kl) > 0.0
                        and teacher_clean_out is not None
                    ):
                        loss_teacher_sat_kl_l = one_way_kl_from_teacher(
                            out_sat["tx_logits"],
                            teacher_clean_out["tx_logits"],
                            temperature=float(args.teacher_distill_temperature),
                        )
                if (
                    concat_sat_full_batch
                    and concat_sat_clean_bsz > 0
                    and int(y_l.numel()) >= 2 * concat_sat_clean_bsz
                ):
                    clean_slice = slice(0, concat_sat_clean_bsz)
                    sat_slice = slice(concat_sat_clean_bsz, 2 * concat_sat_clean_bsz)
                    sat_z_id_l = out_l["z_id"][sat_slice]
                    sat_zid_pair_applied = bool(float(concat_sat_info.get("applied", 0.0)) > 0.0)
                    sat_nuisance = concat_sat_info.get("crra_nuisance")
                    sat_nuisance_valid = concat_sat_info.get("crra_nuisance_valid")
                    sat_nuisance_fields = tuple(concat_sat_info.get("crra_nuisance_fields") or ())

                if bool(getattr(args, "use_crra", False)):
                    clean_crra_slice = (
                        slice(0, concat_sat_clean_bsz)
                        if concat_sat_full_batch and concat_sat_clean_bsz > 0
                        else slice(None)
                    )
                    clean_z_crra = out_l["z_id"][clean_crra_slice]
                    clean_energy = out_l.get("crra_correction_energy")
                    clean_gate = out_l.get("crra_gate")
                    clean_alpha = out_l.get("crra_alpha")
                    if torch.is_tensor(clean_energy):
                        clean_energy = clean_energy[clean_crra_slice]
                    if torch.is_tensor(clean_gate):
                        clean_gate = clean_gate[clean_crra_slice]
                    if torch.is_tensor(clean_alpha):
                        clean_alpha = clean_alpha[clean_crra_slice]
                    sat_z_crra = sat_z_id_l if sat_z_id_l is not None and sat_zid_pair_applied else None
                    if torch.is_tensor(sat_z_crra):
                        clean_y_crra = y_l
                        sat_y_crra = y_l
                        if concat_sat_full_batch and concat_sat_clean_bsz > 0:
                            clean_y_crra = y_l[:concat_sat_clean_bsz]
                            sat_y_crra = y_l[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz]
                        pair_count = min(int(clean_z_crra.size(0)), int(sat_z_crra.size(0)))
                        if pair_count > 0:
                            pair_cos = (
                                F.normalize(clean_z_crra[:pair_count].float(), dim=1)
                                * F.normalize(sat_z_crra[:pair_count].float(), dim=1)
                            ).sum(dim=1)
                            crra_pair_cosine_l = pair_cos.mean()
                        if float(getattr(args, "lambda_crra_pair", 0.0)) > 0.0 and pair_count > 0:
                            loss_crra_pair_l = (1.0 - pair_cos).mean()
                        if float(getattr(args, "lambda_crra_sat_shell", 0.0)) > 0.0:
                            if crra_satellite_shell_loss is None:
                                raise ImportError(
                                    "cvsrffi.losses.crra_satellite_shell_loss is required for CRRA satellite shell loss"
                                )
                            loss_crra_sat_shell_l, crra_sat_shell_info = crra_satellite_shell_loss(
                                clean_z_crra,
                                clean_y_crra,
                                sat_z_crra,
                                sat_y_crra,
                                shell_width_rad=math.radians(
                                    float(getattr(args, "crra_sat_shell_width_deg", 12.0))
                                ),
                            )
                    energy_terms = [value for value in (clean_energy,) if torch.is_tensor(value)]
                    gate_terms = [value for value in (clean_gate, clean_alpha) if torch.is_tensor(value)]
                    sat_aux_regularizers_active = _crra_satellite_aux_regularizers_active(args)
                    if out_sat is not None and sat_aux_regularizers_active:
                        sat_energy = out_sat.get("crra_correction_energy")
                        sat_gate = out_sat.get("crra_gate")
                        sat_alpha = out_sat.get("crra_alpha")
                        if torch.is_tensor(sat_energy):
                            energy_terms.append(sat_energy)
                        if torch.is_tensor(sat_gate):
                            gate_terms.append(sat_gate)
                        if torch.is_tensor(sat_alpha):
                            gate_terms.append(sat_alpha)
                    elif (
                        concat_sat_full_batch
                        and concat_sat_clean_bsz > 0
                        and sat_aux_regularizers_active
                    ):
                        sat_slice = slice(concat_sat_clean_bsz, 2 * concat_sat_clean_bsz)
                        full_energy = out_l.get("crra_correction_energy")
                        full_gate = out_l.get("crra_gate")
                        full_alpha = out_l.get("crra_alpha")
                        if torch.is_tensor(full_energy):
                            energy_terms.append(full_energy[sat_slice])
                        if torch.is_tensor(full_gate):
                            gate_terms.append(full_gate[sat_slice])
                        if torch.is_tensor(full_alpha):
                            gate_terms.append(full_alpha[sat_slice])
                    if energy_terms and float(getattr(args, "lambda_crra_energy", 0.0)) > 0.0:
                        loss_crra_energy_l = torch.cat([value.reshape(-1) for value in energy_terms]).mean()
                    if gate_terms and float(getattr(args, "lambda_crra_gate_l1", 0.0)) > 0.0:
                        loss_crra_gate_l1_l = torch.cat([value.reshape(-1) for value in gate_terms]).abs().mean()
                    nuisance_pred = None
                    if out_sat is not None:
                        nuisance_pred = out_sat.get("crra_nuisance_pred")
                    elif concat_sat_full_batch and concat_sat_clean_bsz > 0:
                        nuisance_pred = out_l.get("crra_nuisance_pred")
                        if torch.is_tensor(nuisance_pred):
                            nuisance_pred = nuisance_pred[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz]
                    if torch.is_tensor(sat_nuisance) and torch.is_tensor(nuisance_pred):
                        if sat_nuisance.size(0) == int(y_l.numel()) and concat_sat_full_batch:
                            sat_nuisance = sat_nuisance[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz]
                            if torch.is_tensor(sat_nuisance_valid):
                                sat_nuisance_valid = sat_nuisance_valid[concat_sat_clean_bsz : 2 * concat_sat_clean_bsz]
                        if crra_nuisance_huber_loss is None:
                            raise ImportError("cvsrffi.losses.crra_nuisance_huber_loss is required for CRRA nuisance regression")
                        loss_crra_nuisance_l, crra_nuisance_info = crra_nuisance_huber_loss(
                            nuisance_pred,
                            sat_nuisance,
                            sat_nuisance_valid,
                        )
                    else:
                        crra_nuisance_info = {"valid_count": 0.0, "field_count": 0.0}
                    crra_condition_tx_adv = out_l.get("crra_condition_tx_adv_logits")
                    if (
                        torch.is_tensor(crra_condition_tx_adv)
                        and int(crra_condition_tx_adv.size(0)) == int(y_l.numel())
                    ):
                        crra_condition_tx_adv_acc_l = (
                            crra_condition_tx_adv.detach().argmax(dim=1) == y_l.long()
                        ).float().mean()
                        if float(getattr(args, "lambda_crra_condition_tx_adv", 0.0)) > 0.0:
                            loss_crra_condition_tx_adv_l = F.cross_entropy(
                                crra_condition_tx_adv.float(), y_l.long()
                            )
                else:
                    crra_nuisance_info = {"valid_count": 0.0, "field_count": 0.0}
                if (
                    sat_z_id_l is not None
                    and sat_zid_pair_applied
                    and float(args.lambda_zid_channel_invariance) > 0.0
                    and float(args.zid_channel_pair_weight) > 0.0
                ):
                    if tx_conditional_domain_invariance_loss is None:
                        raise ImportError("cvsrffi.losses.tx_conditional_domain_invariance_loss is required")
                    loss_sat_zid_pair_l, sat_zid_pair_info = _labeled_channel_pair_invariance_loss(
                        z_id_l,
                        sat_z_id_l,
                        y_l,
                        channel_weight=float(args.lambda_zid_channel_invariance),
                        channel_pair_weight=float(args.zid_channel_pair_weight),
                        min_groups=int(args.zid_invariance_min_groups),
                        min_samples_per_group=int(args.zid_invariance_min_samples_per_group),
                    )
                    loss_zid_invariance_l = loss_zid_invariance_l + loss_sat_zid_pair_l
                    zid_invariance_info["active"] = max(
                        float(zid_invariance_info.get("active", 0.0)),
                        float(sat_zid_pair_info.get("active", 0.0)),
                    )
                    zid_invariance_info["channel_active"] = max(
                        float(zid_invariance_info.get("channel_active", 0.0)),
                        float(sat_zid_pair_info.get("channel_active", 0.0)),
                    )
                    zid_invariance_info["channel_pair_active"] = max(
                        float(zid_invariance_info.get("channel_pair_active", 0.0)),
                        float(sat_zid_pair_info.get("channel_pair_active", 0.0)),
                    )
                    zid_invariance_info["channel_loss"] = float(
                        zid_invariance_info.get("channel_loss", 0.0)
                    ) + float(sat_zid_pair_info.get("channel_loss", 0.0))
                    zid_invariance_info["channel_pair_loss"] = float(
                        zid_invariance_info.get("channel_pair_loss", 0.0)
                    ) + float(sat_zid_pair_info.get("channel_pair_loss", 0.0))
                    zid_invariance_info["channel_pair_count"] = max(
                        float(zid_invariance_info.get("channel_pair_count", 0.0)),
                        float(sat_zid_pair_info.get("channel_pair_count", 0.0)),
                    )
                    zid_invariance_info["channel_pair_angle_deg"] = float(
                        sat_zid_pair_info.get("channel_pair_angle_deg", float("nan"))
                    )
                loss_closed_l = (
                    loss_tx_l
                    + loss_muse_local_l
                    + cur_w["dom"] * loss_dom_l
                    + cur_w["adv"] * loss_adv_l
                    + cur_w["orth"] * loss_orth_l
                    + cur_w["cons"] * loss_cons_l
                    + cur_w["group_ce"] * loss_group_ce_l
                    + cur_w["fishr"] * loss_fishr_l
                    + cur_w["sat_cls"] * loss_sat_cls_l
                    + cur_w["sat_cons"] * loss_sat_cons_l
                    + crra_stage_scale * float(getattr(args, "lambda_crra_sat_kl", 0.0)) * sanitize_loss(
                        "crra_sat_kl", loss_sat_cons_l, z_id_l, loss_warn_counts
                    )
                    + crra_stage_scale * float(getattr(args, "lambda_crra_pair", 0.0)) * sanitize_loss(
                        "crra_pair", loss_crra_pair_l, z_id_l, loss_warn_counts
                    )
                    + crra_stage_scale * float(getattr(args, "lambda_crra_sat_shell", 0.0)) * sanitize_loss(
                        "crra_sat_shell", loss_crra_sat_shell_l, z_id_l, loss_warn_counts
                    )
                    + crra_stage_scale * float(getattr(args, "lambda_crra_energy", 0.0)) * sanitize_loss(
                        "crra_energy", loss_crra_energy_l, z_id_l, loss_warn_counts
                    )
                    + crra_stage_scale * float(getattr(args, "lambda_crra_gate_l1", 0.0)) * sanitize_loss(
                        "crra_gate_l1", loss_crra_gate_l1_l, z_id_l, loss_warn_counts
                    )
                    + crra_stage_scale * float(getattr(args, "lambda_crra_nuisance", 0.0)) * sanitize_loss(
                        "crra_nuisance", loss_crra_nuisance_l, z_id_l, loss_warn_counts
                    )
                    + crra_stage_scale * float(getattr(args, "lambda_crra_condition_tx_adv", 0.0)) * sanitize_loss(
                        "crra_condition_tx_adv", loss_crra_condition_tx_adv_l, z_id_l, loss_warn_counts
                    )
                    + (float(args.lambda_teacher_clean_kl) * teacher_scale) * sanitize_loss("teacher_clean_kl", loss_teacher_clean_kl_l, z_id_l, loss_warn_counts)
                    + (float(args.lambda_teacher_sat_kl) * teacher_scale) * sanitize_loss("teacher_sat_kl", loss_teacher_sat_kl_l, z_id_l, loss_warn_counts)
                    + (float(args.lambda_teacher_zid_mse) * teacher_scale) * sanitize_loss("teacher_zid_mse", loss_teacher_zid_mse_l, z_id_l, loss_warn_counts)
                )
                loss_open_invariant_l = (
                    sanitize_loss("ssdg_zid_domain_invariance", loss_zid_invariance_l, z_id_l, loss_warn_counts)
                    + cur_w["proto"] * sanitize_loss("ssdg_proto", loss_proto_l, z_id_l, loss_warn_counts)
                )
                loss_open_boundary_l = (
                    (cur_w["open_world_feat"] * ow_feat_stage_scale) * sanitize_loss("ssdg_open_world_feat", loss_open_world_feat_l, z_id_l, loss_warn_counts)
                    + (cur_w["zid_compact"] * zid_warm) * sanitize_loss("ssdg_zid_compact", loss_zid_compact_l, z_id_l, loss_warn_counts)
                    + (cur_w["proxy_unknown"] * proxy_stage_scale) * sanitize_loss("ssdg_proxy_unknown", loss_proxy_unknown_l, z_id_l, loss_warn_counts)
                    + (cur_w["soft_unknown_mixup"] * soft_unknown_mixup_stage_scale) * sanitize_loss("ssdg_soft_unknown_mixup", loss_soft_unknown_mixup_l, z_id_l, loss_warn_counts)
                    + (cur_w["direct_metric_accept"] * direct_metric_stage_scale) * sanitize_loss("ssdg_direct_metric_accept", loss_direct_metric_accept_l, z_id_l, loss_warn_counts)
                )
                loss_open_source_l = (cur_w["source_episode"] * source_episode_stage_scale) * sanitize_loss(
                    "ssdg_source_episode", loss_source_episode_l, z_id_l, loss_warn_counts
                )
                loss_open_l = loss_open_invariant_l + loss_open_boundary_l + loss_open_source_l
                loss_l = loss_closed_l + loss_open_l
                if muse_state is not None:
                    if muse_unlabeled_batch is None:
                        raise RuntimeError("MUSE epoch pair is missing its U_s batch")
                    x_u, extra_u = _move_muse_unlabeled_batch(
                        muse_unlabeled_batch, device
                    )
                    unlabeled_count = int(x_u.size(0))
                    receiver_u = _metadata_label_tensor(
                        extra_u, "rx_i", device, unlabeled_count
                    )
                    d_u = domain_from_extra(
                        extra_u, data_ctx["domain_label_map"], device
                    )
                    schedule = muse_state["schedule_state"]
                    pseudo_source = ema_model if ema_model is not None else model
                    with torch.no_grad():
                        pseudo_source.eval()
                        out_w2 = None
                        if bool(muse_state.get("fasttrust_rc4", False)):
                            x_w2 = _strong_augment(
                                x_u, max(1e-5, float(args.strong_noise_std) * 0.25)
                            )
                            combined_w = torch.cat([x_u, x_w2], dim=0)
                            combined_d = torch.cat([d_u, d_u], dim=0)
                            combined_out = pseudo_source(
                                combined_w, y_tx=None,
                                grl_lambda=float(schedule.grl_lambda), return_aux=True,
                                domain_labels=combined_d,
                            )
                            out_w, out_w2 = _split_muse_output(
                                combined_out, unlabeled_count, int(combined_w.shape[0])
                            )
                        else:
                            out_w = pseudo_source(
                                x_u, y_tx=None,
                                grl_lambda=float(schedule.grl_lambda), return_aux=True,
                                domain_labels=d_u,
                            )
                        out_anchor = None
                        if bool(muse_state.get("sat_anchor_ssl", False)):
                            if teacher_model is None:
                                raise RuntimeError("SAT_ANCHOR_FROZEN_TEACHER_MISSING")
                            anchor_cache = muse_state.get("anchor_logit_cache")
                            if (
                                bool(muse_state.get("fasttrust_rc4", False))
                                and anchor_cache is not None
                            ):
                                out_anchor = {
                                    "tx_logits": _lookup_anchor_logits(
                                        anchor_cache,
                                        extra_u,
                                        expected_count=unlabeled_count,
                                        device=device,
                                    )
                                }
                            else:
                                out_anchor = teacher_model(
                                    x_u,
                                    y_tx=None,
                                    grl_lambda=0.0,
                                    return_aux=True,
                                    domain_labels=d_u,
                                )
                        if ema_model is None:
                            model.train()
                    x_s = (
                        _strong_augment(x_u, float(args.strong_noise_std))
                        if bool(muse_state.get("fasttrust_rc4", False))
                        else (
                            x_u if bool(muse_state.get("sat_anchor_ssl", False))
                            else _strong_augment(x_u, float(args.strong_noise_std))
                        )
                    )
                    u_sat_probability, _ = adv3b02_core90_u_satellite_policy(
                        int(epoch)
                    )
                    u_sat_scenario = select_adv3b02_u_satellite_scenario(
                        int(epoch), int(batch_idx), int(args.seed)
                    )
                    sat_anchor_route = None
                    rc4_route = None
                    sat_anchor_pair_active = False
                    if bool(muse_state.get("sat_anchor_ssl", False)):
                        if bool(muse_state.get("fasttrust_rc4", False)):
                            calibration = muse_state.get("rc4_calibration")
                            if calibration is None or out_w2 is None:
                                raise RuntimeError("RC4_VCAL_PACKAGE_MISSING")
                            anchor_logits = (
                                out_anchor["tx_logits"] if bool(args.rc4_use_anchor)
                                else out_w["tx_logits"]
                            )
                            rc4_route = route_fasttrust_rc4(
                                anchor_logits, out_w["tx_logits"], out_w2["tx_logits"],
                                domains=d_u, receivers=receiver_u,
                                z_norm=out_w["z_id"].float().norm(dim=-1),
                                calibration=calibration,
                                hard_max_fraction=float(args.sat_anchor_hard_max_fraction),
                                hard_effective_budget=float(args.rc4_hard_effective_budget),
                                candidate_max_classes=int(args.muse_candidate_max_classes),
                                partial_effective_budget=float(args.rc4_partial_effective_budget),
                                negative_effective_budget=float(args.rc4_negative_effective_budget),
                                total_identity_effective_budget=float(
                                    args.rc4_total_identity_effective_budget
                                ),
                                use_calibrated_partial_threshold=bool(
                                    args.rc4_use_calibrated_partial_threshold
                                ),
                                enable_hard=bool(args.rc4_enable_hard),
                                enable_partial=bool(args.rc4_enable_partial),
                                enable_negative=bool(args.rc4_enable_negative),
                                class_receiver_cap=bool(args.rc4_class_receiver_cap),
                                class_receiver_effective_budget=float(
                                    args.rc4_class_receiver_effective_budget
                                ),
                                use_calibrated_risk=bool(args.rc4_use_correctness_calibration),
                            )
                            sat_anchor_route = argparse.Namespace(
                                pseudo=rc4_route.pseudo,
                                confidence=rc4_route.risk,
                                margin=1.0 - rc4_route.disagreement,
                                agreement=rc4_route.agreement,
                                strict=rc4_route.hard,
                                trusted=rc4_route.hard,
                                filled=torch.zeros_like(rc4_route.hard),
                                no_identity=~rc4_route.hard,
                                class_cap=rc4_route.class_receiver_cap,
                                receiver_cap=rc4_route.class_receiver_cap,
                            )
                        else:
                            thresholds = muse_state.get("sat_anchor_thresholds")
                            if thresholds is None:
                                raise RuntimeError("SAT_ANCHOR_VCAL_THRESHOLDS_MISSING")
                            sat_anchor_route = route_sat_anchor_trusted(
                                out_anchor["tx_logits"].float().softmax(dim=-1),
                                out_w["tx_logits"].float().softmax(dim=-1),
                                confidence_thresholds=thresholds.confidence,
                                margin_thresholds=thresholds.margin,
                                receivers=receiver_u, beta=float(args.sat_anchor_beta),
                                hard_max_fraction=float(args.sat_anchor_hard_max_fraction),
                                fill_to_fraction=float(args.sat_anchor_fill_to_fraction),
                                class_balanced_cap=bool(args.muse_class_balanced_cap),
                                receiver_balanced_cap=bool(args.sat_anchor_receiver_balanced_cap),
                            )
                        sat_anchor_pair_active = (
                            not bool(muse_state.get("fasttrust_rc4", False))
                            and
                            float(args.sat_anchor_lambda_pair) > 0.0
                            and _sat_anchor_pair_step(
                                int(batch_idx), int(args.sat_anchor_pair_interval)
                            )
                        )
                        identity_active = (
                            int(epoch) >= int(args.rc4_identity_start_epoch)
                            if bool(muse_state.get("fasttrust_rc4", False))
                            else int(epoch) >= int(args.sat_anchor_identity_start_epoch)
                        )
                        satellite_mask = (
                            torch.ones(
                                unlabeled_count,
                                dtype=torch.bool,
                                device=x_u.device,
                            )
                            if sat_anchor_pair_active
                            else (
                                (
                                    (
                                        rc4_route.hard
                                        if bool(args.rc4_satellite_hard_only)
                                        else torch.zeros_like(rc4_route.hard)
                                    ) if bool(muse_state.get("fasttrust_rc4", False))
                                    else sat_anchor_route.trusted
                                ) if identity_active
                                else torch.zeros(
                                    unlabeled_count,
                                    dtype=torch.bool,
                                    device=x_u.device,
                                )
                            )
                        )
                        need_nuisance_view = bool(satellite_mask.any().item())
                    else:
                        need_nuisance_view = (
                            float(args.muse_lambda_nuisance) > 0.0
                            or bool(muse_state["capabilities"]["satellite"])
                        )
                    if need_nuisance_view:
                        x_u_nuisance, simulator_metadata = _build_muse_nuisance_iq(
                            x_u,
                            args,
                            generator=sat_gen,
                            scenario=u_sat_scenario,
                        )
                    else:
                        x_u_nuisance = None
                        simulator_metadata = {
                            "nuisance": None,
                            "nuisance_valid": None,
                            "scenario": u_sat_scenario,
                        }
                    if bool(muse_state.get("sat_anchor_ssl", False)):
                        sat_anchor_include_clean = (
                            bool(muse_state.get("fasttrust_rc4", False))
                            or sat_anchor_pair_active
                            or float(args.sat_anchor_lambda_clean_kl) > 0.0
                        )
                        student_views = _forward_sat_anchor_student_views(
                            model,
                            x_s,
                            x_u_nuisance,
                            d_u,
                            satellite_indices=satellite_mask.nonzero(
                                as_tuple=False
                            ).reshape(-1),
                            grl_lambda=0.0,
                            detach_backbone=(
                                str(args.sat_anchor_u_gradient_scope)
                                == "adapter_tail"
                            ),
                            include_clean=sat_anchor_include_clean,
                        )
                        if student_views["clean"] is None:
                            zero_link = next(
                                parameter
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ).sum() * 0.0
                            student_views["clean"] = {
                                "tx_logits": out_w["tx_logits"].detach() + zero_link,
                                "z_id": out_w["z_id"].detach() + zero_link,
                            }
                        out_s = student_views["clean"]
                        out_u_nuisance = student_views["satellite"]
                    else:
                        student_views = _forward_muse_student_views(
                            model,
                            x_s,
                            x_u_nuisance,
                            d_u,
                            grl_lambda=float(schedule.grl_lambda),
                            fused=bool(args.muse_fused_student_forward),
                        )
                        out_s = student_views["strong"]
                        out_u_nuisance = student_views["nuisance"]
                    meta_u = _meta_from_extra(extra_u) or {}
                    memory_keys = stable_sample_keys(meta_u)
                    if len(memory_keys) != unlabeled_count:
                        raise ValueError("MUSE metadata cardinality does not match the U_s batch")
                    if not bool(muse_state.get("sat_anchor_ssl", False)):
                        satellite_mask = select_satellite_student_mask(
                            memory_keys,
                            epoch=int(epoch),
                            probability=(
                                float(u_sat_probability)
                                if muse_state["capabilities"]["satellite"]
                                else 0.0
                            ),
                            seed=int(args.seed),
                        ).to(device=x_u.device)
                    out_u_sat = out_u_nuisance if bool(satellite_mask.any()) else None
                    if bool(muse_state.get("sat_anchor_ssl", False)):
                        if bool(muse_state.get("fasttrust_rc4", False)):
                            muse_losses = _compute_rc4_unlabeled_losses(
                                route=rc4_route, ema_outputs=out_w,
                                anchor_outputs=out_anchor,
                                student_views=student_views, domains=d_u,
                                model=model,
                                muse_state=muse_state, epoch=int(epoch),
                            )
                        else:
                            muse_losses = _compute_sat_anchor_unlabeled_losses(
                                route=sat_anchor_route, anchor_outputs=out_anchor,
                                student_views=student_views, muse_state=muse_state,
                                epoch=int(epoch), pair_active=sat_anchor_pair_active,
                            )
                    else:
                        muse_losses = _compute_muse_unlabeled_losses(
                            x_u=x_u,
                            metadata={
                                "domains": d_u,
                                "receivers": receiver_u,
                                "memory_keys": memory_keys,
                                "satellite_mask": satellite_mask,
                                "epoch": int(epoch),
                            },
                            teacher_outputs=out_w,
                            student_outputs={
                                "weak": out_w,
                                "strong": out_s,
                                "nuisance": out_u_nuisance,
                                "satellite": out_u_sat,
                            },
                            muse_state=muse_state,
                            simulator_metadata=simulator_metadata,
                        )
                    identity_forbidden_through = (
                        int(args.rc4_identity_start_epoch) - 1
                        if bool(muse_state.get("fasttrust_rc4", False)) else 16
                    )
                    if int(epoch) <= identity_forbidden_through:
                        if (
                            float(muse_losses["u_identity_selected_count"].detach().item()) != 0.0
                            or float(muse_losses["u_satellite_identity_selected_count"].detach().item()) != 0.0
                            or float(muse_losses["identity"].detach().item()) != 0.0
                        ):
                            raise RuntimeError(
                                "MUSE_S1_IDENTITY_GRADIENT_PROTOCOL_VIOLATION"
                            )
                    if batch_idx == 1:
                        identity_objective = (
                            muse_losses["identity"] + muse_losses["satellite"]
                        )
                        identity_grads = _autograd_grad_or_none(
                            identity_objective,
                            _optimizer_parameters(model, muse_state),
                            retain_graph=True,
                        )
                        identity_norm_sq = sum(
                            float(gradient.detach().float().square().sum().item())
                            for gradient in identity_grads
                            if gradient is not None
                        )
                        muse_identity_grad_norm = math.sqrt(identity_norm_sq)
                        if bool(muse_state.get("sat_anchor_ssl", False)):
                            parameters = _optimizer_parameters(model, muse_state)
                            component_gradients = {}
                            for component in ("self", "satellite", "clean_anchor"):
                                component_gradients[component] = _autograd_grad_or_none(
                                    muse_losses[component],
                                    parameters,
                                    retain_graph=True,
                                )

                            def component_norm(gradients):
                                return math.sqrt(
                                    sum(
                                        float(gradient.detach().float().square().sum().item())
                                        for gradient in gradients
                                        if gradient is not None
                                    )
                                )

                            sat_anchor_pair_grad_norm = component_norm(
                                component_gradients["self"]
                            )
                            sat_anchor_sat_grad_norm = component_norm(
                                component_gradients["satellite"]
                            )
                            sat_anchor_anchor_grad_norm = component_norm(
                                component_gradients["clean_anchor"]
                            )
                            dot = sum(
                                float(
                                    (left.detach().float() * right.detach().float())
                                    .sum()
                                    .item()
                                )
                                for left, right in zip(
                                    component_gradients["self"],
                                    component_gradients["satellite"],
                                )
                                if left is not None and right is not None
                            )
                            denominator = (
                                sat_anchor_pair_grad_norm * sat_anchor_sat_grad_norm
                            )
                            if denominator > 0.0:
                                sat_anchor_pair_sat_grad_cos = dot / denominator
                    mask = muse_losses["route"].high | muse_losses["route"].mid
                    pseudo_total = int(muse_losses["pseudo"].numel())
                    pseudo_selected = int(mask.sum().detach().item())
                    pseudo_correct = 0
                    pseudo_truth_available = 0.0
                    pseudo = muse_losses["pseudo"]
                    zero_u = out_s["tx_logits"].sum() * 0.0
                    loss_u = muse_losses["total"]
                    loss_ent = zero_u
                    loss_u_domain = zero_u
                    loss_u_adv = zero_u
                    loss_u_sat_cons = zero_u
                    loss_u_direct_metric = zero_u
                    loss_u_quarantine = zero_u
                    loss_u_zid_invariance = zero_u
                    u_sat_pair_count = int(satellite_mask.sum().detach().item())
                    u_dm_info = {
                        "active": 0.0,
                        "selected": 0.0,
                        "valid_domain_selected": 0.0,
                    }
                    u_quarantine_info = {
                        "active": 0.0,
                        "anchor_count": 0.0,
                        "query_count": 0.0,
                        "active_classes": 0.0,
                        "accept_rate": float("nan"),
                    }
                    u_zid_invariance_info = {
                        "active": 0.0,
                        "receiver_active": 0.0,
                        "day_active": 0.0,
                        "channel_active": 0.0,
                        "receiver_loss": 0.0,
                        "day_loss": 0.0,
                        "channel_loss": 0.0,
                    }
                    reliable_ratio = mask.float().mean()
                    pseudo_conf = muse_losses["reliability"].mean()
                    domain_mask = (
                        (d_u >= 0)
                        if torch.is_tensor(d_u)
                        else torch.zeros_like(mask)
                    )
                    temporal_mask = muse_losses["stable"]
                    strong_mask = mask
                    domain_pass, temporal_pass, strong_pass = _pseudo_gate_pass_rates(
                        domain_mask=domain_mask,
                        temporal_mask=temporal_mask,
                        strong_mask=strong_mask,
                    )
                elif legacy_unlabeled_active:
                    try:
                        unlabeled_batch = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(data_ctx["unlabeled_loader"])
                        unlabeled_batch = next(unlabeled_iter)
                    x_u, y_u, extra_u = move_batch(unlabeled_batch, device)
                    unlabeled_count = int(y_u.numel())
                    receiver_u = _metadata_label_tensor(extra_u, "rx_i", device, unlabeled_count)
                    day_u = _metadata_label_tensor(extra_u, "day_i", device, unlabeled_count)
                    d_u = domain_from_extra(extra_u, data_ctx["domain_label_map"], device)
                    pseudo_source = ema_model if ema_model is not None else model
                    with torch.no_grad():
                        pseudo_source.eval()
                        out_w = pseudo_source(x_u, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=d_u)
                        if ema_model is None:
                            model.train()
                        prob_w = out_w["tx_logits"].softmax(dim=1)
                        conf, pseudo = prob_w.max(dim=1)
                        conf_mask = _threshold_mask(conf, d_u, args)
                        if bool(args.pseudo_domain_gate):
                            if d_u is None or "dom_logits" not in out_w:
                                domain_mask = torch.zeros_like(conf_mask)
                            else:
                                domain_mask = out_w["dom_logits"].argmax(dim=1) == d_u
                        else:
                            domain_mask = torch.ones_like(conf_mask)
                        if bool(args.pseudo_temporal_gate):
                            if str(getattr(args, "pseudo_temporal_mode", "batch_neighbor")) == "epoch_bank":
                                temporal_mask = _temporal_bank_mask_tensor(
                                    pseudo,
                                    conf,
                                    extra_u,
                                    args,
                                    device,
                                    epoch=epoch,
                                    bank=pseudo_temporal_bank,
                                )
                            else:
                                temporal_mask = _temporal_mask_tensor(pseudo, conf, extra_u, args, device)
                        else:
                            temporal_mask = torch.ones_like(conf_mask)
                        base_mask = conf_mask & domain_mask & temporal_mask
                    x_s = _strong_augment(x_u, float(args.strong_noise_std))
                    out_s = model(x_s, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=d_u)
                    if bool(args.pseudo_strong_agreement):
                        strong_mask = out_s["tx_logits"].argmax(dim=1) == pseudo
                    else:
                        strong_mask = torch.ones_like(base_mask)
                    mask = base_mask & strong_mask
                    pseudo_total = int(pseudo.numel())
                    pseudo_selected = int(mask.sum().detach().item())
                    pseudo_correct = int(((pseudo == y_u) & mask).sum().detach().item())
                    pseudo_truth_available = 1.0
                    if bool(mask.any()):
                        loss_u = F.cross_entropy(out_s["tx_logits"][mask], pseudo[mask])
                    else:
                        loss_u = out_s["tx_logits"].sum() * 0.0
                    prob_s = out_s["tx_logits"].softmax(dim=1)
                    entropy_per_sample = -(prob_s * prob_s.clamp_min(1e-8).log()).sum(dim=1)
                    loss_ent = entropy_per_sample.mean()
                    zero_u = out_s["tx_logits"].sum() * 0.0
                    loss_u_domain = zero_u
                    loss_u_adv = zero_u
                    loss_u_sat_cons = zero_u
                    loss_u_direct_metric = zero_u
                    loss_u_quarantine = zero_u
                    loss_u_zid_invariance = zero_u
                    u_zid_invariance_info: Dict[str, float] = {
                        "active": 0.0,
                        "receiver_active": 0.0,
                        "day_active": 0.0,
                        "channel_active": 0.0,
                        "receiver_loss": 0.0,
                        "day_loss": 0.0,
                        "channel_loss": 0.0,
                    }
                    u_sat_pair_count = 0
                    u_dm_info: Dict[str, float] = {
                        "active": 0.0,
                        "active_classes": 0.0,
                        "inactive_reason_code": 1.0,
                        "selected": float(pseudo_selected),
                        "valid_domain_selected": float("nan"),
                        "zid_p50_deg": float("nan"),
                        "zid_p95_deg": float("nan"),
                        "zid_p99_deg": float("nan"),
                        "zid_tail_cvar_deg": float("nan"),
                        "source_overflow": float("nan"),
                        "proxy_vaccept": float("nan"),
                        "bridge_accept_rate": float("nan"),
                        "low_density_accept_rate": float("nan"),
                        "tail_accept_rate": float("nan"),
                        "overflow_accept_rate": float("nan"),
                        "radius_to_inter_ratio": float("nan"),
                        "sat_pair_angle_p95_deg": float("nan"),
                    }
                    u_quarantine_info: Dict[str, float] = {
                        "active": 0.0,
                        "anchor_count": 0.0,
                        "query_count": 0.0,
                        "active_classes": 0.0,
                        "accept_rate": float("nan"),
                        "accept_loss": 0.0,
                        "low_density_accept_rate": float("nan"),
                        "nearest_angle_p50_deg": float("nan"),
                        "nearest_angle_p95_deg": float("nan"),
                        "nearest_angle_p99_deg": float("nan"),
                        "radius_to_inter_ratio": float("nan"),
                        "quarantine_rate": float("nan"),
                        "valid_domain_rate": float("nan"),
                    }
                    valid_u_domain = d_u is not None and bool(torch.is_tensor(d_u)) and d_u.numel() == pseudo.numel()
                    if valid_u_domain:
                        valid_u_mask = d_u >= 0
                    else:
                        valid_u_mask = torch.zeros_like(pseudo, dtype=torch.bool)
                    if (
                        float(args.lambda_u_domain) > 0.0
                        and epoch >= int(args.u_domain_start_epoch)
                        and bool(valid_u_mask.any())
                        and "dom_logits" in out_s
                    ):
                        loss_u_domain = F.cross_entropy(out_s["dom_logits"][valid_u_mask].float(), d_u[valid_u_mask].long())
                    if (
                        float(args.lambda_u_adv) > 0.0
                        and epoch >= int(args.u_domain_start_epoch)
                        and bool(valid_u_mask.any())
                        and "adv_dom_logits" in out_s
                    ):
                        loss_u_adv = F.cross_entropy(out_s["adv_dom_logits"][valid_u_mask].float(), d_u[valid_u_mask].long())
                    out_u_sat = None
                    u_sat_applied = False
                    need_u_sat_view = bool(
                        concat_sat_aug is not None
                        and (
                            (float(args.lambda_u_sat_cons) > 0.0 and epoch >= int(args.u_sat_cons_start_epoch))
                            or (
                                float(args.lambda_u_direct_metric_accept) > 0.0
                                and bool(args.u_direct_metric_use_sat_pair)
                                and epoch >= int(args.u_direct_metric_start_epoch)
                            )
                            or (
                                float(args.lambda_u_quarantine_accept) > 0.0
                                and bool(args.u_quarantine_include_sat_view)
                                and epoch >= int(args.u_quarantine_start_epoch)
                            )
                            or (float(args.lambda_u_domain) > 0.0 and epoch >= int(args.u_domain_start_epoch))
                            or (float(args.lambda_u_adv) > 0.0 and epoch >= int(args.u_domain_start_epoch))
                        )
                    )
                    if need_u_sat_view:
                        u_sat_view = concat_sat_aug.transform(x_u, args=args, epoch=epoch, batch_idx=int(batch_idx) + 200000)
                        u_sat_applied = bool(u_sat_view.applied)
                        x_u_sat = _safe_iq_tensor(u_sat_view.x)
                        out_u_sat = model(x_u_sat, y_tx=None, grl_lambda=1.0, return_aux=True, domain_labels=d_u)
                        if bool(mask.any()):
                            loss_u = 0.5 * (
                                loss_u + F.cross_entropy(out_u_sat["tx_logits"][mask], pseudo[mask])
                            )
                        if float(args.lambda_u_sat_cons) > 0.0 and epoch >= int(args.u_sat_cons_start_epoch):
                            loss_u_sat_kl = F.kl_div(
                                F.log_softmax(out_u_sat["tx_logits"], dim=1),
                                prob_s.detach(),
                                reduction="batchmean",
                            )
                            loss_u_sat_zid = F.mse_loss(
                                F.normalize(out_u_sat["z_id"].float(), dim=1),
                                F.normalize(out_s["z_id"].detach().float(), dim=1),
                            )
                            loss_u_sat_cons = loss_u_sat_kl + float(args.u_sat_zid_cons_weight) * loss_u_sat_zid
                        if (
                            float(args.lambda_u_domain) > 0.0
                            and epoch >= int(args.u_domain_start_epoch)
                            and bool(valid_u_mask.any())
                            and "dom_logits" in out_u_sat
                        ):
                            loss_u_domain = 0.5 * (
                                loss_u_domain
                                + F.cross_entropy(out_u_sat["dom_logits"][valid_u_mask].float(), d_u[valid_u_mask].long())
                            )
                        if (
                            float(args.lambda_u_adv) > 0.0
                            and epoch >= int(args.u_domain_start_epoch)
                            and bool(valid_u_mask.any())
                            and "adv_dom_logits" in out_u_sat
                        ):
                            loss_u_adv = 0.5 * (
                                loss_u_adv
                                + F.cross_entropy(out_u_sat["adv_dom_logits"][valid_u_mask].float(), d_u[valid_u_mask].long())
                            )
                    strict_pseudo_mask = mask.clone()
                    u_geometry_core_mask = torch.ones_like(mask, dtype=torch.bool)
                    u_geometry_direct_mask = torch.ones_like(mask, dtype=torch.bool)
                    route_start_epochs = []
                    if float(args.lambda_u_quarantine_accept) > 0.0 or bool(
                        getattr(args, "u_tri_state_required", False)
                    ):
                        route_start_epochs.append(int(args.u_quarantine_start_epoch))
                    if float(args.lambda_u_direct_metric_accept) > 0.0:
                        route_start_epochs.append(int(args.u_direct_metric_start_epoch))
                    if any(
                        float(value) > 0.0
                        for value in (
                            args.lambda_u_zid_receiver_invariance,
                            args.lambda_u_zid_day_invariance,
                            args.lambda_u_zid_channel_invariance,
                        )
                    ):
                        route_start_epochs.append(int(args.u_domain_start_epoch))
                    route_requested = bool(route_start_epochs)
                    route_start_epoch = min(route_start_epochs) if route_start_epochs else 10**9
                    if (
                        route_requested
                        and epoch >= route_start_epoch
                        and unlabeled_known_acceptance_quarantine_loss is not None
                    ):
                        route_z_id_l = z_id_l
                        route_y_l = y_l
                        route_d_l = d_l
                        route_labeled_view_count = (
                            concat_sat_clean_bsz
                            if concat_sat_full_batch
                            else int(y_l.numel())
                        )
                        route_labeled_sat_applied = (
                            concat_sat_full_batch
                            and float(concat_sat_info.get("applied", 0.0)) > 0.0
                            and u_sat_applied
                        )
                        if (
                            bool(getattr(args, "u_route_use_reference_bank", False))
                            and dm_bank_clean is not None
                        ):
                            route_z_id_l = dm_bank_clean
                            route_y_l = dm_bank_clean_y
                            route_d_l = dm_bank_clean_d
                            route_labeled_view_count = int(dm_bank_clean_y.numel())
                            route_labeled_sat_applied = False
                            if (
                                bool(args.u_quarantine_include_sat_view)
                                and dm_bank_sat is not None
                            ):
                                route_z_id_l = torch.cat(
                                    [dm_bank_clean, dm_bank_sat], dim=0
                                )
                                route_y_l = torch.cat(
                                    [dm_bank_clean_y, dm_bank_clean_y], dim=0
                                )
                                if dm_bank_clean_d is not None:
                                    route_d_l = torch.cat(
                                        [dm_bank_clean_d, dm_bank_clean_d], dim=0
                                    )
                                route_labeled_sat_applied = True
                        route_out_s = out_s
                        if bool(getattr(args, "u_route_use_teacher_weak", False)):
                            route_out_s = {"z_id": out_w["z_id"].detach()}
                        (
                            loss_u_quarantine,
                            u_quarantine_info,
                            u_geometry_core_mask,
                            u_geometry_direct_mask,
                        ) = _route_unlabeled_known_geometry(
                            args=args,
                            z_id_l=route_z_id_l,
                            y_l=route_y_l,
                            d_l=route_d_l,
                            out_s=route_out_s,
                            out_u_sat=out_u_sat,
                            pseudo=pseudo,
                            d_u=d_u,
                            pseudo_mask=mask,
                            valid_u_mask=valid_u_mask,
                            labeled_view_count=route_labeled_view_count,
                            labeled_sat_applied=route_labeled_sat_applied,
                        )
                    u_outside_mask = u_quarantine_info.pop(
                        "_tri_outside_full_mask", torch.zeros_like(mask, dtype=torch.bool)
                    )
                    u_quarantine_info["route_teacher_weak"] = (
                        1.0 if bool(getattr(args, "u_route_use_teacher_weak", False)) else 0.0
                    )
                    u_quarantine_info["route_reference_bank"] = (
                        1.0
                        if bool(getattr(args, "u_route_use_reference_bank", False))
                        and dm_bank_clean is not None
                        else 0.0
                    )
                    routed_pseudo_mask, u_direct_geometry_mask, u_invariance_mask = _select_unlabeled_geometry_masks(
                        strict_pseudo_mask,
                        u_geometry_core_mask,
                        u_geometry_direct_mask,
                        valid_u_mask,
                        all_valid_queries=bool(getattr(args, "u_geometry_all_valid_queries", False)),
                        direct_valid_domain_only=bool(getattr(args, "u_direct_metric_valid_domain_only", True)),
                    )
                    if (
                        bool(getattr(args, "u_outside_stop_gradient", False))
                        and out_u_sat is not None
                        and float(args.lambda_u_sat_cons) > 0.0
                        and epoch >= int(args.u_sat_cons_start_epoch)
                    ):
                        pair_mask = ~u_outside_mask.bool()
                        if bool(pair_mask.any()):
                            loss_u_sat_kl = F.kl_div(
                                F.log_softmax(out_u_sat["tx_logits"][pair_mask], dim=1),
                                prob_s[pair_mask].detach(),
                                reduction="batchmean",
                            )
                            loss_u_sat_zid = F.mse_loss(
                                F.normalize(out_u_sat["z_id"][pair_mask].float(), dim=1),
                                F.normalize(out_s["z_id"][pair_mask].detach().float(), dim=1),
                            )
                            loss_u_sat_cons = (
                                loss_u_sat_kl
                                + float(args.u_sat_zid_cons_weight) * loss_u_sat_zid
                            )
                        else:
                            loss_u_sat_cons = zero_u
                    if bool(getattr(args, "u_geometry_all_valid_queries", False)):
                        if bool(routed_pseudo_mask.any()):
                            loss_u = F.cross_entropy(
                                out_s["tx_logits"][routed_pseudo_mask], pseudo[routed_pseudo_mask]
                            )
                            if out_u_sat is not None:
                                loss_u = 0.5 * (
                                    loss_u
                                    + F.cross_entropy(
                                        out_u_sat["tx_logits"][routed_pseudo_mask],
                                        pseudo[routed_pseudo_mask],
                                    )
                                )
                        else:
                            loss_u = out_s["tx_logits"].sum() * 0.0
                        pseudo_selected = int(routed_pseudo_mask.sum().detach().item())
                        pseudo_correct = int(
                            ((pseudo == y_u) & routed_pseudo_mask).sum().detach().item()
                        )
                        mask = routed_pseudo_mask
                        loss_ent = (
                            entropy_per_sample[u_geometry_core_mask].mean()
                            if bool(u_geometry_core_mask.any())
                            else entropy_per_sample.sum() * 0.0
                        )
                    if (
                        float(args.lambda_u_direct_metric_accept) > 0.0
                        and epoch >= int(args.u_direct_metric_start_epoch)
                        and direct_metric_acceptance_loss is not None
                    ):
                        dm_mask = u_direct_geometry_mask
                        dm_selected = int(dm_mask.sum().detach().item())
                        u_dm_info["selected"] = float(dm_selected)
                        u_dm_info["valid_domain_selected"] = float(int((dm_mask & valid_u_mask).sum().detach().item()))
                        if dm_selected >= int(args.u_direct_metric_min_selected):
                            dm_anchor_count = (
                                int(concat_sat_clean_bsz)
                                if concat_sat_full_batch
                                else int(y_l.numel())
                            )
                            dm_reference_clean = z_id_l[:dm_anchor_count].detach()
                            dm_reference_y = y_l[:dm_anchor_count].detach().long()
                            dm_reference_d = (
                                d_l[:dm_anchor_count].detach().long()
                                if d_l is not None
                                else None
                            )
                            dm_reference_sat = dm_reference_clean
                            if concat_sat_full_batch and int(z_id_l.size(0)) >= 2 * dm_anchor_count:
                                dm_reference_sat = z_id_l[
                                    dm_anchor_count : 2 * dm_anchor_count
                                ].detach()
                            if dm_bank_clean is not None:
                                dm_reference_clean = dm_bank_clean
                                dm_reference_y = dm_bank_clean_y
                                dm_reference_d = dm_bank_clean_d
                                dm_reference_sat = (
                                    dm_bank_sat
                                    if dm_bank_sat is not None
                                    else dm_bank_clean
                                )
                            dm_z_clean = out_s["z_id"][dm_mask]
                            dm_y = pseudo[dm_mask].long()
                            dm_d = d_u[dm_mask].long() if valid_u_domain else None
                            paired_view_count_u = 0
                            if (
                                bool(args.u_direct_metric_use_sat_pair)
                                and out_u_sat is not None
                                and bool(getattr(args, "direct_metric_multiview_separate", False))
                                and multiview_direct_metric_acceptance_loss is not None
                            ):
                                paired_view_count_u = int(dm_selected)
                                u_sat_pair_count = paired_view_count_u
                                loss_u_direct_metric, u_dm_info = multiview_direct_metric_acceptance_loss(
                                    dm_z_clean,
                                    out_u_sat["z_id"][dm_mask],
                                    dm_y,
                                    dm_d,
                                    clean_weight=float(args.direct_metric_clean_weight),
                                    sat_weight=float(args.direct_metric_sat_weight),
                                    pair_weight=float(args.direct_metric_sat_pair_weight),
                                    sat_pair_target_rad=math.radians(float(args.direct_metric_sat_pair_target_deg)),
                                    clean_reference_z=dm_reference_clean,
                                    sat_reference_z=dm_reference_sat,
                                    reference_y=dm_reference_y,
                                    reference_d=dm_reference_d,
                                    **_direct_metric_kwargs(args),
                                )
                            else:
                                dm_z = dm_z_clean
                                dm_reference_z = dm_reference_clean
                                dm_reference_labels = dm_reference_y
                                dm_reference_domains = dm_reference_d
                                if bool(args.u_direct_metric_use_sat_pair) and out_u_sat is not None:
                                    dm_z = torch.cat([dm_z_clean, out_u_sat["z_id"][dm_mask]], dim=0)
                                    dm_y = torch.cat([dm_y, dm_y], dim=0)
                                    dm_reference_z = torch.cat(
                                        [dm_reference_clean, dm_reference_sat], dim=0
                                    )
                                    dm_reference_labels = torch.cat(
                                        [dm_reference_y, dm_reference_y], dim=0
                                    )
                                    if dm_d is not None:
                                        dm_d = torch.cat([dm_d, dm_d], dim=0)
                                    if dm_reference_d is not None:
                                        dm_reference_domains = torch.cat(
                                            [dm_reference_d, dm_reference_d], dim=0
                                        )
                                    paired_view_count_u = int(dm_selected)
                                    u_sat_pair_count = paired_view_count_u
                                loss_u_direct_metric, u_dm_info = direct_metric_acceptance_loss(
                                    dm_z,
                                    dm_y,
                                    dm_d,
                                    paired_view_count=paired_view_count_u,
                                    sat_pair_target_rad=math.radians(float(args.direct_metric_sat_pair_target_deg)),
                                    sat_pair_weight=float(args.direct_metric_sat_pair_weight) if paired_view_count_u > 0 else 0.0,
                                    reference_z=dm_reference_z,
                                    reference_y=dm_reference_labels,
                                    reference_d=dm_reference_domains,
                                    **_direct_metric_kwargs(args),
                                )
                            u_dm_info["selected"] = float(dm_selected)
                            u_dm_info["valid_domain_selected"] = float(int((dm_mask & valid_u_mask).sum().detach().item()))
                            u_dm_info["inactive_reason_code"] = 0.0 if float(u_dm_info.get("active", 0.0)) > 0.0 else 3.0
                        else:
                            u_dm_info["inactive_reason_code"] = 2.0
                    if any(
                        float(value) > 0.0
                        for value in (
                            args.lambda_u_zid_receiver_invariance,
                            args.lambda_u_zid_day_invariance,
                            args.lambda_u_zid_channel_invariance,
                        )
                    ):
                        if tx_conditional_domain_invariance_loss is None:
                            raise ImportError("cvsrffi.losses.tx_conditional_domain_invariance_loss is required")
                        invariance_mask = u_invariance_mask
                        if bool(invariance_mask.any()):
                            inv_z = out_s["z_id"][invariance_mask]
                            inv_y = pseudo[invariance_mask].long()
                            inv_receiver = receiver_u[invariance_mask] if receiver_u is not None else None
                            inv_day = day_u[invariance_mask] if day_u is not None else None
                            inv_channel = torch.zeros(inv_y.numel(), device=device, dtype=torch.long)
                            if out_u_sat is not None and u_sat_applied:
                                inv_z = torch.cat([inv_z, out_u_sat["z_id"][invariance_mask]], dim=0)
                                inv_y = torch.cat([inv_y, inv_y], dim=0)
                                if inv_receiver is not None:
                                    inv_receiver = torch.cat([inv_receiver, inv_receiver], dim=0)
                                if inv_day is not None:
                                    inv_day = torch.cat([inv_day, inv_day], dim=0)
                                inv_channel = torch.cat(
                                    [
                                        inv_channel,
                                        torch.ones(inv_channel.numel(), device=device, dtype=torch.long),
                                    ],
                                    dim=0,
                                )
                            loss_u_zid_invariance, u_zid_invariance_info = tx_conditional_domain_invariance_loss(
                                inv_z,
                                inv_y,
                                receiver_labels=inv_receiver,
                                day_labels=inv_day,
                                channel_labels=inv_channel,
                                receiver_weight=float(args.lambda_u_zid_receiver_invariance),
                                day_weight=float(args.lambda_u_zid_day_invariance),
                                channel_weight=float(args.lambda_u_zid_channel_invariance),
                                channel_pair_weight=float(args.zid_channel_pair_weight),
                                paired_view_count=(int(invariance_mask.sum().item()) if u_sat_applied else 0),
                                min_groups=int(args.zid_invariance_min_groups),
                                min_samples_per_group=int(args.zid_invariance_min_samples_per_group),
                            )
                    reliable_ratio = mask.float().mean()
                    pseudo_conf = conf.mean()
                    domain_pass, temporal_pass, strong_pass = _pseudo_gate_pass_rates(
                        domain_mask=domain_mask,
                        temporal_mask=temporal_mask,
                        strong_mask=strong_mask,
                    )
                else:
                    z = out_l["tx_logits"].sum() * 0.0
                    loss_u = z
                    loss_ent = z
                    loss_u_domain = z
                    loss_u_adv = z
                    loss_u_sat_cons = z
                    loss_u_direct_metric = z
                    loss_u_quarantine = z
                    loss_u_zid_invariance = z
                    u_sat_pair_count = 0
                    u_dm_info = {
                        "active": 0.0,
                        "selected": 0.0,
                        "valid_domain_selected": 0.0,
                        "zid_p50_deg": float("nan"),
                        "zid_p95_deg": float("nan"),
                        "zid_p99_deg": float("nan"),
                        "zid_tail_cvar_deg": float("nan"),
                        "source_overflow": float("nan"),
                        "proxy_vaccept": float("nan"),
                        "bridge_accept_rate": float("nan"),
                        "low_density_accept_rate": float("nan"),
                        "tail_accept_rate": float("nan"),
                        "overflow_accept_rate": float("nan"),
                        "radius_to_inter_ratio": float("nan"),
                        "sat_pair_angle_p95_deg": float("nan"),
                    }
                    u_quarantine_info = {
                        "active": 0.0,
                        "anchor_count": 0.0,
                        "query_count": 0.0,
                        "active_classes": 0.0,
                        "accept_rate": float("nan"),
                        "accept_loss": 0.0,
                        "low_density_accept_rate": float("nan"),
                        "nearest_angle_p50_deg": float("nan"),
                        "nearest_angle_p95_deg": float("nan"),
                        "nearest_angle_p99_deg": float("nan"),
                        "radius_to_inter_ratio": float("nan"),
                        "quarantine_rate": float("nan"),
                        "valid_domain_rate": float("nan"),
                    }
                    u_zid_invariance_info = {
                        "active": 0.0,
                        "receiver_active": 0.0,
                        "day_active": 0.0,
                        "channel_active": 0.0,
                        "receiver_loss": 0.0,
                        "day_loss": 0.0,
                        "channel_loss": 0.0,
                    }
                    reliable_ratio = z.detach()
                    pseudo_conf = z.detach()
                    domain_pass = z.detach()
                    temporal_pass = z.detach()
                    strong_pass = z.detach()
                    pseudo_total = 0
                    pseudo_selected = 0
                    pseudo_correct = 0
                    pseudo_truth_available = 0.0
                loss_closed = _compose_unlabeled_closed_loss(
                    loss_closed_l,
                    args=args,
                    muse_state=muse_state,
                    muse_total=loss_u,
                    identity=loss_u,
                    entropy=loss_ent,
                    domain=sanitize_loss(
                        "ssdg_u_domain", loss_u_domain, z_id_l, loss_warn_counts
                    ),
                    adv=sanitize_loss(
                        "ssdg_u_adv", loss_u_adv, z_id_l, loss_warn_counts
                    ),
                    satellite=sanitize_loss(
                        "ssdg_u_sat_cons", loss_u_sat_cons, z_id_l, loss_warn_counts
                    ),
                )
                loss_open_u = (
                    sanitize_loss("ssdg_u_zid_domain_invariance", loss_u_zid_invariance, z_id_l, loss_warn_counts)
                    + float(args.lambda_u_direct_metric_accept)
                    * sanitize_loss("ssdg_u_direct_metric_accept", loss_u_direct_metric, z_id_l, loss_warn_counts)
                    + float(args.lambda_u_quarantine_accept)
                    * sanitize_loss("ssdg_u_quarantine_accept", loss_u_quarantine, z_id_l, loss_warn_counts)
                )
                loss_open = loss_open_l + loss_open_u
                open_objective_losses = {
                    "boundary": loss_open_boundary_l,
                    "source": loss_open_source_l,
                    "invariant": loss_open_invariant_l,
                    "u_geometry": loss_open_u,
                }
                effective_os_min_budget = (
                    float(args.os_eff_min_budget)
                    if float(dg_health_open_scale) >= 1.0 - 1e-8
                    else 0.0
                )
                control_os_min_budget = effective_os_min_budget
                if effective_os_min_budget > 0.0:
                    control_os_min_budget = effective_os_min_budget + max(
                        0.0, float(getattr(args, "os_budget_target_reserve", 0.0))
                    )
                    if float(args.os_eff_max_budget) > 0.0:
                        control_os_min_budget = min(
                            control_os_min_budget,
                            float(args.os_eff_max_budget) - 1e-6,
                        )
                os_budget_info = {
                    "active": 0.0,
                    "os_scale": 1.0,
                    "closed_scale": 1.0,
                    "pre_budget": 0.0,
                    "post_budget": 0.0,
                    "target_budget": float(control_os_min_budget),
                    "configured_target_budget": float(args.os_eff_min_budget),
                    "max_budget": float(args.os_eff_max_budget),
                    "reason_code": 0.0,
                }
                scaled_closed_loss = float(tail_closed_scale) * loss_closed
                scaled_open_loss = float(dg_health_open_scale) * loss_open
                loss = scaled_closed_loss + scaled_open_loss
            telemetry_epochs = {
                int(value)
                for value in str(getattr(args, "rc4_gradient_telemetry_epochs", "")).split(",")
                if str(value).strip()
            }
            if _rc4_should_collect_gradient_telemetry(
                muse_state,
                batch_idx=batch_idx,
                epoch=epoch,
                telemetry_epochs=telemetry_epochs,
            ):
                gradient_losses = muse_losses.get("rc4_gradient_losses", {})
                probe_parameters = _rc4_gradient_probe_parameters(model)
                relationships = _rc4_gradient_relationships(
                    {
                        "labeled": loss_tx_l,
                        "hard": gradient_losses.get("hard"),
                        "partial_set": gradient_losses.get("partial_set"),
                        "partial_conditional": gradient_losses.get("partial_conditional"),
                        "adv_u": gradient_losses.get("adv"),
                        "adv_l": float(cur_w["adv"]) * loss_adv_l,
                    },
                    probe_parameters,
                )
                gradient_norms = relationships["norms"]
                gradient_cosines = relationships["cosines_to_labeled"]
                g_l = gradient_norms.get("labeled", float("nan"))
                g_h = gradient_norms.get("hard", float("nan"))
                g_pset = gradient_norms.get("partial_set", float("nan"))
                g_pcond = gradient_norms.get("partial_conditional", float("nan"))
                g_adv_u = gradient_norms.get("adv_u", float("nan"))
                g_adv_l = gradient_norms.get("adv_l", float("nan"))
                identity_terms = [value for value in (g_h, g_pset, g_pcond) if math.isfinite(value)]
                adv_terms = [value for value in (g_adv_l, g_adv_u) if math.isfinite(value)]
                g_identity = math.sqrt(sum(value * value for value in identity_terms)) if identity_terms else float("nan")
                g_adv = math.sqrt(sum(value * value for value in adv_terms)) if adv_terms else float("nan")
                rc4_gradient_metrics = {
                    "rc4/g_L": g_l,
                    "rc4/g_H": g_h,
                    "rc4/g_Pset": g_pset,
                    "rc4/g_Pcond": g_pcond,
                    "rc4/g_adv": g_adv,
                    "rc4/g_identity_to_labeled": (
                        g_identity / max(g_l, 1e-12)
                        if math.isfinite(g_identity) and math.isfinite(g_l)
                        else float("nan")
                    ),
                    "rc4/g_adv_to_labeled": (
                        g_adv / max(g_l, 1e-12)
                        if math.isfinite(g_adv) and math.isfinite(g_l)
                        else float("nan")
                    ),
                    "rc4/cos_H_to_labeled": gradient_cosines.get("hard", float("nan")),
                    "rc4/cos_Pset_to_labeled": gradient_cosines.get("partial_set", float("nan")),
                    "rc4/cos_Pcond_to_labeled": gradient_cosines.get("partial_conditional", float("nan")),
                    "rc4/gradient_telemetry_active": 1.0,
                }
            loss_is_finite = bool(torch.isfinite(loss.detach()).item())
            skipped_nonfinite_loss = 0
            skipped_nonfinite_grad = 0
            optimizer_step_applied = False
            first_nonfinite_gradient = None
            os_grad_info = {
                "active": 0.0,
                "conflict": 0.0,
                "pre_cosine": float("nan"),
                "post_cosine": float("nan"),
                "closed_grad_norm": float("nan"),
                "open_grad_norm": float("nan"),
                "total_closed_grad_norm": float("nan"),
                "total_open_grad_norm": float("nan"),
                "shared_param_count": 0.0,
                "budget_scope_shared_trainable_params": 0.0,
                "budget_scope_shared_zid_path": 0.0,
                "balanced_closed_grad_norm": float("nan"),
                "balanced_open_grad_norm": float("nan"),
                "effective_closed_grad_norm": float("nan"),
                "effective_open_grad_norm": float("nan"),
                "os_scale": 1.0,
                "closed_scale": 1.0,
                "pre_budget": 0.0,
                "post_budget": 0.0,
                "reason_code": 0.0,
                "conflict_projection_priority_code": 0.0,
                "nonfinite_gradient_bundle": 0.0,
            }
            if loss_is_finite:
                os_control_epoch_ready = bool(getattr(args, "phase1_v2_os_eff_all_phases", True)) or (
                    epoch >= int(args.direct_metric_start_epoch)
                )
                open_loss_has_signal = bool(torch.isfinite(scaled_open_loss.detach()).item()) and (
                    abs(float(scaled_open_loss.detach().float().item())) > 1e-12
                )
                use_os_control = (
                    (
                        bool(getattr(args, "os_gradient_surgery", False))
                        or bool(getattr(args, "os_budget_controller", False))
                    )
                    and os_control_epoch_ready
                    and open_loss_has_signal
                    and (batch_idx % max(1, int(args.os_gradient_surgery_interval)) == 0)
                    and bool(scaled_closed_loss.requires_grad)
                    and bool(scaled_open_loss.requires_grad)
                )
                if use_os_control:
                    os_grad_info = _backward_with_open_set_projection(
                        model,
                        scaler,
                        scaled_closed_loss,
                        scaled_open_loss,
                        project_conflicts=bool(getattr(args, "os_gradient_surgery", False)),
                        budget_controller=bool(getattr(args, "os_budget_controller", False)),
                        min_budget=float(control_os_min_budget),
                        max_budget=float(args.os_eff_max_budget),
                        max_os_scale=float(args.os_budget_max_scale),
                        min_closed_scale=float(args.os_budget_min_closed_scale),
                        protect_closed_on_conflict=bool(getattr(args, "os_gradient_protect_closed", False)),
                        open_loss_groups=(
                            {
                                key: float(dg_health_open_scale) * value
                                for key, value in open_objective_losses.items()
                            }
                            if bool(getattr(args, "os_objective_budget_controller", False))
                            else None
                        ),
                        open_group_shares={
                            "boundary": float(getattr(args, "os_objective_boundary_share", 0.40)),
                            "source": float(getattr(args, "os_objective_source_share", 0.25)),
                            "invariant": float(getattr(args, "os_objective_invariant_share", 0.20)),
                            "u_geometry": float(getattr(args, "os_objective_u_share", 0.15)),
                        },
                        open_group_min_scale=float(getattr(args, "os_objective_min_scale", 0.25)),
                        open_group_max_scale=float(getattr(args, "os_objective_max_scale", 8.0)),
                        budget_param_filter=(
                            (lambda name: str(name).startswith("id_backbone."))
                            if str(getattr(args, "os_budget_scope", "all_shared")) == "zid_path"
                            else None
                        ),
                    )
                    os_budget_info = {
                        "active": 1.0 if float(os_grad_info["reason_code"]) in {1.0, 3.0} else 0.0,
                        "os_scale": float(os_grad_info["os_scale"]),
                        "closed_scale": float(os_grad_info["closed_scale"]),
                        "pre_budget": float(os_grad_info["pre_budget"]),
                        "post_budget": float(os_grad_info["post_budget"]),
                        "target_budget": float(effective_os_min_budget),
                        "configured_target_budget": float(args.os_eff_min_budget),
                        "max_budget": float(args.os_eff_max_budget),
                        "reason_code": float(os_grad_info["reason_code"]),
                    }
                    loss = (
                        float(os_grad_info["closed_scale"]) * scaled_closed_loss
                        + float(os_grad_info["os_scale"]) * scaled_open_loss
                    )
                else:
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                first_nonfinite_gradient = _first_nonfinite_gradient(model)
                if first_nonfinite_gradient is None and muse_state is not None:
                    first_nonfinite_gradient = _first_nonfinite_gradient(muse_state["heads"])
                    if first_nonfinite_gradient is not None:
                        first_nonfinite_gradient = dict(first_nonfinite_gradient)
                        first_nonfinite_gradient["parameter_name"] = (
                            "muse_heads." + str(first_nonfinite_gradient["parameter_name"])
                        )
                grad_norm_before_clip = _grad_norm(model)
                grads_finite = first_nonfinite_gradient is None
                if grads_finite and float(getattr(args, "max_grad_norm", 0.0)) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        _optimizer_parameters(model, muse_state),
                        max_norm=float(args.max_grad_norm),
                        error_if_nonfinite=False,
                    )
                grad_total = _grad_norm(model)
                grad_backbone = _grad_norm(model, lambda name: "backbone" in name)
                grad_aux = _grad_norm(model, lambda name: "aux" in name)
                grad_domain = _grad_norm(model, lambda name: "dom" in name or "domain" in name)
                if grads_finite:
                    scaler.step(optimizer)
                    optimizer_step_applied = True
                    if ema_model is not None:
                        ema_decay = (
                            float(muse_state["schedule_state"].ema_decay)
                            if muse_state is not None
                            else float(args.ema_decay)
                        )
                        _update_ema_model(ema_model, model, ema_decay)
                else:
                    skipped_nonfinite_grad = 1
                    optimizer.zero_grad(set_to_none=True)
                scaler.update()
            else:
                skipped_nonfinite_loss = 1
                optimizer.zero_grad(set_to_none=True)
                grad_norm_before_clip = float("nan")
                grad_total = float("nan")
                grad_backbone = float("nan")
                grad_aux = float("nan")
                grad_domain = float("nan")
            if (
                muse_state is not None
                and bool(muse_state.get("fasttrust_rc4", False))
                and not rc4_anomaly_written
                and (skipped_nonfinite_loss or skipped_nonfinite_grad)
            ):
                save_payload(
                    rc4_anomaly_path,
                    {
                        "schema": "cvs.phase1.fasttrust_qb3_first_anomaly.v1",
                        "run_id": str(getattr(args, "run_id", "")),
                        "candidate_id": str(getattr(args, "candidate_id", "")),
                        "epoch": int(epoch),
                        "batch_index": int(batch_idx),
                        "loss_finite": bool(loss_is_finite),
                        "skipped_nonfinite_loss": int(skipped_nonfinite_loss),
                        "skipped_nonfinite_grad": int(skipped_nonfinite_grad),
                        "loss": float(loss.detach().float().item()),
                        "grad_before_clip": float(grad_norm_before_clip),
                        "grad_total": float(grad_total),
                        "first_nonfinite_gradient": first_nonfinite_gradient,
                        "loss_components": {
                            "labeled_tx": loss_tx_l.detach().float().cpu(),
                            "labeled_domain": loss_dom_l.detach().float().cpu(),
                            "labeled_identity_domain": loss_adv_l.detach().float().cpu(),
                            "rc4_hard": muse_losses.get("hard", 0.0),
                            "rc4_partial_set": muse_losses.get("partial_set", 0.0),
                            "rc4_partial_conditional": muse_losses.get("partial_conditional", 0.0),
                            "rc4_domain_discriminator": muse_losses.get("domain_discriminator", 0.0),
                            "rc4_domain_confusion": muse_losses.get("domain_confusion", 0.0),
                        },
                        "optimizer_lrs": [float(group["lr"]) for group in optimizer.param_groups],
                        "model": model.state_dict(),
                        "ema_model": ema_model.state_dict() if ema_model is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "rng_state": _capture_training_rng_state(sat_gen),
                        "args": vars(args),
                        "rc4_calibration": muse_state.get("rc4_calibration"),
                        "rc4_route_counts": {
                            "hard": int(rc4_route.hard.sum().item()) if rc4_route is not None else 0,
                            "partial": int(rc4_route.partial.sum().item()) if rc4_route is not None else 0,
                            "negative": int(rc4_route.negative.sum().item()) if rc4_route is not None else 0,
                            "representation": int(rc4_route.representation.sum().item()) if rc4_route is not None else 0,
                        },
                    },
                )
                rc4_anomaly_written = True
            if proto_bank is not None and optimizer_step_applied:
                proto_bank.update(out_l["z_id"].detach(), y_l.detach(), d_l.detach() if d_l is not None else None)
                if proto_bank.class_count is not None:
                    active = proto_bank.class_count >= int(args.proto_min_count)
                    proto_info["proto_active_classes"] = float(int(active.sum().detach().item()))
            u_tri_query_count = float(u_quarantine_info.get("query_count", 0.0) or 0.0)
            q_tri_trusted_core_count = float(u_quarantine_info.get("tri_trusted_core_count", float("nan")))
            q_tri_ambiguous_tail_count = float(u_quarantine_info.get("tri_ambiguous_tail_count", float("nan")))
            q_tri_outside_reject_count = float(u_quarantine_info.get("tri_outside_reject_count", float("nan")))
            q_tri_counts = [
                q_tri_trusted_core_count,
                q_tri_ambiguous_tail_count,
                q_tri_outside_reject_count,
            ]
            has_geometry_tri_state = (
                all(math.isfinite(v) and v >= 0.0 for v in q_tri_counts)
                and sum(q_tri_counts) > 0.0
                and u_tri_query_count > 0.0
            )
            if has_geometry_tri_state:
                u_tri_state_source = "geometry"
                u_tri_trusted_core_count = q_tri_trusted_core_count
                u_tri_ambiguous_tail_count = q_tri_ambiguous_tail_count
                u_tri_outside_reject_count = q_tri_outside_reject_count
            else:
                u_tri_state_source = "fallback"
                u_tri_trusted_core_count = float(u_dm_info.get("selected", 0.0) or 0.0)
                u_tri_accept_rate = float(u_quarantine_info.get("accept_rate", 0.0) or 0.0)
                if not math.isfinite(u_tri_trusted_core_count):
                    u_tri_trusted_core_count = 0.0
                if not math.isfinite(u_tri_query_count):
                    u_tri_query_count = 0.0
                if not math.isfinite(u_tri_accept_rate):
                    u_tri_accept_rate = 0.0
                u_tri_ambiguous_tail_count = max(0.0, u_tri_query_count * max(0.0, min(1.0, u_tri_accept_rate)))
                u_tri_outside_reject_count = max(0.0, u_tri_query_count - u_tri_ambiguous_tail_count)
            epoch_logs.append(_detach_log_mapping(
                {
                    "train/loss": loss.detach(),
                    "train/loss_labeled": loss_l.detach(),
                    "train/loss_closed_group": loss_closed.detach(),
                    "train/loss_open_group": loss_open.detach(),
                    "train/os_budget_controller_active": float(os_budget_info["active"]),
                    "train/os_budget_controller_os_scale": float(os_budget_info["os_scale"]),
                    "train/os_budget_controller_closed_scale": float(os_budget_info["closed_scale"]),
                    "train/os_budget_controller_pre": float(os_budget_info["pre_budget"]),
                    "train/os_budget_controller_post": float(os_budget_info["post_budget"]),
                    "train/os_budget_controller_target": float(os_budget_info["target_budget"]),
                    "train/os_budget_controller_configured_target": float(
                        os_budget_info["configured_target_budget"]
                    ),
                    "train/os_budget_controller_max": float(os_budget_info["max_budget"]),
                    "train/os_budget_controller_reason_code": float(os_budget_info["reason_code"]),
                    "train/source_val_dg_health_open_scale": float(dg_health_open_scale),
                    "train/tail_rollback_cooldown_active": 1.0 if tail_rollback_cooldown_active else 0.0,
                    "train/tail_rollback_cooldown_remaining": float(stage_state["tail_rollback_cooldown_remaining"]),
                    "train/tail_rollback_closed_scale": float(tail_closed_scale),
                    "train/os_gradient_surgery_active": float(os_grad_info["active"]),
                    "train/os_gradient_conflict": float(os_grad_info["conflict"]),
                    "train/os_gradient_pre_cosine": float(os_grad_info["pre_cosine"]),
                    "train/os_gradient_post_cosine": float(os_grad_info["post_cosine"]),
                    "train/os_gradient_conflict_projection_priority_code": float(
                        os_grad_info["conflict_projection_priority_code"]
                    ),
                    "train/os_gradient_closed_norm": float(os_grad_info["closed_grad_norm"]),
                    "train/os_gradient_open_norm": float(os_grad_info["open_grad_norm"]),
                    "train/os_gradient_total_closed_norm": float(os_grad_info["total_closed_grad_norm"]),
                    "train/os_gradient_total_open_norm": float(os_grad_info["total_open_grad_norm"]),
                    "train/os_gradient_shared_param_count": float(os_grad_info["shared_param_count"]),
                    "train/os_gradient_budget_scope_shared_trainable_params": float(
                        os_grad_info.get("budget_scope_shared_trainable_params", 0.0)
                    ),
                    "train/os_gradient_budget_scope_shared_zid_path": float(
                        os_grad_info["budget_scope_shared_zid_path"]
                    ),
                    "train/os_gradient_nonfinite_bundle": float(
                        os_grad_info.get("nonfinite_gradient_bundle", 0.0)
                    ),
                    "train/os_gradient_balanced_closed_norm": float(os_grad_info["balanced_closed_grad_norm"]),
                    "train/os_gradient_balanced_open_norm": float(os_grad_info["balanced_open_grad_norm"]),
                    "train/os_gradient_effective_open_norm": float(os_grad_info["effective_open_grad_norm"]),
                    "train/os_gradient_effective_closed_norm": float(os_grad_info["effective_closed_grad_norm"]),
                    "train/os_objective_boundary_raw_norm": float(
                        os_grad_info.get("objective_boundary_raw_norm", float("nan"))
                    ),
                    "train/os_objective_boundary_scale": float(
                        os_grad_info.get("objective_boundary_scale", float("nan"))
                    ),
                    "train/os_objective_boundary_effective_norm": float(
                        os_grad_info.get("objective_boundary_effective_norm", float("nan"))
                    ),
                    "train/os_objective_source_raw_norm": float(
                        os_grad_info.get("objective_source_raw_norm", float("nan"))
                    ),
                    "train/os_objective_source_scale": float(
                        os_grad_info.get("objective_source_scale", float("nan"))
                    ),
                    "train/os_objective_source_effective_norm": float(
                        os_grad_info.get("objective_source_effective_norm", float("nan"))
                    ),
                    "train/os_objective_invariant_raw_norm": float(
                        os_grad_info.get("objective_invariant_raw_norm", float("nan"))
                    ),
                    "train/os_objective_invariant_scale": float(
                        os_grad_info.get("objective_invariant_scale", float("nan"))
                    ),
                    "train/os_objective_invariant_effective_norm": float(
                        os_grad_info.get("objective_invariant_effective_norm", float("nan"))
                    ),
                    "train/os_objective_u_geometry_raw_norm": float(
                        os_grad_info.get("objective_u_geometry_raw_norm", float("nan"))
                    ),
                    "train/os_objective_u_geometry_scale": float(
                        os_grad_info.get("objective_u_geometry_scale", float("nan"))
                    ),
                    "train/os_objective_u_geometry_effective_norm": float(
                        os_grad_info.get("objective_u_geometry_effective_norm", float("nan"))
                    ),
                    "train/loss_tx_labeled": loss_tx_l.detach(),
                    "train/loss_muse_local_labeled": loss_muse_local_l.detach(),
                    "train/loss_domain_labeled": loss_dom_l.detach(),
                    "train/loss_adv_labeled": loss_adv_l.detach(),
                    "train/loss_adv_discriminator_labeled": loss_adv_discriminator_l.detach(),
                    "train/loss_adv_confusion_labeled": loss_adv_confusion_l.detach(),
                    "train/loss_cons_labeled": loss_cons_l.detach(),
                    "train/loss_orth_labeled": loss_orth_l.detach(),
                    "train/loss_group_ce_labeled": loss_group_ce_l.detach(),
                    "train/loss_fishr_labeled": loss_fishr_l.detach(),
                    "train/loss_zid_domain_invariance": loss_zid_invariance_l.detach(),
                    "train/zid_invariance_active": zid_invariance_info.get("active", 0.0),
                    "train/zid_receiver_invariance_active": zid_invariance_info.get("receiver_active", 0.0),
                    "train/zid_day_invariance_active": zid_invariance_info.get("day_active", 0.0),
                    "train/zid_channel_invariance_active": zid_invariance_info.get("channel_active", 0.0),
                    "train/zid_receiver_invariance_loss": zid_invariance_info.get("receiver_loss", 0.0),
                    "train/zid_day_invariance_loss": zid_invariance_info.get("day_loss", 0.0),
                    "train/zid_channel_invariance_loss": zid_invariance_info.get("channel_loss", 0.0),
                    "train/zid_channel_pair_loss": zid_invariance_info.get("channel_pair_loss", 0.0),
                    "train/zid_channel_pair_count": zid_invariance_info.get("channel_pair_count", 0.0),
                    "train/zid_channel_pair_angle_deg": zid_invariance_info.get(
                        "channel_pair_angle_deg", float("nan")
                    ),
                    "train/zid_receiver_center_angle_deg": zid_invariance_info.get(
                        "receiver_mean_center_angle_deg", float("nan")
                    ),
                    "train/zid_day_center_angle_deg": zid_invariance_info.get(
                        "day_mean_center_angle_deg", float("nan")
                    ),
                    "train/zid_channel_center_angle_deg": zid_invariance_info.get(
                        "channel_mean_center_angle_deg", float("nan")
                    ),
                    "train/loss_proto_labeled": loss_proto_l.detach(),
                    "train/loss_open_world_feat": loss_open_world_feat_l.detach(),
                    "train/loss_zid_compact": loss_zid_compact_l.detach(),
                    "train/loss_proxy_unknown": loss_proxy_unknown_l.detach(),
                    "train/loss_soft_unknown_mixup": loss_soft_unknown_mixup_l.detach(),
                    "train/loss_source_episode": loss_source_episode_l.detach(),
                    "train/loss_direct_metric_accept": loss_direct_metric_accept_l.detach(),
                    "train/loss_sat_cls_labeled": loss_sat_cls_l.detach(),
                    "train/loss_sat_cons_labeled": loss_sat_cons_l.detach(),
                    "train/loss_crra_pair_labeled": loss_crra_pair_l.detach(),
                    "train/loss_crra_energy_labeled": loss_crra_energy_l.detach(),
                    "train/loss_crra_gate_l1_labeled": loss_crra_gate_l1_l.detach(),
                    "train/loss_crra_nuisance_labeled": loss_crra_nuisance_l.detach(),
                    "train/loss_crra_sat_shell_labeled": loss_crra_sat_shell_l.detach(),
                    "train/loss_crra_condition_tx_adv_labeled": loss_crra_condition_tx_adv_l.detach(),
                    "train/crra_pair_cosine_labeled": crra_pair_cosine_l.detach(),
                    "train/crra_condition_tx_adv_accuracy": crra_condition_tx_adv_acc_l.detach(),
                    "train/crra_stage_scale": float(crra_stage_scale),
                    "train/crra_correction_energy": out_l.get(
                        "crra_correction_energy", zero_sat
                    ).detach().float().mean(),
                    "train/crra_gate": out_l.get("crra_gate", zero_sat).detach().float().mean(),
                    "train/crra_alpha": out_l.get("crra_alpha", zero_sat).detach().float().mean(),
                    "train/crra_support_distance": out_l.get(
                        "crra_support_distance", zero_sat
                    ).detach().float().mean(),
                    "train/crra_nuisance_valid_count": float(crra_nuisance_info.get("valid_count", 0.0)),
                    "train/crra_sat_shell_valid_count": float(crra_sat_shell_info.get("valid_count", 0.0)),
                    "train/crra_sat_shell_active_classes": float(crra_sat_shell_info.get("active_classes", 0.0)),
                    "train/crra_sat_shell_mean_excess_rad": float(
                        crra_sat_shell_info.get("mean_excess_rad", 0.0)
                    ),
                    "train/loss_teacher_clean_kl": loss_teacher_clean_kl_l.detach(),
                    "train/loss_teacher_sat_kl": loss_teacher_sat_kl_l.detach(),
                    "train/loss_teacher_zid_mse": loss_teacher_zid_mse_l.detach(),
                    "train/w_loss_tx_labeled": loss_tx_l.detach(),
                    "train/w_loss_domain_labeled": (cur_w["dom"] * loss_dom_l).detach(),
                    "train/w_loss_adv_labeled": (cur_w["adv"] * loss_adv_l).detach(),
                    "train/w_loss_cons_labeled": (cur_w["cons"] * loss_cons_l).detach(),
                    "train/w_loss_orth_labeled": (cur_w["orth"] * loss_orth_l).detach(),
                    "train/w_loss_group_ce_labeled": (cur_w["group_ce"] * loss_group_ce_l).detach(),
                    "train/w_loss_fishr_labeled": (cur_w["fishr"] * loss_fishr_l).detach(),
                    "train/w_loss_proto_labeled": (cur_w["proto"] * loss_proto_l).detach(),
                    "train/w_loss_open_world_feat": ((cur_w["open_world_feat"] * ow_feat_stage_scale) * loss_open_world_feat_l).detach(),
                    "train/w_loss_zid_compact": ((cur_w["zid_compact"] * zid_warm) * loss_zid_compact_l).detach(),
                    "train/w_loss_proxy_unknown": ((cur_w["proxy_unknown"] * proxy_stage_scale) * loss_proxy_unknown_l).detach(),
                    "train/w_loss_soft_unknown_mixup": ((cur_w["soft_unknown_mixup"] * soft_unknown_mixup_stage_scale) * loss_soft_unknown_mixup_l).detach(),
                    "train/w_loss_source_episode": ((cur_w["source_episode"] * source_episode_stage_scale) * loss_source_episode_l).detach(),
                    "train/w_loss_direct_metric_accept": ((cur_w["direct_metric_accept"] * direct_metric_stage_scale) * loss_direct_metric_accept_l).detach(),
                    "train/w_loss_sat_cls_labeled": (cur_w["sat_cls"] * loss_sat_cls_l).detach(),
                    "train/w_loss_sat_cons_labeled": (cur_w["sat_cons"] * loss_sat_cons_l).detach(),
                    "train/w_loss_crra_pair_labeled": (
                        crra_stage_scale * float(getattr(args, "lambda_crra_pair", 0.0)) * loss_crra_pair_l
                    ).detach(),
                    "train/w_loss_crra_sat_kl_labeled": (
                        crra_stage_scale * float(getattr(args, "lambda_crra_sat_kl", 0.0)) * loss_sat_cons_l
                    ).detach(),
                    "train/w_loss_crra_energy_labeled": (
                        crra_stage_scale * float(getattr(args, "lambda_crra_energy", 0.0)) * loss_crra_energy_l
                    ).detach(),
                    "train/w_loss_crra_gate_l1_labeled": (
                        crra_stage_scale * float(getattr(args, "lambda_crra_gate_l1", 0.0)) * loss_crra_gate_l1_l
                    ).detach(),
                    "train/w_loss_crra_sat_shell_labeled": (
                        crra_stage_scale
                        * float(getattr(args, "lambda_crra_sat_shell", 0.0))
                        * loss_crra_sat_shell_l
                    ).detach(),
                    "train/w_loss_crra_nuisance_labeled": (
                        crra_stage_scale * float(getattr(args, "lambda_crra_nuisance", 0.0)) * loss_crra_nuisance_l
                    ).detach(),
                    "train/w_loss_crra_condition_tx_adv_labeled": (
                        crra_stage_scale
                        * float(getattr(args, "lambda_crra_condition_tx_adv", 0.0))
                        * loss_crra_condition_tx_adv_l
                    ).detach(),
                    "train/w_loss_teacher_clean_kl": ((float(args.lambda_teacher_clean_kl) * teacher_scale) * loss_teacher_clean_kl_l).detach(),
                    "train/w_loss_teacher_sat_kl": ((float(args.lambda_teacher_sat_kl) * teacher_scale) * loss_teacher_sat_kl_l).detach(),
                    "train/w_loss_teacher_zid_mse": ((float(args.lambda_teacher_zid_mse) * teacher_scale) * loss_teacher_zid_mse_l).detach(),
                    "train/teacher_distill_scale": float(teacher_scale),
                    "train/concat_sat_active": float(concat_sat_info.get("active", 0.0)),
                    "train/concat_sat_expanded": float(concat_sat_info.get("expanded", 0.0)),
                    "train/concat_sat_applied": float(concat_sat_info.get("applied", 0.0)),
                    "train/concat_sat_clean_batch_size": float(concat_sat_info.get("clean_batch_size", 0.0)),
                    "train/concat_sat_total_batch_size": float(concat_sat_info.get("total_batch_size", 0.0)),
                    "train/concat_sat_view_prob": float(concat_sat_info.get("view_prob", 0.0)),
                    "train/concat_sat_stage_start_epoch": float(concat_sat_info.get("stage_start_epoch", float("nan"))),
                    "train/concat_sat_stage_index": float(concat_sat_info.get("stage_index", float("nan"))),
                    "train/loss_unlabeled": loss_u.detach(),
                    "train/loss_u_domain": loss_u_domain.detach(),
                    "train/loss_u_adv": loss_u_adv.detach(),
                    "train/loss_u_sat_cons": loss_u_sat_cons.detach(),
                    "train/loss_u_direct_metric_accept": loss_u_direct_metric.detach(),
                    "train/loss_u_quarantine_accept": loss_u_quarantine.detach(),
                    "train/loss_u_zid_domain_invariance": loss_u_zid_invariance.detach(),
                    "train/u_zid_invariance_active": u_zid_invariance_info.get("active", 0.0),
                    "train/u_zid_receiver_invariance_active": u_zid_invariance_info.get("receiver_active", 0.0),
                    "train/u_zid_day_invariance_active": u_zid_invariance_info.get("day_active", 0.0),
                    "train/u_zid_channel_invariance_active": u_zid_invariance_info.get("channel_active", 0.0),
                    "train/u_zid_receiver_invariance_loss": u_zid_invariance_info.get("receiver_loss", 0.0),
                    "train/u_zid_day_invariance_loss": u_zid_invariance_info.get("day_loss", 0.0),
                    "train/u_zid_channel_invariance_loss": u_zid_invariance_info.get("channel_loss", 0.0),
                    "train/u_zid_channel_pair_loss": u_zid_invariance_info.get("channel_pair_loss", 0.0),
                    "train/u_zid_channel_pair_count": u_zid_invariance_info.get("channel_pair_count", 0.0),
                    "train/w_loss_u_domain": (float(args.lambda_u_domain) * loss_u_domain).detach(),
                    "train/w_loss_u_adv": (float(args.lambda_u_adv) * loss_u_adv).detach(),
                    "train/w_loss_u_sat_cons": (float(args.lambda_u_sat_cons) * loss_u_sat_cons).detach(),
                    "train/w_loss_u_direct_metric_accept": (
                        float(args.lambda_u_direct_metric_accept) * loss_u_direct_metric
                    ).detach(),
                    "train/w_loss_u_quarantine_accept": (
                        float(args.lambda_u_quarantine_accept) * loss_u_quarantine
                    ).detach(),
                    "train/u_dm_accept_active": u_dm_info.get("active", float("nan")),
                    "train/u_dm_accept_active_classes": u_dm_info.get("active_classes", float("nan")),
                    "train/u_dm_accept_inactive_reason_code": u_dm_info.get("inactive_reason_code", float("nan")),
                    "train/u_dm_accept_selected": u_dm_info.get("selected", float("nan")),
                    "train/u_dm_accept_valid_domain_selected": u_dm_info.get("valid_domain_selected", float("nan")),
                    "train/u_dm_accept_sat_pair_count": float(u_sat_pair_count),
                    "train/u_dm_accept_zid_p50_deg": u_dm_info.get("zid_p50_deg", float("nan")),
                    "train/u_dm_accept_zid_p95_deg": u_dm_info.get("zid_p95_deg", float("nan")),
                    "train/u_dm_accept_zid_p99_deg": u_dm_info.get("zid_p99_deg", float("nan")),
                    "train/u_dm_accept_zid_tail_cvar_deg": u_dm_info.get("zid_tail_cvar_deg", float("nan")),
                    "train/u_dm_accept_source_overflow": u_dm_info.get("source_overflow", float("nan")),
                    "train/u_dm_accept_proxy_vaccept": u_dm_info.get("proxy_vaccept", float("nan")),
                    "train/u_dm_accept_bridge_accept_rate": u_dm_info.get("bridge_accept_rate", float("nan")),
                    "train/u_dm_accept_low_density_accept_rate": u_dm_info.get("low_density_accept_rate", float("nan")),
                    "train/u_dm_accept_tail_accept_rate": u_dm_info.get("tail_accept_rate", float("nan")),
                    "train/u_dm_accept_overflow_accept_rate": u_dm_info.get("overflow_accept_rate", float("nan")),
                    "train/u_dm_accept_radius_to_inter_ratio": u_dm_info.get("radius_to_inter_ratio", float("nan")),
                    "train/u_dm_accept_component_min_inter_deg": u_dm_info.get(
                        "component_min_inter_deg", float("nan")
                    ),
                    "train/u_dm_accept_component_inter_margin_loss": u_dm_info.get(
                        "component_inter_margin_loss", float("nan")
                    ),
                    "train/u_dm_accept_component_overlap_loss": u_dm_info.get(
                        "component_overlap_loss", float("nan")
                    ),
                    "train/u_dm_accept_global_zid_quantile_loss": u_dm_info.get(
                        "global_zid_quantile_loss", float("nan")
                    ),
                    "train/u_dm_accept_reference_anchor_count": u_dm_info.get(
                        "reference_anchor_count", float("nan")
                    ),
                    "train/u_dm_accept_query_count": u_dm_info.get("query_count", float("nan")),
                    "train/u_dm_accept_sat_pair_angle_p95_deg": u_dm_info.get("sat_pair_angle_p95_deg", float("nan")),
                    "train/u_quarantine_active": u_quarantine_info.get("active", float("nan")),
                    "train/u_quarantine_anchor_count": u_quarantine_info.get("anchor_count", float("nan")),
                    "train/u_quarantine_query_count": u_quarantine_info.get("query_count", float("nan")),
                "train/u_quarantine_active_classes": u_quarantine_info.get("active_classes", float("nan")),
                "train/u_quarantine_local_component_count": u_quarantine_info.get("local_component_count", float("nan")),
                    "train/u_quarantine_multiview_local_components": u_quarantine_info.get(
                        "multiview_local_components", 0.0
                    ),
                    "train/u_quarantine_global_component_fallback": u_quarantine_info.get(
                        "global_component_fallback", 1.0
                    ),
                    "train/u_quarantine_route_teacher_weak": u_quarantine_info.get(
                        "route_teacher_weak", 0.0
                    ),
                    "train/u_quarantine_route_reference_bank": u_quarantine_info.get(
                        "route_reference_bank", 0.0
                    ),
                    "train/u_quarantine_accept_rate": u_quarantine_info.get("accept_rate", float("nan")),
                    "train/u_quarantine_accept_loss": u_quarantine_info.get("accept_loss", float("nan")),
                    "train/u_quarantine_core_keep_loss": u_quarantine_info.get("core_keep_loss", float("nan")),
                    "train/u_quarantine_tail_quarantine_loss": u_quarantine_info.get("tail_quarantine_loss", float("nan")),
                    "train/u_quarantine_outside_reject_loss": u_quarantine_info.get("outside_reject_loss", float("nan")),
                    "train/u_quarantine_low_density_accept_rate": u_quarantine_info.get("low_density_accept_rate", float("nan")),
                    "train/u_quarantine_nearest_angle_p50_deg": u_quarantine_info.get("nearest_angle_p50_deg", float("nan")),
                    "train/u_quarantine_nearest_angle_p95_deg": u_quarantine_info.get("nearest_angle_p95_deg", float("nan")),
                    "train/u_quarantine_nearest_angle_p99_deg": u_quarantine_info.get("nearest_angle_p99_deg", float("nan")),
                    "train/u_quarantine_radius_to_inter_ratio": u_quarantine_info.get("radius_to_inter_ratio", float("nan")),
                    "train/u_quarantine_rate": u_quarantine_info.get("quarantine_rate", float("nan")),
                    "train/u_quarantine_valid_domain_rate": u_quarantine_info.get("valid_domain_rate", float("nan")),
                    "train/u_quarantine_outside_known_negative_disabled": u_quarantine_info.get(
                        "outside_known_negative_disabled", float("nan")
                    ),
                    "train/u_tri_state_source_code": 1.0 if u_tri_state_source == "geometry" else 0.0,
                    "train/u_tri_state_geometry": 1.0 if u_tri_state_source == "geometry" else 0.0,
                    "train/u_tri_query_count": u_tri_query_count,
                    "train/u_tri_trusted_core_count": u_tri_trusted_core_count,
                    "train/u_tri_ambiguous_tail_count": u_tri_ambiguous_tail_count,
                    "train/u_tri_outside_reject_count": u_tri_outside_reject_count,
                    "train/u_tri_class_coverage": u_quarantine_info.get("tri_class_coverage", float("nan")),
                    "train/u_tri_domain_coverage": u_quarantine_info.get("tri_domain_coverage", float("nan")),
                    "train/u_tri_pair_disagreement_rate": u_quarantine_info.get(
                        "tri_pair_disagreement_rate", float("nan")
                    ),
                    "train/u_tri_direct_count": u_quarantine_info.get("tri_direct_count", float("nan")),
                    "train/u_tri_direct_rate": u_quarantine_info.get("tri_direct_rate", float("nan")),
                    "train/u_tri_direct_eligible_count": u_quarantine_info.get(
                        "tri_direct_eligible_count", float("nan")
                    ),
                    "train/u_tri_direct_eligible_rate": u_quarantine_info.get(
                        "tri_direct_eligible_rate", float("nan")
                    ),
                    "train/u_tri_quota_routing": u_quarantine_info.get(
                        "tri_quota_routing", 0.0
                    ),
                    "train/u_tri_route_core_rate": u_quarantine_info.get(
                        "tri_route_core_rate", float("nan")
                    ),
                    "train/u_tri_route_ambiguous_rate": u_quarantine_info.get(
                        "tri_route_ambiguous_rate", float("nan")
                    ),
                    "train/u_tri_route_outside_rate": u_quarantine_info.get(
                        "tri_route_outside_rate", float("nan")
                    ),
                    "train/u_tri_route_accept_score_mean": u_quarantine_info.get(
                        "tri_route_accept_score_mean", float("nan")
                    ),
                    "train/u_tri_route_label_match_rate": u_quarantine_info.get(
                        "tri_route_label_match_rate", float("nan")
                    ),
                    "train/u_tri_tail_pair_loss": u_quarantine_info.get("tri_tail_pair_loss", float("nan")),
                    "train/u_tri_outside_pair_loss": u_quarantine_info.get(
                        "tri_outside_pair_loss", float("nan")
                    ),
                    "train/u_tri_pseudo_component_agreement_rate": u_quarantine_info.get(
                        "tri_pseudo_component_agreement_rate", float("nan")
                    ),
                    "train/tx_acc": 100.0 * (out_l["tx_logits"].argmax(dim=1) == y_l).float().mean().detach(),
                    "train/dom_acc": core_losses.get("dom_acc", float("nan")),
                    "train/cons_cos": core_losses.get("cons_cos", float("nan")),
                    "train/grad_before_clip": grad_norm_before_clip,
                    "train/grad_clip_active": 1.0 if float(getattr(args, "max_grad_norm", 0.0)) > 0.0 else 0.0,
                    "train/grad_clip_limit": float(getattr(args, "max_grad_norm", 0.0)),
                    "train/grad_total": grad_total,
                    "train/grad_backbone": grad_backbone,
                    "train/grad_aux": grad_aux,
                    "train/grad_domain": grad_domain,
                    "train/skipped_nonfinite_loss": skipped_nonfinite_loss,
                    "train/skipped_nonfinite_grad": skipped_nonfinite_grad,
                    "train/optimizer_step_applied": 1.0 if optimizer_step_applied else 0.0,
                    "train/reliable_ratio": reliable_ratio.detach(),
                    "train/pseudo_conf": pseudo_conf.detach(),
                    "train/domain_pass": domain_pass.detach(),
                    "train/temporal_pass": temporal_pass.detach(),
                    "train/strong_pass": strong_pass.detach(),
                    "train/pseudo_total": pseudo_total,
                    "train/pseudo_selected": pseudo_selected,
                    "train/pseudo_correct": pseudo_correct,
                    "train/pseudo_truth_available": pseudo_truth_available,
                    "muse/high_ratio": muse_losses["route"].high.float().mean().detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/mid_ratio": muse_losses["route"].mid.float().mean().detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/low_ratio": muse_losses["route"].low.float().mean().detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/effective_weight": (
                        float(muse_state["schedule_state"].lambda_u)
                        * (
                            float(args.muse_lambda_high)
                            * muse_losses["route"].high.float().mean()
                            + float(args.muse_lambda_mid)
                            * muse_losses["route"].mid.float().mean()
                            + float(args.muse_lambda_low_entropy)
                            * muse_losses["route"].low.float().mean()
                        )
                    ).detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/head_js": muse_losses["head_js"].mean().detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/proto_update_weight": muse_losses["proto_update_weight"].detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/u_identity_selected_count": muse_losses[
                        "u_identity_selected_count"
                    ].detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/u_identity_loss": muse_losses["identity"].detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/u_identity_gradient_norm": muse_identity_grad_norm,
                    "muse/u_satellite_identity_selected_count": muse_losses[
                        "u_satellite_identity_selected_count"
                    ].detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/prototype_update_count": muse_losses[
                        "prototype_update_count"
                    ].detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/three_head_agreement_rate": muse_losses[
                        "three_head_agreement_rate"
                    ].detach()
                    if muse_state is not None
                    else 0.0,
                    "muse/u_samples": float(unlabeled_count)
                    if muse_state is not None
                    else 0.0,
                    "muse/u_forward_samples": float(
                        (
                            unlabeled_count + int(satellite_mask.sum().detach().item())
                            if bool(muse_state.get("sat_anchor_ssl", False))
                            else unlabeled_count
                            * (2 if out_u_nuisance is not None else 1)
                        )
                    )
                    if muse_state is not None
                    else 0.0,
                    "rc4/hard_count": rc4_route.hard.sum().detach() if rc4_route is not None else 0.0,
                    "rc4/partial_count": rc4_route.partial.sum().detach() if rc4_route is not None else 0.0,
                    "rc4/negative_count": rc4_route.negative.sum().detach() if rc4_route is not None else 0.0,
                    "rc4/representation_count": rc4_route.representation.sum().detach() if rc4_route is not None else 0.0,
                    "rc4/effective_weighted_coverage": rc4_route.weights.sum().detach() / float(max(1, unlabeled_count)) if rc4_route is not None else 0.0,
                    "rc4/risk_mean": rc4_route.risk.mean().detach() if rc4_route is not None else 0.0,
                    "rc4/p_correct_mean": rc4_route.p_correct.mean().detach() if rc4_route is not None else 0.0,
                    "rc4/estimated_error_mean": (1.0 - rc4_route.p_correct).mean().detach() if rc4_route is not None else 0.0,
                    "rc4/p_set_safe_mean": rc4_route.p_set_safe.mean().detach() if rc4_route is not None else 0.0,
                    "rc4/p_exclusion_safe_mean": rc4_route.p_exclusion_safe.mean().detach() if rc4_route is not None else 0.0,
                    "rc4/partial_safety_threshold": rc4_route.partial_threshold.detach() if rc4_route is not None else 0.0,
                    "rc4/hard_effective_coverage": rc4_route.weights[rc4_route.hard].sum().detach() / float(max(1, unlabeled_count)) if rc4_route is not None else 0.0,
                    "rc4/partial_effective_coverage": rc4_route.weights[rc4_route.partial].sum().detach() / float(max(1, unlabeled_count)) if rc4_route is not None else 0.0,
                    "rc4/negative_effective_coverage": rc4_route.weights[rc4_route.negative].sum().detach() / float(max(1, unlabeled_count)) if rc4_route is not None else 0.0,
                    "rc4/partial_set_loss": muse_losses.get("rc4_partial_set", 0.0) if muse_state is not None else 0.0,
                    "rc4/partial_conditional_loss": muse_losses.get("rc4_partial_conditional", 0.0) if muse_state is not None else 0.0,
                    "rc4/negative_set_loss": muse_losses.get("rc4_negative_set", 0.0) if muse_state is not None else 0.0,
                    "rc4/domain_discriminator_loss": muse_losses.get("domain_discriminator", 0.0) if muse_state is not None else 0.0,
                    "rc4/domain_confusion_loss": muse_losses.get("domain_confusion", 0.0) if muse_state is not None else 0.0,
                    "rc4/feature_anchor_loss": muse_losses.get("feature_anchor", 0.0) if muse_state is not None else 0.0,
                    "rc4/tail_scale": muse_losses.get("rc4_tail_scale", 1.0) if muse_state is not None else 1.0,
                    "rc4/hard_tail_scale": muse_losses.get("rc4_hard_tail_scale", 1.0) if muse_state is not None else 1.0,
                    "rc4/partial_set_tail_scale": muse_losses.get("rc4_partial_set_tail_scale", 1.0) if muse_state is not None else 1.0,
                    "rc4/partial_conditional_tail_scale": muse_losses.get("rc4_partial_conditional_tail_scale", 1.0) if muse_state is not None else 1.0,
                    "rc4/components_finite": muse_losses.get("rc4_components_finite", 1.0) if muse_state is not None else 1.0,
                    "rc4/candidate_size_mean": rc4_route.candidate_mask.sum(dim=-1).float().mean().detach() if rc4_route is not None else 0.0,
                    **rc4_gradient_metrics,
                    "sat_anchor/trusted_count": (
                        sat_anchor_route.trusted.sum().detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/strict_count": (
                        sat_anchor_route.strict.sum().detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/filled_count": (
                        muse_losses["filled_count"].detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/agreement_rate": (
                        sat_anchor_route.agreement.float().mean().detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/confidence_mean": (
                        sat_anchor_route.confidence.mean().detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/margin_mean": (
                        sat_anchor_route.margin.mean().detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/class_coverage": (
                        float(torch.unique(
                            sat_anchor_route.pseudo[sat_anchor_route.trusted]
                        ).numel())
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/receiver_coverage": (
                        float(torch.unique(
                            receiver_u[sat_anchor_route.trusted]
                        ).numel())
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        and torch.is_tensor(receiver_u)
                        else 0.0
                    ),
                    "sat_anchor/pair_active": (
                        muse_losses["pair_active"].detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/pair_drift": (
                        muse_losses["pair_drift"].detach()
                        if muse_state is not None
                        and bool(muse_state.get("sat_anchor_ssl", False))
                        else 0.0
                    ),
                    "sat_anchor/pair_grad_norm": sat_anchor_pair_grad_norm,
                    "sat_anchor/satellite_grad_norm": sat_anchor_sat_grad_norm,
                    "sat_anchor/anchor_grad_norm": sat_anchor_anchor_grad_norm,
                    "sat_anchor/pair_sat_grad_cos": sat_anchor_pair_sat_grad_cos,
                    "train/proto_pull_cos": proto_info.get("proto_pull_cos", float("nan")),
                    "train/proto_domain_align": proto_info.get("proto_domain_align", float("nan")),
                    "train/proto_push": proto_info.get("proto_push", float("nan")),
                    "train/proto_active_classes": proto_info.get("proto_active_classes", float("nan")),
                    "train/ow_feat_compact": ow_feat_info.get("compact", float("nan")),
                    "train/ow_feat_inter": ow_feat_info.get("inter", float("nan")),
                    "train/ow_feat_sample_margin": ow_feat_info.get("sample_margin", float("nan")),
                    "train/ow_feat_domain_align": ow_feat_info.get("domain_align", float("nan")),
                    "train/ow_feat_active_classes": ow_feat_info.get("active_classes", float("nan")),
                    "train/ow_feat_pos_angle_deg": ow_feat_info.get("pos_angle_deg", float("nan")),
                    "train/ow_feat_min_inter_deg": ow_feat_info.get("min_inter_angle_deg", float("nan")),
                    "train/ow_feat_pos_angle_p50_deg": ow_feat_info.get("pos_angle_p50_deg", float("nan")),
                    "train/ow_feat_pos_angle_p95_deg": ow_feat_info.get("pos_angle_p95_deg", float("nan")),
                    "train/ow_feat_pos_angle_p99_deg": ow_feat_info.get("pos_angle_p99_deg", float("nan")),
                    "train/ow_feat_pos_angle_max_deg": ow_feat_info.get("pos_angle_max_deg", float("nan")),
                    "train/ow_feat_tail_loss": ow_feat_info.get("tail_loss", float("nan")),
                    "train/ow_feat_tail_cvar_deg": ow_feat_info.get("tail_cvar_deg", float("nan")),
                    "train/ow_feat_tail_frac_gt_3sigma": ow_feat_info.get("tail_frac_gt_3sigma", float("nan")),
                    "train/ow_feat_tail_radius_3sigma_deg": ow_feat_info.get("tail_radius_3sigma_deg", float("nan")),
                    "train/ow_feat_vacuum_loss": ow_feat_info.get("vacuum_loss", float("nan")),
                    "train/ow_feat_vacuum_violation_rate": ow_feat_info.get("vacuum_violation_rate", float("nan")),
                    "train/ow_feat_vacuum_min_neg_angle_deg": ow_feat_info.get("vacuum_min_neg_angle_deg", float("nan")),
                    "train/ow_feat_vacuum_margin_deg": ow_feat_info.get("vacuum_margin_deg", float("nan")),
                    "train/ow_feat_vacuum_boundary_deg": ow_feat_info.get("vacuum_boundary_deg", float("nan")),
                    "train/ow_feat_stage_scale": float(ow_feat_stage_scale),
                    "train/zid_compact_supcon": zid_compact_info.get("supcon", float("nan")),
                    "train/zid_compact_radius": zid_compact_info.get("radius", float("nan")),
                    "train/zid_compact_tail_cvar": zid_compact_info.get("tail_cvar", float("nan")),
                    "train/zid_compact_active_classes": zid_compact_info.get("active_classes", float("nan")),
                    "train/zid_compact_pos_angle_p50_deg": zid_compact_info.get("pos_angle_p50_deg", float("nan")),
                    "train/zid_compact_pos_angle_p95_deg": zid_compact_info.get("pos_angle_p95_deg", float("nan")),
                    "train/zid_compact_pos_angle_p99_deg": zid_compact_info.get("pos_angle_p99_deg", float("nan")),
                    "train/zid_compact_tail_cvar_deg": zid_compact_info.get("tail_cvar_deg", float("nan")),
                    "train/zid_compact_warmup_scale": float(zid_warm),
                    "train/proxy_unknown_active": proxy_unknown_info.get("active", float("nan")),
                    "train/proxy_unknown_known_count": proxy_unknown_info.get("known_count", float("nan")),
                    "train/proxy_unknown_count": proxy_unknown_info.get("proxy_unknown_count", float("nan")),
                    "train/proxy_unknown_virtual_count": proxy_unknown_info.get("virtual_count", float("nan")),
                    "train/proxy_unknown_core_count": proxy_unknown_info.get("core_count", float("nan")),
                    "train/proxy_unknown_tail_count": proxy_unknown_info.get("tail_count", float("nan")),
                    "train/proxy_unknown_overflow_count": proxy_unknown_info.get("overflow_count", float("nan")),
                    "train/proxy_unknown_energy_known": proxy_unknown_info.get("energy_known", float("nan")),
                    "train/proxy_unknown_energy_proxy": proxy_unknown_info.get("energy_proxy", float("nan")),
                    "train/proxy_unknown_energy_virtual": proxy_unknown_info.get("energy_virtual", float("nan")),
                    "train/proxy_unknown_margin": proxy_unknown_info.get("energy_margin", float("nan")),
                    "train/proxy_unknown_accept_energy_threshold": proxy_unknown_info.get("accept_energy_threshold", float("nan")),
                    "train/proxy_unknown_core_energy_threshold": proxy_unknown_info.get("core_energy_threshold", float("nan")),
                    "train/proxy_unknown_vaccept_surrogate": proxy_unknown_info.get("vaccept_surrogate", float("nan")),
                    "train/proxy_unknown_vaccept_surrogate_CVaR": proxy_unknown_info.get("vaccept_surrogate_CVaR", proxy_unknown_info.get("vaccept_surrogate", float("nan"))),
                    "train/proxy_unknown_core_accept_loss": proxy_unknown_info.get("core_accept_loss", float("nan")),
                    "train/proxy_unknown_component_gate_unknown": proxy_unknown_info.get("component_gate_unknown", float("nan")),
                    "train/proxy_unknown_component_gate_accept_prob": proxy_unknown_info.get("component_gate_accept_prob", float("nan")),
                    "train/proxy_unknown_component_gate_accept_prob_max": proxy_unknown_info.get("component_gate_accept_prob_max", float("nan")),
                    "train/proxy_unknown_tail_quarantine_loss": proxy_unknown_info.get("tail_quarantine_loss", float("nan")),
                    "train/proxy_unknown_source_safe_loss": proxy_unknown_info.get("source_safe_loss", float("nan")),
                    "train/proxy_unknown_bridge_governance_loss": proxy_unknown_info.get("bridge_governance_loss", float("nan")),
                    "train/proxy_unknown_shell_outward_accept_loss": proxy_unknown_info.get("shell_outward_accept_loss", float("nan")),
                    "train/proxy_unknown_low_density_accept_loss": proxy_unknown_info.get("low_density_accept_loss", float("nan")),
                    "train/proxy_unknown_energy_margin_quantile_loss": proxy_unknown_info.get("energy_margin_quantile_loss", float("nan")),
                    "train/proxy_unknown_radius_budget_loss": proxy_unknown_info.get("radius_budget_loss", float("nan")),
                    "train/proxy_unknown_radius_inter_ratio_loss": proxy_unknown_info.get("radius_inter_ratio_loss", float("nan")),
                    "train/proxy_unknown_tail_accept_loss": proxy_unknown_info.get("tail_accept_loss", float("nan")),
                    "train/proxy_unknown_overflow_accept_loss": proxy_unknown_info.get("overflow_accept_loss", float("nan")),
                    "train/proxy_unknown_energy_margin_q05": proxy_unknown_info.get("energy_margin_q05", float("nan")),
                    "train/proxy_unknown_energy_margin_q10": proxy_unknown_info.get("energy_margin_q10", float("nan")),
                    "train/proxy_unknown_component_radius_p95_deg": proxy_unknown_info.get("component_radius_p95_deg", float("nan")),
                    "train/proxy_unknown_component_radius_max_deg": proxy_unknown_info.get("component_radius_max_deg", float("nan")),
                    "train/proxy_unknown_component_radius_mode_code": proxy_unknown_info.get("component_radius_mode_code", float("nan")),
                    "train/proxy_unknown_component_gate_radius_p95_deg": proxy_unknown_info.get("component_gate_radius_p95_deg", float("nan")),
                    "train/proxy_unknown_component_gate_radius_max_deg": proxy_unknown_info.get("component_gate_radius_max_deg", float("nan")),
                    "train/proxy_unknown_radius_inter_ratio": proxy_unknown_info.get("radius_inter_ratio", float("nan")),
                    "train/proxy_unknown_radius_to_inter_ratio": proxy_unknown_info.get("radius_to_inter_ratio", float("nan")),
                    "train/proxy_unknown_low_density_accept_prob": proxy_unknown_info.get("low_density_accept_prob", float("nan")),
                    "train/proxy_unknown_low_density_accept_rate": proxy_unknown_info.get("low_density_accept_rate", float("nan")),
                    "train/proxy_unknown_auc_proxy": proxy_unknown_info.get("proxy_unknown_auc", float("nan")),
                    "train/proxy_unknown_virtual_accept_rate": proxy_unknown_info.get("virtual_accept_rate", float("nan")),
                    "train/proxy_unknown_proxy_vaccept": proxy_unknown_info.get("proxy_vaccept", proxy_unknown_info.get("virtual_accept_rate", float("nan"))),
                    "train/proxy_unknown_proxy_vaccept_proxy_only": proxy_unknown_info.get("proxy_vaccept_proxy_only", proxy_unknown_info.get("virtual_accept_rate", float("nan"))),
                    "train/proxy_unknown_proxy_reject_claim_allowed": proxy_unknown_info.get("proxy_reject_claim_allowed", 0.0),
                    "train/proxy_unknown_virtual_accept_rate_core": proxy_unknown_info.get("virtual_accept_rate_core", float("nan")),
                    "train/proxy_unknown_proxy_accept_rate": proxy_unknown_info.get("proxy_accept_rate", float("nan")),
                    "train/proxy_unknown_hard_proxy_accept_rate": proxy_unknown_info.get("hard_proxy_accept_rate", float("nan")),
                    "train/proxy_unknown_shell_accept_rate": proxy_unknown_info.get("shell_accept_rate", float("nan")),
                    "train/proxy_unknown_bridge_accept_rate": proxy_unknown_info.get("bridge_accept_rate", float("nan")),
                    "train/proxy_unknown_outward_accept_rate": proxy_unknown_info.get("outward_accept_rate", float("nan")),
                    "train/proxy_unknown_vacuum_loss": proxy_unknown_info.get("vacuum_loss", float("nan")),
                    "train/proxy_unknown_vacuum_violation_rate": proxy_unknown_info.get("vacuum_violation_rate", float("nan")),
                    "train/proxy_unknown_vacuum_margin_deg": proxy_unknown_info.get("vacuum_margin_deg", float("nan")),
                    "train/proxy_unknown_vacuum_min_angle_deg": proxy_unknown_info.get("vacuum_min_angle_deg", float("nan")),
                    "train/proxy_unknown_stage_scale": float(proxy_stage_scale),
                    "train/soft_unknown_mixup_count": soft_unknown_mixup_info.get("soft_unknown_mixup_count", float("nan")),
                    "train/soft_unknown_mixup_order": soft_unknown_mixup_info.get("soft_unknown_mixup_order", float("nan")),
                    "train/soft_unknown_mixup_ce": soft_unknown_mixup_info.get("soft_unknown_mixup_ce", float("nan")),
                    "train/soft_unknown_mixup_energy": soft_unknown_mixup_info.get("soft_unknown_mixup_energy", float("nan")),
                    "train/soft_unknown_mixup_vacuum": soft_unknown_mixup_info.get("soft_unknown_mixup_vacuum", float("nan")),
                    "train/soft_unknown_mixup_virtual_accept_rate": soft_unknown_mixup_info.get("soft_unknown_mixup_virtual_accept_rate", float("nan")),
                    "train/soft_unknown_mixup_vacuum_violation": soft_unknown_mixup_info.get("soft_unknown_mixup_vacuum_violation", float("nan")),
                    "train/soft_unknown_mixup_stage_scale": float(soft_unknown_mixup_stage_scale),
                "train/source_episode_loss": source_episode_info.get("source_episode_loss", float("nan")),
                "train/source_episode_leave_domain_loss": source_episode_info.get(
                    "source_episode_leave_domain_loss", float("nan")
                ),
                    "train/source_episode_overflow_rate": source_episode_info.get("source_episode_overflow_rate", float("nan")),
                    "train/source_overflow": source_episode_info.get("source_overflow", source_episode_info.get("source_episode_overflow_rate", float("nan"))),
                    "train/source_episode_radius_3sigma_deg": source_episode_info.get("source_episode_radius_3sigma_deg", float("nan")),
                    "train/source_episode_radius_core_deg": source_episode_info.get("source_episode_radius_core_deg", float("nan")),
                    "train/source_episode_radius_safe_deg": source_episode_info.get("source_episode_radius_safe_deg", float("nan")),
                    "train/source_episode_radius_mode_code": source_episode_info.get("source_episode_radius_mode_code", float("nan")),
                    "train/source_episode_tail_query_rate": source_episode_info.get("source_episode_tail_query_rate", float("nan")),
                    "train/source_episode_val_angle_deg": source_episode_info.get("source_episode_val_angle_deg", float("nan")),
                    "train/source_episode_classes": source_episode_info.get("source_episode_classes", float("nan")),
                    "train/source_episode_domains": source_episode_info.get("source_episode_domains", float("nan")),
                    "train/source_episode_mixup_count": source_episode_info.get("source_episode_mixup_count", float("nan")),
                    "train/source_episode_mixup_order": source_episode_info.get("source_episode_mixup_order", float("nan")),
                    "train/source_episode_mixup_loss": source_episode_info.get("source_episode_mixup_loss", float("nan")),
                    "train/source_episode_mixup_overflow_rate": source_episode_info.get("source_episode_mixup_overflow_rate", float("nan")),
                    "train/source_episode_mixup_margin_deg": source_episode_info.get("source_episode_mixup_margin_deg", float("nan")),
                    "train/source_episode_stage_scale": float(source_episode_stage_scale),
                    "train/source_episode_structural_stage_scale": float(source_episode_structural_stage_scale),
                    "train/source_episode_loss_upper_bound": source_episode_info.get(
                        "source_episode_loss_upper_bound", float("nan")
                    ),
                    "train/source_episode_receiver_local_component_count": source_episode_info.get(
                        "source_episode_receiver_local_component_count", float("nan")
                    ),
                    "train/source_episode_local_component_coverage": source_episode_info.get(
                        "source_episode_local_component_coverage", float("nan")
                    ),
                    "train/source_episode_local_component_compact_loss": source_episode_info.get(
                        "source_episode_local_component_compact_loss", float("nan")
                    ),
                    "train/source_episode_local_component_invariant_loss": source_episode_info.get(
                        "source_episode_local_component_invariant_loss", float("nan")
                    ),
                    "train/source_episode_local_component_inter_loss": source_episode_info.get(
                        "source_episode_local_component_inter_loss", float("nan")
                    ),
                    "train/source_episode_local_component_overlap_loss": source_episode_info.get(
                        "source_episode_local_component_overlap_loss", float("nan")
                    ),
                    "train/source_episode_leave_domain_target_loss": source_episode_info.get(
                        "source_episode_leave_domain_target_loss", float("nan")
                    ),
                    "train/source_episode_local_component_min_inter_deg": source_episode_info.get(
                        "source_episode_local_component_min_inter_deg", float("nan")
                    ),
                    "train/source_episode_local_component_max_radius_inter_ratio": source_episode_info.get(
                        "source_episode_local_component_max_radius_inter_ratio", float("nan")
                    ),
                    "train/source_episode_local_component_accept_loss": source_episode_info.get(
                        "source_episode_local_component_accept_loss", float("nan")
                    ),
                    "train/source_episode_local_component_density_loss": source_episode_info.get(
                        "source_episode_local_component_density_loss", float("nan")
                    ),
                    "train/source_episode_local_component_accept_raw_loss": source_episode_info.get(
                        "source_episode_local_component_accept_raw_loss", float("nan")
                    ),
                    "train/source_episode_local_component_density_raw_loss": source_episode_info.get(
                        "source_episode_local_component_density_raw_loss", float("nan")
                    ),
                    "train/source_episode_local_component_radius_p95_deg": source_episode_info.get(
                        "source_episode_local_component_radius_p95_deg", float("nan")
                    ),
                    "train/source_episode_local_component_radius_min_deg": source_episode_info.get(
                        "source_episode_local_component_radius_min_deg", float("nan")
                    ),
                    "train/source_episode_local_component_loss_radius_floor_deg": source_episode_info.get(
                        "source_episode_local_component_loss_radius_floor_deg", float("nan")
                    ),
                    "train/source_episode_local_component_radius_floor_rate": source_episode_info.get(
                        "source_episode_local_component_radius_floor_rate", float("nan")
                    ),
                    "train/source_episode_local_component_center_spread_deg": source_episode_info.get(
                        "source_episode_local_component_center_spread_deg", float("nan")
                    ),
                    "train/source_episode_local_component_structural_active": source_episode_info.get(
                        "source_episode_local_component_structural_active", float("nan")
                    ),
                    "train/source_episode_core_count": source_episode_info.get("source_episode_core_count", float("nan")),
                    "train/source_episode_tail_count": source_episode_info.get("source_episode_tail_count", float("nan")),
                    "train/source_episode_outside_count": source_episode_info.get("source_episode_outside_count", float("nan")),
                    "train/source_episode_core_rate": source_episode_info.get("source_episode_core_rate", float("nan")),
                    "train/source_episode_tail_rate": source_episode_info.get("source_episode_tail_rate", float("nan")),
                    "train/source_episode_outside_rate": source_episode_info.get("source_episode_outside_rate", float("nan")),
                    "train/source_episode_zid_p50_deg": source_episode_info.get("source_episode_zid_p50_deg", float("nan")),
                    "train/source_episode_zid_p95_deg": source_episode_info.get("source_episode_zid_p95_deg", float("nan")),
                    "train/source_episode_zid_p99_deg": source_episode_info.get("source_episode_zid_p99_deg", float("nan")),
                    "train/source_episode_zid_tail_cvar_deg": source_episode_info.get(
                        "source_episode_zid_tail_cvar_deg", float("nan")
                    ),
                    "train/source_episode_core_tail_outside_ready": source_episode_info.get(
                        "source_episode_core_tail_outside_ready", float("nan")
                    ),
                "train/source_episode_density_gate_active": source_episode_info.get(
                    "source_episode_density_gate_active", float("nan")
                ),
                "train/source_episode_multiview_separate_geometry": source_episode_info.get(
                    "source_episode_multiview_separate_geometry", 0.0
                ),
                "train/source_episode_multiview_active_weight": source_episode_info.get(
                    "source_episode_multiview_active_weight", float("nan")
                ),
                "train/source_episode_multiview_normalized": source_episode_info.get(
                    "source_episode_multiview_normalized", 0.0
                ),
                "train/source_episode_multiview_weighted_sum_loss": source_episode_info.get(
                    "source_episode_multiview_weighted_sum_loss", float("nan")
                ),
                **{
                    f"train/source_episode_{view}_{key}": source_episode_info.get(
                        f"{view}_{key}", float("nan")
                    )
                    for view in ("clean", "sat")
                    for key in (
                        "source_episode_overflow_rate",
                        "source_episode_zid_p95_deg",
                        "source_episode_zid_p99_deg",
                        "source_episode_zid_tail_cvar_deg",
                        "source_episode_local_component_radius_p95_deg",
                        "source_episode_local_component_radius_min_deg",
                        "source_episode_local_component_radius_floor_rate",
                        "source_episode_leave_domain_loss",
                        "source_episode_local_component_compact_loss",
                        "source_episode_local_component_invariant_loss",
                        "source_episode_local_component_inter_loss",
                        "source_episode_local_component_accept_loss",
                        "source_episode_local_component_density_loss",
                        "source_episode_local_component_accept_raw_loss",
                        "source_episode_local_component_density_raw_loss",
                    )
                },
                    "train/dm_accept_active": direct_metric_info.get("active", float("nan")),
                    "train/dm_accept_active_classes": direct_metric_info.get("active_classes", float("nan")),
                    "train/dm_accept_zid_p50_deg": direct_metric_info.get("zid_p50_deg", float("nan")),
                    "train/dm_accept_zid_p95_deg": direct_metric_info.get("zid_p95_deg", float("nan")),
                    "train/dm_accept_zid_p99_deg": direct_metric_info.get("zid_p99_deg", float("nan")),
                    "train/dm_accept_zid_tail_cvar_deg": direct_metric_info.get("zid_tail_cvar_deg", float("nan")),
                    "train/dm_accept_source_overflow": direct_metric_info.get("source_overflow", float("nan")),
                    "train/dm_accept_source_overflow_hard": direct_metric_info.get(
                        "source_overflow_hard", float("nan")
                    ),
                    "train/dm_accept_source_overflow_loss": direct_metric_info.get("source_overflow_loss", float("nan")),
                    "train/dm_accept_proxy_vaccept": direct_metric_info.get("proxy_vaccept", float("nan")),
                    "train/dm_accept_proxy_vaccept_loss": direct_metric_info.get("proxy_vaccept_loss", float("nan")),
                    "train/dm_accept_bridge_accept_rate": direct_metric_info.get("bridge_accept_rate", float("nan")),
                    "train/dm_accept_bridge_accept_loss": direct_metric_info.get("bridge_accept_loss", float("nan")),
                    "train/dm_accept_shell_accept_rate": direct_metric_info.get("shell_accept_rate", float("nan")),
                    "train/dm_accept_outward_accept_rate": direct_metric_info.get("outward_accept_rate", float("nan")),
                    "train/dm_accept_low_density_accept_rate": direct_metric_info.get("low_density_accept_rate", float("nan")),
                    "train/dm_accept_low_density_accept_loss": direct_metric_info.get("low_density_accept_loss", float("nan")),
                    "train/dm_accept_tail_accept_rate": direct_metric_info.get("tail_accept_rate", float("nan")),
                    "train/dm_accept_tail_accept_loss": direct_metric_info.get("tail_accept_loss", float("nan")),
                    "train/dm_accept_overflow_accept_rate": direct_metric_info.get("overflow_accept_rate", float("nan")),
                    "train/dm_accept_overflow_accept_loss": direct_metric_info.get("overflow_accept_loss", float("nan")),
                    "train/dm_accept_radius_to_inter_ratio": direct_metric_info.get("radius_to_inter_ratio", float("nan")),
                    "train/dm_accept_radius_inter_ratio_loss": direct_metric_info.get("radius_inter_ratio_loss", float("nan")),
                    "train/dm_accept_component_inter_margin_loss": direct_metric_info.get(
                        "component_inter_margin_loss", float("nan")
                    ),
                    "train/dm_accept_component_overlap_loss": direct_metric_info.get(
                        "component_overlap_loss", float("nan")
                    ),
                    "train/dm_accept_query_inter_margin_loss": direct_metric_info.get(
                        "query_inter_margin_loss", float("nan")
                    ),
                    "train/dm_accept_query_overlap_loss": direct_metric_info.get(
                        "query_overlap_loss", float("nan")
                    ),
                    "train/dm_accept_component_min_inter_deg": direct_metric_info.get(
                        "component_min_inter_deg", float("nan")
                    ),
                    "train/dm_accept_global_zid_quantile_loss": direct_metric_info.get(
                        "global_zid_quantile_loss", float("nan")
                    ),
                    "train/dm_accept_hierarchical_class_gate": direct_metric_info.get(
                        "hierarchical_class_gate", float("nan")
                    ),
                    "train/dm_accept_core_accept_rate": direct_metric_info.get("core_accept_rate", float("nan")),
                    "train/dm_accept_core_accept_loss": direct_metric_info.get("core_accept_loss", float("nan")),
                    "train/dm_accept_core_hard_tpr": direct_metric_info.get("core_hard_tpr", float("nan")),
                    "train/dm_accept_core_soft_tpr": direct_metric_info.get("core_soft_tpr", float("nan")),
                    "train/dm_accept_core_tpr_loss": direct_metric_info.get("core_tpr_loss", float("nan")),
                    "train/dm_accept_known_accept_rate": direct_metric_info.get(
                        "known_accept_rate", float("nan")
                    ),
                    "train/dm_accept_known_hard_tpr": direct_metric_info.get(
                        "known_hard_tpr", float("nan")
                    ),
                    "train/dm_accept_known_soft_tpr": direct_metric_info.get(
                        "known_soft_tpr", float("nan")
                    ),
                    "train/dm_accept_known_coverage_loss": direct_metric_info.get(
                        "known_coverage_loss", float("nan")
                    ),
                    "train/dm_accept_known_probability_loss": direct_metric_info.get(
                        "known_probability_loss", float("nan")
                    ),
                    "train/dm_accept_known_radius_loss": direct_metric_info.get(
                        "known_radius_loss", float("nan")
                    ),
                    "train/dm_accept_known_margin_loss": direct_metric_info.get(
                        "known_margin_loss", float("nan")
                    ),
                    "train/dm_accept_known_density_loss": direct_metric_info.get(
                        "known_density_loss", float("nan")
                    ),
                    "train/dm_accept_known_tpr_loss": direct_metric_info.get(
                        "known_tpr_loss", float("nan")
                    ),
                    "train/dm_accept_negative_risk_scale": direct_metric_info.get(
                        "negative_risk_scale", float("nan")
                    ),
                    "train/dm_accept_proxy_gradient_active": direct_metric_info.get(
                        "proxy_gradient_active", float("nan")
                    ),
                    "train/dm_reference_bank_anchor_count": float(
                        direct_metric_reference_bank.anchor_count
                        if direct_metric_reference_bank is not None
                        else 0
                    ),
                    "train/dm_reference_bank_version": float(
                        direct_metric_reference_bank.version
                        if direct_metric_reference_bank is not None
                        else 0
                    ),
                    "train/dm_reference_bank_active_epoch": float(
                        direct_metric_reference_bank.active_epoch
                        if direct_metric_reference_bank is not None
                        else -1
                    ),
                    "train/dm_accept_sat_pair_angle_p95_deg": direct_metric_info.get("sat_pair_angle_p95_deg", float("nan")),
                    "train/dm_accept_sat_pair_loss": direct_metric_info.get("sat_pair_loss", float("nan")),
                    "train/dm_accept_zid_quantile_loss": direct_metric_info.get("zid_quantile_loss", float("nan")),
                    "train/dm_accept_virtual_count": direct_metric_info.get("virtual_count", float("nan")),
                    "train/dm_accept_multiview_separate_geometry": direct_metric_info.get(
                        "multiview_separate_geometry", 0.0
                    ),
                    "train/dm_accept_domain_local_component_gate": direct_metric_info.get(
                        "domain_local_component_gate", 0.0
                    ),
                    "train/dm_accept_global_ball_accept": direct_metric_info.get("global_ball_accept", 1.0),
                    "train/dm_accept_local_component_count": direct_metric_info.get(
                        "local_component_count", 0.0
                    ),
                    "train/dm_accept_local_component_class_coverage": direct_metric_info.get(
                        "local_component_class_coverage", 0.0
                    ),
                    "train/dm_accept_local_zid_p50_deg": direct_metric_info.get("local_zid_p50_deg", float("nan")),
                    "train/dm_accept_local_zid_p95_deg": direct_metric_info.get("local_zid_p95_deg", float("nan")),
                    "train/dm_accept_local_zid_p99_deg": direct_metric_info.get("local_zid_p99_deg", float("nan")),
                    "train/dm_accept_local_zid_tail_cvar_deg": direct_metric_info.get(
                        "local_zid_tail_cvar_deg", float("nan")
                    ),
                    "train/dm_accept_global_diag_zid_p50_deg": direct_metric_info.get(
                        "global_diag_zid_p50_deg", float("nan")
                    ),
                    "train/dm_accept_global_diag_zid_p95_deg": direct_metric_info.get(
                        "global_diag_zid_p95_deg", float("nan")
                    ),
                    "train/dm_accept_global_diag_zid_p99_deg": direct_metric_info.get(
                        "global_diag_zid_p99_deg", float("nan")
                    ),
                    "train/dm_accept_global_diag_zid_tail_cvar_deg": direct_metric_info.get(
                        "global_diag_zid_tail_cvar_deg", float("nan")
                    ),
                    "train/dm_accept_quantile_optimization_scope_local": direct_metric_info.get(
                        "quantile_optimization_scope_local", 0.0
                    ),
                    "train/dm_accept_geometry_stabilized": direct_metric_info.get("geometry_stabilized", float("nan")),
                    "train/dm_accept_geometry_reference_detached": direct_metric_info.get(
                        "geometry_reference_detached", float("nan")
                    ),
                    "train/dm_accept_angle_clamp_eps": direct_metric_info.get("angle_clamp_eps", float("nan")),
                    "train/dm_accept_softplus_clip": direct_metric_info.get("softplus_clip", float("nan")),
                    **{
                        f"train/dm_accept_{view}_{key}": direct_metric_info.get(f"{view}_{key}", float("nan"))
                        for view in ("clean", "sat")
                        for key in (
                            "active",
                            "active_classes",
                            "zid_p50_deg",
                            "zid_p95_deg",
                            "zid_p99_deg",
                            "zid_tail_cvar_deg",
                            "source_overflow",
                            "proxy_vaccept",
                            "bridge_accept_rate",
                            "low_density_accept_rate",
                            "tail_accept_rate",
                            "overflow_accept_rate",
                            "radius_to_inter_ratio",
                            "domain_local_component_gate",
                            "global_ball_accept",
                            "local_component_count",
                            "local_zid_p95_deg",
                            "local_zid_p99_deg",
                        )
                    },
                    "train/dm_accept_stage_scale": float(direct_metric_stage_scale),
                }
            ))
            if bool(muse_state is not None and muse_state.get("fasttrust_rc4", False)):
                rc4_nonfinite_batches += int(
                    bool(skipped_nonfinite_loss) or bool(skipped_nonfinite_grad)
                )
                if _rc4_nonfinite_guard_triggered(
                    rc4_nonfinite_batches,
                    batch_idx,
                    min_count=int(args.rc4_nonfinite_guard_min_count),
                    fraction=float(args.rc4_nonfinite_guard_fraction),
                ):
                    raise RuntimeError(
                        "RC4_SYSTEMIC_NONFINITE_BATCH_GUARD "
                        f"epoch={int(epoch)} batch={int(batch_idx)} "
                        f"nonfinite={int(rc4_nonfinite_batches)} "
                        f"fraction={rc4_nonfinite_batches / float(max(1, batch_idx)):.6f}"
                    )

        train_batches_seconds = time.time() - t0
        base_validation_started = time.time()
        val_stats = evaluate_loader(model, data_ctx["val_loader"], device, data_ctx["domain_label_map"], max_batches=int(args.eval_max_batches))
        base_validation_seconds = time.time() - base_validation_started
        current_source_val = float(val_stats.get("tx_acc", float("nan")))
        if math.isfinite(current_source_val):
            dg_health_best_val = max(float(dg_health_best_val), current_source_val)
            dg_health_drop_pp = max(0.0, float(dg_health_best_val) - current_source_val)
        else:
            dg_health_drop_pp = float("inf")
        if bool(getattr(args, "source_val_dg_health_guard", False)) and epoch >= int(
            getattr(args, "source_val_dg_health_start_epoch", 10)
        ):
            warning_drop = float(getattr(args, "source_val_dg_health_warning_drop_pp", 3.0))
            stop_drop = float(getattr(args, "source_val_dg_health_stop_drop_pp", 8.0))
            health_floor = float(getattr(args, "source_val_dg_health_floor", 60.0))
            if dg_health_drop_pp >= warning_drop:
                dg_health_open_scale = min(
                    float(dg_health_open_scale),
                    float(getattr(args, "source_val_dg_health_min_open_scale", 0.20)),
                )
            elif dg_health_drop_pp <= max(0.5, 0.25 * warning_drop):
                dg_health_open_scale = min(1.0, float(dg_health_open_scale) + 0.20)
            hard_bad = (
                not math.isfinite(current_source_val)
                or current_source_val < health_floor
                or dg_health_drop_pp >= stop_drop
            )
            dg_health_bad_epochs = int(dg_health_bad_epochs + 1) if hard_bad else 0
            if dg_health_bad_epochs >= int(getattr(args, "source_val_dg_health_stop_patience", 1)):
                dg_health_early_stop_requested = True
        source_val_heavy_eval_ran = _should_run_source_val_heavy_eval(epoch, total_epochs, args)
        if source_val_heavy_eval_ran:
            heavy_source_validation_started = time.time()
            source_val_tail_geometry = _evaluate_source_val_tail_geometry(model, data_ctx, device, args)
            source_val_sat_stats = _evaluate_source_val_sat_if_enabled(model, data_ctx, device, args)
            heavy_source_validation_seconds = time.time() - heavy_source_validation_started
            last_source_val_tail_geometry = deepcopy(source_val_tail_geometry)
            last_source_val_sat_stats = deepcopy(source_val_sat_stats)
            last_source_val_heavy_eval_epoch = int(epoch)
        else:
            source_val_tail_geometry = deepcopy(last_source_val_tail_geometry)
            source_val_sat_stats = deepcopy(last_source_val_sat_stats)
        source_val_tail_observation = {
            "train/dm_accept_zid_p95_deg": source_val_tail_geometry.get("local_zid_p95_deg", float("nan")),
            "train/dm_accept_zid_p99_deg": source_val_tail_geometry.get("local_zid_p99_deg", float("nan")),
            "train/dm_accept_zid_tail_cvar_deg": source_val_tail_geometry.get(
                "local_zid_tail_cvar_deg", float("nan")
            ),
            "train/dm_accept_proxy_vaccept": source_val_tail_geometry.get("proxy_vaccept", float("nan")),
        }
        stage_state["tail_observation_fixed_source_val"] = 1.0 if source_val_heavy_eval_ran else 0.0
        stage_state["source_val_heavy_eval_ran"] = 1.0 if source_val_heavy_eval_ran else 0.0
        stage_state["source_val_heavy_eval_stale"] = 0.0 if source_val_heavy_eval_ran else 1.0
        stage_state["source_val_heavy_eval_epoch"] = float(last_source_val_heavy_eval_epoch)
        stage_state["source_val_heavy_eval_start_epoch"] = float(args.source_val_heavy_eval_start_epoch)
        stage_state["source_val_heavy_eval_interval"] = float(args.source_val_heavy_eval_interval)
        stage_state["source_val_heavy_eval_final_window"] = float(args.source_val_heavy_eval_final_window)
        stage_state["source_val_heavy_eval_final_interval"] = float(args.source_val_heavy_eval_final_interval)
        source_val_sat_scores = _satellite_tx_scores(source_val_sat_stats)
        stage_state["source_val_sat_mean_tx"] = (
            sum(source_val_sat_scores) / len(source_val_sat_scores) if source_val_sat_scores else float("nan")
        )
        stage_state["source_val_sat_floor_tx"] = min(source_val_sat_scores) if source_val_sat_scores else float("nan")
        # Phase1 is source-only. Held-out receiver/day and satellite-test views are
        # evaluated exactly once after the source-val-selected checkpoint is frozen.
        test_ran_this_epoch = False
        if test_ran_this_epoch:
            named_stats = evaluate_named_loaders(
                model,
                data_ctx["named_test_loaders"],
                device,
                data_ctx["domain_label_map"],
                max_batches=int(args.eval_max_batches),
            )
        else:
            named_stats = {}
        train_logs = mean_logs(epoch_logs)
        if _fasttrust_lr_enabled(args):
            fasttrust_zero_step_streak = _next_fasttrust_zero_step_streak(
                train_logs, fasttrust_zero_step_streak
            )
        else:
            fasttrust_zero_step_streak = 0
        for tail_key in (
            "train/dm_accept_zid_p95_deg",
            "train/dm_accept_zid_p99_deg",
            "train/dm_accept_zid_tail_cvar_deg",
            "train/dm_accept_proxy_vaccept",
        ):
            train_logs[tail_key] = _max_log_value(epoch_logs, tail_key)
        train_logs["train/tail_epoch_aggregation_max_batch"] = 1.0
        train_logs["train/pseudo_total"] = _sum_log_values(epoch_logs, "train/pseudo_total")
        train_logs["train/pseudo_selected"] = _sum_log_values(epoch_logs, "train/pseudo_selected")
        train_logs["train/pseudo_correct"] = _sum_log_values(epoch_logs, "train/pseudo_correct")
        train_logs["train/pseudo_truth_available"] = _sum_log_values(
            epoch_logs, "train/pseudo_truth_available"
        )
        test_stats = _aggregate_main_test(named_stats, str(args.dataset)) if test_ran_this_epoch else {
            "tx_acc": float("nan"),
            "tx_correct": 0,
            "tx_total": 0,
        }
        sat_test_stats = _evaluate_sat_if_enabled(model, data_ctx, device, args) if test_ran_this_epoch else {}
        stage_state["test_eval_ran"] = 1.0 if test_ran_this_epoch else 0.0
        stage_state["test_eval_interval"] = float(getattr(args, "test_eval_interval", 0))
        stage_state["test_eval_final_window"] = float(getattr(args, "test_eval_final_window", 0))
        stage_state["test_eval_final_interval"] = float(getattr(args, "test_eval_final_interval", 0))
        stats = {
            "train": train_logs,
            "val": val_stats,
            "source_val_sat_named": source_val_sat_stats,
            "source_val_tail_geometry": source_val_tail_geometry,
            "satellite_protocol": dict(getattr(args, "sat_protocol_manifest", {}) or {}),
            "named_test": named_stats,
            "test": test_stats,
            "sat_test_named": sat_test_stats,
        }
        payload = {
            "checkpoint_schema": "ssdg_phase1_training_state_v2",
            "checkpoint_role": "training_epoch_state_in_memory",
            "checkpoint_selection": str(args.checkpoint_selection),
            "run_id": str(getattr(args, "run_id", "")),
            "candidate_id": str(getattr(args, "candidate_id", "")),
            "ablation_id": str(getattr(args, "ablation_id", "")),
            "ablation_config_hash": str(
                getattr(args, "ablation_config_hash", "")
            ),
            "git_commit": str(getattr(args, "git_commit", "")),
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "prototype_memory": proto_bank.state_dict() if proto_bank is not None else None,
            "rng_state": _capture_training_rng_state(sat_gen),
            "tail_state_machine": phase1_v2_tail_machine.state_dict() if phase1_v2_tail_machine is not None else None,
            "tail_rollback_events": deepcopy(tail_rollback_events),
            "guard_history": {
                "previous_protected_metrics": deepcopy(previous_protected_metrics),
                "previous_train_logs": deepcopy(previous_train_logs),
            },
            "baseline_ckpt": args.baseline_ckpt,
            "epoch": epoch,
            "observed_epoch": epoch,
            "state_epoch": epoch,
            "phase": phase,
            "args": vars(args),
            "baseline_args": vars(model_args),
            "split_info": data_ctx["split_info"],
            "stats": stats,
        }
        payload.update(_muse_checkpoint_state(muse_state))
        latest_path = out_dir / "NOT_SAVED_FINAL_ONLY"
        best_path = final_path
        protected_metrics = protected_metric_snapshot(
            val_stats=val_stats,
            test_stats=test_stats,
            named_test_stats=named_stats,
            sat_test_stats=sat_test_stats,
        )
        drop_decision = detect_one_epoch_drop(
            protected_metrics,
            previous_protected_metrics,
            threshold_pp=float(args.one_epoch_drop_guard_pp),
        ) if _joint_safe_guard_enabled(args) and phase == "pseudo" and test_ran_this_epoch else None
        paic_decision = detect_paic_variance_guard(
            train_logs,
            previous_train_logs,
            sat_ce_delta=float(args.paic_guard_sat_ce_delta),
            grad_delta=float(args.paic_guard_grad_delta),
            reliable_drop=float(args.paic_guard_reliable_drop),
            domain_delta=float(args.paic_guard_domain_delta),
            sat_cons_delta=float(args.paic_guard_sat_cons_delta),
        ) if _paic_guard_enabled(args) and phase == "pseudo" else None
        drop_guard_fired = bool(drop_decision and drop_decision.fired)
        paic_guard_fired = bool(paic_decision and paic_decision.fired)
        guard_enabled = _joint_safe_guard_enabled(args)
        test_eval_skipped_guard_block = bool(_joint_safe_guard_enabled(args) and phase == "pseudo" and not test_ran_this_epoch)
        missing_required_metrics = (
            tuple(
                missing_joint_safe_metrics(
                    protected_metrics,
                    require_satellite=bool(getattr(args, "joint_guard_require_satellite", True)),
                )
            )
            if guard_enabled and test_ran_this_epoch and missing_joint_safe_metrics is not None
            else tuple()
        )
        missing_required_guard_fired = bool(missing_required_metrics)
        checkpoint_safe = not bool(
            guard_enabled and (test_eval_skipped_guard_block or missing_required_guard_fired or drop_guard_fired or paic_guard_fired)
        )
        if paic_guard_fired:
            paic_cooldown_remaining = max(paic_cooldown_remaining, max(0, int(args.paic_guard_cooldown_epochs)))
        guard_reasons = []
        if test_eval_skipped_guard_block:
            guard_reasons.append("test_eval_skipped")
        if missing_required_guard_fired:
            guard_reasons.append("missing_required_metrics:" + ",".join(missing_required_metrics))
        if drop_guard_fired:
            guard_reasons.append(drop_decision.reason)
        if paic_guard_fired:
            guard_reasons.append(paic_decision.reason)
        guard_state = {
            "enabled": bool(guard_enabled),
            "checkpoint_safe": bool(checkpoint_safe),
            "missing_required_guard_fired": bool(missing_required_guard_fired),
            "missing_required_metric_count": int(len(missing_required_metrics)),
            "missing_required_metrics": ",".join(missing_required_metrics),
            "test_eval_ran": bool(test_ran_this_epoch),
            "test_eval_skipped_guard_block": bool(test_eval_skipped_guard_block),
            "drop_guard_fired": bool(drop_guard_fired),
            "paic_guard_fired": bool(paic_guard_fired),
            "paic_cooldown_active": bool(paic_cooldown_active),
            "paic_cooldown_remaining": int(paic_cooldown_remaining),
            "reason": ";".join(guard_reasons),
        }
        if drop_decision is not None:
            guard_state.update({f"drop_{key}": value for key, value in drop_decision.details.items()})
        if paic_decision is not None:
            guard_state.update({f"paic_{key}": value for key, value in paic_decision.details.items()})
        phase1_v2_guard_fired = False
        phase1_v2_final_blocked = False
        phase1_v2_reasons: List[str] = []
        if dg_health_early_stop_requested:
            phase1_v2_reasons.append("SOURCE_VAL_DG_HEALTH_STOP")
            phase1_v2_final_blocked = True
            checkpoint_safe = False
        tail_reference_saved = False
        tail_rollback_applied = False
        tail_rollback_reference_epoch = -1
        tail_rejected_checkpoint_path = ""
        if bool(getattr(args, "phase1_v2_hard_gates", False)):
            if (
                assess_endpoint_contract is None
                or assess_open_set_effective_budget is None
                or assess_unlabeled_tri_state is None
                or assess_phase1_v2_final_export_policy is None
                or assess_source_episode_density_gate is None
                or should_skip_phase1_v2_final_export is None
            ):
                raise ImportError("cvsrffi.phase1_v2_control is required for --phase1_v2_hard_gates.")
            endpoint_decision = assess_endpoint_contract(
                {
                    "phase": "Phase1_source_only",
                    "source_only": True,
                    "endpoint_policy_id": str(args.endpoint_accept_policy_id),
                    "endpoint_accept_boundary_exported": False,
                    "endpoint_artifact_required": False,
                    "endpoint_threshold_source": str(args.endpoint_threshold_source),
                    "endpoint_calibration_split": str(args.endpoint_calibration_split),
                    "loss_gate_exported": bool(args.loss_gate_exported),
                    "phase1_proxy_only": True,
                    "real_unknown_eval_available": False,
                    "stage2_success_claim": False,
                    "deployment_success_claim": False,
                }
            )
            guard_state["phase1_v2_endpoint_contract_fired"] = bool(endpoint_decision.fired)
            guard_state.update({f"phase1_v2_endpoint_{key}": value for key, value in endpoint_decision.details.items()})
            if endpoint_decision.fired:
                phase1_v2_reasons.append(endpoint_decision.reason)
            if float(getattr(args, "os_eff_min_budget", 0.0)) > 0.0 and (
                bool(getattr(args, "phase1_v2_os_eff_all_phases", True)) or phase == "pseudo"
            ):
                os_eff_decision = assess_open_set_effective_budget(
                    train_logs,
                    min_budget=float(args.os_eff_min_budget),
                    max_budget=float(args.os_eff_max_budget),
                )
                guard_state["phase1_v2_os_eff_fired"] = bool(os_eff_decision.fired)
                guard_state.update({f"phase1_v2_os_{key}": value for key, value in os_eff_decision.details.items()})
                if os_eff_decision.fired:
                    phase1_v2_reasons.append(os_eff_decision.reason)
            if bool(getattr(args, "u_tri_state_required", False)) and phase == "pseudo":
                u_tri_decision = assess_unlabeled_tri_state(
                    train_logs,
                    required=bool(args.u_direct_idle_blocks_promotion),
                    min_selected=int(args.u_direct_metric_min_selected),
                    min_core_rate=float(args.u_tri_min_core_rate),
                    max_core_rate=float(args.u_tri_max_core_rate),
                    min_ambiguous_rate=float(args.u_tri_min_ambiguous_rate),
                    max_outside_rate=float(args.u_tri_max_outside_rate),
                    min_class_coverage=int(args.u_tri_min_class_coverage),
                    min_domain_coverage=int(args.u_tri_min_domain_coverage),
                    max_pair_disagreement_rate=float(args.u_tri_max_pair_disagreement_rate),
                    min_pseudo_component_agreement=float(args.u_tri_min_pseudo_component_agreement),
                )
                guard_state["phase1_v2_u_tri_state_fired"] = bool(u_tri_decision.fired)
                guard_state.update({f"phase1_v2_u_{key}": value for key, value in u_tri_decision.details.items()})
                if u_tri_decision.fired:
                    phase1_v2_reasons.append(u_tri_decision.reason)
            if (
                bool(getattr(args, "source_episode_density_gate", False))
                and phase == "pseudo"
                and float(getattr(args, "lambda_source_episode", 0.0)) > 0.0
            ):
                source_ep_decision = assess_source_episode_density_gate(
                    train_logs,
                    overflow_warn=float(args.source_episode_overflow_warn),
                    min_local_components=int(args.source_episode_min_local_components),
                )
                guard_state["phase1_v2_source_episode_density_fired"] = bool(source_ep_decision.fired)
                guard_state.update({f"phase1_v2_source_episode_{key}": value for key, value in source_ep_decision.details.items()})
                if source_ep_decision.fired:
                    phase1_v2_reasons.append(source_ep_decision.reason)
        if bool(getattr(args, "feasibility_gate", False)):
            if assess_feasibility_gate is None:
                raise ImportError("cvsrffi.phase1_v2_control is required for --feasibility_gate.")
            feasibility_decision = assess_feasibility_gate(
                {
                    "stage": str(args.feasibility_stage),
                    "relaxed_pass": bool(args.feasibility_relaxed_pass),
                    "local_pass": bool(args.feasibility_local_pass),
                    "loss_response_slope": float(args.feasibility_loss_response_slope),
                    "overflow_excess_cvar95_delta": float(args.feasibility_overflow_excess_cvar95_delta),
                }
            )
            guard_state["phase1_v2_feasibility_fired"] = bool(feasibility_decision.fired)
            guard_state.update({f"phase1_v2_feasibility_{key}": value for key, value in feasibility_decision.details.items()})
            if feasibility_decision.fired:
                phase1_v2_reasons.append(feasibility_decision.reason)
        promotion_stage_ready = not bool(getattr(args, "u_tri_state_required", False)) or phase == "pseudo"
        guard_state["phase1_v2_promotion_stage_ready"] = bool(promotion_stage_ready)
        if phase1_v2_tail_machine is not None and not promotion_stage_ready:
            phase1_v2_reasons.append("US_STAGE_NOT_READY")
            phase1_v2_final_blocked = True
            checkpoint_safe = False
            guard_state["phase1_v2_tail_fired"] = True
            guard_state["phase1_v2_tail_blocks_best"] = True
            guard_state["phase1_v2_tail_blocks_final"] = True
            guard_state["phase1_v2_tail_state_code"] = 4.0
            guard_state["phase1_v2_tail_action_code"] = 0.0
            guard_state["phase1_v2_tail_reference_ready"] = 0.0
            guard_state["phase1_v2_tail_deferred_to_pseudo"] = 1.0
        if phase1_v2_tail_machine is not None and promotion_stage_ready and source_val_heavy_eval_ran:
            tail_decision = phase1_v2_tail_machine.update(source_val_tail_observation)
            guard_state["phase1_v2_tail_fired"] = bool(tail_decision.fired)
            guard_state["phase1_v2_tail_blocks_best"] = bool(tail_decision.blocks_best)
            guard_state["phase1_v2_tail_blocks_final"] = bool(tail_decision.blocks_final)
            guard_state["phase1_v2_tail_state_code"] = {"NORMAL": 0.0, "WARNING": 1.0, "ROLLBACK": 2.0, "STOP": 3.0, "INSUFFICIENT": 4.0}.get(tail_decision.state, -1.0)
            guard_state["phase1_v2_tail_action_code"] = {"NONE": 0.0, "WARNING": 1.0, "ROLLBACK": 2.0, "STOP": 3.0}.get(tail_decision.action, -1.0)
            guard_state.update({f"phase1_v2_{key}": value for key, value in tail_decision.details.items()})
            if (
                int(epoch) < int(total_epochs)
                and
                float(tail_decision.details.get("tail_reference_improved", 0.0)) > 0.0
                and float(tail_decision.details.get("tail_reference_ready", 0.0)) > 0.0
                and tail_decision.state == "NORMAL"
                and float(tail_decision.details.get("tail_expansion_blocks_final", 0.0)) < 1.0
                and float(tail_decision.details.get("tail_expansion_blocks_promotion", 0.0)) < 1.0
                and (
                    float(tail_decision.details.get("tail_absolute_violation", 0.0)) < 1.0
                    or not bool(args.tail_safety_reference_requires_absolute_safe)
                )
            ):
                phase1_v2_tail_machine.commit_reference(tail_decision)
                tail_reference_geometry = deepcopy(source_val_tail_geometry)
                tail_reference_geometry["reference_epoch"] = int(epoch)
                tail_reference_geometry["final_epoch_excluded"] = True
                tail_reference_geometry["reference_kind"] = "metric_only_robust_tail_reference"
                tail_reference_geometry["reference_decision"] = dict(tail_decision.details)
                tail_reference_epoch = int(epoch)
                tail_reference_saved = True
            if tail_decision.action == "STOP" and bool(args.tail_safety_training_stop_enabled):
                phase1_v2_final_blocked = True
                tail_early_stop_requested = True
            if tail_decision.blocks_best:
                phase1_v2_reasons.append(tail_decision.reason or f"tail_state_{tail_decision.state}")
            if tail_decision.blocks_final and bool(getattr(args, "tail_stop_blocks_final", True)):
                phase1_v2_final_blocked = True
                checkpoint_safe = False
                if not tail_decision.blocks_best:
                    phase1_v2_reasons.append(
                        tail_decision.reason or f"tail_final_block_{tail_decision.state}"
                    )
        elif phase1_v2_tail_machine is not None and promotion_stage_ready:
            guard_state["phase1_v2_tail_eval_deferred"] = True
            guard_state["phase1_v2_tail_fired"] = False
            guard_state["phase1_v2_tail_blocks_best"] = False
            guard_state["phase1_v2_tail_blocks_final"] = False
        if bool(getattr(args, "phase1_v2_hard_gates", False)) and assess_phase1_v2_final_export_policy is not None:
            final_export_decision = assess_phase1_v2_final_export_policy(
                phase1_v2_reasons,
                tail_blocks_final=bool(phase1_v2_final_blocked),
                fail_closed=bool(getattr(args, "phase1_v2_guard_blocks_final", True)),
            )
            guard_state["phase1_v2_final_export_policy_fired"] = bool(final_export_decision.fired)
            guard_state.update({f"phase1_v2_final_export_{key}": value for key, value in final_export_decision.details.items()})
            if final_export_decision.fired:
                phase1_v2_final_blocked = True
        phase1_v2_reasons = [reason for reason in phase1_v2_reasons if str(reason).strip()]
        phase1_v2_guard_fired = bool(phase1_v2_reasons)
        if phase1_v2_guard_fired:
            checkpoint_safe = False
        guard_state["phase1_v2_guard_fired"] = bool(phase1_v2_guard_fired)
        guard_state["phase1_v2_final_blocked"] = bool(phase1_v2_final_blocked)
        guard_state["phase1_v2_tail_reference_saved"] = bool(tail_reference_saved)
        guard_state["phase1_v2_tail_reference_path"] = "METRIC_ONLY"
        guard_state["phase1_v2_tail_rollback_applied"] = bool(tail_rollback_applied)
        guard_state["phase1_v2_tail_rollback_reference_epoch"] = int(tail_rollback_reference_epoch)
        guard_state["phase1_v2_tail_rejected_checkpoint_path"] = tail_rejected_checkpoint_path
        guard_state["phase1_v2_tail_early_stop_requested"] = bool(tail_early_stop_requested)
        guard_state["source_val_dg_health_guard_enabled"] = bool(
            getattr(args, "source_val_dg_health_guard", False)
        )
        guard_state["source_val_dg_health_best_val"] = float(dg_health_best_val)
        guard_state["source_val_dg_health_drop_pp"] = float(dg_health_drop_pp)
        guard_state["source_val_dg_health_open_scale"] = float(dg_health_open_scale)
        guard_state["source_val_dg_health_bad_epochs"] = int(dg_health_bad_epochs)
        guard_state["source_val_dg_health_early_stop_requested"] = bool(
            dg_health_early_stop_requested
        )
        if phase1_v2_reasons:
            guard_state["reason"] = ";".join([r for r in [guard_state.get("reason", ""), *phase1_v2_reasons] if str(r).strip()])
        payload["joint_guard"] = dict(guard_state)
        payload["tail_state_machine"] = phase1_v2_tail_machine.state_dict() if phase1_v2_tail_machine is not None else None
        payload["checkpoint_status"] = {
            "state": (
                "STOPPED_TAIL"
                if tail_early_stop_requested
                else "STOPPED_DG_HEALTH"
                if dg_health_early_stop_requested
                else "RUNNING"
            ),
            "checkpoint_safe": bool(checkpoint_safe),
            "phase1_v2_guard_fired": bool(phase1_v2_guard_fired),
            "phase1_v2_final_blocked": bool(phase1_v2_final_blocked),
            "reason": str(guard_state.get("reason", "")),
        }
        recovery_interval = int(getattr(args, "rc4_recovery_checkpoint_interval", 0))
        recovery_saved = False
        if (
            muse_state is not None
            and bool(muse_state.get("fasttrust_rc4", False))
            and recovery_interval > 0
            and int(epoch) % recovery_interval == 0
        ):
            recovery_loss_keys = (
                "train/loss",
                "train/loss_labeled",
                "train/loss_closed_group",
                "train/loss_open_group",
                "train/loss_adv_labeled",
                "rc4/partial_set_loss",
                "rc4/partial_conditional_loss",
                "rc4/domain_discriminator_loss",
                "rc4/domain_confusion_loss",
            )
            finite_losses = all(
                key not in train_logs or math.isfinite(float(train_logs[key]))
                for key in recovery_loss_keys
            )
            if finite_losses and float(train_logs.get("train/optimizer_step_applied", 0.0)) > 0.0:
                checkpoint_io_started = time.time()
                recovery_payload = dict(payload)
                recovery_payload["checkpoint_role"] = "recovery_latest_finite_not_for_selection"
                save_payload(out_dir / "latest_finite_ssdg.pth", recovery_payload)
                if int(epoch) == 90:
                    e90_payload = dict(recovery_payload)
                    e90_payload["checkpoint_role"] = "recovery_e90_not_for_selection"
                    save_payload(out_dir / "recovery_e90_ssdg.pth", e90_payload)
                checkpoint_io_seconds += time.time() - checkpoint_io_started
                recovery_saved = True
        train_logs["rc4/recovery_checkpoint_saved"] = 1.0 if recovery_saved else 0.0
        train_logs["rc4/first_anomaly_packet_written"] = 1.0 if rc4_anomaly_written else 0.0
        safe_checkpoint_saved = False
        is_best = False
        best_metric_name = str(args.best_metric)
        best_metric_needs_test = best_metric_name not in {"clean_val_tx", "source_val_sat_hmean"}
        current_best_score = (
            _best_score(
                val_stats,
                test_stats,
                source_val_sat_stats if best_metric_name == "source_val_sat_hmean" else sat_test_stats,
                best_metric_name,
                named_stats,
                args,
            )
            if (test_ran_this_epoch or not best_metric_needs_test)
            else float("-inf")
        )
        source_val_best_is_fresh = best_metric_name != "source_val_sat_hmean" or source_val_heavy_eval_ran
        allow_best_update = (test_ran_this_epoch or not best_metric_needs_test) and source_val_best_is_fresh
        if allow_best_update and current_best_score > best_score:
            best_score = float(current_best_score)
            best_val = float(val_stats["tx_acc"])
            best_test = float(test_stats["tx_acc"])
            best_epoch = int(epoch)
            is_best = True
        elapsed = time.time() - t0
        if muse_state is not None:
            peak_memory_bytes = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            train_logs.update(
                _fasttrust_epoch_resource_metrics(
                    u_samples_per_step=float(train_logs.get("muse/u_samples", 0.0)),
                    u_forward_samples_per_step=float(
                        train_logs.get("muse/u_forward_samples", 0.0)
                    ),
                    steps=len(epoch_logs),
                    elapsed_s=elapsed,
                    peak_memory_bytes=peak_memory_bytes,
                )
            )
            train_logs.update(
                _fasttrust_epoch_stage_timing_metrics(
                    train_batches_s=train_batches_seconds,
                    base_validation_s=base_validation_seconds,
                    heavy_source_validation_s=heavy_source_validation_seconds,
                    checkpoint_io_s=checkpoint_io_seconds,
                    epoch_elapsed_s=elapsed,
                )
            )
        telemetry_rows.append(
            _build_ssdg_epoch_telemetry_row(
                args=args,
                epoch=epoch,
                epochs=total_epochs,
                lr=float(optimizer.param_groups[0]["lr"]),
                epoch_time_s=elapsed,
                phase=phase,
                train_logs=train_logs,
                val_stats=val_stats,
                test_stats=test_stats,
                named_test_stats=named_stats,
                sat_test_stats=sat_test_stats,
                stage_state=stage_state,
                mixstyle_state=mixstyle_state,
                aug_state=aug_state,
                loss_weights=cur_w,
                best_score=best_score,
                best_val=best_val,
                best_test=best_test,
                best_epoch=best_epoch,
                latest_path=str(latest_path),
                best_path=str(best_path),
                is_best=is_best,
                protected_metrics=protected_metrics,
                guard_state=guard_state,
                phase2_audit_state=phase2_audit_state,
                safe_latest_path=str(safe_latest_path),
                safe_best_path=str(safe_best_path),
                safe_checkpoint_saved=safe_checkpoint_saved,
            )
        )
        _write_ssdg_epoch_telemetry(metrics_csv_path, metrics_jsonl_path, telemetry_rows)
        print(
            format_ssdg_epoch_block(
                epoch=epoch,
                epochs=total_epochs,
                lr=float(optimizer.param_groups[0]["lr"]),
                epoch_time_s=elapsed,
                phase=phase,
                train_logs=train_logs,
                val_stats=val_stats,
                test_stats=test_stats,
                named_test_stats=named_stats,
                named_test_meta=data_ctx["split_info"].get("named_test_meta", {}),
                sat_test_stats=sat_test_stats,
                stage_state=stage_state,
                mixstyle_state=mixstyle_state,
                aug_state=aug_state,
                loss_weights=cur_w,
                best_val=best_val,
                best_test=best_test,
                best_epoch=best_epoch,
                latest_path=str(latest_path),
                best_path=str(best_path),
                is_best=is_best,
                protected_metrics=protected_metrics,
                guard_state=guard_state,
                phase2_audit_state=phase2_audit_state,
                safe_latest_path=str(safe_latest_path),
                safe_best_path=str(safe_best_path),
                safe_checkpoint_saved=safe_checkpoint_saved,
            ),
            flush=True,
        )
        if _fasttrust_zero_step_abort_required(fasttrust_zero_step_streak):
            print(
                f"[FASTTRUST-SYSTEMIC-FAILURE] epoch={int(epoch)} "
                f"zero_step_epoch_streak={int(fasttrust_zero_step_streak)} "
                "reason=consecutive_full_epochs_without_optimizer_update",
                flush=True,
            )
            raise RuntimeError("FASTTRUST_CONSECUTIVE_ZERO_OPTIMIZER_STEP_EPOCHS")
        if test_ran_this_epoch and not tail_rollback_applied:
            previous_protected_metrics = dict(protected_metrics)
        if not tail_rollback_applied:
            previous_train_logs = dict(train_logs)
        if tail_early_stop_requested or dg_health_early_stop_requested:
            print(
                f"[PHASE1-V2] training_stopped=1 epoch={int(epoch)} "
                f"reason={'tail_safety_fail_closed' if tail_early_stop_requested else 'source_val_dg_health'} "
                "final_export_blocked=1",
                flush=True,
            )
            break
    selected_checkpoint = final_path
    selected_epoch = int(epoch)
    selected_role = "training_final_only"
    selected_source = "training_final_only"
    final_source_val_tail = _evaluate_source_val_tail_geometry(model, data_ctx, device, args)
    final_zid_leakage_probe = _evaluate_zid_leakage_probes(model, data_ctx, device, args)
    leakage_decision = _assess_zid_leakage_probe(final_zid_leakage_probe, args)
    if bool(leakage_decision["fired"]):
        phase1_v2_final_blocked = True
        phase1_v2_reasons.extend(str(leakage_decision["reason"]).split(";"))
    if bool(args.direct_metric_require_domain_local_components):
        local_gate_ready = (
            float(final_source_val_tail.get("domain_local_component_gate", 0.0)) > 0.0
            and float(final_source_val_tail.get("global_ball_accept", 1.0)) == 0.0
            and float(final_source_val_tail.get("local_component_count", 0.0)) > 0.0
        )
        if not local_gate_ready:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.append("FINAL_LOCAL_COMPONENT_GEOMETRY_INCOMPLETE")

    final_payload = deepcopy(payload)
    final_payload.update(
        {
            "checkpoint_role": selected_role,
            "checkpoint_selection": str(args.checkpoint_selection),
            "model": getattr(model, "_orig_mod", model).state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "prototype_memory": proto_bank.state_dict() if proto_bank is not None else None,
            "rng_state": _capture_training_rng_state(sat_gen),
            "tail_state_machine": phase1_v2_tail_machine.state_dict() if phase1_v2_tail_machine is not None else None,
            "tail_rollback_events": deepcopy(tail_rollback_events),
            "final_epoch": int(selected_epoch),
            "observed_epoch": int(epoch),
            "selection_epoch": int(selected_epoch),
        }
    )
    final_payload.update(_muse_checkpoint_state(muse_state))
    final_payload.setdefault("stats", {})
    final_payload["stats"]["source_val_tail_geometry"] = final_source_val_tail
    final_payload["stats"]["zid_leakage_probe"] = final_zid_leakage_probe
    final_payload["stats"]["satellite_protocol"] = dict(getattr(args, "sat_protocol_manifest", {}) or {})
    final_payload["final_only_evidence"] = {
        "selection_source": "training_final_only",
        "telemetry_best_metric": str(args.best_metric),
        "telemetry_best_epoch": int(best_epoch),
        "telemetry_best_score": float(best_score) if math.isfinite(float(best_score)) else None,
        "source_val_tail_geometry": final_source_val_tail,
        "zid_leakage_probe": final_zid_leakage_probe,
        "satellite_protocol": dict(getattr(args, "sat_protocol_manifest", {}) or {}),
    }
    reference_source_val_tail = (
        deepcopy(tail_reference_geometry)
        if tail_reference_geometry
        else {"status": "NOT_AVAILABLE", "reason": "tail_reference_metric_missing"}
    )
    reference_final_tail_gate: Dict[str, Any] = {
        "status": "FAILED",
        "protocol": "fixed_source_val_multiview_local_component_v2",
        "selection_policy": str(args.checkpoint_selection),
        "reference": reference_source_val_tail,
        "final": final_source_val_tail,
        "p99_delta_deg": None,
        "tail_cvar_delta_deg": None,
        "proxy_vaccept_delta": None,
        "satellite_protocol_registry_sha256": (args.sat_protocol_manifest or {}).get("registry_sha256", ""),
        "blocks_final_export": True,
        "blocks_promotion": True,
    }
    try:
        reference_p99 = float(reference_source_val_tail.get("local_zid_p99_deg", float("nan")))
        final_p99 = float(final_source_val_tail.get("local_zid_p99_deg", float("nan")))
        reference_cvar = float(reference_source_val_tail.get("local_zid_tail_cvar_deg", float("nan")))
        final_cvar = float(final_source_val_tail.get("local_zid_tail_cvar_deg", float("nan")))
        reference_proxy = float(reference_source_val_tail.get("proxy_vaccept", float("nan")))
        final_proxy = float(final_source_val_tail.get("proxy_vaccept", float("nan")))
        final_p95 = float(final_source_val_tail.get("local_zid_p95_deg", float("nan")))
        p99_delta = final_p99 - reference_p99
        cvar_delta = final_cvar - reference_cvar
        proxy_delta = final_proxy - reference_proxy
        if (
            str(reference_source_val_tail.get("status", "")) == "COMPLETE"
            and str(final_source_val_tail.get("status", "")) == "COMPLETE"
            and math.isfinite(reference_p99)
            and math.isfinite(final_p99)
            and math.isfinite(reference_cvar)
            and math.isfinite(final_cvar)
            and math.isfinite(reference_proxy)
            and math.isfinite(final_proxy)
            and math.isfinite(final_p95)
            and float(reference_source_val_tail.get("domain_local_component_gate", 0.0)) > 0.0
            and float(final_source_val_tail.get("domain_local_component_gate", 0.0)) > 0.0
            and float(reference_source_val_tail.get("global_ball_accept", 1.0)) == 0.0
            and float(final_source_val_tail.get("global_ball_accept", 1.0)) == 0.0
        ):
            absolute_unsafe = (
                final_p95 > float(args.tail_safety_p95_target_deg)
                or final_p99 > float(args.tail_safety_p99_target_deg)
                or final_cvar > float(args.tail_safety_cvar_target_deg)
                or final_proxy > float(args.tail_safety_proxy_vaccept_target)
            )
            block_final = (
                absolute_unsafe
                or p99_delta > float(args.tail_safety_p99_expansion_block_final_delta)
                or cvar_delta > float(args.tail_safety_cvar_expansion_block_final_delta)
            )
            block_promotion = (
                absolute_unsafe
                or p99_delta > float(args.tail_safety_p99_expansion_block_best_delta)
                or cvar_delta > float(args.tail_safety_cvar_expansion_block_best_delta)
            )
            reference_final_tail_gate.update(
                {
                    "status": "COMPLETE",
                    "p99_delta_deg": float(p99_delta),
                    "tail_cvar_delta_deg": float(cvar_delta),
                    "proxy_vaccept_delta": float(proxy_delta),
                    "absolute_unsafe": bool(absolute_unsafe),
                    "final_local_p95_deg": final_p95,
                    "final_local_p99_deg": final_p99,
                    "final_local_tail_cvar_deg": final_cvar,
                    "final_proxy_vaccept": final_proxy,
                    "blocks_final_export": bool(block_final),
                    "blocks_promotion": bool(block_promotion),
                }
            )
            if block_final:
                phase1_v2_final_blocked = True
                phase1_v2_reasons.append("REFERENCE_FINAL_TAIL_SAFETY_BLOCKS_FINAL")
            if block_promotion:
                phase1_v2_final_blocked = True
                phase1_v2_reasons.append("REFERENCE_FINAL_TAIL_SAFETY_BLOCKS_PROMOTION")
        elif phase1_v2_tail_machine is not None:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.append("REFERENCE_FINAL_TAIL_GEOMETRY_INCOMPLETE")
        else:
            reference_final_tail_gate.update(
                {
                    "status": "NOT_REQUIRED",
                    "blocks_final_export": False,
                    "blocks_promotion": False,
                }
            )
    except Exception as exc:
        reference_final_tail_gate["reason"] = str(exc)
        if phase1_v2_tail_machine is not None:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.append("REFERENCE_FINAL_TAIL_GEOMETRY_INCOMPLETE")
    if bool(getattr(args, "phase1_v2_hard_gates", False)):
        final_train_logs = (final_payload.get("stats", {}) or {}).get("train", {}) or {}
        p1_preexport_checks = {
            "SATELLITE_PROTOCOL_REQUIREMENT_FAILED": (
                not bool(args.sat_protocol_disjoint_required)
                or bool((args.sat_protocol_manifest or {}).get("disjoint", False))
            ),
            "DIRECT_METRIC_LOCAL_COMPONENTS_DISABLED": bool(args.direct_metric_domain_local_components)
            and bool(args.direct_metric_require_domain_local_components),
            "DIRECT_METRIC_LOCAL_COMPONENT_RUNTIME_INACTIVE": _log_value(
                final_train_logs, "train/dm_accept_domain_local_component_gate", 0.0
            )
            > 0.0
            and _log_value(final_train_logs, "train/dm_accept_global_ball_accept", 1.0) == 0.0,
            "LABELED_ZID_INVARIANCE_RUNTIME_INACTIVE": all(
                _log_value(final_train_logs, f"train/zid_{name}_invariance_active", 0.0) > 0.0
                for name in ("receiver", "day", "channel")
            ),
            "UNLABELED_ZID_INVARIANCE_RUNTIME_INACTIVE": _log_value(
                final_train_logs, "train/u_zid_invariance_active", 0.0
            )
            > 0.0,
            "UNLABELED_MULTIVIEW_LOCAL_ROUTING_INACTIVE": _log_value(
                final_train_logs, "train/u_quarantine_multiview_local_components", 0.0
            )
            > 0.0
            and _log_value(final_train_logs, "train/u_quarantine_global_component_fallback", 1.0)
            == 0.0,
            "ZID_LEAKAGE_PROBE_NOT_REQUIRED": bool(args.zid_leakage_probe_required),
            "ZID_LEAKAGE_PROBE_GATE_FAILED": not bool(leakage_decision["fired"]),
        }
        failed_p1_checks = [reason for reason, passed in p1_preexport_checks.items() if not passed]
        if failed_p1_checks:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.extend(failed_p1_checks)
        final_payload["p1_preexport_checks"] = p1_preexport_checks
    guard_state["phase1_v2_final_blocked"] = bool(phase1_v2_final_blocked)
    guard_state["reference_final_p99_delta_deg"] = reference_final_tail_gate.get("p99_delta_deg")
    guard_state["reference_final_tail_gate_status"] = reference_final_tail_gate.get("status")
    guard_state["zid_leakage_probe_fired"] = bool(leakage_decision["fired"])
    guard_state.update(
        {f"zid_leakage_{key}": value for key, value in leakage_decision["details"].items()}
    )
    if phase1_v2_reasons:
        guard_state["reason"] = ";".join(
            dict.fromkeys(
                [
                    part
                    for value in [str(guard_state.get("reason", "")), *phase1_v2_reasons]
                    for part in str(value).split(";")
                    if part
                ]
            )
        )
    final_payload["joint_guard"] = dict(guard_state)
    final_payload["checkpoint_status"] = {
        "state": "FINAL_ONLY",
        "checkpoint_safe": not bool(phase1_v2_final_blocked),
        "phase1_v2_final_blocked": bool(phase1_v2_final_blocked),
        "reason": str(guard_state.get("reason", "")),
    }
    final_payload["final_only_evidence"]["reference_to_final_tail_safety"] = reference_final_tail_gate
    save_payload(selected_checkpoint, final_payload)
    (out_dir / "reference_to_final_tail_safety.json").write_text(
        json.dumps(reference_final_tail_gate, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        frozen_eval = _run_final_heldout_evaluation(
            args,
            model,
            data_ctx,
            device,
            selected_checkpoint,
        )
    except Exception as exc:
        frozen_eval = {
            "status": "FAILED",
            "reason": str(exc),
            "selection_source": selected_source,
            "checkpoint": str(selected_checkpoint),
            "claim": "PHASE1_FROZEN_HELDOUT_EVAL_INCOMPLETE",
        }
    heldout_eval_path = out_dir / (
        "external_final_eval_pending.json"
        if str(frozen_eval.get("status", "")).upper() == "DELEGATED_TO_MUSE_LAUNCHER"
        else "frozen_phase1_heldout_eval.json"
    )
    heldout_eval_path.write_text(
        json.dumps(frozen_eval, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    export_status: Dict[str, Any] = {"status": "NOT_REQUESTED"}
    exit_code = 0
    if should_skip_phase1_v2_final_export is not None and should_skip_phase1_v2_final_export(
        phase1_v2_final_blocked=bool(phase1_v2_final_blocked),
        tail_stop_blocks_final=bool(getattr(args, "tail_stop_blocks_final", True)),
    ):
        print(
            "[PHASE1-V2] prototype_export_skipped=1 reason=phase1_v2_guard_blocks_final "
            "claim=NON_PROMOTABLE_DIAGNOSTIC",
            flush=True,
        )
        export_status = {
            "status": "SKIPPED_FAIL_CLOSED",
            "reason": str((guard_state or {}).get("reason", "phase1_v2_guard_blocks_final")),
        }
        if bool(getattr(args, "phase1_export_diagnostic_on_block", False)):
            try:
                diagnostic_package = _maybe_export_phase2_prototypes_ssdg(
                    args,
                    model,
                    data_ctx,
                    device,
                    default_checkpoint=selected_checkpoint,
                    diagnostic_only=True,
                )
                if diagnostic_package is not None:
                    diagnostic_paths = diagnostic_package.get("paths", {}) or {}
                    export_status.update(
                        {
                            "status": "DIAGNOSTIC_COMPLETE",
                            "diagnostic_only": True,
                            "promotion_ready": False,
                            "prototype_path": diagnostic_paths.get("pt_path", ""),
                            "prototype_json_path": diagnostic_paths.get("json_path", ""),
                            "endpoint_artifact_ready": False,
                            "true_unknown_validated": False,
                        }
                    )
            except Exception as exc:
                export_status.update(
                    {
                        "status": "DIAGNOSTIC_FAILED",
                        "diagnostic_only": True,
                        "diagnostic_reason": str(exc),
                    }
                )
    else:
        try:
            default_export_checkpoint = selected_checkpoint
            exported_package = _maybe_export_phase2_prototypes_ssdg(
                args,
                model,
                data_ctx,
                device,
                default_checkpoint=default_export_checkpoint,
            )
            if exported_package is not None:
                endpoint_manifest = exported_package.get("endpoint_accept_v1", {})
                export_status = {
                    "status": "COMPLETE",
                    "prototype_path": (exported_package.get("paths", {}) or {}).get("pt_path", ""),
                    "prototype_json_path": (exported_package.get("paths", {}) or {}).get("json_path", ""),
                    "endpoint_artifact_ready": bool(endpoint_manifest),
                    "endpoint_boundary_version": endpoint_manifest.get("boundary_version", ""),
                    "endpoint_boundary_hash": endpoint_manifest.get("boundary_hash", ""),
                    "source_checkpoint_sha256": (
                        endpoint_manifest.get("inference_identity", {}) or {}
                    ).get("source_checkpoint_sha256", ""),
                    "true_unknown_validated": False,
                }
        except Exception as exc:
            level = "ERROR" if bool(getattr(args, "endpoint_require_artifact_on_export", True)) else "WARN"
            print(f"[{level}] SSDG Phase2 prototype export failed: {exc}", flush=True)
            if bool(getattr(args, "phase2_export_prototypes", False)) and bool(
                getattr(args, "endpoint_require_artifact_on_export", True)
            ):
                exit_code = 3
            export_status = {"status": "FAILED", "reason": str(exc)}
    selected_checkpoint_exists = selected_checkpoint.is_file()
    selected_checkpoint_evidence: Mapping[str, Any] = {}
    selected_train_logs: Mapping[str, Any] = {}
    if selected_checkpoint_exists:
        try:
            selected_checkpoint_evidence = _validate_phase1_checkpoint_payload(
                load_checkpoint(str(selected_checkpoint), torch.device("cpu")),
                args,
                selected_checkpoint,
            )
            selected_train_logs = (
                (selected_checkpoint_evidence.get("stats", {}) or {}).get("train", {}) or {}
            )
        except Exception:
            selected_checkpoint_evidence = {}
            selected_train_logs = {}
    selected_checkpoint_sha256 = _sha256_file(selected_checkpoint) if selected_checkpoint_exists else ""
    p0_mechanism_flags = {
        "run_id_present": bool(str(getattr(args, "run_id", "")).strip()),
        "candidate_id_present": bool(str(getattr(args, "candidate_id", "")).strip()),
        "phase1_v2_hard_gates": bool(getattr(args, "phase1_v2_hard_gates", False)),
        "tail_safety_state_machine": bool(getattr(args, "tail_safety_state_machine", False)),
        "tail_final_only_policy_valid": (
            str(getattr(args, "checkpoint_selection", "")) == "final_only"
            and not bool(getattr(args, "tail_rollback_enabled", False))
        ),
        "os_budget_controller": bool(getattr(args, "os_budget_controller", False)),
        "os_budget_target_positive": float(getattr(args, "os_eff_min_budget", 0.0)) > 0.0,
        "os_gradient_surgery": bool(getattr(args, "os_gradient_surgery", False)),
        "u_tri_state_required": bool(getattr(args, "u_tri_state_required", False)),
        "u_geometry_all_valid_queries": bool(getattr(args, "u_geometry_all_valid_queries", False)),
        "source_episode_density_gate": bool(getattr(args, "source_episode_density_gate", False)),
        "feasibility_gate": bool(getattr(args, "feasibility_gate", False)),
        "direct_metric_multiview_separate": bool(getattr(args, "direct_metric_multiview_separate", False)),
        "concat_sat_channel_aug": bool(getattr(args, "use_concat_sat_channel_aug", False)),
        "rho_label_protocol": float((data_ctx.get("split_info", {}) or {}).get("rho_label", 1.0)) <= 0.1 + 1e-8,
        "direct_metric_loss_active": float(getattr(args, "lambda_direct_metric_accept", 0.0)) > 0.0,
        "source_episode_loss_active": float(getattr(args, "lambda_source_episode", 0.0)) > 0.0,
        "proxy_unknown_loss_active": float(getattr(args, "lambda_proxy_unknown", 0.0)) > 0.0,
        "u_direct_metric_loss_active": float(getattr(args, "lambda_u_direct_metric_accept", 0.0)) > 0.0,
        "u_quarantine_loss_active": float(getattr(args, "lambda_u_quarantine_accept", 0.0)) > 0.0,
        "u_domain_loss_active": float(getattr(args, "lambda_u_domain", 0.0)) > 0.0,
        "u_adv_loss_active": float(getattr(args, "lambda_u_adv", 0.0)) > 0.0,
        "u_sat_consistency_active": float(getattr(args, "lambda_u_sat_cons", 0.0)) > 0.0,
        "selected_checkpoint_evidence_bound": bool(selected_checkpoint_evidence),
        "direct_metric_runtime_active": _log_value(selected_train_logs, "train/dm_accept_active", 0.0) > 0.0,
        "source_episode_runtime_active": _log_value(
            selected_train_logs, "train/source_episode_local_component_structural_active", 0.0
        ) > 0.0,
        "proxy_unknown_runtime_active": _log_value(selected_train_logs, "train/proxy_unknown_active", 0.0) > 0.0,
        "u_direct_metric_runtime_active": _log_value(selected_train_logs, "train/u_dm_accept_active", 0.0) > 0.0,
        "u_quarantine_runtime_active": _log_value(selected_train_logs, "train/u_quarantine_active", 0.0) > 0.0,
        "open_gradient_runtime_active": _log_value(
            selected_train_logs, "train/os_gradient_effective_open_norm", 0.0
        ) > 0.0,
        "open_gradient_budget_met": (
            _log_value(selected_train_logs, "train/os_budget_controller_post", 0.0) + 1e-8
            >= float(getattr(args, "os_eff_min_budget", 0.0))
            and (
                float(getattr(args, "os_eff_max_budget", 0.0)) <= 0.0
                or _log_value(selected_train_logs, "train/os_budget_controller_post", 1.0)
                <= float(getattr(args, "os_eff_max_budget", 0.0)) + 1e-8
            )
        ),
        "phase2_export_prototypes": bool(getattr(args, "phase2_export_prototypes", False)),
        "phase2_fuse_prototypes": bool(getattr(args, "phase2_fuse_prototypes", False)),
        "endpoint_artifact_required": bool(getattr(args, "endpoint_require_artifact_on_export", False)),
        "endpoint_checkpoint_identity_match": bool(selected_checkpoint_sha256)
        and str(export_status.get("source_checkpoint_sha256", "")) == selected_checkpoint_sha256,
    }
    p0_mechanisms_ready = all(p0_mechanism_flags.values())
    legacy_p0_mechanism_flags = dict(p0_mechanism_flags)
    probe_from_checkpoint = (
        (selected_checkpoint_evidence.get("stats", {}) or {}).get("zid_leakage_probe", {})
        if selected_checkpoint_evidence
        else {}
    )
    sat_disjoint_required = bool(getattr(args, "sat_protocol_disjoint_required", False))
    sat_protocol_disjoint = bool((getattr(args, "sat_protocol_manifest", {}) or {}).get("disjoint", False))
    p1_mechanism_flags = {
        "checkpoint_selection_final_only": str(getattr(args, "checkpoint_selection", "")) == "final_only",
        "selected_checkpoint_is_final": selected_checkpoint.name == "final_ssdg.pth",
        "selected_checkpoint_role_final": str(selected_checkpoint_evidence.get("checkpoint_role", ""))
        == "training_final_only",
        "sat_protocol_requirement_satisfied": (
            sat_protocol_requirement_satisfied(
                required=sat_disjoint_required,
                actual_disjoint=sat_protocol_disjoint,
            )
            if sat_protocol_requirement_satisfied is not None
            else ((not sat_disjoint_required) or sat_protocol_disjoint)
        ),
        "direct_metric_domain_local_components": bool(
            getattr(args, "direct_metric_domain_local_components", False)
        ),
        "direct_metric_local_components_required": bool(
            getattr(args, "direct_metric_require_domain_local_components", False)
        ),
        "direct_metric_runtime_local_component_gate": _log_value(
            selected_train_logs, "train/dm_accept_domain_local_component_gate", 0.0
        )
        > 0.0,
        "direct_metric_runtime_global_ball_disabled": _log_value(
            selected_train_logs, "train/dm_accept_global_ball_accept", 1.0
        )
        == 0.0,
        "labeled_receiver_invariance_active": _log_value(
            selected_train_logs, "train/zid_receiver_invariance_active", 0.0
        )
        > 0.0,
        "labeled_day_invariance_active": _log_value(
            selected_train_logs, "train/zid_day_invariance_active", 0.0
        )
        > 0.0,
        "labeled_channel_invariance_active": _log_value(
            selected_train_logs, "train/zid_channel_invariance_active", 0.0
        )
        > 0.0,
        "unlabeled_invariance_configured": any(
            float(getattr(args, key, 0.0)) > 0.0
            for key in (
                "lambda_u_zid_receiver_invariance",
                "lambda_u_zid_day_invariance",
                "lambda_u_zid_channel_invariance",
            )
        ),
        "unlabeled_invariance_runtime_active": _log_value(
            selected_train_logs, "train/u_zid_invariance_active", 0.0
        )
        > 0.0,
        "unlabeled_multiview_local_routing_active": _log_value(
            selected_train_logs, "train/u_quarantine_multiview_local_components", 0.0
        )
        > 0.0,
        "unlabeled_global_component_fallback_disabled": _log_value(
            selected_train_logs, "train/u_quarantine_global_component_fallback", 1.0
        )
        == 0.0,
        "zid_leakage_probe_required": bool(getattr(args, "zid_leakage_probe_required", False)),
        "zid_leakage_probe_complete": str(probe_from_checkpoint.get("status", "")) == "COMPLETE",
        "zid_leakage_probe_gate_passed": not bool(leakage_decision["fired"]),
    }
    p1_mechanisms_ready = all(p1_mechanism_flags.values())
    legacy_p1_mechanism_flags = dict(p1_mechanism_flags)
    if bool(getattr(args, "formal_ablation", False)):
        p0_mechanism_flags, p1_mechanism_flags = (
            _formal_ablation_terminal_flags(
                args,
                selected_checkpoint=selected_checkpoint,
                selected_checkpoint_evidence=selected_checkpoint_evidence,
                selected_checkpoint_sha256=selected_checkpoint_sha256,
                export_status=export_status,
                source_split_receipt=(
                    data_ctx.get("split_info", {}) or {}
                ).get("source_split_receipt", {}),
            )
        )
        p0_mechanisms_ready = all(p0_mechanism_flags.values())
        p1_mechanisms_ready = all(p1_mechanism_flags.values())
    endpoint_export_ready = bool(
        export_status.get("status") == "COMPLETE" and export_status.get("endpoint_artifact_ready", False)
    )
    external_final_eval = _external_final_eval_requested(args)
    terminal_status = _resolve_phase1_terminal_status(
        tail_stopped=bool(tail_early_stop_requested),
        export_failed=bool(exit_code),
        final_blocked=bool(phase1_v2_final_blocked),
        selected_checkpoint_exists=bool(selected_checkpoint_exists),
        heldout_eval_status=str(frozen_eval.get("status", "")),
        external_final_eval=bool(external_final_eval),
        p0_mechanisms_ready=bool(p0_mechanisms_ready),
        p1_mechanisms_ready=bool(p1_mechanisms_ready),
        endpoint_export_ready=bool(endpoint_export_ready),
        mechanism_gates_required=not external_final_eval,
        endpoint_export_required=(
            bool(getattr(args, "endpoint_require_artifact_on_export", True))
            and not external_final_eval
        ),
    )
    terminal_exit_code = int(exit_code)
    if terminal_exit_code == 0 and terminal_status != "COMPLETE":
        terminal_exit_code = {
            "STOPPED_TAIL": 4,
            "NON_PROMOTABLE_GUARD_BLOCKED": 5,
            "NO_SAFE_CHECKPOINT": 6,
            "HELDOUT_EVAL_INCOMPLETE": 7,
            "NON_PROMOTABLE_P0_DISABLED": 8,
            "NON_PROMOTABLE_P1_DISABLED": 11,
            "NON_PROMOTABLE_ENDPOINT_NOT_EXPORTED": 9,
        }.get(terminal_status, 10)
    resource_summary = {
        "schema": "cvs.phase1.resource_summary.v1",
        "run_id": str(getattr(args, "run_id", "")),
        "row_key": str(getattr(args, "row_key", "")),
        "ablation_id": str(getattr(args, "ablation_id", "")),
        "train_seed": int(getattr(args, "seed", -1)),
        "wall_time_seconds": float(
            time.time() - training_wall_started
        ),
        "trainable_parameter_count": int(trainable_params),
        "total_parameter_count": int(total_params),
        "device_type": str(device.type),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        ),
        "resource_claim": (
            "THROUGHPUT_SCHEDULE_OBSERVATION_NOT_ISOLATED_LATENCY"
        ),
    }
    resource_summary_path = out_dir / "phase1_resource_summary.json"
    resource_summary_path.write_text(
        json.dumps(
            resource_summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    terminal_manifest = {
        "schema": "phase1_terminal_status_v2",
        "run_id": str(getattr(args, "run_id", "")),
        "candidate_id": str(getattr(args, "candidate_id", "")),
        "ablation_id": str(getattr(args, "ablation_id", "")),
        "ablation_config_hash": str(
            getattr(args, "ablation_config_hash", "")
        ),
        "git_commit": str(getattr(args, "git_commit", "")),
        "row_key": str(getattr(args, "row_key", "")),
        "sealed_plan_sha256": str(
            getattr(args, "sealed_plan_sha256", "")
        ),
        "seed_registry_sha256": str(
            getattr(args, "seed_registry_sha256", "")
        ),
        "wisig_pkl_sha256": str(
            getattr(args, "wisig_pkl_sha256", "")
        ),
        "dataset_receipt_sha256": str(
            getattr(args, "dataset_receipt_sha256", "")
        ),
        "environment_receipt_sha256": str(
            getattr(args, "environment_receipt_sha256", "")
        ),
        "source_split_receipt": (
            data_ctx.get("split_info", {}) or {}
        ).get("source_split_receipt", {}),
        "dataset_receipt": formal_dataset_receipt,
        "environment_receipt": formal_environment_receipt,
        "status": terminal_status,
        "exit_code": int(terminal_exit_code),
        "selection_source": selected_source,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_exists": bool(selected_checkpoint_exists),
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "best_epoch": int(best_epoch),
        "best_source_val_tx": float(best_val) if math.isfinite(float(best_val)) else None,
        "heldout_eval": frozen_eval,
        "heldout_eval_path": str(heldout_eval_path),
        "heldout_eval_sha256": _sha256_file(heldout_eval_path),
        "tail_reference_path": "METRIC_ONLY",
        "tail_reference_exists": bool(tail_reference_geometry),
        "tail_reference_sha256": "",
        "tail_reference_epoch": int(tail_reference_epoch),
        "tail_rollback_count": int(phase1_v2_tail_machine.rollback_count) if phase1_v2_tail_machine is not None else 0,
        "tail_rollback_events": tail_rollback_events,
        "reference_to_final_tail_safety": reference_final_tail_gate,
        "satellite_protocol": dict(getattr(args, "sat_protocol_manifest", {}) or {}),
        "zid_leakage_probe": final_zid_leakage_probe,
        "phase1_v2_final_blocked": bool(phase1_v2_final_blocked),
        "final_guard_reason": str((guard_state or {}).get("reason", "")),
        "prototype_export": export_status,
        "resource_summary": resource_summary,
        "p0_mechanism_flags": p0_mechanism_flags,
        "legacy_p0_mechanism_flags_diagnostic_only": legacy_p0_mechanism_flags,
        "p0_mechanism_evidence_checkpoint_epoch": int(
            selected_checkpoint_evidence.get("epoch", -1)
        ) if selected_checkpoint_evidence else -1,
        "p0_mechanisms_ready": bool(p0_mechanisms_ready),
        "p1_mechanism_flags": p1_mechanism_flags,
        "legacy_p1_mechanism_flags_diagnostic_only": legacy_p1_mechanism_flags,
        "p1_mechanism_facts": {
            "sat_protocol_disjoint_required": sat_disjoint_required,
            "sat_protocol_disjoint": sat_protocol_disjoint,
        },
        "p1_mechanisms_ready": bool(p1_mechanisms_ready),
        "endpoint_export_ready": bool(endpoint_export_ready),
        "promotion_ready": terminal_status == "COMPLETE",
        "claim": "PHASE1_SOURCE_ONLY_NO_TRUE_UNKNOWN_SUCCESS_CLAIM",
    }
    (out_dir / "phase1_terminal_status.json").write_text(
        json.dumps(terminal_manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    terminal_manifest_path = out_dir / "phase1_terminal_status.json"
    prototype_paths = {
        key: str(export_status.get(key, "") or "")
        for key in ("prototype_path", "prototype_json_path")
    }
    prototype_hashes = {
        key: _sha256_file(path)
        for key, path in prototype_paths.items()
        if path and Path(path).is_file()
    }
    completion_receipt = {
        "schema": "cvs.phase1.training_completion_receipt.v1",
        "run_id": str(getattr(args, "run_id", "")),
        "row_key": str(getattr(args, "row_key", "")),
        "ablation_id": str(getattr(args, "ablation_id", "")),
        "train_seed": int(getattr(args, "seed", -1)),
        "git_commit": str(getattr(args, "git_commit", "")),
        "sealed_plan_sha256": str(
            getattr(args, "sealed_plan_sha256", "")
        ),
        "seed_registry_sha256": str(
            getattr(args, "seed_registry_sha256", "")
        ),
        "wisig_pkl_sha256": str(
            getattr(args, "wisig_pkl_sha256", "")
        ),
        "dataset_receipt_sha256": str(
            getattr(args, "dataset_receipt_sha256", "")
        ),
        "environment_receipt_sha256": str(
            getattr(args, "environment_receipt_sha256", "")
        ),
        "method_config_hash": str(
            getattr(args, "ablation_method_config_hash", "")
        ),
        "resolved_config_hash": str(
            getattr(args, "ablation_config_hash", "")
        ),
        "source_split_receipt": (
            data_ctx.get("split_info", {}) or {}
        ).get("source_split_receipt", {}),
        "dataset_receipt": formal_dataset_receipt,
        "environment_receipt": formal_environment_receipt,
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "terminal_manifest_sha256": _sha256_file(
            terminal_manifest_path
        ),
        "prototype_paths": prototype_paths,
        "prototype_hashes": prototype_hashes,
        "heldout_eval_path": str(heldout_eval_path),
        "heldout_eval_sha256": _sha256_file(heldout_eval_path),
        "resource_summary_sha256": _sha256_file(
            resource_summary_path
        ),
        "resource_summary": resource_summary,
        "terminal_status": terminal_status,
        "exit_code": int(terminal_exit_code),
        "phase1_training_complete": terminal_status == "COMPLETE",
        "deployment_bundle_status": "PENDING_PHASE1_W2_SEAL",
        "formal_performance_claim": False,
        "claim": "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT",
    }
    (out_dir / "phase1_training_completion_receipt.json").write_text(
        json.dumps(
            completion_receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"[PHASE1-TERMINAL] status={terminal_status} exit_code={int(terminal_exit_code)} "
        f"promotion_ready={int(terminal_status == 'COMPLETE')} "
        f"endpoint_export_ready={int(endpoint_export_ready)}",
        flush=True,
    )
    return int(terminal_exit_code)


def _bicad_xr_jsonable(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        resolved = detached.item() if detached.numel() == 1 else detached.tolist()
        return _bicad_xr_jsonable(resolved)
    if isinstance(value, Mapping):
        return {str(key): _bicad_xr_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bicad_xr_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _bicad_xr_mean_epoch_loss(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("BiCAD-XR epoch loss requires at least one training step")
    mean_loss = float(sum(values) / len(values))
    if not math.isfinite(mean_loss):
        raise FloatingPointError("non-finite BiCAD-XR train loss")
    return mean_loss


def _bicad_xr_metadata(extra, key: str, device, expected_count: int) -> torch.Tensor:
    value = _metadata_label_tensor(extra, key, device, expected_count)
    if value is None:
        return torch.zeros(expected_count, dtype=torch.long, device=device)
    return value.to(device=device, dtype=torch.long).reshape(expected_count)


def _bicad_xr_required_metadata(extra, key: str, device, expected_count: int) -> torch.Tensor:
    value = _metadata_label_tensor(extra, key, device, expected_count)
    if value is None:
        raise ValueError(f"BiCAD-XR requires source-only {key} metadata for every row")
    return value.to(device=device, dtype=torch.long).reshape(expected_count)


def _move_bicad_xr_unlabeled_batch(batch, device):
    """Move U_s IQ/domain/metadata without reading or materializing TX labels."""

    if not isinstance(batch, (tuple, list)):
        raise ValueError("BiCAD-XR U_s batch must be a tuple or list")
    if len(batch) >= 4:
        x_u = batch[0]
        domain_u = batch[2]
        metadata_u = batch[3]
    elif len(batch) == 3:
        x_u = batch[0]
        domain_u = batch[1]
        metadata_u = batch[2]
    else:
        raise ValueError("BiCAD-XR U_s batch must contain IQ, domain and metadata")
    if not torch.is_tensor(x_u):
        raise TypeError("BiCAD-XR U_s IQ batch must be a tensor")
    x_u = _safe_iq_tensor(x_u.to(device, non_blocking=True))
    if torch.is_tensor(domain_u):
        domain_u = domain_u.to(device, non_blocking=True)
    return x_u, (domain_u, metadata_u)


def _bicad_xr_local_domain_labels(
    labels: torch.Tensor,
    source_indices: Sequence[int],
    *,
    name: str,
) -> torch.Tensor:
    """Map raw WiSig domain indices to contiguous labels for local heads."""

    resolved = [int(value) for value in source_indices]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError(f"BiCAD-XR {name} source indices must be unique and non-empty")
    raw = labels.to(dtype=torch.long)
    local = torch.full_like(raw, -1)
    for local_index, source_index in enumerate(resolved):
        local = torch.where(
            raw == int(source_index),
            torch.as_tensor(local_index, dtype=torch.long, device=raw.device),
            local,
        )
    if bool((local < 0).any()):
        unexpected = sorted(set(raw[local < 0].detach().cpu().tolist()))
        raise ValueError(
            f"BiCAD-XR {name} labels are outside the frozen source set: {unexpected}"
        )
    return local


def _apply_bicad_xr_entry_protocol(args, protocol: BiCADXRProtocol) -> None:
    args.phase1_method = "bicad_xr"
    args.candidate_id = protocol.candidate_id
    args.phase1_source_only_eval = True
    args.use_concat_sat_channel_aug = True
    args.concat_sat_ce_only = True
    args.concat_sat_ce_weight = protocol.lambda_sat_cls_start
    args.concat_sat_start_epoch = protocol.concat_sat_start_epoch
    args.sat_train_scenarios = _BICAD_XR_LEO_WEAK_CSV
    args.sat_train_scenario_list = list(_BICAD_XR_LEO_WEAK)
    args.sat_view_schedule = _BICAD_XR_SAT_VIEW_SCHEDULE
    args.sat_training_mode = protocol.satellite_supervision_mode
    args.lambda_sat_cls = 0.68
    args.lambda_sat_cons = 0.0
    args.strict_pair_concat = protocol.strict_pair_concat
    args.pair_identity = protocol.pair_identity
    args.pair_vicreg = protocol.pair_vicreg
    args.pair_delta = protocol.pair_delta
    args.dynamic_adversarial_dose = protocol.dynamic_adversarial_dose
    args.pair_projector_dim = protocol.pair_projector_dim
    args.factor_interaction_dim = protocol.factor_interaction_dim
    args.lambda_sat_cls_start = protocol.lambda_sat_cls_start
    args.lambda_sat_cls_end = protocol.lambda_sat_cls_end
    args.use_mixstyle = False
    args.muse_lr_schedule = "off"
    if hasattr(args, "use_fasttrust"):
        args.use_fasttrust = False


def _apply_bicad_xr_model_defaults(model_args) -> None:
    """Resolve model defaults that checkpoint merging normally supplies."""

    if float(getattr(model_args, "sample_rate_hz", 0.0)) <= 0.0:
        model_args.sample_rate_hz = (
            25e6
            if str(getattr(model_args, "dataset", "wisig")).lower() == "wisig"
            else 5e6
        )


def _build_bicad_xr_concat_augmenter(args):
    if ConcatSatChannelAugment is None or apply_sat_channel_for_scenario is None:
        raise ImportError("BiCAD-XR requires ConcatSatChannelAugment and LEO weak channels")
    return ConcatSatChannelAugment(
        scenarios=list(_BICAD_XR_LEO_WEAK),
        schedule=_BICAD_XR_SAT_VIEW_SCHEDULE,
        p=1.0,
        seed=int(getattr(args, "sat_view_seed", getattr(args, "seed", 0))),
        apply_fn=apply_sat_channel_for_scenario,
    )


class _BiCADXRConcatForward(torch.nn.Module):
    """Expose clean/satellite pair tensors from one concatenated model forward."""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self._clean_count: Optional[int] = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base_model"), name)

    def set_concat_clean_count(self, clean_count: Optional[int]) -> None:
        self._clean_count = None if clean_count is None else int(clean_count)

    def forward(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)
        clean_count = self._clean_count
        self._clean_count = None
        if clean_count is None:
            return output
        if not isinstance(output, Mapping):
            raise ValueError("BiCAD-XR concat model must return a mapping")
        logits = output.get("tx_logits", output.get("logits"))
        z_id = output.get("z_id", output.get("identity_features"))
        if not torch.is_tensor(logits) or not torch.is_tensor(z_id):
            raise ValueError("BiCAD-XR concat forward requires tx_logits and z_id")
        if logits.size(0) != 2 * clean_count or z_id.size(0) != 2 * clean_count:
            raise ValueError("BiCAD-XR concat forward must contain clean then satellite rows")
        resolved = dict(output)
        existing_pair = output.get("concat_pair")
        pair_payload: Dict[str, Any] = (
            dict(existing_pair) if isinstance(existing_pair, Mapping) else {}
        )
        pair_payload.update(
            {
                "clean_z_id": z_id[:clean_count],
                "satellite_z_id": z_id[clean_count:],
                "clean_logits": logits[:clean_count],
                "satellite_logits": logits[clean_count:],
            }
        )
        z_dom = output.get("z_dom", output.get("domain_features"))
        if torch.is_tensor(z_dom):
            if z_dom.size(0) != 2 * clean_count:
                raise ValueError("BiCAD-XR concat z_dom must contain clean then satellite rows")
            pair_payload["clean_z_dom"] = z_dom[:clean_count]
            pair_payload["satellite_z_dom"] = z_dom[clean_count:]
        for name in ("z_r", "z_d", "z_c", "z_int"):
            factor = output.get(name)
            if torch.is_tensor(factor):
                if factor.size(0) != 2 * clean_count:
                    raise ValueError(f"BiCAD-XR concat {name} must contain clean then satellite rows")
                pair_payload[f"clean_{name}"] = factor[:clean_count]
                pair_payload[f"satellite_{name}"] = factor[clean_count:]
        factor_outputs = output.get("factor_outputs", output.get("domain_factors"))
        if factor_outputs is not None and not isinstance(factor_outputs, Mapping):
            factor_outputs = {
                name: getattr(factor_outputs, name)
                for name in ("z_r", "z_d", "z_c", "z_int")
                if torch.is_tensor(getattr(factor_outputs, name, None))
            }
        if isinstance(factor_outputs, Mapping):
            for name, factor in factor_outputs.items():
                if not torch.is_tensor(factor):
                    continue
                if factor.size(0) != 2 * clean_count:
                    raise ValueError(
                        f"BiCAD-XR concat factor {name} must contain clean then satellite rows"
                    )
                pair_payload[f"clean_{name}"] = factor[:clean_count]
                pair_payload[f"satellite_{name}"] = factor[clean_count:]
        resolved["concat_pair"] = pair_payload
        return resolved


def _bicad_xr_labeled_step(
    trainer,
    concat_sat_aug,
    x,
    tx,
    receiver,
    day,
    x_u=None,
    receiver_u=None,
    day_u=None,
    *,
    args,
    epoch: int,
    batch_idx: int,
    update: int,
    total_updates: int,
):
    from cvsrffi.phase1_bicad_xr.trainer import BiCADXRBatch

    is_pairbicad = bool(getattr(trainer.config, "strict_pair_concat", False))
    if is_pairbicad:
        tx = torch.as_tensor(tx, device=x.device, dtype=torch.long).reshape(-1)
        receiver = torch.as_tensor(receiver, device=x.device, dtype=torch.long).reshape(-1)
        day = torch.as_tensor(day, device=x.device, dtype=torch.long).reshape(-1)
        if x_u is None or receiver_u is None or day_u is None:
            raise ValueError("strict PairBiCAD requires both labeled and unlabeled source batches")
        x = _safe_iq_tensor(x)
        x_u = _safe_iq_tensor(torch.as_tensor(x_u, device=x.device))
        receiver_u = torch.as_tensor(receiver_u, device=x.device, dtype=torch.long).reshape(-1)
        day_u = torch.as_tensor(day_u, device=x.device, dtype=torch.long).reshape(-1)
        if (
            tx.numel() != 16
            or x.size(0) != 16
            or x_u.size(0) != 32
            or receiver.numel() != 16
            or day.numel() != 16
            or receiver_u.numel() != 32
            or day_u.numel() != 32
        ):
            raise ValueError(
                "strict PairBiCAD requires exactly 16 labeled and 32 unlabeled physical samples"
            )
        if x.ndim < 2 or x_u.ndim != x.ndim or tuple(x.shape[1:]) != tuple(x_u.shape[1:]):
            raise ValueError("strict PairBiCAD labeled and unlabeled IQ shapes must align")
        physical_x = torch.cat((x, x_u), dim=0)
        physical_receiver = torch.cat((receiver, receiver_u), dim=0)
        physical_day = torch.cat((day, day_u), dim=0)
        clean_count = int(physical_x.size(0))
        sat_view = concat_sat_aug.transform(
            physical_x,
            args=args,
            epoch=int(epoch),
            batch_idx=int(batch_idx),
        )
        satellite_x = _safe_iq_tensor(sat_view.x)
        if satellite_x.size(0) != clean_count:
            raise ValueError("strict PairBiCAD satellite view must preserve physical batch size")
        concat_x = torch.cat((physical_x, satellite_x), dim=0)
        concat_receiver = torch.cat((physical_receiver, physical_receiver), dim=0)
        concat_day = torch.cat((physical_day, physical_day), dim=0)
        concat_channel = torch.cat(
            (
                torch.zeros(clean_count, dtype=torch.long, device=x.device),
                torch.ones(clean_count, dtype=torch.long, device=x.device),
            ),
            dim=0,
        )
        labeled_mask = torch.zeros(2 * clean_count, dtype=torch.bool, device=x.device)
        labeled_mask[: tx.numel()] = True
        if not isinstance(trainer.model, _BiCADXRConcatForward):
            raise ValueError("BiCAD-XR trainer requires the concat forward adapter")
        trainer.model.set_concat_clean_count(clean_count)
        batch = BiCADXRBatch(
            x=concat_x,
            tx=None,
            labeled_tx=tx,
            receiver=concat_receiver,
            day=concat_day,
            channel=concat_channel,
            labeled_mask=labeled_mask,
            pair_tx=tx,
            epoch=int(epoch),
        )
        output = trainer.compute_step(
            batch,
            update=int(update),
            total_updates=int(total_updates),
            epoch=int(epoch),
        )
        view = {
            "applied": bool(sat_view.applied),
            "scenario": str(sat_view.scenario),
            "clean_batch_size": clean_count,
            "total_batch_size": 2 * clean_count,
            "physical_batch_size": clean_count,
            "labeled_count": int(tx.numel()),
            "unlabeled_count": int(x_u.size(0)),
            "network_batch_size": int(concat_x.size(0)),
            "satellite_x": satellite_x,
        }
        return output, view

    if int(epoch) < 80:
        raise ValueError("BiCAD-XR concat step starts at epoch 80")
    if x_u is not None or receiver_u is not None or day_u is not None:
        raise ValueError("legacy BiCAD-XR concat step does not accept an unlabeled pair")
    clean_count = int(tx.numel())
    sat_view = concat_sat_aug.transform(
        x,
        args=args,
        epoch=int(epoch),
        batch_idx=int(batch_idx),
    )
    satellite_x = _safe_iq_tensor(sat_view.x)
    concat_x = torch.cat((x, satellite_x), dim=0)
    concat_tx = torch.cat((tx, tx), dim=0)
    concat_receiver = torch.cat((receiver, receiver), dim=0)
    concat_day = torch.cat((day, day), dim=0)
    concat_channel = torch.cat(
        (
            torch.zeros(clean_count, dtype=torch.long, device=x.device),
            torch.ones(clean_count, dtype=torch.long, device=x.device),
        ),
        dim=0,
    )
    labeled_mask = torch.zeros(2 * clean_count, dtype=torch.bool, device=x.device)
    labeled_mask[:clean_count] = True
    if not isinstance(trainer.model, _BiCADXRConcatForward):
        raise ValueError("BiCAD-XR trainer requires the concat forward adapter")
    trainer.model.set_concat_clean_count(clean_count)
    batch = BiCADXRBatch(
        x=concat_x,
        tx=concat_tx,
        receiver=concat_receiver,
        day=concat_day,
        channel=concat_channel,
        labeled_mask=labeled_mask,
        pair_tx=tx,
        epoch=int(epoch),
    )
    output = trainer.compute_step(
        batch,
        update=int(update),
        total_updates=int(total_updates),
        epoch=int(epoch),
    )
    view = {
        "applied": bool(sat_view.applied),
        "scenario": str(sat_view.scenario),
        "clean_batch_size": clean_count,
        "total_batch_size": 2 * clean_count,
        "satellite_x": satellite_x,
    }
    return output, view


def _load_bicad_xr_checkpoint_strict(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    trainer,
) -> Dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("BiCAD-XR checkpoint must be a mapping")
    model_state = checkpoint.get("model")
    runtime = checkpoint.get("bicad_xr_runtime")
    if not isinstance(model_state, Mapping):
        raise ValueError("BiCAD-XR checkpoint is missing model state")
    if not isinstance(runtime, Mapping):
        raise ValueError("BiCAD-XR checkpoint is missing bicad_xr_runtime")
    model.load_state_dict(model_state, strict=True)
    trainer.load_checkpoint_runtime(runtime, strict=True)
    restored_runtime = deepcopy(dict(runtime))
    restored_runtime["strict_reconstruction"] = True
    return restored_runtime


def _train_bicad_xr(args) -> int:
    """Run the source-only BiCAD-XR main step without entering legacy SSDG."""

    from dataclasses import replace as dataclass_replace

    from cvsrffi.phase1_bicad_xr.config import candidate_config
    from cvsrffi.phase1_bicad_xr.trainer import BiCADXRBatch, BiCADXRTrainer

    protocol = resolve_bicad_protocol(args)
    if not isinstance(protocol, BiCADXRProtocol):
        raise ValueError("BiCAD-XR route requires --phase1_method bicad_xr")
    _apply_bicad_xr_entry_protocol(args, protocol)

    set_seed(int(getattr(args, "seed", 0)))
    device = resolve_device(str(getattr(args, "device", "auto")))
    pair_data_overrides = None
    if protocol.strict_pair_concat:
        if int(getattr(args, "batch_size", 0)) != 48:
            raise ValueError("PairBiCAD physical batch contract requires --batch_size 48")
        pair_data_overrides = {
            "batch_size": getattr(args, "batch_size", 48),
            "use_tx_rx_balanced_sampler": getattr(
                args, "use_tx_rx_balanced_sampler", False
            ),
            "use_muse_ssdg": getattr(args, "use_muse_ssdg", False),
            "muse_level": getattr(args, "muse_level", "M0"),
            "muse_unlabeled_batch_size": getattr(
                args, "muse_unlabeled_batch_size", 256
            ),
        }
        # The CLI batch is the frozen 48-physical contract.  The two source
        # loaders are built at 16L and 32U, then one step reassembles 48.
        args.batch_size = 16
        args.use_tx_rx_balanced_sampler = False
        args.use_muse_ssdg = True
        args.muse_level = "M1"
        args.muse_unlabeled_batch_size = 32
    try:
        data_ctx = _build_ssdg_wisig_data(args, device)
    finally:
        if pair_data_overrides is not None:
            for name, value in pair_data_overrides.items():
                setattr(args, name, value)
    train_loader = data_ctx["train_loader"]
    if train_loader is None:
        raise RuntimeError("BiCAD-XR requires the labeled source train loader")
    unlabeled_loader = data_ctx.get("unlabeled_loader")
    if protocol.strict_pair_concat and unlabeled_loader is None:
        raise RuntimeError("PairBiCAD requires the unlabeled source train loader")

    checkpoint = None
    baseline_ckpt = str(getattr(args, "baseline_ckpt", "") or "").strip()
    if baseline_ckpt and not bool(getattr(args, "from_scratch", True)):
        checkpoint = load_checkpoint(baseline_ckpt, device)
        model_args = merge_checkpoint_args(
            checkpoint,
            args,
            input_len=int(data_ctx["input_len"]),
            num_domains=int(data_ctx["num_domains"]),
        )
        _apply_model_cli_args(model_args, args)
    else:
        model_args = deepcopy(args)
        model_args.input_len = int(data_ctx["input_len"])
        model_args.num_domains = int(data_ctx["num_domains"])
        model_args.num_classes = int(len(data_ctx["class_id_to_tx"]))
    model_args.phase1_method = "bicad_xr"
    _apply_bicad_xr_model_defaults(model_args)
    model = build_baseline_model(model_args, device)

    base_config = candidate_config(protocol.candidate_id)
    if protocol.strict_pair_concat:
        config = dataclass_replace(
            base_config,
            phase1_method="bicad_xr",
            use_fasttrust=False,
            use_mixstyle=False,
            sat_train_scenarios=_BICAD_XR_LEO_WEAK,
        )
    else:
        config = dataclass_replace(
            base_config,
            phase1_method="bicad_xr",
            use_fasttrust=False,
            use_mixstyle=False,
            lambda_sat_cls=0.68,
            lambda_sat_cons=0.0,
            concat_sat_ce_only=True,
            concat_sat_start_epoch=80,
            sat_train_scenarios=_BICAD_XR_LEO_WEAK,
        )
    concat_model = _BiCADXRConcatForward(model)
    source_receiver_indices = list(data_ctx["source_receiver_indices"])
    source_day_indices = list(data_ctx["source_day_indices"])
    trainer = BiCADXRTrainer(
        concat_model,
        config,
        num_receivers=len(source_receiver_indices),
        num_days=len(source_day_indices),
        num_channels=2 if protocol.strict_pair_concat else 4,
    ).to(device)
    if checkpoint is not None:
        _load_bicad_xr_checkpoint_strict(
            checkpoint,
            model=model,
            trainer=trainer,
        )
    concat_sat_aug = _build_bicad_xr_concat_augmenter(args)
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=float(getattr(args, "lr", 2e-4)),
        weight_decay=float(getattr(args, "weight_decay", 1e-4)),
    )
    if checkpoint is not None and isinstance(checkpoint.get("optimizer"), Mapping):
        optimizer.load_state_dict(checkpoint["optimizer"])

    epochs = int(getattr(args, "epochs", 0))
    if epochs <= 0:
        epochs = 200
    total_updates = int(config.optimizer_updates)
    update = 0
    epoch_rows: List[Dict[str, Any]] = []
    last_audit: Mapping[str, Any] = {}
    unlabeled_iter = iter(unlabeled_loader) if protocol.strict_pair_concat else None
    trainer.train()
    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        for batch_idx, labeled_batch in enumerate(train_loader, start=1):
            if update >= total_updates:
                break
            x_l, y_l, extra_l = move_batch(labeled_batch, device)
            if y_l is None:
                if protocol.strict_pair_concat:
                    raise RuntimeError("PairBiCAD labeled source loader returned no TX labels")
                continue
            tx = y_l.to(device=device, dtype=torch.long).reshape(-1)
            count = int(tx.numel())
            receiver = _bicad_xr_local_domain_labels(
                (
                    _bicad_xr_required_metadata
                    if protocol.strict_pair_concat
                    else _bicad_xr_metadata
                )(extra_l, "rx_i", device, count),
                source_receiver_indices,
                name="receiver",
            )
            day = _bicad_xr_local_domain_labels(
                (
                    _bicad_xr_required_metadata
                    if protocol.strict_pair_concat
                    else _bicad_xr_metadata
                )(extra_l, "day_i", device, count),
                source_day_indices,
                name="day",
            )
            optimizer.zero_grad(set_to_none=True)
            step_view = None
            if protocol.strict_pair_concat:
                if unlabeled_iter is None:
                    raise RuntimeError("PairBiCAD unlabeled iterator was not initialized")
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    try:
                        unlabeled_batch = next(unlabeled_iter)
                    except StopIteration as exc:
                        raise RuntimeError("PairBiCAD requires a non-empty unlabeled loader") from exc
                x_u, extra_u = _move_bicad_xr_unlabeled_batch(unlabeled_batch, device)
                unlabeled_count = int(x_u.size(0))
                receiver_u = _bicad_xr_local_domain_labels(
                    _bicad_xr_required_metadata(
                        extra_u, "rx_i", device, unlabeled_count
                    ),
                    source_receiver_indices,
                    name="receiver_u",
                )
                day_u = _bicad_xr_local_domain_labels(
                    _bicad_xr_required_metadata(
                        extra_u, "day_i", device, unlabeled_count
                    ),
                    source_day_indices,
                    name="day_u",
                )
                step_output, step_view = _bicad_xr_labeled_step(
                    trainer,
                    concat_sat_aug,
                    x_l,
                    tx,
                    receiver,
                    day,
                    x_u,
                    receiver_u,
                    day_u,
                    args=args,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    update=update + 1,
                    total_updates=total_updates,
                )
            elif epoch >= 80:
                step_output, step_view = _bicad_xr_labeled_step(
                    trainer,
                    concat_sat_aug,
                    x_l,
                    tx,
                    receiver,
                    day,
                    args=args,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    update=update + 1,
                    total_updates=total_updates,
                )
            else:
                channel = torch.zeros(count, dtype=torch.long, device=device)
                batch = BiCADXRBatch(
                    x=x_l,
                    tx=tx,
                    receiver=receiver,
                    day=day,
                    channel=channel,
                    labeled_mask=torch.ones(count, dtype=torch.bool, device=device),
                    epoch=epoch,
                )
                step_output = trainer.compute_step(
                    batch,
                    update=update + 1,
                    total_updates=total_updates,
                    epoch=epoch,
                )
            if step_view is not None:
                step_output.audit["satellite_scenario"] = step_view["scenario"]
                step_output.audit["satellite_view_applied"] = step_view["applied"]
                pair_runtime = step_output.audit.get("pairbicad_runtime")
                if isinstance(pair_runtime, dict):
                    pair_runtime["scenario"] = step_view["scenario"]
                    pair_runtime["satellite_view_applied"] = step_view["applied"]
            trainer.apply_backward_controls(step_output)
            max_grad_norm = float(getattr(args, "max_grad_norm", 0.0))
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_grad_norm)
            optimizer.step()
            update += 1
            epoch_losses.append(float(step_output.total.detach().cpu().item()))
            last_audit = step_output.audit
        if epoch_losses:
            epoch_runtime = trainer.checkpoint_runtime(
                update=update,
                total_updates=total_updates,
            )
            epoch_runtime["target_access"] = False
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "phase1_method": "bicad_xr",
                    "candidate_id": protocol.candidate_id,
                    "train_loss": _bicad_xr_mean_epoch_loss(epoch_losses),
                    "optimizer_update": update,
                    "target_access": False,
                    "bicad_xr_runtime": _bicad_xr_jsonable(epoch_runtime),
                    "bicad_xr_audit": _bicad_xr_jsonable(last_audit),
                }
            )
        if update >= total_updates:
            break
    if update == 0:
        raise RuntimeError("BiCAD-XR received no labeled source main step")

    out_dir = Path(str(getattr(args, "output_dir", "") or "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = str(getattr(args, "metrics_csv", "") or "").strip()
    metrics_jsonl = str(getattr(args, "metrics_jsonl", "") or "").strip()
    _write_ssdg_epoch_telemetry(
        Path(metrics_csv) if metrics_csv else out_dir / "metrics_epoch.csv",
        Path(metrics_jsonl) if metrics_jsonl else out_dir / "metrics_epoch.jsonl",
        epoch_rows,
    )

    runtime = trainer.checkpoint_runtime(update=update, total_updates=total_updates)
    runtime["target_access"] = False
    runtime["entry"] = bicad_xr_runtime(protocol)
    model_state = model.state_dict()
    checkpoint_payload = {
        "schema": "ssdg_phase1_bicad_xr_v1",
        "phase1_method": "bicad_xr",
        "candidate_id": protocol.candidate_id,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch_rows[-1]["epoch"]),
        "optimizer_update": update,
        "args": dict(vars(model_args)),
        "bicad_xr_runtime": runtime,
    }
    reconstructed_model = build_baseline_model(model_args, device)
    reconstructed_trainer = BiCADXRTrainer(
        _BiCADXRConcatForward(reconstructed_model),
        config,
        num_receivers=len(source_receiver_indices),
        num_days=len(source_day_indices),
        num_channels=2 if protocol.strict_pair_concat else 4,
    ).to(device)
    checkpoint_payload["bicad_xr_runtime"] = _load_bicad_xr_checkpoint_strict(
        checkpoint_payload,
        model=reconstructed_model,
        trainer=reconstructed_trainer,
    )
    save_payload(out_dir / "bicad_xr_final.pth", checkpoint_payload)
    return 0


def main() -> int:
    raw_argv = tuple(sys.argv[1:])
    if _cli_phase1_method(raw_argv) == "bicad_xr":
        args = parse(raw_argv)
    else:
        args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
