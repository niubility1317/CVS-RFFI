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

try:
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader

    from dataset_wisig import (
        WiSigCompactDataset,
        WiSigMetaSslSubsetDataset,
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
    from baseline_origin_sat_view import parse_sat_view_schedule
    from concat_sat_channel_aug import ConcatSatChannelAugment
    from cvsrffi.tensors import build_domain_label_map
    from cvsrffi.phase1_ccpc_leo import (
        CCPCLEOConfig,
        CCPCLEOConfigurationError,
        CCPCLEORuntimeError,
        add_ccpc_to_loss,
        ccpc_config_receipt,
        ccpc_leo_gradient_status,
        ccpc_leo_loss,
        ccpc_leo_unscaled_gradient,
        require_finite_ccpc_leo_gradient,
        strict_ccpc_warm_start,
        update_ccpc_receipt,
        update_ccpc_optimizer_receipt,
        validate_ccpc_terminal_receipt,
        validate_ccpc_leo_args,
        write_ccpc_failure_receipt,
    )
    from cvsrffi.phase1_pamr import (
        PAMRConfig,
        PAMRConfigurationError,
        PAMRRuntimeError,
        add_pamr_to_loss,
        pamr_config_receipt,
        pamr_gradient_status,
        pamr_loss,
        remap_pamr_local_labels_to_head_rows,
        resolve_pamr_local_head_class_binding,
        pamr_shared_encoder_parameters,
        pamr_shared_gradient_relation,
        pamr_unscaled_gradient,
        require_finite_pamr_gradient,
        resolve_pamr_classifier_weight,
        strict_pamr_warm_start,
        update_pamr_gradient_receipt,
        update_pamr_gradient_relation_receipt,
        update_pamr_receipt,
        validate_pamr_args,
        validate_pamr_binding,
        validate_pamr_terminal_receipt,
        write_pamr_failure_receipt,
    )
    from cvsrffi.phase1_cb_sfce import (
        CBSFCEConfig,
        CBSFCEConfigurationError,
        CBSFCERuntimeError,
        FROZEN_CB_SFCE_SCENARIOS,
        add_cb_sfce_to_loss,
        cb_sfce_config_receipt,
        cb_sfce_loss,
        cb_sfce_shared_encoder_and_head_parameters,
        cb_sfce_shared_gradient_relation,
        remap_cb_sfce_local_labels_to_head_rows,
        resolve_cb_sfce_classifier_weight,
        resolve_cb_sfce_local_head_class_binding,
        strict_cb_sfce_warm_start,
        update_cb_sfce_gradient_relation_receipt,
        update_cb_sfce_receipt,
        validate_cb_sfce_args,
        validate_cb_sfce_logit_binding,
        validate_cb_sfce_terminal_receipt,
        write_cb_sfce_failure_receipt,
    )
    from cvsrffi.phase1_gd_proto_nll import (
        GDProtoNLLConfig,
        GDProtoNLLConfigurationError,
        GDProtoNLLRuntimeError,
        FROZEN_GD_PROTO_NLL_SCENARIOS,
        add_gd_proto_nll_to_loss,
        advance_gd_proto_nll_state,
        gd_proto_nll_config_receipt,
        gd_proto_nll_loss,
        gd_proto_nll_shared_encoder_and_head_parameters,
        gd_proto_nll_shared_gradient_relation,
        make_gd_proto_nll_state,
        remap_gd_proto_nll_local_labels_to_head_rows,
        resolve_gd_proto_nll_classifier_weight,
        resolve_gd_proto_nll_local_head_class_binding,
        strict_gd_proto_nll_warm_start,
        update_gd_proto_nll_gradient_relation_receipt,
        update_gd_proto_nll_receipt,
        update_gd_proto_nll_state_receipt,
        validate_gd_proto_nll_args,
        validate_gd_proto_nll_feature_binding,
        validate_gd_proto_nll_terminal_receipt,
        write_gd_proto_nll_failure_receipt,
    )
    from cvsrffi.phase1_cp_sfce import (
        CPSFCEConfig,
        CPSFCEConfigurationError,
        CPSFCERuntimeError,
        FROZEN_CP_SFCE_SCENARIOS,
        cp_sfce_capture_optimizer_steps,
        cp_sfce_capture_optimizer_steps_for_model,
        cp_sfce_config_receipt,
        cp_sfce_loss,
        cp_sfce_scaled_backward_and_project,
        finalize_cp_sfce_amp_overflow_skip,
        remap_cp_sfce_local_labels_to_head_rows,
        resolve_cp_sfce_classifier_weight,
        resolve_cp_sfce_local_head_class_binding,
        strict_cp_sfce_warm_start,
        update_cp_sfce_coverage_receipt,
        update_cp_sfce_amp_overflow_receipt,
        update_cp_sfce_optimizer_step_receipt,
        update_cp_sfce_projection_receipt,
        validate_cp_sfce_args,
        validate_cp_sfce_logit_binding,
        validate_cp_sfce_terminal_receipt,
        write_cp_sfce_failure_receipt,
    )
    from cvsrffi.balanced_tx_rx_sampler import BalancedTxDomainBatchSampler
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
        real_oe_energy_ranking_loss,
        one_way_kl_from_teacher,
        sanitize_loss,
        soft_unknown_mixup_loss,
        source_episode_three_sigma_loss,
        tx_conditional_domain_invariance_loss,
        unlabeled_known_acceptance_quarantine_loss,
        zid_compactness_loss,
    )
    from cvsrffi.leakage_probe import frozen_ridge_linear_probe
    from cvsrffi.phase2_prototypes import (
        PrototypeFusionConfig,
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
    WiSigCompactDataset = WiSigMetaSslSubsetDataset = WiSigSubsetDataset = None
    PrototypeMemoryBank = None
    direct_metric_acceptance_loss = None
    multiview_direct_metric_acceptance_loss = None
    multiview_source_episode_three_sigma_loss = None
    make_soft_unknown_mixup = None
    open_world_feature_space_loss = None
    proxy_unknown_energy_loss = None
    real_oe_energy_ranking_loss = None
    sanitize_loss = None
    soft_unknown_mixup_loss = None
    source_episode_three_sigma_loss = None
    tx_conditional_domain_invariance_loss = None
    unlabeled_known_acceptance_quarantine_loss = None
    zid_compactness_loss = None
    export_phase2_prototypes = None
    fuse_tx_domain_prototypes = None
    save_phase2_prototype_export = None
    PrototypeFusionConfig = None
    attach_endpoint_accept_v1_manifest = verify_endpoint_accept_v1_manifest = None
    LocalComponentHardGate = None
    calibrate_endpoint_accept_v1 = extract_endpoint_calibration_features = None
    _resolve_days = _resolve_rxs = load_wisig_compact_pkl = make_wisig_trainval_test_by_day_rx = None
    build_baseline_model = domain_from_extra = ensure_dir = load_checkpoint = None
    mean_logs = merge_checkpoint_args = move_batch = resolve_device = save_payload = set_seed = None
    parse_sat_scenarios = satellite_protocol_manifest = None
    parse_sat_view_schedule = None
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
    CCPCLEOConfig = None
    CCPCLEOConfigurationError = CCPCLEORuntimeError = None
    add_ccpc_to_loss = ccpc_config_receipt = ccpc_leo_gradient_status = ccpc_leo_loss = None
    ccpc_leo_unscaled_gradient = None
    require_finite_ccpc_leo_gradient = strict_ccpc_warm_start = update_ccpc_receipt = None
    update_ccpc_optimizer_receipt = None
    validate_ccpc_terminal_receipt = validate_ccpc_leo_args = write_ccpc_failure_receipt = None
    PAMRConfig = None
    PAMRConfigurationError = PAMRRuntimeError = None
    add_pamr_to_loss = pamr_config_receipt = pamr_gradient_status = pamr_loss = None
    remap_pamr_local_labels_to_head_rows = resolve_pamr_local_head_class_binding = None
    pamr_shared_encoder_parameters = pamr_shared_gradient_relation = pamr_unscaled_gradient = None
    require_finite_pamr_gradient = resolve_pamr_classifier_weight = strict_pamr_warm_start = None
    update_pamr_gradient_receipt = update_pamr_gradient_relation_receipt = update_pamr_receipt = None
    validate_pamr_args = validate_pamr_binding = validate_pamr_terminal_receipt = write_pamr_failure_receipt = None
    CBSFCEConfig = None
    CBSFCEConfigurationError = CBSFCERuntimeError = None
    FROZEN_CB_SFCE_SCENARIOS = tuple()
    add_cb_sfce_to_loss = cb_sfce_config_receipt = cb_sfce_loss = None
    cb_sfce_shared_encoder_and_head_parameters = cb_sfce_shared_gradient_relation = None
    remap_cb_sfce_local_labels_to_head_rows = resolve_cb_sfce_classifier_weight = None
    resolve_cb_sfce_local_head_class_binding = strict_cb_sfce_warm_start = None
    update_cb_sfce_gradient_relation_receipt = update_cb_sfce_receipt = None
    validate_cb_sfce_args = validate_cb_sfce_logit_binding = validate_cb_sfce_terminal_receipt = None
    write_cb_sfce_failure_receipt = None
    GDProtoNLLConfig = None
    GDProtoNLLConfigurationError = GDProtoNLLRuntimeError = None
    FROZEN_GD_PROTO_NLL_SCENARIOS = tuple()
    add_gd_proto_nll_to_loss = advance_gd_proto_nll_state = gd_proto_nll_config_receipt = None
    gd_proto_nll_loss = gd_proto_nll_shared_encoder_and_head_parameters = None
    gd_proto_nll_shared_gradient_relation = make_gd_proto_nll_state = None
    remap_gd_proto_nll_local_labels_to_head_rows = resolve_gd_proto_nll_classifier_weight = None
    resolve_gd_proto_nll_local_head_class_binding = strict_gd_proto_nll_warm_start = None
    update_gd_proto_nll_gradient_relation_receipt = update_gd_proto_nll_receipt = None
    update_gd_proto_nll_state_receipt = validate_gd_proto_nll_args = None
    validate_gd_proto_nll_feature_binding = validate_gd_proto_nll_terminal_receipt = None
    write_gd_proto_nll_failure_receipt = None
    CPSFCEConfig = None
    CPSFCEConfigurationError = CPSFCERuntimeError = None
    FROZEN_CP_SFCE_SCENARIOS = tuple()
    cp_sfce_capture_optimizer_steps = cp_sfce_capture_optimizer_steps_for_model = None
    cp_sfce_config_receipt = cp_sfce_loss = None
    cp_sfce_scaled_backward_and_project = finalize_cp_sfce_amp_overflow_skip = None
    remap_cp_sfce_local_labels_to_head_rows = None
    resolve_cp_sfce_classifier_weight = resolve_cp_sfce_local_head_class_binding = None
    strict_cp_sfce_warm_start = None
    update_cp_sfce_coverage_receipt = update_cp_sfce_amp_overflow_receipt = None
    update_cp_sfce_optimizer_step_receipt = None
    update_cp_sfce_projection_receipt = validate_cp_sfce_args = validate_cp_sfce_logit_binding = None
    validate_cp_sfce_terminal_receipt = write_cp_sfce_failure_receipt = None


_MANYTX_REAL_OE_LOCKED_TARGET_NEW_TX = tuple(
    "1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,"
    "13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6".split(",")
)
_MANYTX_REAL_OE_TRAIN_TX = tuple(
    "10-4,3-1,7-8,16-20,11-17,8-14,19-1,2-13,11-1,19-19,18-1,4-1,13-19,18-4,13-3,11-10,"
    "19-11,7-20,1-11,18-11,14-8,3-19,13-20,14-9,19-4,18-17,19-7,2-17,7-10,1-10,2-7,9-1,"
    "18-14,11-4,18-15,20-18,19-2,14-12,3-20,1-12,3-2,5-1,7-13,11-20,20-4,18-5,18-2,6-1,20-7,"
    "10-17,8-1,18-16,17-10,20-1,2-19,14-20,8-8,10-7,9-20,6-6,19-20,2-6,20-5,1-15,1-14,8-13,"
    "18-20,8-18,7-11,8-7,9-7,18-12,11-7,16-16,14-14,20-14,15-19,2-8,14-13,20-8".split(",")
)
_MANYTX_REAL_OE_PROXY_TX = tuple(
    "20-20,20-16,19-3,1-19,3-18,19-12,5-20,7-14,12-7,7-9,17-11,20-3,12-20,16-1,18-7,2-3,19-10,18-9,2-4,15-6".split(",")
)
_MANYTX_REAL_OE_RESERVE_TX = tuple(
    "2-14,10-11,9-14,13-7,2-12,7-12,5-5,2-15,18-13,5-16,19-14,15-1,12-19,3-13,7-7,4-11".split(",")
)
_MANYTX_REAL_OE_PARTITION_ROOT_SHA256 = (
    "ca3ed65a533359d2abb022fa513c49101ad93235738a39b362b5cdd15879c3d1"
)
_MANYTX_REAL_OE_SOURCE_RX_LABELS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
_MANYTX_REAL_OE_SOURCE_DAY_LABELS = ("2021_03_01", "2021_03_08")
_MANYTX_REAL_OE_TARGET_RX_LABELS = ("20-1", "3-19", "7-14", "7-7", "8-8")
_MANYTX_REAL_OE_INSUFFICIENT_TX = (
    "1-1", "1-2", "10-1", "12-1", "13-18", "16-5", "2-1", "2-20"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train two-stage SSDG from a Stable-SAT baseline checkpoint.")
    parser.add_argument("--baseline_ckpt", type=str, default="", help="Optional checkpoint. Empty means train SSDG from scratch.")
    parser.add_argument("--from_scratch", type=str2bool, default=True)
    parser.add_argument("--split_mode", type=str, default="tx_rx_day_1_6_3", choices=["tx_rx_day_1_6_3", "tx_rx_day_1_7_2"])
    parser.add_argument("--labeled_ratio", type=float, default=0.08)
    parser.add_argument("--unlabeled_ratio", type=float, default=0.72)
    parser.add_argument("--source_val_ratio", type=float, default=0.20)
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
    parser.add_argument("--concat_sat_ce_weight", type=float, default=1.0)
    parser.add_argument("--concat_sat_deduplicate_tx_ce", type=str2bool, default=False)
    parser.add_argument("--concat_sat_teacher_clean_only", type=str2bool, default=False)
    parser.add_argument("--concat_sat_start_epoch", type=int, default=1)
    parser.add_argument("--sat_view_prob", type=float, default=1.0)
    parser.add_argument("--sat_view_seed", type=int, default=2027)
    parser.add_argument("--sat_protocol_disjoint_required", type=str2bool, default=False)
    parser.add_argument("--lambda_sat_cls", type=float, default=0.10)
    parser.add_argument("--lambda_sat_cons", type=float, default=0.0)
    parser.add_argument("--sat_cons_start_epoch", type=int, default=20)
    parser.add_argument(
        "--phase1_ccpc_leo_frozen_mode",
        type=str2bool,
        default=False,
        help="Enable the frozen P1-CCPC-LEO C/G continuation contract.",
    )
    parser.add_argument(
        "--phase1_ccpc_leo_enabled",
        type=str2bool,
        default=False,
        help="Enable the sole CCPC loss inside a frozen P1-CCPC-LEO G arm.",
    )
    parser.add_argument(
        "--phase1_ccpc_leo_gradient_audit_only",
        type=str2bool,
        default=False,
        help="Permit the frozen 15-epoch, G-only CCPC unscaled-gradient health audit.",
    )
    parser.add_argument(
        "--lambda_ccpc_leo",
        type=float,
        default=0.0,
        help="Frozen at 0 for C and 0.02 for G when CCPC-LEO mode is enabled.",
    )
    parser.add_argument(
        "--ccpc_leo_temperature",
        type=float,
        default=0.12,
        help="Frozen paired-contrastive temperature for P1-CCPC-LEO.",
    )
    parser.add_argument(
        "--phase1_pamr_frozen_mode",
        type=str2bool,
        default=False,
        help="Enable the frozen P1-PAMR C/G continuation contract.",
    )
    parser.add_argument(
        "--phase1_pamr_enabled",
        type=str2bool,
        default=False,
        help="Enable the sole angular-margin restoration loss in a frozen P1-PAMR G arm.",
    )
    parser.add_argument(
        "--phase1_pamr_audit_only",
        type=str2bool,
        default=False,
        help="Permit the frozen one-epoch, G-only P1-PAMR technical gradient audit.",
    )
    parser.add_argument(
        "--lambda_pamr",
        type=float,
        default=0.0,
        help="Frozen at 0 for C and 0.05 for G when P1-PAMR mode is enabled.",
    )
    parser.add_argument(
        "--phase1_cb_sfce_frozen_mode",
        type=str2bool,
        default=False,
        help="Enable the frozen P1-CB-SFCE C/G continuation contract.",
    )
    parser.add_argument(
        "--phase1_cb_sfce_enabled",
        type=str2bool,
        default=False,
        help="Enable the sole class-balanced satellite focal CE in a frozen P1-CB-SFCE G arm.",
    )
    parser.add_argument(
        "--lambda_cb_sfce",
        type=float,
        default=0.0,
        help="Frozen at 0 for C and 0.10 for G when P1-CB-SFCE mode is enabled.",
    )
    parser.add_argument(
        "--cb_sfce_gamma",
        type=float,
        default=1.0,
        help="Frozen focal exponent for P1-CB-SFCE; must remain 1.",
    )
    parser.add_argument(
        "--phase1_gd_proto_nll_frozen_mode",
        type=str2bool,
        default=False,
        help="Enable the frozen P1-GD-ProtoNLL C/G continuation contract.",
    )
    parser.add_argument(
        "--phase1_gd_proto_nll_enabled",
        type=str2bool,
        default=False,
        help="Enable lagged-EMA local4xscenario DRO prototype focal CE in a frozen G arm.",
    )
    parser.add_argument(
        "--lambda_gd_proto_nll",
        type=float,
        default=0.0,
        help="Frozen at 0 for C and 0.10 for G when P1-GD-ProtoNLL mode is enabled.",
    )
    parser.add_argument(
        "--gd_proto_nll_gamma",
        type=float,
        default=1.0,
        help="Frozen focal exponent for P1-GD-ProtoNLL; must remain 1.",
    )
    parser.add_argument(
        "--phase1_cp_sfce_frozen_mode",
        type=str2bool,
        default=False,
        help="Enable the frozen P1-CP-SFCE C/G continuation contract.",
    )
    parser.add_argument(
        "--phase1_cp_sfce_enabled",
        type=str2bool,
        default=False,
        help="Enable conflict projection of the frozen satellite focal-CE gradient in a P1-CP-SFCE G arm.",
    )
    parser.add_argument(
        "--lambda_cp_sfce",
        type=float,
        default=0.0,
        help="Frozen at 0 for C and 0.10 for G when P1-CP-SFCE mode is enabled.",
    )
    parser.add_argument(
        "--cp_sfce_gamma",
        type=float,
        default=1.0,
        help="Frozen focal exponent for P1-CP-SFCE; must remain 1.",
    )
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
        choices=["final_only", "source_validation_only"],
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
    parser.add_argument(
        "--ow_feat_key",
        type=str,
        default="z_id",
        choices=["z_id", "id_feat_cls"],
        help="Feature consumed only by the optional known-only open-world geometry loss.",
    )
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
    parser.add_argument(
        "--phase1_source_train_tx_ids",
        type=str,
        default="",
        help="Comma-separated source TX labels admitted to Phase1 training.",
    )
    parser.add_argument(
        "--phase1_source_known_validation_tx_ids",
        type=str,
        default="",
        help="Comma-separated TX-exclusive held-known labels; never loaded by training.",
    )
    parser.add_argument(
        "--phase1_source_proxy_unknown_tx_ids",
        type=str,
        default="",
        help="Comma-separated TX-exclusive proxy-unknown labels; never loaded by training.",
    )
    parser.add_argument(
        "--phase1_allow_empty_proxy_unknown",
        type=str2bool,
        default=False,
        help="Allow a known-train/known-validation-only main TX split only for frozen external real-OE runs.",
    )
    parser.add_argument("--manytx_real_oe_protocol_enabled", type=str2bool, default=False)
    parser.add_argument("--manytx_real_oe_enabled", type=str2bool, default=False)
    parser.add_argument("--manytx_real_oe_pkl", type=str, default="")
    parser.add_argument("--manytx_real_oe_train_tx_ids", type=str, default="")
    parser.add_argument("--manytx_real_oe_proxy_tx_ids", type=str, default="")
    parser.add_argument("--manytx_real_oe_reserve_tx_ids", type=str, default="")
    parser.add_argument("--manytx_locked_target_new_tx_ids", type=str, default="")
    parser.add_argument("--manytx_real_oe_partition_root_sha256", type=str, default="")
    parser.add_argument("--manytx_real_oe_days", type=str, default="2021_03_01,2021_03_08")
    parser.add_argument("--manytx_real_oe_rxs", type=str, default="1-1,1-19,14-7,18-2,19-2,2-1")
    parser.add_argument("--manytx_real_oe_equalized", type=int, default=1)
    parser.add_argument("--manytx_real_oe_tx_per_batch", type=int, default=16)
    parser.add_argument("--manytx_real_oe_samples_per_tx", type=int, default=8)
    parser.add_argument("--lambda_manytx_real_oe", type=float, default=0.0)
    parser.add_argument("--manytx_real_oe_start_epoch", type=int, default=61)
    parser.add_argument("--manytx_real_oe_warmup_epochs", type=int, default=10)
    parser.add_argument("--manytx_real_oe_temperature", type=float, default=1.0)
    parser.add_argument("--manytx_real_oe_margin", type=float, default=1.0)
    parser.add_argument("--manytx_real_oe_tau", type=float, default=1.0)
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


def _phase1_tx_partition_view(
    ds_w: Mapping[str, Any],
    *,
    train_spec: str = "",
    known_validation_spec: str = "",
    proxy_unknown_spec: str = "",
    allow_empty_proxy_unknown: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a contiguous TX-reindexed training view and a disjoint-role receipt."""

    def _parse(value: str) -> List[str]:
        return [item.strip() for item in str(value or "").split(",") if item.strip()]

    train = _parse(train_spec)
    known_validation = _parse(known_validation_spec)
    proxy_unknown = _parse(proxy_unknown_spec)
    enabled = bool(train or known_validation or proxy_unknown)
    if not enabled:
        labels = [str(value) for value in list(ds_w.get("tx_list", []))]
        return dict(ds_w), {
            "schema": "cvs.phase1.tx_partition_receipt.v1",
            "enabled": False,
            "source_known_train_tx": labels,
            "source_known_validation_tx": [],
            "source_proxy_unknown_tx": [],
            "dataset_tx_order": labels,
        }
    if not train or not known_validation or (not proxy_unknown and not bool(allow_empty_proxy_unknown)):
        raise ValueError(
            "Phase1 TX-exclusive mode requires non-empty train, known-validation, and proxy-unknown TX sets"
        )
    roles = {
        "source_known_train_tx": train,
        "source_known_validation_tx": known_validation,
        "source_proxy_unknown_tx": proxy_unknown,
    }
    for role, values in roles.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate TX label in {role}: {values}")
    role_names = list(roles)
    for i, left in enumerate(role_names):
        for right in role_names[i + 1 :]:
            overlap = sorted(set(roles[left]).intersection(roles[right]))
            if overlap:
                raise ValueError(f"Phase1 TX roles overlap: {left} vs {right}: {overlap}")

    tx_list = [str(value) for value in list(ds_w.get("tx_list", []))]
    index_by_label = {label: index for index, label in enumerate(tx_list)}
    requested = [value for values in roles.values() for value in values]
    missing = sorted(set(requested).difference(index_by_label))
    if missing:
        raise ValueError(f"Phase1 TX role labels are absent from dataset: {missing}")
    train_indices = [index_by_label[label] for label in train]
    data = ds_w.get("data")
    if data is None:
        raise ValueError("WiSig dataset lacks data")
    filtered = dict(ds_w)
    filtered["tx_list"] = list(train)
    filtered["data"] = [data[index] for index in train_indices]
    receipt = {
        "schema": "cvs.phase1.tx_partition_receipt.v1",
        "enabled": True,
        **roles,
        "dataset_tx_order": list(tx_list),
        "dataset_tx_count": len(tx_list),
        "training_tx_count": len(train),
        "allow_empty_proxy_unknown": bool(allow_empty_proxy_unknown),
        "training_view_contiguous_reindex": {
            str(new_index): label for new_index, label in enumerate(train)
        },
        "held_tx_loaded_by_training": False,
    }
    receipt["partition_sha256"] = hashlib.sha256(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return filtered, receipt


def _parse_manytx_tx_ids(value: Any) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if len(items) != len(set(items)):
        raise ValueError(f"duplicate ManyTx identity in frozen role list: {items}")
    return items


def _parse_wisig_axis_spec(value: Any) -> Optional[List[Any]]:
    """Parse numeric indices without coercing physical labels such as dates.

    Python's ``int`` accepts underscores, so the generic CSV helper turns the
    physical day label ``2021_03_01`` into ``20210301``.  WiSig axes must only
    interpret an item as an index when every character (apart from an optional
    sign) is a decimal digit.
    """

    raw = str(value or "").strip()
    if not raw:
        return None
    parsed: List[Any] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        numeric = item[1:] if item[:1] in {"+", "-"} else item
        parsed.append(int(item) if numeric.isdigit() else item)
    return parsed or None


def _require_frozen_physical_labels(
    value: Any,
    expected: Sequence[str],
    *,
    field: str,
) -> Tuple[str, ...]:
    labels = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if labels == tuple(expected):
        return labels
    if any(item.isdigit() for item in labels):
        raise ValueError(
            f"{field} rejects raw index strings; use the frozen physical labels"
        )
    raise ValueError(
        f"{field} must exactly equal the frozen physical labels: {','.join(expected)}"
    )


def _manytx_tx_list_sha256(items: Sequence[str]) -> str:
    # Match the authority manifest convention, rather than a display-oriented
    # comma join: compact JSON preserves list order and unambiguously encodes
    # each canonical transmitter identifier.
    payload = json.dumps(
        [str(item) for item in items],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_manytx_real_oe_config(args: Any) -> Dict[str, Any]:
    """Fail closed on the immutable ManyTx OE/proxy/reserve/locked partition."""

    protocol_enabled = bool(getattr(args, "manytx_real_oe_protocol_enabled", False))
    enabled = bool(getattr(args, "manytx_real_oe_enabled", False))
    weight = float(getattr(args, "lambda_manytx_real_oe", 0.0))
    if not protocol_enabled:
        if enabled or weight != 0.0:
            raise ValueError(
                "ManyTx real-OE activity requires --manytx_real_oe_protocol_enabled true"
            )
        return {
            "schema": "cvs.phase1.manytx_real_oe_receipt.v2",
            "protocol_enabled": False,
            "enabled": False,
            "training_loader_constructed": False,
            "proxy_loaded_by_training": False,
            "locked_target_new_loaded_by_training": False,
            "reserve_loaded_by_training": False,
        }
    if enabled and weight <= 0.0:
        raise ValueError("enabled ManyTx real-OE requires --lambda_manytx_real_oe > 0")
    if (not enabled) and weight != 0.0:
        raise ValueError("disabled ManyTx real-OE requires --lambda_manytx_real_oe = 0")
    if enabled and not str(getattr(args, "manytx_real_oe_pkl", "")).strip():
        raise ValueError("enabled ManyTx real-OE requires --manytx_real_oe_pkl")
    # This frozen arm measures one mechanism only: observed, label-masked
    # source OE.  Do not silently stack historical virtual/proxy/geometry
    # routes onto it.
    forbidden_stacked_losses = (
        "lambda_open_world_feat",
        "lambda_proxy_unknown",
        "lambda_soft_unknown_mixup",
        "lambda_source_episode",
        "lambda_direct_metric_accept",
    )
    if enabled:
        nonzero_stacked = [
            name
            for name in forbidden_stacked_losses
            if float(getattr(args, name, 0.0)) != 0.0
        ]
        if nonzero_stacked:
            raise ValueError(
                "ManyTx real-OE forbids stacked proxy/virtual/geometry losses: "
                + ",".join(nonzero_stacked)
            )
    if str(getattr(args, "manytx_real_oe_partition_root_sha256", "")).strip().lower() != _MANYTX_REAL_OE_PARTITION_ROOT_SHA256:
        raise ValueError("ManyTx real-OE partition root does not match the frozen authority lock")

    oe_tx = _parse_manytx_tx_ids(getattr(args, "manytx_real_oe_train_tx_ids", ""))
    proxy_tx = _parse_manytx_tx_ids(getattr(args, "manytx_real_oe_proxy_tx_ids", ""))
    reserve_tx = _parse_manytx_tx_ids(getattr(args, "manytx_real_oe_reserve_tx_ids", ""))
    locked_target_new_tx = _parse_manytx_tx_ids(
        getattr(args, "manytx_locked_target_new_tx_ids", "")
    )
    frozen_roles = {
        "manytx_real_oe_train_tx": oe_tx,
        "manytx_real_oe_proxy_tx": proxy_tx,
        "manytx_real_oe_reserve_tx": reserve_tx,
        "locked_target_new_tx": locked_target_new_tx,
    }
    expected_roles = {
        "manytx_real_oe_train_tx": _MANYTX_REAL_OE_TRAIN_TX,
        "manytx_real_oe_proxy_tx": _MANYTX_REAL_OE_PROXY_TX,
        "manytx_real_oe_reserve_tx": _MANYTX_REAL_OE_RESERVE_TX,
        "locked_target_new_tx": _MANYTX_REAL_OE_LOCKED_TARGET_NEW_TX,
    }
    for role, values in frozen_roles.items():
        if values != expected_roles[role]:
            raise ValueError(f"{role} does not match the frozen ManyTx authority list")
    role_names = list(frozen_roles)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1 :]:
            overlap = sorted(set(frozen_roles[left]).intersection(frozen_roles[right]))
            if overlap:
                raise ValueError(f"ManyTx frozen roles overlap: {left} vs {right}: {overlap}")
    if len(oe_tx) != 80 or len(proxy_tx) != 20 or len(reserve_tx) != 16 or len(locked_target_new_tx) != 20:
        raise ValueError("ManyTx frozen partition must be 80 OE / 20 proxy / 16 reserve / 20 locked target-new")
    eligible_tx = set(oe_tx).union(proxy_tx, reserve_tx)
    if len(eligible_tx) != 116 or len(set(eligible_tx).union(locked_target_new_tx)) != 136:
        raise ValueError("ManyTx frozen partition must contain 116 eligible extras plus 20 locked target-new identities")
    insufficient_overlap = sorted(eligible_tx.intersection(_MANYTX_REAL_OE_INSUFFICIENT_TX))
    if insufficient_overlap:
        raise ValueError(f"ManyTx frozen roles include insufficient physical-RX identities: {insufficient_overlap}")

    source_days = _require_frozen_physical_labels(
        getattr(args, "wisig_train_days", ""),
        _MANYTX_REAL_OE_SOURCE_DAY_LABELS,
        field="--wisig_train_days",
    )
    source_rxs = _require_frozen_physical_labels(
        getattr(args, "wisig_train_rxs", ""),
        _MANYTX_REAL_OE_SOURCE_RX_LABELS,
        field="--wisig_train_rxs",
    )
    target_rxs = _require_frozen_physical_labels(
        getattr(args, "wisig_test_rxs", ""),
        _MANYTX_REAL_OE_TARGET_RX_LABELS,
        field="--wisig_test_rxs",
    )
    oe_days = _require_frozen_physical_labels(
        getattr(args, "manytx_real_oe_days", ""),
        _MANYTX_REAL_OE_SOURCE_DAY_LABELS,
        field="--manytx_real_oe_days",
    )
    oe_rxs = _require_frozen_physical_labels(
        getattr(args, "manytx_real_oe_rxs", ""),
        _MANYTX_REAL_OE_SOURCE_RX_LABELS,
        field="--manytx_real_oe_rxs",
    )
    if set(source_rxs).intersection(target_rxs):
        raise ValueError("frozen source and target physical receiver labels must be disjoint")
    if int(getattr(args, "manytx_real_oe_equalized", -1)) != 1:
        raise ValueError("ManyTx real-OE is frozen to equalized=1")
    if int(getattr(args, "manytx_real_oe_tx_per_batch", 0)) <= 0 or int(
        getattr(args, "manytx_real_oe_samples_per_tx", 0)
    ) <= 0:
        raise ValueError("ManyTx real-OE balanced batch dimensions must be positive")
    if int(getattr(args, "manytx_real_oe_tx_per_batch", 0)) > len(oe_tx):
        raise ValueError("ManyTx real-OE tx_per_batch exceeds the frozen OE identity count")
    if float(getattr(args, "manytx_real_oe_temperature", 0.0)) <= 0.0 or float(
        getattr(args, "manytx_real_oe_tau", 0.0)
    ) <= 0.0:
        raise ValueError("ManyTx real-OE temperature and tau must be positive")

    known_ids = set(
        _parse_manytx_tx_ids(getattr(args, "phase1_source_train_tx_ids", ""))
        + _parse_manytx_tx_ids(getattr(args, "phase1_source_known_validation_tx_ids", ""))
        + _parse_manytx_tx_ids(getattr(args, "phase1_source_proxy_unknown_tx_ids", ""))
    )
    external_ids = set().union(*[set(values) for values in frozen_roles.values()])
    overlap = sorted(known_ids.intersection(external_ids))
    if overlap:
        raise ValueError(f"ManyTx roles overlap the known Phase1 TX roles: {overlap}")
    return {
        "schema": "cvs.phase1.manytx_real_oe_receipt.v2",
        "protocol_enabled": True,
        "enabled": bool(enabled),
        "partition_root_sha256": _MANYTX_REAL_OE_PARTITION_ROOT_SHA256,
        "oe_train_tx": list(oe_tx),
        "proxy_tx": list(proxy_tx),
        "reserve_tx": list(reserve_tx),
        "locked_target_new_tx": list(locked_target_new_tx),
        "oe_train_tx_sha256": _manytx_tx_list_sha256(oe_tx),
        "proxy_tx_sha256": _manytx_tx_list_sha256(proxy_tx),
        "reserve_tx_sha256": _manytx_tx_list_sha256(reserve_tx),
        "locked_target_new_tx_sha256": _manytx_tx_list_sha256(locked_target_new_tx),
        "known_source_day_labels": list(source_days),
        "known_source_receiver_labels": list(source_rxs),
        "known_target_receiver_labels": list(target_rxs),
        "oe_source_day_labels": list(oe_days),
        "oe_source_receiver_labels": list(oe_rxs),
        "insufficient_physical_rx_tx": list(_MANYTX_REAL_OE_INSUFFICIENT_TX),
        "eligible_extra_count": 116,
        "equalized": 1,
        "proxy_loaded_by_training": False,
        "locked_target_new_loaded_by_training": False,
        "reserve_loaded_by_training": False,
        "training_loader_constructed": False,
    }


class _ManyTxRealOeBalancedBatchSampler:
    """Deterministically emits equal-count batches over real OE transmitter IDs."""

    def __init__(
        self,
        dataset: Any,
        *,
        tx_per_batch: int,
        samples_per_tx: int,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        self.tx_per_batch = int(tx_per_batch)
        self.samples_per_tx = int(samples_per_tx)
        self.batches_per_epoch = max(1, int(batches_per_epoch))
        self.seed = int(seed)
        self.epoch = 0
        grouped: Dict[int, List[int]] = defaultdict(list)
        direct_index = getattr(dataset, "index", None)
        if direct_index is not None:
            indexed_items = enumerate(direct_index)
        else:
            # WiSigMetaSslSubsetDataset deliberately exposes only its selected
            # positions; recover their immutable base index for batch balance
            # without recovering labels in the returned batch.
            base = getattr(dataset, "base", None)
            selected = getattr(dataset, "selected", None)
            base_index = getattr(base, "index", None)
            if selected is None or base_index is None:
                raise ValueError("ManyTx real-OE sampler cannot locate dataset TX metadata")
            indexed_items = (
                (sample_index, base_index[int(base_sample_index)])
                for sample_index, base_sample_index in enumerate(selected)
            )
        for sample_index, item in indexed_items:
            grouped[int(item.tx_i)].append(int(sample_index))
        self.groups = {tx_i: tuple(indices) for tx_i, indices in sorted(grouped.items()) if indices}
        if len(self.groups) < self.tx_per_batch:
            raise ValueError("ManyTx real-OE sampler has fewer populated TX identities than tx_per_batch")

    @property
    def batch_size(self) -> int:
        return int(self.tx_per_batch * self.samples_per_tx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.batches_per_epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        tx_ids = np.asarray(sorted(self.groups), dtype=np.int64)
        for _ in range(self.batches_per_epoch):
            chosen_tx = rng.choice(tx_ids, size=self.tx_per_batch, replace=False)
            batch: List[int] = []
            for tx_i in chosen_tx.tolist():
                members = np.asarray(self.groups[int(tx_i)], dtype=np.int64)
                picks = rng.choice(members, size=self.samples_per_tx, replace=members.size < self.samples_per_tx)
                batch.extend(int(value) for value in picks.tolist())
            rng.shuffle(batch)
            yield batch


def _manytx_real_oe_coverage(
    ds_w: Mapping[str, Any],
    tx_index: int,
    *,
    rx_indices: Sequence[int],
    day_indices: Sequence[int],
    eq_index: int,
) -> Tuple[int, set[int], set[int]]:
    data = ds_w.get("data")
    if data is None:
        raise ValueError("ManyTx real-OE dataset lacks data")
    total = 0
    seen_days: set[int] = set()
    seen_rxs: set[int] = set()
    for rx_i in rx_indices:
        for day_i in day_indices:
            arr = data[tx_index][int(rx_i)][int(day_i)][int(eq_index)]
            if arr is None:
                continue
            count = int(arr.shape[0])
            if count > 0:
                total += count
                seen_days.add(int(day_i))
                seen_rxs.add(int(rx_i))
    return total, seen_days, seen_rxs


def _manytx_real_oe_coverage_meets_contract(
    count: int,
    seen_days: Sequence[int],
    seen_rxs: Sequence[int],
    *,
    expected_days: Sequence[int],
) -> bool:
    return (
        int(count) >= 400
        and set(seen_days) == set(expected_days)
        and len(set(seen_rxs)) >= 2
    )


def _build_manytx_real_oe_data(
    args: Any,
    device: Any,
    *,
    batches_per_epoch: int,
) -> Dict[str, Any]:
    receipt = _validate_manytx_real_oe_config(args)
    if not bool(receipt.get("enabled", False)):
        return {"loader": None, "sampler": None, "receipt": receipt}
    if WiSigCompactDataset is None or WiSigMetaSslSubsetDataset is None or DataLoader is None:
        raise ImportError("WiSig dataset and DataLoader support are required for ManyTx real-OE")

    ds_w = load_wisig_compact_pkl(str(args.manytx_real_oe_pkl))
    tx_list = [str(value) for value in list(ds_w.get("tx_list", []))]
    tx_index = {tx: index for index, tx in enumerate(tx_list)}
    missing = sorted(set(receipt["oe_train_tx"]).difference(tx_index))
    if missing:
        raise ValueError(f"ManyTx frozen identity is absent from the supplied pkl: {missing}")
    day_list = list(ds_w.get("capture_date_list", []))
    rx_list = list(ds_w.get("rx_list", []))
    day_indices = _resolve_days(
        day_list,
        list(receipt["oe_source_day_labels"]),
        [],
    )
    rx_indices = _resolve_rxs(
        rx_list,
        list(receipt["oe_source_receiver_labels"]),
        [],
    )
    if tuple(str(day_list[index]) for index in day_indices) != tuple(receipt["oe_source_day_labels"]):
        raise ValueError("ManyTx real-OE physical day-label resolution drift")
    if tuple(str(rx_list[index]) for index in rx_indices) != tuple(receipt["oe_source_receiver_labels"]):
        raise ValueError("ManyTx real-OE physical receiver-label resolution drift")
    eq_list = list(ds_w.get("equalized_list", [0]))
    if 1 not in eq_list:
        raise ValueError("ManyTx real-OE requires equalized=1")
    eq_index = int(eq_list.index(1))
    for tx in receipt["oe_train_tx"]:
        count, seen_days, seen_rxs = _manytx_real_oe_coverage(
            ds_w,
            tx_index[str(tx)],
            rx_indices=rx_indices,
            day_indices=day_indices,
            eq_index=eq_index,
        )
        if not _manytx_real_oe_coverage_meets_contract(
            count,
            seen_days,
            seen_rxs,
            expected_days=day_indices,
        ):
            raise ValueError(
                f"ManyTx real-OE coverage contract failed for {tx}: "
                f"count={count} days={sorted(seen_days)} rxs={sorted(seen_rxs)}"
            )
    base = WiSigCompactDataset(
        ds_w,
        out_len=int(args.wisig_out_len),
        crop_mode="center",
        normalize=True,
        equalized=1,
        tx_keep=[tx_index[str(tx)] for tx in receipt["oe_train_tx"]],
        rx_keep=rx_indices,
        day_keep=day_indices,
        domain="rx_day",
        seed=int(args.seed) + 611,
        build_index=True,
    )
    oe_dataset = WiSigMetaSslSubsetDataset(
        base,
        list(range(len(base))),
        split_source="manytx_real_oe_train_tx_hidden",
        role="manytx_real_oe",
        tx_label_visible=False,
    )
    sampler = _ManyTxRealOeBalancedBatchSampler(
        oe_dataset,
        tx_per_batch=int(args.manytx_real_oe_tx_per_batch),
        samples_per_tx=int(args.manytx_real_oe_samples_per_tx),
        batches_per_epoch=int(batches_per_epoch),
        seed=int(args.seed) + 611,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": int(args.num_workers),
        "pin_memory": getattr(device, "type", "cpu") == "cuda",
        "persistent_workers": int(args.num_workers) > 0,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    loader = DataLoader(oe_dataset, **loader_kwargs)
    receipt = {
        **receipt,
        "training_loader_constructed": True,
        "oe_loader_sample_count": int(len(oe_dataset)),
        "oe_batch_size": int(sampler.batch_size),
        "oe_batches_per_epoch": int(len(sampler)),
        "oe_labels_masked_to_minus_one": True,
        "oe_source_day_indices": [int(value) for value in day_indices],
        "oe_source_receiver_indices": [int(value) for value in rx_indices],
        "oe_source_day_labels_resolved": [str(day_list[value]) for value in day_indices],
        "oe_source_receiver_labels_resolved": [str(rx_list[value]) for value in rx_indices],
    }
    return {"loader": loader, "sampler": sampler, "receipt": receipt}


def _select_open_world_feature(
    out: Mapping[str, Any],
    z_id: "torch.Tensor",
    *,
    key: str,
) -> "torch.Tensor":
    """Select the explicitly configured feature for the optional geometry loss.

    ``z_id`` is returned by identity for the default path so legacy runs keep
    their exact feature tensor and do not enter any new validation branch.
    The ``id_feat_cls`` route is deliberately strict because it is a separate
    frozen experimental intervention rather than a best-effort alias.
    """

    feature_key = str(key or "z_id")
    if feature_key == "z_id":
        return z_id
    if feature_key != "id_feat_cls":
        raise ValueError(f"Unsupported open-world feature key: {feature_key!r}")
    if not isinstance(out, Mapping):
        raise ValueError("id_feat_cls selection requires a mapping model output")
    feature = out.get("id_feat_cls")
    if not torch.is_tensor(feature):
        raise ValueError("id_feat_cls selection requires a top-level tensor output")
    if feature.dim() != 2:
        raise ValueError(
            "id_feat_cls selection requires a 2D [B,D] tensor, "
            f"got shape={tuple(feature.shape)}"
        )
    if not bool(torch.isfinite(feature).all().item()):
        raise ValueError("id_feat_cls selection rejects non-finite values")
    if not torch.is_tensor(z_id) or z_id.dim() != 2:
        raise ValueError("id_feat_cls selection requires the current z_id to be a 2D tensor")
    if feature.size(0) != z_id.size(0):
        raise ValueError(
            "id_feat_cls selection row mismatch with z_id: "
            f"{feature.size(0)} != {z_id.size(0)}"
        )
    if feature.size(1) != z_id.size(1):
        raise ValueError(
            "id_feat_cls selection dimension mismatch with z_id: "
            f"{feature.size(1)} != {z_id.size(1)}"
        )
    return feature


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
    requested_labeled_ratio: float | None = None,
    requested_unlabeled_ratio: float | None = None,
    requested_source_val_ratio: float | None = None,
    realized_rho_tolerance: float = 0.002,
    realized_source_val_tolerance: float = 0.002,
) -> Dict[str, Any]:
    labeled_size = int(len(labeled_indices))
    unlabeled_size = int(len(unlabeled_indices))
    source_validation_size = int(len(source_validation_indices))
    train_size = labeled_size + unlabeled_size
    source_pool_size = train_size + source_validation_size
    realized_rho = float(labeled_size) / float(max(1, train_size))
    realized_source_val_fraction = float(source_validation_size) / float(
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
        "labeled_size": labeled_size,
        "unlabeled_size": unlabeled_size,
        "source_validation_size": source_validation_size,
        "source_pool_size": source_pool_size,
        "requested_labeled_ratio": float(requested_labeled_ratio),
        "requested_unlabeled_ratio": float(requested_unlabeled_ratio),
        "requested_source_val_ratio": float(requested_source_val_ratio),
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


def _manytx_known_source_physical_receipt(
    args: Any,
    *,
    day_list: Sequence[Any],
    rx_list: Sequence[Any],
    train_days: Sequence[int],
    train_rxs: Sequence[int],
    test_rxs: Sequence[int],
) -> Dict[str, Any]:
    if not bool(getattr(args, "manytx_real_oe_protocol_enabled", False)):
        return {}
    resolved_source_days = tuple(str(day_list[index]) for index in train_days)
    resolved_source_rxs = tuple(str(rx_list[index]) for index in train_rxs)
    resolved_target_rxs = tuple(str(rx_list[index]) for index in test_rxs)
    if resolved_source_days != _MANYTX_REAL_OE_SOURCE_DAY_LABELS:
        raise ValueError("known source physical day-label resolution drift")
    if resolved_source_rxs != _MANYTX_REAL_OE_SOURCE_RX_LABELS:
        raise ValueError("known source physical receiver-label resolution drift")
    if resolved_target_rxs != _MANYTX_REAL_OE_TARGET_RX_LABELS:
        raise ValueError("known target physical receiver-label resolution drift")
    if set(train_rxs).intersection(test_rxs):
        raise ValueError("known source and target physical receiver indices overlap")
    return {
        "known_source_day_labels": list(resolved_source_days),
        "known_source_day_indices": [int(value) for value in train_days],
        "known_source_receiver_labels": list(resolved_source_rxs),
        "known_source_receiver_indices": [int(value) for value in train_rxs],
        "known_target_receiver_labels": list(resolved_target_rxs),
        "known_target_receiver_indices": [int(value) for value in test_rxs],
        "known_source_target_receiver_disjoint": True,
    }


def _build_ssdg_wisig_data(args, device: torch.device):
    ds_w = load_wisig_compact_pkl(args.wisig_pkl)
    ds_w, tx_partition_receipt = _phase1_tx_partition_view(
        ds_w,
        train_spec=str(getattr(args, "phase1_source_train_tx_ids", "")),
        known_validation_spec=str(
            getattr(args, "phase1_source_known_validation_tx_ids", "")
        ),
        proxy_unknown_spec=str(
            getattr(args, "phase1_source_proxy_unknown_tx_ids", "")
        ),
        allow_empty_proxy_unknown=bool(
            getattr(args, "phase1_allow_empty_proxy_unknown", False)
            and getattr(args, "manytx_real_oe_protocol_enabled", False)
        ),
    )
    infer_nc = len(ds_w.get("tx_list", []))
    if infer_nc > 0:
        args.num_classes = infer_nc
    eq = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
    day_list = list(ds_w.get("capture_date_list", []))
    rx_list = list(ds_w.get("rx_list", []))
    train_days = _resolve_days(day_list, _parse_wisig_axis_spec(args.wisig_train_days), list(range(min(3, len(day_list)))))
    test_days = _resolve_days(day_list, _parse_wisig_axis_spec(args.wisig_test_days), [len(day_list) - 1])
    train_rxs = _resolve_rxs(rx_list, _parse_wisig_axis_spec(args.wisig_train_rxs), list(range(len(rx_list))))
    test_rxs = _resolve_rxs(rx_list, _parse_wisig_axis_spec(args.wisig_test_rxs), [])
    train_days = [d for d in train_days if d not in test_days]
    train_rxs = [r for r in train_rxs if r not in test_rxs]
    known_physical_receipt = _manytx_known_source_physical_receipt(
        args,
        day_list=day_list,
        rx_list=rx_list,
        train_days=train_days,
        train_rxs=train_rxs,
        test_rxs=test_rxs,
    )

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
    split_fn = split_tx_rx_day_1_6_3 if str(args.split_mode) == "tx_rx_day_1_6_3" else split_tx_rx_day_1_7_2
    labeled_idx, unlabeled_idx, val_idx = split_fn(
        source_base,
        labeled_ratio=float(args.labeled_ratio),
        unlabeled_ratio=float(args.unlabeled_ratio),
        source_val_ratio=float(args.source_val_ratio),
    )
    labeled_ds = WiSigSubsetDataset(source_base, labeled_idx, split_source="ssdg_labeled_tx_visible")
    unlabeled_ds = WiSigSubsetDataset(source_base, unlabeled_idx, split_source="ssdg_unlabeled_tx_hidden")
    val_ds = WiSigSubsetDataset(source_base, val_idx, split_source="ssdg_source_val")

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
        int(args.batch_size),
        bool(getattr(args, "u_unlabeled_shuffle", True)),
        int(args.num_workers),
        device,
        True,
        int(args.prefetch_factor),
    )
    val_loader = make_loader(val_ds, int(args.eval_batch_size), False, int(args.num_workers), device, False, int(args.prefetch_factor))
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
        requested_labeled_ratio=float(args.labeled_ratio),
        requested_unlabeled_ratio=float(args.unlabeled_ratio),
        requested_source_val_ratio=float(args.source_val_ratio),
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
        "val_loader": val_loader,
        "named_test_loaders": named_test_loaders,
        "domain_label_map": domain_label_map,
        "num_domains": max(1, len(domain_label_map)),
        "input_len": int(args.wisig_out_len),
        "num_classes": int(infer_nc),
        "class_id_to_tx": [str(value) for value in list(getattr(source_base, "tx_list", []) or [])],
        "split_info": {
            "mode": str(args.split_mode),
            "labeled_size": len(labeled_ds),
            "unlabeled_size": len(unlabeled_ds),
            "source_val_size": len(val_ds),
            "rho_label": float(len(labeled_ds)) / float(max(1, len(labeled_ds) + len(unlabeled_ds))),
            "balanced_sampler_active": bool(balanced_sampler is not None),
            "balanced_sampler_batch_size": int(balanced_sampler.batch_size) if balanced_sampler is not None else int(args.batch_size),
            "test": test_split_info,
            "named_test_meta": named_meta,
            "source_split_receipt": split_receipt,
            "tx_partition_receipt": tx_partition_receipt,
            "manytx_known_physical_receipt": known_physical_receipt,
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
                "formal Phase1 export must use the "
                "source-validation-selected checkpoint"
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
                data_ctx["val_loader"],
                device=device,
                feature_key=str(getattr(args, "phase2_export_feature_key", "z_id") or "z_id"),
                max_batches=int(getattr(args, "phase2_export_max_batches", 0) or 0),
            )
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


def _resolve_frozen_phase1_evaluation(
    args,
    model,
    data_ctx,
    device,
    checkpoint_path: str | Path,
    *,
    technical_only: bool,
    selection_source: str,
) -> Dict[str, Any]:
    """Return the only legal frozen-evaluation receipt for the run mode."""

    if bool(technical_only):
        return {
            "status": "SKIPPED_TECHNICAL_AUDIT",
            "selection_source": "TECHNICAL_ONLY",
            "claim": "NO_PERFORMANCE_RESULT",
        }
    try:
        return _evaluate_frozen_phase1_checkpoint(args, model, data_ctx, device, checkpoint_path)
    except Exception as exc:
        return {
            "status": "FAILED",
            "reason": str(exc),
            "selection_source": str(selection_source),
            "checkpoint": str(checkpoint_path),
            "claim": "PHASE1_FROZEN_HELDOUT_EVAL_INCOMPLETE",
        }


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
    p0_mechanisms_ready: bool = True,
    p1_mechanisms_ready: bool = True,
    endpoint_export_ready: bool = True,
    technical_only: bool = False,
) -> str:
    """Resolve a fail-closed Phase1 terminal state without overstating promotion readiness."""

    if tail_stopped:
        return "STOPPED_TAIL"
    if export_failed:
        return "FAILED_EXPORT"
    if not selected_checkpoint_exists:
        return "NO_SAFE_CHECKPOINT"
    if technical_only:
        return "TECHNICAL_AUDIT_COMPLETE"
    if final_blocked:
        return "NON_PROMOTABLE_GUARD_BLOCKED"
    if not p0_mechanisms_ready:
        return "NON_PROMOTABLE_P0_DISABLED"
    if not p1_mechanisms_ready:
        return "NON_PROMOTABLE_P1_DISABLED"
    if str(heldout_eval_status).upper() != "COMPLETE":
        return "HELDOUT_EVAL_INCOMPLETE"
    if not endpoint_export_ready:
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
        "source_validation_only_selection": (
            bool(getattr(args, "phase1_source_val_selection_only", False))
            and str(getattr(args, "checkpoint_selection", ""))
            == "source_validation_only"
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
        "selected_checkpoint_is_source_validation": selected_checkpoint.name
        == "best_source_validation_ssdg.pth",
        "selected_checkpoint_role_source_validation": str(
            selected_checkpoint_evidence.get("checkpoint_role", "")
        )
        == "source_validation_selected",
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
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor]:
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
        "phase1_ccpc_leo_frozen_mode": bool(getattr(args, "phase1_ccpc_leo_frozen_mode", False)),
        "phase1_ccpc_leo_enabled": bool(getattr(args, "phase1_ccpc_leo_enabled", False)),
        "phase1_ccpc_leo_gradient_audit_only": bool(
            getattr(args, "phase1_ccpc_leo_gradient_audit_only", False)
        ),
        "lambda_ccpc_leo": float(getattr(args, "lambda_ccpc_leo", 0.0)),
        "ccpc_leo_temperature": float(getattr(args, "ccpc_leo_temperature", 0.12)),
        "phase1_pamr_frozen_mode": bool(getattr(args, "phase1_pamr_frozen_mode", False)),
        "phase1_pamr_enabled": bool(getattr(args, "phase1_pamr_enabled", False)),
        "phase1_pamr_audit_only": bool(getattr(args, "phase1_pamr_audit_only", False)),
        "lambda_pamr": float(getattr(args, "lambda_pamr", 0.0)),
        "phase1_cb_sfce_frozen_mode": bool(getattr(args, "phase1_cb_sfce_frozen_mode", False)),
        "phase1_cb_sfce_enabled": bool(getattr(args, "phase1_cb_sfce_enabled", False)),
        "lambda_cb_sfce": float(getattr(args, "lambda_cb_sfce", 0.0)),
        "cb_sfce_gamma": float(getattr(args, "cb_sfce_gamma", 1.0)),
        "phase1_gd_proto_nll_frozen_mode": bool(getattr(args, "phase1_gd_proto_nll_frozen_mode", False)),
        "phase1_gd_proto_nll_enabled": bool(getattr(args, "phase1_gd_proto_nll_enabled", False)),
        "lambda_gd_proto_nll": float(getattr(args, "lambda_gd_proto_nll", 0.0)),
        "gd_proto_nll_gamma": float(getattr(args, "gd_proto_nll_gamma", 1.0)),
        "phase1_cp_sfce_frozen_mode": bool(getattr(args, "phase1_cp_sfce_frozen_mode", False)),
        "phase1_cp_sfce_enabled": bool(getattr(args, "phase1_cp_sfce_enabled", False)),
        "lambda_cp_sfce": float(getattr(args, "lambda_cp_sfce", 0.0)),
        "cp_sfce_gamma": float(getattr(args, "cp_sfce_gamma", 1.0)),
        "sat_train_scenario": str(getattr(args, "sat_train_scenario", "")),
        "sat_train_scenarios": ",".join(getattr(args, "sat_train_scenario_list", []) or []),
        "sat_view_schedule": str(getattr(args, "sat_view_schedule", "") or ""),
        "use_concat_sat_channel_aug": bool(getattr(args, "use_concat_sat_channel_aug", False)),
        "concat_sat_ce_only": bool(getattr(args, "concat_sat_ce_only", False)),
        "concat_sat_deduplicate_tx_ce": bool(getattr(args, "concat_sat_deduplicate_tx_ce", False)),
        "concat_sat_teacher_clean_only": bool(getattr(args, "concat_sat_teacher_clean_only", False)),
        "id_feature_key": str(getattr(args, "id_feature_key", "feat_joint")),
        "ow_feat_key": str(getattr(args, "ow_feat_key", "z_id")),
        "manytx_real_oe_protocol_enabled": bool(
            getattr(args, "manytx_real_oe_protocol_enabled", False)
        ),
        "manytx_real_oe_enabled": bool(getattr(args, "manytx_real_oe_enabled", False)),
        "manytx_real_oe_partition_root_sha256": str(
            getattr(args, "manytx_real_oe_partition_root_sha256", "")
        ),
        "os_budget_scope": str(getattr(args, "os_budget_scope", "all_shared")),
        "direct_metric_hierarchical_combine": str(
            getattr(args, "direct_metric_hierarchical_combine", "product")
        ),
        "eval_sat_channel": bool(getattr(args, "eval_sat_channel", False)),
        "eval_sat_scenarios": str(getattr(args, "eval_sat_scenarios", "")),
        "nonfinite_train_metric_count": _count_nonfinite(train_logs),
        "nonfinite_val_metric_count": _count_nonfinite(val_stats),
        "nonfinite_test_metric_count": _count_nonfinite(test_stats),
    }
    _flatten_telemetry(row, "protected", protected_metrics or {})
    _flatten_telemetry(row, "joint_guard", guard_state or {})
    _flatten_telemetry(row, "phase2_audit", phase2_audit_state or {})
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
        "lambda_manytx_real_oe",
        "manytx_real_oe_start_epoch",
        "manytx_real_oe_warmup_epochs",
        "manytx_real_oe_temperature",
        "manytx_real_oe_margin",
        "manytx_real_oe_tau",
        "manytx_real_oe_tx_per_batch",
        "manytx_real_oe_samples_per_tx",
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
    loss_ccpc = _log_value(train_logs, "train/loss_ccpc_leo")
    loss_pamr = _log_value(train_logs, "train/loss_pamr")
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
    w_ccpc = _log_value(train_logs, "train/w_loss_ccpc_leo", loss_ccpc)
    w_pamr = _log_value(train_logs, "train/w_loss_pamr", loss_pamr)
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
    lines.append(
        "[CCPC-LEO] "
        f"enabled={int(_log_value(train_logs, 'train/ccpc_enabled', 0.0) >= 0.5)} "
        f"loss={loss_ccpc:.6f} weighted={w_ccpc:.6f} "
        f"rows={int(round(_log_value(train_logs, 'train/ccpc_rows', 0.0)))} "
        f"classes={int(round(_log_value(train_logs, 'train/ccpc_classes', 0.0)))} "
        f"positive_pairs={int(round(_log_value(train_logs, 'train/ccpc_positive_pairs', 0.0)))} "
        f"clean_detached={int(_log_value(train_logs, 'train/ccpc_clean_detached', 0.0) >= 0.5)} "
        f"leo_grad_nonzero={int(_log_value(train_logs, 'train/ccpc_leo_grad_nonzero', 0.0) >= 0.5)} "
        f"leo_grad_zero={int(_log_value(train_logs, 'train/ccpc_leo_grad_zero', 0.0) >= 0.5)} "
        f"leo_grad_nonfinite={int(_log_value(train_logs, 'train/ccpc_leo_grad_nonfinite', 0.0) >= 0.5)}"
    )
    lines.append(
        "[P1-PAMR] "
        f"enabled={int(_log_value(train_logs, 'train/pamr_enabled', 0.0) >= 0.5)} "
        f"loss={loss_pamr:.6f} weighted={w_pamr:.6f} "
        f"rows={int(round(_log_value(train_logs, 'train/pamr_rows', 0.0)))} "
        f"classes={int(round(_log_value(train_logs, 'train/pamr_classes', 0.0)))} "
        f"valid_anchors={int(round(_log_value(train_logs, 'train/pamr_valid_anchors', 0.0)))} "
        f"active_hinges={int(round(_log_value(train_logs, 'train/pamr_active_hinges', 0.0)))} "
        f"clean_margin_detached={int(_log_value(train_logs, 'train/pamr_clean_margin_detached', 0.0) >= 0.5)} "
        f"weight_detached={int(_log_value(train_logs, 'train/pamr_class_weight_detached', 0.0) >= 0.5)} "
        f"leo_grad_nonzero={int(_log_value(train_logs, 'train/pamr_leo_grad_nonzero', 0.0) >= 0.5)} "
        f"leo_grad_zero={int(_log_value(train_logs, 'train/pamr_leo_grad_zero', 0.0) >= 0.5)} "
        f"leo_grad_nonfinite={int(_log_value(train_logs, 'train/pamr_leo_grad_nonfinite', 0.0) >= 0.5)} "
        f"shared_cos={_log_value(train_logs, 'train/pamr_shared_grad_cosine'):.6f} "
        f"shared_ratio={_log_value(train_logs, 'train/pamr_shared_grad_norm_ratio'):.6f}"
    )
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
        f"total={pseudo_total} selected={pseudo_selected}/{pseudo_total} correct={pseudo_correct} "
        f"precision={pseudo_precision:.3f}%"
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
    if bool(getattr(args, "concat_sat_ce_only", False)):
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
        }
    )
    return _safe_iq_tensor(concat_batch.x), concat_batch.y, concat_batch.d_raw, None, info


def _persist_ccpc_failure_receipt(
    *,
    out_dir: Path,
    args: Any,
    ccpc_receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str = "pre_scaled_backward_unscaled_ccpc_gradient_audit",
) -> Optional[Path]:
    """Best-effort persistence that never masks the primary CCPC failure."""

    def _emit_writer_failure(exception_type: str) -> None:
        try:
            print(
                "[CCPC-LEO-FAILURE-RECEIPT] persistence_failed "
                f"writer_exception_type={exception_type}",
                flush=True,
            )
        except Exception:
            pass

    if write_ccpc_failure_receipt is None:
        _emit_writer_failure("ImportError")
        return None
    try:
        return write_ccpc_failure_receipt(
            out_dir,
            candidate_id=str(getattr(args, "candidate_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            receipt=ccpc_receipt,
            error=error,
            failure_stage=str(failure_stage),
        )
    except Exception as receipt_error:
        _emit_writer_failure(type(receipt_error).__name__)
        return None


def _persist_pamr_failure_receipt(
    *,
    out_dir: Path,
    args: Any,
    pamr_receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Optional[Path]:
    """Best-effort persistence that never masks the primary PAMR failure."""

    def _emit_writer_failure(exception_type: str) -> None:
        try:
            print(
                "[P1-PAMR-FAILURE-RECEIPT] persistence_failed "
                f"writer_exception_type={exception_type}",
                flush=True,
            )
        except Exception:
            pass

    if write_pamr_failure_receipt is None:
        _emit_writer_failure("ImportError")
        return None
    try:
        return write_pamr_failure_receipt(
            out_dir,
            candidate_id=str(getattr(args, "candidate_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            receipt=pamr_receipt,
            error=error,
            failure_stage=str(failure_stage),
        )
    except Exception as receipt_error:
        _emit_writer_failure(type(receipt_error).__name__)
        return None


def _persist_cb_sfce_failure_receipt(
    *,
    out_dir: Path,
    args: Any,
    cb_sfce_receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Optional[Path]:
    """Best-effort persistence that never masks the primary CB-SFCE failure."""

    def _emit_writer_failure(exception_type: str) -> None:
        try:
            print(
                "[P1-CB-SFCE-FAILURE-RECEIPT] persistence_failed "
                f"writer_exception_type={exception_type}",
                flush=True,
            )
        except Exception:
            pass

    if write_cb_sfce_failure_receipt is None:
        _emit_writer_failure("ImportError")
        return None
    try:
        return write_cb_sfce_failure_receipt(
            out_dir,
            candidate_id=str(getattr(args, "candidate_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            receipt=cb_sfce_receipt,
            error=error,
            failure_stage=str(failure_stage),
        )
    except Exception as receipt_error:
        _emit_writer_failure(type(receipt_error).__name__)
        return None


def _persist_gd_proto_nll_failure_receipt(
    *,
    out_dir: Path,
    args: Any,
    gd_proto_nll_receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Optional[Path]:
    """Best-effort persistence that never masks the primary GD-ProtoNLL failure."""

    def _emit_writer_failure(exception_type: str) -> None:
        try:
            print(
                "[P1-GD-PROTO-NLL-FAILURE-RECEIPT] persistence_failed "
                f"writer_exception_type={exception_type}",
                flush=True,
            )
        except Exception:
            pass

    if write_gd_proto_nll_failure_receipt is None:
        _emit_writer_failure("ImportError")
        return None
    try:
        return write_gd_proto_nll_failure_receipt(
            out_dir,
            candidate_id=str(getattr(args, "candidate_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            receipt=gd_proto_nll_receipt,
            error=error,
            failure_stage=str(failure_stage),
        )
    except Exception as receipt_error:
        _emit_writer_failure(type(receipt_error).__name__)
        return None


def _persist_cp_sfce_failure_receipt(
    *,
    out_dir: Path,
    args: Any,
    cp_sfce_receipt: Mapping[str, Any],
    error: BaseException,
    failure_stage: str,
) -> Optional[Path]:
    """Best-effort persistence that never masks the primary CP-SFCE failure."""

    def _emit_writer_failure(exception_type: str) -> None:
        try:
            print(
                "[P1-CP-SFCE-FAILURE-RECEIPT] persistence_failed "
                f"writer_exception_type={exception_type}",
                flush=True,
            )
        except Exception:
            pass

    if write_cp_sfce_failure_receipt is None:
        _emit_writer_failure("ImportError")
        return None
    try:
        return write_cp_sfce_failure_receipt(
            out_dir,
            candidate_id=str(getattr(args, "candidate_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            receipt=cp_sfce_receipt,
            error=error,
            failure_stage=str(failure_stage),
        )
    except Exception as receipt_error:
        _emit_writer_failure(type(receipt_error).__name__)
        return None


def _pamr_technical_audit_skip_receipt(scope: str) -> Dict[str, str]:
    """Fixed non-performance placeholder for the isolated PAMR audit path."""

    return {
        "status": "SKIPPED_TECHNICAL_AUDIT",
        "selection_source": "TECHNICAL_ONLY",
        "claim": "NO_PERFORMANCE_RESULT",
        "scope": str(scope),
    }


def train(args) -> int:
    training_wall_started = time.time()
    ablation_manifest = None
    if bool(getattr(args, "formal_ablation", False)):
        ablation_manifest = apply_phase1_ablation(args)
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
    args.epochs = total_epochs
    if validate_ccpc_leo_args is None or ccpc_config_receipt is None or strict_ccpc_warm_start is None:
        if bool(getattr(args, "phase1_ccpc_leo_frozen_mode", False)) or bool(
            getattr(args, "phase1_ccpc_leo_enabled", False)
        ):
            raise ImportError("cvsrffi.phase1_ccpc_leo is required for P1-CCPC-LEO")
        ccpc_config = None
        ccpc_receipt: Dict[str, Any] = {
            "schema": "cvs.phase1.ccpc_leo_receipt.v3",
            "frozen_mode": False,
            "enabled": False,
            "lambda": 0.0,
            "temperature": float(getattr(args, "ccpc_leo_temperature", 0.12)),
            "gradient_audit_only": False,
            "gradient_audit_method": "AUTOGRAD_UNSCALED_CCPC_FEATURE_V1",
            "technical_only": False,
            "performance_result_available": False,
            "technical_only_claim": "",
            "rows": 0,
            "classes": 0,
            "positive_pairs": 0,
            "clean_detached": False,
            "leo_grad_nonzero": False,
            "ccpc_batches": 0,
            "ccpc_grad_nonzero_batches": 0,
            "ccpc_grad_zero_batches": 0,
            "ccpc_grad_nonfinite_batches": 0,
            "ccpc_param_grad_finite_batches": 0,
            "ccpc_param_grad_nonfinite_batches": 0,
            "ccpc_optimizer_step_applied_batches": 0,
            "ccpc_optimizer_step_not_applied_batches": 0,
            "ccpc_terminal_gradient_contract": "PENDING",
            "ccpc_terminal_gradient_contract_passed": False,
            "proxy_rows": 0,
            "held_rows": 0,
        }
    else:
        ccpc_config = validate_ccpc_leo_args(args)
        ccpc_receipt = ccpc_config_receipt(ccpc_config)
    ccpc_frozen_mode = bool(getattr(ccpc_config, "frozen_mode", False))
    ccpc_gradient_audit_only = bool(
        getattr(ccpc_config, "gradient_audit_only", False)
    )
    if (
        validate_pamr_args is None
        or pamr_config_receipt is None
        or strict_pamr_warm_start is None
    ):
        if bool(getattr(args, "phase1_pamr_frozen_mode", False)) or bool(
            getattr(args, "phase1_pamr_enabled", False)
        ):
            raise ImportError("cvsrffi.phase1_pamr is required for P1-PAMR")
        pamr_config = None
        pamr_receipt: Dict[str, Any] = {
            "schema": "cvs.phase1.pamr_receipt.v1",
            "frozen_mode": False,
            "enabled": False,
            "lambda": 0.0,
            "audit_only": False,
            "technical_only": False,
            "technical_only_claim": "",
            "pamr_batches": 0,
            "pamr_grad_nonzero_batches": 0,
            "pamr_grad_zero_batches": 0,
            "pamr_grad_nonfinite_batches": 0,
            "pamr_valid_anchors_by_tx": {},
            "pamr_active_hinges_by_tx": {},
            "pamr_terminal_gradient_contract": "PENDING",
            "pamr_terminal_gradient_contract_passed": False,
            "proxy_rows": 0,
            "held_rows": 0,
        }
    else:
        pamr_config = validate_pamr_args(args)
        pamr_receipt = pamr_config_receipt(pamr_config)
    pamr_frozen_mode = bool(getattr(pamr_config, "frozen_mode", False))
    pamr_audit_only = bool(getattr(pamr_config, "audit_only", False))
    pamr_local_to_head_class_ids: Optional[Tuple[int, ...]] = None
    if pamr_audit_only:
        pamr_receipt["evaluation"] = _pamr_technical_audit_skip_receipt(
            "all_source_val_leo_tail_leakage_and_heldout"
        )
    if (
        validate_cb_sfce_args is None
        or cb_sfce_config_receipt is None
        or strict_cb_sfce_warm_start is None
    ):
        if bool(getattr(args, "phase1_cb_sfce_frozen_mode", False)) or bool(
            getattr(args, "phase1_cb_sfce_enabled", False)
        ):
            raise ImportError("cvsrffi.phase1_cb_sfce is required for P1-CB-SFCE")
        cb_sfce_config = None
        cb_sfce_receipt: Dict[str, Any] = {
            "schema": "cvs.phase1.cb_sfce_receipt.v1",
            "frozen_mode": False,
            "enabled": False,
            "lambda": 0.0,
            "gamma": 1.0,
            "cb_sfce_cells": {},
            "cb_sfce_gradient_relation_completed": False,
            "cb_sfce_terminal_contract": "PENDING",
            "cb_sfce_terminal_contract_passed": False,
            "proxy_rows": 0,
            "held_rows": 0,
        }
    else:
        cb_sfce_config = validate_cb_sfce_args(args)
        cb_sfce_receipt = cb_sfce_config_receipt(cb_sfce_config)
    cb_sfce_frozen_mode = bool(getattr(cb_sfce_config, "frozen_mode", False))
    cb_sfce_local_to_head_class_ids: Optional[Tuple[int, ...]] = None
    if (
        validate_gd_proto_nll_args is None
        or gd_proto_nll_config_receipt is None
        or strict_gd_proto_nll_warm_start is None
    ):
        if bool(getattr(args, "phase1_gd_proto_nll_frozen_mode", False)) or bool(
            getattr(args, "phase1_gd_proto_nll_enabled", False)
        ):
            raise ImportError("cvsrffi.phase1_gd_proto_nll is required for P1-GD-ProtoNLL")
        gd_proto_nll_config = None
        gd_proto_nll_receipt: Dict[str, Any] = {
            "schema": "cvs.phase1.gd_proto_nll_receipt.v1",
            "frozen_mode": False,
            "enabled": False,
            "lambda": 0.0,
            "gamma": 1.0,
            "gd_proto_nll_batches": 0,
            "gd_proto_nll_total_rows": 0,
            "gd_proto_nll_valid_rows": 0,
            "gd_proto_nll_zero_rows": 0,
            "gd_proto_nll_all_local4_valid_batches": 0,
            "gd_proto_nll_cells": {},
            "gd_proto_nll_state_update_batches": 0,
            "gd_proto_nll_gradient_relation_completed": False,
            "gd_proto_nll_terminal_contract": "PENDING",
            "gd_proto_nll_terminal_contract_passed": False,
            "proxy_rows": 0,
            "held_rows": 0,
        }
    else:
        gd_proto_nll_config = validate_gd_proto_nll_args(args)
        gd_proto_nll_receipt = gd_proto_nll_config_receipt(gd_proto_nll_config)
    gd_proto_nll_frozen_mode = bool(getattr(gd_proto_nll_config, "frozen_mode", False))
    gd_proto_nll_local_to_head_class_ids: Optional[Tuple[int, ...]] = None
    gd_proto_nll_state: Optional[Dict[str, torch.Tensor]] = None
    if (
        validate_cp_sfce_args is None
        or cp_sfce_config_receipt is None
        or strict_cp_sfce_warm_start is None
    ):
        if bool(getattr(args, "phase1_cp_sfce_frozen_mode", False)) or bool(
            getattr(args, "phase1_cp_sfce_enabled", False)
        ):
            raise ImportError("cvsrffi.phase1_cp_sfce is required for P1-CP-SFCE")
        cp_sfce_config = None
        cp_sfce_receipt: Dict[str, Any] = {
            "schema": "cvs.phase1.cp_sfce_receipt.v1",
            "frozen_mode": False,
            "enabled": False,
            "lambda": 0.0,
            "gamma": 1.0,
            "cp_sfce_cells": {},
            "cp_sfce_terminal_contract": "PENDING",
            "cp_sfce_terminal_contract_passed": False,
            "proxy_rows": 0,
            "held_rows": 0,
        }
    else:
        cp_sfce_config = validate_cp_sfce_args(args)
        cp_sfce_receipt = cp_sfce_config_receipt(cp_sfce_config)
    cp_sfce_frozen_mode = bool(getattr(cp_sfce_config, "frozen_mode", False))
    cp_sfce_local_to_head_class_ids: Optional[Tuple[int, ...]] = None
    args.lambda_dom = float(args.lambda_domain)
    if float(args.tau_conf) > 0.0:
        args.tau_min = float(args.tau_conf)
    if not bool(args.use_unlabeled):
        args.lambda_u = 0.0
    args.sat_train_scenario = str(args.sat_train_scenario or "mixed_orbit").strip().lower().replace("-", "_")
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
    expected_checkpoint_selection = (
        "source_validation_only"
        if bool(getattr(args, "formal_ablation", False))
        else "final_only"
    )
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
        out_dir / "best_source_validation_ssdg.pth",
        out_dir / f"best_{args.best_metric}_ssdg.pth",
        out_dir / "final_ssdg.pth",
        out_dir / "latest_ssdg.pth",
        out_dir / "tail_reference_ssdg.pth",
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
    source_validation_path = out_dir / "best_source_validation_ssdg.pth"
    default_safe_best_name = f"best_{args.best_metric}_ssdg.pth" if _joint_safe_guard_enabled(args) else f"best_{args.best_metric}_safe_ssdg.pth"
    safe_best_path = Path(str(args.safe_best_path).strip()) if str(args.safe_best_path).strip() else out_dir / default_safe_best_name
    safe_latest_path = Path(str(args.safe_latest_path).strip()) if str(args.safe_latest_path).strip() else out_dir / "latest_safe_ssdg.pth"
    phase2_audit_state = _phase2_audit_state(args)
    manytx_real_oe_preflight = _validate_manytx_real_oe_config(args)
    if bool(getattr(args, "phase1_allow_empty_proxy_unknown", False)) and not bool(
        manytx_real_oe_preflight.get("protocol_enabled", False)
    ):
        raise ValueError(
            "--phase1_allow_empty_proxy_unknown requires the frozen ManyTx real-OE protocol"
        )
    data_ctx = _build_ssdg_wisig_data(args, device)
    if ccpc_frozen_mode or pamr_frozen_mode or cb_sfce_frozen_mode or gd_proto_nll_frozen_mode or cp_sfce_frozen_mode:
        tx_partition_receipt = (
            (data_ctx.get("split_info", {}) or {}).get("tx_partition_receipt", {})
        )
        if not bool(tx_partition_receipt.get("enabled", False)):
            if pamr_frozen_mode:
                raise PAMRConfigurationError(
                    "Frozen Phase1 continuation requires an explicit TX-role partition receipt"
                )
            if cb_sfce_frozen_mode:
                raise CBSFCEConfigurationError(
                    "Frozen Phase1 continuation requires an explicit TX-role partition receipt"
                )
            if gd_proto_nll_frozen_mode:
                raise GDProtoNLLConfigurationError(
                    "Frozen Phase1 continuation requires an explicit TX-role partition receipt"
                )
            if cp_sfce_frozen_mode:
                raise CPSFCEConfigurationError(
                    "Frozen Phase1 continuation requires an explicit TX-role partition receipt"
                )
            raise CCPCLEOConfigurationError(
                "Frozen Phase1 continuation requires an explicit TX-role partition receipt"
            )
        if bool(tx_partition_receipt.get("held_tx_loaded_by_training", True)):
            if pamr_frozen_mode:
                raise PAMRConfigurationError(
                    "Frozen Phase1 continuation rejects any held/proxy TX loaded by training"
                )
            if cb_sfce_frozen_mode:
                raise CBSFCEConfigurationError(
                    "Frozen Phase1 continuation rejects any held/proxy TX loaded by training"
                )
            if gd_proto_nll_frozen_mode:
                raise GDProtoNLLConfigurationError(
                    "Frozen Phase1 continuation rejects any held/proxy TX loaded by training"
                )
            if cp_sfce_frozen_mode:
                raise CPSFCEConfigurationError(
                    "Frozen Phase1 continuation rejects any held/proxy TX loaded by training"
                )
            raise CCPCLEOConfigurationError(
                "Frozen Phase1 continuation rejects any held/proxy TX loaded by training"
            )
        frozen_source_roles = {
            "source_train_tx": list(tx_partition_receipt.get("source_known_train_tx", [])),
            "source_known_validation_tx": list(
                tx_partition_receipt.get("source_known_validation_tx", [])
            ),
            "source_proxy_unknown_tx": list(
                tx_partition_receipt.get("source_proxy_unknown_tx", [])
            ),
            "source_partition_sha256": str(tx_partition_receipt.get("partition_sha256", "")),
            "proxy_rows": 0,
            "held_rows": 0,
        }
        if ccpc_frozen_mode:
            ccpc_receipt.update(frozen_source_roles)
        if pamr_frozen_mode:
            local_data_class_count = int(data_ctx.get("num_classes", 0))
            local_tx_order = list(data_ctx.get("class_id_to_tx", []) or [])
            if local_data_class_count != len(local_tx_order):
                raise PAMRConfigurationError(
                    "P1-PAMR data_ctx local class count must equal its local TX class-order receipt"
                )
            if local_tx_order != frozen_source_roles["source_train_tx"]:
                raise PAMRConfigurationError(
                    "P1-PAMR data_ctx local TX class order must equal the source-train TX receipt"
                )
            pamr_receipt.update(
                {
                    **frozen_source_roles,
                    "local_data_class_count": local_data_class_count,
                    "local_tx_class_order": local_tx_order,
                }
            )
        if cb_sfce_frozen_mode:
            local_data_class_count = int(data_ctx.get("num_classes", 0))
            local_tx_order = list(data_ctx.get("class_id_to_tx", []) or [])
            if local_data_class_count != len(local_tx_order):
                raise CBSFCEConfigurationError(
                    "P1-CB-SFCE data_ctx local class count must equal its local TX class-order receipt"
                )
            if local_tx_order != frozen_source_roles["source_train_tx"]:
                raise CBSFCEConfigurationError(
                    "P1-CB-SFCE data_ctx local TX class order must equal the source-train TX receipt"
                )
            cb_sfce_receipt.update(
                {
                    **frozen_source_roles,
                    "local_data_class_count": local_data_class_count,
                    "local_tx_class_order": local_tx_order,
                }
            )
        if gd_proto_nll_frozen_mode:
            local_data_class_count = int(data_ctx.get("num_classes", 0))
            local_tx_order = list(data_ctx.get("class_id_to_tx", []) or [])
            if local_data_class_count != len(local_tx_order):
                raise GDProtoNLLConfigurationError(
                    "P1-GD-ProtoNLL data_ctx local class count must equal its local TX class-order receipt"
                )
            if local_tx_order != frozen_source_roles["source_train_tx"]:
                raise GDProtoNLLConfigurationError(
                    "P1-GD-ProtoNLL data_ctx local TX class order must equal the source-train TX receipt"
                )
            gd_proto_nll_receipt.update(
                {
                    **frozen_source_roles,
                    "local_data_class_count": local_data_class_count,
                    "local_tx_class_order": local_tx_order,
                }
            )
        if cp_sfce_frozen_mode:
            local_data_class_count = int(data_ctx.get("num_classes", 0))
            local_tx_order = list(data_ctx.get("class_id_to_tx", []) or [])
            if local_data_class_count != len(local_tx_order):
                raise CPSFCEConfigurationError(
                    "P1-CP-SFCE data_ctx local class count must equal its local TX class-order receipt"
                )
            if local_tx_order != frozen_source_roles["source_train_tx"]:
                raise CPSFCEConfigurationError(
                    "P1-CP-SFCE data_ctx local TX class order must equal the source-train TX receipt"
                )
            cp_sfce_receipt.update(
                {
                    **frozen_source_roles,
                    "local_data_class_count": local_data_class_count,
                    "local_tx_class_order": local_tx_order,
                }
            )
    manytx_real_oe_data = _build_manytx_real_oe_data(
        args,
        device,
        batches_per_epoch=len(data_ctx["train_loader"]),
    )
    data_ctx["manytx_real_oe_loader"] = manytx_real_oe_data["loader"]
    data_ctx["manytx_real_oe_sampler"] = manytx_real_oe_data["sampler"]
    data_ctx["split_info"]["manytx_real_oe_receipt"] = {
        **manytx_real_oe_data["receipt"],
        "known_physical_rx_receipt": data_ctx["split_info"].get(
            "manytx_known_physical_receipt", {}
        ),
    }
    use_ckpt = bool(str(args.baseline_ckpt).strip()) and not bool(args.from_scratch)
    ckpt = load_checkpoint(args.baseline_ckpt, device) if use_ckpt else {"model": None, "args": {}, "stats": {}, "split_info": None}
    model_args = merge_checkpoint_args(ckpt, args, input_len=int(data_ctx["input_len"]), num_domains=int(data_ctx["num_domains"]))
    model_args = _apply_model_cli_args(model_args, args)
    model = build_baseline_model(model_args, device)
    if use_ckpt:
        if ccpc_frozen_mode:
            ccpc_receipt.update(
                strict_ccpc_warm_start(
                    model,
                    ckpt["model"],
                    baseline_path=str(args.baseline_ckpt),
                    baseline_sha256=_sha256_file(args.baseline_ckpt),
                    checkpoint_epoch=ckpt.get("epoch", -1),
                    checkpoint_role=ckpt.get(
                        "checkpoint_role",
                        ckpt.get("checkpoint_selection", "UNSPECIFIED"),
                    ),
                )
            )
        elif pamr_frozen_mode:
            pamr_receipt.update(
                strict_pamr_warm_start(
                    model,
                    ckpt["model"],
                    baseline_path=str(args.baseline_ckpt),
                    baseline_sha256=_sha256_file(args.baseline_ckpt),
                    checkpoint_epoch=ckpt.get("epoch", -1),
                    checkpoint_role=ckpt.get(
                        "checkpoint_role",
                        ckpt.get("checkpoint_selection", "UNSPECIFIED"),
                    ),
                )
            )
        elif cb_sfce_frozen_mode:
            cb_sfce_receipt.update(
                strict_cb_sfce_warm_start(
                    model,
                    ckpt["model"],
                    baseline_path=str(args.baseline_ckpt),
                    baseline_sha256=_sha256_file(args.baseline_ckpt),
                    checkpoint_epoch=ckpt.get("epoch", -1),
                    checkpoint_role=ckpt.get(
                        "checkpoint_role",
                        ckpt.get("checkpoint_selection", "UNSPECIFIED"),
                    ),
                )
            )
        elif gd_proto_nll_frozen_mode:
            gd_proto_nll_receipt.update(
                strict_gd_proto_nll_warm_start(
                    model,
                    ckpt["model"],
                    baseline_path=str(args.baseline_ckpt),
                    baseline_sha256=_sha256_file(args.baseline_ckpt),
                    checkpoint_epoch=ckpt.get("epoch", -1),
                    checkpoint_role=ckpt.get(
                        "checkpoint_role",
                        ckpt.get("checkpoint_selection", "UNSPECIFIED"),
                    ),
                )
            )
        elif cp_sfce_frozen_mode:
            cp_sfce_receipt.update(
                strict_cp_sfce_warm_start(
                    model,
                    ckpt["model"],
                    baseline_path=str(args.baseline_ckpt),
                    baseline_sha256=_sha256_file(args.baseline_ckpt),
                    checkpoint_epoch=ckpt.get("epoch", -1),
                    checkpoint_role=ckpt.get(
                        "checkpoint_role",
                        ckpt.get("checkpoint_selection", "UNSPECIFIED"),
                    ),
                )
            )
        else:
            model.load_state_dict(ckpt["model"], strict=False)
    if ccpc_frozen_mode:
        (out_dir / "phase1_ccpc_leo_config_receipt.json").write_text(
            json.dumps(ccpc_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if pamr_frozen_mode:
        # Resolve the exact live GeoSat-C head path before training.  The PAMR
        # loss later uses a detached view of this parameter, never a new head.
        if (
            resolve_pamr_classifier_weight is None
            or resolve_pamr_local_head_class_binding is None
            or remap_pamr_local_labels_to_head_rows is None
        ):
            raise ImportError("cvsrffi.phase1_pamr classifier binding support is required")
        pamr_weight = resolve_pamr_classifier_weight(model)
        checkpoint_args = ckpt.get("args", {}) or {}
        if not isinstance(checkpoint_args, Mapping):
            raise PAMRConfigurationError(
                "P1-PAMR strict baseline checkpoint must contain an argument mapping"
            )
        checkpoint_train_tx = [
            item.strip()
            for item in str(checkpoint_args.get("phase1_source_train_tx_ids", "") or "").split(",")
            if item.strip()
        ]
        tx_partition_receipt = (data_ctx.get("split_info", {}) or {}).get(
            "tx_partition_receipt", {}
        )
        pamr_head_binding = resolve_pamr_local_head_class_binding(
            local_class_order=list(data_ctx.get("class_id_to_tx", []) or []),
            source_train_tx=list(tx_partition_receipt.get("source_known_train_tx", []) or []),
            checkpoint_train_tx=checkpoint_train_tx,
            dataset_class_order=list(tx_partition_receipt.get("dataset_tx_order", []) or []),
            local_data_class_count=data_ctx.get("num_classes", 0),
            checkpoint_head_class_count=checkpoint_args.get("num_classes", None),
            live_head_class_count=int(pamr_weight.size(0)),
        )
        pamr_local_to_head_class_ids = tuple(
            int(value) for value in pamr_head_binding["local_to_head_class_ids"]
        )
        pamr_receipt.update(pamr_head_binding)
        (out_dir / "phase1_pamr_config_receipt.json").write_text(
            json.dumps(pamr_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if cb_sfce_frozen_mode:
        if (
            resolve_cb_sfce_classifier_weight is None
            or resolve_cb_sfce_local_head_class_binding is None
            or remap_cb_sfce_local_labels_to_head_rows is None
        ):
            raise ImportError("cvsrffi.phase1_cb_sfce classifier binding support is required")
        cb_sfce_weight = resolve_cb_sfce_classifier_weight(model)
        checkpoint_args = ckpt.get("args", {}) or {}
        if not isinstance(checkpoint_args, Mapping):
            raise CBSFCEConfigurationError(
                "P1-CB-SFCE strict baseline checkpoint must contain an argument mapping"
            )
        checkpoint_train_tx = [
            item.strip()
            for item in str(checkpoint_args.get("phase1_source_train_tx_ids", "") or "").split(",")
            if item.strip()
        ]
        tx_partition_receipt = (data_ctx.get("split_info", {}) or {}).get(
            "tx_partition_receipt", {}
        )
        cb_sfce_head_binding = resolve_cb_sfce_local_head_class_binding(
            local_class_order=list(data_ctx.get("class_id_to_tx", []) or []),
            source_train_tx=list(tx_partition_receipt.get("source_known_train_tx", []) or []),
            checkpoint_train_tx=checkpoint_train_tx,
            dataset_class_order=list(tx_partition_receipt.get("dataset_tx_order", []) or []),
            local_data_class_count=data_ctx.get("num_classes", 0),
            checkpoint_head_class_count=checkpoint_args.get("num_classes", None),
            live_head_class_count=int(cb_sfce_weight.size(0)),
        )
        cb_sfce_local_to_head_class_ids = tuple(
            int(value) for value in cb_sfce_head_binding["local_to_head_class_ids"]
        )
        cb_sfce_receipt.update(cb_sfce_head_binding)
        (out_dir / "phase1_cb_sfce_config_receipt.json").write_text(
            json.dumps(cb_sfce_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if gd_proto_nll_frozen_mode:
        if (
            resolve_gd_proto_nll_classifier_weight is None
            or resolve_gd_proto_nll_local_head_class_binding is None
            or remap_gd_proto_nll_local_labels_to_head_rows is None
            or make_gd_proto_nll_state is None
        ):
            raise ImportError("cvsrffi.phase1_gd_proto_nll classifier binding support is required")
        gd_proto_nll_weight = resolve_gd_proto_nll_classifier_weight(model)
        checkpoint_args = ckpt.get("args", {}) or {}
        if not isinstance(checkpoint_args, Mapping):
            raise GDProtoNLLConfigurationError(
                "P1-GD-ProtoNLL strict baseline checkpoint must contain an argument mapping"
            )
        checkpoint_train_tx = [
            item.strip()
            for item in str(checkpoint_args.get("phase1_source_train_tx_ids", "") or "").split(",")
            if item.strip()
        ]
        tx_partition_receipt = (data_ctx.get("split_info", {}) or {}).get(
            "tx_partition_receipt", {}
        )
        gd_proto_nll_head_binding = resolve_gd_proto_nll_local_head_class_binding(
            local_class_order=list(data_ctx.get("class_id_to_tx", []) or []),
            source_train_tx=list(tx_partition_receipt.get("source_known_train_tx", []) or []),
            checkpoint_train_tx=checkpoint_train_tx,
            dataset_class_order=list(tx_partition_receipt.get("dataset_tx_order", []) or []),
            local_data_class_count=data_ctx.get("num_classes", 0),
            checkpoint_head_class_count=checkpoint_args.get("num_classes", None),
            live_head_class_count=int(gd_proto_nll_weight.size(0)),
        )
        gd_proto_nll_local_to_head_class_ids = tuple(
            int(value) for value in gd_proto_nll_head_binding["local_to_head_class_ids"]
        )
        gd_proto_nll_receipt.update(gd_proto_nll_head_binding)
        gd_proto_nll_state = make_gd_proto_nll_state(device)
        (out_dir / "phase1_gd_proto_nll_config_receipt.json").write_text(
            json.dumps(gd_proto_nll_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if cp_sfce_frozen_mode:
        if (
            resolve_cp_sfce_classifier_weight is None
            or resolve_cp_sfce_local_head_class_binding is None
            or remap_cp_sfce_local_labels_to_head_rows is None
        ):
            raise ImportError("cvsrffi.phase1_cp_sfce classifier binding support is required")
        checkpoint_args = ckpt.get("args", {}) or {}
        if not isinstance(checkpoint_args, Mapping):
            raise CPSFCEConfigurationError(
                "P1-CP-SFCE strict baseline checkpoint must contain an argument mapping"
            )
        cp_sfce_weight = resolve_cp_sfce_classifier_weight(model)
        live_head_rows = int(cp_sfce_weight.size(0))
        checkpoint_train_tx = [
            item.strip()
            for item in str(checkpoint_args.get("phase1_source_train_tx_ids", "") or "").split(",")
            if item.strip()
        ]
        tx_partition_receipt = (data_ctx.get("split_info", {}) or {}).get(
            "tx_partition_receipt", {}
        )
        cp_sfce_head_binding = resolve_cp_sfce_local_head_class_binding(
            local_class_order=list(data_ctx.get("class_id_to_tx", []) or []),
            source_train_tx=list(tx_partition_receipt.get("source_known_train_tx", []) or []),
            checkpoint_train_tx=checkpoint_train_tx,
            dataset_class_order=list(tx_partition_receipt.get("dataset_tx_order", []) or []),
            local_data_class_count=data_ctx.get("num_classes", 0),
            checkpoint_head_class_count=checkpoint_args.get("num_classes", None),
            live_head_class_count=live_head_rows,
        )
        cp_sfce_local_to_head_class_ids = tuple(
            int(value) for value in cp_sfce_head_binding["local_to_head_class_ids"]
        )
        cp_sfce_receipt.update(cp_sfce_head_binding)
        (out_dir / "phase1_cp_sfce_config_receipt.json").write_text(
            json.dumps(cp_sfce_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if bool(args.freeze_backbone):
        for name, param in model.named_parameters():
            param.requires_grad = any(key in name for key in ("cls_head", "dom_head", "adv_head"))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_params = int(sum(p.numel() for p in model.parameters()))
    ema_model = None
    if bool(args.use_ema_teacher):
        ema_model = deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    teacher_model = None
    if _teacher_distill_requested(args):
        if not str(getattr(args, "teacher_ckpt", "")).strip():
            raise ValueError("--teacher_ckpt is required when any teacher distillation weight is non-zero")
        if one_way_kl_from_teacher is None:
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
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
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
                f"L/U/V={data_ctx['split_info']['labeled_size']}/{data_ctx['split_info']['unlabeled_size']}/{data_ctx['split_info']['source_val_size']} "
                f"ratios={float(args.labeled_ratio):.3f}/{float(args.unlabeled_ratio):.3f}/{float(args.source_val_ratio):.3f}",
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
                f"ow_feat_key={str(getattr(args, 'ow_feat_key', 'z_id'))} "
                f"lambda_zid_compact={float(args.lambda_zid_compact):.6g} lambda_proxy_unknown={float(args.lambda_proxy_unknown):.6g} "
                f"lambda_manytx_real_oe={float(getattr(args, 'lambda_manytx_real_oe', 0.0)):.6g} "
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
                "[CONFIG-MANYTX-REAL-OE] "
                f"protocol={int(bool(manytx_real_oe_preflight.get('protocol_enabled', False)))} "
                f"enabled={int(bool(manytx_real_oe_preflight.get('enabled', False)))} "
                f"partition_root={str(manytx_real_oe_preflight.get('partition_root_sha256', '-'))} "
                f"oe_train={len(manytx_real_oe_preflight.get('oe_train_tx', []))} "
                f"proxy={len(manytx_real_oe_preflight.get('proxy_tx', []))} "
                f"reserve={len(manytx_real_oe_preflight.get('reserve_tx', []))} "
                f"locked_target_new={len(manytx_real_oe_preflight.get('locked_target_new_tx', []))} "
                f"known_days={','.join(manytx_real_oe_preflight.get('known_source_day_labels', [])) or '-'} "
                f"known_rxs={','.join(manytx_real_oe_preflight.get('known_source_receiver_labels', [])) or '-'} "
                f"target_rxs={','.join(manytx_real_oe_preflight.get('known_target_receiver_labels', [])) or '-'} "
                f"oe_days={','.join(manytx_real_oe_preflight.get('oe_source_day_labels', [])) or '-'} "
                f"oe_rxs={','.join(manytx_real_oe_preflight.get('oe_source_receiver_labels', [])) or '-'} "
                f"equalized={manytx_real_oe_preflight.get('equalized', '-')} "
                f"start={int(getattr(args, 'manytx_real_oe_start_epoch', 61))} "
                f"warmup={int(getattr(args, 'manytx_real_oe_warmup_epochs', 10))} "
                f"T={float(getattr(args, 'manytx_real_oe_temperature', 1.0)):.6g} "
                f"margin={float(getattr(args, 'manytx_real_oe_margin', 1.0)):.6g} "
                f"tau={float(getattr(args, 'manytx_real_oe_tau', 1.0)):.6g} "
                f"tx_per_batch={int(getattr(args, 'manytx_real_oe_tx_per_batch', 0))} "
                f"samples_per_tx={int(getattr(args, 'manytx_real_oe_samples_per_tx', 0))} "
                f"proxy_loaded=0 locked_loaded=0 reserve_loaded=0 labels_masked=1",
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
                f"sat_view_schedule={getattr(args, 'sat_view_schedule', '') or '<none>'} "
                f"sat_cons_start_epoch={int(args.sat_cons_start_epoch)} eval_sat_channel={int(bool(args.eval_sat_channel))} "
                f"eval_sat_scenarios={args.eval_sat_scenarios}",
                "[CONFIG-CONCAT-SAT] "
                f"enabled={int(concat_sat_aug is not None)} "
                f"mode={'ce_only_aux' if bool(getattr(args, 'concat_sat_ce_only', False)) else 'full_2b_core_domain'} "
                f"scenario_cycle={','.join(concat_sat_aug.scenarios) if concat_sat_aug is not None else '<none>'} "
                f"start_epoch={int(getattr(args, 'concat_sat_start_epoch', 1))} "
                f"view_prob={float(getattr(args, 'sat_view_prob', 1.0)):.3f} "
                f"seed={int(getattr(args, 'sat_view_seed', args.seed))} "
                f"ce_weight={float(getattr(args, 'concat_sat_ce_weight', 1.0)):.3f}",
                "[CONFIG-CCPC-LEO] "
                f"frozen_mode={int(bool(ccpc_receipt.get('frozen_mode', False)))} "
                f"enabled={int(bool(ccpc_receipt.get('enabled', False)))} "
                f"gradient_audit_only={int(bool(ccpc_receipt.get('gradient_audit_only', False)))} "
                f"gradient_audit={str(ccpc_receipt.get('gradient_audit_method', '') or '-')} "
                f"lambda={float(ccpc_receipt.get('lambda', 0.0)):.6g} "
                f"T={float(ccpc_receipt.get('temperature', 0.12)):.6g} "
                f"clean_detached={int(bool(ccpc_receipt.get('clean_detached', False)))} "
                f"baseline_strict={int(bool(ccpc_receipt.get('strict_model_keys', False)))} "
                f"checkpoint_epoch={int(ccpc_receipt.get('checkpoint_epoch', -1))} "
                f"checkpoint_role={str(ccpc_receipt.get('checkpoint_role', '') or '-')} "
                f"optimizer_state_restored={int(bool(ccpc_receipt.get('optimizer_state_restored', False)))} "
                f"rng_state_restored={int(bool(ccpc_receipt.get('rng_state_restored', False)))} "
                f"rx_or_domain_labels=0 proxy_rows=0 held_rows=0",
                "[CONFIG-P1-PAMR] "
                f"frozen_mode={int(bool(pamr_receipt.get('frozen_mode', False)))} "
                f"enabled={int(bool(pamr_receipt.get('enabled', False)))} "
                f"audit_only={int(bool(pamr_receipt.get('audit_only', False)))} "
                f"lambda={float(pamr_receipt.get('lambda', 0.0)):.6g} "
                f"feature_key={str(pamr_receipt.get('id_feature_key', '') or '-')} "
                f"weight_path={str(pamr_receipt.get('class_weight_path', '') or '-')} "
                f"clean_margin_detached={int(bool(pamr_receipt.get('clean_margin_detached', False)))} "
                f"weight_detached={int(bool(pamr_receipt.get('class_weight_detached', False)))} "
                f"baseline_strict={int(bool(pamr_receipt.get('strict_model_keys', False)))} "
                f"checkpoint_epoch={int(pamr_receipt.get('checkpoint_epoch', -1))} "
                f"checkpoint_role={str(pamr_receipt.get('checkpoint_role', '') or '-')} "
                "rx_or_domain_labels=0 proxy_rows=0 held_rows=0",
                "[CONFIG-P1-CB-SFCE] "
                f"frozen_mode={int(bool(cb_sfce_receipt.get('frozen_mode', False)))} "
                f"enabled={int(bool(cb_sfce_receipt.get('enabled', False)))} "
                f"lambda={float(cb_sfce_receipt.get('lambda', 0.0)):.6g} "
                f"gamma={float(cb_sfce_receipt.get('gamma', 1.0)):.6g} "
                f"scenarios={','.join(cb_sfce_receipt.get('satellite_scenarios', [])) or '-'} "
                f"baseline_strict={int(bool(cb_sfce_receipt.get('strict_model_keys', False)))} "
                f"checkpoint_epoch={int(cb_sfce_receipt.get('checkpoint_epoch', -1))} "
                f"checkpoint_role={str(cb_sfce_receipt.get('checkpoint_role', '') or '-')} "
                "rx_or_domain_labels=0 proxy_rows=0 held_rows=0",
                "[CONFIG-P1-GD-PROTO-NLL] "
                f"frozen_mode={int(bool(gd_proto_nll_receipt.get('frozen_mode', False)))} "
                f"enabled={int(bool(gd_proto_nll_receipt.get('enabled', False)))} "
                f"lambda={float(gd_proto_nll_receipt.get('lambda', 0.0)):.6g} "
                f"gamma={float(gd_proto_nll_receipt.get('gamma', 1.0)):.6g} "
                f"beta={float(gd_proto_nll_receipt.get('beta', 0.0)):.6g} "
                f"eta={float(gd_proto_nll_receipt.get('eta_dro', 0.0)):.6g} "
                f"scenarios={','.join(gd_proto_nll_receipt.get('satellite_scenarios', [])) or '-'} "
                f"baseline_strict={int(bool(gd_proto_nll_receipt.get('strict_model_keys', False)))} "
                f"checkpoint_epoch={int(gd_proto_nll_receipt.get('checkpoint_epoch', -1))} "
                f"checkpoint_role={str(gd_proto_nll_receipt.get('checkpoint_role', '') or '-')} "
                "rx_or_domain_labels=0 proxy_rows=0 held_rows=0",
                "[CONFIG-P1-CP-SFCE] "
                f"frozen_mode={int(bool(cp_sfce_receipt.get('frozen_mode', False)))} "
                f"enabled={int(bool(cp_sfce_receipt.get('enabled', False)))} "
                f"lambda={float(cp_sfce_receipt.get('lambda', 0.0)):.6g} "
                f"gamma={float(cp_sfce_receipt.get('gamma', 1.0)):.6g} "
                f"projection={str(cp_sfce_receipt.get('gradient_rule', '') or '-')} "
                f"scenarios={','.join(cp_sfce_receipt.get('satellite_scenarios', [])) or '-'} "
                f"baseline_strict={int(bool(cp_sfce_receipt.get('strict_model_keys', False)))} "
                f"checkpoint_epoch={int(cp_sfce_receipt.get('checkpoint_epoch', -1))} "
                f"checkpoint_role={str(cp_sfce_receipt.get('checkpoint_role', '') or '-')} "
                "rx_or_domain_labels=0 proxy_rows=0 held_rows=0",
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
        f"L/U/V={data_ctx['split_info']['labeled_size']}/{data_ctx['split_info']['unlabeled_size']}/{data_ctx['split_info']['source_val_size']} "
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
    cb_sfce_satellite_step = 0
    cp_sfce_satellite_step = 0
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
        model.train()
        balanced_train_sampler = data_ctx.get("balanced_train_sampler")
        if balanced_train_sampler is not None and hasattr(balanced_train_sampler, "set_epoch"):
            balanced_train_sampler.set_epoch(epoch)
        manytx_real_oe_sampler = data_ctx.get("manytx_real_oe_sampler")
        if manytx_real_oe_sampler is not None and hasattr(manytx_real_oe_sampler, "set_epoch"):
            manytx_real_oe_sampler.set_epoch(epoch)
        epoch_logs = []
        unlabeled_iter = iter(data_ctx["unlabeled_loader"]) if phase == "pseudo" and bool(args.use_unlabeled) else None
        manytx_real_oe_iter = (
            iter(data_ctx["manytx_real_oe_loader"])
            if data_ctx.get("manytx_real_oe_loader") is not None
            else None
        )
        for batch_idx, labeled_batch in enumerate(data_ctx["train_loader"], start=1):
            x_l, y_l, extra_l = move_batch(labeled_batch, device)
            if pamr_frozen_mode and bool(getattr(pamr_config, "enabled", False)):
                try:
                    if pamr_local_to_head_class_ids is None or remap_pamr_local_labels_to_head_rows is None:
                        raise PAMRRuntimeError("P1-PAMR local-to-head class binding is unavailable")
                    y_l = remap_pamr_local_labels_to_head_rows(
                        y_l, pamr_local_to_head_class_ids
                    )
                except Exception as error:
                    _persist_pamr_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        pamr_receipt=pamr_receipt,
                        error=error,
                        failure_stage="local_tx_label_to_live_head_row_binding",
                    )
                    raise
            if cb_sfce_frozen_mode:
                try:
                    if (
                        cb_sfce_local_to_head_class_ids is None
                        or remap_cb_sfce_local_labels_to_head_rows is None
                    ):
                        raise CBSFCERuntimeError(
                            "P1-CB-SFCE local-to-head class binding is unavailable"
                        )
                    y_l = remap_cb_sfce_local_labels_to_head_rows(
                        y_l, cb_sfce_local_to_head_class_ids
                    )
                except Exception as error:
                    _persist_cb_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cb_sfce_receipt=cb_sfce_receipt,
                        error=error,
                        failure_stage="local_tx_label_to_live_head_row_binding",
                    )
                    raise
            if gd_proto_nll_frozen_mode:
                try:
                    if (
                        gd_proto_nll_local_to_head_class_ids is None
                        or remap_gd_proto_nll_local_labels_to_head_rows is None
                    ):
                        raise GDProtoNLLRuntimeError(
                            "P1-GD-ProtoNLL local-to-head class binding is unavailable"
                        )
                    y_l = remap_gd_proto_nll_local_labels_to_head_rows(
                        y_l, gd_proto_nll_local_to_head_class_ids
                    )
                except Exception as error:
                    _persist_gd_proto_nll_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        gd_proto_nll_receipt=gd_proto_nll_receipt,
                        error=error,
                        failure_stage="local_tx_label_to_live_head_row_binding",
                    )
                    raise
            if cp_sfce_frozen_mode:
                try:
                    if (
                        cp_sfce_local_to_head_class_ids is None
                        or remap_cp_sfce_local_labels_to_head_rows is None
                    ):
                        raise CPSFCERuntimeError(
                            "P1-CP-SFCE local-to-head class binding is unavailable"
                        )
                    y_l = remap_cp_sfce_local_labels_to_head_rows(
                        y_l, cp_sfce_local_to_head_class_ids
                    )
                except Exception as error:
                    _persist_cp_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cp_sfce_receipt=cp_sfce_receipt,
                        error=error,
                        failure_stage="local_tx_label_to_live_head_row_binding",
                    )
                    raise
            labeled_clean_count = int(y_l.numel())
            if ccpc_frozen_mode or pamr_frozen_mode or cb_sfce_frozen_mode or cp_sfce_frozen_mode:
                # The frozen C/G experiment is intentionally blind to RX/day/domain
                # metadata.  Frozen candidates receive only source TX labels.
                receiver_l_base = None
                day_l_base = None
                d_l = None
            else:
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
                out_l = model(x_l_main, y_tx=y_l, grl_lambda=1.0, return_aux=True, domain_labels=d_l)
                domain_stats = {"valid": (d_l >= 0) if d_l is not None else None}
                domain_gates = {
                    "dom": d_l is not None and "dom_logits" in out_l and cur_w["dom"] > 0.0,
                    "adv": d_l is not None and "adv_dom_logits" in out_l and cur_w["adv"] > 0.0,
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
                loss_manytx_real_oe_l = z_id_l.sum() * 0.0
                manytx_real_oe_info: Dict[str, float] = {
                    "active": 0.0,
                    "known_energy": float("nan"),
                    "oe_energy": float("nan"),
                    "energy_gap": float("nan"),
                    "batch_size": 0.0,
                    "labels_masked": 0.0,
                }
                manytx_real_oe_stage_scale = _stage_gate_scale(
                    epoch,
                    start_epoch=int(getattr(args, "manytx_real_oe_start_epoch", 61)),
                    warmup_epochs=int(getattr(args, "manytx_real_oe_warmup_epochs", 10)),
                )
                manytx_real_oe_active = (
                    bool(getattr(args, "manytx_real_oe_enabled", False))
                    and float(getattr(args, "lambda_manytx_real_oe", 0.0)) > 0.0
                    and manytx_real_oe_stage_scale > 0.0
                )
                if manytx_real_oe_active:
                    if real_oe_energy_ranking_loss is None:
                        raise ImportError("cvsrffi.losses.real_oe_energy_ranking_loss is required for ManyTx real-OE")
                    if manytx_real_oe_iter is None:
                        raise RuntimeError("ManyTx real-OE is active but its training loader is unavailable")
                    try:
                        oe_batch = next(manytx_real_oe_iter)
                    except StopIteration:
                        manytx_real_oe_iter = iter(data_ctx["manytx_real_oe_loader"])
                        oe_batch = next(manytx_real_oe_iter)
                    x_oe, y_oe, _ = move_batch(oe_batch, device)
                    if y_oe.numel() == 0 or not bool((y_oe == -1).all().item()):
                        raise ValueError("ManyTx real-OE labels must be fully masked to -1 before the model")
                    out_oe = model(
                        x_oe,
                        y_tx=None,
                        grl_lambda=1.0,
                        return_aux=True,
                        domain_labels=None,
                    )
                    loss_manytx_real_oe_l, manytx_real_oe_info = real_oe_energy_ranking_loss(
                        out_l["tx_logits"][:labeled_clean_count],
                        out_oe["tx_logits"],
                        margin=float(getattr(args, "manytx_real_oe_margin", 1.0)),
                        temperature=float(getattr(args, "manytx_real_oe_temperature", 1.0)),
                        tau=float(getattr(args, "manytx_real_oe_tau", 1.0)),
                    )
                    manytx_real_oe_info = {
                        **manytx_real_oe_info,
                        "batch_size": float(y_oe.numel()),
                        "labels_masked": 1.0,
                    }
                loss_zid_invariance_l = z_id_l.sum() * 0.0
                zid_invariance_info: Dict[str, float] = {
                    "active": 0.0,
                    "receiver_active": 0.0,
                    "day_active": 0.0,
                    "channel_active": 0.0,
                    "receiver_loss": 0.0,
                    "day_loss": 0.0,
                    "channel_loss": 0.0,
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
                    ow_feat_z = _select_open_world_feature(
                        out_l,
                        z_id_l,
                        key=str(getattr(args, "ow_feat_key", "z_id")),
                    )
                    loss_open_world_feat_l, ow_feat_info = open_world_feature_space_loss(
                        ow_feat_z,
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
                out_sat = None
                loss_ccpc_leo_l = zero_sat
                loss_pamr_l = zero_sat
                loss_cb_sfce_l = zero_sat
                loss_gd_proto_nll_l = zero_sat
                loss_cp_sfce_l = zero_sat
                cb_sfce_base_loss_l = None
                gd_proto_nll_base_loss_l = None
                cb_sfce_satellite_scenario = ""
                gd_proto_nll_satellite_scenario = ""
                cp_sfce_satellite_scenario = ""
                cb_sfce_batch_info: Dict[str, Any] = {
                    "rows": 0,
                    "classes": 0,
                    "per_tx_rows": {},
                    "per_tx_loss": {},
                    "per_tx_finite": {},
                    "per_tx_nonzero_logit_gradient": {},
                }
                cp_sfce_batch_info: Dict[str, Any] = {
                    "rows": 0,
                    "classes": 0,
                    "per_tx_rows": {},
                    "per_tx_loss": {},
                    "per_tx_finite": {},
                    "per_tx_nonzero_logit_gradient": {},
                }
                gd_proto_nll_batch_info: Dict[str, Any] = {
                    "total_rows": 0,
                    "valid_rows": 0,
                    "zero_rows": 0,
                    "classes": 0,
                    "scenario": "",
                    "per_tx_total_rows": {},
                    "per_tx_valid_rows": {},
                    "per_tx_zero_rows": {},
                    "per_tx_valid_loss": {},
                    "per_tx_finite": {},
                    "per_tx_analytic_nonzero_logit_gradient_witness": {},
                }
                ccpc_batch_info: Dict[str, Any] = {
                    "rows": 0,
                    "classes": 0,
                    "positive_pairs": 0,
                    "clean_detached": False,
                    "leo_grad_nonzero": False,
                }
                ccpc_leo_feature = None
                pamr_leo_feature = None
                pamr_base_loss_l = None
                pamr_batch_info: Dict[str, Any] = {
                    "rows": 0,
                    "classes": 0,
                    "valid_anchors": 0,
                    "active_hinges": 0,
                    "valid_anchors_by_tx": {},
                    "active_hinges_by_tx": {},
                    "clean_margin_detached": False,
                    "class_weight_detached": False,
                    "clean_gate_raw_cosine": False,
                    "tx_equal_aggregation": False,
                }
                use_sat_train = (
                    bool(args.use_sat_consistency)
                    and (not concat_sat_full_batch)
                    and concat_sat_ce_view is None
                    and epoch >= int(args.sat_cons_start_epoch)
                    and (cur_w["sat_cls"] > 0.0 or cur_w["sat_cons"] > 0.0)
                )
                if (
                    concat_sat_full_batch
                    and concat_sat_clean_bsz > 0
                    and int(y_l.numel()) >= 2 * concat_sat_clean_bsz
                    and epoch >= int(args.sat_cons_start_epoch)
                    and (cur_w["sat_cls"] > 0.0 or cur_w["sat_cons"] > 0.0)
                ):
                    clean_slice = slice(0, concat_sat_clean_bsz)
                    sat_slice = slice(concat_sat_clean_bsz, 2 * concat_sat_clean_bsz)
                    sat_logits = out_l["tx_logits"][sat_slice]
                    sat_y = y_l[sat_slice]
                    if cur_w["sat_cls"] > 0.0:
                        loss_sat_cls_l = F.cross_entropy(sat_logits, sat_y)
                    if cur_w["sat_cons"] > 0.0:
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
                    if cb_sfce_frozen_mode:
                        # C/G share an explicit global clear/low/rain cycle.
                        sat_train_scenarios = list(FROZEN_CB_SFCE_SCENARIOS)
                        sat_train_scenario = sat_train_scenarios[
                            cb_sfce_satellite_step % len(sat_train_scenarios)
                        ]
                        cb_sfce_satellite_step += 1
                    elif cp_sfce_frozen_mode:
                        # CP-SFCE retains the identical globally balanced
                        # clear/low/rain training cycle of its CB control.
                        sat_train_scenarios = list(FROZEN_CP_SFCE_SCENARIOS)
                        sat_train_scenario = sat_train_scenarios[
                            cp_sfce_satellite_step % len(sat_train_scenarios)
                        ]
                        cp_sfce_satellite_step += 1
                    else:
                        sat_train_scenario = sat_train_scenarios[
                            (int(epoch) + int(batch_idx) - 2) % max(1, len(sat_train_scenarios))
                        ]
                    with torch.no_grad():
                        x_sat, _ = apply_sat_channel_for_scenario(
                            x_l,
                            str(sat_train_scenario),
                            args,
                            gen=sat_gen,
                            return_meta=False,
                        )
                    out_sat = model(x_sat, y_tx=y_l, grl_lambda=1.0, return_aux=True, domain_labels=d_l)
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
                    if cur_w["sat_cls"] > 0.0:
                        loss_sat_cls_l = float(args.concat_sat_ce_weight) * F.cross_entropy(out_sat["tx_logits"], y_l)
                    if cur_w["sat_cons"] > 0.0:
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
                if bool(getattr(cb_sfce_config, "enabled", False)):
                    try:
                        if out_sat is None:
                            raise CBSFCERuntimeError(
                                "Enabled P1-CB-SFCE requires one single satellite forward per batch"
                            )
                        if cb_sfce_loss is None or validate_cb_sfce_logit_binding is None:
                            raise ImportError("cvsrffi.phase1_cb_sfce loss and binding support is required")
                        validate_cb_sfce_logit_binding(
                            model=model,
                            tx_logits=out_sat["tx_logits"],
                            tx_labels=y_l,
                            expected_class_ids=cb_sfce_receipt.get("expected_tx_class_ids", []),
                        )
                        loss_cb_sfce_l, cb_sfce_batch_info = cb_sfce_loss(
                            out_sat["tx_logits"],
                            y_l,
                            gamma=float(getattr(cb_sfce_config, "gamma", 1.0)),
                        )
                        cb_sfce_satellite_scenario = str(sat_train_scenario)
                    except Exception as error:
                        _persist_cb_sfce_failure_receipt(
                            out_dir=out_dir,
                            args=args,
                            cb_sfce_receipt=cb_sfce_receipt,
                            error=error,
                            failure_stage="satellite_logit_binding_or_cb_sfce_loss",
                        )
                        raise
                if bool(getattr(gd_proto_nll_config, "enabled", False)):
                    try:
                        if out_sat is None or gd_proto_nll_state is None:
                            raise GDProtoNLLRuntimeError(
                                "Enabled P1-GD-ProtoNLL requires one satellite forward and prior DRO state"
                            )
                        if (
                            gd_proto_nll_loss is None
                            or validate_gd_proto_nll_feature_binding is None
                        ):
                            raise ImportError(
                                "cvsrffi.phase1_gd_proto_nll loss and binding support is required"
                            )
                        gd_proto_nll_weight = validate_gd_proto_nll_feature_binding(
                            model=model,
                            satellite_feature=out_sat["z_id"],
                            tx_labels=y_l,
                            expected_class_ids=gd_proto_nll_receipt.get(
                                "expected_tx_class_ids", []
                            ),
                            z_id_key=str(out_sat.get("z_id_key", "")),
                        )
                        loss_gd_proto_nll_l, gd_proto_nll_batch_info = gd_proto_nll_loss(
                            out_sat["z_id"],
                            gd_proto_nll_weight,
                            y_l,
                            scenario=str(sat_train_scenario),
                            state=gd_proto_nll_state,
                            gamma=float(getattr(gd_proto_nll_config, "gamma", 1.0)),
                        )
                        gd_proto_nll_satellite_scenario = str(sat_train_scenario)
                    except Exception as error:
                        _persist_gd_proto_nll_failure_receipt(
                            out_dir=out_dir,
                            args=args,
                            gd_proto_nll_receipt=gd_proto_nll_receipt,
                            error=error,
                            failure_stage="satellite_feat_joint_binding_or_gd_proto_nll_loss",
                        )
                        raise
                if bool(getattr(cp_sfce_config, "enabled", False)):
                    try:
                        if out_sat is None:
                            raise CPSFCERuntimeError(
                                "Enabled P1-CP-SFCE requires one single satellite forward per batch"
                            )
                        if cp_sfce_loss is None or validate_cp_sfce_logit_binding is None:
                            raise ImportError("cvsrffi.phase1_cp_sfce loss and binding support is required")
                        validate_cp_sfce_logit_binding(
                            model=model,
                            tx_logits=out_sat["tx_logits"],
                            tx_labels=y_l,
                            expected_class_ids=cp_sfce_receipt.get("expected_tx_class_ids", []),
                        )
                        loss_cp_sfce_l, cp_sfce_batch_info = cp_sfce_loss(
                            out_sat["tx_logits"],
                            y_l,
                            gamma=float(getattr(cp_sfce_config, "gamma", 1.0)),
                        )
                        cp_sfce_satellite_scenario = str(sat_train_scenario)
                    except Exception as error:
                        _persist_cp_sfce_failure_receipt(
                            out_dir=out_dir,
                            args=args,
                            cp_sfce_receipt=cp_sfce_receipt,
                            error=error,
                            failure_stage="satellite_logit_binding_or_cp_sfce_loss",
                        )
                        raise
                if bool(getattr(ccpc_config, "enabled", False)):
                    if out_sat is None:
                        raise CCPCLEORuntimeError(
                            "Enabled CCPC-LEO requires one paired clean/LEO forward per batch"
                        )
                    if ccpc_leo_loss is None:
                        raise ImportError("cvsrffi.phase1_ccpc_leo.ccpc_leo_loss is required")
                    ccpc_leo_feature = out_sat["z_id"]
                    if not torch.is_tensor(ccpc_leo_feature):
                        raise CCPCLEORuntimeError("CCPC-LEO requires the LEO z_id feature")
                    loss_ccpc_leo_l, ccpc_batch_info = ccpc_leo_loss(
                        ccpc_leo_feature,
                        z_id_l,
                        y_l,
                        temperature=float(getattr(ccpc_config, "temperature", 0.12)),
                    )
                if bool(getattr(pamr_config, "enabled", False)):
                    try:
                        if out_sat is None:
                            raise PAMRRuntimeError(
                                "Enabled P1-PAMR requires one paired clean/LEO forward per batch"
                            )
                        if pamr_loss is None or validate_pamr_binding is None:
                            raise ImportError("cvsrffi.phase1_pamr loss and binding support are required")
                        pamr_weight = validate_pamr_binding(
                            model=model,
                            out_clean=out_l,
                            out_leo=out_sat,
                            tx_labels=y_l,
                        )
                        pamr_leo_feature = out_sat["z_id"]
                        loss_pamr_l, pamr_batch_info = pamr_loss(
                            pamr_leo_feature,
                            z_id_l,
                            y_l,
                            pamr_weight,
                        )
                    except Exception as error:
                        _persist_pamr_failure_receipt(
                            out_dir=out_dir,
                            args=args,
                            pamr_receipt=pamr_receipt,
                            error=error,
                            failure_stage="paired_feature_binding_or_pamr_loss",
                        )
                        raise
                loss_closed_l = (
                    loss_tx_l
                    + cur_w["dom"] * loss_dom_l
                    + cur_w["adv"] * loss_adv_l
                    + cur_w["orth"] * loss_orth_l
                    + cur_w["cons"] * loss_cons_l
                    + cur_w["group_ce"] * loss_group_ce_l
                    + cur_w["fishr"] * loss_fishr_l
                    + cur_w["sat_cls"] * loss_sat_cls_l
                    + cur_w["sat_cons"] * loss_sat_cons_l
                    + (float(args.lambda_teacher_clean_kl) * teacher_scale) * sanitize_loss("teacher_clean_kl", loss_teacher_clean_kl_l, z_id_l, loss_warn_counts)
                    + (float(args.lambda_teacher_sat_kl) * teacher_scale) * sanitize_loss("teacher_sat_kl", loss_teacher_sat_kl_l, z_id_l, loss_warn_counts)
                    + (float(args.lambda_teacher_zid_mse) * teacher_scale) * sanitize_loss("teacher_zid_mse", loss_teacher_zid_mse_l, z_id_l, loss_warn_counts)
                )
                pamr_base_loss_l = loss_closed_l
                cb_sfce_base_loss_l = loss_closed_l
                gd_proto_nll_base_loss_l = loss_closed_l
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
                    + (float(getattr(args, "lambda_manytx_real_oe", 0.0)) * manytx_real_oe_stage_scale)
                    * sanitize_loss("ssdg_manytx_real_oe", loss_manytx_real_oe_l, z_id_l, loss_warn_counts)
                )
                loss_open_source_l = (cur_w["source_episode"] * source_episode_stage_scale) * sanitize_loss(
                    "ssdg_source_episode", loss_source_episode_l, z_id_l, loss_warn_counts
                )
                loss_open_l = loss_open_invariant_l + loss_open_boundary_l + loss_open_source_l
                if add_ccpc_to_loss is not None:
                    loss_closed_l = add_ccpc_to_loss(
                        loss_closed_l,
                        loss_ccpc_leo_l if bool(getattr(ccpc_config, "enabled", False)) else None,
                        ccpc_config,
                    )
                if add_pamr_to_loss is not None:
                    loss_closed_l = add_pamr_to_loss(
                        loss_closed_l,
                        loss_pamr_l if bool(getattr(pamr_config, "enabled", False)) else None,
                        pamr_config,
                    )
                if add_cb_sfce_to_loss is not None:
                    loss_closed_l = add_cb_sfce_to_loss(
                        loss_closed_l,
                        loss_cb_sfce_l if bool(getattr(cb_sfce_config, "enabled", False)) else None,
                        cb_sfce_config,
                    )
                if add_gd_proto_nll_to_loss is not None:
                    loss_closed_l = add_gd_proto_nll_to_loss(
                        loss_closed_l,
                        loss_gd_proto_nll_l
                        if bool(getattr(gd_proto_nll_config, "enabled", False))
                        else None,
                        gd_proto_nll_config,
                    )
                loss_l = loss_closed_l + loss_open_l
                if phase == "pseudo" and bool(args.use_unlabeled):
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
                    domain_pass = domain_mask.float().mean()
                    temporal_pass = temporal_mask.float().mean()
                    strong_pass = strong_mask.float().mean()
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
                loss_closed = (
                    loss_closed_l
                    + float(args.lambda_u) * loss_u
                    + float(args.lambda_ent) * loss_ent
                    + float(args.lambda_u_domain) * sanitize_loss("ssdg_u_domain", loss_u_domain, z_id_l, loss_warn_counts)
                    + float(args.lambda_u_adv) * sanitize_loss("ssdg_u_adv", loss_u_adv, z_id_l, loss_warn_counts)
                    + float(args.lambda_u_sat_cons) * sanitize_loss("ssdg_u_sat_cons", loss_u_sat_cons, z_id_l, loss_warn_counts)
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
            loss_is_finite = bool(torch.isfinite(loss.detach()).item())
            skipped_nonfinite_loss = 0
            skipped_nonfinite_grad = 0
            optimizer_step_applied = False
            ccpc_leo_grad_nonzero = False
            ccpc_leo_grad_zero = False
            ccpc_leo_grad_nonfinite = False
            pamr_leo_grad_nonzero = False
            pamr_leo_grad_zero = False
            pamr_leo_grad_nonfinite = False
            # C never invokes CP projection; keep telemetry total and C-safe.
            cp_sfce_projection_info: Dict[str, Any] = {}
            cp_sfce_amp_overflow_info: Dict[str, Any] = {}
            cp_sfce_amp_overflow_pending = False
            pamr_shared_gradient_info: Dict[str, Any] = {
                "shared_parameter_count": 0.0,
                "base_norm": float("nan"),
                "pamr_norm": float("nan"),
                "cosine": None,
                "norm_ratio": float("nan"),
            }
            cb_sfce_gradient_relation_info: Dict[str, Any] = {
                "shared_encoder": {
                    "parameter_count": 0.0,
                    "base_norm": float("nan"),
                    "cb_sfce_norm": float("nan"),
                    "cosine": None,
                    "norm_ratio": float("nan"),
                },
                "classifier_head": {
                    "parameter_count": 0.0,
                    "base_norm": float("nan"),
                    "cb_sfce_norm": float("nan"),
                    "cosine": None,
                    "norm_ratio": float("nan"),
                },
            }
            gd_proto_nll_gradient_relation_info: Dict[str, Any] = {
                "shared_encoder": {
                    "parameter_count": 0.0,
                    "base_norm": float("nan"),
                    "gd_proto_nll_norm": float("nan"),
                    "cosine": None,
                    "norm_ratio": float("nan"),
                },
                "classifier_head": {
                    "parameter_count": 0.0,
                    "base_norm": float("nan"),
                    "gd_proto_nll_norm": float("nan"),
                    "cosine": None,
                    "norm_ratio": float("nan"),
                },
            }
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
            if loss_is_finite and bool(getattr(ccpc_config, "enabled", False)):
                if (
                    ccpc_leo_unscaled_gradient is None
                    or ccpc_leo_gradient_status is None
                    or require_finite_ccpc_leo_gradient is None
                    or update_ccpc_receipt is None
                ):
                    raise ImportError("cvsrffi.phase1_ccpc_leo unscaled gradient audit support is required")
                try:
                    ccpc_gradient_status = ccpc_leo_gradient_status(
                        ccpc_leo_unscaled_gradient(
                            loss_ccpc_leo_l,
                            ccpc_leo_feature,
                            loss_weight=float(getattr(ccpc_config, "loss_weight", 0.0)),
                        )
                    )
                    ccpc_leo_grad_nonzero = bool(ccpc_gradient_status["nonzero"])
                    ccpc_leo_grad_zero = bool(ccpc_gradient_status["zero"])
                    ccpc_leo_grad_nonfinite = bool(ccpc_gradient_status["nonfinite"])
                    ccpc_receipt = update_ccpc_receipt(
                        ccpc_receipt,
                        ccpc_batch_info,
                        leo_grad_nonzero=ccpc_leo_grad_nonzero,
                        leo_grad_zero=ccpc_leo_grad_zero,
                        leo_grad_nonfinite=ccpc_leo_grad_nonfinite,
                    )
                    require_finite_ccpc_leo_gradient(ccpc_gradient_status)
                except CCPCLEORuntimeError as error:
                    _persist_ccpc_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        ccpc_receipt=ccpc_receipt,
                        error=error,
                        failure_stage="pre_scaled_backward_unscaled_ccpc_gradient_audit",
                    )
                    raise
            if bool(getattr(pamr_config, "enabled", False)):
                try:
                    if update_pamr_receipt is None:
                        raise ImportError("cvsrffi.phase1_pamr coverage receipt support is required")
                    # Every formal G batch proves only its frozen source-TX coverage;
                    # raw autograd diagnostics are intentionally excluded from 40E.
                    pamr_receipt = update_pamr_receipt(pamr_receipt, pamr_batch_info)
                except Exception as error:
                    _persist_pamr_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        pamr_receipt=pamr_receipt,
                        error=error,
                        failure_stage="pamr_forward_coverage_receipt",
                    )
                    raise
            if bool(getattr(cb_sfce_config, "enabled", False)):
                try:
                    if update_cb_sfce_receipt is None:
                        raise ImportError("cvsrffi.phase1_cb_sfce coverage receipt support is required")
                    cb_sfce_receipt = update_cb_sfce_receipt(
                        cb_sfce_receipt,
                        cb_sfce_batch_info,
                        scenario=cb_sfce_satellite_scenario,
                    )
                except Exception as error:
                    _persist_cb_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cb_sfce_receipt=cb_sfce_receipt,
                        error=error,
                        failure_stage="cb_sfce_local_tx_scenario_coverage_receipt",
                    )
                    raise
            if bool(getattr(gd_proto_nll_config, "enabled", False)):
                try:
                    if update_gd_proto_nll_receipt is None:
                        raise ImportError("cvsrffi.phase1_gd_proto_nll coverage receipt support is required")
                    gd_proto_nll_receipt = update_gd_proto_nll_receipt(
                        gd_proto_nll_receipt,
                        gd_proto_nll_batch_info,
                        scenario=gd_proto_nll_satellite_scenario,
                    )
                except Exception as error:
                    _persist_gd_proto_nll_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        gd_proto_nll_receipt=gd_proto_nll_receipt,
                        error=error,
                        failure_stage="gd_proto_nll_local4_scenario_coverage_receipt",
                    )
                    raise
            if bool(getattr(cp_sfce_config, "enabled", False)):
                try:
                    if update_cp_sfce_coverage_receipt is None:
                        raise ImportError("cvsrffi.phase1_cp_sfce coverage receipt support is required")
                    cp_sfce_receipt = update_cp_sfce_coverage_receipt(
                        cp_sfce_receipt,
                        cp_sfce_batch_info,
                        scenario=cp_sfce_satellite_scenario,
                    )
                except Exception as error:
                    _persist_cp_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cp_sfce_receipt=cp_sfce_receipt,
                        error=error,
                        failure_stage="cp_sfce_local_tx_scenario_coverage_receipt",
                    )
                    raise
            if (
                loss_is_finite
                and bool(getattr(cb_sfce_config, "enabled", False))
                and not bool(cb_sfce_receipt.get("cb_sfce_gradient_relation_completed", False))
            ):
                try:
                    if (
                        cb_sfce_shared_encoder_and_head_parameters is None
                        or cb_sfce_shared_gradient_relation is None
                        or update_cb_sfce_gradient_relation_receipt is None
                    ):
                        raise ImportError("cvsrffi.phase1_cb_sfce raw gradient audit support is required")
                    if cb_sfce_base_loss_l is None:
                        raise CBSFCERuntimeError(
                            "P1-CB-SFCE requires the common base loss for first-batch audit"
                        )
                    cb_sfce_gradient_relation_info = cb_sfce_shared_gradient_relation(
                        cb_sfce_base_loss_l,
                        loss_cb_sfce_l,
                        cb_sfce_shared_encoder_and_head_parameters(model),
                        loss_weight=float(getattr(cb_sfce_config, "loss_weight", 0.0)),
                    )
                    cb_sfce_receipt = update_cb_sfce_gradient_relation_receipt(
                        cb_sfce_receipt,
                        cb_sfce_gradient_relation_info,
                    )
                except Exception as error:
                    _persist_cb_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cb_sfce_receipt=cb_sfce_receipt,
                        error=error,
                        failure_stage="pre_scaled_backward_first_effective_cb_sfce_gradient_audit",
                    )
                    raise
            if (
                loss_is_finite
                and bool(getattr(gd_proto_nll_config, "enabled", False))
                and not bool(gd_proto_nll_receipt.get("gd_proto_nll_gradient_relation_completed", False))
            ):
                try:
                    if (
                        gd_proto_nll_shared_encoder_and_head_parameters is None
                        or gd_proto_nll_shared_gradient_relation is None
                        or update_gd_proto_nll_gradient_relation_receipt is None
                    ):
                        raise ImportError("cvsrffi.phase1_gd_proto_nll raw gradient audit support is required")
                    if gd_proto_nll_base_loss_l is None:
                        raise GDProtoNLLRuntimeError(
                            "P1-GD-ProtoNLL requires the common base loss for first-batch audit"
                        )
                    gd_proto_nll_gradient_relation_info = gd_proto_nll_shared_gradient_relation(
                        gd_proto_nll_base_loss_l,
                        loss_gd_proto_nll_l,
                        gd_proto_nll_shared_encoder_and_head_parameters(model),
                        loss_weight=float(getattr(gd_proto_nll_config, "loss_weight", 0.0)),
                    )
                    gd_proto_nll_receipt = update_gd_proto_nll_gradient_relation_receipt(
                        gd_proto_nll_receipt,
                        gd_proto_nll_gradient_relation_info,
                    )
                except Exception as error:
                    _persist_gd_proto_nll_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        gd_proto_nll_receipt=gd_proto_nll_receipt,
                        error=error,
                        failure_stage="pre_scaled_backward_first_effective_gd_proto_nll_gradient_audit",
                    )
                    raise
            pamr_audit_has_effective_batch = (
                int(pamr_batch_info.get("valid_anchors", 0)) > 0
                and int(pamr_batch_info.get("active_hinges", 0)) > 0
            )
            if (
                loss_is_finite
                and bool(getattr(pamr_config, "enabled", False))
                and pamr_audit_only
                and pamr_audit_has_effective_batch
                and not bool(pamr_receipt.get("pamr_gradient_audit_completed", False))
            ):
                try:
                    if (
                        pamr_unscaled_gradient is None
                        or pamr_gradient_status is None
                        or require_finite_pamr_gradient is None
                        or update_pamr_gradient_receipt is None
                        or pamr_shared_encoder_parameters is None
                        or pamr_shared_gradient_relation is None
                        or update_pamr_gradient_relation_receipt is None
                    ):
                        raise ImportError("cvsrffi.phase1_pamr raw gradient audit support is required")
                    if pamr_base_loss_l is None or pamr_leo_feature is None:
                        raise PAMRRuntimeError("P1-PAMR requires paired loss and baseline loss bindings")
                    pamr_gradient_state = pamr_gradient_status(
                        pamr_unscaled_gradient(
                            loss_pamr_l,
                            pamr_leo_feature,
                            loss_weight=float(getattr(pamr_config, "loss_weight", 0.0)),
                        )
                    )
                    pamr_receipt = update_pamr_gradient_receipt(
                        pamr_receipt,
                        leo_grad_nonzero=bool(pamr_gradient_state["nonzero"]),
                        leo_grad_zero=bool(pamr_gradient_state["zero"]),
                        leo_grad_nonfinite=bool(pamr_gradient_state["nonfinite"]),
                    )
                    require_finite_pamr_gradient(pamr_gradient_state)
                    pamr_shared_gradient_info = pamr_shared_gradient_relation(
                        pamr_base_loss_l,
                        loss_pamr_l,
                        pamr_shared_encoder_parameters(model),
                        loss_weight=float(getattr(pamr_config, "loss_weight", 0.0)),
                    )
                    pamr_receipt = update_pamr_gradient_relation_receipt(
                        pamr_receipt,
                        pamr_shared_gradient_info,
                    )
                    pamr_receipt["pamr_gradient_audit_completed"] = True
                except Exception as error:
                    _persist_pamr_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        pamr_receipt=pamr_receipt,
                        error=error,
                        failure_stage="pre_scaled_backward_first_effective_pamr_gradient_audit",
                    )
                    raise
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
                if bool(getattr(cp_sfce_config, "enabled", False)):
                    try:
                        if use_os_control:
                            raise CPSFCERuntimeError(
                                "P1-CP-SFCE forbids stacked open-set gradient control"
                            )
                        if (
                            cp_sfce_scaled_backward_and_project is None
                            or update_cp_sfce_amp_overflow_receipt is None
                        ):
                            raise ImportError(
                                "cvsrffi.phase1_cp_sfce scaled projection support is required"
                            )
                        cp_sfce_projection_info = cp_sfce_scaled_backward_and_project(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            base_loss=loss,
                            sfce_loss=loss_cp_sfce_l,
                            loss_weight=float(getattr(cp_sfce_config, "loss_weight", 0.0)),
                        )
                        if bool(cp_sfce_projection_info.get("amp_overflow_detected", False)):
                            cp_sfce_amp_overflow_info = dict(cp_sfce_projection_info)
                            if cp_sfce_projection_info.get("amp_overflow_recoverable") is not True:
                                cp_sfce_receipt = update_cp_sfce_amp_overflow_receipt(
                                    cp_sfce_receipt,
                                    overflow=cp_sfce_projection_info,
                                )
                                raise CPSFCERuntimeError(
                                    "P1-CP-SFCE raw gradient audit failed after scaled AMP overflow"
                                )
                            cp_sfce_amp_overflow_pending = True
                        else:
                            cp_sfce_projection_info["projection_applied"] = True
                    except Exception as error:
                        _persist_cp_sfce_failure_receipt(
                            out_dir=out_dir,
                            args=args,
                            cp_sfce_receipt=cp_sfce_receipt,
                            error=error,
                            failure_stage="scaled_base_backward_unscale_aux_vjp_projection",
                        )
                        raise
                elif use_os_control:
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
                    if bool(getattr(gd_proto_nll_config, "enabled", False)):
                        try:
                            if (
                                gd_proto_nll_state is None
                                or advance_gd_proto_nll_state is None
                                or update_gd_proto_nll_state_receipt is None
                            ):
                                raise GDProtoNLLRuntimeError(
                                    "P1-GD-ProtoNLL post-backward EMA state support is unavailable"
                                )
                            gd_proto_nll_state = advance_gd_proto_nll_state(
                                gd_proto_nll_state,
                                gd_proto_nll_batch_info,
                            )
                            gd_proto_nll_receipt = update_gd_proto_nll_state_receipt(
                                gd_proto_nll_receipt,
                                gd_proto_nll_state,
                            )
                        except Exception as error:
                            _persist_gd_proto_nll_failure_receipt(
                                out_dir=out_dir,
                                args=args,
                                gd_proto_nll_receipt=gd_proto_nll_receipt,
                                error=error,
                                failure_stage="post_backward_detached_ema_then_full12_softmax",
                            )
                            raise
                if cp_sfce_amp_overflow_pending:
                    try:
                        if (
                            finalize_cp_sfce_amp_overflow_skip is None
                            or update_cp_sfce_amp_overflow_receipt is None
                        ):
                            raise ImportError(
                                "cvsrffi.phase1_cp_sfce AMP overflow recovery support is required"
                            )
                        cp_sfce_amp_skip = finalize_cp_sfce_amp_overflow_skip(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            overflow=cp_sfce_amp_overflow_info,
                        )
                        cp_sfce_receipt = update_cp_sfce_amp_overflow_receipt(
                            cp_sfce_receipt,
                            overflow=cp_sfce_amp_overflow_info,
                            finalized_skip=cp_sfce_amp_skip,
                        )
                    except Exception as error:
                        _persist_cp_sfce_failure_receipt(
                            out_dir=out_dir,
                            args=args,
                            cp_sfce_receipt=cp_sfce_receipt,
                            error=error,
                            failure_stage="raw_finite_amp_overflow_skip_backoff",
                        )
                        raise
                    grad_norm_before_clip = float("nan")
                    grad_total = float("nan")
                    grad_backbone = float("nan")
                    grad_aux = float("nan")
                    grad_domain = float("nan")
                    grads_finite = False
                    skipped_nonfinite_grad = 1
                else:
                    if not bool(getattr(cp_sfce_config, "enabled", False)):
                        scaler.unscale_(optimizer)
                    grad_norm_before_clip = _grad_norm(model)
                    if float(getattr(args, "max_grad_norm", 0.0)) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=float(args.max_grad_norm),
                            error_if_nonfinite=False,
                        )
                    grad_total = _grad_norm(model)
                    grad_backbone = _grad_norm(model, lambda name: "backbone" in name)
                    grad_aux = _grad_norm(model, lambda name: "aux" in name)
                    grad_domain = _grad_norm(model, lambda name: "dom" in name or "domain" in name)
                    grads_finite = _grads_are_finite(model)
                    if grads_finite:
                        scaler.step(optimizer)
                        if bool(getattr(cp_sfce_config, "enabled", False)):
                            try:
                                if (
                                    cp_sfce_capture_optimizer_steps_for_model is None
                                    or update_cp_sfce_optimizer_step_receipt is None
                                    or update_cp_sfce_projection_receipt is None
                                ):
                                    raise ImportError(
                                        "cvsrffi.phase1_cp_sfce optimizer-state receipt support is required"
                                    )
                                cp_sfce_receipt = update_cp_sfce_optimizer_step_receipt(
                                    cp_sfce_receipt,
                                    before=cp_sfce_projection_info.get("optimizer_state_before", {}),
                                    after=cp_sfce_capture_optimizer_steps_for_model(model, optimizer),
                                )
                                cp_sfce_receipt = update_cp_sfce_projection_receipt(
                                    cp_sfce_receipt,
                                    cp_sfce_projection_info,
                                    epoch=epoch,
                                    batch_index=batch_idx,
                                    scenario=cp_sfce_satellite_scenario,
                                    batch_info=cp_sfce_batch_info,
                                )
                            except Exception as error:
                                _persist_cp_sfce_failure_receipt(
                                    out_dir=out_dir,
                                    args=args,
                                    cp_sfce_receipt=cp_sfce_receipt,
                                    error=error,
                                    failure_stage="post_scaler_step_optimizer_state_increment",
                                )
                                raise
                        optimizer_step_applied = True
                        if ema_model is not None:
                            _update_ema_model(ema_model, model, float(args.ema_decay))
                    else:
                        if bool(getattr(cp_sfce_config, "enabled", False)):
                            error = CPSFCERuntimeError(
                                "P1-CP-SFCE combined parameter gradient is non-finite"
                            )
                            _persist_cp_sfce_failure_receipt(
                                out_dir=out_dir,
                                args=args,
                                cp_sfce_receipt=cp_sfce_receipt,
                                error=error,
                                failure_stage="post_projection_combined_gradient_nonfinite",
                            )
                            raise error
                        skipped_nonfinite_grad = 1
                        optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                if bool(getattr(ccpc_config, "enabled", False)):
                    if update_ccpc_optimizer_receipt is None:
                        raise ImportError(
                            "cvsrffi.phase1_ccpc_leo optimizer receipt support is required"
                        )
                    ccpc_receipt = update_ccpc_optimizer_receipt(
                        ccpc_receipt,
                        parameter_grad_finite=bool(grads_finite),
                        optimizer_step_applied=bool(optimizer_step_applied),
                    )
            else:
                if bool(getattr(ccpc_config, "enabled", False)):
                    raise CCPCLEORuntimeError(
                        "CCPC-LEO fail-closed: total loss is non-finite before backward"
                    )
                if bool(getattr(pamr_config, "enabled", False)):
                    error = PAMRRuntimeError(
                        "P1-PAMR fail-closed: total loss is non-finite before backward"
                    )
                    _persist_pamr_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        pamr_receipt=pamr_receipt,
                        error=error,
                        failure_stage="pre_backward_total_loss_nonfinite",
                    )
                    raise error
                if bool(getattr(cb_sfce_config, "enabled", False)):
                    error = CBSFCERuntimeError(
                        "P1-CB-SFCE fail-closed: total loss is non-finite before backward"
                    )
                    _persist_cb_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cb_sfce_receipt=cb_sfce_receipt,
                        error=error,
                        failure_stage="pre_backward_total_loss_nonfinite",
                    )
                    raise error
                if bool(getattr(gd_proto_nll_config, "enabled", False)):
                    error = GDProtoNLLRuntimeError(
                        "P1-GD-ProtoNLL fail-closed: total loss is non-finite before backward"
                    )
                    _persist_gd_proto_nll_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        gd_proto_nll_receipt=gd_proto_nll_receipt,
                        error=error,
                        failure_stage="pre_backward_total_loss_nonfinite",
                    )
                    raise error
                if bool(getattr(cp_sfce_config, "enabled", False)):
                    error = CPSFCERuntimeError(
                        "P1-CP-SFCE fail-closed: common base loss is non-finite before backward"
                    )
                    _persist_cp_sfce_failure_receipt(
                        out_dir=out_dir,
                        args=args,
                        cp_sfce_receipt=cp_sfce_receipt,
                        error=error,
                        failure_stage="pre_backward_common_base_loss_nonfinite",
                    )
                    raise error
                skipped_nonfinite_loss = 1
                optimizer.zero_grad(set_to_none=True)
                grad_norm_before_clip = float("nan")
                grad_total = float("nan")
                grad_backbone = float("nan")
                grad_aux = float("nan")
                grad_domain = float("nan")
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
            epoch_logs.append(
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
                    "train/loss_domain_labeled": loss_dom_l.detach(),
                    "train/loss_adv_labeled": loss_adv_l.detach(),
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
                    "train/loss_manytx_real_oe": loss_manytx_real_oe_l.detach(),
                    "train/manytx_real_oe_active": manytx_real_oe_info.get("active", 0.0),
                    "train/manytx_real_oe_known_energy": manytx_real_oe_info.get("known_energy", float("nan")),
                    "train/manytx_real_oe_energy": manytx_real_oe_info.get("oe_energy", float("nan")),
                    "train/manytx_real_oe_energy_gap": manytx_real_oe_info.get("energy_gap", float("nan")),
                    "train/manytx_real_oe_batch_size": manytx_real_oe_info.get("batch_size", 0.0),
                    "train/manytx_real_oe_labels_masked": manytx_real_oe_info.get("labels_masked", 0.0),
                    "train/loss_soft_unknown_mixup": loss_soft_unknown_mixup_l.detach(),
                    "train/loss_source_episode": loss_source_episode_l.detach(),
                    "train/loss_direct_metric_accept": loss_direct_metric_accept_l.detach(),
                    "train/loss_sat_cls_labeled": loss_sat_cls_l.detach(),
                    "train/loss_sat_cons_labeled": loss_sat_cons_l.detach(),
                    "train/loss_ccpc_leo": loss_ccpc_leo_l.detach(),
                    "train/ccpc_enabled": 1.0 if bool(getattr(ccpc_config, "enabled", False)) else 0.0,
                    "train/ccpc_rows": float(ccpc_batch_info.get("rows", 0)),
                    "train/ccpc_classes": float(ccpc_batch_info.get("classes", 0)),
                    "train/ccpc_positive_pairs": float(ccpc_batch_info.get("positive_pairs", 0)),
                    "train/ccpc_clean_detached": 1.0 if bool(ccpc_batch_info.get("clean_detached", False)) else 0.0,
                    "train/ccpc_leo_grad_nonzero": 1.0 if ccpc_leo_grad_nonzero else 0.0,
                    "train/ccpc_leo_grad_zero": 1.0 if ccpc_leo_grad_zero else 0.0,
                    "train/ccpc_leo_grad_nonfinite": 1.0 if ccpc_leo_grad_nonfinite else 0.0,
                    "train/loss_pamr": loss_pamr_l.detach(),
                    "train/pamr_enabled": 1.0 if bool(getattr(pamr_config, "enabled", False)) else 0.0,
                    "train/pamr_rows": float(pamr_batch_info.get("rows", 0)),
                    "train/pamr_classes": float(pamr_batch_info.get("classes", 0)),
                    "train/pamr_valid_anchors": float(pamr_batch_info.get("valid_anchors", 0)),
                    "train/pamr_active_hinges": float(pamr_batch_info.get("active_hinges", 0)),
                    "train/pamr_clean_margin_detached": 1.0 if bool(pamr_batch_info.get("clean_margin_detached", False)) else 0.0,
                    "train/pamr_class_weight_detached": 1.0 if bool(pamr_batch_info.get("class_weight_detached", False)) else 0.0,
                    "train/pamr_leo_grad_nonzero": 1.0 if pamr_leo_grad_nonzero else 0.0,
                    "train/pamr_leo_grad_zero": 1.0 if pamr_leo_grad_zero else 0.0,
                    "train/pamr_leo_grad_nonfinite": 1.0 if pamr_leo_grad_nonfinite else 0.0,
                    "train/pamr_shared_grad_cosine": (
                        float(pamr_shared_gradient_info["cosine"])
                        if pamr_shared_gradient_info.get("cosine") is not None
                        else float("nan")
                    ),
                    "train/pamr_shared_grad_norm_ratio": float(
                        pamr_shared_gradient_info.get("norm_ratio", float("nan"))
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
                    "train/w_loss_manytx_real_oe": (
                        (float(getattr(args, "lambda_manytx_real_oe", 0.0)) * manytx_real_oe_stage_scale)
                        * loss_manytx_real_oe_l
                    ).detach(),
                    "train/w_loss_soft_unknown_mixup": ((cur_w["soft_unknown_mixup"] * soft_unknown_mixup_stage_scale) * loss_soft_unknown_mixup_l).detach(),
                    "train/w_loss_source_episode": ((cur_w["source_episode"] * source_episode_stage_scale) * loss_source_episode_l).detach(),
                    "train/w_loss_direct_metric_accept": ((cur_w["direct_metric_accept"] * direct_metric_stage_scale) * loss_direct_metric_accept_l).detach(),
                    "train/w_loss_sat_cls_labeled": (cur_w["sat_cls"] * loss_sat_cls_l).detach(),
                    "train/w_loss_sat_cons_labeled": (cur_w["sat_cons"] * loss_sat_cons_l).detach(),
                    "train/w_loss_ccpc_leo": (
                        float(getattr(ccpc_config, "loss_weight", 0.0)) * loss_ccpc_leo_l
                    ).detach(),
                    "train/w_loss_pamr": (
                        float(getattr(pamr_config, "loss_weight", 0.0)) * loss_pamr_l
                    ).detach(),
                    "train/loss_cb_sfce": loss_cb_sfce_l.detach(),
                    "train/cb_sfce_enabled": 1.0 if bool(getattr(cb_sfce_config, "enabled", False)) else 0.0,
                    "train/cb_sfce_rows": float(cb_sfce_batch_info.get("rows", 0)),
                    "train/cb_sfce_classes": float(cb_sfce_batch_info.get("classes", 0)),
                    "train/cb_sfce_gradient_relation_completed": 1.0 if bool(
                        cb_sfce_receipt.get("cb_sfce_gradient_relation_completed", False)
                    ) else 0.0,
                    "train/cb_sfce_encoder_grad_cosine": (
                        float(cb_sfce_gradient_relation_info["shared_encoder"]["cosine"])
                        if cb_sfce_gradient_relation_info["shared_encoder"].get("cosine") is not None
                        else float("nan")
                    ),
                    "train/cb_sfce_encoder_grad_norm_ratio": float(
                        cb_sfce_gradient_relation_info["shared_encoder"].get("norm_ratio", float("nan"))
                    ),
                    "train/cb_sfce_head_grad_cosine": (
                        float(cb_sfce_gradient_relation_info["classifier_head"]["cosine"])
                        if cb_sfce_gradient_relation_info["classifier_head"].get("cosine") is not None
                        else float("nan")
                    ),
                    "train/cb_sfce_head_grad_norm_ratio": float(
                        cb_sfce_gradient_relation_info["classifier_head"].get("norm_ratio", float("nan"))
                    ),
                    "train/w_loss_cb_sfce": (
                        float(getattr(cb_sfce_config, "loss_weight", 0.0)) * loss_cb_sfce_l
                    ).detach(),
                    "train/loss_gd_proto_nll": loss_gd_proto_nll_l.detach(),
                    "train/gd_proto_nll_enabled": 1.0 if bool(getattr(gd_proto_nll_config, "enabled", False)) else 0.0,
                    "train/gd_proto_nll_total_rows": float(gd_proto_nll_batch_info.get("total_rows", 0)),
                    "train/gd_proto_nll_valid_rows": float(gd_proto_nll_batch_info.get("valid_rows", 0)),
                    "train/gd_proto_nll_zero_rows": float(gd_proto_nll_batch_info.get("zero_rows", 0)),
                    "train/gd_proto_nll_classes": float(gd_proto_nll_batch_info.get("classes", 0)),
                    "train/gd_proto_nll_state_updates": float(
                        gd_proto_nll_receipt.get("gd_proto_nll_state_update_batches", 0)
                    ),
                    "train/gd_proto_nll_gradient_relation_completed": 1.0 if bool(
                        gd_proto_nll_receipt.get("gd_proto_nll_gradient_relation_completed", False)
                    ) else 0.0,
                    "train/gd_proto_nll_encoder_grad_cosine": (
                        float(gd_proto_nll_gradient_relation_info["shared_encoder"]["cosine"])
                        if gd_proto_nll_gradient_relation_info["shared_encoder"].get("cosine") is not None
                        else float("nan")
                    ),
                    "train/gd_proto_nll_head_grad_cosine": (
                        float(gd_proto_nll_gradient_relation_info["classifier_head"]["cosine"])
                        if gd_proto_nll_gradient_relation_info["classifier_head"].get("cosine") is not None
                        else float("nan")
                    ),
                    "train/w_loss_gd_proto_nll": (
                        float(getattr(gd_proto_nll_config, "loss_weight", 0.0)) * loss_gd_proto_nll_l
                    ).detach(),
                    "train/loss_cp_sfce": loss_cp_sfce_l.detach(),
                    "train/cp_sfce_enabled": 1.0 if bool(getattr(cp_sfce_config, "enabled", False)) else 0.0,
                    "train/cp_sfce_rows": float(cp_sfce_batch_info.get("rows", 0)),
                    "train/cp_sfce_classes": float(cp_sfce_batch_info.get("classes", 0)),
                    "train/cp_sfce_projection_applied": 1.0 if bool(
                        cp_sfce_projection_info.get("projection_applied", False)
                    ) else 0.0,
                    "train/cp_sfce_first_epoch_marker_completed": 1.0 if bool(
                        cp_sfce_receipt.get("cp_sfce_first_epoch_marker_completed", False)
                    ) else 0.0,
                    "train/cp_sfce_encoder_conflict": 1.0 if bool(
                        cp_sfce_projection_info.get("shared_encoder", {}).get("conflict", False)
                    ) else 0.0,
                    "train/cp_sfce_head_conflict": 1.0 if bool(
                        cp_sfce_projection_info.get("classifier_head", {}).get("conflict", False)
                    ) else 0.0,
                    "train/cp_sfce_encoder_projected_dot": float(
                        cp_sfce_projection_info.get("shared_encoder", {}).get("projected_dot", float("nan"))
                    ),
                    "train/cp_sfce_head_projected_dot": float(
                        cp_sfce_projection_info.get("classifier_head", {}).get("projected_dot", float("nan"))
                    ),
                    "train/w_loss_cp_sfce": (
                        float(getattr(cp_sfce_config, "loss_weight", 0.0)) * loss_cp_sfce_l
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
            )

        if pamr_audit_only:
            val_stats = {
                **_pamr_technical_audit_skip_receipt("source_val_clean"),
                "tx_acc": float("nan"),
                "tx_correct": 0,
                "tx_total": 0,
            }
            current_source_val = float("nan")
            dg_health_drop_pp = 0.0
        else:
            val_stats = evaluate_loader(
                model,
                data_ctx["val_loader"],
                device,
                data_ctx["domain_label_map"],
                max_batches=int(args.eval_max_batches),
            )
            current_source_val = float(val_stats.get("tx_acc", float("nan")))
            if math.isfinite(current_source_val):
                dg_health_best_val = max(float(dg_health_best_val), current_source_val)
                dg_health_drop_pp = max(0.0, float(dg_health_best_val) - current_source_val)
            else:
                dg_health_drop_pp = float("inf")
        if (not pamr_audit_only) and bool(getattr(args, "source_val_dg_health_guard", False)) and epoch >= int(
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
        source_val_heavy_eval_ran = (not pamr_audit_only) and _should_run_source_val_heavy_eval(
            epoch, total_epochs, args
        )
        if source_val_heavy_eval_ran:
            source_val_tail_geometry = _evaluate_source_val_tail_geometry(model, data_ctx, device, args)
            source_val_sat_stats = _evaluate_source_val_sat_if_enabled(model, data_ctx, device, args)
            last_source_val_tail_geometry = deepcopy(source_val_tail_geometry)
            last_source_val_sat_stats = deepcopy(source_val_sat_stats)
            last_source_val_heavy_eval_epoch = int(epoch)
        elif pamr_audit_only:
            source_val_tail_geometry = _pamr_technical_audit_skip_receipt("source_val_tail_geometry")
            source_val_sat_stats = _pamr_technical_audit_skip_receipt("source_val_leo")
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
        if ccpc_frozen_mode:
            payload["ccpc_leo_receipt"] = dict(ccpc_receipt)
        if pamr_frozen_mode:
            payload["pamr_receipt"] = dict(pamr_receipt)
        if cb_sfce_frozen_mode:
            payload["cb_sfce_receipt"] = dict(cb_sfce_receipt)
        if gd_proto_nll_frozen_mode:
            payload["gd_proto_nll_receipt"] = dict(gd_proto_nll_receipt)
        if cp_sfce_frozen_mode:
            payload["cp_sfce_receipt"] = dict(cp_sfce_receipt)
        latest_path = out_dir / "NOT_SAVED_FINAL_ONLY"
        best_path = (
            source_validation_path
            if bool(getattr(args, "formal_ablation", False))
            else final_path
        )
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
            if bool(getattr(args, "formal_ablation", False)):
                selected_payload = deepcopy(payload)
                selected_payload.update(
                    {
                        "checkpoint_role": "source_validation_selected",
                        "checkpoint_selection": "source_validation_only",
                        "selection_source": "source_validation_only",
                        "selection_metric": str(args.best_metric),
                        "selection_score": float(current_best_score),
                        "selection_epoch": int(epoch),
                    }
                )
                save_payload(source_validation_path, selected_payload)
        elapsed = time.time() - t0
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
    if bool(getattr(args, "formal_ablation", False)):
        selected_checkpoint = source_validation_path
        if not selected_checkpoint.is_file():
            raise RuntimeError(
                "formal Phase1 ablation produced no source-validation-selected checkpoint"
            )
        selected_epoch_payload = _validate_phase1_checkpoint_payload(
            load_checkpoint(str(selected_checkpoint), device),
            args,
            selected_checkpoint,
        )
        _load_phase1_checkpoint_strict(
            model,
            selected_epoch_payload,
            selected_checkpoint,
        )
        payload = deepcopy(selected_epoch_payload)
        selected_epoch = int(selected_epoch_payload.get("epoch", best_epoch))
        selected_role = "source_validation_selected"
        selected_source = "source_validation_only"
    else:
        selected_checkpoint = final_path
        selected_epoch = int(epoch)
        selected_role = "training_final_only"
        selected_source = "training_final_only"
    if pamr_audit_only:
        final_source_val_tail = _pamr_technical_audit_skip_receipt("final_source_val_tail_geometry")
        final_zid_leakage_probe = _pamr_technical_audit_skip_receipt("zid_leakage_probe")
        leakage_decision = {
            "fired": False,
            "reason": "SKIPPED_TECHNICAL_AUDIT",
            "details": _pamr_technical_audit_skip_receipt("zid_leakage_probe_gate"),
        }
    else:
        final_source_val_tail = _evaluate_source_val_tail_geometry(model, data_ctx, device, args)
        final_zid_leakage_probe = _evaluate_zid_leakage_probes(model, data_ctx, device, args)
        leakage_decision = _assess_zid_leakage_probe(final_zid_leakage_probe, args)
    if bool(leakage_decision["fired"]):
        phase1_v2_final_blocked = True
        phase1_v2_reasons.extend(str(leakage_decision["reason"]).split(";"))
    if (not pamr_audit_only) and bool(args.direct_metric_require_domain_local_components):
        local_gate_ready = (
            float(final_source_val_tail.get("domain_local_component_gate", 0.0)) > 0.0
            and float(final_source_val_tail.get("global_ball_accept", 1.0)) == 0.0
            and float(final_source_val_tail.get("local_component_count", 0.0)) > 0.0
        )
        if not local_gate_ready:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.append("FINAL_LOCAL_COMPONENT_GEOMETRY_INCOMPLETE")
    if ccpc_frozen_mode:
        if validate_ccpc_terminal_receipt is None:
            raise ImportError("cvsrffi.phase1_ccpc_leo.validate_ccpc_terminal_receipt is required")
        ccpc_receipt = validate_ccpc_terminal_receipt(ccpc_receipt)
    if pamr_frozen_mode:
        if validate_pamr_terminal_receipt is None:
            raise ImportError("cvsrffi.phase1_pamr.validate_pamr_terminal_receipt is required")
        try:
            pamr_receipt = validate_pamr_terminal_receipt(pamr_receipt)
        except Exception as error:
            _persist_pamr_failure_receipt(
                out_dir=out_dir,
                args=args,
                pamr_receipt=pamr_receipt,
                error=error,
                failure_stage="terminal_pamr_receipt_validation",
            )
            raise
    if cb_sfce_frozen_mode:
        if validate_cb_sfce_terminal_receipt is None:
            raise ImportError("cvsrffi.phase1_cb_sfce.validate_cb_sfce_terminal_receipt is required")
        try:
            cb_sfce_receipt = validate_cb_sfce_terminal_receipt(cb_sfce_receipt)
        except Exception as error:
            _persist_cb_sfce_failure_receipt(
                out_dir=out_dir,
                args=args,
                cb_sfce_receipt=cb_sfce_receipt,
                error=error,
                failure_stage="terminal_cb_sfce_receipt_validation",
            )
            raise
    if gd_proto_nll_frozen_mode:
        if validate_gd_proto_nll_terminal_receipt is None:
            raise ImportError("cvsrffi.phase1_gd_proto_nll.validate_gd_proto_nll_terminal_receipt is required")
        try:
            gd_proto_nll_receipt = validate_gd_proto_nll_terminal_receipt(gd_proto_nll_receipt)
        except Exception as error:
            _persist_gd_proto_nll_failure_receipt(
                out_dir=out_dir,
                args=args,
                gd_proto_nll_receipt=gd_proto_nll_receipt,
                error=error,
                failure_stage="terminal_gd_proto_nll_receipt_validation",
            )
            raise
    if cp_sfce_frozen_mode:
        if validate_cp_sfce_terminal_receipt is None:
            raise ImportError("cvsrffi.phase1_cp_sfce.validate_cp_sfce_terminal_receipt is required")
        try:
            cp_sfce_receipt = validate_cp_sfce_terminal_receipt(cp_sfce_receipt)
        except Exception as error:
            _persist_cp_sfce_failure_receipt(
                out_dir=out_dir,
                args=args,
                cp_sfce_receipt=cp_sfce_receipt,
                error=error,
                failure_stage="terminal_cp_sfce_receipt_validation",
            )
            raise

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
    if ccpc_frozen_mode:
        final_payload["ccpc_leo_receipt"] = dict(ccpc_receipt)
    if pamr_frozen_mode:
        final_payload["pamr_receipt"] = dict(pamr_receipt)
    if cb_sfce_frozen_mode:
        final_payload["cb_sfce_receipt"] = dict(cb_sfce_receipt)
    if gd_proto_nll_frozen_mode:
        final_payload["gd_proto_nll_receipt"] = dict(gd_proto_nll_receipt)
    if cp_sfce_frozen_mode:
        final_payload["cp_sfce_receipt"] = dict(cp_sfce_receipt)
    final_payload.setdefault("stats", {})
    final_payload["stats"]["source_val_tail_geometry"] = final_source_val_tail
    final_payload["stats"]["zid_leakage_probe"] = final_zid_leakage_probe
    final_payload["stats"]["satellite_protocol"] = dict(getattr(args, "sat_protocol_manifest", {}) or {})
    final_payload["final_only_evidence"] = {
        "selection_source": "TECHNICAL_ONLY" if pamr_audit_only else selected_source,
        "status": "SKIPPED_TECHNICAL_AUDIT" if pamr_audit_only else "FINAL_ONLY",
        "claim": "NO_PERFORMANCE_RESULT" if pamr_audit_only else "",
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
        "status": "SKIPPED_TECHNICAL_AUDIT" if pamr_audit_only else "FAILED",
        "protocol": "fixed_source_val_multiview_local_component_v2",
        "selection_policy": str(args.checkpoint_selection),
        "selection_source": "TECHNICAL_ONLY" if pamr_audit_only else selected_source,
        "claim": "NO_PERFORMANCE_RESULT" if pamr_audit_only else "",
        "reference": reference_source_val_tail,
        "final": final_source_val_tail,
        "p99_delta_deg": None,
        "tail_cvar_delta_deg": None,
        "proxy_vaccept_delta": None,
        "satellite_protocol_registry_sha256": (args.sat_protocol_manifest or {}).get("registry_sha256", ""),
        "blocks_final_export": False if pamr_audit_only else True,
        "blocks_promotion": False if pamr_audit_only else True,
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
        elif (not pamr_audit_only) and phase1_v2_tail_machine is not None:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.append("REFERENCE_FINAL_TAIL_GEOMETRY_INCOMPLETE")
        elif not pamr_audit_only:
            reference_final_tail_gate.update(
                {
                    "status": "NOT_REQUIRED",
                    "blocks_final_export": False,
                    "blocks_promotion": False,
                }
            )
    except Exception as exc:
        reference_final_tail_gate["reason"] = str(exc)
        if (not pamr_audit_only) and phase1_v2_tail_machine is not None:
            phase1_v2_final_blocked = True
            phase1_v2_reasons.append("REFERENCE_FINAL_TAIL_GEOMETRY_INCOMPLETE")
    if (not pamr_audit_only) and bool(getattr(args, "phase1_v2_hard_gates", False)):
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
        "state": (
            "SOURCE_VALIDATION_SELECTED"
            if bool(getattr(args, "formal_ablation", False))
            else "FINAL_ONLY"
        ),
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
    if ccpc_gradient_audit_only:
        frozen_eval = _resolve_frozen_phase1_evaluation(
            args,
            model,
            data_ctx,
            device,
            selected_checkpoint,
            technical_only=ccpc_gradient_audit_only,
            selection_source=selected_source,
        )
    elif pamr_audit_only:
        frozen_eval = {
            **_pamr_technical_audit_skip_receipt("frozen_phase1_heldout"),
            "promotion_ready": False,
            "performance_result_available": False,
        }
    else:
        frozen_eval = _resolve_frozen_phase1_evaluation(
            args,
            model,
            data_ctx,
            device,
            selected_checkpoint,
            technical_only=False,
            selection_source=selected_source,
        )
    heldout_eval_path = out_dir / "frozen_phase1_heldout_eval.json"
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
    terminal_status = _resolve_phase1_terminal_status(
        tail_stopped=bool(tail_early_stop_requested),
        export_failed=bool(exit_code),
        final_blocked=bool(phase1_v2_final_blocked),
        selected_checkpoint_exists=bool(selected_checkpoint_exists),
        heldout_eval_status=str(frozen_eval.get("status", "")),
        p0_mechanisms_ready=bool(p0_mechanisms_ready),
        p1_mechanisms_ready=bool(p1_mechanisms_ready),
        endpoint_export_ready=bool(endpoint_export_ready),
        technical_only=(
            ccpc_gradient_audit_only if ccpc_gradient_audit_only else pamr_audit_only
        ),
    )
    terminal_exit_code = int(exit_code)
    if terminal_exit_code == 0 and terminal_status not in {
        "COMPLETE",
        "TECHNICAL_AUDIT_COMPLETE",
    }:
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
        "ow_feat_key": str(getattr(args, "ow_feat_key", "z_id")),
        "manytx_real_oe_receipt": (
            data_ctx.get("split_info", {}) or {}
        ).get("manytx_real_oe_receipt", {}),
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
        "technical_only": bool(ccpc_gradient_audit_only) if ccpc_gradient_audit_only else bool(pamr_audit_only),
        "promotion_ready": (
            False
            if ccpc_gradient_audit_only
            else (False if pamr_audit_only else terminal_status == "COMPLETE")
        ),
        "performance_result_available": False,
        "claim": (
            "NO_PERFORMANCE_RESULT"
            if ccpc_gradient_audit_only
            else (
                "NO_PERFORMANCE_RESULT"
                if pamr_audit_only
                else "PHASE1_SOURCE_ONLY_NO_TRUE_UNKNOWN_SUCCESS_CLAIM"
            )
        ),
    }
    if ccpc_frozen_mode:
        terminal_manifest["ccpc_leo_receipt"] = dict(ccpc_receipt)
    if pamr_frozen_mode:
        terminal_manifest["pamr_receipt"] = dict(pamr_receipt)
    if cb_sfce_frozen_mode:
        terminal_manifest["cb_sfce_receipt"] = dict(cb_sfce_receipt)
    if gd_proto_nll_frozen_mode:
        terminal_manifest["gd_proto_nll_receipt"] = dict(gd_proto_nll_receipt)
    if cp_sfce_frozen_mode:
        terminal_manifest["cp_sfce_receipt"] = dict(cp_sfce_receipt)
    (out_dir / "phase1_terminal_status.json").write_text(
        json.dumps(terminal_manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    terminal_manifest_path = out_dir / "phase1_terminal_status.json"
    if ccpc_frozen_mode:
        (out_dir / "phase1_ccpc_leo_terminal_receipt.json").write_text(
            json.dumps(
                {
                    **dict(ccpc_receipt),
                    "terminal_status": terminal_status,
                    "terminal_exit_code": int(terminal_exit_code),
                    "selected_checkpoint": str(selected_checkpoint),
                    "selected_checkpoint_sha256": selected_checkpoint_sha256,
                    "technical_only": bool(ccpc_gradient_audit_only),
                    "promotion_ready": (
                        False
                        if ccpc_gradient_audit_only
                        else terminal_status == "COMPLETE"
                    ),
                    "performance_result_available": False,
                    "claim": (
                        "NO_PERFORMANCE_RESULT"
                        if ccpc_gradient_audit_only
                        else "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if pamr_frozen_mode:
        (out_dir / "phase1_pamr_terminal_receipt.json").write_text(
            json.dumps(
                {
                    **dict(pamr_receipt),
                    "terminal_status": terminal_status,
                    "terminal_exit_code": int(terminal_exit_code),
                    "selected_checkpoint": str(selected_checkpoint),
                    "selected_checkpoint_sha256": selected_checkpoint_sha256,
                    "technical_only": bool(pamr_audit_only),
                    "promotion_ready": (
                        False if pamr_audit_only else terminal_status == "COMPLETE"
                    ),
                    "performance_result_available": False,
                    "claim": (
                        "NO_PERFORMANCE_RESULT"
                        if pamr_audit_only
                        else "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if cb_sfce_frozen_mode:
        (out_dir / "phase1_cb_sfce_terminal_receipt.json").write_text(
            json.dumps(
                {
                    **dict(cb_sfce_receipt),
                    "terminal_status": terminal_status,
                    "terminal_exit_code": int(terminal_exit_code),
                    "selected_checkpoint": str(selected_checkpoint),
                    "selected_checkpoint_sha256": selected_checkpoint_sha256,
                    "technical_only": False,
                    "promotion_ready": terminal_status == "COMPLETE",
                    "performance_result_available": False,
                    "claim": "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if gd_proto_nll_frozen_mode:
        (out_dir / "phase1_gd_proto_nll_terminal_receipt.json").write_text(
            json.dumps(
                {
                    **dict(gd_proto_nll_receipt),
                    "terminal_status": terminal_status,
                    "terminal_exit_code": int(terminal_exit_code),
                    "selected_checkpoint": str(selected_checkpoint),
                    "selected_checkpoint_sha256": selected_checkpoint_sha256,
                    "technical_only": False,
                    "promotion_ready": terminal_status == "COMPLETE",
                    "performance_result_available": False,
                    "claim": "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if cp_sfce_frozen_mode:
        (out_dir / "phase1_cp_sfce_terminal_receipt.json").write_text(
            json.dumps(
                {
                    **dict(cp_sfce_receipt),
                    "terminal_status": terminal_status,
                    "terminal_exit_code": int(terminal_exit_code),
                    "selected_checkpoint": str(selected_checkpoint),
                    "selected_checkpoint_sha256": selected_checkpoint_sha256,
                    "technical_only": False,
                    "promotion_ready": terminal_status == "COMPLETE",
                    "performance_result_available": False,
                    "claim": "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
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
        "ow_feat_key": str(getattr(args, "ow_feat_key", "z_id")),
        "manytx_real_oe_receipt": (
            data_ctx.get("split_info", {}) or {}
        ).get("manytx_real_oe_receipt", {}),
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
        "technical_only": bool(ccpc_gradient_audit_only) if ccpc_gradient_audit_only else bool(pamr_audit_only),
        "deployment_bundle_status": "PENDING_PHASE1_W2_SEAL",
        "formal_performance_claim": False,
        "claim": (
            "NO_PERFORMANCE_RESULT"
            if ccpc_gradient_audit_only
            else (
                "NO_PERFORMANCE_RESULT"
                if pamr_audit_only
                else "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT"
            )
        ),
    }
    if gd_proto_nll_frozen_mode:
        completion_receipt["gd_proto_nll_receipt"] = dict(gd_proto_nll_receipt)
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
        f"promotion_ready={int(bool(terminal_manifest['promotion_ready']))} "
        f"endpoint_export_ready={int(endpoint_export_ready)}",
        flush=True,
    )
    return int(terminal_exit_code)


def main() -> int:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
