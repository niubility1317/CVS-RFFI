from __future__ import annotations
import argparse
import math
from typing import Any, Dict, Mapping, Optional, List, Sequence, Tuple
import torch
from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train two-stage SSDG from a Stable-SAT baseline checkpoint.")
    parser.add_argument("--baseline_ckpt", type=str, default="", help="Optional checkpoint. Empty means train SSDG from scratch.")
    parser.add_argument("--from_scratch", type=str2bool, default=True)
    parser.add_argument("--split_mode", type=str, default="tx_rx_day_1_6_3", choices=["tx_rx_day_1_6_3", "tx_rx_day_1_7_2"])
    parser.add_argument("--labeled_ratio", type=float, default=0.10)
    parser.add_argument("--unlabeled_ratio", type=float, default=0.60)
    parser.add_argument("--source_val_ratio", type=float, default=0.30)
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
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_ent", type=float, default=0.01)
    parser.add_argument("--lambda_domain", "--lambda_dom", dest="lambda_domain", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.45)
    parser.add_argument("--lambda_orth", type=float, default=0.05)
    parser.add_argument("--lambda_cons", type=float, default=0.08)
    parser.add_argument("--lambda_group_ce", type=float, default=0.10)
    parser.add_argument("--group_ce_top_frac", type=float, default=0.35)
    parser.add_argument("--group_ce_min_domains", type=int, default=4)
    parser.add_argument("--group_ce_mode", type=str, default="hard")
    parser.add_argument("--lambda_fishr", type=float, default=0.02)
    parser.add_argument("--fishr_min_domains", type=int, default=4)
    parser.add_argument("--strong_noise_std", type=float, default=0.015)
    parser.add_argument("--use_unlabeled", type=str2bool, default=True)
    parser.add_argument("--pseudo_domain_gate", type=str2bool, default=True)
    parser.add_argument("--pseudo_temporal_gate", type=str2bool, default=True)
    parser.add_argument("--pseudo_temporal_window", type=int, default=2)
    parser.add_argument("--pseudo_temporal_min_conf", type=float, default=0.80)
    parser.add_argument("--pseudo_strong_agreement", type=str2bool, default=True)
    parser.add_argument("--use_ema_teacher", type=str2bool, default=False)
    parser.add_argument("--ema_decay", type=float, default=0.999)
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
    parser.add_argument("--lambda_sat_cls", type=float, default=0.10)
    parser.add_argument("--lambda_sat_cons", type=float, default=0.0)
    parser.add_argument("--sat_cons_start_epoch", type=int, default=20)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="clean_val_tx",
        choices=["clean_val_tx", "test_overall_tx", "sat_mean_tx", "sat_worst_tx", "joint_safe"],
        help="Metric used to update the best SSDG checkpoint.",
    )
    parser.add_argument("--safe_best_path", type=str, default="", help="Optional path for the guarded best checkpoint.")
    parser.add_argument("--safe_latest_path", type=str, default="", help="Optional path for the latest guarded checkpoint.")
    parser.add_argument(
        "--test_eval_policy",
        type=str,
        default="every_epoch",
        choices=["every_epoch", "val_improved_final", "interval_final"],
        help="When to run named test-set and satellite evaluation during SSDG training.",
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
    parser.add_argument("--use_phase2_ground_prototypes", type=str2bool, default=False)
    parser.add_argument("--use_feature_masks", type=str2bool, default=False)
    parser.add_argument("--use_txrx_geometry_losses", type=str2bool, default=False)
    parser.add_argument("--use_tx_rx_balanced_sampler", type=str2bool, default=False)
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
    parser.add_argument("--proxy_unknown_vaccept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_core_accept_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_component_gate_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_tail_quarantine_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_source_safe_weight", type=float, default=0.0)
    parser.add_argument("--proxy_unknown_vaccept_cvar_alpha", type=float, default=0.25)
    parser.add_argument("--proxy_unknown_unknown_margin", type=float, default=0.08)
    parser.add_argument("--proxy_unknown_known_margin", type=float, default=0.05)
    parser.add_argument("--proxy_unknown_energy_softplus_temperature", type=float, default=0.04)
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
    parser.add_argument("--source_episode_mixup_weight", type=float, default=0.0)
    parser.add_argument("--source_episode_mixup_hard_k", type=int, default=2)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--candidate_id", type=str, default="")
    parser.add_argument("--base_candidate", type=str, default="")
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
    parser.add_argument("--phase2_fuse_global_ball_accept", type=str2bool, default=False)
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--model_variant", type=str, default="lite_d")
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
    add_sat_eval_args(parser)
    return parser

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
    }

def _stage_gate_scale(epoch: int, *, start_epoch: int = 1, warmup_epochs: int = 0) -> float:
    start = max(1, int(start_epoch))
    if int(epoch) < start:
        return 0.0
    warm = max(0, int(warmup_epochs))
    if warm <= 0:
        return 1.0
    return min(1.0, max(0.0, float(int(epoch) - start + 1) / float(warm)))

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


def _update_ema_model(ema_model, model, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(float(decay)).add_(p.data, alpha=1.0 - float(decay))
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)
