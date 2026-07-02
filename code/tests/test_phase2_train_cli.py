from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_phase2_export_train_cli_is_default_off_and_uses_best_primary_checkpoint():
    text = (PROJECT_ROOT / "code" / "train.py").read_text(encoding="utf-8")

    assert 'add_bool_arg(parser, "phase2_export_prototypes", False' in text
    assert '"--phase2_export_path"' in text
    assert '"--phase2_export_feature_key"' in text
    assert '"--phase2_export_checkpoint"' in text
    assert "best_primary_save_path" in text
    assert "export_phase2_prototypes(" in text


def test_train_cli_exposes_default_off_open_world_feature_space_loss():
    text = (PROJECT_ROOT / "code" / "train.py").read_text(encoding="utf-8")

    assert '"--lambda_open_world_feat"' in text
    assert "default=0.0" in text
    assert '"--ow_feat_radius_deg"' in text
    assert '"--ow_feat_inter_margin_deg"' in text
    assert '"--ow_feat_sample_margin_deg"' in text
    assert '"--ow_feat_domain_align_weight"' in text
    assert "open_world_feature_space_loss(" in text


def test_train_cli_exposes_dense_tail_test_eval_schedule():
    text = (PROJECT_ROOT / "code" / "train.py").read_text(encoding="utf-8")

    assert '"--test_eval_final_window"' in text
    assert '"--test_eval_final_interval"' in text
    assert "final_window=args.test_eval_final_window" in text
    assert "final_interval=args.test_eval_final_interval" in text


def test_ssdg_cli_exposes_dense_tail_test_eval_schedule():
    text = (PROJECT_ROOT / "code" / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert '"--test_eval_policy"' in text
    assert '"--test_eval_interval"' in text
    assert '"--test_eval_final_window"' in text
    assert '"--test_eval_final_interval"' in text
    assert "test_eval_skipped_guard_block" in text


def test_ssdg_cli_exposes_default_off_zid_feature_space_bridge_and_export():
    text = (PROJECT_ROOT / "code" / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert '"--use_proto_memory"' in text
    assert '"--lambda_proto"' in text
    assert "default=0.0" in text
    assert '"--lambda_open_world_feat"' in text
    assert '"--ow_feat_start_epoch"' in text
    assert '"--ow_feat_warmup_epochs"' in text
    assert '"--ow_feat_radius_deg"' in text
    assert '"--phase2_export_prototypes"' in text
    assert '"--phase2_export_feature_key"' in text
    assert "PrototypeMemoryBank(" in text
    assert "open_world_feature_space_loss(" in text
    assert "export_phase2_prototypes(" in text
    assert "Non-zero legacy Phase1 prototype/mask/geometry audit losses" in text


def test_ssdg_cli_exposes_default_off_three_sigma_tail_and_fusion_controls():
    text = (PROJECT_ROOT / "code" / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert '"--ow_feat_tail_mode"' in text
    assert 'default="none"' in text
    assert '"--ow_feat_tail_weight"' in text
    assert "default=0.0" in text
    assert '"--ow_feat_soft_gate"' in text
    assert '"--phase2_fuse_prototypes"' in text
    assert "default=False" in text
    assert '"--phase2_fuse_max_components"' in text
    assert '"--phase2_fuse_merge_angle_deg"' in text
    assert '"--lambda_source_episode"' in text
    assert '"--source_episode_start_epoch"' in text
    assert '"--source_episode_warmup_epochs"' in text
    assert '"--source_episode_min_domains"' in text
    assert "fuse_tx_domain_prototypes(" in text
    assert "source_episode_three_sigma_loss(" in text


def test_ssdg_cli_exposes_default_off_v2_compactness_proxy_unknown_and_local_acceptance():
    text = (PROJECT_ROOT / "code" / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert '"--lambda_zid_compact"' in text
    assert '"--zid_compact_start_epoch"' in text
    assert '"--zid_compact_cvar_alpha"' in text
    assert '"--zid_compact_radius_deg"' in text
    assert '"--lambda_proxy_unknown"' in text
    assert '"--proxy_unknown_start_epoch"' in text
    assert '"--proxy_unknown_warmup_epochs"' in text
    assert '"--proxy_unknown_virtual_count"' in text
    assert '"--ow_feat_vacuum_weight"' in text
    assert '"--ow_feat_vacuum_width_deg"' in text
    assert '"--proxy_unknown_vacuum_weight"' in text
    assert '"--proxy_unknown_vacuum_width_deg"' in text
    assert '"--phase2_fuse_accept_policy"' in text
    assert '"--phase2_fuse_global_ball_accept"' in text
    assert "zid_compactness_loss(" in text
    assert "proxy_unknown_energy_loss(" in text
    assert "vacuum_weight=float(args.ow_feat_vacuum_weight)" in text
    assert "vacuum_weight=float(args.proxy_unknown_vacuum_weight)" in text


def test_ssdg_cli_exposes_core_safe_accept_domain_controls():
    text = (PROJECT_ROOT / "code" / "SSDG" / "train_ssdg.py").read_text(encoding="utf-8")

    assert '"--source_episode_radius_mode"' in text
    assert 'default="min_three_sigma_core"' in text
    assert '"core_quantile"' in text
    assert '"min_three_sigma_core"' in text
    assert '"--source_episode_core_quantile"' in text
    assert '"--source_episode_min_sigma_deg"' in text
    assert '"--proxy_unknown_component_radius_mode"' in text
    assert 'default="core_quantile"' in text
    assert '"--proxy_unknown_component_radius_quantile"' in text
    assert '"--phase2_fuse_tail_auto_accept"' in text
    assert "radius_mode=str(args.source_episode_radius_mode)" in text
    assert "component_radius_mode=str(args.proxy_unknown_component_radius_mode)" in text
