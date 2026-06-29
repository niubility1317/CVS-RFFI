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
