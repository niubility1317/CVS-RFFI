import json
from pathlib import Path

from paper_reproduction.scripts.run_cvs_extreme_light_matrix import (
    ARMS,
    CORE_ARMS,
    build_rows,
    row_config,
)


def _base_config():
    return {
        "method": "cvs_qknnv42",
        "stage": "Stage2-C",
        "feature_npz_by_scenario": {
            "leo_clear_weak": "clear.npz",
            "leo_low_elev_weak": "low.npz",
            "leo_rain_weak": "rain.npz",
        },
        "target_receiver_labels": ["20-1"],
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "target_old_tx_labels": ["o1", "o2"],
        "target_new_tx_labels": [f"n{i}" for i in range(20)],
        "target_unknown_tx_labels": [],
        "k_shot": 20,
        "support_pool_max_k": 30,
        "query_per_tx": 20,
        "seed": 1,
        "split_seed": 1,
        "unknown_rejection_enabled": False,
        "qknnv42_aux_feature_key": "fft_logmag_features",
        "qknnv42_aux_feature_dim": 96,
        "qknnv42_aux_score_weight": 0.0,
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "prototype_only",
        "qknnv42_feature_adapter_mode": "none",
        "qknnv42_head_mode": "extreme_light_diag_cosine",
        "extreme_light_aux_weight": 2.0,
        "extreme_light_epochs": 20,
        "extreme_light_max_trainable_parameters": 50000,
        "extreme_light_max_persistent_state_bytes": 131072,
    }


def test_smoke_shape_and_nested_new_class_configuration():
    rows = build_rows(
        arms=CORE_ARMS,
        new_class_counts=(5, 10, 20),
        receivers=("20-1", "8-8"),
        seeds=(713101, 713102),
        k_grid=(20,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )
    assert len(rows) == 36
    row = next(value for value in rows if value.arm == "el_diag_aux2p0" and value.new_class_count == 10)
    config = row_config(_base_config(), row, device="cpu")
    assert config["target_new_tx_labels"] == [f"n{i}" for i in range(10)]
    assert config["qknnv42_decision_mode"] == "per_sample_argmax"
    assert config["qknnv42_labelprop_mode"] == "disabled"
    assert config["extreme_light_aux_weight"] == 2.0


def test_baseline_remains_non_oracle_and_non_transductive():
    row = build_rows(
        arms=("baseline_single_qknn",),
        new_class_counts=(5,),
        receivers=("20-1",),
        seeds=(713101,),
        k_grid=(20,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_decision_mode"] == "per_sample_argmax"
    assert config["qknnv42_labelprop_mode"] == "support_prototype"
    assert config["qknnv42_head_mode"] == "qknn"


def test_five_epoch_zid_only_arm_compresses_adaptation():
    row = build_rows(
        arms=("el_zid_anchor5_e5",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(20,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["extreme_light_aux_weight"] == 0.0
    assert config["extreme_light_epochs"] == 5
    assert config["extreme_light_prototype_anchor_weight"] == 5.0


def test_twenty_epoch_noise_regularized_arm_is_explicit():
    row = build_rows(
        arms=("el_aux2p0_anchor20_noise5_e20",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["extreme_light_aux_weight"] == 2.0
    assert config["extreme_light_epochs"] == 20
    assert config["extreme_light_prototype_anchor_weight"] == 20.0
    assert config["extreme_light_feature_noise_std"] == 0.05


def test_closed_form_prototype_arm_has_zero_epochs():
    row = build_rows(
        arms=("el_proto_aux2p0",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_head_mode"] == "extreme_light_prototype_cosine"
    assert config["extreme_light_epochs"] == 0
    assert config["extreme_light_aux_weight"] == 2.0


def test_closed_form_ridge_arm_has_explicit_lambda_and_zero_epochs():
    row = build_rows(
        arms=("el_ridge_aux2p0_lam1em1",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_head_mode"] == "extreme_light_support_ridge"
    assert config["extreme_light_epochs"] == 0
    assert config["extreme_light_ridge_lambda"] == 0.1


def test_low_rank_margin_arm_is_explicit_and_twenty_epoch():
    row = build_rows(
        arms=("el_lowrank_r16_m0p1",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_head_mode"] == "extreme_light_low_rank_cosine"
    assert config["extreme_light_epochs"] == 20
    assert config["extreme_light_low_rank_width"] == 16
    assert config["extreme_light_cosine_margin"] == 0.1


def test_support_augmented_arm_keeps_physical_k_and_one_view_query():
    row = build_rows(
        arms=("el_lowrank_r8_m0p05_aug3_e10",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["k_shot"] == 30
    assert config.get("qknnv42_expected_tta_view_count", 1) == 1
    assert config["extreme_light_support_aug_scenarios"] == [
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ]
    assert config["extreme_light_epochs"] == 10


def test_frozen_source_logit_arm_is_per_sample_and_support_augmented():
    row = build_rows(
        arms=("el_diag_aug3_logit0p5_e20",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_decision_mode"] == "per_sample_argmax"
    assert config["qknnv42_labelprop_mode"] == "disabled"
    assert config["extreme_light_source_logit_weight"] == 0.5
    assert config["extreme_light_epochs"] == 20
    assert len(config["extreme_light_support_aug_scenarios"]) == 3


def test_frozen_source_bank_anchor_arm_has_no_query_role_gate():
    row = build_rows(
        arms=("el_diag_aug3_logit0p25_srca0p1_e20",),
        new_class_counts=(10,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(30,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_decision_mode"] == "per_sample_argmax"
    assert config["extreme_light_source_logit_weight"] == 0.25
    assert config["extreme_light_source_bank_anchor_strength"] == 0.1
    assert config["extreme_light_source_bank_anchor_blend"] == 0.25
    assert config["qknnv42_labelprop_mode"] == "disabled"


def test_multiprototype_arm_is_closed_form_and_selects_registered_feature_block():
    row = build_rows(
        arms=("el_mp3_fftrf_w4p0",),
        new_class_counts=(20,),
        receivers=("8-8",),
        seeds=(713101,),
        k_grid=(10,),
        output_root=Path("runs"),
        log_root=Path("logs"),
    )[0]
    config = row_config(_base_config(), row, device="cpu")
    assert config["qknnv42_head_mode"] == "extreme_light_multiprototype_cosine"
    assert config["qknnv42_aux_feature_key"] == "fft_rf_features"
    assert config["qknnv42_aux_feature_dim"] == 128
    assert config["extreme_light_aux_weight"] == 4.0
    assert config["extreme_light_epochs"] == 0
    assert config["extreme_light_prototypes_per_class"] == 3
    assert len(config["extreme_light_support_aug_scenarios"]) == 3


def test_k10_primary_config_locks_k5_sensitivity_and_new_thresholds():
    path = Path("paper_reproduction/configs/cvs_qknnv42_extreme_light_20new_stage2c_k10_20260715_n607.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["primary_k_shot"] == 10
    assert config["sensitivity_k_shot"] == 5
    assert config["support_pool_max_k"] == 10
    assert config["k5_max_drop_pp"] == 3.0
    assert config["success_thresholds"] == {
        "old_acc": 0.95,
        "min_old_class_acc": 0.88,
        "seen_new_acc_5": 0.92,
        "seen_new_acc_10": 0.90,
        "seen_new_acc_20": 0.86,
    }
