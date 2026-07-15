#!/usr/bin/env python3
"""Generate strict non-dense qKNNV42 Stage2-C configs for adapter epoch ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EPOCHS = (2, 5, 10, 20, 30, 60)
RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def build_config(epoch: int, remote_root: str) -> dict:
    feature_root = (
        f"{remote_root}/runs/qknnv42_nondense_adapter_epoch_features_20260715/"
        f"E{epoch}"
    )
    feature_map = {}
    for receiver in RECEIVERS:
        feature_path = (
            f"{feature_root}/FULL_RX_{receiver}/"
            f"ADV3B02_FULL_ADAPTER5_FFT96_E{epoch}/"
            f"features_full_adapter5_fft96_e{epoch}.npz"
        )
        feature_map[receiver] = {scenario: feature_path for scenario in SCENARIOS}
    if epoch <= 20:
        resource_tier = "EXTREME_LIGHT_PREFERRED"
        protocol_status = "LAUNCHABLE_NONDENSE_STAGE2C"
    elif epoch <= 40:
        resource_tier = "PERFORMANCE_RELAXED"
        protocol_status = "LAUNCHABLE_NONDENSE_STAGE2C_RESOURCE_ABLATION"
    else:
        resource_tier = "NON_EXTREME_LIGHT_RESOURCE_CONTROL"
        protocol_status = "LAUNCHABLE_NONDENSE_STAGE2C_NONPROMOTABLE_RESOURCE_CONTROL"
    return {
        "experiment_id": f"cvs_qknnv42_nondense_adapter_e{epoch}_stage2c_20260715",
        "launchable": True,
        "protocol_status": protocol_status,
        "method": "cvs_qknnv42",
        "stage": "Stage2-C",
        "cvs_proposed_method": True,
        "backbone_id": f"ADV3B02_CORE90_SOFT_E200_STRICT_LOAD_ID_NORM_LATE_FEATURE_E{epoch}",
        "adapter_training_epochs": epoch,
        "adapter_resource_tier": resource_tier,
        "feature_npz_by_receiver_scenario": feature_map,
        "target_receiver_labels": ["20-1"],
        "publication_target_receiver_grid": list(RECEIVERS),
        "target_old_tx_labels": ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"],
        "target_new_tx_labels": ["1-16", "1-18"],
        "target_unknown_tx_labels": [],
        "k_shot": 5,
        "query_per_tx": 20,
        "support_pool_max_k": 20,
        "target_sample_strategy": "seeded_nested",
        "split_seed": 713101,
        "seed": 713101,
        "target_channel_view": "satellite/LEO",
        "target_channel_scenarios": list(SCENARIOS),
        "target_labels_scope": "registered_support_only",
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
        "unknown_rejection_enabled": False,
        "qknnv42_aux_feature_key": "fft_logmag_features",
        "qknnv42_aux_feature_dim": 96,
        "qknnv42_aux_score_weight": 0.34,
        "qknnv42_expected_tta_view_count": 5,
        "qknnv42_head_mode": "qknn",
        "qknnv42_decision_mode": "per_sample_argmax",
        "qknnv42_labelprop_mode": "disabled",
        "qknnv42_support_representation": "all_support",
        "qknnv42_feature_adapter_mode": "support_diag_whiten_fisher",
        "qknnv42_old_anchor_bias": 0.001,
        "non_deployment_oracle_diagnostic": False,
        "publication_protocol": "same_125_tasks_full_qknn_no_dense_query_no_role_or_class_quota_oracle_strict_checkpoint_load",
        "claim_boundary": (
            "Stage2-C old-domain adaptation plus two-real-new-TX enrollment; "
            "per-sample inference; no dense query graph; no role/quota Oracle"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_reproduction/configs"),
    )
    parser.add_argument(
        "--remote-root",
        default="/home/szu2070436088/2510044040/CV-SincNet",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in EPOCHS:
        path = args.output_dir / (
            f"cvs_qknnv42_nondense_adapter_e{epoch}_stage2c_20260715_n607.json"
        )
        path.write_text(
            json.dumps(build_config(epoch, args.remote_root), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
