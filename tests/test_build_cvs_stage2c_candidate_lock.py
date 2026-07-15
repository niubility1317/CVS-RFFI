from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from paper_reproduction.scripts.build_cvs_stage2c_candidate_lock import (
    build_candidate_lock,
    verify_candidate_lock,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    adapter = tmp_path / "adapter.pt"
    stats = tmp_path / "stats.npz"
    checkpoint.write_bytes(b"checkpoint")
    adapter.write_bytes(b"adapter")
    np.savez(stats, mean=np.zeros(8), std=np.ones(8))
    validation = tmp_path / "source_validation.json"
    validation_payload = {
        "source_validation_pass": True,
        "clean_samples_used_for_validation": False,
        "failed_gates": [],
        "gates": {"all_source_checks": True},
        "scenarios": [
            "leo_clear_weak",
            "leo_low_elev_weak",
            "leo_rain_weak",
        ],
        "permissions": {
            "target_support_used": False,
            "target_query_features_used": False,
            "target_query_labels_used": False,
        },
        "symmetric_head_lock": {
            "selection_source": "disjoint_source_receiver_holdout_k1_episodes",
            "target_support_used_for_selection": False,
            "target_query_features_used": False,
            "old_new_role_oracle_used": False,
            "class_quota_used": False,
            "support_view_policy": "three_leo_weak_scenario_base_views",
            "support_receive_views_per_physical_sample": 3,
            "allowed_k": [1, 5, 10, 20],
            "selected": {
                "use_alignment": True,
                "prototype_rule": "mean",
                "ridge": 0.1,
            },
        },
        "nested_k_source_lock": {
            "k_values": [1, 5, 10, 20],
            "target_rows_used": False,
            "role_labels_used": False,
            "class_quota_used": False,
        },
        "calibration": {
            "selected": {
                "thresholds": {
                    "base_margin": 0.1,
                    "shift3_margin": 0.05,
                    "shift3_disagreement": 1.0 / 3.0,
                }
            }
        },
        "source_feature_statistics": {
            "path": str(stats),
            "sha256": _sha(stats),
            "target_rows_used": False,
            "feature_kind": "normalized_z_id_plus_fft96_weight2",
            "fft_dim": 96,
            "fft_weight": 2.0,
        },
    }
    validation.write_text(json.dumps(validation_payload), encoding="utf-8")
    promotion = tmp_path / "promotion.json"
    promotion_payload = {
        "method": "ground_source_effective_feature_lora_v1",
        "source_validation_pass": True,
        "source_only": True,
        "target_receiver_data_used_for_training": False,
        "clean_samples_used_for_training": False,
        "formal_training_view": "leo_weak_only",
        "proxy_data_used_for_training": False,
        "checkpoint_sha256": _sha(checkpoint),
        "adapter_state_sha256": _sha(adapter),
        "source_validation_manifest": str(validation),
        "source_validation_manifest_sha256": _sha(validation),
    }
    promotion.write_text(json.dumps(promotion_payload), encoding="utf-8")
    direct_mapping = tmp_path / "direct_mapping.json"
    direct_mapping.write_text(
        json.dumps({"class_id_to_tx": ["o0", "o1"]}), encoding="utf-8"
    )
    split = tmp_path / "class_split.json"
    split.write_text(
        json.dumps(
            {
                "target_old_tx_labels": ["o0", "o1"],
                "nested_target_new_tx_labels": {
                    "5": [f"n{i}" for i in range(5)],
                    "10": [f"n{i}" for i in range(10)],
                    "20": [f"n{i}" for i in range(20)],
                },
                "direct_adv3b02_class_mapping_source": str(direct_mapping),
                "direct_adv3b02_class_mapping_sha256": _sha(direct_mapping),
                "direct_adv3b02_class_id_to_tx": ["o0", "o1"],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, adapter, promotion, split, direct_mapping


def test_source_candidate_lock_pins_code_and_all_formal_k(tmp_path) -> None:
    checkpoint, adapter, promotion, split, direct_mapping = _artifacts(tmp_path)
    lock = build_candidate_lock(
        candidate_id="effective8-r16-e12",
        checkpoint=checkpoint,
        adapter_state=adapter,
        promotion_manifest=promotion,
        class_split_manifest=split,
    )
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    config = {
        "target_receiver_labels": ["20-1"],
        "k_shot": 5,
        "support_pool_max_k": 20,
        "target_new_tx_labels": [f"n{i}" for i in range(10)],
        "target_old_tx_labels": ["o0", "o1"],
        "direct_adv3b02_class_mapping_source": str(direct_mapping),
        "direct_adv3b02_class_mapping_sha256": _sha(direct_mapping),
        "direct_adv3b02_class_id_to_tx": ["o0", "o1"],
    }
    verified = verify_candidate_lock(
        lock_path,
        checkpoint=checkpoint,
        adapter_state=adapter,
        promotion_manifest=promotion,
        config=config,
    )
    assert verified["locked_candidate"]["formal_matrix"]["k_values"] == [
        1,
        5,
        10,
        20,
    ]
    assert verified["locked_candidate"]["head"]["mode"] == "symmetric_locked"


def test_candidate_lock_rejects_support_pool_drift(tmp_path) -> None:
    checkpoint, adapter, promotion, split, direct_mapping = _artifacts(tmp_path)
    lock = build_candidate_lock(
        candidate_id="candidate",
        checkpoint=checkpoint,
        adapter_state=adapter,
        promotion_manifest=promotion,
        class_split_manifest=split,
    )
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="support_pool_max_k"):
        verify_candidate_lock(
            lock_path,
            checkpoint=checkpoint,
            adapter_state=adapter,
            promotion_manifest=promotion,
            config={
                "target_receiver_labels": ["8-8"],
                "k_shot": 1,
                "support_pool_max_k": 10,
                "target_new_tx_labels": [f"n{i}" for i in range(5)],
                "target_old_tx_labels": ["o0", "o1"],
                "direct_adv3b02_class_mapping_source": str(direct_mapping),
                "direct_adv3b02_class_mapping_sha256": _sha(direct_mapping),
                "direct_adv3b02_class_id_to_tx": ["o0", "o1"],
            },
        )


def test_candidate_lock_rejects_same_count_with_different_tx(tmp_path) -> None:
    checkpoint, adapter, promotion, split, direct_mapping = _artifacts(tmp_path)
    lock = build_candidate_lock(
        candidate_id="candidate",
        checkpoint=checkpoint,
        adapter_state=adapter,
        promotion_manifest=promotion,
        class_split_manifest=split,
    )
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="target-new labels"):
        verify_candidate_lock(
            lock_path,
            checkpoint=checkpoint,
            adapter_state=adapter,
            promotion_manifest=promotion,
            config={
                "target_receiver_labels": ["8-8"],
                "k_shot": 1,
                "support_pool_max_k": 20,
                "target_old_tx_labels": ["o0", "o1"],
                "target_new_tx_labels": ["easy0", "easy1", "easy2", "easy3", "easy4"],
                "direct_adv3b02_class_mapping_source": str(direct_mapping),
                "direct_adv3b02_class_mapping_sha256": _sha(direct_mapping),
                "direct_adv3b02_class_id_to_tx": ["o0", "o1"],
            },
        )


def test_candidate_lock_rejects_mutated_class_split_artifact(tmp_path) -> None:
    checkpoint, adapter, promotion, split, direct_mapping = _artifacts(tmp_path)
    lock = build_candidate_lock(
        candidate_id="candidate",
        checkpoint=checkpoint,
        adapter_state=adapter,
        promotion_manifest=promotion,
        class_split_manifest=split,
    )
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    split.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable artifact drift: class_split"):
        verify_candidate_lock(
            lock_path,
            checkpoint=checkpoint,
            adapter_state=adapter,
            promotion_manifest=promotion,
            config={
                "target_receiver_labels": ["8-8"],
                "k_shot": 1,
                "support_pool_max_k": 20,
                "target_old_tx_labels": ["o0", "o1"],
                "target_new_tx_labels": [f"n{i}" for i in range(5)],
                "direct_adv3b02_class_mapping_source": str(direct_mapping),
                "direct_adv3b02_class_mapping_sha256": _sha(direct_mapping),
                "direct_adv3b02_class_id_to_tx": ["o0", "o1"],
            },
        )
