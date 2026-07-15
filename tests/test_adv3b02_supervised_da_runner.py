from __future__ import annotations

import torch
import json
from pathlib import Path

from paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner import (
    _nearest_prototype,
    _select_registered_support,
    _target_predictor_bundle_path,
    _validate_config,
)

import numpy as np


def _config(tmp_path: Path) -> dict:
    evidence = {
        "sealed_inference_package_sha256": "a" * 64,
        "package_root_sha256": "b" * 64,
        "runtime_code_sha256": "c" * 64,
        "artifact_member_allowlist_sha256": "d" * 64,
        "os_isolation_mode": "equivalent_verified_isolation",
        "os_isolation_attestation_sha256": "e" * 64,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": "f" * 64,
        "predict_score_process_isolation": True,
    }
    evidence_path = tmp_path / "rx_20_1" / "seed_713101" / "runtime_isolation_evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return {
        "method_id": "mrior_sda", "stage": "Stage2-B",
        "target_new_tx_labels": [], "target_unknown_tx_labels": [],
        "target_receiver_labels": ["20-1"], "target_old_tx_labels": ["a", "b"],
        "source_receiver_labels": ["s0"], "k_shot": 5, "support_pool_max_k": 20,
        "query_per_tx": 2, "adapt_steps": 2, "seed": 713101, "split_seed": 713101,
        "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
        "target_channel_view": "leo_weak_only",
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False, "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "source_leo_weak_cache_set_manifest": "source/cache_set.json",
        "target_predictor_bundle_root": "target",
        "target_predictor_seal_root": "seals",
        "phase2_runtime_isolation_evidence_root": str(tmp_path),
    }


def test_adv3b02_da_protocol_accepts_old_only(tmp_path: Path) -> None:
    _validate_config(_config(tmp_path))


def test_adv3b02_da_protocol_rejects_new_classes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["target_new_tx_labels"] = ["new"]
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "target-old" in str(exc)
    else:
        raise AssertionError("Stage2-B accepted target-new classes")


def test_adv3b02_protonet_support_prediction() -> None:
    support = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[0.0, 0.1], [3.0, 3.1]])
    assert _nearest_prototype(support, labels, query).tolist() == [0, 1]


def test_adv3b02_da_protocol_rejects_raw_dataset_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["manysig_pkl"] = "ManySig.pkl"
    try:
        _validate_config(config)
    except ValueError as exc:
        assert "raw/clean inputs" in str(exc)
    else:
        raise AssertionError("Phase2 config accepted a raw dataset path")


def test_target_bundle_path_is_receiver_and_seed_specific(tmp_path: Path) -> None:
    path = _target_predictor_bundle_path(_config(tmp_path))
    assert path.as_posix().endswith("target/rx_20_1/seed_713101")


def test_registered_support_selector_uses_first_k_per_class(tmp_path: Path) -> None:
    count_per_class = 20
    arrays = {
        "support_pool_class_indices": np.asarray([0] * count_per_class + [1] * count_per_class),
        "support_pool_tokens": np.asarray([f"sid_{index:032x}" for index in range(40)]),
        "support_pool_leo_weak_iq": np.arange(40 * 4, dtype=np.float32).reshape(40, 2, 2),
    }
    config = _config(tmp_path)
    _x5, y5, ids5 = _select_registered_support(arrays, config)
    config["k_shot"] = 1
    _x1, y1, ids1 = _select_registered_support(arrays, config)
    assert y5.tolist() == [0] * 5 + [1] * 5
    assert y1.tolist() == [0, 1]
    assert ids1 == [f"sid_{0:032x}", f"sid_{count_per_class:032x}"]
