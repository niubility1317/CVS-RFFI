from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from paper_reproduction.scripts.benchmark_qknnv42_tta_policies import (
    METRICS,
    _aggregate,
    _apply_head_profile,
    _load_historical_reference,
    _validate_frozen_feature_caches,
    _validate_historical_reference_metrics,
)


def _row(value: float) -> dict[str, float | int | str]:
    return {
        "run_key": "rx_target/seed_1/k_1",
        "tta_view_count": 1,
        **{metric: value for metric in METRICS},
        "latency_per_query_ms": 1.0,
        "estimated_head_macs": 2.0,
        "persistent_state_bytes": 3.0,
        "decision_workspace_bytes_lower_bound": 4.0,
        "estimated_decision_cubic_work_units": 5.0,
    }


def test_full_history_profile_keeps_oracle_but_moves_adaptation_to_qknn() -> None:
    config: dict[str, object] = {}
    _apply_head_profile(config, profile="full_legacy_oracle", old_anchor_bias=-0.001)
    assert config["qknnv42_feature_adapter_mode"] == "support_diag_whiten_fisher"
    assert config["qknnv42_decision_mode"] == "legacy_role_quota_oracle"
    assert config["qknnv42_labelprop_mode"] == "dense_transductive"
    assert config["qknnv42_support_representation"] == "all_support"
    assert config["non_deployment_oracle_diagnostic"] is True


def test_full_history_prototype_profile_keeps_oracle_without_dense_graph() -> None:
    config: dict[str, object] = {}
    _apply_head_profile(
        config,
        profile="full_legacy_oracle_prototype",
        old_anchor_bias=-0.001,
    )
    assert config["qknnv42_decision_mode"] == "legacy_role_quota_oracle"
    assert config["qknnv42_labelprop_mode"] == "support_prototype"
    assert config["qknnv42_support_representation"] == "prototype_only"
    assert config["non_deployment_oracle_diagnostic"] is True


def test_historical_gate_includes_exact_three_pp_boundary() -> None:
    candidate = _row(0.47)
    baseline = {str(candidate["run_key"]): candidate}
    historical = {str(candidate["run_key"]): _row(0.50)}
    summary = _aggregate([candidate], baseline, historical)
    assert summary["performance_gate_reference"] == "historical"
    assert summary["performance_gate_pass"] is True

    failed_row = deepcopy(candidate)
    failed_row[METRICS[0]] = 0.4699
    failed = _aggregate([failed_row], baseline, historical)
    assert failed["performance_gate_pass"] is False


def test_historical_reference_loader_keeps_row_metrics_and_split_hash(tmp_path) -> None:
    run_dir = tmp_path / "rx_target" / "seed_1" / "k_1" / "cvs_qknnv42"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "target_receiver_label": "target",
            "seed": 1,
            "metrics": {metric: 0.5 for metric in METRICS},
        }),
        encoding="utf-8",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps({"splits_by_scenario": {"leo_clear_weak": {"support": [1]}}}),
        encoding="utf-8",
    )
    reference = _load_historical_reference(tmp_path)
    assert list(reference) == ["rx_target/seed_1/k_1"]
    assert reference["rx_target/seed_1/k_1"]["old_acc_mean"] == 0.5
    assert len(reference["rx_target/seed_1/k_1"]["split_manifest_sha256"]) == 64


def test_historical_reference_metrics_are_locked() -> None:
    reference = {"only": _row(0.5)}
    with pytest.raises(ValueError, match="locked 125-run baseline"):
        _validate_historical_reference_metrics(
            reference,
            {
                "old_acc_mean": 84.07,
                "seen_new_acc_mean": 93.24,
                "H_old_new_mean": 88.23,
            },
        )


def test_frozen_cache_manifest_is_required(tmp_path) -> None:
    path = tmp_path / "features.npz"
    checkpoint_hash = "a" * 64
    manifest = {
        "payload_source": "qknnv42_frozen_adv3b02_identity_only_features_v1",
        "source_checkpoint_sha256": checkpoint_hash,
        "feature_name": "z_id",
        "identity_only_forward": True,
        "domain_branch_executed_for_qknn": False,
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": {
            "missing_keys": 0,
            "unexpected_keys": 0,
            "skipped_mismatch": 0,
        },
        "adapter": {"skip_adapter_training": True, "adv3b02_gradient_updates": 0},
    }
    np.savez(path, manifest_json=np.asarray(json.dumps(manifest)))
    evidence = _validate_frozen_feature_caches(
        {"target": {"leo_clear_weak": str(path)}}, checkpoint_hash
    )
    assert evidence["validated_cache_count"] == 1

    manifest["adapter"]["adv3b02_gradient_updates"] = 60
    np.savez(path, manifest_json=np.asarray(json.dumps(manifest)))
    with pytest.raises(ValueError, match="not a frozen qKNN export"):
        _validate_frozen_feature_caches(
            {"target": {"leo_clear_weak": str(path)}}, checkpoint_hash
        )
