from __future__ import annotations

import json
from copy import deepcopy

from paper_reproduction.scripts.benchmark_qknnv42_tta_policies import (
    METRICS,
    _aggregate,
    _apply_head_profile,
    _load_historical_reference,
)


def _row(value: float) -> dict[str, float | int | str]:
    return {
        "run_key": "rx_target/seed_1/k_1",
        "tta_view_count": 1,
        **{metric: value for metric in METRICS},
        "latency_per_query_ms": 1.0,
        "estimated_head_macs": 2.0,
        "persistent_state_bytes": 3.0,
    }


def test_full_history_profile_keeps_oracle_but_moves_adaptation_to_qknn() -> None:
    config: dict[str, object] = {}
    _apply_head_profile(config, profile="full_legacy_oracle", old_anchor_bias=-0.001)
    assert config["qknnv42_feature_adapter_mode"] == "support_diag_whiten_fisher"
    assert config["qknnv42_decision_mode"] == "legacy_role_quota_oracle"
    assert config["qknnv42_labelprop_mode"] == "dense_transductive"
    assert config["qknnv42_support_representation"] == "all_support"
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
