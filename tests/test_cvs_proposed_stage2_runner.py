from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from paper_reproduction.cvs_aligned.cvs_method_runner import SCENARIOS, run


OLD = ["old0", "old1"]
NEW = ["new0"]


def _cache(path: Path, scenario: str) -> None:
    rng = np.random.default_rng(71)
    rows = []
    for label_i, label in enumerate(OLD):
        center = np.eye(4)[label_i]
        for sample in range(24):
            rows.append((center + rng.normal(0, 0.02, 4), label, "source", "src", sample, ""))
        for sample in range(8):
            rows.append((center + rng.normal(0, 0.02, 4), label, "target_old", "target", sample, scenario))
    center = np.eye(4)[2]
    for sample in range(8):
        rows.append((center + rng.normal(0, 0.02, 4), "new0", "target_new", "target", sample, scenario))
    np.savez(
        path,
        features=np.asarray([row[0] for row in rows], dtype=np.float32),
        fft_logmag_features=np.asarray([row[0] for row in rows], dtype=np.float32),
        tx_ids=np.asarray([row[1] for row in rows]),
        rx_ids=np.asarray([row[3] for row in rows]),
        day_ids=np.asarray([0] * len(rows)),
        eq_ids=np.asarray([0] * len(rows)),
        sig_ids=np.asarray([row[4] for row in rows]),
        dataset_role=np.asarray([row[2] for row in rows]),
        sat_scenarios=np.asarray([row[5] for row in rows]),
        manifest_json=np.asarray(
            json.dumps(
                {
                    "satellite_tta_view_count": 1,
                    "aux_fft_view_alignment": "same_post_channel_view_as_backbone",
                }
            )
        ),
    )


def _config(tmp_path: Path, method: str) -> dict:
    mapping = {}
    for scenario in SCENARIOS:
        path = tmp_path / f"{scenario}.npz"
        _cache(path, scenario)
        mapping[scenario] = str(path)
    return {
        "experiment_id": method,
        "method": method,
        "stage": "Stage2-B" if method == "cvs_opgac" else "Stage2-C",
        "feature_npz_by_scenario": mapping,
        "target_receiver_labels": ["target"],
        "target_old_tx_labels": OLD,
        "target_new_tx_labels": NEW,
        "target_unknown_tx_labels": [],
        "k_shot": 2,
        "query_per_tx": 3,
        "support_pool_max_k": 4,
        "target_sample_strategy": "seeded_nested",
        "split_seed": 713101,
        "seed": 713101,
        "target_channel_scenarios": list(SCENARIOS),
        "unknown_rejection_enabled": False,
    }


def test_cvs_opgac_writes_publication_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "opgac"
    result = run(_config(tmp_path, "cvs_opgac"), run_dir)
    assert result["metrics"]["target_old_accuracy_mean"] == 1.0
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["support_query_overlap"] is False
    assert manifest["all_tests_satellite_augmented"] is True
    assert manifest["target_new_tx_labels"] == []
    with (run_dir / "score_table.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2 * 3 * 3


def test_cvs_qknnv42_uses_nested_disjoint_split_and_four_detail_levels(tmp_path: Path) -> None:
    run_dir = tmp_path / "qknn"
    result = run(_config(tmp_path, "cvs_qknnv42"), run_dir)
    assert result["metrics"]["H_old_new_mean"] == 1.0
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    split = manifest["splits_by_scenario"][SCENARIOS[0]]
    assert not set(split["support_sample_ids"]) & set(split["query_sample_ids"])
    with (run_dir / "detailed_metrics.csv").open(encoding="utf-8", newline="") as handle:
        groups = {row["group_type"] for row in csv.DictReader(handle)}
    assert groups == {
        "per_receiver", "per_transmitter", "per_receiver_transmitter", "per_receiver_transmitter_day"
    }


def test_cvs_qknnv42_fft_aux_and_legacy_oracle_are_explicit(tmp_path: Path) -> None:
    config = _config(tmp_path, "cvs_qknnv42")
    config.update(
        {
            "qknnv42_aux_feature_key": "fft_logmag_features",
            "qknnv42_aux_feature_dim": 4,
            "qknnv42_aux_score_weight": 0.34,
            "qknnv42_expected_tta_view_count": 1,
            "qknnv42_decision_mode": "legacy_role_quota_oracle",
        }
    )
    run_dir = tmp_path / "qknn_fft_oracle"
    result = run(config, run_dir)
    assert result["metrics"]["H_old_new_mean"] == 1.0
    first = result["metrics_by_scenario"][SCENARIOS[0]]
    assert first["aux_feature_enabled"] is True
    assert first["aux_score_weight"] == 0.34
    assert first["decision_mode"] == "legacy_role_quota_oracle"
    manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["non_deployment_oracle_diagnostic"] is True
