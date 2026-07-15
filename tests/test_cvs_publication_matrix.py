from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from paper_reproduction.scripts.run_cvs_publication_matrix import (
    DEFAULT_K,
    DEFAULT_RECEIVERS,
    DEFAULT_SEEDS,
    PHASE_METHODS,
    _assert_cvs_config_uses_independent_query_decisions,
    _artifact_status,
    _matrix_manifest_path,
    build_rows,
)


def test_full_stage2_matrices_cover_methods_receivers_k_and_seeds(tmp_path: Path) -> None:
    for phase in ("stage2b", "stage2c"):
        rows = build_rows(
            phase=phase,
            methods=PHASE_METHODS[phase],
            receivers=DEFAULT_RECEIVERS,
            k_grid=DEFAULT_K,
            seeds=DEFAULT_SEEDS,
            output_root=tmp_path / phase / "runs",
            log_root=tmp_path / phase / "logs",
        )
        assert len(rows) == len(PHASE_METHODS[phase]) * 5 * 5 * 5
        assert len({row.experiment_id for row in rows}) == len(rows)
        assert {row.receiver for row in rows} == set(DEFAULT_RECEIVERS)
        assert {row.k_shot for row in rows} == set(DEFAULT_K)
        assert {row.seed for row in rows} == set(DEFAULT_SEEDS)


def test_subset_worker_cannot_overwrite_canonical_manifest(tmp_path: Path) -> None:
    canonical = _matrix_manifest_path(
        tmp_path,
        phase="stage2b",
        methods=PHASE_METHODS["stage2b"],
        receivers=DEFAULT_RECEIVERS,
        k_grid=DEFAULT_K,
        seeds=DEFAULT_SEEDS,
    )
    subset = _matrix_manifest_path(
        tmp_path,
        phase="stage2b",
        methods=("cvs_opgac",),
        receivers=DEFAULT_RECEIVERS[1:],
        k_grid=DEFAULT_K,
        seeds=DEFAULT_SEEDS,
    )
    assert canonical.name == "matrix_manifest.json"
    assert subset.name.startswith("matrix_manifest_subset_")
    assert subset != canonical


def test_artifact_contract_requires_satellite_scores_details_and_loss_trace(tmp_path: Path) -> None:
    row = build_rows(
        phase="stage2b",
        methods=("protonet_cda",),
        receivers=("20-1",),
        k_grid=(5,),
        seeds=(713101,),
        output_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
    )[0]
    run_dir = Path(row.run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(
        json.dumps({"support_query_overlap": False, "all_tests_satellite_augmented": True}),
        encoding="utf-8",
    )
    (run_dir / "detailed_metrics.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "loss_trace.json").write_text('[{"loss":1.0}]\n', encoding="utf-8")
    with (run_dir / "score_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        for index in range(360):
            writer.writerow({"sample_id": index})
    with (run_dir / "detailed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_type"])
        writer.writeheader()
        for group_type in (
            "per_receiver",
            "per_transmitter",
            "per_receiver_transmitter",
            "per_receiver_transmitter_day",
        ):
            writer.writerow({"group_type": group_type})
    with (run_dir / "loss_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loss"])
        writer.writeheader()
        writer.writerow({"loss": 1.0})

    assert _artifact_status(row)["complete"] is True
    (run_dir / "loss_trace.csv").write_text("loss\nnan\n", encoding="utf-8")
    status = _artifact_status(row)
    assert status["complete"] is False
    assert "nonfinite_loss_trace" in status["errors"]


def test_publication_preflight_rejects_role_oracle_and_role_partition(tmp_path: Path) -> None:
    config_path = tmp_path / "oracle.json"
    config_path.write_text(
        json.dumps(
            {
                "launchable": False,
                "protocol_status": "PROTOCOL_INVALID_FOR_DEPLOYMENT_HISTORICAL_ARTIFACT_ONLY",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not launchable"):
        _assert_cvs_config_uses_independent_query_decisions(config_path)

    config_path.write_text(
        json.dumps({"qknnv42_decision_mode": "legacy_role_quota_oracle"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="role Oracle and class-quota"):
        _assert_cvs_config_uses_independent_query_decisions(config_path)

    config_path.write_text(
        json.dumps(
            {
                "qknnv42_decision_mode": "per_sample_argmax",
                "qknnv42_feature_adapter_mode": "support_role_center",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query-role partition"):
        _assert_cvs_config_uses_independent_query_decisions(config_path)

def test_publication_artifact_contract_rejects_oracle_metadata(tmp_path: Path) -> None:
    row = build_rows(
        phase="stage2c",
        methods=("cvs_qknnv42",),
        receivers=("20-1",),
        k_grid=(5,),
        seeds=(713101,),
        output_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
    )[0]
    run_dir = Path(row.run_dir)
    run_dir.mkdir(parents=True)
    unsafe_scenario = {
        "role_oracle_used": True,
        "equal_class_quota_used": True,
        "query_query_graph_used": True,
        "query_batch_state_required": True,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics_by_scenario": {
                    "leo_clear_weak": unsafe_scenario,
                    "leo_low_elev_weak": unsafe_scenario,
                    "leo_rain_weak": unsafe_scenario,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"qknnv42_feature_adapter_mode": "none"}),
        encoding="utf-8",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "support_query_overlap": False,
                "all_tests_satellite_augmented": True,
                "qknnv42_decision_mode": "legacy_role_quota_oracle",
                "qknnv42_labelprop_mode": "dense_transductive",
                "non_deployment_oracle_diagnostic": True,
                "query_used_for_joint_decision": True,
                "query_used_for_transductive_inference": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "detailed_metrics.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "loss_trace.json").write_text('[{"loss":1.0}]\n', encoding="utf-8")
    with (run_dir / "score_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        for index in range(480):
            writer.writerow({"sample_id": index})
    with (run_dir / "detailed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_type"])
        writer.writeheader()
        for group_type in (
            "per_receiver",
            "per_transmitter",
            "per_receiver_transmitter",
            "per_receiver_transmitter_day",
        ):
            writer.writerow({"group_type": group_type})
    with (run_dir / "loss_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loss"])
        writer.writeheader()
        writer.writerow({"loss": 1.0})

    status = _artifact_status(row)
    assert status["complete"] is False
    assert "non_independent_query_decision" in status["errors"]
    assert "oracle_diagnostic_status_missing_or_true" in status["errors"]
    assert "query_used_for_joint_decision_missing_or_true" in status["errors"]
    assert "role_oracle_used_missing_or_true" in status["errors"]

    (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(
        json.dumps({"support_query_overlap": False, "all_tests_satellite_augmented": True}),
        encoding="utf-8",
    )
    missing_status = _artifact_status(row)
    assert missing_status["complete"] is False
    assert "non_independent_query_decision" in missing_status["errors"]
    assert "scenario_decision_metadata_missing_or_incomplete" in missing_status["errors"]
    assert "feature_adapter_mode_missing" in missing_status["errors"]
