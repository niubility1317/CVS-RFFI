import os
import subprocess
import sys
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_export_cli_accepts_target_old_stage2_arguments(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "code")

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "export_spaceborne_features.py"),
            "--dry_run_synthetic",
            "--out_npz",
            str(tmp_path / "features.npz"),
            "--target_old_tx_ids",
            "old0,old1",
            "--target_old_channel_view",
            "satellite",
            "--target_old_sat_scenarios",
            "clear_leo",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_eval_cli_accepts_ftrc_target_old_support(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "code")

    features = []
    tx_ids = []
    roles = []
    rx_ids = []
    day_ids = []
    channel_views = []
    sat_scenarios = []

    def add(tx, role, base, count, *, rx="rx7", scenario="clear_leo"):
        for idx in range(count):
            features.append([base, float(idx)])
            tx_ids.append(tx)
            roles.append(role)
            rx_ids.append(rx)
            day_ids.append("2021_03_15")
            channel_views.append("satellite")
            sat_scenarios.append(scenario)

    add("old0", "source", 1.0, 5, rx="rx0", scenario="")
    add("old1", "source", 2.0, 5, rx="rx1", scenario="")
    add("old0", "target_old", 1.2, 5, rx="rx7", scenario="low_elev_leo")
    add("old1", "target_old", 2.2, 5, rx="rx8", scenario="rain_leo")
    add("new0", "target_new", -1.0, 3, rx="rx9", scenario="storm_mp")
    add("unk0", "target_new", -2.0, 3, rx="rx10", scenario="mixed_orbit")

    feature_npz = tmp_path / "features.npz"
    np.savez(
        feature_npz,
        features=np.asarray(features, dtype=np.float32),
        tx_ids=np.asarray(tx_ids, dtype=str),
        dataset_role=np.asarray(roles, dtype=str),
        rx_ids=np.asarray(rx_ids, dtype=str),
        day_ids=np.asarray(day_ids, dtype=str),
        channel_views=np.asarray(channel_views, dtype=str),
        sat_scenarios=np.asarray(sat_scenarios, dtype=str),
    )

    output_json = tmp_path / "metrics.json"
    manifest_json = tmp_path / "manifest.json"
    score_csv = tmp_path / "score.csv"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "eval_spaceborne_fewshot.py"),
            "--protocol",
            "ftrc",
            "--feature_npz",
            str(feature_npz),
            "--output_json",
            str(output_json),
            "--manifest_json",
            str(manifest_json),
            "--score_table_csv",
            str(score_csv),
            "--source_tx_ids",
            "old0,old1",
            "--target_old_tx_ids",
            "old0,old1",
            "--new_tx_ids",
            "new0",
            "--unknown_tx_ids",
            "unk0",
            "--gate_mode",
            "oa_mse",
            "--oa_mse_adapter_steps",
            "2",
            "--oa_mse_adapter_selection_policy",
            "target_boundary_guard",
            "--pseudo_unknown_target_shift_samples_per_class",
            "2",
            "--pseudo_unknown_target_halo_samples_per_class",
            "2",
            "--pseudo_unknown_target_ring_samples_per_class",
            "2",
            "--oa_mse_old_bridge_weight",
            "0.15",
            "--oa_mse_support_contrast_weight",
            "0.20",
            "--oa_mse_soft_proto_weight",
            "0.15",
            "--soft_proto_topk",
            "2",
            "--soft_proto_temperature",
            "0.10",
            "--source_proto_per_tx",
            "2",
            "--target_old_support_per_tx",
            "1",
            "--target_old_query_per_tx",
            "2",
            "--shots",
            "0",
            "--query_per_tx",
            "1",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert output_json.exists()
    assert manifest_json.exists()
    assert score_csv.exists()
    with score_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {
        "query_role",
        "query_rx_id",
        "query_sat_scenario",
        "query_sample_index",
        "old_support_quality",
        "old_support_quality_delta",
    } <= set(rows[0])
    assert {row["query_role"] for row in rows} == {"target_old_query", "new_query", "unknown_query"}
    assert {"rx7", "rx8", "rx9", "rx10"} <= {row["query_rx_id"] for row in rows}
    assert {"low_elev_leo", "rain_leo", "storm_mp", "mixed_orbit"} <= {
        row["query_sat_scenario"] for row in rows
    }
    metrics = json.loads(output_json.read_text(encoding="utf-8"))
    assert metrics["loss_trace_status"] == "PRESENT"
    assert len(metrics["loss_trace"]) == 2
    adapter = metrics["telemetry"]["oa_mse_onboard_adaptation"]["target_adapter"]
    pseudo_unknown = metrics["telemetry"]["oa_mse_onboard_adaptation"]["pseudo_unknown_energy"]
    assert adapter["pseudo_unknown_target_shift_count"] > 0
    assert adapter["pseudo_unknown_target_halo_count"] > 0
    assert adapter["pseudo_unknown_target_ring_count"] > 0
    assert adapter["old_bridge_count"] > 0
    assert adapter["support_contrast_weight"] == 0.20
    assert adapter["soft_proto_weight"] == 0.15
    assert adapter["soft_proto_anchor_count"] > 0
    assert pseudo_unknown["target_shift_sample_count"] > 0
    assert pseudo_unknown["target_halo_sample_count"] > 0
    assert pseudo_unknown["target_ring_sample_count"] > 0
