import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_eval_manifest_promotes_embedded_satellite_target_view(tmp_path):
    feature_npz = tmp_path / "features.npz"
    output_json = tmp_path / "metrics.json"
    manifest_json = tmp_path / "manifest.json"

    centers = {
        "old0": np.asarray([1.0, 0.0], dtype=np.float32),
        "old1": np.asarray([0.0, 1.0], dtype=np.float32),
        "new0": np.asarray([0.7, 0.7], dtype=np.float32),
        "unk0": np.asarray([-1.0, -1.0], dtype=np.float32),
    }
    tx_values = []
    feature_rows = []
    for tx_id, center in centers.items():
        for idx in range(8):
            tx_values.append(tx_id)
            feature_rows.append(center + np.asarray([idx * 0.001, -idx * 0.001], dtype=np.float32))
    tx_ids = np.asarray(tx_values, dtype=str)
    features = np.asarray(feature_rows, dtype=np.float32)
    embedded_manifest = {
        "channel_profile": {
            "target_new": {
                "view": "satellite",
                "scenarios": ["clear_leo", "rain_leo"],
            }
        }
    }
    np.savez(
        feature_npz,
        features=features,
        tx_ids=tx_ids,
        manifest_json=np.asarray(json.dumps(embedded_manifest)),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "code")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "eval_spaceborne_fewshot.py"),
            "--protocol",
            "source_open_set",
            "--feature_npz",
            str(feature_npz),
            "--output_json",
            str(output_json),
            "--manifest_json",
            str(manifest_json),
            "--source_tx_ids",
            "old0,old1",
            "--new_tx_ids",
            "new0",
            "--unknown_tx_ids",
            "unk0",
            "--source_proto_per_tx",
            "1",
            "--source_query_per_tx",
            "1",
            "--query_per_tx",
            "1",
            "--seed",
            "7",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["target_new_channel_view"] == "satellite"
    assert manifest["target_channel_view"] == "satellite/LEO"
    assert manifest["target_channel_scenarios"] == ["clear_leo", "rain_leo"]


def test_low_compute_target_adapter_records_step_loss_trace():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
    from cvsrffi.spaceborne_fewshot import PrototypeSet, fit_low_compute_target_adapter

    source = PrototypeSet(
        labels=torch.tensor([0, 1], dtype=torch.long),
        vectors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        counts=torch.tensor([4, 4], dtype=torch.long),
    )
    support = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    _, telemetry = fit_low_compute_target_adapter(source, support, labels, steps=3, lr=0.01, source_anchor_weight=0.05)

    assert telemetry["loss_trace_schema"] == "target_adapter_step_loss_v1"
    assert telemetry["optimizer"] == "Adam"
    assert len(telemetry["loss_trace"]) == 3
    first = telemetry["loss_trace"][0]
    assert {"step", "loss_total", "loss_ce", "loss_source_anchor_weighted", "grad_norm", "support_acc"} <= set(first)
    assert first["step"] == 1
    assert telemetry["loss_initial"] == telemetry["loss_trace"][0]["loss_total"]
    assert telemetry["loss_final"] == telemetry["loss_trace"][-1]["loss_total"]


def test_eval_cli_uses_target_old_role_for_stage2_queries(tmp_path):
    feature_npz = tmp_path / "features_target_old.npz"
    output_json = tmp_path / "metrics_target_old.json"
    manifest_json = tmp_path / "manifest_target_old.json"

    rows = []
    tx_ids = []
    roles = []

    def add(tx_id, role, center, count):
        for idx in range(count):
            rows.append(np.asarray(center, dtype=np.float32) + idx * 0.001)
            tx_ids.append(tx_id)
            roles.append(role)

    add("old0", "source", [1.0, 0.0], 4)
    add("old1", "source", [0.0, 1.0], 4)
    add("old0", "target_old", [0.9, 0.1], 3)
    add("old1", "target_old", [0.1, 0.9], 3)
    add("new0", "target_new", [-1.0, 0.0], 3)
    add("unk0", "target_new", [0.0, -1.0], 3)

    embedded_manifest = {
        "channel_profile": {
            "target_old": {"view": "satellite", "scenarios": ["clear_leo"]},
            "target_new": {"view": "satellite", "scenarios": ["clear_leo"]},
        }
    }
    np.savez(
        feature_npz,
        features=np.asarray(rows, dtype=np.float32),
        tx_ids=np.asarray(tx_ids, dtype=str),
        dataset_role=np.asarray(roles, dtype=str),
        manifest_json=np.asarray(json.dumps(embedded_manifest)),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "code")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "eval_spaceborne_fewshot.py"),
            "--protocol",
            "source_open_set",
            "--feature_npz",
            str(feature_npz),
            "--output_json",
            str(output_json),
            "--manifest_json",
            str(manifest_json),
            "--source_tx_ids",
            "old0,old1",
            "--target_old_tx_ids",
            "old0,old1",
            "--new_tx_ids",
            "new0",
            "--unknown_tx_ids",
            "unk0",
            "--shots",
            "1",
            "--source_proto_per_tx",
            "2",
            "--source_query_per_tx",
            "1",
            "--target_old_query_per_tx",
            "2",
            "--query_per_tx",
            "1",
            "--seed",
            "7",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["target_channel_view"] == "satellite/LEO"
    assert manifest["target_old_tx_ids"] == ["old0", "old1"]
    assert manifest["counts"]["target_old_query"] == 4


def test_eval_cli_remaps_index_style_target_old_ids_from_embedded_manifest(tmp_path):
    feature_npz = tmp_path / "features_target_old_index_remap.npz"
    output_json = tmp_path / "metrics_target_old_index_remap.json"
    manifest_json = tmp_path / "manifest_target_old_index_remap.json"

    rows = []
    tx_ids = []
    roles = []

    def add(tx_id, role, center, count):
        for idx in range(count):
            rows.append(np.asarray(center, dtype=np.float32) + idx * 0.001)
            tx_ids.append(tx_id)
            roles.append(role)

    add("old0", "source", [1.0, 0.0], 4)
    add("old1", "source", [0.0, 1.0], 4)
    add("old0", "target_old", [0.9, 0.1], 3)
    add("old1", "target_old", [0.1, 0.9], 3)
    add("new0", "target_new", [-1.0, 0.0], 3)
    add("unk0", "target_new", [0.0, -1.0], 3)

    embedded_manifest = {
        "source_tx_ids": ["old0", "old1"],
        "target_old_tx_ids": ["old0", "old1"],
        "new_tx_ids": ["new0"],
        "unknown_tx_ids": ["unk0"],
        "channel_profile": {
            "target_old": {"view": "satellite", "scenarios": ["clear_leo"]},
            "target_new": {"view": "satellite", "scenarios": ["clear_leo"]},
        },
    }
    np.savez(
        feature_npz,
        features=np.asarray(rows, dtype=np.float32),
        tx_ids=np.asarray(tx_ids, dtype=str),
        dataset_role=np.asarray(roles, dtype=str),
        manifest_json=np.asarray(json.dumps(embedded_manifest)),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "code")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "code" / "eval_spaceborne_fewshot.py"),
            "--protocol",
            "source_open_set",
            "--feature_npz",
            str(feature_npz),
            "--output_json",
            str(output_json),
            "--manifest_json",
            str(manifest_json),
            "--source_tx_ids",
            "0,1",
            "--target_old_tx_ids",
            "0,1",
            "--new_tx_ids",
            "2",
            "--unknown_tx_ids",
            "3",
            "--shots",
            "1",
            "--source_proto_per_tx",
            "2",
            "--source_query_per_tx",
            "1",
            "--target_old_query_per_tx",
            "2",
            "--query_per_tx",
            "1",
            "--seed",
            "7",
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["source_tx_ids"] == ["old0", "old1"]
    assert manifest["target_old_tx_ids"] == ["old0", "old1"]
    assert manifest["new_tx_ids"] == ["new0"]
    assert manifest["unknown_tx_ids"] == ["unk0"]
    assert manifest["counts"]["target_old_query"] == 4
