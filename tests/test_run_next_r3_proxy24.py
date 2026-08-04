from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_next_r3_proxy24.py"
SPEC = importlib.util.spec_from_file_location("run_next_r3_proxy24_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_missing_real_inputs_fail_closed(tmp_path: Path):
    absent = tmp_path / "absent.npz"
    args = SimpleNamespace(
        received_iq=absent,
        received_iq_sha256="0" * 64,
        source_held_archive=absent,
        source_held_archive_sha256="1" * 64,
        phase1_cells=absent,
        phase1_cells_sha256="2" * 64,
    )
    with pytest.raises(runner.MissingRealInputArtifacts, match=r"^MISSING_REAL_INPUT_ARTIFACTS"):
        runner._load_real_rows(args)


def test_new_run_root_refuses_overwrite(tmp_path: Path):
    root = runner._new_root(tmp_path / "run")
    assert (root / "rows").is_dir()
    with pytest.raises(runner.NextR3Proxy24Error, match="new absolute child"):
        runner._new_root(root)


def _truth_free_fixture(tmp_path: Path):
    receivers = ("1-1", "18-2", "r3", "r4", "r5", "r6", "r7")
    classes = tuple(f"c{index}" for index in range(6))
    physical_ids: list[str] = []
    observation_ids: list[str] = []
    receiver_ids: list[str] = []
    class_ids: list[str] = []
    by_receiver_class: dict[tuple[str, str], list[str]] = {}
    for receiver in receivers:
        for class_id in classes:
            values: list[str] = []
            for index in range(14):
                physical_id = f"{receiver}-{class_id}-{index:02d}"
                values.append(physical_id)
                physical_ids.append(physical_id)
                observation_ids.append(f"obs-{physical_id}")
                receiver_ids.append(receiver)
                class_ids.append(class_id)
            by_receiver_class[(receiver, class_id)] = values
    rows = runner.SourceRows(
        received_iq=np.ones((runner.ROW_COUNT, 2, 1), dtype=np.float32),
        receiver_ids=tuple(receiver_ids),
        day_ids=tuple("day" for _ in physical_ids),
        physical_ids=tuple(physical_ids),
        scenario_names=tuple("leo_clear_weak" for _ in physical_ids),
        observation_ids=tuple(observation_ids),
        receiver_registry=receivers,
        received_iq_sha256="a" * 64,
    )
    cells = runner.CellRows(
        receiver_ids=tuple(receiver_ids),
        class_ids=tuple(class_ids),
        physical_ids=tuple(physical_ids),
    )
    plan = runner.matrix.build_next_r3_proxy24_plan(classes)
    observation_by_physical = dict(zip(physical_ids, observation_ids, strict=True))
    split_rows: list[dict[str, object]] = []
    for planned in plan["rows"]:
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        active_k = int(planned["active_k"])
        support = [
            physical_id
            for class_id in classes
            for physical_id in by_receiver_class[(held_receiver, class_id)][:active_k]
        ]
        reg1_query = [
            physical_id
            for class_id in classes
            for physical_id in by_receiver_class[(held_receiver, class_id)][5:]
        ]
        reg0_query = [
            physical_id
            for class_id in classes
            if class_id != held_class
            for physical_id in by_receiver_class[(held_receiver, class_id)][5:]
        ]
        split_rows.append(
            {
                "row_id": planned["row_id"],
                "held_receiver": held_receiver,
                "held_class": held_class,
                "active_k": active_k,
                "support_physical_ids": support,
                "support_observation_ids": [observation_by_physical[item] for item in support],
                "support_labels": [class_id for class_id in classes for _ in range(active_k)],
                "reg0_query_physical_ids": reg0_query,
                "reg0_query_observation_ids": [observation_by_physical[item] for item in reg0_query],
                "reg1_query_physical_ids": reg1_query,
                "reg1_query_observation_ids": [observation_by_physical[item] for item in reg1_query],
            }
        )
    document: dict[str, object] = {
        "schema": runner.TRUTH_FREE_SPLIT_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "received_iq_sha256": rows.received_iq_sha256,
        "capsule_id": "b" * 64,
        "split_id": "c" * 64,
        "class_registry": list(classes),
        "rows": split_rows,
    }

    def write_receipt() -> SimpleNamespace:
        path = tmp_path / "truth_free_split.json"
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(payload)
        return SimpleNamespace(
            truth_free_split=path,
            truth_free_split_sha256=hashlib.sha256(payload).hexdigest(),
            capsule_id="b" * 64,
            split_id="c" * 64,
        )

    return rows, cells, document, write_receipt


def test_truth_free_split_rejects_query_truth_like_field(tmp_path: Path):
    rows, cells, document, write_receipt = _truth_free_fixture(tmp_path)
    assert isinstance(document["rows"], list)
    document["rows"][0]["query_labels"] = []
    with pytest.raises(runner.NextR3Proxy24Error, match="truth-like or unknown"):
        runner._load_truth_free_split(write_receipt(), rows, cells)


def test_truth_free_split_rejects_cross_k_query_drift(tmp_path: Path):
    rows, cells, document, write_receipt = _truth_free_fixture(tmp_path)
    assert isinstance(document["rows"], list)
    k5 = next(item for item in document["rows"] if item["active_k"] == 5)
    query_ids = k5["reg0_query_physical_ids"]
    query_observations = k5["reg0_query_observation_ids"]
    reg1_ids = k5["reg1_query_physical_ids"]
    reg1_observations = k5["reg1_query_observation_ids"]
    first_reg1 = reg1_ids.index(query_ids[0])
    second_reg1 = reg1_ids.index(query_ids[1])
    query_ids[0], query_ids[1] = query_ids[1], query_ids[0]
    query_observations[0], query_observations[1] = query_observations[1], query_observations[0]
    reg1_ids[first_reg1], reg1_ids[second_reg1] = reg1_ids[second_reg1], reg1_ids[first_reg1]
    reg1_observations[first_reg1], reg1_observations[second_reg1] = (
        reg1_observations[second_reg1],
        reg1_observations[first_reg1],
    )
    with pytest.raises(runner.NextR3Proxy24Error, match="K1/K5"):
        runner._load_truth_free_split(write_receipt(), rows, cells)


def test_bridge_cache_rejects_unbound_received_iq(tmp_path: Path):
    del tmp_path
    rows = runner.SourceRows(
        received_iq=np.ones((2, 2, 1), dtype=np.float32),
        receiver_ids=("r", "r"),
        day_ids=("d", "d"),
        physical_ids=("p0", "p1"),
        scenario_names=("leo_clear_weak", "leo_clear_weak"),
        observation_ids=("o0", "o1"),
        receiver_registry=("r",),
        received_iq_sha256="a" * 64,
    )
    bridge = SimpleNamespace(
        checkpoint_sha256="b" * 64,
        rows=SimpleNamespace(
            received_iq=np.zeros((2, 2, 1), dtype=np.float32),
            physical_ids=("p0", "p1"),
            observation_ids=("o0", "o1"),
        ),
    )
    with pytest.raises(runner.NextR3Proxy24Error, match="bridge input binding drift"):
        runner.BridgeFeatureCache(bridge, rows, "b" * 64)
