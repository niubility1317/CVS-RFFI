import csv
import hashlib
import importlib
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_wisig_canonical_union import main as audit_main


FORMAL_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _write_asset(path: Path, value: float) -> Path:
    samples = np.full((1, 4, 2), value, dtype=np.float32)
    payload = {
        "data": [[[[[samples]]]]],
        "tx_list": ["tx-A"],
        "rx_list": ["rx-X"],
        "capture_date_list": ["day-0"],
        "equalized_list": [1],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_cli_four_assets_writes_auditable_schemas_without_iq_bytes(tmp_path: Path):
    assets = {
        name: _write_asset(tmp_path / f"{name}.pkl", 7.25)
        for name in ("ManySig", "SingleDay", "ManyRx", "ManyTx")
    }
    sqlite_path = tmp_path / "canonical.sqlite"
    summary_path = tmp_path / "summary.json"
    coverage_path = tmp_path / "coverage.csv"
    conflicts_path = tmp_path / "conflicts.csv"
    argv = [
        "--sqlite-out",
        str(sqlite_path),
        "--summary-json",
        str(summary_path),
        "--coverage-csv",
        str(coverage_path),
        "--conflicts-csv",
        str(conflicts_path),
        "--equalized",
        "1",
    ]
    for name, path in assets.items():
        argv.extend(("--asset", f"{name}={path}"))

    assert audit_main(argv) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with coverage_path.open(encoding="utf-8", newline="") as handle:
        coverage_rows = list(csv.DictReader(handle))
    with conflicts_path.open(encoding="utf-8", newline="") as handle:
        conflict_rows = list(csv.DictReader(handle))

    assert summary == {
        "canonical_record_count": 1,
        "conflict_count": 0,
        "eligible_record_count": 1,
        "equalized": "1",
        "merged_duplicate_count": 3,
        "protocol_schema": "p2_min_v1",
        "source_record_count": 4,
    }
    assert list(coverage_rows[0]) == ["tx_id", "rx_id", "day_id", "record_count", "asset_count"]
    assert coverage_rows[0] == {
        "tx_id": "tx-A",
        "rx_id": "rx-X",
        "day_id": "day-0",
        "record_count": "1",
        "asset_count": "4",
    }
    assert conflict_rows == []
    text_outputs = "\n".join(
        path.read_text(encoding="utf-8") for path in (summary_path, coverage_path, conflicts_path)
    )
    assert "7.25" not in text_outputs
    assert "[[" not in text_outputs


def test_cli_rejects_duplicate_assets_and_existing_output_paths(tmp_path: Path):
    asset = _write_asset(tmp_path / "ManySig.pkl", 7.25)
    sqlite_path = tmp_path / "canonical.sqlite"
    summary_path = tmp_path / "summary.json"
    coverage_path = tmp_path / "coverage.csv"
    conflicts_path = tmp_path / "conflicts.csv"
    base_argv = [
        "--sqlite-out",
        str(sqlite_path),
        "--summary-json",
        str(summary_path),
        "--coverage-csv",
        str(coverage_path),
        "--conflicts-csv",
        str(conflicts_path),
    ]

    assert audit_main(
        base_argv + ["--asset", f"ManySig={asset}", "--asset", f"ManySig={asset}"]
    ) == 2
    assert not sqlite_path.exists()
    summary_path.write_text("already exists", encoding="utf-8")
    assert audit_main(base_argv + ["--asset", f"ManySig={asset}"]) == 2


def _build_split_main():
    try:
        return importlib.import_module("scripts.build_phase2_canonical_splits").main
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 4 split builder CLI is missing: {error}")


def _split_profile_payload():
    return {
        "schema": "cvs.phase2.canonical_union_profile.v1",
        "protocol_schema": "p2_min_v1",
        "source_profile_id": "TEST_MAXQ_BAL4D",
        "source_receivers": ["source-rx"],
        "receiver_tiers": {
            "dense": ["rx-a"],
            "single_day": [],
            "many_tx": [],
        },
        "old_tx_ids": [f"old-{index:02d}" for index in range(6)],
        "new_tx_candidates": [f"tx-{index:02d}" for index in range(22)],
        "new_class_sizes": [5, 10, 20],
        "k_values": [1, 5, 10, 20],
        "k_max": 20,
        "scenarios": list(FORMAL_SCENARIOS),
        "query_policies": ["MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE"],
    }


def _write_split_inventory(path: Path, *, sufficient: bool = True) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE canonical_records (
          physical_sample_id TEXT PRIMARY KEY,
          tx_id TEXT NOT NULL,
          rx_id TEXT NOT NULL,
          day_id TEXT NOT NULL,
          eq_id TEXT NOT NULL,
          sig_id TEXT NOT NULL,
          iq_sha256 TEXT NOT NULL,
          preferred_asset TEXT NOT NULL,
          preferred_source_record_index INTEGER NOT NULL,
          eligible INTEGER NOT NULL CHECK (eligible IN (0,1))
        )
        """
    )
    if sufficient:
        registered = tuple(f"old-{index:02d}" for index in range(6)) + tuple(
            f"tx-{index:02d}" for index in range(20)
        )
        source_index = 0
        rows = []
        for tx_id in registered:
            for day_index in range(4):
                for sample_index in range(63):
                    sample_id = f"cli-{tx_id}-d{day_index}-{sample_index:02d}"
                    rows.append(
                        (
                            sample_id,
                            tx_id,
                            "rx-a",
                            f"day-{day_index}",
                            "1",
                            str(sample_index),
                            "IQ_SECRET_DIGEST",
                            "ManyTx",
                            source_index,
                            1,
                        )
                    )
                    source_index += 1
        connection.executemany(
            """
            INSERT INTO canonical_records (
              physical_sample_id, tx_id, rx_id, day_id, eq_id, sig_id, iq_sha256,
              preferred_asset, preferred_source_record_index, eligible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    connection.commit()
    connection.close()
    return path


def _write_split_profile(path: Path) -> Path:
    path.write_text(
        json.dumps(_split_profile_payload(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_split_builder_cli_writes_exact_tree_deterministically_without_leakage(
    tmp_path: Path,
):
    inventory = _write_split_inventory(tmp_path / "canonical.sqlite")
    profile = _write_split_profile(tmp_path / "profile.json")
    out_root = tmp_path / "splits"
    before_inventory = hashlib.sha256(inventory.read_bytes()).hexdigest()

    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile),
            "--out-root",
            str(out_root),
            "--seed",
            "713101",
        ]
    ) == 0

    expected_files = {
        "class_selection.json",
        *{
            f"{policy}/k{k}.json"
            for policy in ("MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE")
            for k in (1, 5, 10, 20)
        },
    }
    actual_files = {
        path.relative_to(out_root).as_posix() for path in out_root.rglob("*") if path.is_file()
    }
    assert actual_files == expected_files
    assert hashlib.sha256(inventory.read_bytes()).hexdigest() == before_inventory

    selection = json.loads((out_root / "class_selection.json").read_text(encoding="utf-8"))
    assert selection["protocol_schema"] == "p2_min_v1"
    assert selection["profile_id"] == "TEST_MAXQ_BAL4D"
    assert selection["seed"] == 713101
    assert selection["Y_new5"] == selection["Y_new10"][:5]
    assert selection["Y_new10"] == selection["Y_new20"][:10]
    assert len(selection["Y_new20"]) == 20

    balanced_query_ids = []
    maxq_eligible_counts = []
    for policy in ("MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE"):
        for k in (1, 5, 10, 20):
            manifest_path = out_root / policy / f"k{k}.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert payload["protocol_schema"] == "p2_min_v1"
            assert payload["profile_id"] == "TEST_MAXQ_BAL4D"
            assert payload["query_policy"] == policy
            assert payload["k"] == k
            assert len(payload["registered_tx_ids"]) == 26
            assert payload["eligible_receivers"] == ["rx-a"]
            assert payload["counts"]["row_count"] == len(payload["rows"])
            assert payload["counts"]["support_count"] == sum(
                row["role"] == "support" for row in payload["rows"]
            )
            assert payload["counts"]["query_count"] == sum(
                row["role"] == "query" for row in payload["rows"]
            )
            support_rows = [
                row for row in payload["rows"] if row["role"] == "support"
            ]
            query_rows = [row for row in payload["rows"] if row["role"] == "query"]
            assert support_rows
            assert query_rows
            assert all(
                set(row)
                == {
                    "physical_sample_id",
                    "source_asset",
                    "source_record_index",
                    "tx_id",
                    "rx_id",
                    "day_id",
                    "scene",
                    "role",
                    "rank",
                }
                for row in support_rows
            )
            assert all(
                set(row)
                == {
                    "physical_sample_id",
                    "source_asset",
                    "source_record_index",
                    "rx_id",
                    "day_id",
                    "scene",
                    "role",
                    "rank",
                }
                for row in query_rows
            )
            query_truth_aliases = {
                "tx_id",
                "true_tx_id",
                "tx_label",
                "class_id",
                "class_label",
                "label",
                "truth",
                "query_truth",
            }
            assert all(not query_truth_aliases.intersection(row) for row in query_rows)
            serialized = manifest_path.read_text(encoding="utf-8").lower()
            for forbidden in (
                "iq_secret_digest",
                "iq_sha256",
                "dataset_path",
                "query_truth",
                "prediction",
                "class_quota",
            ):
                assert forbidden not in serialized
            if policy == "MAXQ_ALL_UNIQUE":
                maxq_eligible_counts.append(payload["counts"]["eligible_count"])
                assert payload["counts"]["eligible_count"] == payload["counts"]["row_count"]
            else:
                balanced_query_ids.append(
                    {
                        row["physical_sample_id"]
                        for row in payload["rows"]
                        if row["role"] == "query"
                    }
                )
    assert len(set(maxq_eligible_counts)) == 1
    assert balanced_query_ids[0] == balanced_query_ids[1] == balanced_query_ids[2] == balanced_query_ids[3]


def test_split_builder_cli_rejects_existing_root_without_mutating_inventory(tmp_path: Path):
    inventory = _write_split_inventory(tmp_path / "canonical.sqlite")
    profile = _write_split_profile(tmp_path / "profile.json")
    out_root = tmp_path / "existing"
    out_root.mkdir()
    marker = out_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    before_inventory = inventory.read_bytes()

    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile),
            "--out-root",
            str(out_root),
            "--seed",
            "713101",
        ]
    ) == 2
    assert marker.read_text(encoding="utf-8") == "keep"
    assert inventory.read_bytes() == before_inventory


def test_split_builder_cli_builds_all_manifests_before_creating_root(tmp_path: Path):
    inventory = _write_split_inventory(tmp_path / "empty.sqlite", sufficient=False)
    profile = _write_split_profile(tmp_path / "profile.json")
    out_root = tmp_path / "must-not-exist"

    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile),
            "--out-root",
            str(out_root),
            "--seed",
            "713101",
        ]
    ) == 2
    assert not out_root.exists()
