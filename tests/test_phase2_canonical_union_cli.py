import csv
import json
import pickle
from pathlib import Path

import numpy as np

from scripts.audit_wisig_canonical_union import main


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

    assert main(argv) == 0
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

    assert main(base_argv + ["--asset", f"ManySig={asset}", "--asset", f"ManySig={asset}"]) == 2
    assert not sqlite_path.exists()
    summary_path.write_text("already exists", encoding="utf-8")
    assert main(base_argv + ["--asset", f"ManySig={asset}"]) == 2
