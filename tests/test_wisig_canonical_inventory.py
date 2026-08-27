import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.wisig_canonical_inventory as canonical_inventory
from cvsrffi.wisig_canonical_inventory import (
    RawRecordRef,
    build_inventory,
    canonical_coordinate,
    canonical_physical_id,
    iter_wisig_records,
)


REQUIRED_KEYS = (
    "data",
    "tx_list",
    "rx_list",
    "capture_date_list",
    "equalized_list",
)


def _payload(*, values=(1.0, 2.0), num_tx=2, num_rx=2, num_days=2):
    data = []
    for tx_i in range(num_tx):
        tx_rows = []
        for rx_i in range(num_rx):
            day_rows = []
            for day_i in range(num_days):
                base = float((tx_i + 1) * 100 + (rx_i + 1) * 10 + day_i)
                eq0 = np.stack(
                    [np.full((4, 2), base + value, dtype=np.float32) for value in values]
                )
                eq1 = np.stack(
                    [np.full((4, 2), base + 1000.0 + value, dtype=np.float32) for value in values]
                )
                day_rows.append([eq0, eq1])
            tx_rows.append(day_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": ["tx-A", "tx-B"][:num_tx],
        "rx_list": ["rx-X", "rx-Y"][:num_rx],
        "capture_date_list": ["day-0", "day-1"][:num_days],
        "equalized_list": [0, 1],
    }


def _write_payload(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


@pytest.fixture
def fake_wisig_pkl(tmp_path: Path) -> Path:
    return _write_payload(tmp_path / "ManyTx.pkl", _payload(num_tx=1, num_rx=1, num_days=1))


def test_canonical_identity_is_path_independent(tmp_path: Path):
    first_path = _write_payload(tmp_path / "first" / "ManyTx.pkl", _payload(num_tx=1, num_rx=1, num_days=1))
    second_path = _write_payload(tmp_path / "second" / "renamed.pkl", _payload(num_tx=1, num_rx=1, num_days=1))

    first = list(iter_wisig_records(first_path, "first-asset", equalized=1))
    second = list(iter_wisig_records(second_path, "renamed-asset", equalized=1))

    assert [row.physical_sample_id for row in first] == [row.physical_sample_id for row in second]
    assert first[0].dataset_path != second[0].dataset_path
    assert first[0].asset_name != second[0].asset_name
    assert canonical_coordinate("tx0", "rx0", "day0", "1", "7") == (
        "tx0",
        "rx0",
        "day0",
        "1",
        "7",
    )
    expected = hashlib.sha256(
        json.dumps(("tx0", "rx0", "day0", "1", "7"), separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    ).hexdigest()
    assert canonical_physical_id("tx0", "rx0", "day0", "1", "7") == expected


def test_reader_filters_equalized_one_and_hashes_non_empty_iq(fake_wisig_pkl: Path):
    rows = list(iter_wisig_records(fake_wisig_pkl, "ManyTx", equalized=1))

    assert rows
    assert all(isinstance(row, RawRecordRef) for row in rows)
    assert {row.eq_id for row in rows} == {"1"}
    assert all(row.iq_sha256 for row in rows)
    assert all(len(row.iq_sha256) == 64 for row in rows)
    expected = hashlib.sha256(
        np.ascontiguousarray(np.full((4, 2), 1111.0, dtype=np.float32)).tobytes(order="C")
    ).hexdigest()
    assert rows[0].iq_sha256 == expected


@pytest.mark.parametrize("missing_key", REQUIRED_KEYS)
def test_reader_rejects_missing_required_key(fake_wisig_pkl: Path, missing_key: str):
    payload = _payload(num_tx=1, num_rx=1, num_days=1)
    del payload[missing_key]
    path = _write_payload(fake_wisig_pkl.parent / f"missing-{missing_key}.pkl", payload)

    with pytest.raises(KeyError, match=missing_key):
        list(iter_wisig_records(path, "ManyTx", equalized=1))


def test_reader_rejects_absent_equalization_label(fake_wisig_pkl: Path):
    with pytest.raises(ValueError, match="equalized"):
        list(iter_wisig_records(fake_wisig_pkl, "ManyTx", equalized=7))


def test_reader_resolves_labels_and_traverses_deterministically(tmp_path: Path):
    path = _write_payload(tmp_path / "ManyTx.pkl", _payload())

    first = list(iter_wisig_records(path, "ManyTx", equalized=1))
    second = list(iter_wisig_records(path, "ManyTx", equalized=1))

    assert len(first) == 16
    assert first == second
    assert [row.source_record_index for row in first] == list(range(16))
    assert [(row.tx_id, row.rx_id, row.day_id, row.eq_id, row.sig_id) for row in first[:4]] == [
        ("tx-A", "rx-X", "day-0", "1", "0"),
        ("tx-A", "rx-X", "day-0", "1", "1"),
        ("tx-A", "rx-X", "day-1", "1", "0"),
        ("tx-A", "rx-X", "day-1", "1", "1"),
    ]
    assert {row.tx_id for row in first} == {"tx-A", "tx-B"}
    assert {row.rx_id for row in first} == {"rx-X", "rx-Y"}
    assert {row.day_id for row in first} == {"day-0", "day-1"}
    assert {row.eq_id for row in first} == {"1"}
    with pytest.raises((AttributeError, TypeError)):
        first[0].eq_id = "changed"


def test_inventory_merges_overlapping_assets_with_summary_counts(tmp_path: Path):
    many_tx = _write_payload(
        tmp_path / "ManyTx.pkl", _payload(values=(1.0, 2.0, 3.0), num_tx=1, num_rx=1, num_days=1)
    )
    many_rx = _write_payload(
        tmp_path / "ManyRx.pkl", _payload(values=(1.0, 2.0), num_tx=1, num_rx=1, num_days=1)
    )

    summary = build_inventory(
        {"ManyTx": many_tx, "ManyRx": many_rx}, tmp_path / "canonical.sqlite", equalized=1
    )

    assert summary.source_record_count == 5
    assert summary.canonical_record_count == 3
    assert summary.eligible_record_count == 3
    assert summary.merged_duplicate_count == 2
    assert summary.conflict_count == 0


def test_inventory_marks_different_digest_at_same_coordinate_ineligible(tmp_path: Path):
    first = _write_payload(
        tmp_path / "ManyTx.pkl", _payload(values=(1.0,), num_tx=1, num_rx=1, num_days=1)
    )
    conflicting = _write_payload(
        tmp_path / "ManyRx.pkl", _payload(values=(9.5,), num_tx=1, num_rx=1, num_days=1)
    )

    summary = build_inventory(
        {"ManyTx": first, "ManyRx": conflicting}, tmp_path / "canonical.sqlite", equalized=1
    )

    assert summary.source_record_count == 2
    assert summary.canonical_record_count == 1
    assert summary.eligible_record_count == 0
    assert summary.merged_duplicate_count == 1
    assert summary.conflict_count == 1


def test_inventory_uses_asset_priority_only_for_preferred_reference(tmp_path: Path):
    asset_paths = {}
    for asset_name in ("ManyTx", "ManyRx", "SingleDay", "ManySig"):
        asset_paths[asset_name] = _write_payload(
            tmp_path / f"{asset_name}.pkl",
            _payload(values=(1.0,), num_tx=1, num_rx=1, num_days=1),
        )
    sqlite_path = tmp_path / "canonical.sqlite"

    summary = build_inventory(asset_paths, sqlite_path, equalized=1)
    with sqlite3.connect(sqlite_path) as connection:
        preferred_asset, preferred_source_record_index = connection.execute(
            "SELECT preferred_asset, preferred_source_record_index FROM canonical_records"
        ).fetchone()

    assert summary.source_record_count == 4
    assert summary.canonical_record_count == 1
    assert summary.merged_duplicate_count == 3
    assert preferred_asset == "ManySig"
    assert preferred_source_record_index == 0


def test_inventory_rejects_existing_sqlite_output(tmp_path: Path):
    asset = _write_payload(
        tmp_path / "ManyTx.pkl", _payload(values=(1.0,), num_tx=1, num_rx=1, num_days=1)
    )
    sqlite_path = tmp_path / "canonical.sqlite"
    sqlite_path.write_text("already exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_inventory({"ManyTx": asset}, sqlite_path, equalized=1)


def test_inventory_does_not_adopt_or_delete_concurrently_reserved_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    asset = _write_payload(
        tmp_path / "ManyTx.pkl", _payload(values=(1.0,), num_tx=1, num_rx=1, num_days=1)
    )
    sqlite_path = tmp_path / "canonical.sqlite"
    sqlite_path.write_text("reserved by another caller", encoding="utf-8")
    real_exists = Path.exists
    stale_check_used = False

    def stale_once(path: Path) -> bool:
        nonlocal stale_check_used
        if path == sqlite_path and not stale_check_used:
            stale_check_used = True
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", stale_once)

    with pytest.raises(FileExistsError):
        build_inventory({"ManyTx": asset}, sqlite_path, equalized=1)

    assert sqlite_path.read_text(encoding="utf-8") == "reserved by another caller"


def test_inventory_closes_owned_connection_before_removing_incomplete_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    asset = _write_payload(
        tmp_path / "ManyTx.pkl", _payload(values=(1.0,), num_tx=1, num_rx=1, num_days=1)
    )
    sqlite_path = tmp_path / "canonical.sqlite"
    events: list[str] = []
    closed = False
    real_connect = canonical_inventory.sqlite3.connect
    real_unlink = Path.unlink

    class ObservedConnection:
        def __init__(self, connection: sqlite3.Connection):
            self._connection = connection

        def __getattr__(self, name: str):
            return getattr(self._connection, name)

        def close(self) -> None:
            nonlocal closed
            closed = True
            events.append("close")
            self._connection.close()

    def observing_connect(*args, **kwargs):
        return ObservedConnection(real_connect(*args, **kwargs))

    def fail_iteration(*args, **kwargs):
        raise RuntimeError("injected source read failure")
        yield None

    def unlink_after_close(path: Path, *args, **kwargs):
        events.append("unlink")
        assert closed
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(canonical_inventory.sqlite3, "connect", observing_connect)
    monkeypatch.setattr(canonical_inventory, "iter_wisig_records", fail_iteration)
    monkeypatch.setattr(Path, "unlink", unlink_after_close)

    with pytest.raises(RuntimeError, match="injected source read failure"):
        build_inventory({"ManyTx": asset}, sqlite_path, equalized=1)

    assert events == ["close", "unlink"]
    assert not sqlite_path.exists()
