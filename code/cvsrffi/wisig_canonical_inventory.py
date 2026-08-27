"""Canonical identities and deterministic raw-record iteration for WiSig PKLs."""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


_REQUIRED_KEYS = (
    "data",
    "tx_list",
    "rx_list",
    "capture_date_list",
    "equalized_list",
)


@dataclass(frozen=True)
class RawRecordRef:
    physical_sample_id: str
    asset_name: str
    dataset_path: str
    source_record_index: int
    tx_id: str
    rx_id: str
    day_id: str
    eq_id: str
    sig_id: str
    iq_sha256: str


@dataclass(frozen=True)
class InventorySummary:
    source_record_count: int
    canonical_record_count: int
    eligible_record_count: int
    merged_duplicate_count: int
    conflict_count: int


_ASSET_PRIORITY = {"ManySig": 0, "SingleDay": 1, "ManyRx": 2, "ManyTx": 3}


def _asset_preference_key(asset_name: str, source_record_index: int) -> tuple[int, str, int]:
    return (_ASSET_PRIORITY.get(asset_name, len(_ASSET_PRIORITY)), asset_name, source_record_index)


def canonical_coordinate(
    tx_id: str,
    rx_id: str,
    day_id: str,
    eq_id: str,
    sig_id: str,
) -> tuple[str, str, str, str, str]:
    """Return the stable, path-independent WiSig coordinate labels."""

    return tuple(map(str, (tx_id, rx_id, day_id, eq_id, sig_id)))


def canonical_physical_id(
    tx_id: str,
    rx_id: str,
    day_id: str,
    eq_id: str,
    sig_id: str,
) -> str:
    """Hash the compact ASCII JSON representation of a canonical coordinate."""

    payload = canonical_coordinate(tx_id, rx_id, day_id, eq_id, sig_id)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _resolve_label(labels: Any, index: int, name: str) -> str:
    try:
        value = labels[index]
    except (IndexError, KeyError, TypeError):
        raise ValueError(f"WiSig {name}_list has no label for index {index}") from None
    return _label(value)


def _load_payload(dataset_path: str | Path) -> Mapping[str, Any]:
    path = Path(dataset_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"WiSig PKL must contain a mapping, got {type(payload).__name__}")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"WiSig PKL missing required key(s): {', '.join(missing)}")
    return payload


def _equalization_index(equalized_list: Any, requested: Any) -> tuple[int, str]:
    requested_label = _label(requested)
    try:
        labels = list(equalized_list)
    except TypeError:
        raise ValueError("WiSig equalized_list must be a sequence") from None
    for index, value in enumerate(labels):
        if _label(value) == requested_label:
            return index, _label(value)
    raise ValueError(
        f"equalized={requested!r} is absent from equalized_list={[_label(value) for value in labels]!r}"
    )


def iter_wisig_records(
    dataset_path: str | Path,
    asset_name: str,
    equalized: Any = 1,
) -> Iterator[RawRecordRef]:
    """Yield canonical raw-record references in nested WiSig dataset order."""

    payload = _load_payload(dataset_path)
    data = payload["data"]
    tx_list = payload["tx_list"]
    rx_list = payload["rx_list"]
    day_list = payload["capture_date_list"]
    eq_index, eq_id = _equalization_index(payload["equalized_list"], equalized)

    source_record_index = 0
    dataset_path_text = str(dataset_path)
    asset_name_text = str(asset_name)
    for tx_index, tx_data in enumerate(data):
        tx_id = _resolve_label(tx_list, tx_index, "tx")
        for rx_index, rx_data in enumerate(tx_data):
            rx_id = _resolve_label(rx_list, rx_index, "rx")
            for day_index, day_data in enumerate(rx_data):
                day_id = _resolve_label(day_list, day_index, "capture_date")
                try:
                    cell = day_data[eq_index]
                except (IndexError, KeyError, TypeError):
                    raise ValueError(
                        f"WiSig data cell at tx={tx_index}, rx={rx_index}, day={day_index} "
                        f"has no equalization index {eq_index}"
                    ) from None
                if cell is None:
                    continue
                try:
                    samples = iter(cell)
                except TypeError:
                    raise ValueError(
                        f"WiSig data cell at tx={tx_index}, rx={rx_index}, day={day_index}, "
                        f"eq={eq_index} must be a sample sequence"
                    ) from None
                for sig_index, iq in enumerate(samples):
                    iq_array = np.ascontiguousarray(np.asarray(iq, dtype=np.float32))
                    iq_sha256 = hashlib.sha256(iq_array.tobytes(order="C")).hexdigest()
                    sig_id = str(sig_index)
                    physical_sample_id = canonical_physical_id(
                        tx_id,
                        rx_id,
                        day_id,
                        eq_id,
                        sig_id,
                    )
                    yield RawRecordRef(
                        physical_sample_id=physical_sample_id,
                        asset_name=asset_name_text,
                        dataset_path=dataset_path_text,
                        source_record_index=source_record_index,
                        tx_id=tx_id,
                        rx_id=rx_id,
                        day_id=day_id,
                        eq_id=eq_id,
                        sig_id=sig_id,
                        iq_sha256=iq_sha256,
                    )
                    source_record_index += 1


def build_inventory(
    asset_paths: Mapping[str, str | Path],
    sqlite_path: str | Path,
    equalized: Any = 1,
) -> InventorySummary:
    """Build a non-overwriting canonical WiSig inventory from named PKL assets."""

    output_path = Path(sqlite_path)
    connection: sqlite3.Connection | None = None
    reserved_file_identity: tuple[int, int] | None = None
    try:
        with output_path.open("xb"):
            pass
        stat_result = output_path.stat()
        reserved_file_identity = (stat_result.st_dev, stat_result.st_ino)
        connection = sqlite3.connect(output_path)
        connection.execute("BEGIN")
        connection.executescript(
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
            );
            CREATE TABLE record_sources (
              physical_sample_id TEXT NOT NULL,
              asset_name TEXT NOT NULL,
              dataset_path TEXT NOT NULL,
              source_record_index INTEGER NOT NULL,
              iq_sha256 TEXT NOT NULL,
              PRIMARY KEY (physical_sample_id, asset_name, source_record_index)
            );
            CREATE TABLE identity_conflicts (
              physical_sample_id TEXT NOT NULL,
              first_iq_sha256 TEXT NOT NULL,
              conflicting_iq_sha256 TEXT NOT NULL,
              asset_name TEXT NOT NULL
            );
            """
        )
        for asset_name, dataset_path in sorted(asset_paths.items(), key=lambda item: str(item[0])):
            asset_name_text = str(asset_name)
            for record in iter_wisig_records(dataset_path, asset_name_text, equalized=equalized):
                connection.execute(
                    """
                    INSERT INTO record_sources (
                      physical_sample_id, asset_name, dataset_path, source_record_index, iq_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.physical_sample_id,
                        record.asset_name,
                        record.dataset_path,
                        record.source_record_index,
                        record.iq_sha256,
                    ),
                )
                existing = connection.execute(
                    """
                    SELECT iq_sha256, preferred_asset, preferred_source_record_index
                    FROM canonical_records WHERE physical_sample_id = ?
                    """,
                    (record.physical_sample_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO canonical_records (
                          physical_sample_id, tx_id, rx_id, day_id, eq_id, sig_id, iq_sha256,
                          preferred_asset, preferred_source_record_index, eligible
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            record.physical_sample_id,
                            record.tx_id,
                            record.rx_id,
                            record.day_id,
                            record.eq_id,
                            record.sig_id,
                            record.iq_sha256,
                            record.asset_name,
                            record.source_record_index,
                        ),
                    )
                elif existing[0] != record.iq_sha256:
                    connection.execute(
                        """
                        INSERT INTO identity_conflicts (
                          physical_sample_id, first_iq_sha256, conflicting_iq_sha256, asset_name
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (record.physical_sample_id, existing[0], record.iq_sha256, record.asset_name),
                    )
                    connection.execute(
                        "UPDATE canonical_records SET eligible = 0 WHERE physical_sample_id = ?",
                        (record.physical_sample_id,),
                    )
                elif _asset_preference_key(record.asset_name, record.source_record_index) < _asset_preference_key(
                    existing[1], existing[2]
                ):
                    connection.execute(
                        """
                        UPDATE canonical_records
                        SET preferred_asset = ?, preferred_source_record_index = ?
                        WHERE physical_sample_id = ?
                        """,
                        (record.asset_name, record.source_record_index, record.physical_sample_id),
                    )

        source_record_count = connection.execute("SELECT COUNT(*) FROM record_sources").fetchone()[0]
        canonical_record_count = connection.execute("SELECT COUNT(*) FROM canonical_records").fetchone()[0]
        eligible_record_count = connection.execute(
            "SELECT COUNT(*) FROM canonical_records WHERE eligible = 1"
        ).fetchone()[0]
        conflict_count = connection.execute("SELECT COUNT(*) FROM identity_conflicts").fetchone()[0]
        connection.commit()
        return InventorySummary(
            source_record_count=source_record_count,
            canonical_record_count=canonical_record_count,
            eligible_record_count=eligible_record_count,
            merged_duplicate_count=source_record_count - canonical_record_count,
            conflict_count=conflict_count,
        )
    except Exception:
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()
                connection = None
        if reserved_file_identity is not None:
            try:
                stat_result = output_path.stat()
            except OSError:
                stat_result = None
            if stat_result is not None and (stat_result.st_dev, stat_result.st_ino) == reserved_file_identity:
                output_path.unlink()
        raise
    finally:
        if connection is not None:
            connection.close()
