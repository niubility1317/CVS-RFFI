"""Canonical identities and deterministic raw-record iteration for WiSig PKLs."""

from __future__ import annotations

import hashlib
import json
import pickle
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
