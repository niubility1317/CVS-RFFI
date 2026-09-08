"""Read the public Tweak LoRa cf32 recordings without materializing them in RAM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


FRAME_LENGTH = 128
_MEMMAPS: dict[Path, np.memmap] = {}


@dataclass(frozen=True)
class OfficialLoRaRecord:
    configuration: str
    device_id: int
    data_path: Path
    metadata_path: Path
    total_samples: int


@dataclass(frozen=True)
class RecordFrameSplit:
    total_frames: int
    training: range
    calibration: range
    testing: range


def _metadata_datatype(metadata_path: Path) -> str:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return str(metadata["_metadata"]["global"]["core:datatype"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read SigMF datatype from {metadata_path}") from exc


def load_configuration_records(
    root: Path,
    configuration: str,
    *,
    device_ids: Sequence[int],
) -> list[OfficialLoRaRecord]:
    """Return requested public LoRa records and reject non-cf32 or truncated data."""
    configuration_root = Path(root) / configuration
    records: list[OfficialLoRaRecord] = []
    for device_id in device_ids:
        data_path = configuration_root / f"IQ_{device_id}.dat"
        metadata_path = data_path.with_suffix(".sigmf-meta")
        if not data_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"missing public LoRa record for {configuration}/IQ_{device_id}")
        if _metadata_datatype(metadata_path).lower() != "cf32":
            raise ValueError(f"{metadata_path} must declare cf32 samples")
        byte_count = data_path.stat().st_size
        if byte_count == 0 or byte_count % np.dtype(np.complex64).itemsize:
            raise ValueError(f"{data_path} is empty or does not contain whole cf32 samples")
        records.append(
            OfficialLoRaRecord(
                configuration=configuration,
                device_id=int(device_id),
                data_path=data_path,
                metadata_path=metadata_path,
                total_samples=byte_count // np.dtype(np.complex64).itemsize,
            )
        )
    return records


def read_iq_frames(record: OfficialLoRaRecord, *, frame_indices: Iterable[int]) -> torch.Tensor:
    """Read [I,Q] raw-IQ frames in the paper's 2x128 input order."""
    indices = [int(index) for index in frame_indices]
    if not indices:
        return torch.empty((0, 2, FRAME_LENGTH), dtype=torch.float32)
    total_frames = record.total_samples // FRAME_LENGTH
    if min(indices) < 0 or max(indices) >= total_frames:
        raise IndexError(f"frame index is outside {record.data_path}")
    raw = _MEMMAPS.get(record.data_path)
    if raw is None:
        raw = np.memmap(record.data_path, dtype=np.complex64, mode="r", shape=(record.total_samples,))
        _MEMMAPS[record.data_path] = raw
    offsets = np.asarray(indices, dtype=np.int64)[:, None] * FRAME_LENGTH + np.arange(FRAME_LENGTH, dtype=np.int64)
    selected = np.asarray(raw[offsets])
    iq = np.ascontiguousarray(np.stack((selected.real, selected.imag), axis=1), dtype=np.float32)
    return torch.frombuffer(memoryview(iq), dtype=torch.float32).reshape(iq.shape)


def split_record_frames(
    *,
    total_samples: int,
    training_fraction: float = 0.75,
    calibration_fraction_of_training: float = 0.10,
) -> RecordFrameSplit:
    """Create the paper's 75/25 train/test split and N=10% training calibration prefix."""
    if total_samples <= 0 or total_samples % FRAME_LENGTH:
        raise ValueError("total_samples must contain a whole positive number of 128-sample frames")
    if not 0 < training_fraction < 1 or not 0 < calibration_fraction_of_training <= 1:
        raise ValueError("split fractions must be in (0, 1]")
    total_frames = total_samples // FRAME_LENGTH
    training_end = int(total_frames * training_fraction)
    calibration_end = int(training_end * calibration_fraction_of_training)
    if training_end == 0 or calibration_end == 0 or training_end >= total_frames:
        raise ValueError("split leaves an empty training, calibration, or test region")
    return RecordFrameSplit(
        total_frames=total_frames,
        training=range(0, training_end),
        calibration=range(0, calibration_end),
        testing=range(training_end, total_frames),
    )
