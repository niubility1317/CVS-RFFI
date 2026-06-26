from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .spectrogram import ensure_iq_2xl


@dataclass(frozen=True)
class RFFISample:
    path: str
    label: int
    receiver: int
    split: str = ""
    snr: float | None = None
    packet_id: str | None = None


def load_iq_npy(path: str) -> torch.Tensor:
    arr = np.load(path)
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        x = torch.from_numpy(np.stack([arr.real, arr.imag], axis=0)).float()
    else:
        x = torch.from_numpy(arr).float()
        if x.dim() == 2 and x.shape[-1] == 2 and x.shape[0] != 2:
            x = x.t().contiguous()
    return ensure_iq_2xl(x)[0]


def read_csv_index(path: str, root: str | None = None) -> List[RFFISample]:
    root = root or ""
    out: List[RFFISample] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_path = row.get("path") or row.get("iq_path")
            if not sample_path:
                raise ValueError("CSV index requires a 'path' or 'iq_path' column.")
            if root and not os.path.isabs(sample_path):
                sample_path = os.path.join(root, sample_path)
            out.append(
                RFFISample(
                    path=sample_path,
                    label=int(row.get("label", row.get("tx_label", 0))),
                    receiver=int(row.get("receiver", row.get("rx_label", 0))),
                    split=str(row.get("split", "")),
                    snr=float(row["snr"]) if row.get("snr") not in (None, "") else None,
                    packet_id=row.get("packet_id") or None,
                )
            )
    return out


class CSVIQDataset(Dataset):
    def __init__(
        self,
        index_csv: str,
        root: str | None = None,
        split: str | None = None,
        receivers: Sequence[int] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        samples = read_csv_index(index_csv, root=root)
        if split:
            samples = [s for s in samples if s.split == split]
        if receivers is not None:
            keep = {int(r) for r in receivers}
            samples = [s for s in samples if int(s.receiver) in keep]
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[int(idx)]
        iq = load_iq_npy(s.path)
        if self.transform is not None:
            iq = self.transform(iq)
        return {
            "iq": iq,
            "label": int(s.label),
            "receiver": int(s.receiver),
            "snr": s.snr,
            "packet_id": s.packet_id,
            "path": s.path,
        }


class SyntheticIQDataset(Dataset):
    """Small deterministic IQ dataset for pipeline smoke tests."""

    def __init__(
        self,
        num_tx: int = 4,
        num_rx: int = 3,
        samples_per_pair: int = 8,
        length: int = 256,
        seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        g = torch.Generator().manual_seed(int(seed))
        samples = []
        t = torch.linspace(0, 1, int(length))
        for rx in range(int(num_rx)):
            rx_gain = 0.85 + 0.12 * rx
            rx_phase = 0.2 * rx
            for tx in range(int(num_tx)):
                freq = 2.0 + tx
                for k in range(int(samples_per_pair)):
                    phase = rx_phase + 0.03 * k
                    i = rx_gain * torch.sin(2 * torch.pi * freq * t + phase)
                    q = rx_gain * torch.cos(2 * torch.pi * (freq + 0.25) * t + phase)
                    x = torch.stack([i, q], dim=0)
                    x = x + 0.03 * torch.randn(x.shape, generator=g)
                    samples.append((x.float(), tx, rx, f"syn_{tx}_{rx}_{k}"))
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        iq, label, receiver, packet_id = self.samples[int(idx)]
        if self.transform is not None:
            iq = self.transform(iq)
        return {
            "iq": iq,
            "label": int(label),
            "receiver": int(receiver),
            "snr": 20.0 + float(receiver),
            "packet_id": packet_id,
        }


def collate_iq_dict(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "iq": torch.stack([b["iq"] for b in batch], dim=0),
        "label": torch.tensor([int(b["label"]) for b in batch], dtype=torch.long),
        "receiver": torch.tensor([int(b["receiver"]) for b in batch], dtype=torch.long),
        "snr": torch.tensor([float(b["snr"] or 0.0) for b in batch], dtype=torch.float32),
        "packet_id": [b.get("packet_id") for b in batch],
    }


def receiver_split(all_receivers: Iterable[int], target_receivers: Iterable[int]):
    target = {int(x) for x in target_receivers}
    source = [int(r) for r in all_receivers if int(r) not in target]
    return source, sorted(target)
