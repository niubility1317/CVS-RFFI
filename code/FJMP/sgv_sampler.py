"""Sampler helpers for paired SGV batches."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Sampler


def validate_paired_batch(batch: Mapping[str, torch.Tensor]) -> bool:
    """Validate the clean/sat pairing contract used by SGV-BP training."""

    required = ["x_clean", "x_sat", "y"]
    for key in required:
        if key not in batch:
            raise KeyError(f"paired SGV batch missing {key}")
    if batch["x_clean"].shape != batch["x_sat"].shape:
        raise ValueError("x_clean and x_sat must have identical shapes.")
    if "y_sat" in batch and not torch.equal(batch["y"], batch["y_sat"]):
        raise ValueError("y_clean and y_sat must match.")
    for key in ["rx", "day", "domain"]:
        sat_key = f"{key}_sat"
        if key in batch and sat_key in batch and not torch.equal(batch[key], batch[sat_key]):
            raise ValueError(f"{key} clean/sat metadata must match.")
    return True


class PairedSGVBatchSampler(Sampler[list[int]]):
    """Class-rx-day balanced batch sampler for paired clean/sat generation."""

    def __init__(
        self,
        labels: Sequence[int],
        rx: Sequence[int] | None = None,
        day: Sequence[int] | None = None,
        *,
        batch_size: int = 256,
        generator: torch.Generator | None = None,
    ) -> None:
        self.labels = [int(x) for x in labels]
        self.rx = [0 for _ in self.labels] if rx is None else [int(x) for x in rx]
        self.day = [0 for _ in self.labels] if day is None else [int(x) for x in day]
        self.batch_size = int(batch_size)
        self.generator = generator
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for idx, key in enumerate(zip(self.labels, self.rx, self.day)):
            self.groups[key].append(idx)
        self.keys = list(self.groups.keys())

    def __len__(self) -> int:
        return max(1, len(self.labels) // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        if not self.keys:
            return
        key_order = torch.randperm(len(self.keys), generator=self.generator).tolist()
        cursor = 0
        for _ in range(len(self)):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                key = self.keys[key_order[cursor % len(key_order)]]
                pool = self.groups[key]
                pick = int(torch.randint(len(pool), (1,), generator=self.generator).item())
                batch.append(pool[pick])
                cursor += 1
            yield batch


__all__ = ["PairedSGVBatchSampler", "validate_paired_batch"]
