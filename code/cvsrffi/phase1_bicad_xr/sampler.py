"""Source-only TX/RX balanced sampling for BiCAD-XR episodes.

The sampler treats each row of the input metadata as one physical sample.
Only complete ``(tx, receiver)`` cells are emitted, so a structured episode
can be sparse without manufacturing samples for a missing cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any

import torch


def _as_integer_tensor(name: str, values: Any) -> torch.Tensor:
    """Normalize one-dimensional integer metadata and fail closed."""

    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must contain integer labels") from exc

    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if tensor.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer labels")

    if tensor.is_complex():
        raise ValueError(f"{name} must contain integer labels")
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain finite integer labels")
        converted = tensor.to(dtype=torch.long)
        if not torch.equal(tensor, converted.to(dtype=tensor.dtype)):
            raise ValueError(f"{name} must contain integer labels")
    else:
        try:
            converted = tensor.to(dtype=torch.long)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain integer labels") from exc

    return converted.detach().cpu().contiguous()


def _positive_integer(name: str, value: Any) -> int:
    """Return a strict positive integer parameter."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        resolved = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(resolved)


def _optional_positive_integer(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _positive_integer(name, value)


def _resolve_label_extent(name: str, labels: torch.Tensor, requested: Any) -> int:
    """Resolve a contiguous non-negative label extent and validate bounds."""

    minimum = int(labels.min().item())
    maximum = int(labels.max().item())
    extent = _optional_positive_integer(name, requested)
    if extent is None:
        if minimum < 0:
            raise ValueError(f"{name} labels out of range: expected non-negative labels")
        return maximum + 1
    if minimum < 0 or maximum >= extent:
        raise ValueError(
            f"{name} labels out of range: expected values in [0, {extent})"
        )
    return extent


def _validate_generator(generator: torch.Generator | None) -> torch.Generator:
    if generator is None:
        return torch.default_generator
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator or None")
    return generator


@dataclass(frozen=True)
class StructuredEpisode:
    """A sparse structured episode containing only sampled physical rows."""

    indices: tuple[int, ...]
    tx: torch.Tensor
    receiver: torch.Tensor
    day: torch.Tensor
    valid_cells: torch.Tensor

    def __post_init__(self) -> None:
        lengths = {len(self.indices), self.tx.numel(), self.receiver.numel(), self.day.numel()}
        if len(lengths) != 1:
            raise ValueError("episode fields must have the same length")
        if self.valid_cells.ndim != 2 or self.valid_cells.dtype != torch.bool:
            raise ValueError("valid_cells must be a two-dimensional boolean mask")


class BalancedIndexPool:
    """Index a source metadata pool by ``(tx, receiver)`` physical cells.

    The default physical index for a row is its position in the input arrays.
    ``physical_indices`` can be supplied when the arrays are a filtered view
    of a larger source table; those IDs must remain unique.
    """

    def __init__(
        self,
        tx: Any,
        receiver: Any,
        day: Any,
        *,
        num_classes: int | None = None,
        num_receivers: int | None = None,
        physical_indices: Any | None = None,
    ) -> None:
        tx_tensor = _as_integer_tensor("tx", tx)
        receiver_tensor = _as_integer_tensor("receiver", receiver)
        day_tensor = _as_integer_tensor("day", day)
        if not (len(tx_tensor) == len(receiver_tensor) == len(day_tensor)):
            raise ValueError("tx, receiver, and day must have the same length")

        if int(day_tensor.min().item()) < 0:
            raise ValueError("day labels out of range: expected non-negative labels")

        resolved_classes = _resolve_label_extent("num_classes", tx_tensor, num_classes)
        resolved_receivers = _resolve_label_extent(
            "num_receivers", receiver_tensor, num_receivers
        )

        if physical_indices is None:
            physical_tensor = torch.arange(len(tx_tensor), dtype=torch.long)
        else:
            physical_tensor = _as_integer_tensor("physical_indices", physical_indices)
            if len(physical_tensor) != len(tx_tensor):
                raise ValueError("physical_indices must have the same length as metadata")
            if int(physical_tensor.min().item()) < 0:
                raise ValueError("physical_indices must be non-negative")
            if torch.unique(physical_tensor).numel() != physical_tensor.numel():
                raise ValueError("physical_indices must be unique")

        cells: list[list[tuple[int, ...]]] = [
            [[] for _ in range(resolved_receivers)] for _ in range(resolved_classes)
        ]
        for local_index, (class_id, receiver_id) in enumerate(
            zip(tx_tensor.tolist(), receiver_tensor.tolist())
        ):
            cells[int(class_id)][int(receiver_id)].append(int(local_index))

        self._tx = tx_tensor
        self._receiver = receiver_tensor
        self._day = day_tensor
        self._physical_indices = physical_tensor
        self._cells = tuple(
            tuple(tuple(cell) for cell in class_cells) for class_cells in cells
        )
        self.num_classes = resolved_classes
        self.num_receivers = resolved_receivers

    @property
    def tx(self) -> torch.Tensor:
        return self._tx.clone()

    @property
    def receiver(self) -> torch.Tensor:
        return self._receiver.clone()

    @property
    def day(self) -> torch.Tensor:
        return self._day.clone()

    @property
    def cells(self) -> dict[tuple[int, int], tuple[int, ...]]:
        return {
            (class_id, receiver_id): self._cells[class_id][receiver_id]
            for class_id in range(self.num_classes)
            for receiver_id in range(self.num_receivers)
        }

    def __len__(self) -> int:
        return len(self._tx)

    def valid_cell_mask(self, samples_per_cell: int) -> torch.Tensor:
        """Return the complete-cell mask without consuming a random generator."""

        count = _positive_integer("samples_per_cell", samples_per_cell)
        mask = torch.zeros(
            (self.num_classes, self.num_receivers), dtype=torch.bool
        )
        for class_id in range(self.num_classes):
            for receiver_id in range(self.num_receivers):
                mask[class_id, receiver_id] = (
                    len(self._cells[class_id][receiver_id]) >= count
                )
        return mask

    def sample(
        self,
        samples_per_cell: int,
        *,
        generator: torch.Generator | None = None,
    ) -> StructuredEpisode:
        """Sample each complete cell without replacement.

        Incomplete cells contribute no rows at all.  This is the fail-closed
        behavior that prevents duplicate or cross-cell placeholder samples.
        """

        count = _positive_integer("samples_per_cell", samples_per_cell)
        resolved_generator = _validate_generator(generator)
        valid_cells = self.valid_cell_mask(count)
        selected_local_indices: list[int] = []

        generator_device = getattr(resolved_generator, "device", torch.device("cpu"))
        for class_id in range(self.num_classes):
            for receiver_id in range(self.num_receivers):
                cell = self._cells[class_id][receiver_id]
                if len(cell) < count:
                    continue
                permutation = torch.randperm(
                    len(cell), generator=resolved_generator, device=generator_device
                )
                selected_local_indices.extend(
                    cell[int(position)] for position in permutation[:count].tolist()
                )

        local_indices = torch.tensor(selected_local_indices, dtype=torch.long)
        physical_indices = tuple(
            int(value) for value in self._physical_indices[local_indices].tolist()
        )
        return StructuredEpisode(
            indices=physical_indices,
            tx=self._tx[local_indices].clone(),
            receiver=self._receiver[local_indices].clone(),
            day=self._day[local_indices].clone(),
            valid_cells=valid_cells,
        )


def build_structured_episode(
    tx: Any,
    receiver: Any,
    day: Any,
    samples_per_cell: int,
    generator: torch.Generator | None = None,
    *,
    num_classes: int | None = None,
    num_receivers: int | None = None,
    physical_indices: Any | None = None,
) -> StructuredEpisode:
    """Build one sparse TX/RX structured episode from source metadata."""

    pool = BalancedIndexPool(
        tx,
        receiver,
        day,
        num_classes=num_classes,
        num_receivers=num_receivers,
        physical_indices=physical_indices,
    )
    return pool.sample(samples_per_cell, generator=generator)


__all__ = [
    "BalancedIndexPool",
    "StructuredEpisode",
    "build_structured_episode",
]
