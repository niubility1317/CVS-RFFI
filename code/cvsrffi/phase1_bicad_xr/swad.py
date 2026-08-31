"""In-memory SWAD checkpoint admission and deterministic state averaging."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _Checkpoint:
    state_dict: dict[str, torch.Tensor]
    score: float
    floors: tuple[float, float, float]


def _finite_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be on the 0--1 scale")
    return result


class SWADAccumulator:
    """Collect near-best checkpoints without material floor regression."""

    def __init__(
        self,
        *,
        score_tolerance: float = 0.005,
        floor_tolerance: float = 0.005,
    ) -> None:
        self.score_tolerance = _finite_metric(
            score_tolerance, name="score_tolerance"
        )
        self.floor_tolerance = _finite_metric(
            floor_tolerance, name="floor_tolerance"
        )
        self._window: list[_Checkpoint] = []
        self._best_score: float | None = None
        self._best_floors: tuple[float, float, float] | None = None
        self._schema: dict[str, tuple[torch.dtype, torch.Size, torch.device]] | None = None

    @property
    def window_size(self) -> int:
        return len(self._window)

    def _copy_state(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("state_dict must be a non-empty mapping")
        copied: dict[str, torch.Tensor] = {}
        schema: dict[str, tuple[torch.dtype, torch.Size, torch.device]] = {}
        for name, tensor in state_dict.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("state_dict must map string keys to tensors")
            if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise ValueError(f"state tensor {name!r} must be finite")
            copied[name] = tensor.detach().clone()
            schema[name] = (tensor.dtype, tensor.shape, tensor.device)
        if self._schema is None:
            self._schema = schema
        elif schema != self._schema:
            raise ValueError("state_dict schema must remain unchanged")
        return copied

    def _is_admissible(self, checkpoint: _Checkpoint) -> bool:
        assert self._best_score is not None
        assert self._best_floors is not None
        if checkpoint.score < self._best_score - self.score_tolerance:
            return False
        return all(
            value >= reference - self.floor_tolerance
            for value, reference in zip(checkpoint.floors, self._best_floors)
        )

    def consider(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        score: float,
        clean_floor: float,
        leo_floor: float,
        receiver_floor: float,
    ) -> bool:
        """Admit one caller-provided checkpoint and return its admission state."""

        score_value = _finite_metric(score, name="score")
        floors = (
            _finite_metric(clean_floor, name="clean_floor"),
            _finite_metric(leo_floor, name="leo_floor"),
            _finite_metric(receiver_floor, name="receiver_floor"),
        )
        copied = self._copy_state(state_dict)
        checkpoint = _Checkpoint(copied, score_value, floors)

        if self._best_score is None or score_value > self._best_score:
            self._best_score = score_value
            self._best_floors = floors
            self._window.append(checkpoint)
            self._window = [item for item in self._window if self._is_admissible(item)]
            return True
        if not self._is_admissible(checkpoint):
            return False
        self._window.append(checkpoint)
        return True

    def averaged_state_dict(self) -> dict[str, torch.Tensor]:
        """Average floating tensors and copy integer/bool buffers from latest."""

        if not self._window:
            raise RuntimeError("cannot average an empty SWAD window")
        latest = self._window[-1].state_dict
        result: dict[str, torch.Tensor] = {}
        for name, latest_tensor in latest.items():
            if torch.is_floating_point(latest_tensor) or torch.is_complex(latest_tensor):
                result[name] = torch.stack(
                    [item.state_dict[name] for item in self._window], dim=0
                ).mean(dim=0)
            else:
                result[name] = latest_tensor.detach().clone()
        return result
