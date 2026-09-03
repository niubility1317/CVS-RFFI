from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import torch
import torch.nn.functional as F


class PersistentConflictProjector:
    """Protect the primary task only after a persistent auxiliary conflict."""

    def __init__(self, *, window: int = 3, threshold: float = -0.1) -> None:
        if int(window) < 1 or not -1.0 <= float(threshold) < 0.0:
            raise ValueError("invalid persistent conflict configuration")
        self.window = int(window)
        self.threshold = float(threshold)
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def project(
        self,
        name: str,
        *,
        auxiliary: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if auxiliary.shape != base.shape:
            raise ValueError("auxiliary and base gradients must align")
        aux_flat = auxiliary.reshape(-1).float()
        base_flat = base.reshape(-1).float()
        cosine = float(F.cosine_similarity(aux_flat, base_flat, dim=0, eps=1e-12).item())
        history = self._history[str(name)]
        history.append(cosine)
        persistent = len(history) == self.window and all(value < self.threshold for value in history)
        if not persistent:
            return auxiliary, {"cosine": cosine, "projected": False}
        coefficient = torch.dot(aux_flat, base_flat) / base_flat.square().sum().clamp_min(1e-12)
        projected = aux_flat - coefficient.clamp_max(0.0) * base_flat
        return projected.reshape_as(auxiliary).to(dtype=auxiliary.dtype), {
            "cosine": cosine,
            "projected": True,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "threshold": self.threshold,
            "history": {key: list(values) for key, values in self._history.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["window"]) != self.window or float(state["threshold"]) != self.threshold:
            raise ValueError("gradient-controller state configuration does not match")
        self._history.clear()
        for key, values in dict(state.get("history", {})).items():
            history = self._history[str(key)]
            history.extend(float(value) for value in values[-self.window :])
