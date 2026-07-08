from __future__ import annotations

from collections.abc import Sequence

import torch


def top1_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes")
    if target.numel() == 0:
        return 0.0
    return float((pred == target).float().mean().item())


def harmonic_accuracy(*, old_accuracy: float, new_accuracy: float) -> float:
    old = float(old_accuracy)
    new = float(new_accuracy)
    denom = old + new
    if denom <= 0:
        return 0.0
    return 2.0 * old * new / denom


def forgetting_by_session(accuracy_matrix: torch.Tensor) -> list[float]:
    """Compute formula (38) from a lower-triangular task accuracy matrix."""

    if accuracy_matrix.ndim != 2 or accuracy_matrix.size(0) != accuracy_matrix.size(1):
        raise ValueError("accuracy_matrix must be square")
    values: list[float] = []
    for t in range(1, accuracy_matrix.size(0)):
        drops = []
        for k in range(t):
            current = accuracy_matrix[t, k]
            history = accuracy_matrix[:t, k]
            history = history[torch.isfinite(history)]
            if torch.isfinite(current) and history.numel() > 0:
                drops.append(float((history.max() - current).clamp_min(0).item()))
        values.append(sum(drops) / len(drops) if drops else 0.0)
    return values


def average_incremental_metrics(
    *,
    session_accuracies: Sequence[float],
    old_accuracies: Sequence[float],
    new_accuracies: Sequence[float],
    accuracy_matrix: torch.Tensor,
) -> dict[str, float]:
    if not session_accuracies:
        raise ValueError("session_accuracies must not be empty")
    if len(old_accuracies) != len(new_accuracies):
        raise ValueError("old_accuracies and new_accuracies must have matching lengths")
    h_values = [harmonic_accuracy(old_accuracy=o, new_accuracy=n) for o, n in zip(old_accuracies, new_accuracies)]
    f_values = forgetting_by_session(accuracy_matrix)
    return {
        "A_bar": float(sum(map(float, session_accuracies)) / len(session_accuracies)),
        "H_bar": float(sum(h_values) / len(h_values)) if h_values else 0.0,
        "F_bar": float(sum(f_values) / len(f_values)) if f_values else 0.0,
    }
