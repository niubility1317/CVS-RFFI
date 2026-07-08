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
                drops.append(float((history.max() - current).item()))
        values.append(sum(drops) / len(drops) if drops else 0.0)
    return values


def average_incremental_metrics(
    *,
    session_accuracies: Sequence[float],
    old_accuracies: Sequence[float],
    new_accuracies: Sequence[float],
    accuracy_matrix: torch.Tensor,
    average_denominator: str = "incremental_sessions",
    forgetting_denominator: str = "incremental_sessions",
) -> dict[str, float]:
    if not session_accuracies:
        raise ValueError("session_accuracies must not be empty")
    if not old_accuracies or not new_accuracies:
        raise ValueError("old_accuracies and new_accuracies must not be empty")
    if len(old_accuracies) != len(new_accuracies):
        raise ValueError("old_accuracies and new_accuracies must have matching lengths")
    if len(session_accuracies) != len(old_accuracies):
        raise ValueError("session_accuracies must match old/new accuracy length")
    if accuracy_matrix.size(0) != len(session_accuracies):
        raise ValueError("accuracy_matrix size must match session_accuracies length")
    if average_denominator not in {"total_sessions", "incremental_sessions"}:
        raise ValueError("average_denominator must be total_sessions or incremental_sessions")
    if forgetting_denominator not in {"total_sessions", "incremental_sessions"}:
        raise ValueError("forgetting_denominator must be total_sessions or incremental_sessions")
    h_values = [harmonic_accuracy(old_accuracy=o, new_accuracy=n) for o, n in zip(old_accuracies, new_accuracies)]
    a_values = list(map(float, session_accuracies))
    if average_denominator == "incremental_sessions" and len(a_values) > 1:
        a_values = a_values[1:]
        h_values = h_values[1:]
    f_values = forgetting_by_session(accuracy_matrix)
    f_denominator = len(session_accuracies) if forgetting_denominator == "total_sessions" else len(f_values)
    return {
        "A_bar": float(sum(a_values) / len(a_values)),
        "H_bar": float(sum(h_values) / len(h_values)) if h_values else 0.0,
        "F_bar": float(sum(f_values) / f_denominator) if f_denominator else 0.0,
    }
