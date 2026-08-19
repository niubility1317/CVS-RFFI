"""Late, bounded RF-lite residual gate for ERBT-IDR M2.4."""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_rf_residual(
    base_coefficient: Any,
    base_bias: Any,
    rf_coefficient: Any,
    rf_bias: Any,
    support_if: Any,
    support_rf: Any,
    support_targets: Any,
    *,
    k_shot: int,
    alpha_max: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    base = np.asarray(base_coefficient, dtype=np.float64)
    bias = np.asarray(base_bias, dtype=np.float64)
    residual = np.asarray(rf_coefficient, dtype=np.float64)
    residual_bias = np.asarray(rf_bias, dtype=np.float64)
    support = np.asarray(support_if, dtype=np.float64)
    rf_rows = np.asarray(support_rf, dtype=np.float64)
    targets = np.asarray(support_targets, dtype=np.int64)
    if base.ndim != 2 or residual.ndim != 2 or base.shape[0] != residual.shape[0]:
        raise ValueError("RF residual geometry is invalid")
    if bias.shape != (base.shape[0],) or residual_bias.shape != (base.shape[0],) or support.ndim != 2 or support.shape[1] != base.shape[1] or len(support) != len(targets):
        raise ValueError("support geometry is invalid")
    if rf_rows.shape != (len(support), residual.shape[1]):
        raise ValueError("RF support geometry is invalid")
    if not 0.0 <= alpha_max <= 0.1:
        raise ValueError("alpha_max must be in [0, 0.1]")
    if k_shot <= 2:
        return np.zeros_like(residual), np.zeros(base.shape[0]), {"mode": "forced_off_k_le_2", "alpha_max": 0.0, "global_help": 0, "global_harm": 0}

    base_score = support @ base.T + bias[None, :]
    augmented_score = base_score + alpha_max * (rf_rows @ residual.T + residual_bias[None, :])
    base_prediction = np.argmax(base_score, axis=1)
    augmented_prediction = np.argmax(augmented_score, axis=1)
    base_correct = base_prediction == targets
    augmented_correct = augmented_prediction == targets
    help_mask = (~base_correct) & augmented_correct
    harm_mask = base_correct & (~augmented_correct)
    global_help = int(np.sum(help_mask))
    global_harm = int(np.sum(harm_mask))
    global_alpha = alpha_max if global_harm == 0 and global_help >= 2 else 0.0
    if k_shot < 10:
        alpha = np.full(base.shape[0], global_alpha)
        mode = "global_no_harm"
    else:
        counts = np.bincount(targets, minlength=base.shape[0]).astype(np.float64)
        shrinkage = counts / (counts + 5.0)
        class_help = np.bincount(targets[help_mask], minlength=base.shape[0])
        class_harm = np.bincount(targets[harm_mask], minlength=base.shape[0])
        class_safe = (class_help >= 2) & (class_harm == 0)
        alpha = global_alpha * shrinkage * class_safe
        mode = "global_then_class_hierarchical"
    gated_coefficient = alpha[:, None] * residual
    gated_bias = alpha * residual_bias
    return gated_coefficient, gated_bias, {
        "mode": mode,
        "alpha_by_class": alpha.tolist(),
        "alpha_max": float(np.max(alpha, initial=0.0)),
        "global_help": global_help,
        "global_harm": global_harm,
    }


__all__ = ["safe_rf_residual"]
