"""Scale-relative PSD covariance repair for M2.4."""

from __future__ import annotations

from typing import Any

import numpy as np


class M24CovarianceError(ValueError):
    pass


def relative_psd_jitter(value: Any, *, relative_floor: float = 1.0e-4) -> tuple[np.ndarray, dict[str, float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise M24CovarianceError("covariance must be square")
    if not np.isfinite(matrix).all() or relative_floor <= 0.0:
        raise M24CovarianceError("covariance controls are invalid")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalue, eigenvector = np.linalg.eigh(symmetric)
    scale = max(abs(float(np.trace(symmetric))) / matrix.shape[0], float(np.max(np.abs(eigenvalue))), np.finfo(np.float64).tiny)
    floor = relative_floor * scale
    clipped = np.maximum(eigenvalue, floor)
    repaired = (eigenvector * clipped[None, :]) @ eigenvector.T
    repaired = 0.5 * (repaired + repaired.T)
    return repaired, {
        "relative_floor": float(relative_floor),
        "scale": scale,
        "jitter": float(max(0.0, floor - float(np.min(eigenvalue)))),
        "minimum_eigenvalue": float(np.min(clipped)),
    }


__all__ = ["M24CovarianceError", "relative_psd_jitter"]
