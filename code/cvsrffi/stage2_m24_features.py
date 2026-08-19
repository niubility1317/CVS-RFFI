"""Physical identity+FFT feature geometry for ERBT-IDR M2.4."""

from __future__ import annotations

from typing import Any

import numpy as np


IDENTITY_DIM = 160
FFT_DIM = 96
IF_DIM = IDENTITY_DIM + FFT_DIM
_EPS = 1.0e-12


class M24FeatureError(ValueError):
    pass


def _unit_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm <= _EPS):
        raise M24FeatureError("identity/FFT block cannot be degenerate")
    return value / norm


def physical_if256(value: Any) -> np.ndarray:
    """Return normalize([normalize(identity160); 4 normalize(FFT96)])."""

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] <= 0 or rows.shape[1] < IF_DIM:
        raise M24FeatureError("M2.4 features must be finite N x >=256")
    if not np.isfinite(rows).all():
        raise M24FeatureError("M2.4 features must be finite")
    joined = np.concatenate(
        [_unit_rows(rows[:, :IDENTITY_DIM]), 4.0 * _unit_rows(rows[:, IDENTITY_DIM:IF_DIM])],
        axis=1,
    )
    return _unit_rows(joined).astype(np.float32)


__all__ = ["FFT_DIM", "IDENTITY_DIM", "IF_DIM", "M24FeatureError", "physical_if256"]
