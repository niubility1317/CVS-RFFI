from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from export_spaceborne_features import _rf_statistics_batch  # noqa: E402


def test_rf_statistics_are_finite_32d_and_gain_invariant() -> None:
    rng = np.random.default_rng(713101)
    raw = rng.normal(size=(4, 2, 256)).astype(np.float32)
    descriptor = _rf_statistics_batch(raw)
    scaled = _rf_statistics_batch(raw * 7.5)

    assert descriptor.shape == (4, 32)
    assert np.isfinite(descriptor).all()
    np.testing.assert_allclose(np.linalg.norm(descriptor, axis=1), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(descriptor, scaled, atol=2.0e-6)


def test_rf_statistics_reject_misaligned_iq_shape() -> None:
    with pytest.raises(ValueError, match="expected IQ batch"):
        _rf_statistics_batch(np.zeros((2, 256), dtype=np.float32))
