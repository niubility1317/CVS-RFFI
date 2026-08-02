"""Build the non-formal D114 variance aggregate from the fixed 588-row tap."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from cvsrffi.stage2_d111_g0_source_bundle import _load_strict_tap
from cvsrffi.stage2_d112_g0_source_bundle import (
    _centres_from_strict_tap,
    _derive_assets,
    _load_tap_receipt,
    _quantize_positive,
)
from cvsrffi.stage2_d114_hbpd_qknn import D114Bundle, build_d114_bundle


def build_d114_g0_source_bundle(
    tap_path: str | Path,
    *,
    receipt_path: str | Path,
    checkpoint_sha256: str,
    expected_tap_sha256: str,
    allowed_config_lock_digests: Sequence[str],
) -> D114Bundle:
    pre_relu, labels, receivers, days, _physical, tap_sha = _load_strict_tap(
        Path(tap_path), expected_tap_sha256=expected_tap_sha256
    )
    _load_tap_receipt(
        Path(receipt_path), tap_sha256=tap_sha, checkpoint_sha256=checkpoint_sha256
    )
    x, centres, classes, cells = _centres_from_strict_tap(
        pre_relu, labels, receivers, days
    )
    assets = _derive_assets(x, centres, labels, receivers, days, classes, cells)
    sigma0_amb = np.asarray(assets[4], dtype=np.float64)
    codes, scales, decoded = _quantize_positive(sigma0_amb)
    digest = hashlib.sha256()
    digest.update(str(tap_sha).encode("ascii"))
    digest.update(np.ascontiguousarray(codes).tobytes(order="C"))
    digest.update(np.ascontiguousarray(scales).tobytes(order="C"))
    digest.update(np.ascontiguousarray(decoded).tobytes(order="C"))
    return build_d114_bundle(
        class_registry=classes,
        sigma0_old=decoded,
        checkpoint_sha256=checkpoint_sha256,
        source_aggregate_sha256=digest.hexdigest(),
        allowed_config_lock_digests=allowed_config_lock_digests,
    )


__all__ = ["build_d114_g0_source_bundle"]
