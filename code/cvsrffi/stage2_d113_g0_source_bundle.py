"""Build the non-formal D113 G0 aggregate from the fixed 588-row tap."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from cvsrffi.stage2_d111_g0_source_bundle import _load_strict_tap
from cvsrffi.stage2_d112_g0_source_bundle import (
    EPSILON_VARIANCE_AMB,
    _centres_from_strict_tap,
    _derive_assets,
    _load_tap_receipt,
    _quantize_positive,
    _quantize_rows,
)
from cvsrffi.stage2_d113_bcat_qknn import D113Bundle, FEATURE_DIM, build_d113_bundle


class D113G0SourceBundleError(ValueError):
    """Raised when the fixed G0 source aggregate cannot be derived."""


def build_d113_g0_source_bundle(
    tap_path: str | Path,
    *,
    receipt_path: str | Path,
    checkpoint_sha256: str,
    expected_tap_sha256: str,
    allowed_config_lock_digests: Sequence[str],
) -> D113Bundle:
    """Return only quantized aggregate assets; never retain source rows or IDs."""

    pre_relu, labels, receivers, days, _physical, tap_sha = _load_strict_tap(
        Path(tap_path), expected_tap_sha256=expected_tap_sha256
    )
    _load_tap_receipt(
        Path(receipt_path),
        tap_sha256=tap_sha,
        checkpoint_sha256=checkpoint_sha256,
    )
    x, centres, classes, cells = _centres_from_strict_tap(
        pre_relu, labels, receivers, days
    )
    (
        ground,
        _q0,
        _basis,
        _sigma0_r,
        sigma0_amb,
        _v_g_r,
        _v_g_amb,
        _tau_h_r,
        _endpoint_quantization,
    ) = _derive_assets(x, centres, labels, receivers, days, classes, cells)

    projected = np.empty_like(centres)
    identity = np.eye(FEATURE_DIM, dtype=np.float64)
    for class_index in range(len(classes)):
        projector = identity - np.outer(ground[class_index], ground[class_index])
        projected[:, class_index, :] = (
            centres[:, class_index, :] - ground[class_index][None, :]
        ) @ projector
    shared = np.mean(projected, axis=1)
    tau_b2 = EPSILON_VARIANCE_AMB + float(np.mean(np.square(shared)))
    v_ground = EPSILON_VARIANCE_AMB + np.mean(
        np.square(projected - shared[:, None, :]), axis=(0, 2)
    )
    if (
        not math.isfinite(tau_b2)
        or tau_b2 <= 0.0
        or not np.isfinite(v_ground).all()
        or np.any(v_ground <= 0.0)
    ):
        raise D113G0SourceBundleError("D113 common-shift source moments are invalid")

    _g_codes, _g_scales, decoded_ground = _quantize_rows(ground)
    q_ground = np.mean(np.square(decoded_ground - ground), axis=1)
    _sigma_codes, _sigma_scale, decoded_sigma0 = _quantize_positive(sigma0_amb)
    _vg_codes, _vg_scale, decoded_v_ground = _quantize_positive(v_ground)
    _tau_codes, _tau_scale, decoded_tau = _quantize_positive(np.asarray(tau_b2))

    digest = hashlib.sha256()
    digest.update(str(tap_sha).encode("ascii"))
    for value in (
        decoded_ground,
        decoded_sigma0,
        decoded_v_ground,
        q_ground,
        np.asarray([decoded_tau], dtype=np.float64),
    ):
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return build_d113_bundle(
        class_registry=classes,
        ground=decoded_ground,
        sigma0=decoded_sigma0,
        v_ground=decoded_v_ground,
        quantization_mse=q_ground,
        tau_b2=float(decoded_tau),
        checkpoint_sha256=checkpoint_sha256,
        source_aggregate_sha256=digest.hexdigest(),
        allowed_config_lock_digests=allowed_config_lock_digests,
    )


__all__ = ["D113G0SourceBundleError", "build_d113_g0_source_bundle"]
