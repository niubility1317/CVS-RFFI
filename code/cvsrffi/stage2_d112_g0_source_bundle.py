"""In-memory non-formal Phase1 aggregate for the D112 SEAM-qKNN G0 check.

The sole public function consumes the fixed 588-row D106 strict tap, retains
no source row or identifier, quantizes every numeric Phase1 asset to int8 and
only then constructs the typed :class:`D112Bundle`.  Its result is explicitly
functional-only: it is not a Phase2 deployment bundle and it exposes no truth
or performance surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cvsrffi.stage2_d111_g0_source_bundle import (
    EXPECTED_CELLS,
    EXPECTED_CLASSES,
    EXPECTED_ROWS,
    _load_strict_tap,
)
from cvsrffi.stage2_d112_seam_bundle import (
    FEATURE_DIM,
    G0_COMPONENT_STATE,
    SHARED_RANK,
    D112Bundle,
    build_d112_g0_bundle,
)
from cvsrffi.stage2_d112_seam_qknn import (
    EPSILON_GEO,
    EPSILON_VARIANCE_AMB,
    EPSILON_VARIANCE_R,
    D112SEAMError,
    sphere_exp,
    sphere_log,
    sphere_parallel_transport,
)


SCHEMA = "cvs.phase1.d112.g0_source_aggregate.v1"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
ROUNDING_SCHEMA = "numpy_rint_ties_to_even_symmetric_int8_v1"
CANONICAL_REDUCTION_SCHEMA = "float64_byte_order_pairwise_reduction_v1"


class D112G0SourceBundleError(ValueError):
    """Raised when the D112 G0-only Phase1 aggregate cannot be constructed."""


class D112G0GeometryDegeneracy(D112G0SourceBundleError):
    """A legitimate Phase1 sphere-chart/rank degeneration that resolves to M0."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_row_key(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes(order="C")


def _ordered_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D112G0SourceBundleError("canonical row reduction received an invalid matrix")
    return np.asarray(sorted((row.copy() for row in rows), key=_canonical_row_key), dtype=np.float64)


def _pairwise_sum(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise D112G0SourceBundleError("pairwise reduction received no finite rows")
    pending = values.copy()
    while len(pending) > 1:
        pairs = len(pending) // 2
        merged = pending[: 2 * pairs : 2] + pending[1 : 2 * pairs : 2]
        if len(pending) % 2:
            merged = np.concatenate((merged, pending[-1:, :]), axis=0)
        pending = merged
    return pending[0]


def _canonical_mean_rows(rows: np.ndarray) -> np.ndarray:
    ordered = _ordered_rows(rows)
    return _pairwise_sum(ordered) / float(len(ordered))


def _canonical_mean_scalars(values: Sequence[float]) -> float:
    rows = np.asarray(values, dtype=np.float64).reshape((-1, 1))
    if len(rows) == 0 or not np.isfinite(rows).all():
        raise D112G0SourceBundleError("scalar canonical reduction received no finite values")
    ordered = np.asarray(
        sorted((row.copy() for row in rows), key=lambda row: np.asarray(row, dtype="<f8").tobytes()),
        dtype=np.float64,
    )
    return float(_pairwise_sum(np.pad(ordered, ((0, 0), (0, FEATURE_DIM - 1))))[0] / len(ordered))


def _unit(value: np.ndarray, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
        raise D112G0SourceBundleError(f"{field} must be finite [{FEATURE_DIM}]")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= EPSILON_GEO:
        raise D112G0SourceBundleError(f"{field} is geometrically degenerate")
    return vector / norm


def _sphere_log(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    try:
        return sphere_log(base, point)
    except D112SEAMError as exc:
        if "antipodal chart" in str(exc):
            raise D112G0GeometryDegeneracy("D112 Phase1 Log chart is degenerate") from exc
        raise D112G0SourceBundleError("D112 Phase1 Log computation failed") from exc


def _sphere_parallel_transport(
    base: np.ndarray, destination: np.ndarray, tangent: np.ndarray
) -> np.ndarray:
    try:
        return sphere_parallel_transport(base, destination, tangent)
    except D112SEAMError as exc:
        if "antipodal chart" in str(exc):
            raise D112G0GeometryDegeneracy("D112 Phase1 PT chart is degenerate") from exc
        raise D112G0SourceBundleError("D112 Phase1 PT computation failed") from exc


def _sphere_exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    try:
        return sphere_exp(base, tangent)
    except D112SEAMError as exc:
        raise D112G0SourceBundleError("D112 Phase1 Exp computation failed") from exc


def _canonical_sign(rows: np.ndarray) -> np.ndarray:
    result = np.asarray(rows, dtype=np.float64).copy()
    for index in range(len(result)):
        pivot = int(np.argmax(np.abs(result[index])))
        if result[index, pivot] < 0.0:
            result[index] *= -1.0
    return result


def _quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D112G0SourceBundleError("int8 vector quantizer received an invalid asset")
    maximum = np.max(np.abs(rows), axis=1)
    if np.any(maximum <= 0.0):
        raise D112G0SourceBundleError("int8 vector quantizer received a zero asset")
    scale = (maximum / 127.0).astype(np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise D112G0SourceBundleError("int8 vector quantization scale is invalid")
    code = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
    if np.any(code == -128):
        raise D112G0SourceBundleError("int8 vector quantizer emitted forbidden -128")
    decoded = code.astype(np.float32) * scale[:, None]
    return code, scale, decoded.astype(np.float64)


def _quantize_nonnegative(
    value: np.ndarray | float, *, upper: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise D112G0SourceBundleError("int8 nonnegative quantizer received an invalid asset")
    maximum = float(np.max(array))
    scale = np.asarray(maximum / 127.0 if maximum > 0.0 else 1.0, dtype=np.float32)
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise D112G0SourceBundleError("int8 nonnegative quantization scale is invalid")
    quotient = array / float(scale)
    code = (
        np.ceil(quotient) if upper else np.rint(quotient)
    )
    code = np.clip(code, 0, 127).astype(np.int8)
    decoded = code.astype(np.float32) * scale
    if np.any(code == -128) or not np.isfinite(decoded).all():
        raise D112G0SourceBundleError("int8 nonnegative quantizer emitted an invalid code")
    if upper and np.any(decoded.astype(np.float64) + 1.0e-30 < array):
        # The scale itself was rounded to FP32.  Advance it once and recompute
        # so an explicitly declared quantization error remains conservative.
        scale = np.nextafter(scale, np.float32(np.inf), dtype=np.float32)
        code = np.clip(np.ceil(array / float(scale)), 0, 127).astype(np.int8)
        decoded = code.astype(np.float32) * scale
        if np.any(decoded.astype(np.float64) + 1.0e-30 < array):
            raise D112G0SourceBundleError("nonnegative upper quantization lost conservatism")
    return code, scale, decoded.astype(np.float64)


def _quantize_positive(value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise D112G0SourceBundleError("int8 positive quantizer received an invalid asset")
    scale = np.asarray(float(np.max(array)) / 127.0, dtype=np.float32)
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise D112G0SourceBundleError("int8 positive quantization scale is invalid")
    code = np.clip(np.rint(array / float(scale)), 1, 127).astype(np.int8)
    decoded = code.astype(np.float32) * scale
    if np.any(code == -128) or not np.isfinite(decoded).all() or np.any(decoded <= 0.0):
        raise D112G0SourceBundleError("int8 positive quantizer emitted an invalid code")
    return code, scale, decoded.astype(np.float64)


def _asset_digest(
    *,
    classes: tuple[str, ...],
    tap_sha256: str,
    tap_receipt_sha256: str,
    assets: Sequence[tuple[str, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema": SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "rounding_schema": ROUNDING_SCHEMA,
                "canonical_reduction_schema": CANONICAL_REDUCTION_SCHEMA,
                "class_registry": list(classes),
                "source_tap_sha256": tap_sha256,
                "source_tap_receipt_sha256": tap_receipt_sha256,
            }
        )
    )
    for name, value in assets:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fallback_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Finite inert geometry used only when Phase1 charts/rank are undefined."""

    q0 = np.zeros(FEATURE_DIM, dtype=np.float64)
    q0[0] = 1.0
    g = np.repeat(q0[None, :], EXPECTED_CLASSES, axis=0)
    basis = np.zeros((SHARED_RANK, FEATURE_DIM), dtype=np.float64)
    for index in range(SHARED_RANK):
        basis[index, index + 1] = 1.0
    variance_r = np.full(EXPECTED_CLASSES, EPSILON_VARIANCE_R, dtype=np.float64)
    variance_amb = np.full(EXPECTED_CLASSES, EPSILON_VARIANCE_AMB, dtype=np.float64)
    endpoint_q = np.zeros(EXPECTED_CLASSES, dtype=np.float64)
    return (
        g,
        q0,
        basis,
        variance_r,
        variance_amb,
        variance_r.copy(),
        variance_amb.copy(),
        EPSILON_VARIANCE_R,
        endpoint_q,
    )


def _load_tap_receipt(
    path: Path,
    *,
    tap_sha256: str,
    checkpoint_sha256: str,
) -> str:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise D112G0SourceBundleError(
            "strict-tap receipt must be one absolute regular non-symlink file"
        )
    raw = path.read_bytes()
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D112G0SourceBundleError("strict-tap receipt is not canonical UTF-8 JSON") from exc
    if not isinstance(receipt, dict):
        raise D112G0SourceBundleError("strict-tap receipt must be one JSON object")
    if (
        receipt.get("schema") != "cvs.phase1.d106.ls_strict_tap_receipt.v1"
        or receipt.get("tap_archive_sha256") != tap_sha256
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("row_count") != EXPECTED_ROWS
        or receipt.get("target_access") is not False
        or receipt.get("formal_query_access") is not False
        or receipt.get("tap_archive_name") != "d106_ls_strict_tap.npz"
    ):
        raise D112G0SourceBundleError("strict-tap receipt identity/permission drift")
    loader = receipt.get("checkpoint_loader")
    if (
        not isinstance(loader, dict)
        or loader.get("exact_frozen_checkpoint_sha256_required") != checkpoint_sha256
        or loader.get("caller_selected_checkpoint_allowed") is not False
    ):
        raise D112G0SourceBundleError("strict-tap checkpoint-loader binding drift")
    return receipt_sha256


def _centres_from_strict_tap(
    pre_relu: np.ndarray,
    labels: Sequence[str],
    receivers: Sequence[str],
    days: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[tuple[str, str], ...]]:
    raw = np.maximum(np.asarray(pre_relu, dtype=np.float64), 0.0)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if raw.shape != (EXPECTED_ROWS, FEATURE_DIM) or np.any(norms <= EPSILON_GEO):
        raise D112G0SourceBundleError("ReLU strict-tap feature contains a zero vector")
    x = raw / norms
    classes = tuple(sorted(set(str(item) for item in labels), key=lambda item: item.encode("utf-8")))
    if len(classes) != EXPECTED_CLASSES:
        raise D112G0SourceBundleError("D112 source class registry drift")
    cells = tuple(
        sorted(
            {(str(receiver), str(day)) for receiver, day in zip(receivers, days, strict=True)},
            key=lambda item: (item[0].encode("utf-8"), item[1].encode("utf-8")),
        )
    )
    if len(cells) != EXPECTED_CELLS:
        raise D112G0SourceBundleError("D112 source receiver-day grid drift")
    labels_array = np.asarray(labels, dtype=object)
    receivers_array = np.asarray(receivers, dtype=object)
    days_array = np.asarray(days, dtype=object)
    centres = np.empty((EXPECTED_CELLS, EXPECTED_CLASSES, FEATURE_DIM), dtype=np.float64)
    for domain_index, (receiver, day) in enumerate(cells):
        domain_mask = (receivers_array == receiver) & (days_array == day)
        for class_index, class_id in enumerate(classes):
            local = x[domain_mask & (labels_array == class_id)]
            if len(local) < 2:
                raise D112G0SourceBundleError("each D112 receiver-day-class cell needs at least two rows")
            centres[domain_index, class_index] = _unit(
                _canonical_mean_rows(local), "D112 receiver-day-class centre"
            )
    return x, centres, classes, cells


def _shared_basis(tangent_centres: np.ndarray, q0: np.ndarray) -> np.ndarray:
    projectors: list[np.ndarray] = []
    for class_index in range(EXPECTED_CLASSES):
        rows = _ordered_rows(tangent_centres[:, class_index, :])
        _left, singular, right = np.linalg.svd(rows, full_matrices=False)
        if (
            len(singular) <= SHARED_RANK
            or singular[SHARED_RANK - 1] <= max(EPSILON_GEO, singular[0] * 1.0e-10)
            or singular[SHARED_RANK - 1] - singular[SHARED_RANK]
            <= max(EPSILON_GEO, singular[0] * 1.0e-10)
        ):
            raise D112G0GeometryDegeneracy("D112 Phase1 class projector lacks an identified rank-three cutoff")
        basis = right[:SHARED_RANK]
        projectors.append(np.ascontiguousarray(basis.T @ basis, dtype=np.float64))
    ordered = sorted(projectors, key=_canonical_row_key)
    projection = _pairwise_sum(
        np.asarray([item.reshape(-1) for item in ordered], dtype=np.float64)
    ).reshape((FEATURE_DIM, FEATURE_DIM)) / float(EXPECTED_CLASSES)
    eigenvalue, eigenvector = np.linalg.eigh(projection)
    order = np.argsort(eigenvalue)[::-1]
    gap = float(eigenvalue[order[SHARED_RANK - 1]] - eigenvalue[order[SHARED_RANK]])
    if not math.isfinite(gap) or gap <= EPSILON_GEO:
        raise D112G0GeometryDegeneracy("D112 shared projection has an undefined rank-three cutoff")
    candidate = eigenvector[:, order[:SHARED_RANK]].T
    tangent = candidate - (candidate @ q0)[:, None] * q0[None, :]
    left, singular, right = np.linalg.svd(tangent, full_matrices=False)
    if singular[-1] <= max(EPSILON_GEO, singular[0] * 1.0e-10):
        raise D112G0GeometryDegeneracy("D112 shared basis lost tangent rank")
    return _canonical_sign(left @ right)


def _canonical_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if values.shape != (EXPECTED_CLASSES - 1, SHARED_RANK) or weights.shape != (EXPECTED_CLASSES - 1,):
        raise D112G0SourceBundleError("D112 P1 LOO weighting layout drift")
    pairs = sorted(
        ((float(weight), value.copy()) for weight, value in zip(weights, values, strict=True)),
        key=lambda pair: np.asarray([pair[0]], dtype="<f8").tobytes() + _canonical_row_key(pair[1]),
    )
    ordered_weights = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    ordered_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if not np.isfinite(ordered_weights).all() or np.any(ordered_weights <= 0.0):
        raise D112G0SourceBundleError("D112 P1 LOO weighting became invalid")
    denominator = _canonical_mean_scalars(ordered_weights.tolist()) * len(ordered_weights)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise D112G0SourceBundleError("D112 P1 LOO weight sum is invalid")
    numerator = _pairwise_sum(ordered_values * ordered_weights[:, None])
    return numerator / denominator


def _endpoint_quantization_mse(endpoints: np.ndarray) -> float:
    ordered = _ordered_rows(endpoints)
    _code, _scale, decoded = _quantize_rows(ordered)
    return _canonical_mean_scalars(
        [float(np.sum(np.square(row - reference)) / FEATURE_DIM) for row, reference in zip(decoded, ordered, strict=True)]
    )


def _derive_assets(
    x: np.ndarray,
    centres: np.ndarray,
    labels: Sequence[str],
    receivers: Sequence[str],
    days: Sequence[str],
    classes: tuple[str, ...],
    cells: tuple[tuple[str, str], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    try:
        g = np.stack(
            [
                _unit(
                    _canonical_mean_rows(centres[:, class_index, :]),
                    "D112 ground anchor",
                )
                for class_index in range(EXPECTED_CLASSES)
            ],
            axis=0,
        )
        q0 = _unit(_canonical_mean_rows(g), "D112 shared reference")
    except D112G0SourceBundleError as exc:
        raise D112G0GeometryDegeneracy(
            "D112 Phase1 ground/reference sphere mean is degenerate"
        ) from exc
    tangent_centres = np.empty_like(centres)
    for domain_index in range(EXPECTED_CELLS):
        for class_index in range(EXPECTED_CLASSES):
            tangent_centres[domain_index, class_index] = _sphere_parallel_transport(
                g[class_index],
                q0,
                _sphere_log(g[class_index], centres[domain_index, class_index]),
            )
    basis = _shared_basis(tangent_centres, q0)
    z = np.einsum("rp,dcp->dcr", basis, tangent_centres, optimize=True)

    labels_array = np.asarray(labels, dtype=object)
    receivers_array = np.asarray(receivers, dtype=object)
    days_array = np.asarray(days, dtype=object)
    sigma_r_cells = np.empty((EXPECTED_CELLS, EXPECTED_CLASSES), dtype=np.float64)
    sigma_amb_cells = np.empty((EXPECTED_CELLS, EXPECTED_CLASSES), dtype=np.float64)
    for domain_index, (receiver, day) in enumerate(cells):
        domain_mask = (receivers_array == receiver) & (days_array == day)
        for class_index, class_id in enumerate(classes):
            local = x[domain_mask & (labels_array == class_id)]
            centre = centres[domain_index, class_index]
            transported = np.asarray(
                [
                    _sphere_parallel_transport(centre, q0, _sphere_log(centre, row))
                    for row in _ordered_rows(local)
                ],
                dtype=np.float64,
            )
            projected = transported @ basis.T
            sigma_r_cells[domain_index, class_index] = _canonical_mean_scalars(
                [float(np.sum(np.square(row)) / SHARED_RANK) for row in projected]
            )
            sigma_amb_cells[domain_index, class_index] = _canonical_mean_scalars(
                [float(np.sum(np.square(row - centre)) / FEATURE_DIM) for row in local]
            )
    sigma0_r = np.asarray(
        [
            EPSILON_VARIANCE_R
            + _canonical_mean_scalars(sigma_r_cells[:, class_index].tolist())
            for class_index in range(EXPECTED_CLASSES)
        ],
        dtype=np.float64,
    )
    sigma0_amb = np.asarray(
        [
            EPSILON_VARIANCE_AMB
            + _canonical_mean_scalars(sigma_amb_cells[:, class_index].tolist())
            for class_index in range(EXPECTED_CLASSES)
        ],
        dtype=np.float64,
    )

    h = np.empty((EXPECTED_CELLS, EXPECTED_CLASSES, SHARED_RANK), dtype=np.float64)
    for class_index in range(EXPECTED_CLASSES):
        donor_indices = np.asarray(
            [index for index in range(EXPECTED_CLASSES) if index != class_index], dtype=np.int64
        )
        weights = 1.0 / sigma0_r[donor_indices]
        for domain_index in range(EXPECTED_CELLS):
            h[domain_index, class_index] = _canonical_weighted_mean(
                z[domain_index, donor_indices, :], weights
            )
    tau_h_r = _canonical_mean_scalars(
        [
            float(np.sum(np.square(h[domain_index, class_index])) / SHARED_RANK)
            for domain_index in range(EXPECTED_CELLS)
            for class_index in range(EXPECTED_CLASSES)
        ]
    )
    if not math.isfinite(tau_h_r) or tau_h_r <= 0.0:
        raise D112G0SourceBundleError("D112 P1 LOO shared-motion power is degenerate")

    v_g_r = np.asarray(
        [
            EPSILON_VARIANCE_R
            + _canonical_mean_scalars(
                [
                    float(
                        np.sum(np.square(z[domain_index, class_index] - h[domain_index, class_index]))
                        / SHARED_RANK
                    )
                    for domain_index in range(EXPECTED_CELLS)
                ]
            )
            for class_index in range(EXPECTED_CLASSES)
        ],
        dtype=np.float64,
    )
    endpoint_mse = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    endpoint_quantization = np.empty(EXPECTED_CLASSES, dtype=np.float64)
    for class_index in range(EXPECTED_CLASSES):
        endpoints = np.empty((EXPECTED_CELLS, FEATURE_DIM), dtype=np.float64)
        for domain_index in range(EXPECTED_CELLS):
            displacement = _sphere_parallel_transport(
                q0, g[class_index], basis.T @ h[domain_index, class_index]
            )
            endpoints[domain_index] = _sphere_exp(g[class_index], displacement)
        endpoint_mse[class_index] = _canonical_mean_scalars(
            [
                float(np.sum(np.square(centres[domain_index, class_index] - endpoints[domain_index])) / FEATURE_DIM)
                for domain_index in range(EXPECTED_CELLS)
            ]
        )
        endpoint_quantization[class_index] = _endpoint_quantization_mse(endpoints)
    v_g_amb = EPSILON_VARIANCE_AMB + endpoint_mse + endpoint_quantization
    arrays = (sigma0_r, sigma0_amb, v_g_r, v_g_amb, endpoint_quantization)
    if any(not np.isfinite(value).all() or np.any(value < 0.0) for value in arrays):
        raise D112G0SourceBundleError("D112 Phase1 variance asset is invalid")
    if any(np.any(value <= 0.0) for value in (sigma0_r, sigma0_amb, v_g_r, v_g_amb)):
        raise D112G0SourceBundleError("D112 Phase1 variance asset lost its fixed positive floor")
    return g, q0, basis, sigma0_r, sigma0_amb, v_g_r, v_g_amb, tau_h_r, endpoint_quantization


def build_d112_g0_source_bundle(
    tap_path: str | Path,
    *,
    receipt_path: str | Path,
    checkpoint_sha256: str,
    expected_tap_sha256: str,
) -> D112Bundle:
    """Construct the non-formal D112 G0 bundle directly from one strict tap.

    No path, source row, source identifier, query, truth or performance datum
    is retained in the returned object.  Geometry/rank chart failure is encoded
    as the specified globally-invalid all-M0 bundle rather than repaired from
    target information.
    """

    try:
        pre_relu, labels, receivers, days, _physical, tap_sha = _load_strict_tap(
            Path(tap_path),
            expected_tap_sha256=expected_tap_sha256,
        )
    except Exception as exc:
        if isinstance(exc, D112G0SourceBundleError):
            raise
        raise D112G0SourceBundleError("D112 strict-tap validation failed") from exc
    tap_receipt_sha256 = _load_tap_receipt(
        Path(receipt_path),
        tap_sha256=tap_sha,
        checkpoint_sha256=checkpoint_sha256,
    )
    x, centres, classes, cells = _centres_from_strict_tap(pre_relu, labels, receivers, days)
    global_valid = True
    try:
        (
            g,
            q0,
            basis,
            sigma0_r,
            sigma0_amb,
            v_g_r,
            v_g_amb,
            tau_h_r,
            endpoint_quantization,
        ) = _derive_assets(x, centres, labels, receivers, days, classes, cells)
    except D112G0GeometryDegeneracy:
        global_valid = False
        (
            g,
            q0,
            basis,
            sigma0_r,
            sigma0_amb,
            v_g_r,
            v_g_amb,
            tau_h_r,
            endpoint_quantization,
        ) = _fallback_assets()

    g_q, g_scale, g_decoded = _quantize_rows(g)
    q0_q, q0_scale, q0_decoded = _quantize_rows(q0[None, :])
    u_q, u_scale, u_decoded = _quantize_rows(basis)
    sigma0_r_q, sigma0_r_scale, sigma0_r_decoded = _quantize_positive(sigma0_r)
    sigma0_amb_q, sigma0_amb_scale, sigma0_amb_decoded = _quantize_positive(sigma0_amb)
    v_g_r_q, v_g_r_scale, v_g_r_decoded = _quantize_positive(v_g_r)
    v_g_amb_q, v_g_amb_scale, v_g_amb_decoded = _quantize_positive(v_g_amb)
    tau_q, tau_scale, tau_decoded = _quantize_positive(np.asarray(tau_h_r))

    g_error_raw = np.linalg.norm(g_decoded - g, axis=1)
    q0_error_raw = np.asarray(float(np.linalg.norm(q0_decoded[0] - q0)), dtype=np.float64)
    u_error_raw = np.asarray(float(np.linalg.norm(u_decoded - basis, ord=2)), dtype=np.float64)
    g_error_q, g_error_scale, g_error_decoded = _quantize_nonnegative(g_error_raw, upper=True)
    q0_error_q, q0_error_scale, q0_error_decoded = _quantize_nonnegative(q0_error_raw, upper=True)
    u_error_q, u_error_scale, u_error_decoded = _quantize_nonnegative(u_error_raw, upper=True)
    endpoint_q, endpoint_scale, endpoint_decoded = _quantize_nonnegative(
        endpoint_quantization, upper=True
    )

    aggregate_sha = _asset_digest(
        classes=classes,
        tap_sha256=tap_sha,
        tap_receipt_sha256=tap_receipt_sha256,
        assets=(
            ("g_q", g_q),
            ("g_scale", g_scale),
            ("q0_q", q0_q),
            ("q0_scale", q0_scale),
            ("U_q", u_q),
            ("U_scale", u_scale),
            ("sigma0_r_q", sigma0_r_q),
            ("sigma0_r_scale", sigma0_r_scale),
            ("sigma0_amb_q", sigma0_amb_q),
            ("sigma0_amb_scale", sigma0_amb_scale),
            ("v_g_r_q", v_g_r_q),
            ("v_g_r_scale", v_g_r_scale),
            ("v_g_amb_q", v_g_amb_q),
            ("v_g_amb_scale", v_g_amb_scale),
            ("tau_h_r_q", tau_q),
            ("tau_h_r_scale", tau_scale),
            ("g_error_q", g_error_q),
            ("g_error_scale", g_error_scale),
            ("q0_error_q", q0_error_q),
            ("q0_error_scale", q0_error_scale),
            ("U_error_q", u_error_q),
            ("U_error_scale", u_error_scale),
            ("endpoint_q", endpoint_q),
            ("endpoint_scale", endpoint_scale),
        ),
    )
    bundle = build_d112_g0_bundle(
        class_registry=classes,
        g=g_decoded,
        q0=q0_decoded[0],
        U=u_decoded,
        sigma0_r=sigma0_r_decoded,
        sigma0_amb=sigma0_amb_decoded,
        v_g_r=v_g_r_decoded,
        v_g_amb=v_g_amb_decoded,
        tau_h_r=float(tau_decoded),
        checkpoint_sha256=checkpoint_sha256,
        source_aggregate_sha256=aggregate_sha,
        global_bundle_valid=global_valid,
        g_quantization_l2_error_bound=g_error_decoded,
        q0_quantization_l2_error_bound=float(q0_error_decoded),
        U_operator_error_upper_bound=float(u_error_decoded),
        endpoint_quantization_chord_mse=endpoint_decoded,
    )
    if (
        bundle.manifest.get("component_state") != G0_COMPONENT_STATE
        or bundle.manifest.get("formal_phase2_eligible") is not False
        or bundle.manifest.get("performance_claim_allowed") is not False
        or bundle.manifest.get("performance_metrics_allowed") is not False
        or bundle.manifest.get("target_access_allowed") is not False
    ):
        raise D112G0SourceBundleError("D112 G0 lifecycle markers drifted during construction")
    return bundle


__all__ = [
    "CANONICAL_REDUCTION_SCHEMA",
    "D112G0SourceBundleError",
    "FEATURE_SCHEMA",
    "ROUNDING_SCHEMA",
    "SCHEMA",
    "build_d112_g0_source_bundle",
]
