"""One-time non-formal Phase1 aggregate for the D111-r2 G0 check.

This module is deliberately smaller than the formal D111 deployment bundle.
It consumes the fixed 588-row D106 strict tap exactly once, writes only a
quantized aggregate, and may only be used by the D111 G0 functional check.
The generated directory contains neither source rows nor source identifiers,
and it is explicitly ineligible for Phase2 performance experiments.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_d111_loo_gat_bundle import D111Bundle, FEATURE_DIM, RANK


SCHEMA = "cvs.d111.r2.g0_source_bundle.v1"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
COMPONENT_STATE = "NONFORMAL_G0_FUNCTIONAL_ONLY"
NPZ_NAME = "d111_r2_g0_source_bundle.npz"
MANIFEST_NAME = "manifest.json"
EXPECTED_ROWS = 588
EXPECTED_CLASSES = 6
EXPECTED_RECEIVERS = 7
EXPECTED_DAYS = 4
EXPECTED_CELLS = EXPECTED_RECEIVERS * EXPECTED_DAYS
TAP_MEMBERS = (
    "pre_relu",
    "z_dom",
    "tx_labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
PAYLOAD_MEMBERS = (
    "schema",
    "feature_schema",
    "class_registry",
    "anchor_q",
    "anchor_scale",
    "basis_q",
    "basis_scale",
    "v_g_q",
    "v_g_scale",
    "v_s_q",
    "v_s_scale",
    "envelope_b_q",
    "envelope_b_scale",
    "epsilon_q",
    "epsilon_scale",
    "anchor_quantization_l2_error_bound",
    "basis_operator_error_upper_bound",
)
ROUNDING_SCHEMA = "numpy_rint_ties_to_even_symmetric_int8_v1"
SUBSPACE_SCHEMA = "equal_class_projection_mean_top3_canonical_sign_v1"
ENVELOPE_SCHEMA = "loo_geometric_median_order_statistic_alpha010_v1"
ALPHA_ENV = 0.10
NUMERIC_EPSILON = 1.0e-12


class D111G0SourceBundleError(ValueError):
    """Raised when the one-time G0-only aggregate is malformed or unsafe."""


def _canonical_bytes(value: Any) -> bytes:
    def _plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): _plain(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [_plain(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise D111G0SourceBundleError(f"{field} must be a lowercase SHA256")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _normalise_rows(value: np.ndarray, field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim < 2 or rows.shape[-1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D111G0SourceBundleError(f"{field} must be finite [...,{FEATURE_DIM}]")
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    if bool(np.any(norms <= NUMERIC_EPSILON)):
        raise D111G0SourceBundleError(f"{field} contains a zero vector")
    return rows / norms


def _canonical_sign(rows: np.ndarray) -> np.ndarray:
    result = np.asarray(rows, dtype=np.float64).copy()
    for index in range(len(result)):
        pivot = int(np.argmax(np.abs(result[index])))
        if result[index, pivot] < 0.0:
            result[index] *= -1.0
    return result


def _canonical_svd_basis(points: np.ndarray, field: str) -> np.ndarray:
    matrix = np.asarray(points, dtype=np.float64)
    if matrix.shape != (EXPECTED_CELLS, FEATURE_DIM) or not np.isfinite(matrix).all():
        raise D111G0SourceBundleError(f"{field} must be finite [{EXPECTED_CELLS},{FEATURE_DIM}]")
    _left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if singular.shape[0] < RANK or singular[RANK - 1] <= max(NUMERIC_EPSILON, singular[0] * 1.0e-10):
        raise D111G0SourceBundleError(f"{field} lacks a defined rank-{RANK} domain basis")
    return _canonical_sign(right[:RANK])


def _geometric_median(points: np.ndarray, *, steps: int = 32) -> np.ndarray:
    rows = np.asarray(points, dtype=np.float64)
    if rows.shape != (EXPECTED_CLASSES - 1, RANK) or not np.isfinite(rows).all():
        raise D111G0SourceBundleError("LOO envelope input must be finite [5,3]")
    ordered = np.asarray(sorted((row.copy() for row in rows), key=lambda row: row.tobytes(order="C")))
    estimate = np.mean(ordered, axis=0)
    for _ in range(steps):
        distance = np.linalg.norm(ordered - estimate[None, :], axis=1)
        if bool(np.any(distance <= NUMERIC_EPSILON)):
            estimate = ordered[int(np.argmin(distance))].copy()
            break
        weight = 1.0 / distance
        estimate = np.sum(weight[:, None] * ordered, axis=0) / np.sum(weight)
    if not np.isfinite(estimate).all():
        raise D111G0SourceBundleError("LOO envelope median became non-finite")
    return estimate


def _quantize_vectors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(value, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[-1] != FEATURE_DIM or not np.isfinite(vectors).all():
        raise D111G0SourceBundleError("vector aggregate layout drift")
    maximum = np.max(np.abs(vectors), axis=1)
    scale32 = np.where(maximum > 0.0, maximum / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if not np.isfinite(scale16).all() or bool(np.any(scale16 <= 0.0)):
        raise D111G0SourceBundleError("vector aggregate scale became invalid")
    code = np.clip(np.rint(vectors / scale32[:, None]), -127, 127).astype(np.int8)
    if bool(np.any(code == -128)):
        raise D111G0SourceBundleError("vector aggregate emitted forbidden -128")
    decoded = code.astype(np.float32) * scale16.astype(np.float32)[:, None]
    return code, scale16, decoded


def _positive_vector(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all() or bool(np.any(vector <= 0.0)):
        raise D111G0SourceBundleError("positive aggregate vector is invalid")
    # Unit-sphere RMS chord variances can validly be much smaller than the
    # smallest representable FP16 per-code scale.  The values remain int8; the
    # scale alone is FP32 so a nonzero physical scatter is not misclassified as
    # an invalid aggregate through storage underflow.
    scale = np.asarray(float(np.max(vector)) / 127.0, dtype=np.float32)
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise D111G0SourceBundleError("positive aggregate vector scale is invalid")
    code = np.clip(np.rint(vector / float(scale)), 1, 127).astype(np.int8)
    decoded = code.astype(np.float32) * np.float32(scale)
    return code, scale, decoded


def _positive_upper_bound(value: float) -> tuple[np.ndarray, np.ndarray, float]:
    if not math.isfinite(value) or value <= 0.0:
        raise D111G0SourceBundleError("positive upper bound is invalid")
    scale = np.asarray(float(value) / 127.0, dtype=np.float32)
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise D111G0SourceBundleError("positive upper-bound scale is invalid")
    decoded = float(np.float32(scale) * np.float32(127.0))
    while decoded < value:
        scale = np.nextafter(scale, np.float32(np.inf), dtype=np.float32)
        if not np.isfinite(scale):
            raise D111G0SourceBundleError("positive upper-bound scale overflow")
        decoded = float(np.float32(scale) * np.float32(127.0))
    return np.asarray(127, dtype=np.int8), scale, decoded


def _conservative_float32(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all() or bool(np.any(array < 0.0)):
        raise D111G0SourceBundleError("quantization error bound is invalid")
    result = np.nextafter(array, np.float32(np.inf), dtype=np.float32)
    if result.shape == ():
        return float(result)
    return result


def _token_array(value: np.ndarray, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"} or array.shape != (EXPECTED_ROWS,):
        raise D111G0SourceBundleError(f"{field} must be text [{EXPECTED_ROWS}]")
    result = tuple(str(item) for item in array.tolist())
    if any(not item.strip() for item in result):
        raise D111G0SourceBundleError(f"{field} contains a blank token")
    return result


def _load_strict_tap(path: Path, *, expected_tap_sha256: str | None) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise D111G0SourceBundleError("strict tap must be one absolute regular non-symlink file")
    raw = path.read_bytes()
    observed = _sha256_bytes(raw)
    if expected_tap_sha256 is not None and observed != _require_sha256(expected_tap_sha256, "expected tap SHA256"):
        raise D111G0SourceBundleError("strict tap SHA256 drift")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if tuple(archive.files) != TAP_MEMBERS:
                raise D111G0SourceBundleError("strict tap member set/order drift")
            arrays = {name: np.asarray(archive[name]) for name in TAP_MEMBERS}
    except D111G0SourceBundleError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise D111G0SourceBundleError("strict tap is not a valid no-pickle NPZ") from exc
    pre_relu = np.asarray(arrays["pre_relu"])
    z_dom = np.asarray(arrays["z_dom"])
    if (
        pre_relu.dtype != np.float32
        or pre_relu.shape != (EXPECTED_ROWS, FEATURE_DIM)
        or not np.isfinite(pre_relu).all()
        or z_dom.dtype != np.float32
        or z_dom.shape != (EXPECTED_ROWS, FEATURE_DIM)
        or not np.isfinite(z_dom).all()
    ):
        raise D111G0SourceBundleError("strict tap feature schema/finite check failed")
    labels = _token_array(arrays["tx_labels"], "tx_labels")
    receivers = _token_array(arrays["receiver_ids"], "receiver_ids")
    days = _token_array(arrays["day_ids"], "day_ids")
    physical = _token_array(arrays["physical_ids"], "physical_ids")
    for field in ("scenario_names", "observation_ids"):
        _token_array(arrays[field], field)
    if len(set(physical)) != EXPECTED_ROWS:
        raise D111G0SourceBundleError("strict tap physical IDs must be unique")
    if len(set(labels)) != EXPECTED_CLASSES:
        raise D111G0SourceBundleError("strict tap must contain exactly six classes")
    cells = {(receiver, day) for receiver, day in zip(receivers, days, strict=True)}
    if len(cells) != EXPECTED_CELLS or len({receiver for receiver, _ in cells}) != EXPECTED_RECEIVERS or len({day for _, day in cells}) != EXPECTED_DAYS:
        raise D111G0SourceBundleError("strict tap must contain the full 7x4 receiver-day grid")
    return pre_relu.copy(), labels, receivers, days, physical, observed


def _aggregate_geometry(
    pre_relu: np.ndarray,
    labels: Sequence[str],
    receivers: Sequence[str],
    days: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Derive the D111 numerical aggregate without retaining any source row."""

    raw = np.maximum(np.asarray(pre_relu, dtype=np.float64), 0.0)
    x = _normalise_rows(raw, "ReLU(pre_relu)")
    classes = tuple(sorted(set(str(label) for label in labels), key=lambda value: value.encode("utf-8")))
    if len(classes) != EXPECTED_CLASSES:
        raise D111G0SourceBundleError("source registry drift")
    cells = tuple(sorted({(str(receiver), str(day)) for receiver, day in zip(receivers, days, strict=True)}, key=lambda value: (value[0].encode("utf-8"), value[1].encode("utf-8"))))
    if len(cells) != EXPECTED_CELLS:
        raise D111G0SourceBundleError("source cell grid drift")
    centres = np.empty((EXPECTED_CELLS, EXPECTED_CLASSES, FEATURE_DIM), dtype=np.float64)
    radii = np.empty((EXPECTED_CELLS, EXPECTED_CLASSES), dtype=np.float64)
    for domain_index, (receiver, day) in enumerate(cells):
        domain_mask = np.asarray(
            [(item_receiver == receiver and item_day == day) for item_receiver, item_day in zip(receivers, days, strict=True)],
            dtype=bool,
        )
        for class_index, class_id in enumerate(classes):
            mask = domain_mask & np.asarray([label == class_id for label in labels], dtype=bool)
            count = int(np.count_nonzero(mask))
            if count < 2:
                raise D111G0SourceBundleError("every receiver-day-class cell needs at least two samples")
            local = x[mask]
            centre = _normalise_rows(np.mean(local, axis=0, keepdims=True), "domain-class mean")[0]
            radius = math.sqrt(float(np.sum(np.square(local - centre[None, :])) / (count * FEATURE_DIM)))
            if not math.isfinite(radius) or radius <= 0.0:
                raise D111G0SourceBundleError("domain-class RMS chord radius must be positive")
            centres[domain_index, class_index] = centre
            radii[domain_index, class_index] = radius
    anchors = _normalise_rows(np.mean(centres, axis=0), "class ground anchors")
    class_bases = np.stack(
        [_canonical_svd_basis(centres[:, class_index] - anchors[class_index], f"class {class_index} SVD") for class_index in range(EXPECTED_CLASSES)],
        axis=0,
    )
    projectors = [np.ascontiguousarray(basis.T @ basis, dtype=np.float64) for basis in class_bases]
    projectors.sort(key=lambda value: value.tobytes(order="C"))
    projection = sum(projectors) / float(EXPECTED_CLASSES)
    eigenvalue, eigenvector = np.linalg.eigh(projection)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = eigenvalue[order]
    basis = _canonical_sign(eigenvector[:, order[:RANK]].T)
    spectral_gap = float(eigenvalue[RANK - 1] - eigenvalue[RANK])
    if not math.isfinite(spectral_gap) or spectral_gap <= NUMERIC_EPSILON:
        raise D111G0SourceBundleError("shared rank-three subspace is not identifiable")
    residual = np.einsum("dcp,rp->dcr", centres - anchors[None, :, :], basis, optimize=True)
    envelope_error: list[float] = []
    for domain_index in range(EXPECTED_CELLS):
        for class_index in range(EXPECTED_CLASSES):
            loo = np.delete(residual[domain_index], class_index, axis=0)
            centre = _geometric_median(loo)
            envelope_error.append(float(np.linalg.norm(residual[domain_index, class_index] - centre)))
    ordered_error = np.sort(np.asarray(envelope_error, dtype=np.float64))
    index = min(len(ordered_error), math.ceil((len(ordered_error) + 1) * (1.0 - ALPHA_ENV))) - 1
    v_g = np.mean(np.square(radii), axis=0)
    v_s = float(np.mean(v_g))
    if not np.isfinite(v_g).all() or bool(np.any(v_g <= 0.0)) or not math.isfinite(v_s) or v_s <= 0.0:
        raise D111G0SourceBundleError("source uncertainty aggregate is invalid")
    return classes, anchors, class_bases, basis, v_g, radii, float(ordered_error[index]), spectral_gap


def _payload(
    *,
    classes: tuple[str, ...],
    anchors: np.ndarray,
    basis: np.ndarray,
    v_g: np.ndarray,
    envelope_error: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | list[float]]]:
    anchor_q, anchor_scale, anchor_decoded = _quantize_vectors(anchors)
    basis_q, basis_scale, basis_decoded = _quantize_vectors(basis)
    anchor_error = _conservative_float32(
        np.linalg.norm(anchor_decoded.astype(np.float64) - anchors, axis=1)
    )
    basis_error = float(
        _conservative_float32(
            np.linalg.norm(basis_decoded.astype(np.float64) - basis, ord=2)
        )
    )
    epsilon_raw = max(float(np.max(anchor_error)) + 2.0 * basis_error, float(np.finfo(np.float32).eps))
    envelope_raw = float(envelope_error + epsilon_raw)
    v_g_q, v_g_scale, _v_g_decoded = _positive_vector(v_g)
    v_s_q, v_s_scale, _v_s_decoded = _positive_upper_bound(float(np.mean(v_g)))
    envelope_b_q, envelope_b_scale, envelope_decoded = _positive_upper_bound(envelope_raw)
    epsilon_q, epsilon_scale, epsilon_decoded = _positive_upper_bound(epsilon_raw)
    payload = {
        "schema": np.asarray(SCHEMA, dtype=np.str_),
        "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
        "class_registry": np.asarray(classes, dtype=np.str_),
        "anchor_q": anchor_q,
        "anchor_scale": anchor_scale,
        "basis_q": basis_q,
        "basis_scale": basis_scale,
        "v_g_q": v_g_q,
        "v_g_scale": v_g_scale,
        "v_s_q": v_s_q,
        "v_s_scale": v_s_scale,
        "envelope_b_q": envelope_b_q,
        "envelope_b_scale": envelope_b_scale,
        "epsilon_q": epsilon_q,
        "epsilon_scale": epsilon_scale,
        "anchor_quantization_l2_error_bound": np.asarray(anchor_error, dtype=np.float32),
        "basis_operator_error_upper_bound": np.asarray(basis_error, dtype=np.float32),
    }
    if tuple(payload) != PAYLOAD_MEMBERS:
        raise AssertionError("D111 G0 payload member order drift")
    return payload, {
        "anchor_quantization_l2_error_bound": [float(value) for value in np.asarray(anchor_error)],
        "basis_operator_error_upper_bound": basis_error,
        "envelope_b_unquantized_upper_bound": envelope_raw,
        "epsilon_unquantized_upper_bound": epsilon_raw,
        "envelope_b_decoded": envelope_decoded,
        "epsilon_decoded": epsilon_decoded,
    }


def _validate_payload(payload: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, float]:
    if tuple(payload) != PAYLOAD_MEMBERS:
        raise D111G0SourceBundleError("G0 bundle payload member order drift")
    if np.asarray(payload["schema"]).shape != () or str(np.asarray(payload["schema"]).item()) != SCHEMA:
        raise D111G0SourceBundleError("G0 bundle payload schema drift")
    if np.asarray(payload["feature_schema"]).shape != () or str(np.asarray(payload["feature_schema"]).item()) != FEATURE_SCHEMA:
        raise D111G0SourceBundleError("G0 bundle feature schema drift")
    registry_raw = np.asarray(payload["class_registry"])
    if registry_raw.dtype.kind not in {"U", "S"} or registry_raw.shape != (EXPECTED_CLASSES,):
        raise D111G0SourceBundleError("G0 bundle registry layout drift")
    classes = tuple(str(item) for item in registry_raw.tolist())
    if len(set(classes)) != EXPECTED_CLASSES or any(not item for item in classes):
        raise D111G0SourceBundleError("G0 bundle registry content drift")
    layouts = {
        "anchor_q": (np.int8, (EXPECTED_CLASSES, FEATURE_DIM)),
        "anchor_scale": (np.float16, (EXPECTED_CLASSES,)),
        "basis_q": (np.int8, (RANK, FEATURE_DIM)),
        "basis_scale": (np.float16, (RANK,)),
        "v_g_q": (np.int8, (EXPECTED_CLASSES,)),
        "v_g_scale": (np.float32, ()),
        "v_s_q": (np.int8, ()),
        "v_s_scale": (np.float32, ()),
        "envelope_b_q": (np.int8, ()),
        "envelope_b_scale": (np.float32, ()),
        "epsilon_q": (np.int8, ()),
        "epsilon_scale": (np.float32, ()),
        "anchor_quantization_l2_error_bound": (np.float32, (EXPECTED_CLASSES,)),
        "basis_operator_error_upper_bound": (np.float32, ()),
    }
    for field, (dtype, shape) in layouts.items():
        value = np.asarray(payload[field])
        if value.dtype != np.dtype(dtype) or value.shape != shape:
            raise D111G0SourceBundleError(f"G0 bundle {field} dtype/shape drift")
        if dtype in {np.float16, np.float32} and not np.isfinite(value).all():
            raise D111G0SourceBundleError(f"G0 bundle {field} is non-finite")
        if dtype is np.int8 and bool(np.any(value == -128)):
            raise D111G0SourceBundleError(f"G0 bundle {field} emitted forbidden -128")
    if bool(np.any(np.asarray(payload["anchor_scale"]) <= 0.0)) or bool(np.any(np.asarray(payload["basis_scale"]) <= 0.0)):
        raise D111G0SourceBundleError("G0 vector scale drift")
    for field in ("v_g_q", "v_s_q", "envelope_b_q", "epsilon_q"):
        if bool(np.any(np.asarray(payload[field]) <= 0)):
            raise D111G0SourceBundleError(f"G0 bundle {field} must be positive")
    for field in ("v_g_scale", "v_s_scale", "envelope_b_scale", "epsilon_scale"):
        if float(np.asarray(payload[field])) <= 0.0:
            raise D111G0SourceBundleError(f"G0 bundle {field} must be positive")
    anchors = np.asarray(payload["anchor_q"], dtype=np.float32) * np.asarray(payload["anchor_scale"], dtype=np.float32)[:, None]
    basis = np.asarray(payload["basis_q"], dtype=np.float32) * np.asarray(payload["basis_scale"], dtype=np.float32)[:, None]
    v_g = np.asarray(payload["v_g_q"], dtype=np.float32) * np.float32(payload["v_g_scale"])
    v_s = float(np.float32(payload["v_s_q"]) * np.float32(payload["v_s_scale"]))
    envelope_b = float(np.float32(payload["envelope_b_q"]) * np.float32(payload["envelope_b_scale"]))
    epsilon = float(np.float32(payload["epsilon_q"]) * np.float32(payload["epsilon_scale"]))
    anchor_error = np.asarray(payload["anchor_quantization_l2_error_bound"], dtype=np.float32)
    basis_error = float(np.asarray(payload["basis_operator_error_upper_bound"], dtype=np.float32))
    if (
        bool(np.any(np.linalg.norm(anchors, axis=1) <= NUMERIC_EPSILON))
        or bool(np.any(np.linalg.norm(basis, axis=1) <= NUMERIC_EPSILON))
        or not np.isfinite(v_g).all()
        or bool(np.any(v_g <= 0.0))
        or not all(math.isfinite(value) and value > 0.0 for value in (v_s, envelope_b, epsilon))
        or bool(np.any(anchor_error < 0.0))
        or basis_error < 0.0
    ):
        raise D111G0SourceBundleError("G0 bundle numerical receipt drift")
    return classes, anchors, basis, v_g, v_s, envelope_b, epsilon, anchor_error, basis_error


def build_d111_g0_source_bundle(
    tap_path: str | Path,
    output_dir: str | Path,
    *,
    expected_tap_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one immutable G0-only aggregate directly from a D106 strict tap."""

    source = Path(tap_path)
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"D111 G0 bundle output already exists: {root}")
    if not root.is_absolute():
        raise D111G0SourceBundleError("D111 G0 bundle output must be absolute")
    pre_relu, labels, receivers, days, _physical, tap_sha = _load_strict_tap(
        source, expected_tap_sha256=expected_tap_sha256
    )
    (
        classes,
        anchors,
        _class_bases,
        basis,
        v_g,
        _radii,
        envelope_error,
        spectral_gap,
    ) = _aggregate_geometry(pre_relu, labels, receivers, days)
    payload, error_receipt = _payload(
        classes=classes,
        anchors=anchors,
        basis=basis,
        v_g=v_g,
        envelope_error=envelope_error,
    )
    _validate_payload(payload)
    root.mkdir(parents=True)
    npz_path = root / NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    payload_sha = _sha256_file(npz_path)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "component_state": COMPONENT_STATE,
        "formal_phase2_eligible": False,
        "performance_claim_allowed": False,
        "performance_metrics_allowed": False,
        "target_access": False,
        "target_access_allowed": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "rank": RANK,
        "class_registry_sha256": _sha256_bytes(_canonical_bytes(list(classes))),
        "source_tap_sha256": tap_sha,
        "member_allowlist": [NPZ_NAME],
        "payload_member_allowlist": list(PAYLOAD_MEMBERS),
        "rounding_schema": ROUNDING_SCHEMA,
        "subspace_schema": SUBSPACE_SCHEMA,
        "envelope_schema": ENVELOPE_SCHEMA,
        "alpha_env": ALPHA_ENV,
        "spectral_gap": spectral_gap,
        "resource_receipt": {
            "numeric_payload_bytes": int(sum(np.asarray(value).nbytes for key, value in payload.items() if key not in {"schema", "feature_schema", "class_registry"})),
            "persistent_source_rows": 0,
            "persistent_source_ids": 0,
            "persistent_query_state_bytes": 0,
        },
        **error_receipt,
        "payload_sha256": payload_sha,
    }
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_bytes(_canonical_bytes(manifest))
    # Validate only the just-written aggregate.  No source row/ID/count is ever
    # carried into this output or the later loader.
    load_d111_g0_source_bundle(root)
    return {
        "root": str(root),
        "source_tap_sha256": tap_sha,
        "payload_sha256": payload_sha,
        "class_registry": classes,
        "component_state": COMPONENT_STATE,
    }


def load_d111_g0_source_bundle(root: str | Path) -> D111Bundle:
    """Return the existing D111 runtime object for G0-only support scoring."""

    directory = Path(root)
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        raise D111G0SourceBundleError("D111 G0 bundle directory must be absolute and regular")
    if {item.name for item in directory.iterdir()} != {NPZ_NAME, MANIFEST_NAME}:
        raise D111G0SourceBundleError("D111 G0 bundle directory member drift")
    manifest_path = directory / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise D111G0SourceBundleError("D111 G0 manifest is invalid") from exc
    required = {
        "schema": SCHEMA,
        "component_state": COMPONENT_STATE,
        "formal_phase2_eligible": False,
        "performance_claim_allowed": False,
        "performance_metrics_allowed": False,
        "target_access": False,
        "target_access_allowed": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "rank": RANK,
        "member_allowlist": [NPZ_NAME],
        "payload_member_allowlist": list(PAYLOAD_MEMBERS),
        "rounding_schema": ROUNDING_SCHEMA,
        "subspace_schema": SUBSPACE_SCHEMA,
        "envelope_schema": ENVELOPE_SCHEMA,
        "alpha_env": ALPHA_ENV,
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required.items()):
        raise D111G0SourceBundleError("D111 G0 manifest lifecycle/schema drift")
    payload_path = directory / NPZ_NAME
    if manifest.get("payload_sha256") != _sha256_file(payload_path):
        raise D111G0SourceBundleError("D111 G0 payload SHA256 drift")
    _require_sha256(str(manifest.get("source_tap_sha256", "")), "source tap SHA256")
    _require_sha256(str(manifest.get("class_registry_sha256", "")), "class registry SHA256")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise D111G0SourceBundleError("D111 G0 payload is invalid") from exc
    classes, anchors, basis, v_g, v_s, envelope_b, epsilon, anchor_error, basis_error = _validate_payload(payload)
    if manifest["class_registry_sha256"] != _sha256_bytes(_canonical_bytes(list(classes))):
        raise D111G0SourceBundleError("D111 G0 registry receipt drift")
    if (
        not math.isfinite(float(manifest.get("spectral_gap", math.nan)))
        or float(manifest["spectral_gap"]) <= NUMERIC_EPSILON
        or not isinstance(manifest.get("resource_receipt"), dict)
        or int(manifest["resource_receipt"].get("persistent_source_rows", -1)) != 0
        or int(manifest["resource_receipt"].get("persistent_source_ids", -1)) != 0
        or int(manifest["resource_receipt"].get("persistent_query_state_bytes", -1)) != 0
    ):
        raise D111G0SourceBundleError("D111 G0 manifest numerical/resource drift")
    if float(manifest.get("envelope_b_decoded", -math.inf)) > envelope_b + 1.0e-7 or float(manifest.get("epsilon_decoded", -math.inf)) > epsilon + 1.0e-7:
        raise D111G0SourceBundleError("D111 G0 conservative bound receipt drift")
    runtime_manifest = MappingProxyType(
        {
            **manifest,
            "effective_bundle_state": COMPONENT_STATE,
            "effective_formal_phase2_eligible": False,
            "g0_functional_only": True,
        }
    )
    return D111Bundle(
        class_registry=classes,
        anchors=_readonly(anchors, np.float32),
        basis=_readonly(basis, np.float32),
        v_g=_readonly(v_g, np.float32),
        v_s=v_s,
        envelope_b=envelope_b,
        epsilon=epsilon,
        anchor_quantization_l2_error_bound=_readonly(anchor_error, np.float32),
        basis_operator_error_upper_bound=basis_error,
        manifest=runtime_manifest,
    )


__all__ = [
    "COMPONENT_STATE",
    "D111G0SourceBundleError",
    "EXPECTED_CELLS",
    "EXPECTED_CLASSES",
    "EXPECTED_ROWS",
    "FEATURE_SCHEMA",
    "MANIFEST_NAME",
    "NPZ_NAME",
    "PAYLOAD_MEMBERS",
    "SCHEMA",
    "TAP_MEMBERS",
    "build_d111_g0_source_bundle",
    "load_d111_g0_source_bundle",
]
