"""Strict joint-sealed Phase1 aggregate asset for D111 LOO-GAT.

The builder accepts aggregate domain/class centres and aggregate class bases;
it has no sample, physical-ID, source-path, query, or truth interface.  The
runtime loader exposes only the small decoded aggregate needed by the later
LOO-GAT state builder.  Scoring and experiment execution intentionally do not
belong in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.d111.loo_gat.joint_sealed_asset.v1"
FEATURE_SCHEMA = "ADV3B02:z_id:unit_l2:160:v1"
FEATURE_DIM = 160
RANK = 3
NPZ_NAME = "d111_loo_gat_asset.npz"
MANIFEST_NAME = "manifest.json"
MANIFEST_SHA_NAME = "manifest.sha256"
STATE = "PENDING_OUTER_JOINT_SEAL"
SOURCE_AGGREGATE_SCHEMA = "cvs.d111.loo_gat.formal_phase1_aggregate.v1"
SOURCE_AGGREGATE_STATE = "FORMAL_PHASE1_AGGREGATE_OUTER_JOINT_SEALED"
OUTER_SEAL_SCHEMA = "cvs.d111.loo_gat.outer_ed25519_seal.v1"
OUTER_SEALED_STATE = "FORMAL_D111_OUTER_JOINT_SEALED"
OUTER_SEAL_NAME = "outer_seal.json"
OUTER_SEAL_SHA_NAME = "outer_seal.sha256"
ROUNDING_SCHEMA = "numpy_rint_ties_to_even_symmetric_int8_v1"
SUBSPACE_SCHEMA = "equal_class_projection_mean_top3_canonical_sign_v1"
ENVELOPE_SCHEMA = "aggregate_domain_class_loo_order_statistic_alpha010_v1"
ALPHA_ENV = 0.10
MAX_U_OPERATOR_QUANTIZATION_ERROR = 0.05

ALLOWED_NPZ_MEMBERS = {
    "schema",
    "feature_schema",
    "class_registry",
    "g_q",
    "g_scale",
    "u_q",
    "u_scale",
    "v_g_q",
    "v_g_scale",
    "v_s_q",
    "v_s_scale",
    "b_q",
    "b_scale",
    "epsilon_q",
    "epsilon_scale",
}

_MANIFEST_FIELDS = {
    "schema",
    "feature_schema",
    "feature_dim",
    "rank",
    "class_count",
    "source_domain_count",
    "class_registry_sha256",
    "source_aggregate_sha256",
    "source_aggregate_manifest_sha256",
    "checkpoint_sha256",
    "method_lock_sha256",
    "generation_code_sha256",
    "generation_config_sha256",
    "outer_bundle_signature_required",
    "component_state",
    "formal_phase2_eligible",
    "member_allowlist",
    "npz_member_allowlist",
    "rounding_schema",
    "subspace_schema",
    "envelope_schema",
    "alpha_env",
    "source_basis_quantization_error_bound",
    "source_center_quantization_error_bound",
    "projection_quantization_error_bound",
    "spectral_gap",
    "spectral_gap_required_minimum",
    "u_operator_quantization_error",
    "u_orthogonality_error",
    "envelope_b_unquantized_upper_bound",
    "epsilon_unquantized_upper_bound",
    "resource_receipt",
    "protocol_receipt",
    "component_npz_sha256",
    "content_root_sha256",
}

_SOURCE_AGGREGATE_MANIFEST_FIELDS = {
    "schema",
    "component_state",
    "formal_phase2_eligible",
    "checkpoint_sha256",
    "signer_id",
    "signature_ed25519_hex",
    "aggregate_content_sha256",
    "aggregate_only",
    "generation_stage",
}

# Trust roots are release-controlled public keys, never caller-provided secrets.
# The production release must enroll the authority keys before any formal asset
# can load. Tests replace these empty immutable maps with test-only public keys.
_TRUSTED_SOURCE_ED25519_PUBLIC_KEYS: Mapping[str, str] = MappingProxyType({})
_TRUSTED_OUTER_ED25519_PUBLIC_KEYS: Mapping[str, str] = MappingProxyType({})

_ED_Q = 2**255 - 19
_ED_L = 2**252 + 27742317777372353535851937790883648493
_ED_D = (-121665 * pow(121666, _ED_Q - 2, _ED_Q)) % _ED_Q
_ED_I = pow(2, (_ED_Q - 1) // 4, _ED_Q)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_Q - 2, _ED_Q) % _ED_Q
    x = pow(xx, (_ED_Q + 3) // 8, _ED_Q)
    if (x * x - xx) % _ED_Q != 0:
        x = x * _ED_I % _ED_Q
    if x & 1:
        x = _ED_Q - x
    return x


_ED_BY = 4 * pow(5, _ED_Q - 2, _ED_Q) % _ED_Q
_ED_B = (_ed_xrecover(_ED_BY), _ED_BY)


def _ed_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    denominator = _ED_D * x1 * x2 * y1 * y2
    x3 = (x1 * y2 + x2 * y1) * pow(1 + denominator, _ED_Q - 2, _ED_Q) % _ED_Q
    y3 = (y1 * y2 + x1 * x2) * pow(1 - denominator, _ED_Q - 2, _ED_Q) % _ED_Q
    return x3, y3


def _ed_scalarmult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    def add(
        p: tuple[int, int, int, int], q: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x1, y1, z1, t1 = p
        x2, y2, z2, t2 = q
        a = (y1 - x1) * (y2 - x2) % _ED_Q
        b = (y1 + x1) * (y2 + x2) % _ED_Q
        c = 2 * _ED_D * t1 * t2 % _ED_Q
        d = 2 * z1 * z2 % _ED_Q
        e, f, g, h = b - a, d - c, d + c, b + a
        return e * f % _ED_Q, g * h % _ED_Q, f * g % _ED_Q, e * h % _ED_Q

    def double(p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, z, _t = p
        a = x * x % _ED_Q
        b = y * y % _ED_Q
        c = 2 * z * z % _ED_Q
        d = -a
        e = (x + y) * (x + y) - a - b
        g, f, h = d + b, d + b - c, d - b
        return e * f % _ED_Q, g * h % _ED_Q, f * g % _ED_Q, e * h % _ED_Q

    x, y = point
    result = (0, 1, 1, 0)
    addend = (x, y, 1, x * y % _ED_Q)
    value = int(scalar)
    while value:
        if value & 1:
            result = add(result, addend)
        addend = double(addend)
        value >>= 1
    x, y, z, _t = result
    inverse_z = pow(z, _ED_Q - 2, _ED_Q)
    return x * inverse_z % _ED_Q, y * inverse_z % _ED_Q


def _ed_encodepoint(point: tuple[int, int]) -> bytes:
    x, y = point
    return int(y | ((x & 1) << 255)).to_bytes(32, "little")


def _ed_decodepoint(encoded: bytes) -> tuple[int, int]:
    if len(encoded) != 32:
        raise D111BundleError("Ed25519 public point must be 32 bytes")
    value = int.from_bytes(encoded, "little")
    y = value & ((1 << 255) - 1)
    if y >= _ED_Q:
        raise D111BundleError("Ed25519 public point is non-canonical")
    x = _ed_xrecover(y)
    if (x & 1) != (value >> 255):
        x = _ED_Q - x
    if (-x * x + y * y - 1 - _ED_D * x * x * y * y) % _ED_Q != 0:
        raise D111BundleError("Ed25519 public point is off-curve")
    return x, y


def _verify_ed25519(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    try:
        public_key = bytes.fromhex(public_key_hex)
        signature = bytes.fromhex(signature_hex)
        if len(public_key) != 32 or len(signature) != 64:
            return False
        a_point = _ed_decodepoint(public_key)
        r_encoded, s_encoded = signature[:32], signature[32:]
        r_point = _ed_decodepoint(r_encoded)
        scalar = int.from_bytes(s_encoded, "little")
        if scalar >= _ED_L:
            return False
        challenge = int.from_bytes(
            hashlib.sha512(r_encoded + public_key + message).digest(), "little"
        ) % _ED_L
        return _ed_encodepoint(_ed_scalarmult(_ED_B, scalar)) == _ed_encodepoint(
            _ed_add(r_point, _ed_scalarmult(a_point, challenge))
        )
    except (D111BundleError, ValueError):
        return False


class D111BundleError(ValueError):
    """Raised when a D111 aggregate asset does not fail closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _aggregate_content_sha256(
    *,
    core: np.ndarray,
    class_bases: np.ndarray,
    domain_class_centers: np.ndarray,
    class_radii: np.ndarray,
    class_registry: Sequence[str],
    source_basis_quantization_error_bound: float,
    source_center_quantization_error_bound: float,
) -> str:
    """Bind the exact decoded aggregate arrays, never sample-level inputs."""

    return _canonical_sha256(
        {
            "schema": SOURCE_AGGREGATE_SCHEMA,
            "core": _array_receipt(core),
            "class_bases": _array_receipt(class_bases),
            "domain_class_centers": _array_receipt(domain_class_centers),
            "class_radii": _array_receipt(class_radii),
            "class_registry": list(_registry(class_registry)),
            "source_basis_quantization_error_bound": float(
                source_basis_quantization_error_bound
            ),
            "source_center_quantization_error_bound": float(
                source_center_quantization_error_bound
            ),
        }
    )


def _validate_source_aggregate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_content_sha256: str,
    expected_checkpoint_sha256: str,
) -> str:
    if set(manifest) != _SOURCE_AGGREGATE_MANIFEST_FIELDS:
        raise D111BundleError("formal source aggregate manifest field set mismatch")
    if (
        manifest.get("schema") != SOURCE_AGGREGATE_SCHEMA
        or manifest.get("component_state") != SOURCE_AGGREGATE_STATE
        or manifest.get("formal_phase2_eligible") is not True
        or manifest.get("aggregate_only") is not True
        or manifest.get("generation_stage") != "phase1_offline_before_target_access"
    ):
        raise D111BundleError("pending or informal source aggregate cannot become formal D111 state")
    checkpoint = _validate_sha(str(manifest.get("checkpoint_sha256", "")), "source checkpoint")
    signer_id = str(manifest.get("signer_id", ""))
    if not signer_id:
        raise D111BundleError("formal source aggregate signer is empty")
    signature = str(manifest.get("signature_ed25519_hex", ""))
    if len(signature) != 128:
        raise D111BundleError("source aggregate Ed25519 signature encoding is invalid")
    content = _validate_sha(
        str(manifest.get("aggregate_content_sha256", "")),
        "source aggregate content",
    )
    if checkpoint != expected_checkpoint_sha256 or content != expected_content_sha256:
        raise D111BundleError("formal source aggregate manifest binding drift")
    public_key = _TRUSTED_SOURCE_ED25519_PUBLIC_KEYS.get(signer_id)
    if public_key is None:
        raise D111BundleError("formal source aggregate signer is not in the trusted keyring")
    signed = {key: value for key, value in manifest.items() if key != "signature_ed25519_hex"}
    if not _verify_ed25519(public_key, _canonical_bytes(signed), signature):
        raise D111BundleError("formal source aggregate signature verification failed")
    return _canonical_sha256(dict(manifest))


def _freeze_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_sha(value: str, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise D111BundleError(f"{field} must be a lowercase SHA256")
    return text


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    registry = tuple(str(value) for value in values)
    if len(registry) < 4 or len(set(registry)) != len(registry) or any(not value for value in registry):
        raise D111BundleError("class registry must contain at least four unique handles")
    return registry


def _normalize_rows(value: np.ndarray, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != FEATURE_DIM or not np.isfinite(array).all():
        raise D111BundleError(f"{field} must be finite [...,{FEATURE_DIM}]")
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if bool(np.any(norms <= 1.0e-12)):
        raise D111BundleError(f"{field} contains a zero aggregate vector")
    return array / norms


def _canonical_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64).copy()
    for index in range(len(rows)):
        pivot = int(np.argmax(np.abs(rows[index])))
        if rows[index, pivot] < 0.0:
            rows[index] *= -1.0
    return rows


def _permutation_invariant_projection_mean(class_bases: np.ndarray) -> np.ndarray:
    """Accumulate the class-projector multiset in a canonical byte order."""

    projectors = [
        np.ascontiguousarray(basis.T @ basis, dtype=np.float64)
        for basis in np.asarray(class_bases, dtype=np.float64)
    ]
    projectors.sort(key=lambda value: value.tobytes(order="C"))
    total = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)
    for projector in projectors:
        total += projector
    return total / float(len(projectors))


def _quantize_vectors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.asarray(value, dtype=np.float32)
    if vectors.ndim < 2 or not np.isfinite(vectors).all():
        raise D111BundleError("vector aggregate must be finite and at least rank two")
    maximum = np.max(np.abs(vectors), axis=-1)
    scale32 = np.where(maximum > 0.0, maximum / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if not np.isfinite(scale16).all() or bool(np.any(scale16 <= 0.0)):
        raise D111BundleError("vector FP16 scale is invalid")
    q = np.clip(np.rint(vectors / scale32[..., None]), -127, 127).astype(np.int8)
    decoded = q.astype(np.float32) * scale16.astype(np.float32)[..., None]
    if bool(np.any(q == -128)):
        raise D111BundleError("forbidden -128 emitted")
    return q, scale16, decoded


def _int8_vector_l2_error_upper_bounds_from_scales(scales: np.ndarray) -> np.ndarray:
    """Return a conservative decoded-INT8 vector error bound per payload row.

    Vector payloads are quantized with a float32 step and decoded with its
    persisted float16 representation.  The bound therefore includes both the
    half-step integer rounding term and the maximum representable float16
    scale-rounding term.  It is computable from the immutable payload alone;
    no source vector or sidecar is needed at Phase2 runtime.
    """

    scale16 = np.asarray(scales, dtype=np.float16)
    if scale16.ndim != 1 or not np.isfinite(scale16).all() or bool(np.any(scale16 <= 0.0)):
        raise D111BundleError("INT8 vector scales must be finite positive [row]")
    decoded_scale = scale16.astype(np.float64)
    previous = np.nextafter(scale16, np.float16(0.0)).astype(np.float64)
    following = np.nextafter(scale16, np.float16(np.inf)).astype(np.float64)
    scale_rounding_bound = 0.5 * np.maximum(
        decoded_scale - previous,
        following - decoded_scale,
    )
    if not np.isfinite(scale_rounding_bound).all() or bool(np.any(scale_rounding_bound < 0.0)):
        raise D111BundleError("INT8 vector scale-rounding bound is invalid")
    unquantized_scale_upper_bound = decoded_scale + scale_rounding_bound
    per_coordinate_bound = (
        0.5 * unquantized_scale_upper_bound + 127.0 * scale_rounding_bound
    )
    bound = np.sqrt(float(FEATURE_DIM)) * per_coordinate_bound
    if not np.isfinite(bound).all() or bool(np.any(bound < 0.0)):
        raise D111BundleError("INT8 vector L2 error upper bound is invalid")
    return _freeze_array(np.asarray(bound, dtype=np.float32))


def _quantize_positive_vector(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or bool(np.any(vector <= 0.0)):
        raise D111BundleError("positive aggregate vector is invalid")
    maximum = float(np.max(vector))
    scale16 = np.asarray(maximum / 127.0, dtype=np.float16)
    if not np.isfinite(scale16) or float(scale16) <= 0.0:
        raise D111BundleError("positive vector FP16 scale is invalid")
    q = np.clip(np.rint(vector / np.float32(scale16)), 1, 127).astype(np.int8)
    decoded = q.astype(np.float32) * np.float32(scale16)
    return q, scale16, decoded


def _quantize_positive_scalar(value: float) -> tuple[np.ndarray, np.ndarray, float]:
    if not math.isfinite(value) or value <= 0.0:
        raise D111BundleError("positive aggregate scalar is invalid")
    scale = np.asarray(np.float32(value / 127.0), dtype=np.float16)
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise D111BundleError("positive scalar FP16 scale is invalid")
    q = np.asarray(127, dtype=np.int8)
    return q, scale, float(np.float32(scale) * np.float32(127.0))


def _quantize_positive_upper_bound(value: float) -> tuple[np.ndarray, np.ndarray, float]:
    q, scale, decoded = _quantize_positive_scalar(value)
    while decoded < value:
        scale = np.nextafter(scale, np.float16(np.inf), dtype=np.float16)
        if not np.isfinite(scale):
            raise D111BundleError("conservative FP16 upper-bound scale overflow")
        decoded = float(np.float32(scale) * np.float32(127.0))
    return q, scale, decoded


def _geometric_median(points: np.ndarray, *, steps: int = 64) -> np.ndarray:
    rows = np.asarray(points, dtype=np.float64)
    if rows.ndim != 2 or len(rows) < 3 or not np.isfinite(rows).all():
        raise D111BundleError("aggregate geometric median input is invalid")
    rows = np.asarray(
        sorted((row.copy() for row in rows), key=lambda row: row.tobytes(order="C"))
    )
    estimate = np.mean(rows, axis=0)
    for _ in range(steps):
        distance = np.linalg.norm(rows - estimate[None, :], axis=1)
        if bool(np.any(distance <= 1.0e-15)):
            estimate = rows[int(np.argmin(distance))].copy()
            break
        weight = 1.0 / distance
        estimate = np.sum(weight[:, None] * rows, axis=0) / np.sum(weight)
    if not np.isfinite(estimate).all():
        raise D111BundleError("aggregate geometric median is not finite")
    return estimate


def _resource_receipt(payload: Mapping[str, np.ndarray], class_count: int, domain_count: int) -> dict[str, int]:
    numeric = sum(
        int(np.asarray(value).nbytes)
        for key, value in payload.items()
        if key not in {"schema", "feature_schema", "class_registry"}
    )
    return {
        "class_count": class_count,
        "domain_count": domain_count,
        "feature_dim": FEATURE_DIM,
        "rank": RANK,
        "numeric_payload_bytes": numeric,
        "fit_projection_accumulate_macs_upper_bound": class_count * RANK * FEATURE_DIM * FEATURE_DIM,
        "fit_eigendecomposition_dimension": FEATURE_DIM,
        "envelope_projection_macs_upper_bound": domain_count * class_count * RANK * FEATURE_DIM,
        "phase2_decode_macs": class_count * FEATURE_DIM + RANK * FEATURE_DIM,
        "aggregate_input_array_bytes": 4
        * (
            class_count * FEATURE_DIM
            + class_count * RANK * FEATURE_DIM
            + domain_count * class_count * FEATURE_DIM
            + domain_count * class_count
        ),
        "temporary_projection_eigendecomposition_peak_bytes_upper_bound": 8
        * (3 * FEATURE_DIM * FEATURE_DIM + FEATURE_DIM),
        "persistent_fp32_source_bank_bytes": 0,
        "sample_or_query_state_bytes": 0,
    }


def _content_root(manifest: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_root_sha256"}
    )


def _validate_npz_payload(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(payload) != ALLOWED_NPZ_MEMBERS:
        raise D111BundleError("D111 NPZ member allowlist mismatch")
    schema = np.asarray(payload["schema"])
    feature = np.asarray(payload["feature_schema"])
    if schema.shape != () or str(schema.item()) != SCHEMA:
        raise D111BundleError("D111 NPZ schema mismatch")
    if feature.shape != () or str(feature.item()) != FEATURE_SCHEMA:
        raise D111BundleError("D111 feature schema mismatch")
    raw_registry = np.asarray(payload["class_registry"])
    if raw_registry.ndim != 1 or raw_registry.dtype.kind not in {"U", "S"}:
        raise D111BundleError("class registry encoding is invalid")
    classes = _registry(raw_registry.astype(str).tolist())
    c = len(classes)
    expected = {
        "g_q": (np.int8, (c, FEATURE_DIM)),
        "g_scale": (np.float16, (c,)),
        "u_q": (np.int8, (RANK, FEATURE_DIM)),
        "u_scale": (np.float16, (RANK,)),
        "v_g_q": (np.int8, (c,)),
        "v_g_scale": (np.float16, ()),
        "v_s_q": (np.int8, ()),
        "v_s_scale": (np.float16, ()),
        "b_q": (np.int8, ()),
        "b_scale": (np.float16, ()),
        "epsilon_q": (np.int8, ()),
        "epsilon_scale": (np.float16, ()),
    }
    for field, (dtype, shape) in expected.items():
        array = np.asarray(payload[field])
        if array.dtype != dtype or array.shape != shape:
            raise D111BundleError(f"{field} dtype/shape mismatch")
        if dtype == np.int8 and bool(np.any(array == -128)):
            raise D111BundleError(f"{field} contains forbidden -128")
        if dtype == np.int8 and field not in {"g_q", "u_q"} and bool(np.any(array < 0)):
            raise D111BundleError(f"{field} violates non-negative INT8 rules")
        if dtype == np.float16 and (not np.isfinite(array).all() or bool(np.any(array <= 0.0))):
            raise D111BundleError(f"{field} has invalid FP16 scale")
    anchors = np.asarray(payload["g_q"], dtype=np.float32) * np.asarray(
        payload["g_scale"], dtype=np.float32
    )[:, None]
    basis = np.asarray(payload["u_q"], dtype=np.float32) * np.asarray(
        payload["u_scale"], dtype=np.float32
    )[:, None]
    if bool(np.any(np.linalg.norm(anchors, axis=1) <= 1.0e-8)):
        raise D111BundleError("decoded D111 anchor is degenerate")
    if bool(np.any(np.linalg.norm(basis, axis=1) <= 1.0e-8)):
        raise D111BundleError("decoded D111 basis is degenerate")
    return {"classes": classes, "anchors": anchors, "basis": basis}


@dataclass(frozen=True)
class D111Bundle:
    class_registry: tuple[str, ...]
    anchors: np.ndarray
    basis: np.ndarray
    v_g: np.ndarray
    v_s: float
    envelope_b: float
    epsilon: float
    manifest: Mapping[str, Any]
    # Runtime-only receipt values derived from the sealed aggregate payload.
    # `None` is retained for old in-memory test fixtures; formal loaders always
    # materialize the auditable per-anchor bound.
    anchor_quantization_l2_error_bound: np.ndarray | None = None
    basis_operator_error_upper_bound: float = 0.0


def build_d111_bundle_from_aggregate(
    *,
    core: np.ndarray,
    class_bases: np.ndarray,
    domain_class_centers: np.ndarray,
    class_radii: np.ndarray,
    class_registry: Sequence[str],
    source_basis_quantization_error_bound: float,
    source_center_quantization_error_bound: float,
    source_aggregate_manifest: Mapping[str, Any],
    checkpoint_sha256: str,
    method_lock_sha256: str,
    generation_code_sha256: str,
    generation_config_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a pending component; class_radii are per-coordinate RMS values."""

    hashes = {
        field: _validate_sha(value, field)
        for field, value in {
            "checkpoint_sha256": checkpoint_sha256,
            "method_lock_sha256": method_lock_sha256,
            "generation_code_sha256": generation_code_sha256,
            "generation_config_sha256": generation_config_sha256,
        }.items()
    }
    classes = _registry(class_registry)
    c = len(classes)
    core32 = np.ascontiguousarray(np.asarray(core, dtype=np.float32))
    bases32 = np.ascontiguousarray(np.asarray(class_bases, dtype=np.float32))
    centers32 = np.ascontiguousarray(np.asarray(domain_class_centers, dtype=np.float32))
    radii32 = np.ascontiguousarray(np.asarray(class_radii, dtype=np.float32))
    anchors = _normalize_rows(core32, "core")
    if anchors.shape != (c, FEATURE_DIM):
        raise D111BundleError("core must match [class,160]")
    bases = np.asarray(bases32, dtype=np.float64)
    if bases.shape != (c, RANK, FEATURE_DIM) or not np.isfinite(bases).all():
        raise D111BundleError("class_bases must be finite [class,3,160]")
    for class_index in range(c):
        gram = bases[class_index] @ bases[class_index].T
        if float(np.max(np.abs(gram - np.eye(RANK)))) > 5.0e-3:
            raise D111BundleError("class aggregate basis is not orthonormal")
    centers = _normalize_rows(centers32, "domain_class_centers")
    if centers.ndim != 3 or centers.shape[1:] != (c, FEATURE_DIM):
        raise D111BundleError("domain_class_centers must be [domain,class,160]")
    radii = np.asarray(radii32, dtype=np.float64)
    if radii.shape != centers.shape[:2] or not np.isfinite(radii).all() or bool(np.any(radii <= 0.0)):
        raise D111BundleError("class_radii must be positive aggregate [domain,class]")
    basis_error = float(source_basis_quantization_error_bound)
    center_error = float(source_center_quantization_error_bound)
    if not math.isfinite(basis_error) or basis_error < 0.0:
        raise D111BundleError("source basis quantization error bound is invalid")
    if not math.isfinite(center_error) or center_error < 0.0:
        raise D111BundleError("source center quantization error bound is invalid")
    aggregate_content_sha = _aggregate_content_sha256(
        core=core32,
        class_bases=bases32,
        domain_class_centers=centers32,
        class_radii=radii32,
        class_registry=classes,
        source_basis_quantization_error_bound=basis_error,
        source_center_quantization_error_bound=center_error,
    )
    source_manifest_sha = _validate_source_aggregate_manifest(
        source_aggregate_manifest,
        expected_content_sha256=aggregate_content_sha,
        expected_checkpoint_sha256=hashes["checkpoint_sha256"],
    )
    hashes["source_aggregate_sha256"] = aggregate_content_sha
    hashes["source_aggregate_manifest_sha256"] = source_manifest_sha

    projection = _permutation_invariant_projection_mean(bases)
    eigenvalue, eigenvector = np.linalg.eigh(projection)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = eigenvalue[order]
    eigenvector = eigenvector[:, order]
    u = _canonical_rows(eigenvector[:, :RANK].T)
    projection_error = 2.0 * basis_error + basis_error * basis_error
    spectral_gap = float(eigenvalue[RANK - 1] - eigenvalue[RANK])
    if spectral_gap <= 2.0 * projection_error:
        raise D111BundleError("shared subspace spectral-gap certificate failed")

    g_q, g_scale, g_hat = _quantize_vectors(anchors)
    u_q, u_scale, u_hat = _quantize_vectors(u)
    u_operator_error = float(np.linalg.norm(u_hat.astype(np.float64) - u, ord=2))
    if u_operator_error > MAX_U_OPERATOR_QUANTIZATION_ERROR:
        raise D111BundleError("shared basis INT8 operator error exceeds fixed limit")
    u_orthogonality_error = float(
        np.linalg.norm(u_hat.astype(np.float64) @ u_hat.astype(np.float64).T - np.eye(RANK), ord=2)
    )
    mathematical_orth_bound = 2.0 * u_operator_error + u_operator_error**2 + 1.0e-7
    if u_orthogonality_error > mathematical_orth_bound:
        raise D111BundleError("shared basis orthogonality certificate failed")

    z = np.einsum("dcp,rp->dcr", centers - anchors[None, :, :], u, optimize=True)
    envelope_error: list[float] = []
    for domain_index in range(len(centers)):
        for class_index in range(c):
            other = np.delete(z[domain_index], class_index, axis=0)
            centre = _geometric_median(other)
            envelope_error.append(float(np.linalg.norm(z[domain_index, class_index] - centre)))
    ordered_error = np.sort(np.asarray(envelope_error, dtype=np.float64))
    order_statistic = min(len(ordered_error), math.ceil((len(ordered_error) + 1) * (1.0 - ALPHA_ENV)))
    g_quant_error = float(np.max(np.linalg.norm(g_hat.astype(np.float64) - anchors, axis=1)))
    epsilon_value = max(
        np.finfo(np.float16).tiny,
        2.0 * center_error + g_quant_error + 2.0 * u_operator_error,
    )
    envelope_b = float(ordered_error[order_statistic - 1] + epsilon_value)
    v_g = np.mean(np.square(radii), axis=0)
    v_s = float(np.mean(v_g))

    v_g_q, v_g_scale, _ = _quantize_positive_vector(v_g)
    v_s_q, v_s_scale, _ = _quantize_positive_scalar(v_s)
    b_q, b_scale, decoded_b = _quantize_positive_upper_bound(envelope_b)
    epsilon_q, epsilon_scale, decoded_epsilon = _quantize_positive_upper_bound(
        epsilon_value
    )
    if decoded_b < envelope_b or decoded_epsilon < epsilon_value:
        raise AssertionError("conservative bound quantization regressed")
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA, dtype=np.str_),
        "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
        "class_registry": np.asarray(classes, dtype=np.str_),
        "g_q": g_q,
        "g_scale": g_scale,
        "u_q": u_q,
        "u_scale": u_scale,
        "v_g_q": v_g_q,
        "v_g_scale": v_g_scale,
        "v_s_q": v_s_q,
        "v_s_scale": v_s_scale,
        "b_q": b_q,
        "b_scale": b_scale,
        "epsilon_q": epsilon_q,
        "epsilon_scale": epsilon_scale,
    }
    _validate_npz_payload(payload)
    registry_sha = _canonical_sha256(list(classes))
    resource = _resource_receipt(payload, c, len(centers))
    protocol = {
        "aggregate_only_input": True,
        "source_sample_access": False,
        "source_identifier_access": False,
        "source_path_access": False,
        "query_access": False,
        "truth_access": False,
        "dense_source_export": False,
        "dequantized_persistent_cache": False,
        "sidecar_substitution": False,
        "phase2_mutable": False,
        "class_radii_semantics": "per_coordinate_rms_chord_scatter_v1",
    }
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"immutable D111 output already exists: {root}")
    root.mkdir(parents=True)
    npz_path = root / NPZ_NAME
    np.savez_compressed(npz_path, **payload)
    npz_sha = _sha256_file(npz_path)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "rank": RANK,
        "class_count": c,
        "source_domain_count": len(centers),
        "class_registry_sha256": registry_sha,
        **hashes,
        "component_state": STATE,
        "formal_phase2_eligible": False,
        "outer_bundle_signature_required": True,
        "member_allowlist": [NPZ_NAME],
        "npz_member_allowlist": sorted(ALLOWED_NPZ_MEMBERS),
        "rounding_schema": ROUNDING_SCHEMA,
        "subspace_schema": SUBSPACE_SCHEMA,
        "envelope_schema": ENVELOPE_SCHEMA,
        "alpha_env": ALPHA_ENV,
        "source_basis_quantization_error_bound": basis_error,
        "source_center_quantization_error_bound": center_error,
        "projection_quantization_error_bound": projection_error,
        "spectral_gap": spectral_gap,
        "spectral_gap_required_minimum": 2.0 * projection_error,
        "u_operator_quantization_error": u_operator_error,
        "u_orthogonality_error": u_orthogonality_error,
        "envelope_b_unquantized_upper_bound": envelope_b,
        "epsilon_unquantized_upper_bound": epsilon_value,
        "resource_receipt": resource,
        "protocol_receipt": protocol,
        "component_npz_sha256": npz_sha,
    }
    manifest["content_root_sha256"] = _content_root(manifest)
    if set(manifest) != _MANIFEST_FIELDS:
        raise AssertionError("internal D111 manifest field drift")
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha = _sha256_file(manifest_path)
    (root / MANIFEST_SHA_NAME).write_text(
        f"{manifest_sha}  {MANIFEST_NAME}\n", encoding="ascii"
    )
    _read_component(root, sealed=False)
    return {
        "root": str(root),
        "component_npz_sha256": npz_sha,
        "manifest_sha256": manifest_sha,
        "content_root_sha256": manifest["content_root_sha256"],
        "resource_receipt": resource,
    }


def _read_component(
    directory: Path, *, sealed: bool
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], str]:
    expected_members = {NPZ_NAME, MANIFEST_NAME, MANIFEST_SHA_NAME}
    if sealed:
        expected_members |= {OUTER_SEAL_NAME, OUTER_SEAL_SHA_NAME}
    actual = {item.name for item in directory.iterdir()} if directory.is_dir() else set()
    if actual != expected_members:
        raise D111BundleError("D111 directory member allowlist mismatch")
    manifest_path = directory / MANIFEST_NAME
    manifest_sha = _sha256_file(manifest_path)
    if (directory / MANIFEST_SHA_NAME).read_text(encoding="ascii") != f"{manifest_sha}  {MANIFEST_NAME}\n":
        raise D111BundleError("D111 manifest SHA sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != _MANIFEST_FIELDS:
        raise D111BundleError("D111 manifest field set mismatch")
    if manifest.get("schema") != SCHEMA or manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise D111BundleError("D111 manifest schema mismatch")
    if (
        manifest.get("component_state") != STATE
        or manifest.get("formal_phase2_eligible") is not False
        or manifest.get("outer_bundle_signature_required") is not True
    ):
        raise D111BundleError("D111 component must remain pending its verified outer seal")
    if manifest.get("member_allowlist") != [NPZ_NAME] or manifest.get("npz_member_allowlist") != sorted(ALLOWED_NPZ_MEMBERS):
        raise D111BundleError("D111 manifest allowlist mismatch")
    if manifest.get("rounding_schema") != ROUNDING_SCHEMA or manifest.get("subspace_schema") != SUBSPACE_SCHEMA:
        raise D111BundleError("D111 fixed method schema mismatch")
    if manifest.get("envelope_schema") != ENVELOPE_SCHEMA or manifest.get("alpha_env") != ALPHA_ENV:
        raise D111BundleError("D111 envelope schema mismatch")
    for field in (
        "source_aggregate_sha256",
        "source_aggregate_manifest_sha256",
        "checkpoint_sha256",
        "method_lock_sha256",
        "generation_code_sha256",
        "generation_config_sha256",
        "class_registry_sha256",
        "component_npz_sha256",
        "content_root_sha256",
    ):
        _validate_sha(str(manifest.get(field, "")), field)
    if manifest.get("content_root_sha256") != _content_root(manifest):
        raise D111BundleError("D111 content-root drift")
    npz_path = directory / NPZ_NAME
    if manifest.get("component_npz_sha256") != _sha256_file(npz_path):
        raise D111BundleError("D111 NPZ SHA drift")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != ALLOWED_NPZ_MEMBERS:
            raise D111BundleError("D111 NPZ archive member drift")
        payload = {name: archive[name] for name in archive.files}
    details = _validate_npz_payload(payload)
    classes = details["classes"]
    if manifest.get("class_count") != len(classes) or manifest.get("class_registry_sha256") != _canonical_sha256(list(classes)):
        raise D111BundleError("D111 registry binding drift")
    if float(manifest.get("spectral_gap", -1.0)) <= float(manifest.get("spectral_gap_required_minimum", math.inf)):
        raise D111BundleError("D111 spectral-gap certificate drift")
    u_operator_error = float(manifest.get("u_operator_quantization_error", math.inf))
    if (
        not math.isfinite(u_operator_error)
        or u_operator_error < 0.0
        or u_operator_error > MAX_U_OPERATOR_QUANTIZATION_ERROR
    ):
        raise D111BundleError("D111 shared-basis operator error receipt drift")
    basis = np.asarray(details["basis"], dtype=np.float32)
    orth_error = float(np.linalg.norm(basis.astype(np.float64) @ basis.astype(np.float64).T - np.eye(RANK), ord=2))
    if abs(orth_error - float(manifest.get("u_orthogonality_error", math.inf))) > 1.0e-6:
        raise D111BundleError("D111 orthogonality receipt drift")
    decoded_b = float(np.float32(payload["b_q"]) * np.float32(payload["b_scale"]))
    decoded_epsilon = float(
        np.float32(payload["epsilon_q"]) * np.float32(payload["epsilon_scale"])
    )
    raw_b = float(manifest.get("envelope_b_unquantized_upper_bound", math.inf))
    raw_epsilon = float(manifest.get("epsilon_unquantized_upper_bound", math.inf))
    if (
        not math.isfinite(raw_b)
        or not math.isfinite(raw_epsilon)
        or raw_b <= 0.0
        or raw_epsilon <= 0.0
        or decoded_b < raw_b
        or decoded_epsilon < raw_epsilon
    ):
        raise D111BundleError("D111 conservative upper-bound receipt drift")
    expected_protocol = {
        "aggregate_only_input": True,
        "source_sample_access": False,
        "source_identifier_access": False,
        "source_path_access": False,
        "query_access": False,
        "truth_access": False,
        "dense_source_export": False,
        "dequantized_persistent_cache": False,
        "sidecar_substitution": False,
        "phase2_mutable": False,
        "class_radii_semantics": "per_coordinate_rms_chord_scatter_v1",
    }
    if manifest.get("protocol_receipt") != expected_protocol:
        raise D111BundleError("D111 protocol receipt drift")
    domain_count = int(manifest.get("source_domain_count", -1))
    if domain_count < 1:
        raise D111BundleError("D111 source domain-count binding drift")
    if manifest.get("resource_receipt") != _resource_receipt(payload, len(classes), domain_count):
        raise D111BundleError("D111 resource receipt drift")
    return manifest, payload, details, manifest_sha


def _outer_seal_unsigned(
    manifest: Mapping[str, Any], manifest_sha: str, signer_id: str
) -> dict[str, Any]:
    if not signer_id:
        raise D111BundleError("outer signer_id must be non-empty")
    return {
        "schema": OUTER_SEAL_SCHEMA,
        "bundle_state": OUTER_SEALED_STATE,
        "formal_phase2_eligible": True,
        "algorithm": "Ed25519",
        "signer_id": signer_id,
        "signed_payload": {
            "component_content_root_sha256": manifest["content_root_sha256"],
            "component_manifest_sha256": manifest_sha,
            "component_npz_sha256": manifest["component_npz_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "method_lock_sha256": manifest["method_lock_sha256"],
            "class_registry_sha256": manifest["class_registry_sha256"],
            "source_aggregate_sha256": manifest["source_aggregate_sha256"],
            "source_aggregate_manifest_sha256": manifest["source_aggregate_manifest_sha256"],
            "generation_code_sha256": manifest["generation_code_sha256"],
            "generation_config_sha256": manifest["generation_config_sha256"],
        },
    }


def d111_outer_signing_payload(root: str | Path, *, signer_id: str) -> bytes:
    """Return canonical component metadata for an external Ed25519 authority."""

    manifest, _payload, _details, manifest_sha = _read_component(
        Path(root), sealed=False
    )
    return _canonical_bytes(_outer_seal_unsigned(manifest, manifest_sha, signer_id))


def install_d111_outer_seal(
    root: str | Path, *, signer_id: str, signature_ed25519_hex: str
) -> dict[str, str]:
    """Install only a signature verified by the release-controlled keyring."""

    directory = Path(root)
    manifest, _payload, _details, manifest_sha = _read_component(directory, sealed=False)
    unsigned = _outer_seal_unsigned(manifest, manifest_sha, signer_id)
    public_key = _TRUSTED_OUTER_ED25519_PUBLIC_KEYS.get(signer_id)
    if public_key is None:
        raise D111BundleError("outer signer is not in the trusted keyring")
    if not _verify_ed25519(
        public_key, _canonical_bytes(unsigned), signature_ed25519_hex
    ):
        raise D111BundleError("D111 outer signature verification failed")
    seal = {**unsigned, "signature_ed25519_hex": signature_ed25519_hex}
    seal_path = directory / OUTER_SEAL_NAME
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal_sha = _sha256_file(seal_path)
    (directory / OUTER_SEAL_SHA_NAME).write_text(
        f"{seal_sha}  {OUTER_SEAL_NAME}\n", encoding="ascii"
    )
    return {
        "outer_seal_sha256": seal_sha,
        "signature_ed25519_hex": signature_ed25519_hex,
    }


def load_d111_bundle(
    root: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_method_lock_sha256: str,
    expected_signer_id: str,
) -> D111Bundle:
    """Load one component only after verifying its trusted outer Ed25519 seal."""

    directory = Path(root)
    manifest, payload, details, manifest_sha = _read_component(directory, sealed=True)
    expected = {
        "checkpoint_sha256": _validate_sha(expected_checkpoint_sha256, "expected checkpoint"),
        "method_lock_sha256": _validate_sha(expected_method_lock_sha256, "expected method lock"),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise D111BundleError(f"D111 {field} drift")
    seal_path = directory / OUTER_SEAL_NAME
    seal_sha = _sha256_file(seal_path)
    if (directory / OUTER_SEAL_SHA_NAME).read_text(encoding="ascii") != f"{seal_sha}  {OUTER_SEAL_NAME}\n":
        raise D111BundleError("D111 outer-seal SHA sidecar mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if set(seal) != {
        "schema",
        "bundle_state",
        "formal_phase2_eligible",
        "algorithm",
        "signer_id",
        "signed_payload",
        "signature_ed25519_hex",
    }:
        raise D111BundleError("D111 outer-seal field set mismatch")
    if (
        seal.get("schema") != OUTER_SEAL_SCHEMA
        or seal.get("bundle_state") != OUTER_SEALED_STATE
        or seal.get("formal_phase2_eligible") is not True
        or seal.get("algorithm") != "Ed25519"
    ):
        raise D111BundleError("D111 outer-seal schema mismatch")
    if not expected_signer_id or seal.get("signer_id") != expected_signer_id:
        raise D111BundleError("D111 outer signer drift")
    signature = str(seal.get("signature_ed25519_hex", ""))
    public_key = _TRUSTED_OUTER_ED25519_PUBLIC_KEYS.get(expected_signer_id)
    if public_key is None:
        raise D111BundleError("outer signer is not in the trusted keyring")
    unsigned = {key: value for key, value in seal.items() if key != "signature_ed25519_hex"}
    if not _verify_ed25519(public_key, _canonical_bytes(unsigned), signature):
        raise D111BundleError("D111 outer signature verification failed")
    expected_signed_payload = {
        "component_content_root_sha256": manifest["content_root_sha256"],
        "component_manifest_sha256": manifest_sha,
        "component_npz_sha256": manifest["component_npz_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "method_lock_sha256": manifest["method_lock_sha256"],
        "class_registry_sha256": manifest["class_registry_sha256"],
        "source_aggregate_sha256": manifest["source_aggregate_sha256"],
        "source_aggregate_manifest_sha256": manifest["source_aggregate_manifest_sha256"],
        "generation_code_sha256": manifest["generation_code_sha256"],
        "generation_config_sha256": manifest["generation_config_sha256"],
    }
    if seal.get("signed_payload") != expected_signed_payload:
        raise D111BundleError("D111 outer signature payload drift")

    anchors = _freeze_array(np.asarray(details["anchors"], dtype=np.float32))
    basis = _freeze_array(np.asarray(details["basis"], dtype=np.float32))
    anchor_quantization_l2_error_bound = _int8_vector_l2_error_upper_bounds_from_scales(
        np.asarray(payload["g_scale"], dtype=np.float16)
    )
    basis_operator_error_upper_bound = float(manifest["u_operator_quantization_error"])
    v_g = _freeze_array(
        np.asarray(payload["v_g_q"], dtype=np.float32) * np.float32(payload["v_g_scale"])
    )
    v_s = float(np.float32(payload["v_s_q"]) * np.float32(payload["v_s_scale"]))
    envelope_b = float(np.float32(payload["b_q"]) * np.float32(payload["b_scale"]))
    epsilon = float(np.float32(payload["epsilon_q"]) * np.float32(payload["epsilon_scale"]))
    runtime_manifest = {
        **manifest,
        "effective_bundle_state": OUTER_SEALED_STATE,
        "effective_formal_phase2_eligible": True,
        "verified_outer_seal": seal,
    }
    return D111Bundle(
        class_registry=details["classes"],
        anchors=anchors,
        basis=basis,
        v_g=v_g,
        v_s=v_s,
        envelope_b=envelope_b,
        epsilon=epsilon,
        manifest=_freeze_json(runtime_manifest),
        anchor_quantization_l2_error_bound=anchor_quantization_l2_error_bound,
        basis_operator_error_upper_bound=basis_operator_error_upper_bound,
    )


__all__ = [
    "ALLOWED_NPZ_MEMBERS",
    "D111Bundle",
    "D111BundleError",
    "ENVELOPE_SCHEMA",
    "FEATURE_SCHEMA",
    "NPZ_NAME",
    "SCHEMA",
    "build_d111_bundle_from_aggregate",
    "d111_outer_signing_payload",
    "install_d111_outer_seal",
    "load_d111_bundle",
]
