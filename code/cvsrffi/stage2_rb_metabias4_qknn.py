"""D102 rank-four receiver MetaBias followed by the typed INT8 z_id qKNN.

The module is deliberately split at the Phase1/Phase2 boundary.  Phase1 owns
the class-free domain encoder, domain bank, MetaBias basis and all tuning.
Phase2 receives only their quantized aggregate asset and one labelled support
row.  It solves exactly four diagonal normal-equation coordinates, applies the
frozen box-then-ellipsoid map, re-encodes every support item from its pre-ReLU
tap, and delegates classification to the existing typed Student-t qKNN.

No fit or prediction API accepts query labels, query roles, receiver/TX names,
class quotas, source samples, or a query batch statistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    TypedINT8ZIDSupportBank,
    TypedSharedPSDMetric,
    audit_int8_margin,
    audit_runtime_state,
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
)


Z_DIM = 160
DOMAIN_DIM = 32
CODE_DIM = 4
INT8_MAX = 127.0
EPSILON = 1.0e-12
MAX_STATE_BYTES = 262_144
MAX_POST_BACKBONE_MAC_PER_QUERY = 262_144

LOCK_SCHEMA = "cvs.phase1.rb_metabias4.lock.v1"
ASSET_SCHEMA = "cvs.phase1.rb_metabias4.asset.v1"
STATE_SCHEMA = "cvs.phase2.rb_metabias4_qknn.state.v1"
FIT_AUDIT_SCHEMA = "cvs.phase2.rb_metabias4_qknn.fit_audit.v1"
GEOMETRY_AUDIT_SCHEMA = "cvs.phase2.rb_metabias4_qknn.geometry_audit.v1"
RESOURCE_AUDIT_SCHEMA = "cvs.phase2.rb_metabias4_qknn.resource_audit.v1"
WIRE_MAGIC = b"CVSRBMETA4\x00\x01"
ALLOWED_STAGES = ("S_B", "S_C")


class RBMetaBias4Error(ValueError):
    """Raised when D102 protocol, shape, provenance, or closure drifts."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RBMetaBias4Error(f"{name} must be a lowercase SHA256")
    return text


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _finite_float32_rows(
    value: np.ndarray, width: int, name: str
) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != width
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise RBMetaBias4Error(f"{name} must be finite float32 [N,{width}]")
    return np.ascontiguousarray(rows)


def _finite_float32_vector(
    value: np.ndarray, length: int, name: str, *, positive: bool = False
) -> np.ndarray:
    vector = np.asarray(value)
    if (
        vector.dtype != np.float32
        or vector.shape != (length,)
        or not np.isfinite(vector).all()
        or (positive and np.any(vector <= 0.0))
    ):
        qualifier = "positive " if positive else ""
        raise RBMetaBias4Error(
            f"{name} must be finite {qualifier}float32 [{length}]"
        )
    return np.ascontiguousarray(vector)


def _normalize_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or len(rows) < 1 or not np.isfinite(rows).all():
        raise RBMetaBias4Error(f"{name} must contain finite rows")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPSILON) or not np.isfinite(norms).all():
        raise RBMetaBias4Error(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _registry(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) < 2 or len(set(result)) != len(result) or any(not x for x in result):
        raise RBMetaBias4Error(
            "registered classes must contain unique non-empty opaque handles"
        )
    return result


def _quantize_rows(value: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _finite_float32_rows(value, int(np.asarray(value).shape[1]), name)
    maximum = np.max(np.abs(rows.astype(np.float64)), axis=1)
    if np.any(maximum <= EPSILON):
        raise RBMetaBias4Error(f"{name} contains an all-zero quantization row")
    scales = maximum / INT8_MAX
    codes = np.rint(rows.astype(np.float64) / scales[:, None])
    codes = np.clip(codes, -127.0, 127.0).astype(np.int8)
    scales16 = scales.astype(np.float16)
    if (
        np.any(scales16 <= 0.0)
        or not np.isfinite(scales16).all()
        or np.any(codes == np.int8(-128))
    ):
        raise RBMetaBias4Error(f"{name} INT8/FP16 quantization closure failed")
    return _readonly(codes, np.int8), _readonly(scales16, np.float16)


def _decode_row_quantized(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(codes, dtype=np.float32)
        * np.asarray(scales, dtype=np.float32)[:, None],
        dtype=np.float32,
    )


def _wire_array(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    name_bytes = name.encode("ascii")
    dtype_bytes = array.dtype.str.encode("ascii")
    shape = b"".join(struct.pack("<Q", int(axis)) for axis in array.shape)
    payload = array.tobytes(order="C")
    return b"".join(
        (
            struct.pack("<H", len(name_bytes)),
            name_bytes,
            struct.pack("<H", len(dtype_bytes)),
            dtype_bytes,
            struct.pack("<H", array.ndim),
            shape,
            struct.pack("<Q", len(payload)),
            payload,
        )
    )


@dataclass(frozen=True, slots=True)
class Phase1MetaBias4Lock:
    """Target-invariant receipts and frozen D102 scalar controls."""

    checkpoint_sha256: str
    runtime_sha256: str
    bundle_sha256: str
    external_seal_sha256: str
    method_lock_sha256: str
    receiver_held_receipt_sha256: str
    class_loco_receipt_sha256: str
    tx_probe_receipt_sha256: str
    label_permutation_receipt_sha256: str
    aggregation_receipt_sha256: str
    quantization_receipt_sha256: str
    tx_probe_balanced_accuracy: float
    query_rows_used_for_fit: int = 0
    rank: int = CODE_DIM
    domain_dim: int = DOMAIN_DIM
    schema: str = LOCK_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != LOCK_SCHEMA
            or type(self.rank) is not int
            or self.rank != CODE_DIM
            or type(self.domain_dim) is not int
            or self.domain_dim != DOMAIN_DIM
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
        ):
            raise RBMetaBias4Error("Phase1 MetaBias4 rank/domain/query lock drift")
        value = float(self.tx_probe_balanced_accuracy)
        if not math.isfinite(value) or not 0.0 <= value <= 0.25:
            raise RBMetaBias4Error(
                "Phase1 class-free domain TX probe balanced accuracy exceeds 25%"
            )
        for field in (
            "checkpoint_sha256",
            "runtime_sha256",
            "bundle_sha256",
            "external_seal_sha256",
            "method_lock_sha256",
            "receiver_held_receipt_sha256",
            "class_loco_receipt_sha256",
            "tx_probe_receipt_sha256",
            "label_permutation_receipt_sha256",
            "aggregation_receipt_sha256",
            "quantization_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), field)

    @property
    def lock_digest(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Phase1MetaBias4Asset:
    """Only aggregate INT8/FP16 MetaBias4 knowledge visible to Phase2."""

    basis_codes_qint8: np.ndarray
    basis_scales_fp16: np.ndarray
    domain_u_codes_qint8: np.ndarray
    domain_u_scales_fp16: np.ndarray
    bank_g_codes_qint8: np.ndarray
    bank_g_scales_fp16: np.ndarray
    bank_t_fp16: np.ndarray
    bank_lambda_fp16: np.ndarray
    bank_sigma_fp16: np.ndarray
    lambda0_fp16: np.ndarray
    a_max_fp16: np.ndarray
    temperature_fp16: np.float16
    ellipsoid_radius_fp16: np.float16
    cell_min_physical_count_int16: np.ndarray
    cell_class_count_int16: np.ndarray
    lock: Phase1MetaBias4Lock
    asset_receipt_sha256: str
    class_ids_present: bool = False
    member_ids_present: bool = False
    raw_source_features_present: bool = False
    schema: str = ASSET_SCHEMA

    def __post_init__(self) -> None:
        if type(self.lock) is not Phase1MetaBias4Lock:
            raise RBMetaBias4Error("asset requires an exact Phase1 MetaBias4 lock")
        b = np.asarray(self.basis_codes_qint8)
        bs = np.asarray(self.basis_scales_fp16)
        u = np.asarray(self.domain_u_codes_qint8)
        us = np.asarray(self.domain_u_scales_fp16)
        g = np.asarray(self.bank_g_codes_qint8)
        gs = np.asarray(self.bank_g_scales_fp16)
        t = np.asarray(self.bank_t_fp16)
        precision = np.asarray(self.bank_lambda_fp16)
        sigma = np.asarray(self.bank_sigma_fp16)
        lambda0 = np.asarray(self.lambda0_fp16)
        a_max = np.asarray(self.a_max_fp16)
        physical = np.asarray(self.cell_min_physical_count_int16)
        classes = np.asarray(self.cell_class_count_int16)
        cells = len(g)
        if (
            self.schema != ASSET_SCHEMA
            or b.dtype != np.int8
            or b.shape != (Z_DIM, CODE_DIM)
            or bs.dtype != np.float16
            or bs.shape != (CODE_DIM,)
            or u.dtype != np.int8
            or u.shape != (DOMAIN_DIM, Z_DIM)
            or us.dtype != np.float16
            or us.shape != (DOMAIN_DIM,)
            or g.dtype != np.int8
            or g.ndim != 2
            or g.shape[1] != DOMAIN_DIM
            or cells < 2
            or gs.dtype != np.float16
            or gs.shape != (cells,)
            or t.dtype != np.float16
            or t.shape != (cells, CODE_DIM)
            or precision.dtype != np.float16
            or precision.shape != (cells, CODE_DIM)
            or sigma.dtype != np.float16
            or sigma.shape != (cells,)
            or lambda0.dtype != np.float16
            or lambda0.shape != (CODE_DIM,)
            or a_max.dtype != np.float16
            or a_max.shape != (CODE_DIM,)
            or physical.dtype != np.int16
            or physical.shape != (cells,)
            or classes.dtype != np.int16
            or classes.shape != (cells,)
            or np.any(physical < 2)
            or np.any(classes < 2)
            or any(
                value is not False
                for value in (
                    self.class_ids_present,
                    self.member_ids_present,
                    self.raw_source_features_present,
                )
            )
            or any(
                np.any(array == np.int8(-128))
                for array in (b, u, g)
            )
            or not all(
                np.isfinite(array).all()
                for array in (bs, us, gs, t, precision, sigma, lambda0, a_max)
            )
            or any(
                np.any(array <= 0.0)
                for array in (bs, us, gs, precision, sigma, lambda0, a_max)
            )
        ):
            raise RBMetaBias4Error("Phase1 MetaBias4 asset shape/aggregation drift")
        temperature = float(self.temperature_fp16)
        radius = float(self.ellipsoid_radius_fp16)
        if (
            not math.isfinite(temperature)
            or temperature <= 0.0
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise RBMetaBias4Error("asset temperature/radius must be positive FP16")
        _require_sha256(self.asset_receipt_sha256, "asset_receipt_sha256")
        arrays = _asset_arrays_from_values(
            b, bs, u, us, g, gs, t, precision, sigma, lambda0, a_max, physical, classes
        )
        expected = _asset_digest(
            arrays,
            self.lock,
            np.float16(self.temperature_fp16),
            np.float16(self.ellipsoid_radius_fp16),
        )
        if expected != self.asset_receipt_sha256:
            raise RBMetaBias4Error("Phase1 MetaBias4 asset receipt verification failed")
        for field, array, dtype in (
            ("basis_codes_qint8", b, np.int8),
            ("basis_scales_fp16", bs, np.float16),
            ("domain_u_codes_qint8", u, np.int8),
            ("domain_u_scales_fp16", us, np.float16),
            ("bank_g_codes_qint8", g, np.int8),
            ("bank_g_scales_fp16", gs, np.float16),
            ("bank_t_fp16", t, np.float16),
            ("bank_lambda_fp16", precision, np.float16),
            ("bank_sigma_fp16", sigma, np.float16),
            ("lambda0_fp16", lambda0, np.float16),
            ("a_max_fp16", a_max, np.float16),
            ("cell_min_physical_count_int16", physical, np.int16),
            ("cell_class_count_int16", classes, np.int16),
        ):
            object.__setattr__(self, field, _readonly(array, dtype))

    @property
    def bank_cell_count(self) -> int:
        return len(self.bank_g_codes_qint8)


def _asset_arrays_from_values(
    b: np.ndarray,
    bs: np.ndarray,
    u: np.ndarray,
    us: np.ndarray,
    g: np.ndarray,
    gs: np.ndarray,
    t: np.ndarray,
    precision: np.ndarray,
    sigma: np.ndarray,
    lambda0: np.ndarray,
    a_max: np.ndarray,
    physical: np.ndarray,
    classes: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("basis_codes_qint8", b),
        ("basis_scales_fp16", bs),
        ("domain_u_codes_qint8", u),
        ("domain_u_scales_fp16", us),
        ("bank_g_codes_qint8", g),
        ("bank_g_scales_fp16", gs),
        ("bank_t_fp16", t),
        ("bank_lambda_fp16", precision),
        ("bank_sigma_fp16", sigma),
        ("lambda0_fp16", lambda0),
        ("a_max_fp16", a_max),
        ("cell_min_physical_count_int16", physical),
        ("cell_class_count_int16", classes),
    )


def _asset_digest(
    arrays: tuple[tuple[str, np.ndarray], ...],
    lock: Phase1MetaBias4Lock,
    temperature: np.float16,
    radius: np.float16,
) -> str:
    return _canonical_sha256(
        {
            "schema": ASSET_SCHEMA,
            "lock": asdict(lock),
            "lock_digest": lock.lock_digest,
            "temperature_fp16_hex": np.float16(temperature).tobytes().hex(),
            "ellipsoid_radius_fp16_hex": np.float16(radius).tobytes().hex(),
            "arrays": {name: _array_receipt(value) for name, value in arrays},
            "class_ids_present": False,
            "member_ids_present": False,
            "raw_source_features_present": False,
        }
    )


def build_phase1_metabias4_asset(
    basis_b: np.ndarray,
    domain_u: np.ndarray,
    bank_g: np.ndarray,
    bank_t: np.ndarray,
    bank_lambda: np.ndarray,
    bank_sigma: np.ndarray,
    lambda0: np.ndarray,
    a_max: np.ndarray,
    *,
    temperature: float,
    ellipsoid_radius: float,
    cell_min_physical_count: np.ndarray,
    cell_class_count: np.ndarray,
    lock: Phase1MetaBias4Lock,
) -> Phase1MetaBias4Asset:
    """Quantize one already-trained Phase1 aggregate; no training occurs here."""

    if type(lock) is not Phase1MetaBias4Lock:
        raise RBMetaBias4Error("asset builder requires an exact Phase1 lock")
    b = _finite_float32_rows(basis_b, CODE_DIM, "MetaBias basis B")
    if b.shape != (Z_DIM, CODE_DIM):
        raise RBMetaBias4Error("MetaBias basis B must have shape [160,4]")
    u = _finite_float32_rows(domain_u, Z_DIM, "class-free domain encoder U")
    if u.shape != (DOMAIN_DIM, Z_DIM):
        raise RBMetaBias4Error("domain encoder U must have shape [32,160]")
    g = _finite_float32_rows(bank_g, DOMAIN_DIM, "class-free domain bank g")
    g = _normalize_rows(g, "class-free domain bank g")
    cells = len(g)
    t = _finite_float32_rows(bank_t, CODE_DIM, "MetaBias bank code t")
    precision = _finite_float32_rows(
        bank_lambda, CODE_DIM, "MetaBias bank precision lambda"
    )
    sigma = _finite_float32_vector(
        bank_sigma, cells, "MetaBias bank sigma", positive=True
    )
    prior = _finite_float32_vector(
        lambda0, CODE_DIM, "MetaBias prior lambda0", positive=True
    )
    limits = _finite_float32_vector(
        a_max, CODE_DIM, "MetaBias box a_max", positive=True
    )
    if t.shape != (cells, CODE_DIM) or precision.shape != (cells, CODE_DIM):
        raise RBMetaBias4Error("MetaBias bank t/lambda cell count drift")
    if np.any(precision <= 0.0):
        raise RBMetaBias4Error("MetaBias bank precision must be strictly positive")
    physical = np.asarray(cell_min_physical_count)
    classes = np.asarray(cell_class_count)
    if (
        physical.dtype != np.int16
        or physical.shape != (cells,)
        or classes.dtype != np.int16
        or classes.shape != (cells,)
        or np.any(physical < 2)
        or np.any(classes < 2)
    ):
        raise RBMetaBias4Error(
            "every class-cell aggregate needs >=2 physical samples and every cell >=2 classes"
        )
    temperature16 = np.float16(float(temperature))
    radius16 = np.float16(float(ellipsoid_radius))
    if (
        not np.isfinite(temperature16)
        or temperature16 <= 0.0
        or not np.isfinite(radius16)
        or radius16 <= 0.0
    ):
        raise RBMetaBias4Error("temperature/radius FP16 closure failed")
    b_codes_t, b_scales = _quantize_rows(b.T.copy(), "MetaBias basis columns")
    b_codes = _readonly(b_codes_t.T, np.int8)
    u_codes, u_scales = _quantize_rows(u, "class-free domain encoder U")
    g_codes, g_scales = _quantize_rows(g, "class-free domain bank g")
    values = (
        b_codes,
        b_scales,
        u_codes,
        u_scales,
        g_codes,
        g_scales,
        _readonly(t, np.float16),
        _readonly(precision, np.float16),
        _readonly(sigma, np.float16),
        _readonly(prior, np.float16),
        _readonly(limits, np.float16),
        _readonly(physical, np.int16),
        _readonly(classes, np.int16),
    )
    arrays = _asset_arrays_from_values(*values)
    receipt = _asset_digest(arrays, lock, temperature16, radius16)
    return Phase1MetaBias4Asset(
        basis_codes_qint8=values[0],
        basis_scales_fp16=values[1],
        domain_u_codes_qint8=values[2],
        domain_u_scales_fp16=values[3],
        bank_g_codes_qint8=values[4],
        bank_g_scales_fp16=values[5],
        bank_t_fp16=values[6],
        bank_lambda_fp16=values[7],
        bank_sigma_fp16=values[8],
        lambda0_fp16=values[9],
        a_max_fp16=values[10],
        temperature_fp16=temperature16,
        ellipsoid_radius_fp16=radius16,
        cell_min_physical_count_int16=values[11],
        cell_class_count_int16=values[12],
        lock=lock,
        asset_receipt_sha256=receipt,
    )


def build_phase1_metabias4_asset_from_bundle(
    bundle: Any,
    *,
    lock: Phase1MetaBias4Lock,
) -> Phase1MetaBias4Asset:
    """Verify and convert the canonical Phase1 bundle into the Stage2 type.

    Phase1 keeps ``B`` and ``t`` row-quantized, whereas the Stage2 runtime uses
    a column-addressable ``B`` and FP16 four-codes.  This is the sole conversion
    boundary.  It verifies all semantic bindings and only materializes the
    aggregate numeric members; no class, receiver, day, member, or physical
    identifier is accepted.
    """

    from cvsrffi.phase1_rb_metabias4_bundle import (
        Phase1RBMetaBias4Bundle,
    )

    if type(bundle) is not Phase1RBMetaBias4Bundle:
        raise RBMetaBias4Error(
            "Stage2 conversion requires an exact Phase1 RB-MetaBias4 bundle"
        )
    if type(lock) is not Phase1MetaBias4Lock:
        raise RBMetaBias4Error("Stage2 conversion requires an exact Phase1 lock")
    bundle.__post_init__()
    bindings = (
        (bundle.checkpoint_sha256, lock.checkpoint_sha256, "checkpoint"),
        (bundle.runtime_sha256, lock.runtime_sha256, "runtime"),
        (bundle.method_lock_sha256, lock.method_lock_sha256, "method lock"),
        (bundle.content_root_sha256, lock.bundle_sha256, "bundle content root"),
        (
            _canonical_sha256(dict(bundle.aggregation_receipt)),
            lock.aggregation_receipt_sha256,
            "aggregation receipt",
        ),
        (
            _canonical_sha256(dict(bundle.quantization_receipt)),
            lock.quantization_receipt_sha256,
            "quantization receipt",
        ),
    )
    for actual, expected, name in bindings:
        if actual != expected:
            raise RBMetaBias4Error(f"Phase1-to-Stage2 {name} binding drift")
    aggregation = dict(bundle.aggregation_receipt)
    cells = int(bundle.bank_count)
    minimum_physical = int(
        aggregation.get("minimum_observed_class_cell_physical_count", 0)
    )
    minimum_classes = int(aggregation.get("minimum_classes_per_bank_cell", 0))
    if (
        aggregation.get("class_free_payload") is not True
        or aggregation.get("payload_contains_class_handles") is not False
        or aggregation.get("payload_contains_receiver_or_day_names") is not False
        or aggregation.get("payload_contains_member_or_physical_ids") is not False
        or minimum_physical < 2
        or minimum_classes < 2
    ):
        raise RBMetaBias4Error("Phase1 class-free aggregation receipt drift")
    return build_phase1_metabias4_asset(
        np.ascontiguousarray(bundle.basis(), dtype=np.float32),
        np.ascontiguousarray(bundle.domain_encoder(), dtype=np.float32),
        np.ascontiguousarray(bundle.bank_g(), dtype=np.float32),
        np.ascontiguousarray(bundle.bank_t(), dtype=np.float32),
        np.ascontiguousarray(bundle.bank_precision_diag_fp16, dtype=np.float32),
        np.ascontiguousarray(bundle.bank_sigma_fp16, dtype=np.float32),
        np.ascontiguousarray(bundle.lambda0_diag_fp16, dtype=np.float32),
        np.ascontiguousarray(bundle.amax_fp16, dtype=np.float32),
        temperature=float(bundle.temperature),
        ellipsoid_radius=float(bundle.trust_radius),
        cell_min_physical_count=np.full(cells, minimum_physical, dtype=np.int16),
        cell_class_count=np.full(cells, minimum_classes, dtype=np.int16),
        lock=lock,
    )


def decode_metabias_basis(asset: Phase1MetaBias4Asset) -> np.ndarray:
    if type(asset) is not Phase1MetaBias4Asset:
        raise RBMetaBias4Error("basis decode requires an exact Phase1 asset")
    return np.ascontiguousarray(
        asset.basis_codes_qint8.astype(np.float32)
        * asset.basis_scales_fp16.astype(np.float32)[None, :],
        dtype=np.float32,
    )


def _decode_domain_u(asset: Phase1MetaBias4Asset) -> np.ndarray:
    return _decode_row_quantized(
        asset.domain_u_codes_qint8, asset.domain_u_scales_fp16
    )


def _decode_domain_bank(asset: Phase1MetaBias4Asset) -> np.ndarray:
    raw = _decode_row_quantized(
        asset.bank_g_codes_qint8, asset.bank_g_scales_fp16
    )
    return _normalize_rows(raw, "decoded class-free domain bank")


def serialize_phase1_metabias4_asset(asset: Phase1MetaBias4Asset) -> bytes:
    if type(asset) is not Phase1MetaBias4Asset:
        raise RBMetaBias4Error("asset serialization requires an exact asset")
    arrays = _asset_arrays_from_values(
        asset.basis_codes_qint8,
        asset.basis_scales_fp16,
        asset.domain_u_codes_qint8,
        asset.domain_u_scales_fp16,
        asset.bank_g_codes_qint8,
        asset.bank_g_scales_fp16,
        asset.bank_t_fp16,
        asset.bank_lambda_fp16,
        asset.bank_sigma_fp16,
        asset.lambda0_fp16,
        asset.a_max_fp16,
        asset.cell_min_physical_count_int16,
        asset.cell_class_count_int16,
    )
    header = _canonical_bytes(
        {
            "schema": ASSET_SCHEMA,
            "asset_receipt_sha256": asset.asset_receipt_sha256,
            "lock": asdict(asset.lock),
            "temperature_fp16_hex": asset.temperature_fp16.tobytes().hex(),
            "ellipsoid_radius_fp16_hex": asset.ellipsoid_radius_fp16.tobytes().hex(),
            "arrays": {name: _array_receipt(value) for name, value in arrays},
        }
    )
    return b"".join(
        (
            WIRE_MAGIC,
            struct.pack("<Q", len(header)),
            header,
            struct.pack("<H", len(arrays)),
            *[_wire_array(name, value) for name, value in arrays],
        )
    )


def _geometry_audit(base: np.ndarray, adapted: np.ndarray) -> dict[str, Any]:
    base_rows = normalize_zid_rows(base)
    adapted_rows = normalize_zid_rows(adapted)
    base_mask = np.asarray(base > 0.0)
    adapted_mask = np.asarray(adapted > 0.0)
    base_cosine = np.clip(
        base_rows.astype(np.float64) @ base_rows.astype(np.float64).T, -1.0, 1.0
    )
    adapted_cosine = np.clip(
        adapted_rows.astype(np.float64) @ adapted_rows.astype(np.float64).T,
        -1.0,
        1.0,
    )
    upper = np.triu_indices(len(base_rows), 1)
    delta = np.abs(adapted_cosine[upper] - base_cosine[upper])
    base_neighbor = base_cosine.copy()
    adapted_neighbor = adapted_cosine.copy()
    np.fill_diagonal(base_neighbor, -np.inf)
    np.fill_diagonal(adapted_neighbor, -np.inf)
    neighbor_change = np.argmax(base_neighbor, axis=1) != np.argmax(
        adapted_neighbor, axis=1
    )
    mask_change = base_mask != adapted_mask
    return {
        "schema": GEOMETRY_AUDIT_SCHEMA,
        "row_count": int(len(base_rows)),
        "relu_mask_change_count": int(np.sum(mask_change)),
        "relu_mask_change_row_count": int(np.sum(np.any(mask_change, axis=1))),
        "relu_mask_change_rate": float(np.mean(mask_change)),
        "pairwise_cosine_abs_delta_mean": float(np.mean(delta)),
        "pairwise_cosine_abs_delta_max": float(np.max(delta)),
        "nearest_neighbor_change_count": int(np.sum(neighbor_change)),
        "non_common_geometry_change": bool(
            np.any(mask_change) or np.max(delta) > 1.0e-7
        ),
    }


def _stage_receipt(
    *,
    asset: Phase1MetaBias4Asset,
    stage: str,
    support_receipt_sha256: str,
    a_fp16: np.ndarray,
    bank: TypedINT8ZIDSupportBank,
    metric: TypedSharedPSDMetric,
    fit_audit: Mapping[str, Any],
    geometry_audit: Mapping[str, Any],
) -> str:
    qwire = serialize_typed_zid_runtime_state(bank, metric)
    return _canonical_sha256(
        {
            "schema": STATE_SCHEMA,
            "asset_receipt_sha256": asset.asset_receipt_sha256,
            "stage": stage,
            "support_receipt_sha256": support_receipt_sha256,
            "a_fp16": _array_receipt(a_fp16),
            "qknn_wire_sha256": _sha256_bytes(qwire),
            "fit_audit": fit_audit,
            "geometry_audit": geometry_audit,
            "query_rows_used_for_fit": 0,
        }
    )


@dataclass(frozen=True, slots=True)
class D102Stage2State:
    asset: Phase1MetaBias4Asset
    stage: str
    a_fp16: np.ndarray
    bank: TypedINT8ZIDSupportBank
    metric: TypedSharedPSDMetric
    support_receipt_sha256: str
    fit_audit: Mapping[str, Any]
    support_geometry_audit: Mapping[str, Any]
    state_receipt_sha256: str
    query_state_updates: int = 0
    query_rows_used_for_fit: int = 0
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != STATE_SCHEMA
            or type(self.asset) is not Phase1MetaBias4Asset
            or self.stage not in ALLOWED_STAGES
            or type(self.bank) is not TypedINT8ZIDSupportBank
            or type(self.metric) is not TypedSharedPSDMetric
            or not self.metric.exact_identity
            or self.bank.config_lock_digest != self.metric.config_lock_digest
            or type(self.query_state_updates) is not int
            or self.query_state_updates != 0
            or type(self.query_rows_used_for_fit) is not int
            or self.query_rows_used_for_fit != 0
        ):
            raise RBMetaBias4Error("D102 Stage2 state lifecycle/type drift")
        a = np.asarray(self.a_fp16)
        if a.dtype != np.float16 or a.shape != (CODE_DIM,) or not np.isfinite(a).all():
            raise RBMetaBias4Error("D102 deployed code must be finite FP16 [4]")
        _require_sha256(self.support_receipt_sha256, "support_receipt_sha256")
        _require_sha256(self.state_receipt_sha256, "state_receipt_sha256")
        fit = dict(self.fit_audit)
        geometry = dict(self.support_geometry_audit)
        expected = _stage_receipt(
            asset=self.asset,
            stage=self.stage,
            support_receipt_sha256=self.support_receipt_sha256,
            a_fp16=a,
            bank=self.bank,
            metric=self.metric,
            fit_audit=fit,
            geometry_audit=geometry,
        )
        if expected != self.state_receipt_sha256:
            raise RBMetaBias4Error("D102 Stage2 state receipt verification failed")
        object.__setattr__(self, "a_fp16", _readonly(a, np.float16))
        object.__setattr__(self, "fit_audit", MappingProxyType(fit))
        object.__setattr__(
            self, "support_geometry_audit", MappingProxyType(geometry)
        )


def _encode_support_domain(
    asset: Phase1MetaBias4Asset, support_zdom: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zdom = _finite_float32_rows(support_zdom, Z_DIM, "support z_dom")
    u = _decode_domain_u(asset).astype(np.float64)
    representation = _normalize_rows(
        zdom.astype(np.float64) @ u.T, "encoded class-free support domain"
    ).astype(np.float64)
    bank_g = _decode_domain_bank(asset).astype(np.float64)
    similarity = np.clip(representation @ bank_g.T, -1.0, 1.0)
    temperature = float(asset.temperature_fp16)
    scaled = similarity / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= np.sum(weights, axis=1, keepdims=True)
    sigma = asset.bank_sigma_fp16.astype(np.float64)
    coverage = np.sum(
        weights * np.exp(-(1.0 - similarity) / (sigma[None, :] ** 2)), axis=1
    )
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(coverage).all()
        or np.any(coverage <= 0.0)
        or np.any(coverage > 1.0 + 1.0e-12)
    ):
        raise RBMetaBias4Error("MetaBias4 continuous coverage closure failed")
    precision = coverage[:, None] * (
        weights @ asset.bank_lambda_fp16.astype(np.float64)
    )
    means = weights @ asset.bank_t_fp16.astype(np.float64)
    if np.any(precision <= 0.0) or not np.isfinite(precision).all():
        raise RBMetaBias4Error("MetaBias4 support precision became non-positive")
    return representation, weights, coverage, precision, means


def _apply_metabias(
    asset: Phase1MetaBias4Asset, pre_relu: np.ndarray, a: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pre = _finite_float32_rows(pre_relu, Z_DIM, "pre-ReLU rows")
    code = np.asarray(a)
    if code.shape != (CODE_DIM,) or not np.isfinite(code).all():
        raise RBMetaBias4Error("MetaBias code must be finite [4]")
    bias = decode_metabias_basis(asset).astype(np.float64) @ code.astype(np.float64)
    shifted = pre.astype(np.float64) + bias[None, :]
    relu = np.maximum(shifted, 0.0).astype(np.float32)
    return normalize_zid_rows(relu), np.ascontiguousarray(bias, dtype=np.float32)


def baseline_zid_from_pre_relu(pre_relu: np.ndarray) -> np.ndarray:
    pre = _finite_float32_rows(pre_relu, Z_DIM, "baseline pre-ReLU rows")
    return normalize_zid_rows(np.maximum(pre, 0.0).astype(np.float32))


def build_d102_baseline_bank(
    support_pre_relu: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    qknn_config: Phase1ZIDStudentTLock,
) -> tuple[TypedINT8ZIDSupportBank, TypedSharedPSDMetric, np.ndarray]:
    base = baseline_zid_from_pre_relu(support_pre_relu)
    bank = build_typed_zid_support_bank(
        base,
        support_labels,
        registered_classes,
        config=qknn_config,
    )
    return bank, identity_shared_psd_metric(config=qknn_config), base


def fit_d102_stage2_state(
    asset: Phase1MetaBias4Asset,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    qknn_config: Phase1ZIDStudentTLock,
    stage: str,
    support_receipt_sha256: str,
) -> D102Stage2State:
    """Fit the unique support-only four-dimensional Stage2 state."""

    if type(asset) is not Phase1MetaBias4Asset:
        raise RBMetaBias4Error("fit requires an exact Phase1 MetaBias4 asset")
    if type(qknn_config) is not Phase1ZIDStudentTLock:
        raise RBMetaBias4Error("fit requires an exact typed qKNN lock")
    if stage not in ALLOWED_STAGES:
        raise RBMetaBias4Error("stage must be S_B or S_C")
    _require_sha256(support_receipt_sha256, "support_receipt_sha256")
    pre = _finite_float32_rows(support_pre_relu, Z_DIM, "support pre-ReLU")
    zdom = _finite_float32_rows(support_zdom, Z_DIM, "support z_dom")
    if len(pre) != len(zdom):
        raise RBMetaBias4Error("support pre-ReLU/z_dom row count drift")
    labels = tuple(str(value) for value in support_labels)
    classes = _registry(registered_classes)
    if len(labels) != len(pre) or any(label not in classes for label in labels):
        raise RBMetaBias4Error("support labels must map to the opaque registry")
    counts = tuple(labels.count(label) for label in classes)
    if any(value != qknn_config.active_k for value in counts):
        raise RBMetaBias4Error("each registered class must provide locked balanced K-shot")

    _, weights, coverage, precision, means = _encode_support_domain(asset, zdom)
    class_a = []
    class_b = []
    for label in classes:
        mask = np.asarray([value == label for value in labels])
        class_a.append(np.mean(precision[mask], axis=0))
        class_b.append(np.mean(precision[mask] * means[mask], axis=0))
    a_data = np.mean(np.stack(class_a), axis=0)
    b_data = np.mean(np.stack(class_b), axis=0)
    lambda0 = asset.lambda0_fp16.astype(np.float64)
    system = lambda0 + a_data
    if np.any(system <= 0.0) or not np.isfinite(system).all():
        raise RBMetaBias4Error("Lambda0+A_data must remain strictly positive")
    a_tilde = b_data / system
    limits = asset.a_max_fp16.astype(np.float64)
    a_box = np.clip(a_tilde, -limits, limits)
    box_active = bool(np.any(a_box != a_tilde))
    quadratic = float(np.sum(lambda0 * a_box * a_box))
    radius = float(asset.ellipsoid_radius_fp16)
    ellipsoid_active = quadratic > radius * radius
    if ellipsoid_active:
        a = radius * a_box / math.sqrt(quadratic)
    else:
        a = a_box
    a16 = np.asarray(a, dtype=np.float16)
    deployed = a16.astype(np.float64)
    deployed_quadratic = float(np.sum(lambda0 * deployed * deployed))
    if (
        not np.isfinite(a16).all()
        or np.any(np.abs(deployed) > limits + 1.0e-6)
        or deployed_quadratic > radius * radius + 1.0e-5
    ):
        raise RBMetaBias4Error("FP16 deployed code violates frozen constraints")

    adapted, bias = _apply_metabias(asset, pre, a16)
    baseline = baseline_zid_from_pre_relu(pre)
    bank = build_typed_zid_support_bank(
        adapted, labels, classes, config=qknn_config
    )
    metric = identity_shared_psd_metric(config=qknn_config)
    geometry = _geometry_audit(baseline, adapted)
    rank = int(np.linalg.matrix_rank(np.diag(a_data), tol=1.0e-10))
    fit_audit = {
        "schema": FIT_AUDIT_SCHEMA,
        "stage": stage,
        "registered_class_count": len(classes),
        "support_row_count": len(pre),
        "active_k": qknn_config.active_k,
        "per_class_weights": [1.0 / len(classes)] * len(classes),
        "old_new_role_weights_present": False,
        "new_class_count_weight_present": False,
        "query_rows_used_for_fit": 0,
        "data_information_rank": rank,
        "system_eigenvalue_min": float(np.min(system)),
        "system_eigenvalue_max": float(np.max(system)),
        "system_condition_number": float(np.max(system) / np.min(system)),
        "prior_fraction": float(np.sum(lambda0) / np.sum(system)),
        "a_tilde_norm": float(np.linalg.norm(a_tilde)),
        "a_mapped_norm": float(np.linalg.norm(a)),
        "a_deployed_norm": float(np.linalg.norm(deployed)),
        "a_tilde": [float(value) for value in a_tilde],
        "a_box": [float(value) for value in a_box],
        "a_mapped": [float(value) for value in a],
        "a_deployed": [float(value) for value in deployed],
        "a_fp16_abs_delta_max": float(np.max(np.abs(deployed - a))),
        "box_constraint_active": box_active,
        "ellipsoid_constraint_active": ellipsoid_active,
        "ellipsoid_value": deployed_quadratic,
        "ellipsoid_radius_squared": radius * radius,
        "coverage_min": float(np.min(coverage)),
        "coverage_mean": float(np.mean(coverage)),
        "coverage_max": float(np.max(coverage)),
        "coverage_hard_gate": False,
        "fallback_present": False,
        "singleton_per_class": qknn_config.active_k == 1,
        "bias_norm": float(np.linalg.norm(bias)),
        "bank_weight_entropy_mean": float(
            np.mean(-np.sum(weights * np.log(np.maximum(weights, EPSILON)), axis=1))
        ),
        "class_formula": "mean_per_class_then_equal_mean_across_registered_classes",
        "constraint_order": "coordinate_box_then_lambda0_ellipsoid_radial_map",
    }
    fit_audit_proxy = MappingProxyType(fit_audit)
    geometry_proxy = MappingProxyType(geometry)
    receipt = _stage_receipt(
        asset=asset,
        stage=stage,
        support_receipt_sha256=support_receipt_sha256,
        a_fp16=a16,
        bank=bank,
        metric=metric,
        fit_audit=fit_audit_proxy,
        geometry_audit=geometry_proxy,
    )
    return D102Stage2State(
        asset=asset,
        stage=stage,
        a_fp16=_readonly(a16, np.float16),
        bank=bank,
        metric=metric,
        support_receipt_sha256=support_receipt_sha256,
        fit_audit=fit_audit_proxy,
        support_geometry_audit=geometry_proxy,
        state_receipt_sha256=receipt,
    )


def transform_d102_query(
    state: D102Stage2State, query_pre_relu: np.ndarray
) -> np.ndarray:
    """Read-only query representation transform; no domain/query fit input exists."""

    if type(state) is not D102Stage2State:
        raise RBMetaBias4Error("query transform requires an exact D102 state")
    transformed, _ = _apply_metabias(
        state.asset, query_pre_relu, state.a_fp16
    )
    return transformed


def predict_d102_logits(
    state: D102Stage2State, query_pre_relu: np.ndarray
) -> np.ndarray:
    """Read-only all-registered-class prediction."""

    query = transform_d102_query(state, query_pre_relu)
    return score_zid_student_t_logits(state.bank, query, metric=state.metric)


def audit_d102_query_geometry(
    state: D102Stage2State,
    baseline_bank: TypedINT8ZIDSupportBank,
    query_pre_relu: np.ndarray,
) -> dict[str, Any]:
    """Truth-free observable proof that D102 changes the deployed geometry."""

    if (
        type(state) is not D102Stage2State
        or type(baseline_bank) is not TypedINT8ZIDSupportBank
        or baseline_bank.classes != state.bank.classes
        or baseline_bank.config_lock_digest != state.bank.config_lock_digest
    ):
        raise RBMetaBias4Error("query geometry baseline/state registry drift")
    before_wire = serialize_d102_runtime_state(state)
    base_query = baseline_zid_from_pre_relu(query_pre_relu)
    adapted_query = transform_d102_query(state, query_pre_relu)
    identity = identity_shared_psd_metric(config=baseline_bank.config)
    base_logits = score_zid_student_t_logits(
        baseline_bank, base_query, metric=identity
    )
    adapted_logits = score_zid_student_t_logits(
        state.bank, adapted_query, metric=state.metric
    )
    base_support = decode_zid_support_bank(baseline_bank).astype(np.float64)
    adapted_support = decode_zid_support_bank(state.bank).astype(np.float64)
    base_cosine = base_query.astype(np.float64) @ base_support.T
    adapted_cosine = adapted_query.astype(np.float64) @ adapted_support.T
    base_order = np.argsort(base_logits, axis=1, kind="stable")
    adapted_order = np.argsort(adapted_logits, axis=1, kind="stable")
    rows = np.arange(len(base_logits))
    base_margin = (
        base_logits[rows, base_order[:, -1]]
        - base_logits[rows, base_order[:, -2]]
    )
    adapted_margin = (
        adapted_logits[rows, adapted_order[:, -1]]
        - adapted_logits[rows, adapted_order[:, -2]]
    )
    after_wire = serialize_d102_runtime_state(state)
    return {
        "schema": "cvs.phase2.rb_metabias4_qknn.query_geometry_audit.v1",
        "query_row_count": int(len(base_query)),
        "relu_mask_change_count": int(
            np.sum(
                (np.asarray(query_pre_relu) > 0.0)
                != (
                    np.asarray(query_pre_relu)
                    + (
                        decode_metabias_basis(state.asset)
                        @ state.a_fp16.astype(np.float32)
                    )[None, :]
                    > 0.0
                )
            )
        ),
        "support_neighbor_change_count": int(
            np.sum(np.argmax(base_cosine, axis=1) != np.argmax(adapted_cosine, axis=1))
        ),
        "neighbor_contribution_abs_delta_mean": float(
            np.mean(np.abs(adapted_cosine - base_cosine))
        ),
        "margin_abs_delta_mean": float(np.mean(np.abs(adapted_margin - base_margin))),
        "argmax_change_count": int(
            np.sum(np.argmax(base_logits, axis=1) != np.argmax(adapted_logits, axis=1))
        ),
        "state_unchanged_after_query": before_wire == after_wire,
        "query_state_updates": 0,
        "query_truth_read": False,
        "all_registered_classes_compete": True,
    }


def audit_d102_int8(
    state: D102Stage2State,
    full_precision_support_pre_relu: np.ndarray,
    support_labels: Sequence[str],
    validation_pre_relu: np.ndarray,
) -> dict[str, Any]:
    """Run the existing FP32-vs-INT8 qKNN audit without validation truth."""

    support = transform_d102_query(state, full_precision_support_pre_relu)
    validation = transform_d102_query(state, validation_pre_relu)
    audit = audit_int8_margin(
        state.bank,
        support,
        support_labels,
        validation,
        metric=state.metric,
    )
    result = dict(audit)
    result.update(
        {
            "required_top1_agreement": 0.995,
            "large_margin_flip_count": int(audit["margin_sign_flip_count"]),
            "large_margin_flip_threshold": "all_positive_teacher_margins_stricter",
            "passes_d102_int8_gate": (
                float(audit["top1_agreement"]) >= 0.995
                and int(audit["margin_sign_flip_count"]) == 0
            ),
            "query_truth_read": False,
        }
    )
    return result


def serialize_d102_runtime_state(state: D102Stage2State) -> bytes:
    if type(state) is not D102Stage2State:
        raise RBMetaBias4Error("state serialization requires an exact D102 state")
    asset_wire = serialize_phase1_metabias4_asset(state.asset)
    qknn_wire = serialize_typed_zid_runtime_state(state.bank, state.metric)
    header = _canonical_bytes(
        {
            "schema": STATE_SCHEMA,
            "stage": state.stage,
            "state_receipt_sha256": state.state_receipt_sha256,
            "support_receipt_sha256": state.support_receipt_sha256,
            "asset_wire_sha256": _sha256_bytes(asset_wire),
            "qknn_wire_sha256": _sha256_bytes(qknn_wire),
            "a_fp16": _array_receipt(state.a_fp16),
            "fit_audit": state.fit_audit,
            "support_geometry_audit": state.support_geometry_audit,
            "query_state_updates": 0,
            "query_rows_used_for_fit": 0,
            "fp32_persistent_sidecar": False,
        }
    )
    return b"".join(
        (
            WIRE_MAGIC,
            struct.pack("<Q", len(header)),
            header,
            struct.pack("<Q", len(asset_wire)),
            asset_wire,
            struct.pack("<Q", len(qknn_wire)),
            qknn_wire,
            _wire_array("a_fp16", state.a_fp16),
        )
    )


def audit_d102_resources(state: D102Stage2State) -> dict[str, Any]:
    if type(state) is not D102Stage2State:
        raise RBMetaBias4Error("resource audit requires an exact D102 state")
    asset_wire = serialize_phase1_metabias4_asset(state.asset)
    qknn_wire = serialize_typed_zid_runtime_state(state.bank, state.metric)
    state_wire = serialize_d102_runtime_state(state)
    qknn = audit_runtime_state(state.bank, state.metric)
    support_rows = state.bank.support_row_count
    cells = state.asset.bank_cell_count
    classes = len(state.bank.classes)
    k_shot = state.bank.active_k
    domain_encode = support_rows * DOMAIN_DIM * Z_DIM
    bank_match = support_rows * cells * DOMAIN_DIM
    precision_mix = support_rows * cells * CODE_DIM
    code_mix = support_rows * cells * CODE_DIM
    support_bias = Z_DIM * CODE_DIM
    qknn_scale_fit = 0 if k_shot == 1 else classes * k_shot * k_shot * Z_DIM
    support_build_mac = (
        domain_encode
        + bank_match
        + precision_mix
        + code_mix
        + support_bias
        + qknn_scale_fit
    )
    post_backbone_query_mac = (
        Z_DIM * CODE_DIM + int(qknn["score_query_variable_matmul_mac_per_query"])
    )
    return {
        "schema": RESOURCE_AUDIT_SCHEMA,
        "stage": state.stage,
        "phase1_asset_serialized_bytes": len(asset_wire),
        "qknn_serialized_bytes": len(qknn_wire),
        "a_code_bytes": int(state.a_fp16.nbytes),
        "actual_serialized_state_bytes": len(state_wire),
        "state_sha256": _sha256_bytes(state_wire),
        "state_bytes_no_cross_arm_sharing_discount": True,
        "fp32_persistent_sidecar_bytes": 0,
        "trainable_parameters_stage2": 0,
        "optimizer_steps_stage2": 0,
        "epochs_stage2": 0,
        "domain_encode_mac_support": domain_encode,
        "bank_matching_mac_support": bank_match,
        "precision_mixture_mac_support": precision_mix,
        "code_mixture_mac_support": code_mix,
        "metabias_support_reencode_mac": support_bias,
        "qknn_class_scale_fit_mac": qknn_scale_fit,
        "support_state_build_mac": support_build_mac,
        "metabias_mac_per_query": Z_DIM * CODE_DIM,
        "qknn_mac_per_query": int(
            qknn["score_query_variable_matmul_mac_per_query"]
        ),
        "post_backbone_mac_per_query": post_backbone_query_mac,
        "state_gate_bytes": MAX_STATE_BYTES,
        "query_gate_mac": MAX_POST_BACKBONE_MAC_PER_QUERY,
        "passes_state_gate": len(state_wire) <= MAX_STATE_BYTES,
        "passes_query_mac_gate": (
            post_backbone_query_mac <= MAX_POST_BACKBONE_MAC_PER_QUERY
        ),
        "query_state_updates": 0,
        "query_batch_dependency": False,
        "dense_query_graph": False,
    }


def audit_d102_stage_lifecycle(
    state_b: D102Stage2State, state_c: D102Stage2State
) -> dict[str, Any]:
    if (
        type(state_b) is not D102Stage2State
        or type(state_c) is not D102Stage2State
        or state_b.stage != "S_B"
        or state_c.stage != "S_C"
        or state_b.asset.asset_receipt_sha256
        != state_c.asset.asset_receipt_sha256
    ):
        raise RBMetaBias4Error("lifecycle audit requires matched S_B then S_C states")
    rb = audit_d102_resources(state_b)
    rc = audit_d102_resources(state_c)
    return {
        "schema": "cvs.phase2.rb_metabias4_qknn.lifecycle_resource_audit.v1",
        "s_b_support_rows": state_b.bank.support_row_count,
        "s_c_support_rows": state_c.bank.support_row_count,
        "s_c_reencodes_all_registered_support": True,
        "s_c_reuses_s_b_bank_prefix": False,
        "s_b_support_state_build_mac": rb["support_state_build_mac"],
        "s_c_support_state_build_mac": rc["support_state_build_mac"],
        "total_support_state_build_mac": (
            rb["support_state_build_mac"] + rc["support_state_build_mac"]
        ),
        "s_b_state_bytes": rb["actual_serialized_state_bytes"],
        "s_c_state_bytes": rc["actual_serialized_state_bytes"],
        "state_bytes_no_cross_arm_sharing_discount": True,
    }


__all__ = [
    "ALLOWED_STAGES",
    "CODE_DIM",
    "DOMAIN_DIM",
    "D102Stage2State",
    "MAX_POST_BACKBONE_MAC_PER_QUERY",
    "MAX_STATE_BYTES",
    "Phase1MetaBias4Asset",
    "Phase1MetaBias4Lock",
    "RBMetaBias4Error",
    "audit_d102_int8",
    "audit_d102_query_geometry",
    "audit_d102_resources",
    "audit_d102_stage_lifecycle",
    "baseline_zid_from_pre_relu",
    "build_d102_baseline_bank",
    "build_phase1_metabias4_asset",
    "build_phase1_metabias4_asset_from_bundle",
    "decode_metabias_basis",
    "fit_d102_stage2_state",
    "predict_d102_logits",
    "serialize_d102_runtime_state",
    "serialize_phase1_metabias4_asset",
    "transform_d102_query",
]
