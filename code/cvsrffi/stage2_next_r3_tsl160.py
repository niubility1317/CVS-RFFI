"""NEXT-R3 TSL-160: sealed Phase1 prior plus a role-free K1/K5 registration head.

The module deliberately owns only the frozen numerical core.  It accepts the
canonical D106 representation, builds a Phase1-only aggregate prior under an
outer source-held receiver/class exclusion, and returns the existing D129
compact affine state so the runtime does not gain a second scoring wire.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_d129_joint6_heads as d129


Z_DIM = d129.Z_DIM
PROTOCOL_SCHEMA = "p2_min_v1"
REPRESENTATION_RULE = "d106_canonical_normalized_relu_zid160"
PRIOR_SCHEMA = "cvs.phase2.next_r3.tsl160.phase1_prior.v1"
FIT_SCHEMA = "cvs.phase2.next_r3.tsl160.fit.v1"
RESOURCE_SCHEMA = "cvs.phase2.next_r3.tsl160.resource.v1"
PRIOR_NUMERIC_PAYLOAD_BYTES = 170
AFFINE_STATE_BYTES_PER_CLASS = 164
CANONICAL_R0 = "canonical_r0"
RDCE_R1_SIGNED_UNIT = "rdce_r1_signed_unit"
PRIOR_SEMANTICS = "pre_adaptation_source_anchor_same_ambient_axes"


class NextR3TSL160Error(ValueError):
    """Raised when the frozen NEXT-R3 TSL-160 contract is violated."""


class NextR3TSL160TieError(NextR3TSL160Error):
    """Raised when a final float32 score has an exact top tie."""


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _canonical_json_bytes(value: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_mapping(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise NextR3TSL160Error(f"{name} must be a lower-case SHA256")
    return text


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _string_tuple(value: Sequence[str], *, name: str, minimum: int = 1) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if (
        len(result) < minimum
        or any(not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise NextR3TSL160Error(f"{name} must be a unique non-empty string sequence")
    return result


def _raw_float32_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != Z_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise NextR3TSL160Error(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(rows)


def canonical_d106_relu_zid160(value: np.ndarray) -> np.ndarray:
    """Return the one allowed D106 representation: ``L2(ReLU(z_id160))``.

    The function is intentionally the only representation conversion in this
    module.  It has no signed pre-ReLU path and fails closed on a zero row.
    """

    rows = _raw_float32_rows(value, name="z_id160")
    positive = np.maximum(rows, np.float32(0.0)).astype(np.float64)
    norms = np.sqrt(np.sum(positive * positive, axis=1, keepdims=True))
    if not np.isfinite(norms).all() or bool(np.any(norms <= 0.0)):
        raise NextR3TSL160Error("D106 ReLU z_id160 contains a zero or non-finite row")
    return _readonly(positive / norms, np.float32)


def _normalized_canonical_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    rows = canonical_d106_relu_zid160(value)
    norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR3TSL160Error(f"{name} canonical normalization drift")
    return rows


def tsl160_cache_sha256(value: np.ndarray) -> str:
    """Hash one exact float32 cache without changing its values or row order."""

    rows = _raw_float32_rows(value, name="TSL160 cache")
    return _sha256_mapping(
        {
            "dtype": "float32",
            "shape": list(rows.shape),
            "values_sha256": _sha256_bytes(rows.tobytes(order="C")),
        }
    )


def _bound_runtime_rows(
    value: np.ndarray,
    *,
    representation_mode: str,
    representation_context_sha256: str,
    cache_sha256: str,
    name: str,
) -> np.ndarray:
    """Validate a runtime-owned cache while preserving its exact float32 bytes.

    R0 is already D106 canonical ReLU/unit data.  R1 is the frozen RDCE bridge
    output in the same ambient axes; its signed unit rows must *not* be pushed
    through ReLU or a second normalization here.
    """

    mode = str(representation_mode)
    if mode not in {CANONICAL_R0, RDCE_R1_SIGNED_UNIT}:
        raise NextR3TSL160Error("representation_mode must be canonical_r0 or rdce_r1_signed_unit")
    _require_sha256(representation_context_sha256, name="representation_context_sha256")
    expected_cache_sha = _require_sha256(cache_sha256, name="cache_sha256")
    rows = _raw_float32_rows(value, name=name)
    if tsl160_cache_sha256(rows) != expected_cache_sha:
        raise NextR3TSL160Error(f"{name} cache SHA256 drift")
    norms = np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-6):
        raise NextR3TSL160Error(f"{name} must be the bound unit-norm cache")
    if mode == CANONICAL_R0 and bool(np.any(rows < 0.0)):
        raise NextR3TSL160Error("canonical_r0 must remain the non-negative D106 ReLU cache")
    return rows


@dataclass(frozen=True, slots=True)
class TSL160RuntimeBinding:
    """The immutable identity required to reuse a sealed TSL Phase1 asset."""

    outer_fold_id: str
    checkpoint_sha256: str
    representation_rule_sha256: str
    phase1_physical_id_root_sha256: str
    phase1_seal_sha256: str

    def __post_init__(self) -> None:
        if not str(self.outer_fold_id):
            raise NextR3TSL160Error("outer_fold_id must be non-empty")
        for name in (
            "checkpoint_sha256",
            "representation_rule_sha256",
            "phase1_physical_id_root_sha256",
            "phase1_seal_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name=name))
        object.__setattr__(self, "outer_fold_id", str(self.outer_fold_id))

    @property
    def mapping(self) -> Mapping[str, str]:
        return _freeze(
            {
                "outer_fold_id": self.outer_fold_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "representation_rule_sha256": self.representation_rule_sha256,
                "phase1_physical_id_root_sha256": self.phase1_physical_id_root_sha256,
                "phase1_seal_sha256": self.phase1_seal_sha256,
            }
        )

    @property
    def binding_sha256(self) -> str:
        return _sha256_mapping(dict(self.mapping))


@dataclass(frozen=True, slots=True)
class TSL160Phase1Cell:
    """One Phase1 receiver/class physical-record cell used only for the prior."""

    receiver_id: str
    class_handle: str
    physical_ids: tuple[str, ...]
    zid160: np.ndarray

    def __post_init__(self) -> None:
        if not str(self.receiver_id) or not str(self.class_handle):
            raise NextR3TSL160Error("Phase1 cell receiver_id and class_handle must be non-empty")
        physical_ids = _string_tuple(self.physical_ids, name="cell physical_ids", minimum=2)
        rows = canonical_d106_relu_zid160(self.zid160)
        if len(rows) != len(physical_ids):
            raise NextR3TSL160Error("cell physical_ids must align exactly with z_id160 rows")
        object.__setattr__(self, "receiver_id", str(self.receiver_id))
        object.__setattr__(self, "class_handle", str(self.class_handle))
        object.__setattr__(self, "physical_ids", physical_ids)
        object.__setattr__(self, "zid160", rows)


@dataclass(frozen=True, slots=True)
class TSL160PhysicalLOOFold:
    """A Phase1 physical leave-one-out audit fold for the trust-radius asset."""

    fold_id: str
    receiver_id: str
    class_handle: str
    registered_classes: tuple[str, ...]
    support_zid160: np.ndarray
    support_labels: tuple[str, ...]
    support_physical_ids: tuple[str, ...]
    validation_zid160: np.ndarray
    validation_labels: tuple[str, ...]
    validation_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.fold_id) or not str(self.receiver_id) or not str(self.class_handle):
            raise NextR3TSL160Error("physical LOO fold identity must be non-empty")
        classes = _string_tuple(self.registered_classes, name="LOO registered_classes", minimum=2)
        support = canonical_d106_relu_zid160(self.support_zid160)
        validation = canonical_d106_relu_zid160(self.validation_zid160)
        support_labels = tuple(str(item) for item in self.support_labels)
        validation_labels = tuple(str(item) for item in self.validation_labels)
        support_ids = _string_tuple(self.support_physical_ids, name="LOO support physical_ids")
        validation_ids = _string_tuple(
            self.validation_physical_ids, name="LOO validation physical_ids"
        )
        if (
            len(support) != len(support_labels)
            or len(support) != len(support_ids)
            or len(validation) != len(validation_labels)
            or len(validation) != len(validation_ids)
            or any(label not in classes for label in support_labels + validation_labels)
            or set(support_ids).intersection(validation_ids)
        ):
            raise NextR3TSL160Error("physical LOO support/validation closure drift")
        indices = np.asarray([classes.index(label) for label in support_labels], dtype=np.int64)
        counts = np.bincount(indices, minlength=len(classes))
        if np.any(counts < 2):
            raise NextR3TSL160Error("physical LOO support must retain at least two rows per class")
        object.__setattr__(self, "fold_id", str(self.fold_id))
        object.__setattr__(self, "receiver_id", str(self.receiver_id))
        object.__setattr__(self, "class_handle", str(self.class_handle))
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "support_zid160", support)
        object.__setattr__(self, "support_labels", support_labels)
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "validation_zid160", validation)
        object.__setattr__(self, "validation_labels", validation_labels)
        object.__setattr__(self, "validation_physical_ids", validation_ids)


def phase1_physical_id_root(cells: Sequence[TSL160Phase1Cell]) -> str:
    """Hash the exact Phase1 physical records after outer-fold exclusions."""

    records: list[Mapping[str, str]] = []
    for cell in cells:
        if type(cell) is not TSL160Phase1Cell:
            raise NextR3TSL160Error("physical-root inputs must be TSL160Phase1Cell values")
        records.extend(
            {
                "receiver_id": cell.receiver_id,
                "class_handle": cell.class_handle,
                "physical_id": physical_id,
            }
            for physical_id in cell.physical_ids
        )
    if not records:
        raise NextR3TSL160Error("Phase1 physical root has no eligible records")
    ids = [str(record["physical_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise NextR3TSL160Error("Phase1 physical IDs must be globally unique")
    return _sha256_mapping(sorted(records, key=lambda record: (
        record["receiver_id"], record["class_handle"], record["physical_id"]
    )))


def _array_sha256(value: np.ndarray) -> str:
    rows = np.ascontiguousarray(value)
    return _sha256_bytes(rows.tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class TSL160Phase1Prior:
    """The 170-byte numerical Phase1 TSL prior wire plus immutable bindings."""

    q_logv0_int8: np.ndarray
    scale_logv0_fp16: np.ndarray
    offset_logv0_fp16: np.ndarray
    nu0_fp16: np.ndarray
    rho_h_mantissa_fp16: np.ndarray
    rho_h_exp2: np.ndarray
    binding: TSL160RuntimeBinding
    schema: str = PRIOR_SCHEMA

    def __post_init__(self) -> None:
        q = np.asarray(self.q_logv0_int8)
        scal = np.asarray(self.scale_logv0_fp16)
        offs = np.asarray(self.offset_logv0_fp16)
        nu0 = np.asarray(self.nu0_fp16)
        mant = np.asarray(self.rho_h_mantissa_fp16)
        exp2 = np.asarray(self.rho_h_exp2)
        if (
            self.schema != PRIOR_SCHEMA
            or type(self.binding) is not TSL160RuntimeBinding
            or q.dtype != np.int8
            or q.shape != (Z_DIM,)
            or scal.dtype != np.float16
            or offs.dtype != np.float16
            or nu0.dtype != np.float16
            or mant.dtype != np.float16
            or exp2.dtype != np.int16
            or scal.shape != (1,)
            or offs.shape != (1,)
            or nu0.shape != (1,)
            or mant.shape != (1,)
            or exp2.shape != (1,)
            or not np.isfinite(scal).all()
            or not np.isfinite(offs).all()
            or not np.isfinite(nu0).all()
            or not np.isfinite(mant).all()
            or float(scal[0]) <= 0.0
            or float(nu0[0]) <= 0.0
            or float(mant[0]) <= 0.0
            or float(mant[0]) >= 1.0
        ):
            raise NextR3TSL160Error("TSL Phase1 prior wire dtype/range closure drift")
        logv0 = q.astype(np.float64) * float(scal[0]) + float(offs[0])
        v0 = np.exp(logv0)
        rho = math.ldexp(float(mant[0]), int(exp2[0]))
        if not np.isfinite(v0).all() or np.any(v0 <= 0.0) or not math.isfinite(rho) or rho <= 0.0:
            raise NextR3TSL160Error("TSL Phase1 prior decode is invalid")
        object.__setattr__(self, "q_logv0_int8", _readonly(q, np.int8))
        object.__setattr__(self, "scale_logv0_fp16", _readonly(scal, np.float16))
        object.__setattr__(self, "offset_logv0_fp16", _readonly(offs, np.float16))
        object.__setattr__(self, "nu0_fp16", _readonly(nu0, np.float16))
        object.__setattr__(self, "rho_h_mantissa_fp16", _readonly(mant, np.float16))
        object.__setattr__(self, "rho_h_exp2", _readonly(exp2, np.int16))

    @property
    def numeric_payload_bytes(self) -> int:
        return int(
            self.q_logv0_int8.nbytes
            + self.scale_logv0_fp16.nbytes
            + self.offset_logv0_fp16.nbytes
            + self.nu0_fp16.nbytes
            + self.rho_h_mantissa_fp16.nbytes
            + self.rho_h_exp2.nbytes
        )

    @property
    def v0(self) -> np.ndarray:
        return _readonly(
            np.exp(
                self.q_logv0_int8.astype(np.float64) * float(self.scale_logv0_fp16[0])
                + float(self.offset_logv0_fp16[0])
            ),
            np.float64,
        )

    @property
    def nu0(self) -> float:
        return float(self.nu0_fp16[0])

    @property
    def rho_h(self) -> float:
        return math.ldexp(float(self.rho_h_mantissa_fp16[0]), int(self.rho_h_exp2[0]))

    @property
    def wire_mapping(self) -> Mapping[str, Any]:
        return _freeze(
            {
                "schema": self.schema,
                "representation_rule": REPRESENTATION_RULE,
                "q_logv0_int8_b64": base64.b64encode(self.q_logv0_int8.tobytes()).decode("ascii"),
                "scale_logv0_fp16_b64": base64.b64encode(self.scale_logv0_fp16.tobytes()).decode("ascii"),
                "offset_logv0_fp16_b64": base64.b64encode(self.offset_logv0_fp16.tobytes()).decode("ascii"),
                "nu0_fp16_b64": base64.b64encode(self.nu0_fp16.tobytes()).decode("ascii"),
                "rho_h_mantissa_fp16_b64": base64.b64encode(self.rho_h_mantissa_fp16.tobytes()).decode("ascii"),
                "rho_h_exp2_b64": base64.b64encode(self.rho_h_exp2.tobytes()).decode("ascii"),
                "binding": dict(self.binding.mapping),
            }
        )

    @property
    def prior_sha256(self) -> str:
        return _sha256_mapping(dict(self.wire_mapping))


@dataclass(frozen=True, slots=True)
class TSL160PriorBuild:
    prior: TSL160Phase1Prior
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.prior) is not TSL160Phase1Prior:
            raise NextR3TSL160Error("prior build must carry an exact TSL160Phase1Prior")
        receipt = dict(self.receipt)
        if receipt.get("prior_sha256") != self.prior.prior_sha256:
            raise NextR3TSL160Error("prior build receipt SHA256 drift")
        object.__setattr__(self, "receipt", _freeze(receipt))


def _decode_b64(value: Any, *, dtype: np.dtype[Any], shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, str):
        raise NextR3TSL160Error(f"{name} wire field must be base64 text")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise NextR3TSL160Error(f"{name} wire base64 is invalid") from error
    result = np.frombuffer(raw, dtype=dtype)
    if result.shape != shape:
        raise NextR3TSL160Error(f"{name} wire byte length drift")
    return np.ascontiguousarray(result.copy())


def serialize_tsl160_prior(prior: TSL160Phase1Prior) -> bytes:
    """Serialize the immutable prior wire; no runtime sidecar is accepted."""

    if type(prior) is not TSL160Phase1Prior:
        raise NextR3TSL160Error("serialize requires an exact TSL160Phase1Prior")
    if prior.numeric_payload_bytes != PRIOR_NUMERIC_PAYLOAD_BYTES:
        raise NextR3TSL160Error("prior numeric payload must remain exactly 170 bytes")
    return _canonical_json_bytes(dict(prior.wire_mapping))


def deserialize_tsl160_prior(value: bytes) -> TSL160Phase1Prior:
    """Deserialize and canonicalize the sealed read-only TSL prior wire."""

    if not isinstance(value, bytes):
        raise NextR3TSL160Error("prior wire must be bytes")
    try:
        mapping = json.loads(value.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR3TSL160Error("prior wire is not canonical ASCII JSON") from error
    if not isinstance(mapping, dict) or mapping.get("schema") != PRIOR_SCHEMA:
        raise NextR3TSL160Error("prior wire schema drift")
    if mapping.get("representation_rule") != REPRESENTATION_RULE:
        raise NextR3TSL160Error("prior wire representation binding drift")
    raw_binding = mapping.get("binding")
    if not isinstance(raw_binding, dict):
        raise NextR3TSL160Error("prior wire binding is missing")
    expected_fields = {
        "outer_fold_id",
        "checkpoint_sha256",
        "representation_rule_sha256",
        "phase1_physical_id_root_sha256",
        "phase1_seal_sha256",
    }
    if set(raw_binding) != expected_fields:
        raise NextR3TSL160Error("prior wire binding fields drift")
    prior = TSL160Phase1Prior(
        q_logv0_int8=_decode_b64(mapping.get("q_logv0_int8_b64"), dtype=np.int8, shape=(Z_DIM,), name="q_logv0"),
        scale_logv0_fp16=_decode_b64(mapping.get("scale_logv0_fp16_b64"), dtype=np.float16, shape=(1,), name="scale_logv0"),
        offset_logv0_fp16=_decode_b64(mapping.get("offset_logv0_fp16_b64"), dtype=np.float16, shape=(1,), name="offset_logv0"),
        nu0_fp16=_decode_b64(mapping.get("nu0_fp16_b64"), dtype=np.float16, shape=(1,), name="nu0"),
        rho_h_mantissa_fp16=_decode_b64(mapping.get("rho_h_mantissa_fp16_b64"), dtype=np.float16, shape=(1,), name="rho_h_mantissa"),
        rho_h_exp2=_decode_b64(mapping.get("rho_h_exp2_b64"), dtype=np.int16, shape=(1,), name="rho_h_exp2"),
        binding=TSL160RuntimeBinding(**raw_binding),
    )
    if serialize_tsl160_prior(prior) != value:
        raise NextR3TSL160Error("prior wire must use the canonical no-sidecar encoding")
    return prior


def roundtrip_tsl160_prior(prior: TSL160Phase1Prior) -> TSL160Phase1Prior:
    """Verify immutable wire roundtrip and return the recovered prior."""

    wire = serialize_tsl160_prior(prior)
    recovered = deserialize_tsl160_prior(wire)
    if serialize_tsl160_prior(recovered) != wire or recovered.prior_sha256 != prior.prior_sha256:
        raise NextR3TSL160Error("prior wire roundtrip drift")
    return recovered


def _balanced_support(
    support_rows: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    allowed_k: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int]:
    rows = _raw_float32_rows(support_rows, name="support_zid160")
    # The returned active state is the D129 wire, whose registry contract has
    # a four-class lower bound.  Enforce it here instead of leaking a D129
    # constructor error after a partial fit.
    classes = _string_tuple(registered_classes, name="registered_classes", minimum=4)
    labels = tuple(str(label) for label in support_labels)
    if len(labels) != len(rows) or any(label not in classes for label in labels):
        raise NextR3TSL160Error("support labels must close over all registered classes")
    class_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_index[label] for label in labels], dtype=np.int64)
    counts = np.bincount(indices, minlength=len(classes))
    if np.any(counts < 1) or len(set(int(count) for count in counts)) != 1:
        raise NextR3TSL160Error("support must be balanced K-shot over all registered classes")
    active_k = int(counts[0])
    if active_k not in allowed_k or len(rows) != len(classes) * active_k:
        raise NextR3TSL160Error("TSL160 only permits the frozen K values")
    return rows, classes, indices, active_k


def _phase1_loo_support(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Validate exact single-record LOO support without requiring equal counts."""

    rows = _normalized_canonical_rows(support_zid160, name="LOO support_zid160")
    classes = _string_tuple(registered_classes, name="LOO registered_classes", minimum=2)
    labels = tuple(str(label) for label in support_labels)
    if len(labels) != len(rows) or any(label not in classes for label in labels):
        raise NextR3TSL160Error("physical LOO support labels must close over registered classes")
    class_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_index[label] for label in labels], dtype=np.int64)
    counts = np.bincount(indices, minlength=len(classes))
    if np.any(counts < 2):
        raise NextR3TSL160Error("physical LOO support must retain at least two rows per class")
    return rows, classes, indices, counts


def _residual_degrees_of_freedom(counts: np.ndarray) -> int:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 2):
        raise NextR3TSL160Error("residual degrees of freedom require per-class counts >=2")
    return int(np.sum(values - 1))


def _geometry(
    rows: np.ndarray,
    indices: np.ndarray,
    classes: tuple[str, ...],
    *,
    prior: TSL160Phase1Prior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, int]:
    """Compute the frozen K>=2 EB diagonal and spherical reference geometry."""

    class_count = len(classes)
    counts = np.bincount(indices, minlength=class_count)
    residual_degrees_of_freedom = _residual_degrees_of_freedom(counts)
    means = np.empty((class_count, Z_DIM), dtype=np.float64)
    residual_energy = np.zeros(Z_DIM, dtype=np.float64)
    for class_index in range(class_count):
        group = rows[indices == class_index].astype(np.float64)
        mean = np.mean(group, axis=0)
        means[class_index] = mean
        # e_ck=((K-1)/K)[z_ck-(K-1)^-1 sum_{j!=k}z_cj] == z_ck-mu_c.
        residuals = group - mean[None, :]
        residual_energy += np.sum(residuals * residuals, axis=0)
    denominator = prior.nu0 + residual_degrees_of_freedom
    v_post = (prior.nu0 * prior.v0 + residual_energy) / denominator
    floor = max(float(np.finfo(np.float32).tiny), 64.0 * float(np.finfo(np.float32).eps) * float(np.mean(prior.v0)))
    if not np.isfinite(v_post).all() or np.any(v_post < floor):
        raise NextR3TSL160Error("empirical-Bayes diagonal variance fell below its frozen floor")
    v_sph = float(np.mean(v_post))
    if not math.isfinite(v_sph) or v_sph < floor:
        raise NextR3TSL160Error("spherical reference variance is invalid")
    weight_ref = means / v_sph
    intercept_ref = -0.5 * np.sum(means * means, axis=1) / v_sph
    weight_hat = means / v_post[None, :]
    intercept_hat = -0.5 * np.sum(means * means / v_post[None, :], axis=1)
    # Class centering is part of the frozen trust-radius geometry only.
    weight_ref_centered = weight_ref - np.mean(weight_ref, axis=0, keepdims=True)
    weight_hat_centered = weight_hat - np.mean(weight_hat, axis=0, keepdims=True)
    intercept_ref_centered = intercept_ref - np.mean(intercept_ref)
    intercept_hat_centered = intercept_hat - np.mean(intercept_hat)
    delta_weight = weight_hat_centered - weight_ref_centered
    delta_intercept = intercept_hat_centered - intercept_ref_centered
    distance = float(np.sqrt(np.sum(delta_weight * delta_weight) + np.sum(delta_intercept * delta_intercept)))
    tolerance = 64.0 * float(np.finfo(np.float64).eps) * max(
        1.0,
        float(np.linalg.norm(weight_ref_centered)),
        float(np.linalg.norm(weight_hat_centered)),
        float(np.linalg.norm(intercept_ref_centered)),
        float(np.linalg.norm(intercept_hat_centered)),
    )
    if not math.isfinite(distance) or distance <= tolerance:
        raise NextR3TSL160Error("TSL geometry has no distinguishable diagonal function")
    # Centering is used once for the trust distance.  The deployed affine
    # formula remains the literal uncentered W_ref/W_hat interpolation; a
    # class-common centering shift would preserve argmax but would no longer
    # be the frozen numeric function.
    return (
        means,
        v_post,
        weight_ref,
        intercept_ref,
        distance,
        v_sph,
        weight_hat,
        intercept_hat,
        residual_degrees_of_freedom,
    )


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) < 1 or not np.isfinite(ordered).all() or not (0.0 <= probability <= 1.0):
        raise NextR3TSL160Error("type-7 quantile input is invalid")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def _encode_positive_downward(value: float) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(value) or value <= 0.0:
        raise NextR3TSL160Error("positive radius encoding requires a finite positive value")
    mantissa, exponent = math.frexp(value)
    if exponent < np.iinfo(np.int16).min or exponent > np.iinfo(np.int16).max:
        raise NextR3TSL160Error("positive radius exponent cannot fit the int16 wire")
    encoded = np.float16(mantissa)
    while float(encoded) > mantissa:
        encoded = np.nextafter(encoded, np.float16(0.0), dtype=np.float16)
    decoded = math.ldexp(float(encoded), exponent)
    if decoded <= 0.0 or decoded > value or not math.isfinite(decoded):
        raise NextR3TSL160Error("positive radius round-down encoding drift")
    return np.asarray([encoded], dtype=np.float16), np.asarray([exponent], dtype=np.int16)


def _quantize_logv0(v0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logv0 = np.log(np.asarray(v0, dtype=np.float64))
    lower = float(np.min(logv0))
    upper = float(np.max(logv0))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise NextR3TSL160Error("Phase1 v0 must have non-degenerate diagonal variation")
    offset64 = 0.5 * (upper + lower)
    offset = np.asarray([offset64], dtype=np.float16)
    if not math.isfinite(float(offset[0])):
        raise NextR3TSL160Error("Phase1 v0 offset is not representable on the FP16 wire")
    # Derive the scale from the *stored* FP16 offset, then round upward.  This
    # avoids an endpoint falling outside INT8 solely because an FP16 midpoint
    # moved by one ULP during serialization.
    required_scale = float(np.max(np.abs(logv0 - float(offset[0])))) / 127.0
    scale_value = np.float16(required_scale)
    while float(scale_value) < required_scale:
        scale_value = np.nextafter(scale_value, np.float16(np.inf), dtype=np.float16)
    scale = np.asarray([scale_value], dtype=np.float16)
    if float(scale[0]) <= 0.0 or not math.isfinite(float(scale[0])):
        raise NextR3TSL160Error("Phase1 v0 scale is not representable on the FP16 wire")
    q = np.rint((logv0 - float(offset[0])) / float(scale[0]))
    if np.any(q < -127.0) or np.any(q > 127.0):
        raise NextR3TSL160Error("Phase1 v0 INT8 wire range drift")
    return np.asarray(q, dtype=np.int8), scale, offset


def _prior_from_cells(cells: Sequence[TSL160Phase1Cell], *, binding: TSL160RuntimeBinding) -> TSL160Phase1Prior:
    variances = []
    degrees = []
    for cell in cells:
        rows = cell.zid160.astype(np.float64)
        variances.append(np.var(rows, axis=0, ddof=1))
        degrees.append(len(rows) - 1)
    v0 = np.mean(np.stack(variances, axis=0), axis=0)
    positive = v0[v0 > 0.0]
    if not len(positive):
        raise NextR3TSL160Error("eligible Phase1 cells have zero diagonal variation")
    floor = max(float(np.finfo(np.float32).tiny), 64.0 * float(np.finfo(np.float32).eps) * float(np.mean(positive)))
    v0 = np.maximum(v0, floor)
    q, scale, offset = _quantize_logv0(v0)
    nu0_value = math.exp(float(np.mean(np.log(np.asarray(degrees, dtype=np.float64)))))
    nu0 = np.asarray([nu0_value], dtype=np.float16)
    if float(nu0[0]) <= 0.0 or not math.isfinite(float(nu0[0])):
        raise NextR3TSL160Error("Phase1 nu0 is not representable on the FP16 wire")
    # A provisional positive radius is replaced by physical LOO calibration.
    return TSL160Phase1Prior(
        q_logv0_int8=q,
        scale_logv0_fp16=scale,
        offset_logv0_fp16=offset,
        nu0_fp16=nu0,
        rho_h_mantissa_fp16=np.asarray([np.float16(0.5)], dtype=np.float16),
        rho_h_exp2=np.asarray([np.int16(1)], dtype=np.int16),
        binding=binding,
    )


def _physical_loo_radius(fold: TSL160PhysicalLOOFold, prior: TSL160Phase1Prior) -> float:
    rows, classes, indices, _ = _phase1_loo_support(
        fold.support_zid160,
        fold.support_labels,
        fold.registered_classes,
    )
    (
        _,
        _,
        weight_ref,
        intercept_ref,
        distance,
        _,
        weight_hat,
        intercept_hat,
        _,
    ) = _geometry(rows, indices, classes, prior=prior)
    validation = _normalized_canonical_rows(fold.validation_zid160, name="LOO validation_zid160")
    class_index = {label: index for index, label in enumerate(classes)}
    reference = validation.astype(np.float64) @ weight_ref.T + intercept_ref[None, :]
    diagonal = validation.astype(np.float64) @ weight_hat.T + intercept_hat[None, :]
    if not np.isfinite(reference).all() or not np.isfinite(diagonal).all():
        raise NextR3TSL160Error("physical LOO geometry scoring became non-finite")
    eta_upper = 1.0
    usable_margin = False
    for row_index, label in enumerate(fold.validation_labels):
        true_index = class_index[label]
        if int(np.argmax(reference[row_index])) != true_index:
            continue
        for other_index in range(len(classes)):
            if other_index == true_index:
                continue
            margin = float(reference[row_index, true_index] - reference[row_index, other_index])
            change = float(
                (diagonal[row_index, true_index] - diagonal[row_index, other_index]) - margin
            )
            if margin > 0.0:
                usable_margin = True
                if change < 0.0:
                    eta_upper = min(eta_upper, margin / (-change))
    if not usable_margin:
        raise NextR3TSL160Error("physical LOO fold has no correctly classified reference margin")
    eta = min(1.0, max(0.0, eta_upper))
    radius = eta * distance
    if not math.isfinite(radius) or radius <= 0.0:
        raise NextR3TSL160Error("physical LOO fold produced a non-positive trust radius")
    return radius


def build_tsl160_phase1_prior(
    cells: Sequence[TSL160Phase1Cell],
    physical_loo_folds: Sequence[TSL160PhysicalLOOFold],
    *,
    binding: TSL160RuntimeBinding,
    held_receiver: str,
    held_class: str,
) -> TSL160PriorBuild:
    """Build a sealed Phase1 TSL prior after simultaneous receiver/class exclusion."""

    if type(binding) is not TSL160RuntimeBinding:
        raise NextR3TSL160Error("Phase1 prior needs an exact runtime binding")
    held_receiver = str(held_receiver)
    held_class = str(held_class)
    if not held_receiver or not held_class:
        raise NextR3TSL160Error("held receiver and held class must be non-empty")
    source_cells = tuple(cells)
    if not source_cells or any(type(cell) is not TSL160Phase1Cell for cell in source_cells):
        raise NextR3TSL160Error("Phase1 prior cells must be exact TSL160Phase1Cell values")
    pair_keys = [(cell.receiver_id, cell.class_handle) for cell in source_cells]
    if len(pair_keys) != len(set(pair_keys)):
        raise NextR3TSL160Error("Phase1 prior input has duplicate receiver/class cells")
    eligible = tuple(
        cell
        for cell in source_cells
        if cell.receiver_id != held_receiver and cell.class_handle != held_class
    )
    if len(eligible) < 2:
        raise NextR3TSL160Error("double exclusion leaves too few Phase1 cells")
    if any(cell.receiver_id == held_receiver or cell.class_handle == held_class for cell in eligible):
        raise NextR3TSL160Error("Phase1 prior double-exclusion closure drift")
    root = phase1_physical_id_root(eligible)
    if root != binding.phase1_physical_id_root_sha256:
        raise NextR3TSL160Error("Phase1 physical-ID root does not match the sealed binding")
    eligible_ids = {physical_id for cell in eligible for physical_id in cell.physical_ids}
    folds = tuple(physical_loo_folds)
    if not folds or any(type(fold) is not TSL160PhysicalLOOFold for fold in folds):
        raise NextR3TSL160Error("Phase1 trust radius requires exact physical LOO folds")
    fold_ids = [fold.fold_id for fold in folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise NextR3TSL160Error("physical LOO fold IDs must be unique")
    validation_id_counts = {physical_id: 0 for physical_id in eligible_ids}
    loo_support_class_counts: dict[str, Mapping[str, int]] = {}
    for fold in folds:
        if fold.receiver_id == held_receiver or fold.class_handle == held_class:
            raise NextR3TSL160Error("physical LOO fold violates held receiver/class exclusion")
        if held_class in fold.registered_classes:
            raise NextR3TSL160Error("held class cannot be a Phase1 pseudo-new LOO class")
        fold_ids_used = set(fold.support_physical_ids).union(fold.validation_physical_ids)
        if not fold_ids_used.issubset(eligible_ids):
            raise NextR3TSL160Error("physical LOO fold uses excluded or unsealed physical IDs")
        if len(fold.validation_physical_ids) != 1:
            raise NextR3TSL160Error(
                "each physical LOO fold must hold out exactly one validation physical ID"
            )
        validation_id = fold.validation_physical_ids[0]
        expected_support_ids = eligible_ids - {validation_id}
        if set(fold.support_physical_ids) != expected_support_ids:
            raise NextR3TSL160Error(
                "physical LOO support IDs must be the exact eligible-universe complement"
            )
        validation_id_counts[validation_id] += 1
        class_counts = {
            class_handle: int(sum(label == class_handle for label in fold.support_labels))
            for class_handle in fold.registered_classes
        }
        if any(count < 2 for count in class_counts.values()):
            raise NextR3TSL160Error("physical LOO support must retain at least two rows per class")
        loo_support_class_counts[fold.fold_id] = class_counts
    if any(count != 1 for count in validation_id_counts.values()):
        raise NextR3TSL160Error(
            "eligible Phase1 physical IDs must each appear exactly once as LOO validation"
        )
    provisional = _prior_from_cells(eligible, binding=binding)
    radii = [_physical_loo_radius(fold, provisional) for fold in folds]
    rho_raw = _type7_quantile(radii, 0.05)
    rho_mantissa, rho_exp2 = _encode_positive_downward(rho_raw)
    prior = TSL160Phase1Prior(
        q_logv0_int8=provisional.q_logv0_int8,
        scale_logv0_fp16=provisional.scale_logv0_fp16,
        offset_logv0_fp16=provisional.offset_logv0_fp16,
        nu0_fp16=provisional.nu0_fp16,
        rho_h_mantissa_fp16=rho_mantissa,
        rho_h_exp2=rho_exp2,
        binding=binding,
    )
    if prior.numeric_payload_bytes != PRIOR_NUMERIC_PAYLOAD_BYTES:
        raise NextR3TSL160Error("TSL prior numeric payload must be exactly 170 bytes")
    recovered = roundtrip_tsl160_prior(prior)
    receipt = {
        "schema": PRIOR_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "representation_rule": REPRESENTATION_RULE,
        "binding": dict(binding.mapping),
        "binding_sha256": binding.binding_sha256,
        "held_receiver": held_receiver,
        "held_class": held_class,
        "outer_fold_dual_exclusion": True,
        "eligible_cell_count": len(eligible),
        "excluded_held_receiver_cell_count": sum(cell.receiver_id == held_receiver for cell in source_cells),
        "excluded_held_class_cell_count": sum(cell.class_handle == held_class for cell in source_cells),
        "eligible_phase1_physical_id_root_sha256": root,
        "physical_loo_fold_count": len(folds),
        "physical_loo_fold_ids": tuple(fold_ids),
        "physical_loo_complete_validation_coverage": True,
        "physical_loo_validation_physical_id_count": len(validation_id_counts),
        "physical_loo_validation_exactly_once": True,
        "physical_loo_support_exact_complement": True,
        "unbalanced_single_holdout_loo": True,
        "physical_loo_residual_degrees_of_freedom_policy": "sum_c(n_c-1)",
        "physical_loo_support_class_counts": loo_support_class_counts,
        "phase1_cell_weighting": "equal_cell",
        "rho_policy": "physical_LOO_positive_radius_type7_q05_round_down",
        "rho_raw_before_round_down": rho_raw,
        "rho_h_after_round_down": prior.rho_h,
        "prior_numeric_payload_bytes": prior.numeric_payload_bytes,
        "runtime_sidecar_used": False,
        "roundtrip_prior_sha256": recovered.prior_sha256,
        "prior_sha256": prior.prior_sha256,
    }
    return TSL160PriorBuild(prior=prior, receipt=receipt)


def _validate_prior_binding(prior: TSL160Phase1Prior, binding: TSL160RuntimeBinding) -> None:
    if type(prior) is not TSL160Phase1Prior or type(binding) is not TSL160RuntimeBinding:
        raise NextR3TSL160Error("TSL fit requires exact prior and runtime binding types")
    if prior.binding.binding_sha256 != binding.binding_sha256:
        raise NextR3TSL160Error("TSL runtime binding does not match the sealed Phase1 prior")
    if prior.numeric_payload_bytes != PRIOR_NUMERIC_PAYLOAD_BYTES:
        raise NextR3TSL160Error("TSL prior numerical payload drift")


def _resource_receipt(
    *,
    active_k: int,
    class_count: int,
    support_rows: int,
    state: d129.D129RegistrationHeadState,
) -> Mapping[str, Any]:
    common = {
        "schema": RESOURCE_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_dim": Z_DIM,
        "class_count": class_count,
        "active_k": active_k,
        "phase1_prior_numeric_payload_bytes": PRIOR_NUMERIC_PAYLOAD_BYTES,
        "phase1_prior_numeric_payload_formula": "int8[160]+4*fp16+int16=170B",
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "explicit_dense_matrix_elements_constructed": 0,
        "explicit_spectral_factorization_count": 0,
        "explicit_linear_system_solve_count": 0,
    }
    if active_k == 1:
        return _freeze(
            {
                **common,
                "fit_mode": "exact_qknn_logit_object_alias",
                "incremental_deployed_numeric_state_bytes": 0,
                "incremental_query_head_macs_per_sample": 0,
                "underlying_qknn_resource_required": True,
                "underlying_qknn_resource_included": False,
                "fit_analytic_mac_equivalent": 0,
                "fit_analytic_mac_formula": "0_incremental_TSL_K1_alias",
            }
        )
    if type(state) is not d129.D129AffineHeadState:
        raise NextR3TSL160Error("K5 resource requires an affine D129 state")
    expected = AFFINE_STATE_BYTES_PER_CLASS * class_count
    if state.numeric_state_bytes != expected:
        raise NextR3TSL160Error("TSL affine numeric state formula drift")
    return _freeze(
        {
            **common,
            "fit_mode": "empirical_bayes_diagonal_with_spherical_trust_region",
            "shared_affine_wire": "int8_W[C,160]+fp16_scale[C]+fp16_intercept[C]",
            "deployed_numeric_state_bytes": state.numeric_state_bytes,
            "deployed_numeric_state_formula": "160C+2C+2C=164C_B",
            "query_head_macs_per_sample": Z_DIM * class_count,
            "query_state_bytes": 0,
            "fit_analytic_mac_equivalent": 4 * support_rows * Z_DIM + 8 * Z_DIM + 2 * class_count * Z_DIM,
            "fit_analytic_mac_formula": "4*N*160+8*160+2*C*160",
        }
    )


def _fit_receipt(
    *,
    active_k: int,
    classes: tuple[str, ...],
    support_rows: int,
    prior: TSL160Phase1Prior,
    binding: TSL160RuntimeBinding,
    state: d129.D129RegistrationHeadState,
    geometry: Mapping[str, Any] | None,
    representation_mode: str,
    representation_context_sha256: str,
    support_cache_sha256: str,
) -> Mapping[str, Any]:
    receipt: dict[str, Any] = {
        "schema": FIT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "head": d129.LITE_HEAD,
        "feature_dim": Z_DIM,
        "class_count": len(classes),
        "registered_classes": classes,
        "active_k": active_k,
        "support_rows": support_rows,
        "support_only": True,
        "same_formula_all_registered_classes": True,
        "role_input": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "query_batch_dependency": False,
        "query_truth_access": False,
        "query_role_access": False,
        "class_quota_access": False,
        "true_batch_class_count_access": False,
        "global_reassignment": False,
        "source_runtime_access": False,
        "clean_runtime_access": False,
        "phase2_optimizer_or_backward": False,
        "phase1_representation_rule": REPRESENTATION_RULE,
        "representation_mode": representation_mode,
        "runtime_representation_rule": (
            REPRESENTATION_RULE
            if representation_mode == CANONICAL_R0
            else "rdce_phi_on_canonical_d106_signed_unit_same_ambient_axes"
        ),
        "representation_context_sha256": representation_context_sha256,
        "support_cache_sha256": support_cache_sha256,
        "cache_binding_required": True,
        "rdce_bridge_binding_required": representation_mode == RDCE_R1_SIGNED_UNIT,
        "prior_semantics": PRIOR_SEMANTICS,
        "prior_transported_by_rdce": False,
        "r1_covariance_claim": False,
        "prior_sha256": prior.prior_sha256,
        "runtime_binding_sha256": binding.binding_sha256,
        "binding": dict(binding.mapping),
        "state_type": type(state).__name__,
        "class_label_permutation_equivariant": True,
    }
    if type(state) is d129.D129AffineHeadState:
        receipt["state_sha256"] = state.state_sha256
    else:
        receipt["state_sha256"] = None
    if active_k == 1:
        receipt.update(
            {
                "fit_mode": "exact_qknn_logit_object_alias",
                "historical_k1_equivalence_claim": False,
                "alias_reason": "K1_covariance_unidentifiable_160d",
            }
        )
    else:
        receipt.update(
            {
                "fit_mode": "empirical_bayes_diagonal_with_spherical_trust_region",
                "trust_geometry": dict(geometry or {}),
            }
        )
    return _freeze(receipt)


def fit_tsl160(
    support_zid160: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    prior: TSL160Phase1Prior,
    runtime_binding: TSL160RuntimeBinding,
    representation_mode: str,
    representation_context_sha256: str,
    support_cache_sha256: str,
) -> d129.D129HeadFit:
    """Fit a bound R0/R1 cache without changing its runtime representation."""

    _validate_prior_binding(prior, runtime_binding)
    bound_support = _bound_runtime_rows(
        support_zid160,
        representation_mode=representation_mode,
        representation_context_sha256=representation_context_sha256,
        cache_sha256=support_cache_sha256,
        name="support_zid160",
    )
    rows, classes, indices, active_k = _balanced_support(
        bound_support,
        support_labels,
        registered_classes,
        allowed_k=(1, 5),
    )
    if active_k == 1:
        state = d129.D129K1QKNNAliasState(head=d129.LITE_HEAD, classes=classes)
        return d129.D129HeadFit(
            state=state,
            fit_receipt=_fit_receipt(
                active_k=active_k,
                classes=classes,
                support_rows=len(rows),
                prior=prior,
                binding=runtime_binding,
                state=state,
                geometry=None,
                representation_mode=representation_mode,
                representation_context_sha256=representation_context_sha256,
                support_cache_sha256=support_cache_sha256,
            ),
            resource_receipt=_resource_receipt(
                active_k=active_k,
                class_count=len(classes),
                support_rows=len(rows),
                state=state,
            ),
        )
    (
        _,
        v_post,
        weight_ref,
        intercept_ref,
        distance,
        v_sph,
        weight_hat,
        intercept_hat,
        residual_degrees_of_freedom,
    ) = _geometry(rows, indices, classes, prior=prior)
    eta = min(1.0, prior.rho_h / distance)
    weights = weight_ref + eta * (weight_hat - weight_ref)
    intercepts = intercept_ref + eta * (intercept_hat - intercept_ref)
    if not np.isfinite(weights).all() or not np.isfinite(intercepts).all() or not (0.0 < eta <= 1.0):
        raise NextR3TSL160Error("TSL trust-region affine interpolation is invalid")
    state, quantization = d129._quantize_shared_affine(
        head=d129.LITE_HEAD,
        classes=classes,
        active_k=5,
        weights=weights,
        intercepts=intercepts,
    )
    deployed_weights = state.weight_qint8.astype(np.float64) * state.scale_fp16.astype(np.float64)[:, None]
    deployed_intercepts = state.intercept_fp16.astype(np.float64)
    deployed_reference_weights = weight_ref * float(quantization["shared_logit_scale"])
    deployed_reference_intercepts = intercept_ref * float(quantization["shared_logit_scale"])
    deployed_delta = float(
        np.sqrt(
            np.sum((deployed_weights - deployed_reference_weights) ** 2)
            + np.sum((deployed_intercepts - deployed_reference_intercepts) ** 2)
        )
    )
    deployed_tolerance = 64.0 * float(np.finfo(np.float64).eps) * max(
        1.0,
        float(np.linalg.norm(deployed_weights)),
        float(np.linalg.norm(deployed_reference_weights)),
    )
    if not math.isfinite(deployed_delta) or deployed_delta <= deployed_tolerance:
        raise NextR3TSL160Error("quantized TSL affine has no function beyond the spherical reference")
    geometry = {
        "formula": "EB_diagonal_plus_centered_spherical_trust_region",
        "v_post_min": float(np.min(v_post)),
        "v_post_max": float(np.max(v_post)),
        "v_sph": v_sph,
        "distance_centered_reference_to_hat": distance,
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "balanced_residual_degrees_of_freedom": len(classes) * (active_k - 1),
        "balanced_dof_reduction_exact": residual_degrees_of_freedom
        == len(classes) * (active_k - 1),
        "rho_h": prior.rho_h,
        "eta": eta,
        "quantized_deployed_delta_from_spherical_reference": deployed_delta,
        "quantized_deployed_delta_tolerance": deployed_tolerance,
        "shared_quantization": dict(quantization),
    }
    return d129.D129HeadFit(
        state=state,
        fit_receipt=_fit_receipt(
            active_k=active_k,
            classes=classes,
            support_rows=len(rows),
            prior=prior,
            binding=runtime_binding,
            state=state,
            geometry=geometry,
            representation_mode=representation_mode,
            representation_context_sha256=representation_context_sha256,
            support_cache_sha256=support_cache_sha256,
        ),
        resource_receipt=_resource_receipt(
            active_k=active_k,
            class_count=len(classes),
            support_rows=len(rows),
            state=state,
        ),
    )


def validate_tsl160_fit_binding(fit: d129.D129HeadFit, runtime_binding: TSL160RuntimeBinding) -> None:
    """Fail closed unless a fit receipt carries the active sealed binding."""

    if type(fit) is not d129.D129HeadFit or type(runtime_binding) is not TSL160RuntimeBinding:
        raise NextR3TSL160Error("TSL fit binding validation requires exact types")
    receipt = dict(fit.fit_receipt)
    if receipt.get("runtime_binding_sha256") != runtime_binding.binding_sha256:
        raise NextR3TSL160Error("TSL fit receipt binding does not match this runtime")
    required = {
        "outer_fold_id": runtime_binding.outer_fold_id,
        "checkpoint_sha256": runtime_binding.checkpoint_sha256,
        "representation_rule_sha256": runtime_binding.representation_rule_sha256,
        "phase1_physical_id_root_sha256": runtime_binding.phase1_physical_id_root_sha256,
        "phase1_seal_sha256": runtime_binding.phase1_seal_sha256,
    }
    if receipt.get("binding") != required:
        raise NextR3TSL160Error("TSL fit receipt binding fields drift")
    if (
        receipt.get("representation_mode") not in {CANONICAL_R0, RDCE_R1_SIGNED_UNIT}
        or not isinstance(receipt.get("representation_context_sha256"), str)
        or not isinstance(receipt.get("support_cache_sha256"), str)
        or receipt.get("prior_semantics") != PRIOR_SEMANTICS
        or receipt.get("prior_transported_by_rdce") is not False
        or receipt.get("r1_covariance_claim") is not False
    ):
        raise NextR3TSL160Error("TSL fit representation/prior semantic receipt drift")
    _require_sha256(receipt["representation_context_sha256"], name="fit representation_context_sha256")
    _require_sha256(receipt["support_cache_sha256"], name="fit support_cache_sha256")


def require_unique_float32_top(logits: np.ndarray) -> None:
    """Fail closed for an exact top-score tie in final float32 logits."""

    value = np.asarray(logits)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 2:
        raise NextR3TSL160Error("final logits must be float32 [N,C] with C>=2")
    if not np.isfinite(value).all():
        raise NextR3TSL160Error("final logits must be finite")
    top = np.max(value, axis=1, keepdims=True)
    if bool(np.any(np.sum(value == top, axis=1) != 1)):
        raise NextR3TSL160TieError("exact final float32 top tie is forbidden")


def alias_k1_qknn_logits(
    fit: d129.D129HeadFit,
    qknn_logits: np.ndarray,
    *,
    runtime_binding: TSL160RuntimeBinding,
) -> np.ndarray:
    """Return the exact original qKNN logits object for the K1 TSL comparator."""

    validate_tsl160_fit_binding(fit, runtime_binding)
    if (
        type(fit.state) is not d129.D129K1QKNNAliasState
        or fit.state.head != d129.LITE_HEAD
        or fit.state.active_k != 1
        or not isinstance(qknn_logits, np.ndarray)
        or qknn_logits.dtype != np.float32
        or qknn_logits.ndim != 2
        or qknn_logits.shape[1] != len(fit.state.classes)
    ):
        raise NextR3TSL160Error("K1 TSL alias requires matching exact D129/qKNN logits")
    require_unique_float32_top(qknn_logits)
    return qknn_logits


def score_tsl160_affine(
    fit: d129.D129HeadFit,
    query_zid160: np.ndarray,
    *,
    runtime_binding: TSL160RuntimeBinding,
    representation_mode: str,
    representation_context_sha256: str,
    query_cache_sha256: str,
) -> np.ndarray:
    """Score a bound cache directly; R1 remains signed-unit and byte-stable."""

    validate_tsl160_fit_binding(fit, runtime_binding)
    receipt = dict(fit.fit_receipt)
    if (
        receipt.get("representation_mode") != representation_mode
        or receipt.get("representation_context_sha256") != representation_context_sha256
    ):
        raise NextR3TSL160Error("TSL query representation/context does not match the fit cache")
    query = _bound_runtime_rows(
        query_zid160,
        representation_mode=representation_mode,
        representation_context_sha256=representation_context_sha256,
        cache_sha256=query_cache_sha256,
        name="query_zid160",
    )
    state = fit.state
    if (
        type(state) is not d129.D129AffineHeadState
        or state.head != d129.LITE_HEAD
        or state.active_k != 5
        or state.numeric_state_bytes != AFFINE_STATE_BYTES_PER_CLASS * len(state.classes)
    ):
        raise NextR3TSL160Error("TSL affine score requires a K5 Lite160 D129 state")
    logits = d129.score_d129_affine_head(state, query)
    require_unique_float32_top(logits)
    return logits


__all__ = [
    "AFFINE_STATE_BYTES_PER_CLASS",
    "CANONICAL_R0",
    "FIT_SCHEMA",
    "NextR3TSL160Error",
    "NextR3TSL160TieError",
    "PRIOR_NUMERIC_PAYLOAD_BYTES",
    "PRIOR_SCHEMA",
    "PRIOR_SEMANTICS",
    "PROTOCOL_SCHEMA",
    "REPRESENTATION_RULE",
    "RESOURCE_SCHEMA",
    "RDCE_R1_SIGNED_UNIT",
    "TSL160Phase1Cell",
    "TSL160Phase1Prior",
    "TSL160PhysicalLOOFold",
    "TSL160PriorBuild",
    "TSL160RuntimeBinding",
    "alias_k1_qknn_logits",
    "build_tsl160_phase1_prior",
    "canonical_d106_relu_zid160",
    "deserialize_tsl160_prior",
    "fit_tsl160",
    "phase1_physical_id_root",
    "require_unique_float32_top",
    "roundtrip_tsl160_prior",
    "score_tsl160_affine",
    "serialize_tsl160_prior",
    "tsl160_cache_sha256",
    "validate_tsl160_fit_binding",
]
