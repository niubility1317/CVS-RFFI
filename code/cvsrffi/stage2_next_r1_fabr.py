"""Frozen NEXT-R1 FABR numerical core and representation primitives.

This module implements only the design-frozen FABR side of NEXT-R1.  It has
no target-query fitting input, scorer, runner, matrix, truth reader, or
fallback.  A runtime integration supplies a *real* functional checkpoint
forward through :func:`fit_fabr_support`'s narrow callback.  The callback must
already use :func:`signed_pre_relu160` on the 160-dimensional ``joint_proj.0``
linear output from that same forward.

The important invariants are deliberately local and fail-closed:

* exactly one Phase1-sealed rank-two INT8/FP16 parameter basis;
* canonical parameter/class/physical ordering;
* K1 separation only and K5 physical leave-one-out compactness;
* four central-difference forwards, a Gauss--Newton ``H_FABR``, and one
  post-quantisation trust re-projection;
* signed pre-ReLU totalisation, never the historical full288/FFT/RF path; and
* exact final-logit top-tie rejection for every K and arm.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA = "cvs.stage2.next_r1.fabr.v2"
ASSET_SCHEMA = "cvs.phase1.next_r1.fabr_asset.v2"
STATE_SCHEMA = "cvs.phase2.next_r1.fabr_state.v2"
RESOURCE_SCHEMA = "cvs.phase2.next_r1.fabr_resource.v2"

Z_DIM = 160
RANK = 2
ALLOWED_K = frozenset((1, 5))

# Design-frozen constants from STAGE2_RD_GOAL_20260731.md section 0.1.
DELTA = 2.0**-6
LAMBDA_F = 1.0
LAMBDA_0 = 1.0e-3
RHO = 0.25
MARGIN_M = 0.20
TEMPERATURE_TAU = 0.10
MAX_CONDITION = 1.0e6
NOISE_FACTOR = 64.0

# This hashes the complete, sole representation rule consumed by FABR.  It is
# an identity seal, not a tunable threshold and not a hash of any target row.
REPRESENTATION_RULE_ID = (
    "joint_proj.0-linear-output160/relu-positive-else-signed/"
    "l2-float64/exact-zero-or-nonfinite-reject/v1"
)
REPRESENTATION_RULE_SHA256 = hashlib.sha256(
    REPRESENTATION_RULE_ID.encode("utf-8")
).hexdigest()

_F16 = np.dtype("<f2")
_I8 = np.dtype("i1")
_F32 = np.dtype("<f4")
_F32_EPS = float(np.finfo(np.float32).eps)
_F32_TINY = float(np.finfo(np.float32).tiny)

# The order is a method-lock order, not a target-selected preference.  The
# flattened tensor order is the listed parameter-key order followed by native
# C-order flattening of each tensor.
BLOCK_PARAMETER_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "t1_norm_affine": ("t1.norm.weight", "t1.norm.bias"),
        "t2_norm_affine": ("t2.norm.weight", "t2.norm.bias"),
        "t3_norm_affine": ("t3.norm.weight", "t3.norm.bias"),
        "joint_proj_bias": ("cls_head.joint_proj.0.bias",),
    }
)
BLOCK_DIMENSIONS: Mapping[str, int] = MappingProxyType(
    {
        "t1_norm_affine": 144,
        "t2_norm_affine": 192,
        "t3_norm_affine": 192,
        "joint_proj_bias": 160,
    }
)
BLOCK_TIE_ORDER = (
    "t1_norm_affine",
    "t2_norm_affine",
    "t3_norm_affine",
    "joint_proj_bias",
)


class FABRError(ValueError):
    """A frozen FABR, representation, or protocol invariant did not close."""


class FABRNoFunctionError(FABRError):
    """A legal computation had no effect beyond the sealed numerical noise."""


class FABRTieError(FABRError):
    """A final float32 logit has an unresolved exact top tie."""


def _reject_no_function(reason: str) -> None:
    raise FABRNoFunctionError(f"REJECT_REVISION_NO_FUNCTION: {reason}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise FABRError("FABR receipt contains a non-canonical or non-finite value")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _deep_freeze(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise FABRError("FABR immutable receipt contains an unsupported value")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise FABRError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FABRError(f"{name} must be a lowercase SHA256") from exc
    return value


def _readonly_exact(value: object, *, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise FABRError(f"{name} must be C-contiguous {dtype.str} with shape {list(shape)}")
    if dtype.kind == "f" and not np.isfinite(value).all():
        raise FABRError(f"{name} must be finite")
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return _deep_freeze(
        {
            "dtype": array.dtype.str,
            "shape": tuple(int(item) for item in array.shape),
            "nbytes": int(array.nbytes),
            "sha256": _sha256(array.tobytes(order="C")),
        }
    )


def canonical_parameter_keys(block_id: str) -> tuple[str, ...]:
    """Return the sole allowed flatten order for a selected parameter block."""

    if block_id not in BLOCK_PARAMETER_KEYS:
        raise FABRError("FABR block is outside the frozen four-block registry")
    return BLOCK_PARAMETER_KEYS[block_id]


def canonical_registered_classes(registered_classes: Sequence[str]) -> tuple[str, ...]:
    """Validate and preserve the frozen registry order (never sort by result)."""

    if isinstance(registered_classes, (str, bytes)):
        raise FABRError("registered classes must be an ordered sequence of strings")
    try:
        classes = tuple(registered_classes)
    except TypeError as exc:
        raise FABRError("registered classes must be an ordered sequence of strings") from exc
    if (
        len(classes) < 2
        or any(not isinstance(item, str) or not item for item in classes)
        or len(set(classes)) != len(classes)
    ):
        raise FABRError("registered classes require at least two unique nonempty strings")
    return classes


def _require_physical_ids(values: Sequence[str], *, expected_rows: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise FABRError("physical IDs must be a row-aligned sequence of opaque strings")
    try:
        physical_ids = tuple(values)
    except TypeError as exc:
        raise FABRError("physical IDs must be a row-aligned sequence of opaque strings") from exc
    if (
        len(physical_ids) != expected_rows
        or any(not isinstance(item, str) or not item for item in physical_ids)
        or len(set(physical_ids)) != len(physical_ids)
    ):
        raise FABRError("physical IDs must be globally unique and exactly row-aligned")
    return physical_ids


def _physical_id_root(values: Sequence[str], *, class_id: str | None = None) -> str:
    payload: dict[str, Any] = {"schema": STATE_SCHEMA, "physical_ids": tuple(values)}
    if class_id is not None:
        payload["class_id"] = class_id
    return _sha256(_canonical_bytes(payload))


def _decode_columns(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        codes.astype(np.float64) * scales.astype(np.float64)[None, :], dtype=np.float64
    )


def _f32_rows(value: object, *, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != _F32
        or array.ndim != 2
        or array.shape[1] != Z_DIM
        or array.shape[0] < 1
        or (rows is not None and array.shape[0] != rows)
        or not np.isfinite(array).all()
    ):
        raise FABRError(f"{name} must be finite float32 [N,{Z_DIM}]")
    return np.ascontiguousarray(array, dtype=np.float32)


def _unit_rows(value: object, *, name: str, rows: int | None = None) -> np.ndarray:
    array = _f32_rows(value, name=name, rows=rows).astype(np.float64, copy=False)
    norms = np.sqrt(np.sum(array * array, axis=1, dtype=np.float64), dtype=np.float64)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise FABRError(f"{name} contains a zero or non-finite row norm")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float64)


def _normalise_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.sqrt(np.sum(vector * vector, dtype=np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FABRError(f"{name} has a zero or non-finite norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float64)


def signed_pre_relu160(pre_relu: object, *, observed_post_relu: object | None = None) -> np.ndarray:
    """Return the frozen 160-D signed-totalised representation.

    ``pre_relu`` must be the *linear output* of ``joint_proj.0`` rather than
    its 320-D input.  The positive ReLU representation is retained whenever it
    is nonzero.  Only an exactly-zero ReLU row uses the signed linear row; an
    exactly-zero or non-finite linear row fails.  Norms are accumulated in
    float64, without an adaptive target-side threshold.
    """

    p = _f32_rows(pre_relu, name="joint_proj.0 pre-ReLU")
    h = np.maximum(p, np.float32(0.0)).astype(np.float32, copy=False)
    if observed_post_relu is not None:
        observed = _f32_rows(
            observed_post_relu, name="observed joint_proj.0 ReLU output", rows=p.shape[0]
        )
        # Byte equality avoids silently tapping another 160-D tensor.
        if h.tobytes(order="C") != observed.tobytes(order="C"):
            raise FABRError("joint_proj.0 pre-ReLU tap does not bind the observed feat_joint")
    use_positive = np.any(h != np.float32(0.0), axis=1)
    use_signed = ~use_positive
    if np.any(use_signed) and np.any(~np.any(p[use_signed] != np.float32(0.0), axis=1)):
        raise FABRError("joint_proj.0 pre-ReLU row is exactly zero")
    selected = np.where(use_positive[:, None], h, p).astype(np.float64, copy=False)
    norms = np.sqrt(np.sum(selected * selected, axis=1, dtype=np.float64), dtype=np.float64)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise FABRError("signed-pre-ReLU160 norm is zero or non-finite")
    result = np.ascontiguousarray(selected / norms[:, None], dtype=np.float32)
    if not np.isfinite(result).all():
        raise FABRError("signed-pre-ReLU160 became non-finite")
    result.setflags(write=False)
    return result


def strict_top1_predictions(final_logits: object) -> np.ndarray:
    """Return unique float32 top-1 indices or reject every exact top tie.

    This deliberately detects equality before ``argmax``.  It never accepts a
    class/registry/physical-ID/hash/role/truth fallback rule.
    """

    logits = np.asarray(final_logits)
    if (
        logits.dtype != _F32
        or logits.ndim != 2
        or logits.shape[0] < 1
        or logits.shape[1] < 2
        or not np.isfinite(logits).all()
    ):
        raise FABRError("final logits must be finite float32 [N,C>=2]")
    maxima = np.max(logits, axis=1, keepdims=True)
    top_count = np.sum(logits == maxima, axis=1)
    if np.any(top_count != 1):
        raise FABRTieError("TIE_UNRESOLVED / NO_PERFORMANCE_RESULT: exact final-logit top tie")
    result = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    result.setflags(write=False)
    return result


def require_exact_logit_alias(reference_logits: object, alias_logits: object) -> None:
    """Require the K1 F/L logit tensor to be an exact Q alias."""

    reference = np.asarray(reference_logits)
    alias = np.asarray(alias_logits)
    if reference.dtype != _F32 or alias.dtype != _F32 or not np.array_equal(reference, alias):
        raise FABRError("K1 F/L logits must be exact float32 aliases of Q logits")


@dataclass(frozen=True, slots=True)
class FABRForwardBatch:
    """One real callback result: signed-pre-ReLU160 and aligned physical IDs."""

    features: np.ndarray
    physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = _f32_rows(self.features, name="FABR forward features")
        rows = np.array(rows, dtype=np.float32, copy=True, order="C")
        rows.setflags(write=False)
        ids = _require_physical_ids(self.physical_ids, expected_rows=rows.shape[0])
        object.__setattr__(self, "features", rows)
        object.__setattr__(self, "physical_ids", ids)


ForwardWithCoeff = Callable[[Any, np.ndarray], FABRForwardBatch]


@dataclass(frozen=True, slots=True)
class FABRAsset:
    """Immutable Phase1-only seal for one selected rank-two FABR block."""

    checkpoint_sha256: str
    phase1_seal_sha256: str
    phase1_selection_sha256: str
    block_id: str
    basis_qint8: np.ndarray
    basis_scale_fp16: np.ndarray
    fisher_k_fp16: np.ndarray
    forward_jitter_tolerance_fp16: np.ndarray
    representation_rule_sha256: str = REPRESENTATION_RULE_SHA256
    schema: str = ASSET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ASSET_SCHEMA:
            raise FABRError("FABR asset schema drift")
        for name in ("checkpoint_sha256", "phase1_seal_sha256", "phase1_selection_sha256"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        representation_rule = _require_sha256(
            self.representation_rule_sha256, "representation_rule_sha256"
        )
        if representation_rule != REPRESENTATION_RULE_SHA256:
            raise FABRError("FABR asset representation-rule identity drift")
        object.__setattr__(self, "representation_rule_sha256", representation_rule)
        if self.block_id not in BLOCK_DIMENSIONS:
            raise FABRError("FABR asset block is outside the frozen four-block registry")
        dimension = BLOCK_DIMENSIONS[self.block_id]
        codes = _readonly_exact(
            self.basis_qint8, name="basis_qint8", dtype=_I8, shape=(dimension, RANK)
        )
        scales = _readonly_exact(
            self.basis_scale_fp16, name="basis_scale_fp16", dtype=_F16, shape=(RANK,)
        )
        fisher_k = _readonly_exact(
            self.fisher_k_fp16, name="fisher_k_fp16", dtype=_F16, shape=(RANK, RANK)
        )
        jitter = _readonly_exact(
            self.forward_jitter_tolerance_fp16,
            name="forward_jitter_tolerance_fp16",
            dtype=_F16,
            shape=(1,),
        )
        if np.any(codes == np.int8(-128)) or not np.all(np.any(codes != 0, axis=0)):
            raise FABRError("FABR basis requires two nonzero symmetric INT8 columns")
        if np.any(scales <= np.float16(0.0)) or jitter[0] < np.float16(0.0):
            raise FABRError("FABR scales must be positive and jitter nonnegative")
        basis = _decode_columns(codes, scales)
        singular_values = np.linalg.svd(basis, compute_uv=False)
        if (
            singular_values.shape != (RANK,)
            or not np.isfinite(singular_values).all()
            or float(singular_values[-1]) <= 0.0
        ):
            raise FABRError("actual dequantized INT8 FABR basis lost rank two")
        if not np.array_equal(fisher_k, fisher_k.T):
            raise FABRError("sealed complete Fisher K must be exactly symmetric FP16")
        _check_positive_definite(fisher_k.astype(np.float64), name="sealed Fisher K")
        object.__setattr__(self, "basis_qint8", codes)
        object.__setattr__(self, "basis_scale_fp16", scales)
        object.__setattr__(self, "fisher_k_fp16", fisher_k)
        object.__setattr__(self, "forward_jitter_tolerance_fp16", jitter)

    @property
    def numeric_payload_bytes(self) -> int:
        # 2P INT8 + 2 FP16 scales + full 2x2 FP16 K + FP16 jitter = 2P + 14.
        return int(
            self.basis_qint8.nbytes
            + self.basis_scale_fp16.nbytes
            + self.fisher_k_fp16.nbytes
            + self.forward_jitter_tolerance_fp16.nbytes
        )

    @property
    def frozen_constants(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "delta": DELTA,
                "lambda_F": LAMBDA_F,
                "lambda_0": LAMBDA_0,
                "rho": RHO,
                "m": MARGIN_M,
                "tau": TEMPERATURE_TAU,
                "max_condition": MAX_CONDITION,
                "noise_factor": NOISE_FACTOR,
            }
        )


def decode_fabr_basis(asset: FABRAsset) -> np.ndarray:
    if type(asset) is not FABRAsset:
        raise FABRError("FABR basis decode requires an exact FABRAsset")
    value = _decode_columns(asset.basis_qint8, asset.basis_scale_fp16)
    value.setflags(write=False)
    return value


def decode_fabr_fisher_k(asset: FABRAsset) -> np.ndarray:
    if type(asset) is not FABRAsset:
        raise FABRError("FABR Fisher K decode requires an exact FABRAsset")
    value = np.array(asset.fisher_k_fp16, dtype=np.float64, copy=True, order="C")
    value.setflags(write=False)
    return value


def serialize_fabr_asset(asset: FABRAsset) -> bytes:
    if type(asset) is not FABRAsset:
        raise FABRError("FABR asset serialization requires an exact FABRAsset")
    header = {
        "schema": asset.schema,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "phase1_seal_sha256": asset.phase1_seal_sha256,
        "phase1_selection_sha256": asset.phase1_selection_sha256,
        "representation_rule_sha256": asset.representation_rule_sha256,
        "block_id": asset.block_id,
        "parameter_keys": canonical_parameter_keys(asset.block_id),
        "frozen_constants": dict(asset.frozen_constants),
        "arrays": {
            "basis_qint8": dict(_array_receipt(asset.basis_qint8)),
            "basis_scale_fp16": dict(_array_receipt(asset.basis_scale_fp16)),
            "fisher_k_fp16": dict(_array_receipt(asset.fisher_k_fp16)),
            "forward_jitter_tolerance_fp16": dict(_array_receipt(asset.forward_jitter_tolerance_fp16)),
        },
    }
    body = b"".join(
        value.tobytes(order="C")
        for value in (
            asset.basis_qint8,
            asset.basis_scale_fp16,
            asset.fisher_k_fp16,
            asset.forward_jitter_tolerance_fp16,
        )
    )
    return _canonical_bytes(header) + b"\n" + body


def fabr_asset_sha256(asset: FABRAsset) -> str:
    return _sha256(serialize_fabr_asset(asset))


@dataclass(frozen=True, slots=True)
class FABRRuntimeBinding:
    """Explicit caller-row identities required at every real FABR boundary.

    ``actual_checkpoint_sha256`` is the hash of the checkpoint file/content
    selected by the owning loader.  It is intentionally distinct from the
    before/after ``state_dict`` fingerprint, which proves only no mutation.
    """

    actual_checkpoint_sha256: str
    phase1_seal_sha256: str
    representation_rule_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "actual_checkpoint_sha256",
            "phase1_seal_sha256",
            "representation_rule_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "schema": SCHEMA,
                    "actual_checkpoint_sha256": self.actual_checkpoint_sha256,
                    "phase1_seal_sha256": self.phase1_seal_sha256,
                    "representation_rule_sha256": self.representation_rule_sha256,
                }
            )
        )


def _require_runtime_binding(
    asset: FABRAsset, runtime_binding: FABRRuntimeBinding | None
) -> FABRRuntimeBinding:
    if type(asset) is not FABRAsset:
        raise FABRError("FABR runtime binding requires an exact FABRAsset")
    if type(runtime_binding) is not FABRRuntimeBinding:
        raise FABRError("FABR requires an explicit immutable runtime binding")
    if runtime_binding.actual_checkpoint_sha256 != asset.checkpoint_sha256:
        raise FABRError("actual checkpoint file/content SHA256 does not match FABR asset")
    if runtime_binding.phase1_seal_sha256 != asset.phase1_seal_sha256:
        raise FABRError("caller-row Phase1 seal SHA256 does not match FABR asset")
    if (
        runtime_binding.representation_rule_sha256 != asset.representation_rule_sha256
        or runtime_binding.representation_rule_sha256 != REPRESENTATION_RULE_SHA256
    ):
        raise FABRError("caller-row representation-rule SHA256 does not match FABR asset")
    return runtime_binding


@dataclass(frozen=True, slots=True)
class Phase1FisherGeometry:
    """Read-only Phase1 ``G``, ``F`` and receiver-mean covariance ``S``."""

    gradient_second_moment: np.ndarray
    fisher: np.ndarray
    receiver_scatter: np.ndarray
    epsilon_f: float

    def __post_init__(self) -> None:
        arrays = {
            "gradient_second_moment": self.gradient_second_moment,
            "fisher": self.fisher,
            "receiver_scatter": self.receiver_scatter,
        }
        dimension: int | None = None
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype != np.float64 or array.ndim != 2 or array.shape[0] != array.shape[1]:
                raise FABRError(f"{name} must be a square float64 matrix")
            if dimension is None:
                dimension = int(array.shape[0])
            if array.shape != (dimension, dimension) or not np.isfinite(array).all():
                raise FABRError("Phase1 Fisher geometry dimensions/non-finite drift")
            frozen = np.array(array, dtype=np.float64, copy=True, order="C")
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if not math.isfinite(self.epsilon_f) or self.epsilon_f <= 0.0:
            raise FABRError("Phase1 eps_F must be finite and positive")


def phase1_fisher_geometry(per_sample_gradients: object, receiver_ids: Sequence[str]) -> Phase1FisherGeometry:
    """Build the fixed Phase1-only ``G``, ``F`` and ``Cov_rx(E[g|rx])``.

    Caller-side fold construction must already exclude the held receiver and
    held class.  This helper neither accepts nor stores target/query data.
    """

    gradients = np.asarray(per_sample_gradients)
    if (
        gradients.dtype != np.float32
        or gradients.ndim != 2
        or gradients.shape[0] < 2
        or gradients.shape[1] < RANK
        or not np.isfinite(gradients).all()
    ):
        raise FABRError("Phase1 gradients must be finite float32 [N,P>=2]")
    if isinstance(receiver_ids, (str, bytes)):
        raise FABRError("Phase1 receiver IDs must be row-aligned strings")
    receivers = tuple(receiver_ids)
    if (
        len(receivers) != gradients.shape[0]
        or any(not isinstance(item, str) or not item for item in receivers)
        or len(set(receivers)) < 2
    ):
        raise FABRError("Phase1 Fisher needs at least two row-aligned receivers")
    g64 = gradients.astype(np.float64)
    second = (g64.T @ g64) / float(g64.shape[0])
    trace = float(np.trace(second))
    if not math.isfinite(trace) or trace <= 0.0:
        raise FABRError("Phase1 tr(G) must be finite and strictly positive")
    epsilon_f = max(_F32_TINY, NOISE_FACTOR * _F32_EPS * trace / float(g64.shape[1]))
    fisher = second + epsilon_f * np.eye(g64.shape[1], dtype=np.float64)
    receiver_means = np.stack(
        [
            np.mean(g64[[index for index, item in enumerate(receivers) if item == receiver]], axis=0)
            for receiver in sorted(set(receivers))
        ],
        axis=0,
    )
    centered = receiver_means - np.mean(receiver_means, axis=0, keepdims=True)
    scatter = (centered.T @ centered) / float(receiver_means.shape[0])
    return Phase1FisherGeometry(
        gradient_second_moment=np.ascontiguousarray(second, dtype=np.float64),
        fisher=np.ascontiguousarray(fisher, dtype=np.float64),
        receiver_scatter=np.ascontiguousarray(scatter, dtype=np.float64),
        epsilon_f=float(epsilon_f),
    )


@dataclass(frozen=True, slots=True)
class FunctionalOverrideReceipt:
    """Evidence that one temporary parameter map left checkpoint state intact."""

    block_id: str
    coefficient_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    actual_checkpoint_sha256: str
    phase1_seal_sha256: str
    representation_rule_sha256: str
    runtime_binding_sha256: str
    signed_rows: int
    positive_rows: int

    def __post_init__(self) -> None:
        canonical_parameter_keys(self.block_id)
        for name in (
            "coefficient_sha256",
            "state_before_sha256",
            "state_after_sha256",
            "actual_checkpoint_sha256",
            "phase1_seal_sha256",
            "representation_rule_sha256",
            "runtime_binding_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        if self.representation_rule_sha256 != REPRESENTATION_RULE_SHA256:
            raise FABRError("functional receipt representation-rule identity drift")
        if self.signed_rows < 0 or self.positive_rows < 0 or self.signed_rows + self.positive_rows < 1:
            raise FABRError("functional override representation-count drift")


@dataclass(frozen=True, slots=True)
class FunctionalOverrideResult:
    """A signed-pre-ReLU160 batch plus its no-write functional-call receipt."""

    batch: FABRForwardBatch
    receipt: FunctionalOverrideReceipt


def _torch_import() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.func import functional_call
    except ImportError as exc:  # pragma: no cover - deployment packaging fault.
        raise FABRError("FABR functional override requires PyTorch") from exc
    return torch, nn, functional_call


def _state_dict_fingerprint(module: Any) -> str:
    torch, nn, _functional_call = _torch_import()
    if not isinstance(module, nn.Module):
        raise FABRError("functional override model must be torch.nn.Module")
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise FABRError("state_dict contains a non-tensor value")
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_bytes({"shape": tuple(int(item) for item in tensor.shape)}))
        try:
            payload = tensor.numpy().tobytes(order="C")
        except TypeError:
            payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(payload)
    return digest.hexdigest()


def canonical_parameter_layout(model: Any, block_id: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Validate actual model names/shapes against the frozen flat block order."""

    torch, nn, _functional_call = _torch_import()
    if not isinstance(model, nn.Module):
        raise FABRError("parameter layout requires torch.nn.Module")
    named = dict(model.named_parameters())
    layout: list[tuple[str, tuple[int, ...]]] = []
    width = 0
    for key in canonical_parameter_keys(block_id):
        parameter = named.get(key)
        if parameter is None or not torch.is_tensor(parameter):
            raise FABRError(f"model lacks frozen FABR parameter {key}")
        if parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all().item()):
            raise FABRError(f"FABR parameter {key} must be finite float32")
        shape = tuple(int(item) for item in parameter.shape)
        layout.append((key, shape))
        width += int(parameter.numel())
    if width != BLOCK_DIMENSIONS[block_id]:
        raise FABRError("actual parameter layout dimension drifts from the frozen block")
    return tuple(layout)


def flatten_canonical_parameter_block(model: Any, block_id: str) -> Any:
    """Flatten a block in frozen key/C-order without modifying the model."""

    torch, _nn, _functional_call = _torch_import()
    canonical_parameter_layout(model, block_id)
    named = dict(model.named_parameters())
    return torch.cat([named[key].detach().reshape(-1) for key in canonical_parameter_keys(block_id)], dim=0)


def functional_override_parameter_map(model: Any, asset: FABRAsset, coefficient: object) -> Mapping[str, Any]:
    """Build ``phi0+B*a`` replacements for one call; never write ``state_dict``."""

    torch, _nn, _functional_call = _torch_import()
    if type(asset) is not FABRAsset:
        raise FABRError("functional override requires an exact FABRAsset")
    coeff = np.asarray(coefficient)
    if coeff.dtype != _F32 or coeff.shape != (RANK,) or not np.isfinite(coeff).all():
        raise FABRError("functional override coefficient must be finite float32 [2]")
    layout = canonical_parameter_layout(model, asset.block_id)
    flat_base = flatten_canonical_parameter_block(model, asset.block_id)
    basis = decode_fabr_basis(asset)
    delta_np = basis @ coeff.astype(np.float64)
    delta = torch.as_tensor(delta_np, dtype=flat_base.dtype, device=flat_base.device)
    override_flat = flat_base + delta
    replacements: dict[str, Any] = {}
    offset = 0
    for key, shape in layout:
        width = int(np.prod(shape, dtype=np.int64))
        replacements[key] = override_flat[offset : offset + width].reshape(shape)
        offset += width
    if offset != int(override_flat.numel()):
        raise FABRError("functional override flatten closure drift")
    return MappingProxyType(replacements)


def functional_forward_signed_pre_relu160(
    model: Any,
    received_iq: Any,
    physical_ids: Sequence[str],
    asset: FABRAsset,
    coefficient: object,
    *,
    functional_kwargs: Mapping[str, Any],
    runtime_binding: FABRRuntimeBinding | None = None,
) -> FunctionalOverrideResult:
    """Run an explicit real functional forward and capture ``joint_proj.0``.

    ``functional_kwargs`` is intentionally mandatory: the owning runtime must
    bind the exact checkpoint call signature locally.  It may not contain a
    truth/role/quota/source/clean input, and any ``y`` value must be ``None``.
    This narrow boundary avoids importing a historical replacement chain.
    """

    binding = _require_runtime_binding(asset, runtime_binding)
    torch, nn, functional_call = _torch_import()
    if not isinstance(model, nn.Module) or model.training:
        raise FABRError("functional override model must be a frozen eval nn.Module")
    if not torch.is_tensor(received_iq) or received_iq.dtype != torch.float32:
        raise FABRError("functional override received IQ must be float32 tensor")
    if received_iq.ndim < 1 or int(received_iq.shape[0]) < 1 or not bool(torch.isfinite(received_iq).all().item()):
        raise FABRError("functional override received IQ must have finite nonempty batch axis")
    ids = _require_physical_ids(physical_ids, expected_rows=int(received_iq.shape[0]))
    if not isinstance(functional_kwargs, Mapping):
        raise FABRError("functional override kwargs must be an explicit mapping")
    forbidden = ("truth", "role", "quota", "source", "clean", "old_class", "query")
    for key, value in functional_kwargs.items():
        if not isinstance(key, str) or any(token in key.lower() for token in forbidden):
            raise FABRError("functional override kwargs contain a forbidden runtime input")
        if key == "y" and value is not None:
            raise FABRError("functional override may not receive labels")
    try:
        tap = model.get_submodule("cls_head.joint_proj.0")
    except AttributeError as exc:
        raise FABRError("model lacks cls_head.joint_proj.0 pre-ReLU tap") from exc
    if not isinstance(tap, nn.Module):
        raise FABRError("joint_proj.0 pre-ReLU tap is not a module")
    captured: list[Any] = []

    def _capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output)

    before = _state_dict_fingerprint(model)
    handle = tap.register_forward_hook(_capture)
    output: Any = None
    error: BaseException | None = None
    try:
        overrides = dict(functional_override_parameter_map(model, asset, coefficient))
        with torch.no_grad():
            output = functional_call(model, overrides, (received_iq,), dict(functional_kwargs))
    except BaseException as exc:  # Fingerprint still has to close after a fault.
        error = exc
    finally:
        handle.remove()
    after = _state_dict_fingerprint(model)
    if before != after:
        raise FABRError("functional_call mutated checkpoint state_dict") from error
    if error is not None:
        raise FABRError("functional override checkpoint forward failed") from error
    if len(captured) != 1 or not torch.is_tensor(captured[0]):
        raise FABRError("functional forward must capture exactly one joint_proj.0 tensor")
    if not isinstance(output, Mapping) or not torch.is_tensor(output.get("feat_joint")):
        raise FABRError("functional forward must expose feat_joint for pre-ReLU binding")
    pre_relu = captured[0]
    observed = output["feat_joint"]
    z = signed_pre_relu160(pre_relu.detach().cpu().numpy(), observed_post_relu=observed.detach().cpu().numpy())
    p_np = _f32_rows(pre_relu.detach().cpu().numpy(), name="functional pre-ReLU")
    h_np = np.maximum(p_np, np.float32(0.0))
    positive_rows = int(np.sum(np.any(h_np != np.float32(0.0), axis=1)))
    coeff = np.ascontiguousarray(np.asarray(coefficient, dtype=np.float32))
    receipt = FunctionalOverrideReceipt(
        block_id=asset.block_id,
        coefficient_sha256=_sha256(coeff.tobytes(order="C")),
        state_before_sha256=before,
        state_after_sha256=after,
        actual_checkpoint_sha256=binding.actual_checkpoint_sha256,
        phase1_seal_sha256=binding.phase1_seal_sha256,
        representation_rule_sha256=binding.representation_rule_sha256,
        runtime_binding_sha256=binding.binding_sha256,
        signed_rows=int(z.shape[0]) - positive_rows,
        positive_rows=positive_rows,
    )
    return FunctionalOverrideResult(batch=FABRForwardBatch(z, ids), receipt=receipt)


@dataclass(frozen=True, slots=True)
class FABRResourceReceipt:
    """Typed support-only resource receipt; backbone calls are counted separately."""

    active_k: int
    registered_class_count: int
    asset_numeric_payload_bytes: int
    dynamic_numeric_state_bytes: int
    support_fit_mac_equivalent: int
    base_support_forward_calls: int
    perturbation_support_forward_calls: int
    final_support_forward_calls: int
    query_rows_used_for_fit: int = 0
    query_state_updates: int = 0
    query_selection_count: int = 0
    query_gradient_calls: int = 0
    phase2_backward_calls: int = 0
    phase2_optimizer_steps: int = 0
    role_inputs: int = 0
    old_class_count_inputs: int = 0
    source_runtime_inputs: int = 0
    clean_runtime_inputs: int = 0
    truth_inputs: int = 0
    quota_inputs: int = 0
    global_reassignment_calls: int = 0
    schema: str = RESOURCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_SCHEMA or self.active_k not in ALLOWED_K or self.registered_class_count < 2:
            raise FABRError("FABR resource receipt schema/K/class drift")
        counters = (
            self.asset_numeric_payload_bytes,
            self.dynamic_numeric_state_bytes,
            self.support_fit_mac_equivalent,
            self.base_support_forward_calls,
            self.perturbation_support_forward_calls,
            self.final_support_forward_calls,
            self.query_rows_used_for_fit,
            self.query_state_updates,
            self.query_selection_count,
            self.query_gradient_calls,
            self.phase2_backward_calls,
            self.phase2_optimizer_steps,
            self.role_inputs,
            self.old_class_count_inputs,
            self.source_runtime_inputs,
            self.clean_runtime_inputs,
            self.truth_inputs,
            self.quota_inputs,
            self.global_reassignment_calls,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise FABRError("FABR resource counters must be nonnegative builtin ints")
        if (
            self.base_support_forward_calls != 1
            or self.perturbation_support_forward_calls != 4
            or self.final_support_forward_calls != 1
        ):
            raise FABRError("FABR support forward accounting must be exactly 1+4+1")

    @property
    def additional_support_forward_calls(self) -> int:
        return self.perturbation_support_forward_calls + self.final_support_forward_calls

    @property
    def protocol_closed(self) -> bool:
        return all(
            value == 0
            for value in (
                self.query_rows_used_for_fit,
                self.query_state_updates,
                self.query_selection_count,
                self.query_gradient_calls,
                self.phase2_backward_calls,
                self.phase2_optimizer_steps,
                self.role_inputs,
                self.old_class_count_inputs,
                self.source_runtime_inputs,
                self.clean_runtime_inputs,
                self.truth_inputs,
                self.quota_inputs,
                self.global_reassignment_calls,
            )
        )

    def as_dict(self) -> Mapping[str, Any]:
        value = {
            "schema": self.schema,
            "component": "FABR",
            "active_k": self.active_k,
            "registered_class_count": self.registered_class_count,
            "asset_numeric_payload_bytes": self.asset_numeric_payload_bytes,
            "dynamic_numeric_state_bytes": self.dynamic_numeric_state_bytes,
            "support_fit_mac_equivalent": self.support_fit_mac_equivalent,
            "base_support_forward_calls": self.base_support_forward_calls,
            "perturbation_support_forward_calls": self.perturbation_support_forward_calls,
            "final_support_forward_calls": self.final_support_forward_calls,
            "additional_support_forward_calls": self.additional_support_forward_calls,
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_state_updates": self.query_state_updates,
            "query_selection_count": self.query_selection_count,
            "query_gradient_calls": self.query_gradient_calls,
            "phase2_backward_calls": self.phase2_backward_calls,
            "phase2_optimizer_steps": self.phase2_optimizer_steps,
            "role_inputs": self.role_inputs,
            "old_class_count_inputs": self.old_class_count_inputs,
            "source_runtime_inputs": self.source_runtime_inputs,
            "clean_runtime_inputs": self.clean_runtime_inputs,
            "truth_inputs": self.truth_inputs,
            "quota_inputs": self.quota_inputs,
            "global_reassignment_calls": self.global_reassignment_calls,
            "protocol_closed": self.protocol_closed,
        }
        value["resource_receipt_sha256"] = _sha256(_canonical_bytes(value))
        return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class _SupportLayout:
    classes: tuple[str, ...]
    active_k: int
    order: np.ndarray
    grouped_indices: tuple[np.ndarray, ...]
    declared_support_physical_id_root_sha256: str
    support_physical_id_root_sha256: str
    class_physical_id_roots: Mapping[str, str]


def canonical_support_order(
    labels: Sequence[str], registered_classes: Sequence[str], physical_ids: Sequence[str]
) -> tuple[int, ...]:
    """Canonical order: frozen registry order, then physical-ID order within class."""

    classes = canonical_registered_classes(registered_classes)
    label_values = tuple(labels)
    ids = _require_physical_ids(physical_ids, expected_rows=len(label_values))
    if any(not isinstance(value, str) or value not in classes for value in label_values):
        raise FABRError("support labels must close exactly over registered classes")
    ordered: list[int] = []
    counts: list[int] = []
    for class_id in classes:
        indices = [index for index, label in enumerate(label_values) if label == class_id]
        if not indices:
            raise FABRError("each registered class requires support")
        indices.sort(key=lambda index: ids[index])
        ordered.extend(indices)
        counts.append(len(indices))
    if len(set(counts)) != 1 or counts[0] not in ALLOWED_K:
        raise FABRError("FABR requires balanced K1 or K5 physical support")
    return tuple(ordered)


def _build_layout(labels: Sequence[str], registered_classes: Sequence[str], physical_ids: Sequence[str]) -> _SupportLayout:
    classes = canonical_registered_classes(registered_classes)
    label_values = tuple(labels)
    ids = _require_physical_ids(physical_ids, expected_rows=len(label_values))
    order = np.asarray(canonical_support_order(label_values, classes, ids), dtype=np.int64)
    active_k = int(len(order) // len(classes))
    grouped = tuple(
        np.arange(index * active_k, (index + 1) * active_k, dtype=np.int64)
        for index in range(len(classes))
    )
    class_roots: dict[str, str] = {}
    for class_index, class_id in enumerate(classes):
        group = order[class_index * active_k : (class_index + 1) * active_k]
        class_roots[class_id] = _physical_id_root(tuple(ids[index] for index in group), class_id=class_id)
    return _SupportLayout(
        classes=classes,
        active_k=active_k,
        order=order,
        grouped_indices=grouped,
        declared_support_physical_id_root_sha256=_physical_id_root(ids),
        support_physical_id_root_sha256=_physical_id_root(tuple(ids[index] for index in order)),
        class_physical_id_roots=MappingProxyType(class_roots),
    )


def _ordered_unit_rows(rows: object, layout: _SupportLayout, *, name: str) -> np.ndarray:
    unit = _unit_rows(rows, name=name)
    return np.ascontiguousarray(unit[layout.order], dtype=np.float64)


def _support_root(unit_rows: np.ndarray, layout: _SupportLayout) -> str:
    payload = {
        "schema": STATE_SCHEMA,
        "registered_classes": layout.classes,
        "support_physical_id_root_sha256": layout.support_physical_id_root_sha256,
        "ordered_z160_sha256": _sha256(np.ascontiguousarray(unit_rows).tobytes(order="C")),
    }
    return _sha256(_canonical_bytes(payload) + b"\n" + np.ascontiguousarray(unit_rows).tobytes(order="C"))


def fabr_support_objective(unit_rows: object, *, labels: Sequence[str], registered_classes: Sequence[str], physical_ids: Sequence[str]) -> tuple[float, float, float]:
    """Evaluate the exact frozen ``L_sep + L_comp`` on canonical support rows."""

    layout = _build_layout(labels, registered_classes, physical_ids)
    rows = _ordered_unit_rows(unit_rows, layout, name="FABR objective rows")
    return _fabr_objective(rows, layout)


def _stable_softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _fabr_objective(unit_rows: np.ndarray, layout: _SupportLayout) -> tuple[float, float, float]:
    class_means = np.stack(
        [
            _normalise_vector(np.mean(unit_rows[group], axis=0, dtype=np.float64), name="FABR class mean")
            for group in layout.grouped_indices
        ],
        axis=0,
    )
    separation = 0.0
    class_count = len(layout.classes)
    for left in range(class_count):
        for right in range(left + 1, class_count):
            cosine = float(np.dot(class_means[left], class_means[right]))
            separation += TEMPERATURE_TAU * _stable_softplus(
                (cosine - MARGIN_M) / TEMPERATURE_TAU
            )
    separation *= 2.0 / float(class_count * (class_count - 1))
    compactness = 0.0
    if layout.active_k == 5:
        terms: list[float] = []
        for group in layout.grouped_indices:
            for local, row_index in enumerate(group):
                others = np.delete(group, local)
                loo = _normalise_vector(
                    np.mean(unit_rows[others], axis=0, dtype=np.float64),
                    name="FABR physical-LOO class mean",
                )
                terms.append(1.0 - float(np.dot(unit_rows[row_index], loo)))
        compactness = float(np.mean(terms, dtype=np.float64))
    elif layout.active_k != 1:
        raise FABRError("FABR active K drift")
    total = separation + compactness
    if not all(math.isfinite(value) for value in (total, separation, compactness)):
        raise FABRError("FABR support objective became non-finite")
    return float(total), float(separation), float(compactness)


def _check_positive_definite(matrix: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    condition = float(np.linalg.cond(symmetric))
    if (
        not np.isfinite(symmetric).all()
        or not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= 0.0
        or not math.isfinite(condition)
        or condition > MAX_CONDITION
    ):
        raise FABRError(f"{name} is non-positive-definite or ill-conditioned")
    return np.ascontiguousarray(symmetric, dtype=np.float64), condition


def _fisher_quadratic(coefficient: np.ndarray, fisher_k: np.ndarray) -> float:
    coeff = np.asarray(coefficient, dtype=np.float64)
    if coeff.shape != (RANK,) or not np.isfinite(coeff).all():
        raise FABRError("FABR coefficient must be finite [2]")
    value = float(coeff @ fisher_k @ coeff)
    if not math.isfinite(value) or value < 0.0:
        raise FABRError("FABR Fisher quadratic is invalid")
    return value


def _quantize_once_with_fisher_reprojection(coefficient: np.ndarray, fisher_k: np.ndarray) -> tuple[np.ndarray, float, int]:
    """FP16 RNE, then at most one radial FP16 re-projection as method-locked."""

    continuous = np.asarray(coefficient, dtype=np.float64)
    initial = _fisher_quadratic(continuous, fisher_k)
    if initial > RHO * RHO:
        continuous = continuous * math.sqrt((RHO * RHO) / initial)
    # NumPy's IEEE conversion is round-to-nearest-even for binary16.
    first_fp16 = np.ascontiguousarray(continuous.astype(_F16))
    first_fp32 = first_fp16.astype(np.float32)
    first_value = _fisher_quadratic(first_fp32, fisher_k)
    if first_value <= RHO * RHO:
        return first_fp16, first_value, 0
    scale = math.sqrt((RHO * RHO) / first_value)
    second_fp16 = np.ascontiguousarray((first_fp32 * np.float32(scale)).astype(_F16))
    second_fp32 = second_fp16.astype(np.float32)
    second_value = _fisher_quadratic(second_fp32, fisher_k)
    if second_value > RHO * RHO:
        raise FABRError("FABR FP16 one-reprojection still violates Fisher trust region")
    return second_fp16, second_value, 1


def _call_support_forward(
    forward_with_coeff: ForwardWithCoeff,
    support_token: Any,
    coefficient: np.ndarray,
    *,
    expected_rows: int | None,
    expected_ids: tuple[str, ...] | None,
    name: str,
) -> FABRForwardBatch:
    if not callable(forward_with_coeff):
        raise FABRError("FABR support forward must be callable")
    coeff = np.asarray(coefficient)
    if coeff.dtype != _F32 or coeff.shape != (RANK,) or not np.isfinite(coeff).all():
        raise FABRError("FABR callback coefficient must be finite float32 [2]")
    try:
        batch = forward_with_coeff(support_token, np.ascontiguousarray(coeff, dtype=np.float32))
    except Exception as exc:
        raise FABRError(f"FABR support callback failed during {name}") from exc
    if type(batch) is not FABRForwardBatch:
        raise FABRError("FABR callback must return an exact FABRForwardBatch")
    if batch.features.shape[0] < 2 or (expected_rows is not None and batch.features.shape[0] != expected_rows):
        raise FABRError(f"FABR callback {name} row count drift")
    if expected_ids is not None and batch.physical_ids != expected_ids:
        raise FABRError(f"FABR callback {name} physical IDs do not exactly match base order")
    return batch


@dataclass(frozen=True, slots=True)
class FABRState:
    """Immutable Phase2 support state: exactly two shared FP16 coefficients."""

    asset_sha256: str
    active_k: int
    registered_classes: tuple[str, ...]
    coeff_fp16: np.ndarray
    support_root_sha256: str
    support_receipt: Mapping[str, Any]
    resource_receipt: FABRResourceReceipt
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATE_SCHEMA or self.active_k not in ALLOWED_K:
            raise FABRError("FABR state schema/K drift")
        object.__setattr__(self, "asset_sha256", _require_sha256(self.asset_sha256, "asset_sha256"))
        object.__setattr__(self, "support_root_sha256", _require_sha256(self.support_root_sha256, "support_root_sha256"))
        classes = canonical_registered_classes(self.registered_classes)
        coeff = _readonly_exact(self.coeff_fp16, name="coeff_fp16", dtype=_F16, shape=(RANK,))
        if not np.any(coeff != np.float16(0.0)):
            raise FABRError("FABR state may not retain an exact zero coefficient")
        if type(self.resource_receipt) is not FABRResourceReceipt or not self.resource_receipt.protocol_closed:
            raise FABRError("FABR state requires a protocol-closed resource receipt")
        if self.resource_receipt.active_k != self.active_k or self.resource_receipt.registered_class_count != len(classes):
            raise FABRError("FABR state/resource receipt closure drift")
        if not isinstance(self.support_receipt, Mapping):
            raise FABRError("FABR support receipt must be a mapping")
        receipt = _deep_freeze(self.support_receipt)
        required = {
            "active_k",
            "registered_classes",
            "actual_checkpoint_sha256",
            "phase1_seal_sha256",
            "representation_rule_sha256",
            "runtime_binding_sha256",
            "support_physical_id_root_sha256",
            "gradient",
            "h_fabr",
            "fisher_k",
            "fisher_quadratic",
            "query_rows_used_for_fit",
            "query_state_updates",
            "query_selection_count",
        }
        if not required.issubset(receipt):
            raise FABRError("FABR support receipt is incomplete")
        if (
            receipt["active_k"] != self.active_k
            or tuple(receipt["registered_classes"]) != classes
            or receipt["representation_rule_sha256"] != REPRESENTATION_RULE_SHA256
            or any(receipt[key] != 0 for key in ("query_rows_used_for_fit", "query_state_updates", "query_selection_count"))
        ):
            raise FABRError("FABR support receipt protocol closure drift")
        for name in (
            "actual_checkpoint_sha256",
            "phase1_seal_sha256",
            "representation_rule_sha256",
            "runtime_binding_sha256",
        ):
            _require_sha256(receipt[name], name)
        k = np.asarray(receipt["fisher_k"], dtype=np.float64)
        value = _fisher_quadratic(coeff.astype(np.float32), k)
        if k.shape != (RANK, RANK) or value > RHO * RHO:
            raise FABRError("FABR state Fisher trust closure drift")
        object.__setattr__(self, "registered_classes", classes)
        object.__setattr__(self, "coeff_fp16", coeff)
        object.__setattr__(self, "support_receipt", receipt)

    @property
    def coeff_float32(self) -> np.ndarray:
        value = np.array(self.coeff_fp16, dtype=np.float32, copy=True, order="C")
        value.setflags(write=False)
        return value


def fit_fabr_support(
    asset: FABRAsset,
    support_token: Any,
    labels: Sequence[str],
    registered_classes: Sequence[str],
    forward_with_coeff: ForwardWithCoeff,
    *,
    support_physical_ids: Sequence[str],
    runtime_binding: FABRRuntimeBinding | None = None,
) -> FABRState:
    """Fit the closed-form FABR coefficient from support only.

    The callback is invoked exactly at ``0``, ``±delta e0``, ``±delta e1`` and
    the final FP16 coefficient.  Query, truth, role, old/new count, quota,
    source/clean data, optimiser, and backward inputs are absent from this API.
    """

    if type(asset) is not FABRAsset:
        raise FABRError("FABR support fit requires an exact FABRAsset")
    binding = _require_runtime_binding(asset, runtime_binding)
    try:
        label_count = len(labels)
    except TypeError as exc:
        raise FABRError("support labels must be a sized sequence") from exc
    physical_ids = _require_physical_ids(support_physical_ids, expected_rows=label_count)
    zero = np.zeros(RANK, dtype=np.float32)
    base = _call_support_forward(
        forward_with_coeff, support_token, zero, expected_rows=None, expected_ids=physical_ids, name="base"
    )
    layout = _build_layout(labels, registered_classes, physical_ids)
    if base.features.shape[0] != label_count:
        raise FABRError("base support forward row count does not close labels")
    base_unit = _ordered_unit_rows(base.features, layout, name="base signed-pre-ReLU160")
    base_total, base_sep, base_comp = _fabr_objective(base_unit, layout)

    plus_units: list[np.ndarray] = []
    minus_units: list[np.ndarray] = []
    gradient = np.empty(RANK, dtype=np.float64)
    for direction in range(RANK):
        plus = np.zeros(RANK, dtype=np.float32)
        minus = np.zeros(RANK, dtype=np.float32)
        plus[direction] = np.float32(DELTA)
        minus[direction] = np.float32(-DELTA)
        plus_batch = _call_support_forward(
            forward_with_coeff,
            support_token,
            plus,
            expected_rows=base.features.shape[0],
            expected_ids=base.physical_ids,
            name=f"plus_direction_{direction}",
        )
        minus_batch = _call_support_forward(
            forward_with_coeff,
            support_token,
            minus,
            expected_rows=base.features.shape[0],
            expected_ids=base.physical_ids,
            name=f"minus_direction_{direction}",
        )
        plus_unit = _ordered_unit_rows(plus_batch.features, layout, name=f"plus signed-pre-ReLU160 {direction}")
        minus_unit = _ordered_unit_rows(minus_batch.features, layout, name=f"minus signed-pre-ReLU160 {direction}")
        plus_loss, _plus_sep, _plus_comp = _fabr_objective(plus_unit, layout)
        minus_loss, _minus_sep, _minus_comp = _fabr_objective(minus_unit, layout)
        gradient[direction] = (plus_loss - minus_loss) / (2.0 * DELTA)
        plus_units.append(plus_unit)
        minus_units.append(minus_unit)

    # Shape is [N,160,2] after the documented canonical class/physical flatten.
    jacobian = np.stack(
        [(plus_units[index] - minus_units[index]) / (2.0 * DELTA) for index in range(RANK)], axis=2
    )
    h_fabr = np.einsum("ndr,nds->rs", jacobian, jacobian, optimize=True) / float(base.features.shape[0])
    h_fabr = 0.5 * (h_fabr + h_fabr.T)
    if not np.isfinite(gradient).all() or not np.isfinite(h_fabr).all():
        raise FABRError("FABR central-difference g/J/H_FABR became non-finite")
    fisher_k = decode_fabr_fisher_k(asset)
    system, condition = _check_positive_definite(
        h_fabr + LAMBDA_F * fisher_k + LAMBDA_0 * np.eye(RANK, dtype=np.float64),
        name="H_FABR + lambda_F*K_F + lambda_0*I",
    )
    try:
        continuous = -np.linalg.solve(system, gradient)
    except np.linalg.LinAlgError as exc:
        raise FABRError("FABR closed-form K_F solve failed") from exc
    if not np.isfinite(continuous).all():
        raise FABRError("FABR closed-form coefficient became non-finite")
    coeff_fp16, final_quadratic, fp16_reprojection_count = _quantize_once_with_fisher_reprojection(
        continuous, fisher_k
    )
    coeff_f32 = coeff_fp16.astype(np.float32)
    if not np.any(coeff_fp16 != np.float16(0.0)) or float(np.max(np.abs(coeff_f32))) <= NOISE_FACTOR * _F32_EPS:
        _reject_no_function("FP16 coefficient is numerically zero")

    final = _call_support_forward(
        forward_with_coeff,
        support_token,
        coeff_f32,
        expected_rows=base.features.shape[0],
        expected_ids=base.physical_ids,
        name="final_quantized",
    )
    final_unit = _ordered_unit_rows(final.features, layout, name="final signed-pre-ReLU160")
    final_total, final_sep, final_comp = _fabr_objective(final_unit, layout)
    feature_delta = float(np.max(np.abs(final_unit - base_unit)))
    gram_delta = float(np.max(np.abs((final_unit @ final_unit.T) - (base_unit @ base_unit.T))))
    jitter = max(float(asset.forward_jitter_tolerance_fp16[0]), NOISE_FACTOR * _F32_EPS)
    if feature_delta <= jitter:
        _reject_no_function("final feature change does not exceed repeated-forward jitter")
    if gram_delta <= jitter:
        _reject_no_function("final Gram change does not exceed repeated-forward jitter")

    support_receipt = _deep_freeze(
        {
            "schema": STATE_SCHEMA,
            "actual_checkpoint_sha256": binding.actual_checkpoint_sha256,
            "phase1_seal_sha256": binding.phase1_seal_sha256,
            "representation_rule_sha256": binding.representation_rule_sha256,
            "runtime_binding_sha256": binding.binding_sha256,
            "active_k": layout.active_k,
            "registered_classes": layout.classes,
            "declared_support_physical_id_root_sha256": layout.declared_support_physical_id_root_sha256,
            "support_physical_id_root_sha256": layout.support_physical_id_root_sha256,
            "class_physical_id_roots": dict(layout.class_physical_id_roots),
            "base_support_z160_sha256": _sha256(np.ascontiguousarray(base_unit, dtype=np.float32).tobytes(order="C")),
            "final_support_z160_sha256": _sha256(np.ascontiguousarray(final_unit, dtype=np.float32).tobytes(order="C")),
            "physical_loo_compactness_active": layout.active_k == 5,
            "objective_base": base_total,
            "objective_final": final_total,
            "class_separation_base": base_sep,
            "class_separation_final": final_sep,
            "physical_loo_compactness_base": base_comp,
            "physical_loo_compactness_final": final_comp,
            "central_difference_delta": DELTA,
            "gradient": tuple(float(value) for value in gradient),
            "h_fabr": tuple(tuple(float(value) for value in row) for row in h_fabr),
            "fisher_k": tuple(tuple(float(value) for value in row) for row in fisher_k),
            "system_condition": condition,
            "fisher_quadratic": final_quadratic,
            "fp16_reprojection_count": fp16_reprojection_count,
            "trust_radius": RHO,
            "base_final_feature_max_abs_delta": feature_delta,
            "base_final_gram_max_abs_delta": gram_delta,
            "forward_jitter_tolerance": jitter,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
            "query_gradient_calls": 0,
            "phase2_backward_calls": 0,
            "phase2_optimizer_steps": 0,
            "role_inputs": 0,
            "old_class_count_inputs": 0,
            "source_runtime_inputs": 0,
            "clean_runtime_inputs": 0,
            "truth_inputs": 0,
            "quota_inputs": 0,
            "global_reassignment_calls": 0,
        }
    )
    class_pairs = len(layout.classes) * (len(layout.classes) - 1) // 2
    # Numeric z-space arithmetic only; real backbone functional forwards are
    # separately counted above and never converted into invented MAC claims.
    support_macs = int(8 * base.features.shape[0] * Z_DIM + 2 * base.features.shape[0] * RANK + class_pairs * Z_DIM)
    resource = FABRResourceReceipt(
        active_k=layout.active_k,
        registered_class_count=len(layout.classes),
        asset_numeric_payload_bytes=asset.numeric_payload_bytes,
        dynamic_numeric_state_bytes=int(coeff_fp16.nbytes),
        support_fit_mac_equivalent=support_macs,
        base_support_forward_calls=1,
        perturbation_support_forward_calls=4,
        final_support_forward_calls=1,
    )
    return FABRState(
        asset_sha256=fabr_asset_sha256(asset),
        active_k=layout.active_k,
        registered_classes=layout.classes,
        coeff_fp16=coeff_fp16,
        support_root_sha256=_support_root(base_unit, layout),
        support_receipt=support_receipt,
        resource_receipt=resource,
    )


__all__ = [
    "ALLOWED_K",
    "ASSET_SCHEMA",
    "BLOCK_DIMENSIONS",
    "BLOCK_PARAMETER_KEYS",
    "BLOCK_TIE_ORDER",
    "DELTA",
    "FABRAsset",
    "FABRError",
    "FABRForwardBatch",
    "FABRNoFunctionError",
    "FABRResourceReceipt",
    "FABRRuntimeBinding",
    "FABRState",
    "FABRTieError",
    "FunctionalOverrideReceipt",
    "FunctionalOverrideResult",
    "ForwardWithCoeff",
    "LAMBDA_0",
    "LAMBDA_F",
    "MARGIN_M",
    "MAX_CONDITION",
    "NOISE_FACTOR",
    "Phase1FisherGeometry",
    "RANK",
    "REPRESENTATION_RULE_ID",
    "REPRESENTATION_RULE_SHA256",
    "RESOURCE_SCHEMA",
    "RHO",
    "SCHEMA",
    "STATE_SCHEMA",
    "TEMPERATURE_TAU",
    "Z_DIM",
    "canonical_parameter_keys",
    "canonical_parameter_layout",
    "canonical_registered_classes",
    "canonical_support_order",
    "decode_fabr_basis",
    "decode_fabr_fisher_k",
    "fabr_asset_sha256",
    "fabr_support_objective",
    "fit_fabr_support",
    "flatten_canonical_parameter_block",
    "functional_forward_signed_pre_relu160",
    "functional_override_parameter_map",
    "phase1_fisher_geometry",
    "require_exact_logit_alias",
    "serialize_fabr_asset",
    "signed_pre_relu160",
    "strict_top1_predictions",
]
