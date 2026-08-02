"""D113 BCAT: support-only common-shift canonicalization for qKNN.

The fitted state reads a sealed Phase1 aggregate and one labelled support
bank. Queries are accepted only by the two scoring functions and never alter
the fitted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


FEATURE_DIM = 160
OLD_CLASS_COUNT = 6
EPSILON = 64.0 * float(np.finfo(np.float32).eps)
SCHEMA = "cvs.stage2.d113.bcat_qknn.v1"


class D113BCATError(ValueError):
    """Raised when the frozen D113 numerical or protocol contract drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _unit_rows(value: np.ndarray, *, field: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_DIM or not np.isfinite(rows).all():
        raise D113BCATError(f"{field} must be finite [N,{FEATURE_DIM}]")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= EPSILON):
        raise D113BCATError(f"{field} contains a degenerate row")
    return rows / norms


@dataclass(frozen=True, slots=True)
class D113Bundle:
    """Minimal immutable Phase1 aggregate; it contains no source rows."""

    class_registry: tuple[str, ...]
    ground: np.ndarray
    sigma0: np.ndarray
    v_ground: np.ndarray
    quantization_mse: np.ndarray
    tau_b2: float
    checkpoint_sha256: str
    source_aggregate_sha256: str
    allowed_config_lock_digests: tuple[str, ...]
    schema: str = SCHEMA
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        registry = tuple(str(value) for value in self.class_registry)
        if len(registry) != OLD_CLASS_COUNT or len(set(registry)) != OLD_CLASS_COUNT:
            raise D113BCATError("D113 requires six unique Phase1 old classes")
        ground = _unit_rows(self.ground, field="ground")
        if ground.shape != (OLD_CLASS_COUNT, FEATURE_DIM):
            raise D113BCATError("ground must be [6,160]")
        vectors = []
        for name in ("sigma0", "v_ground", "quantization_mse"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                value.shape != (OLD_CLASS_COUNT,)
                or not np.isfinite(value).all()
                or np.any(value < 0.0)
                or (name != "quantization_mse" and np.any(value <= 0.0))
            ):
                raise D113BCATError(f"{name} must be a valid six-class variance vector")
            vectors.append(value)
        if not math.isfinite(float(self.tau_b2)) or float(self.tau_b2) <= 0.0:
            raise D113BCATError("tau_b2 must be finite and positive")
        for name in ("checkpoint_sha256", "source_aggregate_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise D113BCATError(f"{name} must be a lowercase SHA256")
        lock_digests = tuple(str(value) for value in self.allowed_config_lock_digests)
        if (
            len(lock_digests) != 3
            or len(set(lock_digests)) != 3
            or any(
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in lock_digests
            )
        ):
            raise D113BCATError("D113 bundle requires three unique K-specific lock digests")
        object.__setattr__(self, "class_registry", registry)
        object.__setattr__(self, "ground", _readonly(ground, np.float32))
        object.__setattr__(self, "sigma0", _readonly(vectors[0], np.float64))
        object.__setattr__(self, "v_ground", _readonly(vectors[1], np.float64))
        object.__setattr__(self, "quantization_mse", _readonly(vectors[2], np.float64))
        digest = hashlib.sha256()
        digest.update(self.schema.encode("ascii"))
        for name in registry:
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        for value in (self.ground, self.sigma0, self.v_ground, self.quantization_mse):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        digest.update(np.asarray([self.tau_b2], dtype="<f8").tobytes())
        digest.update(self.checkpoint_sha256.encode("ascii"))
        digest.update(self.source_aggregate_sha256.encode("ascii"))
        for value in lock_digests:
            digest.update(value.encode("ascii"))
        object.__setattr__(self, "allowed_config_lock_digests", lock_digests)
        object.__setattr__(self, "content_sha256", digest.hexdigest())


def build_d113_bundle(
    *,
    class_registry: Sequence[str],
    ground: np.ndarray,
    sigma0: np.ndarray,
    v_ground: np.ndarray,
    quantization_mse: np.ndarray,
    tau_b2: float,
    checkpoint_sha256: str,
    source_aggregate_sha256: str,
    allowed_config_lock_digests: Sequence[str],
) -> D113Bundle:
    return D113Bundle(
        class_registry=tuple(class_registry),
        ground=np.asarray(ground),
        sigma0=np.asarray(sigma0),
        v_ground=np.asarray(v_ground),
        quantization_mse=np.asarray(quantization_mse),
        tau_b2=float(tau_b2),
        checkpoint_sha256=str(checkpoint_sha256),
        source_aggregate_sha256=str(source_aggregate_sha256),
        allowed_config_lock_digests=tuple(str(value) for value in allowed_config_lock_digests),
    )


def bcat_inverse(value: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact inverse of x=normalize(u+b) for the effective shift ||b||<1/2."""

    rows = _unit_rows(value, field="BCAT input")
    shift = np.asarray(b, dtype=np.float64)
    if shift.shape != (FEATURE_DIM,) or not np.isfinite(shift).all():
        raise D113BCATError("b must be finite [160]")
    norm2 = float(np.dot(shift, shift))
    if norm2 >= 0.25:
        raise D113BCATError("BCAT effective shift must have norm below one half")
    inner = rows @ shift
    radicand = np.square(inner) + 1.0 - norm2
    if np.any(radicand < 0.75 - 1.0e-12):
        raise D113BCATError("BCAT inverse lost its fixed radicand bound")
    scale = inner + np.sqrt(radicand)
    result = scale[:, None] * rows - shift[None, :]
    # Algebraically unit length; normalization only removes round-off.
    return _unit_rows(result, field="BCAT inverse output")


def _shrinkage_jacobian(delta: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm <= EPSILON:
        return np.eye(FEATURE_DIM, dtype=np.float64)
    denominator = 1.0 + 2.0 * norm
    return (
        np.eye(FEATURE_DIM, dtype=np.float64) / denominator
        - 2.0 * np.outer(delta, delta) / (norm * denominator * denominator)
    )


def _inverse_shift_jacobian(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    inner = float(np.dot(x, b))
    root = math.sqrt(inner * inner + 1.0 - float(np.dot(b, b)))
    grad_scale = x + (inner * x - b) / root
    return np.outer(x, grad_scale) - np.eye(FEATURE_DIM, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class D113State:
    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    transformed_bank: TypedINT8ZIDSupportBank
    delta_hat: np.ndarray
    b: np.ndarray
    rho: np.ndarray
    v_shift: np.ndarray
    information_weight: np.ndarray
    raw_bank_receipt_sha256: str
    bundle_content_sha256: str
    bundle_bank_pair_sha256: str
    resource_receipt: Mapping[str, int]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        count = len(self.classes)
        if (
            self.schema != SCHEMA
            or len(self.old_class_indices) != OLD_CLASS_COUNT
            or self.delta_hat.shape != (FEATURE_DIM,)
            or self.b.shape != (FEATURE_DIM,)
            or self.rho.shape != (count,)
            or self.v_shift.shape != (count,)
            or self.information_weight.shape != (count,)
        ):
            raise D113BCATError("D113 state shape/schema drift")
        if any(
            value.flags.writeable
            for value in (self.delta_hat, self.b, self.rho, self.v_shift, self.information_weight)
        ):
            raise D113BCATError("D113 state arrays must be readonly")
        if not isinstance(self.resource_receipt, Mapping):
            raise D113BCATError("D113 resource receipt must be a mapping")


def fit_d113_state(bundle: D113Bundle, bank: TypedINT8ZIDSupportBank) -> D113State:
    """Fit one immutable state from Phase1 assets and labelled support only."""

    if type(bundle) is not D113Bundle or type(bank) is not TypedINT8ZIDSupportBank:
        raise D113BCATError("D113 fit requires exact bundle and support-bank types")
    if float(bank.config.kernel_volume_gamma) != 1.0:
        raise D113BCATError("D113 unit-mass head requires kernel_volume_gamma=1")
    if bank.config_lock_digest not in bundle.allowed_config_lock_digests:
        raise D113BCATError("D113 bundle/support-bank config lineage mismatch")
    if any(name not in bank.classes for name in bundle.class_registry):
        raise D113BCATError("D113 old registry is not contained in the support bank")

    old_indices = tuple(bank.classes.index(name) for name in bundle.class_registry)
    decoded = normalize_zid_rows(decode_zid_support_bank(bank)).astype(np.float64)
    ground = np.asarray(bundle.ground, dtype=np.float64)
    identity = np.eye(FEATURE_DIM, dtype=np.float64)
    precision = 1.0 / float(bundle.tau_b2)
    system = precision * identity
    rhs = np.zeros(FEATURE_DIM, dtype=np.float64)
    information_weight = np.zeros(len(bank.classes), dtype=np.float64)
    raw_prototypes = np.zeros((OLD_CLASS_COUNT, FEATURE_DIM), dtype=np.float64)

    for position, class_index in enumerate(old_indices):
        local = decoded[bank.class_indices_int16 == class_index]
        prototype = _unit_rows(np.sum(local, axis=0, keepdims=True), field="old support sum")[0]
        raw_prototypes[position] = prototype
        if bank.active_k == 1:
            empirical = 0.0
        else:
            empirical = float(
                np.sum(np.square(local - prototype[None, :]))
                / ((bank.active_k - 1) * FEATURE_DIM)
            )
        variance = (
            float(bundle.v_ground[position])
            + float(bundle.quantization_mse[position])
            + (float(bundle.sigma0[position]) + empirical) / bank.active_k
        )
        if not math.isfinite(variance) or variance <= 0.0:
            raise D113BCATError("D113 support information variance is invalid")
        weight = 1.0 / variance
        projector = identity - np.outer(ground[position], ground[position])
        system += weight * projector
        rhs += weight * (projector @ (prototype - ground[position]))
        information_weight[class_index] = weight

    try:
        delta = np.linalg.solve(system, rhs)
        covariance_delta = np.linalg.inv(system)
    except np.linalg.LinAlgError as exc:
        raise D113BCATError("D113 Bayesian normal system is not solvable") from exc
    if not np.isfinite(delta).all() or not np.isfinite(covariance_delta).all():
        raise D113BCATError("D113 common-shift estimate is non-finite")
    delta_norm = float(np.linalg.norm(delta))
    b = delta / (1.0 + 2.0 * delta_norm)
    if float(np.linalg.norm(b)) >= 0.5:
        raise D113BCATError("D113 radial shrinkage lost its bound")

    transformed = bcat_inverse(decoded, b)
    labels = [bank.classes[int(index)] for index in bank.class_indices_int16]
    transformed_bank = build_typed_zid_support_bank(
        transformed.astype(np.float32), labels, bank.classes, config=bank.config
    )

    rho = np.zeros(len(bank.classes), dtype=np.float64)
    v_shift = np.zeros(len(bank.classes), dtype=np.float64)
    jacobian_b = _shrinkage_jacobian(delta)
    covariance_b = jacobian_b @ covariance_delta @ jacobian_b.T
    transformed_decoded = decode_zid_support_bank(transformed_bank).astype(np.float64)
    for position, class_index in enumerate(old_indices):
        local = transformed_decoded[transformed_bank.class_indices_int16 == class_index]
        prototype = _unit_rows(np.sum(local, axis=0, keepdims=True), field="canonical support sum")[0]
        if bank.active_k == 1:
            empirical = 0.0
        else:
            empirical = float(
                np.sum(np.square(local - prototype[None, :]))
                / ((bank.active_k - 1) * FEATURE_DIM)
            )
        v_support = (float(bundle.sigma0[position]) + empirical) / bank.active_k
        h_shift = _inverse_shift_jacobian(raw_prototypes[position], b)
        local_v_shift = float(np.trace(h_shift @ covariance_b @ h_shift.T) / FEATURE_DIM)
        discrepancy = float(np.sum(np.square(prototype - ground[position])) / FEATURE_DIM)
        denominator = (
            v_support
            + float(bundle.v_ground[position])
            + float(bundle.quantization_mse[position])
            + local_v_shift
            + discrepancy
        )
        local_rho = v_support / denominator
        if not math.isfinite(local_rho) or not 0.0 < local_rho < 1.0:
            raise D113BCATError("D113 ground-head unit mass is invalid")
        rho[class_index] = local_rho
        v_shift[class_index] = local_v_shift

    resource = MappingProxyType(
        {
            "persistent_numeric_bytes": int(
                delta.nbytes + b.nbytes + rho.nbytes + v_shift.nbytes + information_weight.nbytes
            ),
            "enrollment_dense_system_dimension": FEATURE_DIM,
            "enrollment_dense_solve_count": 2,
            "extra_query_macs_da_upper_bound": 2 * FEATURE_DIM,
            "extra_query_macs_joint_upper_bound": (2 + OLD_CLASS_COUNT) * FEATURE_DIM,
            "query_dependent_state_bytes": 0,
        }
    )
    return D113State(
        classes=tuple(bank.classes),
        old_class_indices=old_indices,
        transformed_bank=transformed_bank,
        delta_hat=_readonly(delta, np.float64),
        b=_readonly(b, np.float64),
        rho=_readonly(rho, np.float64),
        v_shift=_readonly(v_shift, np.float64),
        information_weight=_readonly(information_weight, np.float64),
        raw_bank_receipt_sha256=bank.bank_receipt_sha256,
        bundle_content_sha256=bundle.content_sha256,
        bundle_bank_pair_sha256=hashlib.sha256(
            (bundle.content_sha256 + bank.bank_receipt_sha256).encode("ascii")
        ).hexdigest(),
        resource_receipt=resource,
    )


def _verify_score_inputs(state: D113State, bank: TypedINT8ZIDSupportBank) -> None:
    if type(state) is not D113State or type(bank) is not TypedINT8ZIDSupportBank:
        raise D113BCATError("D113 score requires exact state and bank types")
    if (
        state.classes != tuple(bank.classes)
        or state.raw_bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.transformed_bank.config_lock_digest != bank.config_lock_digest
        or state.bundle_bank_pair_sha256
        != hashlib.sha256(
            (state.bundle_content_sha256 + bank.bank_receipt_sha256).encode("ascii")
        ).hexdigest()
    ):
        raise D113BCATError("D113 state/support-bank binding drift")


def score_d113_da_logits(
    state: D113State, bank: TypedINT8ZIDSupportBank, query_zid: np.ndarray
) -> np.ndarray:
    """M_DA: score independent canonicalized queries with the base qKNN head."""

    _verify_score_inputs(state, bank)
    canonical_query = bcat_inverse(query_zid, state.b).astype(np.float32)
    metric = identity_shared_psd_metric(config=bank.config)
    return score_zid_student_t_logits(
        state.transformed_bank, canonical_query, metric=metric
    )


def score_d113_joint_logits(
    state: D113State,
    bundle: D113Bundle,
    bank: TypedINT8ZIDSupportBank,
    query_zid: np.ndarray,
) -> np.ndarray:
    """M_JOINT: M_DA plus a unit-mass canonical ground expert for old classes."""

    _verify_score_inputs(state, bank)
    if (
        type(bundle) is not D113Bundle
        or bundle.content_sha256 != state.bundle_content_sha256
        or tuple(bundle.class_registry)
        != tuple(state.classes[index] for index in state.old_class_indices)
    ):
        raise D113BCATError("D113 joint score bundle/registry drift")
    canonical_query = bcat_inverse(query_zid, state.b).astype(np.float64)
    output = np.asarray(score_d113_da_logits(state, bank, query_zid), dtype=np.float64).copy()
    dimension = bank.config.kernel_effective_dim
    nu = float(bank.config.student_nu)
    ground = np.asarray(bundle.ground, dtype=np.float64)
    for position, class_index in enumerate(state.old_class_indices):
        local_rho = float(state.rho[class_index])
        cosine = np.clip(canonical_query @ ground[position], -1.0, 1.0)
        distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
        h = float(state.transformed_bank.class_scales_fp16[class_index])
        anchor = (
            -dimension * math.log(h)
            - 0.5 * (nu + dimension) * np.log1p(distance / (nu * h * h))
        )
        output[:, class_index] = np.logaddexp(
            math.log1p(-local_rho) + output[:, class_index],
            math.log(local_rho) + anchor,
        )
    if not np.isfinite(output).all():
        raise D113BCATError("D113 joint logits became non-finite")
    return _readonly(output, np.float32)


def audit_d113_state(state: D113State) -> Mapping[str, Any]:
    if type(state) is not D113State:
        raise D113BCATError("D113 audit requires an exact state")
    old = np.asarray(state.old_class_indices, dtype=np.int64)
    return MappingProxyType(
        {
            "schema": state.schema,
            "delta_norm": float(np.linalg.norm(state.delta_hat)),
            "b_norm": float(np.linalg.norm(state.b)),
            "positive_b_coordinates": int(np.sum(np.abs(state.b) > 0.0)),
            "positive_rho_count": int(np.sum(state.rho[old] > 0.0)),
            "max_rho": float(np.max(state.rho)),
            "max_v_shift": float(np.max(state.v_shift)),
            "transformed_bank_receipt_sha256": state.transformed_bank.bank_receipt_sha256,
            "resource_receipt": state.resource_receipt,
            "query_rows_used_for_fit": 0,
            "truth_role_quota_inputs": 0,
        }
    )


__all__ = [
    "D113BCATError",
    "D113Bundle",
    "D113State",
    "audit_d113_state",
    "bcat_inverse",
    "build_d113_bundle",
    "fit_d113_state",
    "score_d113_da_logits",
    "score_d113_joint_logits",
]
