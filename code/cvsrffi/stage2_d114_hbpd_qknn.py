"""D114 heteroscedastic Bayesian predictive-density qKNN."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_zid_student_t_qknn import (
    TypedINT8ZIDSupportBank,
    decode_zid_support_bank,
    normalize_zid_rows,
)


FEATURE_DIM = 160
OLD_CLASS_COUNT = 6
SCHEMA = "cvs.stage2.d114.hbpd_qknn.v1"


class D114HBPDError(ValueError):
    """Raised when the frozen HBPD numerical or lineage contract drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class D114Bundle:
    class_registry: tuple[str, ...]
    sigma0_old: np.ndarray
    sigma0_pooled: float
    checkpoint_sha256: str
    source_aggregate_sha256: str
    allowed_config_lock_digests: tuple[str, ...]
    schema: str = SCHEMA
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        registry = tuple(str(value) for value in self.class_registry)
        sigma = np.asarray(self.sigma0_old, dtype=np.float64)
        pooled = float(self.sigma0_pooled)
        locks = tuple(str(value) for value in self.allowed_config_lock_digests)
        if len(registry) != OLD_CLASS_COUNT or len(set(registry)) != OLD_CLASS_COUNT:
            raise D114HBPDError("D114 requires six unique Phase1 old classes")
        if sigma.shape != (OLD_CLASS_COUNT,) or not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise D114HBPDError("sigma0_old must be six finite positive values")
        if not math.isfinite(pooled) or pooled <= 0.0 or not math.isclose(
            pooled, float(np.mean(sigma)), rel_tol=1.0e-12, abs_tol=1.0e-15
        ):
            raise D114HBPDError("sigma0_pooled must equal the old-class arithmetic mean")
        for name in ("checkpoint_sha256", "source_aggregate_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise D114HBPDError(f"{name} must be a lowercase SHA256")
        if (
            len(locks) != 3
            or len(set(locks)) != 3
            or any(
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in locks
            )
        ):
            raise D114HBPDError("D114 requires three unique K-specific lock digests")
        object.__setattr__(self, "class_registry", registry)
        object.__setattr__(self, "sigma0_old", _readonly(sigma, np.float64))
        object.__setattr__(self, "allowed_config_lock_digests", locks)
        digest = hashlib.sha256()
        digest.update(self.schema.encode("ascii"))
        for name in registry:
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        digest.update(np.ascontiguousarray(self.sigma0_old).tobytes(order="C"))
        digest.update(np.asarray([pooled], dtype="<f8").tobytes())
        digest.update(self.checkpoint_sha256.encode("ascii"))
        digest.update(self.source_aggregate_sha256.encode("ascii"))
        for value in locks:
            digest.update(value.encode("ascii"))
        object.__setattr__(self, "content_sha256", digest.hexdigest())


def build_d114_bundle(
    *,
    class_registry: Sequence[str],
    sigma0_old: np.ndarray,
    checkpoint_sha256: str,
    source_aggregate_sha256: str,
    allowed_config_lock_digests: Sequence[str],
) -> D114Bundle:
    sigma = np.asarray(sigma0_old, dtype=np.float64)
    return D114Bundle(
        class_registry=tuple(class_registry),
        sigma0_old=sigma,
        sigma0_pooled=float(np.mean(sigma)),
        checkpoint_sha256=str(checkpoint_sha256),
        source_aggregate_sha256=str(source_aggregate_sha256),
        allowed_config_lock_digests=tuple(allowed_config_lock_digests),
    )


@dataclass(frozen=True, slots=True)
class D114State:
    classes: tuple[str, ...]
    old_class_indices: tuple[int, ...]
    sigma_prior: np.ndarray
    sigma_target: np.ndarray
    sigma_posterior: np.ndarray
    predictive_bandwidth: np.ndarray
    bank_receipt_sha256: str
    bundle_content_sha256: str
    bundle_bank_pair_sha256: str
    config_lock_digest: str
    resource_receipt: Mapping[str, int]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        expected = (len(self.classes),)
        if (
            self.schema != SCHEMA
            or len(self.old_class_indices) != OLD_CLASS_COUNT
            or any(
                value.shape != expected
                for value in (
                    self.sigma_prior,
                    self.sigma_target,
                    self.sigma_posterior,
                    self.predictive_bandwidth,
                )
            )
        ):
            raise D114HBPDError("D114 state shape/schema drift")
        arrays = (
            self.sigma_prior,
            self.sigma_target,
            self.sigma_posterior,
            self.predictive_bandwidth,
        )
        if any(
            value.flags.writeable
            or not np.isfinite(value).all()
            or np.any(value < 0.0)
            for value in arrays
        ) or np.any(self.predictive_bandwidth <= 0.0):
            raise D114HBPDError("D114 state numeric/readonly invariant drift")


def fit_d114_state(bundle: D114Bundle, bank: TypedINT8ZIDSupportBank) -> D114State:
    """Fit class predictive bandwidths from sealed variances and support only."""

    if type(bundle) is not D114Bundle or type(bank) is not TypedINT8ZIDSupportBank:
        raise D114HBPDError("D114 fit requires exact bundle and support-bank types")
    if float(bank.config.kernel_volume_gamma) != 1.0:
        raise D114HBPDError("D114 density normalization requires kernel_volume_gamma=1")
    if bank.config_lock_digest not in bundle.allowed_config_lock_digests:
        raise D114HBPDError("D114 bundle/support-bank config lineage mismatch")
    if any(name not in bank.classes for name in bundle.class_registry):
        raise D114HBPDError("D114 old registry is not contained in the support bank")
    old_indices = tuple(bank.classes.index(name) for name in bundle.class_registry)
    old_map = {index: position for position, index in enumerate(old_indices)}
    decoded = decode_zid_support_bank(bank).astype(np.float64)
    count = len(bank.classes)
    prior = np.full(count, float(bundle.sigma0_pooled), dtype=np.float64)
    target = np.zeros(count, dtype=np.float64)
    posterior = np.zeros(count, dtype=np.float64)
    bandwidth = np.zeros(count, dtype=np.float64)
    for class_index in range(count):
        if class_index in old_map:
            prior[class_index] = float(bundle.sigma0_old[old_map[class_index]])
        local = decoded[bank.class_indices_int16 == class_index]
        if len(local) != bank.active_k:
            raise D114HBPDError("D114 support count drift")
        if bank.active_k > 1:
            total = np.sum(local, axis=0)
            norm = float(np.linalg.norm(total))
            if not math.isfinite(norm) or norm <= 0.0:
                raise D114HBPDError("D114 support prototype is degenerate")
            prototype = total / norm
            target[class_index] = float(
                np.sum(np.square(local - prototype[None, :]))
                / ((bank.active_k - 1) * FEATURE_DIM)
            )
        posterior[class_index] = (
            prior[class_index] + (bank.active_k - 1) * target[class_index]
        ) / bank.active_k
        bandwidth_squared = 2.0 * FEATURE_DIM * posterior[class_index]
        if not math.isfinite(bandwidth_squared) or bandwidth_squared <= 0.0:
            raise D114HBPDError("D114 predictive bandwidth is invalid")
        bandwidth[class_index] = math.sqrt(bandwidth_squared)
    pair = hashlib.sha256(
        (bundle.content_sha256 + bank.bank_receipt_sha256).encode("ascii")
    ).hexdigest()
    resource = MappingProxyType(
        {
            "persistent_numeric_bytes": int(
                prior.nbytes + target.nbytes + posterior.nbytes + bandwidth.nbytes
            ),
            "enrollment_dispersion_macs_upper_bound": int(len(decoded) * FEATURE_DIM),
            "extra_query_macs": 0,
            "query_dependent_state_bytes": 0,
        }
    )
    return D114State(
        classes=tuple(bank.classes),
        old_class_indices=old_indices,
        sigma_prior=_readonly(prior, np.float64),
        sigma_target=_readonly(target, np.float64),
        sigma_posterior=_readonly(posterior, np.float64),
        predictive_bandwidth=_readonly(bandwidth, np.float64),
        bank_receipt_sha256=bank.bank_receipt_sha256,
        bundle_content_sha256=bundle.content_sha256,
        bundle_bank_pair_sha256=pair,
        config_lock_digest=bank.config_lock_digest,
        resource_receipt=resource,
    )


def score_d114_hbpd_logits(
    state: D114State, bank: TypedINT8ZIDSupportBank, query_zid: np.ndarray
) -> np.ndarray:
    """Score each independent query over all registered classes with HBPD."""

    if type(state) is not D114State or type(bank) is not TypedINT8ZIDSupportBank:
        raise D114HBPDError("D114 score requires exact state and bank types")
    pair = hashlib.sha256(
        (state.bundle_content_sha256 + bank.bank_receipt_sha256).encode("ascii")
    ).hexdigest()
    if (
        state.classes != tuple(bank.classes)
        or state.bank_receipt_sha256 != bank.bank_receipt_sha256
        or state.config_lock_digest != bank.config_lock_digest
        or state.bundle_bank_pair_sha256 != pair
    ):
        raise D114HBPDError("D114 state/support-bank binding drift")
    query = normalize_zid_rows(query_zid).astype(np.float64)
    support = decode_zid_support_bank(bank).astype(np.float64)
    cosine = np.clip(query @ support.T, -1.0, 1.0)
    distance = np.maximum(2.0 * (1.0 - cosine), 0.0)
    nu = float(bank.config.student_nu)
    dimension = bank.config.kernel_effective_dim
    columns = []
    for class_index, expected in enumerate(bank.support_counts):
        local = distance[:, bank.class_indices_int16 == class_index]
        h = float(state.predictive_bandwidth[class_index])
        kernel = (
            -dimension * math.log(h)
            - 0.5 * (nu + dimension) * np.log1p(local / (nu * h * h))
        )
        maximum = np.max(kernel, axis=1, keepdims=True)
        columns.append(
            maximum[:, 0]
            + np.log(np.sum(np.exp(kernel - maximum), axis=1))
            - math.log(expected)
        )
    output = np.stack(columns, axis=1)
    if not np.isfinite(output).all():
        raise D114HBPDError("D114 HBPD logits became non-finite")
    return _readonly(output, np.float32)


def audit_d114_state(state: D114State) -> Mapping[str, Any]:
    if type(state) is not D114State:
        raise D114HBPDError("D114 audit requires an exact state")
    return MappingProxyType(
        {
            "schema": state.schema,
            "predictive_bandwidth_min": float(np.min(state.predictive_bandwidth)),
            "predictive_bandwidth_max": float(np.max(state.predictive_bandwidth)),
            "predictive_bandwidth_unique_count": int(
                len(np.unique(state.predictive_bandwidth))
            ),
            "resource_receipt": state.resource_receipt,
            "query_rows_used_for_fit": 0,
            "truth_role_quota_inputs": 0,
        }
    )


__all__ = [
    "D114Bundle",
    "D114HBPDError",
    "D114State",
    "audit_d114_state",
    "build_d114_bundle",
    "fit_d114_state",
    "score_d114_hbpd_logits",
]
