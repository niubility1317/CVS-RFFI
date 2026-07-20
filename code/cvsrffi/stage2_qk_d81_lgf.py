"""Protocol-minimal local-global fusion on top of an immutable D81 head.

The module owns only a quantized target-support qKNN branch.  D81 probabilities
are supplied by the caller and remain the exact output when ``eta == 0``.
There is no query-fit, receiver, scenario, class-role, quota, graph, source,
clean-sample, or multi-view input surface.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FEATURE_DIM = 288
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
BLOCK_COUNT = len(BLOCK_SLICES)
INT8_MAX = 127.0
SCHEMA = "cvs.phase2.qk_d81_lgf.v1"
ALLOWED_ETA_SOURCES = frozenset(
    {"zero_fallback", "phase1_locked_k1_prior", "support_crossfit_phase1_smoothed"}
)


class QKD81LGFError(ValueError):
    """Raised when the frozen method lock or support-only state drifts."""


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _finite_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise QKD81LGFError(
            f"{name} must be finite float32 [N,{FEATURE_DIM}]"
        )
    return np.ascontiguousarray(rows)


def normalize_three_blocks(value: np.ndarray) -> np.ndarray:
    """Give z160, FFT96, and RF32 equal energy, then normalize the concat."""

    rows = _finite_rows(value, "features").astype(np.float64)
    normalized = np.zeros_like(rows)
    for block in BLOCK_SLICES:
        part = rows[:, block]
        norms = np.linalg.norm(part, axis=1, keepdims=True)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise QKD81LGFError("each 288D feature block must have positive norm")
        normalized[:, block] = part / norms
    total = np.linalg.norm(normalized, axis=1, keepdims=True)
    if np.any(total <= 0.0) or not np.isfinite(total).all():
        raise QKD81LGFError("three-block normalization became degenerate")
    return _readonly(normalized / total, np.float32)


@dataclass(frozen=True)
class Phase1LockedConfig:
    """Parameters selected and sealed before any target/query access."""

    beta: float
    temp_base: float
    temp_qk: float
    eta_max: float
    phase1_receipt_sha256: str
    margin_audit_sha256: str
    k1_eta_prior: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.beta,
            self.temp_base,
            self.temp_qk,
            self.eta_max,
            self.k1_eta_prior,
        )
        if (
            not all(math.isfinite(float(value)) for value in values)
            or float(self.beta) <= 0.0
            or float(self.temp_base) <= 0.0
            or float(self.temp_qk) <= 0.0
            or not 0.0 <= float(self.eta_max) <= 1.0
            or not 0.0 <= float(self.k1_eta_prior) <= float(self.eta_max)
            or len(self.phase1_receipt_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.phase1_receipt_sha256)
            or len(self.margin_audit_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.margin_audit_sha256)
        ):
            raise QKD81LGFError("Phase1 method lock is invalid")

    @property
    def lock_digest(self) -> str:
        payload = {
            "beta": float(self.beta),
            "eta_max": float(self.eta_max),
            "k1_eta_prior": float(self.k1_eta_prior),
            "margin_audit_sha256": self.margin_audit_sha256,
            "phase1_receipt_sha256": self.phase1_receipt_sha256,
            "temp_base": float(self.temp_base),
            "temp_qk": float(self.temp_qk),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class QuantizedSupportBank:
    schema: str
    classes: tuple[str, ...]
    support_counts: tuple[int, ...]
    codes_qint8: np.ndarray
    scales_fp16: np.ndarray
    class_indices_int16: np.ndarray
    eta: float
    eta_source: str
    support_cv_receipt_sha256: str | None
    config_lock_digest: str
    config: Phase1LockedConfig
    quantization_audit: dict[str, Any]

    def __post_init__(self) -> None:
        class_count = len(self.classes)
        row_count = int(sum(self.support_counts))
        if (
            self.schema != SCHEMA
            or class_count < 2
            or len(set(self.classes)) != class_count
            or any(not value for value in self.classes)
            or len(self.support_counts) != class_count
            or any(int(value) < 1 for value in self.support_counts)
            or self.codes_qint8.dtype != np.int8
            or self.codes_qint8.shape != (row_count, FEATURE_DIM)
            or self.scales_fp16.dtype != np.float16
            or self.scales_fp16.shape != (row_count, BLOCK_COUNT)
            or self.class_indices_int16.dtype != np.int16
            or self.class_indices_int16.shape != (row_count,)
            or not np.isfinite(self.scales_fp16).all()
            or not bool(np.all(self.scales_fp16 > 0))
            or not np.array_equal(
                np.bincount(self.class_indices_int16, minlength=class_count),
                np.asarray(self.support_counts, dtype=np.int64),
            )
            or not math.isfinite(float(self.eta))
            or not 0.0 <= float(self.eta) <= float(self.config.eta_max)
            or self.eta_source not in ALLOWED_ETA_SOURCES
            or self.config_lock_digest != self.config.lock_digest
            or (
                self.eta_source == "support_crossfit_phase1_smoothed"
                and (
                    self.support_cv_receipt_sha256 is None
                    or len(self.support_cv_receipt_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in self.support_cv_receipt_sha256
                    )
                )
            )
            or (
                self.eta_source != "support_crossfit_phase1_smoothed"
                and self.support_cv_receipt_sha256 is not None
            )
        ):
            raise QKD81LGFError("quantized support bank drift")
        for name, dtype in (
            ("codes_qint8", np.int8),
            ("scales_fp16", np.float16),
            ("class_indices_int16", np.int16),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def is_uniform_k1(self) -> bool:
        return all(int(value) == 1 for value in self.support_counts)

    @property
    def support_row_count(self) -> int:
        return int(len(self.codes_qint8))

    @property
    def registry_state_bytes(self) -> int:
        metadata = {
            "classes": list(self.classes),
            "config": {
                "beta": float(self.config.beta),
                "eta_max": float(self.config.eta_max),
                "k1_eta_prior": float(self.config.k1_eta_prior),
                "lock_digest": self.config.lock_digest,
                "margin_audit_sha256": self.config.margin_audit_sha256,
                "phase1_receipt_sha256": self.config.phase1_receipt_sha256,
                "temp_base": float(self.config.temp_base),
                "temp_qk": float(self.config.temp_qk),
            },
            "eta": float(self.eta),
            "eta_source": self.eta_source,
            "support_cv_receipt_sha256": self.support_cv_receipt_sha256,
            "schema": self.schema,
            "support_counts": list(self.support_counts),
        }
        return len(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        )

    @property
    def persistent_state_bytes(self) -> int:
        audit_bytes = len(
            json.dumps(
                self.quantization_audit,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        return int(
            self.codes_qint8.nbytes
            + self.scales_fp16.nbytes
            + self.class_indices_int16.nbytes
            + self.registry_state_bytes
            + audit_bytes
        )

    @property
    def extra_macs_per_query(self) -> int:
        return 0 if float(self.eta) == 0.0 else self.support_row_count * FEATURE_DIM

    @property
    def extra_scalar_ops_per_query_upper_bound(self) -> int:
        if float(self.eta) == 0.0:
            return 0
        class_count = len(self.classes)
        return int(
            8 * FEATURE_DIM
            + 4 * self.support_row_count * FEATURE_DIM
            + 6 * self.support_row_count
            + 8 * class_count
        )


def _quantize_rows(normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(normalized, dtype=np.float32)
    codes = np.zeros(rows.shape, dtype=np.int8)
    scales = np.zeros((len(rows), BLOCK_COUNT), dtype=np.float16)
    decoded = np.zeros(rows.shape, dtype=np.float32)
    min_scale = float(np.finfo(np.float16).tiny)
    for row_index in range(len(rows)):
        for block_index, block in enumerate(BLOCK_SLICES):
            part = rows[row_index, block]
            scale64 = max(float(np.max(np.abs(part))) / INT8_MAX, min_scale)
            scale16 = np.float16(scale64)
            if not np.isfinite(scale16) or scale16 <= 0:
                raise QKD81LGFError("support quantization scale overflow")
            code = np.clip(np.rint(part / float(scale16)), -127, 127).astype(np.int8)
            codes[row_index, block] = code
            scales[row_index, block_index] = scale16
            decoded[row_index, block] = code.astype(np.float32) * np.float32(scale16)
    decoded = normalize_three_blocks(decoded)
    return codes, scales, decoded


def _canonical_order(
    codes: np.ndarray, scales: np.ndarray, class_indices: np.ndarray
) -> np.ndarray:
    keys = []
    for index in range(len(codes)):
        keys.append(
            (
                int(class_indices[index]),
                np.ascontiguousarray(codes[index]).tobytes(),
                np.ascontiguousarray(scales[index]).tobytes(),
                index,
            )
        )
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def build_support_bank(
    support_features: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
    *,
    config: Phase1LockedConfig,
    support_only_eta: float = 0.0,
    eta_source: str = "zero_fallback",
    support_cv_receipt_sha256: str | None = None,
) -> QuantizedSupportBank:
    """Build state from labeled target support only; no query argument exists."""

    rows = normalize_three_blocks(support_features)
    labels = tuple(str(value) for value in support_labels)
    registry = tuple(str(value) for value in classes)
    if (
        len(labels) != len(rows)
        or len(registry) < 2
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or any(label not in registry for label in labels)
    ):
        raise QKD81LGFError("support labels/classes closure drift")
    class_map = {label: index for index, label in enumerate(registry)}
    class_indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    counts = tuple(int(np.sum(class_indices == index)) for index in range(len(registry)))
    if any(value < 1 for value in counts):
        raise QKD81LGFError("every registered class requires target support")
    if len(set(counts)) != 1:
        raise QKD81LGFError("formal rows require balanced K-shot support")
    is_k1 = all(value == 1 for value in counts)
    requested = float(support_only_eta)
    if not math.isfinite(requested):
        raise QKD81LGFError("support-only eta must be finite")
    if is_k1:
        if requested != 0.0 or eta_source != "zero_fallback":
            raise QKD81LGFError("K1 cannot fit eta from target support")
        eta = float(config.k1_eta_prior)
        resolved_source = (
            "phase1_locked_k1_prior" if eta > 0.0 else "zero_fallback"
        )
    else:
        if not 0.0 <= requested <= float(config.eta_max):
            raise QKD81LGFError("support-only eta exceeds Phase1 lock")
        if requested > 0.0 and eta_source != "support_crossfit_phase1_smoothed":
            raise QKD81LGFError("nonzero eta requires support-crossfit evidence")
        if requested == 0.0 and eta_source != "zero_fallback":
            raise QKD81LGFError("zero eta must use exact fallback source")
        eta = requested
        resolved_source = eta_source
    if resolved_source == "support_crossfit_phase1_smoothed":
        if (
            support_cv_receipt_sha256 is None
            or len(support_cv_receipt_sha256) != 64
            or any(character not in "0123456789abcdef" for character in support_cv_receipt_sha256)
        ):
            raise QKD81LGFError("nonzero eta requires a sealed support-CV receipt")
    elif support_cv_receipt_sha256 is not None:
        raise QKD81LGFError("zero/Phase1-prior eta cannot claim a support-CV receipt")
    codes, scales, decoded = _quantize_rows(rows)
    order = _canonical_order(codes, scales, class_indices)
    codes = codes[order]
    scales = scales[order]
    decoded = decoded[order]
    class_indices = class_indices[order]
    ordered_rows = np.asarray(rows, dtype=np.float32)[order]
    error = np.abs(decoded.astype(np.float64) - ordered_rows.astype(np.float64))
    cosine = np.sum(decoded.astype(np.float64) * ordered_rows.astype(np.float64), axis=1)
    audit = {
        "schema": "cvs.phase2.qk_d81_lgf.quantization_audit.v1",
        "support_only": True,
        "single_view": True,
        "feature_dim": FEATURE_DIM,
        "block_dims": [160, 96, 32],
        "support_rows": int(len(rows)),
        "class_count": int(len(registry)),
        "support_counts": list(counts),
        "class_count_normalization": "per_class_log_mean_exp_divide_by_Kc",
        "quantization_error_mean": float(np.mean(error)),
        "quantization_error_max": float(np.max(error)),
        "reconstruction_cosine_mean": float(np.mean(cosine)),
        "reconstruction_cosine_min": float(np.min(cosine)),
        "eta": eta,
        "eta_source": resolved_source,
        "config_lock_digest": config.lock_digest,
        "phase1_receipt_sha256": config.phase1_receipt_sha256,
        "margin_audit_sha256": config.margin_audit_sha256,
        "query_rows_used_for_fit": 0,
        "query_dependent_state": False,
        "class_label_permutation_equivariant": True,
        "support_order_canonicalized": True,
    }
    return QuantizedSupportBank(
        schema=SCHEMA,
        classes=registry,
        support_counts=counts,
        codes_qint8=codes,
        scales_fp16=scales,
        class_indices_int16=class_indices,
        eta=eta,
        eta_source=resolved_source,
        support_cv_receipt_sha256=support_cv_receipt_sha256,
        config_lock_digest=config.lock_digest,
        config=config,
        quantization_audit=audit,
    )


def decode_support_bank(bank: QuantizedSupportBank) -> np.ndarray:
    if not isinstance(bank, QuantizedSupportBank):
        raise QKD81LGFError("support bank type drift")
    decoded = np.zeros((bank.support_row_count, FEATURE_DIM), dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        decoded[:, block] = (
            bank.codes_qint8[:, block].astype(np.float32)
            * bank.scales_fp16[:, block_index].astype(np.float32)[:, None]
        )
    return normalize_three_blocks(decoded)


def score_qknn_logits(
    bank: QuantizedSupportBank, query_features: np.ndarray
) -> np.ndarray:
    """Return per-class normalized log-mean-exp cosine evidence."""

    query = normalize_three_blocks(query_features).astype(np.float64)
    support = decode_support_bank(bank).astype(np.float64)
    similarities = query @ support.T
    beta = float(bank.config.beta)
    columns = []
    for class_index in range(len(bank.classes)):
        local = similarities[:, bank.class_indices_int16 == class_index]
        if local.shape[1] != bank.support_counts[class_index]:
            raise QKD81LGFError("class support count drift during scoring")
        scaled = beta * local
        maximum = np.max(scaled, axis=1, keepdims=True)
        # mean(exp(.)) is the explicit 1/Kc normalization.
        column = (
            maximum[:, 0]
            + np.log(np.mean(np.exp(scaled - maximum), axis=1))
        ) / beta
        columns.append(column)
    logits = np.stack(columns, axis=1)
    if not np.isfinite(logits).all():
        raise QKD81LGFError("qKNN logits became non-finite")
    return _readonly(logits, np.float32)


def audit_quantized_margin(
    bank: QuantizedSupportBank,
    full_precision_support_features: np.ndarray,
    support_labels: Sequence[str],
    validation_features: np.ndarray,
) -> dict[str, Any]:
    """Compare FP32 and INT8 qKNN margins on Phase1-only validation rows."""

    support = normalize_three_blocks(full_precision_support_features).astype(np.float64)
    validation = normalize_three_blocks(validation_features).astype(np.float64)
    labels = tuple(str(value) for value in support_labels)
    if len(labels) != len(support) or any(label not in bank.classes for label in labels):
        raise QKD81LGFError("margin audit support labels/classes drift")
    fp_columns = []
    similarities = validation @ support.T
    beta = float(bank.config.beta)
    for class_name, expected_count in zip(bank.classes, bank.support_counts):
        local = similarities[:, np.asarray(labels) == class_name]
        if local.shape[1] != expected_count:
            raise QKD81LGFError("margin audit support count drift")
        scaled = beta * local
        maximum = np.max(scaled, axis=1, keepdims=True)
        fp_columns.append(
            (maximum[:, 0] + np.log(np.mean(np.exp(scaled - maximum), axis=1))) / beta
        )
    fp_logits = np.stack(fp_columns, axis=1)
    int8_logits = score_qknn_logits(bank, validation_features).astype(np.float64)
    teacher_order = np.argsort(fp_logits, axis=1, kind="stable")
    winner = teacher_order[:, -1]
    runner_up = teacher_order[:, -2]
    row = np.arange(len(fp_logits))
    fp_margin = fp_logits[row, winner] - fp_logits[row, runner_up]
    int8_teacher_margin = int8_logits[row, winner] - int8_logits[row, runner_up]
    flip = int8_teacher_margin <= 0.0
    return {
        "schema": "cvs.phase2.qk_d81_lgf.phase1_margin_audit.v1",
        "validation_row_count": int(len(validation)),
        "logit_abs_error_mean": float(np.mean(np.abs(fp_logits - int8_logits))),
        "logit_abs_error_max": float(np.max(np.abs(fp_logits - int8_logits))),
        "top1_agreement": float(np.mean(np.argmax(fp_logits, axis=1) == np.argmax(int8_logits, axis=1))),
        "teacher_margin_mean": float(np.mean(fp_margin)),
        "quantized_teacher_margin_mean": float(np.mean(int8_teacher_margin)),
        "margin_sign_flip_count": int(np.sum(flip)),
        "margin_sign_flip_rate": float(np.mean(flip)),
        "query_or_target_rows_used": 0,
    }


def softmax_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(logits)
    temp = float(temperature)
    if (
        scores.dtype != np.float32
        or scores.ndim != 2
        or scores.shape[0] < 1
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
        or not math.isfinite(temp)
        or temp <= 0.0
    ):
        raise QKD81LGFError("softmax input/temperature drift")
    scaled = scores.astype(np.float64) / temp
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    probabilities = exp / np.sum(exp, axis=1, keepdims=True)
    return _readonly(probabilities, np.float32)


def fuse_with_base_probabilities(
    bank: QuantizedSupportBank,
    base_probabilities: np.ndarray,
    query_features: np.ndarray,
    *,
    base_classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse one row-global eta; eta=0 returns the exact base array unchanged."""

    base = np.asarray(base_probabilities)
    query = _finite_rows(query_features, "query features")
    caller_classes = tuple(str(value) for value in base_classes)
    if (
        base.dtype != np.float32
        or base.ndim != 2
        or base.shape != (len(query), len(bank.classes))
        or not np.isfinite(base).all()
        or np.any(base < 0.0)
        or not np.allclose(np.sum(base, axis=1), 1.0, rtol=0.0, atol=2.0e-6)
        or caller_classes != bank.classes
    ):
        raise QKD81LGFError("base probability/class closure drift")
    base_pred = np.asarray(bank.classes, dtype=object)[np.argmax(base, axis=1)]
    if float(bank.eta) == 0.0:
        return base_probabilities, base_pred.astype(str), {
            "schema": "cvs.phase2.qk_d81_lgf.inference_audit.v1",
            "eta": 0.0,
            "eta_source": bank.eta_source,
            "exact_base_fallback": True,
            "qknn_branch_executed": False,
            "extra_macs_per_query": 0,
            "persistent_state_bytes": bank.persistent_state_bytes,
            "extra_scalar_ops_per_query_upper_bound": 0,
            "query_state_updates": 0,
        }
    qk_logits = score_qknn_logits(bank, query)
    qk_prob = softmax_probabilities(qk_logits, bank.config.temp_qk)
    eta = float(bank.eta)
    fused64 = (1.0 - eta) * base.astype(np.float64) + eta * qk_prob.astype(np.float64)
    fused = _readonly(fused64, np.float32)
    prediction = np.asarray(bank.classes, dtype=object)[np.argmax(fused, axis=1)]
    audit = {
        "schema": "cvs.phase2.qk_d81_lgf.inference_audit.v1",
        "eta": eta,
        "eta_source": bank.eta_source,
        "exact_base_fallback": False,
        "qknn_branch_executed": True,
        "extra_macs_per_query": bank.extra_macs_per_query,
        "persistent_state_bytes": bank.persistent_state_bytes,
        "extra_scalar_ops_per_query_upper_bound": bank.extra_scalar_ops_per_query_upper_bound,
        "query_state_updates": 0,
        "query_batch_coupling": False,
        "decision_policy": "per_sample_all_registered_classes_argmax",
    }
    return fused, prediction.astype(str), audit


def predict_from_base_logits(
    bank: QuantizedSupportBank,
    base_logits: np.ndarray,
    query_features: np.ndarray,
    *,
    base_classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    logits = np.asarray(base_logits)
    if (
        logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape[1] != len(bank.classes)
    ):
        raise QKD81LGFError("base logits/class closure drift")
    base_prob = softmax_probabilities(logits, bank.config.temp_base)
    return fuse_with_base_probabilities(
        bank, base_prob, query_features, base_classes=base_classes
    )


__all__ = [
    "BLOCK_SLICES",
    "FEATURE_DIM",
    "Phase1LockedConfig",
    "QKD81LGFError",
    "QuantizedSupportBank",
    "audit_quantized_margin",
    "build_support_bank",
    "decode_support_bank",
    "fuse_with_base_probabilities",
    "normalize_three_blocks",
    "predict_from_base_logits",
    "score_qknn_logits",
    "softmax_probabilities",
]
