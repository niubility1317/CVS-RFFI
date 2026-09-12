"""D42 support-only unified automatic-shrinkage LDA.

The old-only diagonal metric is the exact D38 full-batch B20 trajectory.  LDA
is fitted twice with that frozen metric: first on old support for the immutable
before artifact, then on all old/new support for the registered artifact.
Formal coefficients use fixed three-block two-level residual int8 storage and
formal intercepts use FP16.  No query input exists on the fit path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from cvsrffi.stage2_d38_strong_b3_quantized import (
    FEATURE_NOISE_STD,
    FFT_LOG_SCALE_LIMIT,
    GRAD_CLIP,
    LOG_SCALE_LIMIT,
    PROTOTYPE_ANCHOR_WEIGHT,
    STAGE2B_LR,
    TEMPERATURE,
    WEIGHT_DECAY,
)


FEATURE_DIM = 288
BLOCK_SLICES = (slice(0, 160), slice(160, 256), slice(256, 288))
BLOCK_DIMS = (160, 96, 32)
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
METRIC_EPOCHS = 20
SCHEMA_INT8 = "cvs.phase2.d42.unified_shrinkage_lda_residual_int8.v1"
SCHEMA_FP32 = "cvs.phase2.d42.unified_shrinkage_lda_fp32_ablation.v1"
ENERGY_EPSILON = 1.0e-12
MAX_TRAINABLE_PARAMETERS = 80_000
MAX_PERSISTENT_STATE_BYTES = 256 * 1024
SKLEARN_RUNTIME_VERSION = "1.7.2"


class D42UnifiedShrinkageLDAError(ValueError):
    pass


def _require_sklearn_runtime() -> None:
    actual = str(sklearn.__version__)
    if actual != SKLEARN_RUNTIME_VERSION:
        raise D42UnifiedShrinkageLDAError(
            "D42 sklearn runtime version drift: "
            f"expected {SKLEARN_RUNTIME_VERSION}, got {actual}"
        )


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D42UnifiedShrinkageLDAError(
            f"{name} must be finite float32 [N,{FEATURE_DIM}]"
        )
    return np.ascontiguousarray(rows)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _rows(features, f"{name} features")
    registry = tuple(str(value) for value in classes)
    y = tuple(str(value) for value in labels)
    if (
        len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not value for value in registry)
        or set(y) != set(registry)
    ):
        raise D42UnifiedShrinkageLDAError(f"{name} registry drift")
    counts = [sum(value == label for value in y) for label in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D42UnifiedShrinkageLDAError(f"{name} must be symmetric K-shot")
    targets = np.asarray([registry.index(value) for value in y], dtype=np.int64)
    return rows, targets, registry, int(counts[0])


def _normalize_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or bool(np.any(norms <= ENERGY_EPSILON)):
        raise D42UnifiedShrinkageLDAError(f"{name} contains a zero row")
    return np.asarray(rows / norms, dtype=np.float32)


def _transform(features: np.ndarray, log_diag: np.ndarray) -> np.ndarray:
    rows = _rows(features, "D42 transform features")
    diagonal = np.asarray(log_diag)
    if (
        diagonal.dtype != np.float32
        or diagonal.shape != (FEATURE_DIM,)
        or not np.isfinite(diagonal).all()
    ):
        raise D42UnifiedShrinkageLDAError("D42 frozen logdiag drift")
    scale = np.exp(diagonal).astype(np.float32)
    return _normalize_rows(rows * scale[None, :], "D42 transformed support")


def _tensor_from_numpy(
    value: np.ndarray, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype == torch.float32:
        rows = np.ascontiguousarray(value, dtype=np.float32)
    elif dtype == torch.long:
        rows = np.ascontiguousarray(value, dtype=np.int64)
    else:
        raise D42UnifiedShrinkageLDAError("D42 unsupported tensor bridge dtype")
    return torch.frombuffer(rows, dtype=dtype).reshape(rows.shape).clone().to(device)


def _fit_old_only_b3_metric(
    old_rows: np.ndarray,
    old_targets: np.ndarray,
    old_class_count: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Exact D38 full-batch B20 metric with an old-only input surface."""

    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device)
    x_old = _tensor_from_numpy(old_rows, dtype=torch.float32, device=device)
    y_old = _tensor_from_numpy(old_targets, dtype=torch.long, device=device)
    prototypes = torch.stack(
        [
            F.normalize(x_old[y_old == index].mean(dim=0), dim=0)
            for index in range(old_class_count)
        ]
    )
    log_diag = torch.nn.Parameter(torch.zeros(FEATURE_DIM, device=device))
    old_weights = torch.nn.Parameter(prototypes.detach().clone())
    lower_np = np.full(FEATURE_DIM, -LOG_SCALE_LIMIT, dtype=np.float32)
    upper_np = np.full(FEATURE_DIM, LOG_SCALE_LIMIT, dtype=np.float32)
    lower_np[160:256] = -FFT_LOG_SCALE_LIMIT
    upper_np[160:256] = FFT_LOG_SCALE_LIMIT
    lower = _tensor_from_numpy(lower_np, dtype=torch.float32, device=device)
    upper = _tensor_from_numpy(upper_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        [log_diag, old_weights], lr=STAGE2B_LR, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    trace: list[dict[str, Any]] = []
    for step in range(1, METRIC_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        noisy = x_old + FEATURE_NOISE_STD * torch.randn(
            x_old.shape,
            generator=generator,
            device=device,
            dtype=x_old.dtype,
        )
        effective = torch.minimum(torch.maximum(log_diag, lower), upper)
        transformed = F.normalize(noisy * torch.exp(effective), dim=1)
        logits = TEMPERATURE * (
            transformed @ F.normalize(old_weights, dim=1).T
        )
        ce_loss = F.cross_entropy(logits, y_old)
        anchor_loss = torch.mean(
            (F.normalize(old_weights, dim=1) - prototypes) ** 2
        )
        loss = ce_loss + PROTOTYPE_ANCHOR_WEIGHT * anchor_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [log_diag, old_weights], max_norm=GRAD_CLIP
        )
        optimizer.step()
        with torch.no_grad():
            log_diag.copy_(torch.minimum(torch.maximum(log_diag, lower), upper))
        row = {
            "phase": "stage2b_fullbatch_old_adaptation",
            "epoch": step,
            "optimizer_step": step,
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "prototype_anchor_loss": float(anchor_loss.detach().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "support_accuracy": float(
                (logits.argmax(dim=1) == y_old).float().mean().detach().cpu()
            ),
            "query_rows_used": 0,
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key != "phase"
        ):
            raise D42UnifiedShrinkageLDAError(
                "non-finite D42 old-only B3 metric trace"
            )
        trace.append(row)
    log_diag_np = np.asarray(log_diag.detach().cpu().tolist(), dtype=np.float32)
    trainable_parameters = FEATURE_DIM * (1 + int(old_class_count))
    resource = {
        "trainable_parameters": int(trainable_parameters),
        "adaptation_epochs": METRIC_EPOCHS,
        "optimizer_steps": METRIC_EPOCHS,
        "estimated_adaptation_macs": int(
            3
            * FEATURE_DIM
            * METRIC_EPOCHS
            * len(old_rows)
            * int(old_class_count)
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "runtime_device": str(device),
        "new_support_argument_count": 0,
    }
    return log_diag_np, tuple(trace), resource


def _positive_fp16(value: float) -> np.float16:
    smallest = np.nextafter(np.float16(0), np.float16(1))
    result = np.float16(max(float(value), float(smallest)))
    if not np.isfinite(result) or result <= 0:
        raise D42UnifiedShrinkageLDAError("D42 quantization scale overflow")
    return result


def _decode_coefficients(
    code1: np.ndarray,
    code2: np.ndarray,
    scale1: np.ndarray,
    scale2: np.ndarray,
) -> np.ndarray:
    decoded = np.empty(code1.shape, dtype=np.float32)
    for block_index, block in enumerate(BLOCK_SLICES):
        decoded[:, block] = (
            code1[:, block].astype(np.float32)
            * scale1[:, block_index].astype(np.float32)[:, None]
            + code2[:, block].astype(np.float32)
            * scale2[:, block_index].astype(np.float32)[:, None]
        )
    if not np.isfinite(decoded).all():
        raise D42UnifiedShrinkageLDAError("D42 coefficient decode became non-finite")
    return decoded


def _quantize_coefficients(
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = _rows(np.asarray(coefficients, dtype=np.float32), "D42 LDA coefficients")
    code1 = np.zeros(rows.shape, dtype=np.int8)
    code2 = np.zeros(rows.shape, dtype=np.int8)
    scale1 = np.empty((len(rows), len(BLOCK_SLICES)), dtype=np.float16)
    scale2 = np.empty((len(rows), len(BLOCK_SLICES)), dtype=np.float16)
    for row_index, row in enumerate(rows):
        for block_index, block in enumerate(BLOCK_SLICES):
            values = row[block]
            first_scale = _positive_fp16(float(np.max(np.abs(values))) / 127.0)
            first_code = np.clip(
                np.rint(values / np.float32(first_scale)), -127, 127
            ).astype(np.int8)
            residual = values - np.float32(first_scale) * first_code.astype(
                np.float32
            )
            second_scale = _positive_fp16(
                float(np.max(np.abs(residual))) / 127.0
            )
            second_code = np.clip(
                np.rint(residual / np.float32(second_scale)), -127, 127
            ).astype(np.int8)
            code1[row_index, block] = first_code
            code2[row_index, block] = second_code
            scale1[row_index, block_index] = first_scale
            scale2[row_index, block_index] = second_scale
    decoded = _decode_coefficients(code1, code2, scale1, scale2)
    return code1, code2, scale1, scale2, decoded


def _fit_equal_prior_lda(
    transformed: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    means = np.stack(
        [transformed[targets == index].mean(axis=0) for index in range(class_count)]
    ).astype(np.float32)
    residuals = transformed - means[targets]
    residual_energy = float(np.sum(residuals.astype(np.float64) ** 2))
    residual_rank = int(np.linalg.matrix_rank(residuals.astype(np.float64)))
    fallback = bool(
        int(k_shot) <= 2
        or residual_rank == 0
        or not math.isfinite(residual_energy)
        or residual_energy <= ENERGY_EPSILON
    )
    priors = np.full(class_count, 1.0 / class_count, dtype=np.float64)
    if fallback:
        coefficients = np.asarray(means, dtype=np.float32)
        intercept = np.asarray(
            -0.5 * np.sum(means.astype(np.float64) ** 2, axis=1)
            + np.log(priors),
            dtype=np.float32,
        )
        policy = "unit_covariance_equal_prior_nearest_centroid"
        coefficient_source = "unit_covariance_explicit_lstsq_equivalent"
        covariance_equation_residual_max = 0.0
        sklearn_prediction_equivalent: bool | None = None
    else:
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=priors, store_covariance=True
        )
        estimator.fit(transformed.astype(np.float64), targets)
        if not np.array_equal(
            np.asarray(estimator.classes_, dtype=np.int64),
            np.arange(class_count, dtype=np.int64),
        ):
            raise D42UnifiedShrinkageLDAError("D42 sklearn class order drift")
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        fitted_means = np.asarray(estimator.means_, dtype=np.float64)
        coefficients64 = np.linalg.lstsq(
            covariance, fitted_means.T, rcond=None
        )[0].T
        intercept64 = (
            -0.5 * np.diag(fitted_means @ coefficients64.T) + np.log(priors)
        )
        coefficients = np.asarray(coefficients64, dtype=np.float32)
        intercept = np.asarray(intercept64, dtype=np.float32)
        covariance_equation_residual_max = float(
            np.max(np.abs(covariance @ coefficients64.T - fitted_means.T))
        )
        coefficient_source = (
            "locked_sklearn_covariance_means_explicit_lstsq_sigma_inverse_mu"
        )
        deployed_predictions = np.argmax(
            transformed.astype(np.float64) @ coefficients.astype(np.float64).T
            + intercept.astype(np.float64)[None, :],
            axis=1,
        )
        if not np.array_equal(
            deployed_predictions,
            np.asarray(estimator.predict(transformed.astype(np.float64)), dtype=np.int64),
        ):
            raise D42UnifiedShrinkageLDAError(
                "D42 sklearn coefficient deployment prediction drift"
            )
        sklearn_prediction_equivalent = True
        policy = "sklearn_lsqr_auto_shrinkage_equal_prior"
    if (
        coefficients.shape != (class_count, FEATURE_DIM)
        or intercept.shape != (class_count,)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
    ):
        raise D42UnifiedShrinkageLDAError("D42 LDA solution became non-finite")
    audit = {
        "solver": "lsqr",
        "shrinkage": "auto",
        "prior_policy": "equal_1_over_registered_class_count",
        "covariance_policy": policy,
        "unit_covariance_fallback": fallback,
        "within_class_residual_rank": residual_rank,
        "within_class_residual_energy": residual_energy,
        "support_rows": int(len(transformed)),
        "class_count": int(class_count),
        "k_shot": int(k_shot),
        "coefficient_source": coefficient_source,
        "covariance_equation_residual_max": covariance_equation_residual_max,
        "sklearn_prediction_equivalent": sklearn_prediction_equivalent,
    }
    return coefficients, intercept, audit


@dataclass(frozen=True)
class D42UnifiedShrinkageLDAConfig:
    metric_epochs: int = METRIC_EPOCHS
    lda_solver: str = "lsqr"
    shrinkage: str = "auto"
    prior_policy: str = "equal"
    sklearn_runtime_version: str = SKLEARN_RUNTIME_VERSION

    def __post_init__(self) -> None:
        if (
            int(self.metric_epochs) != METRIC_EPOCHS
            or self.lda_solver != "lsqr"
            or self.shrinkage != "auto"
            or self.prior_policy != "equal"
            or self.sklearn_runtime_version != SKLEARN_RUNTIME_VERSION
        ):
            raise D42UnifiedShrinkageLDAError("D42 mechanism lock drift")
        _require_sklearn_runtime()


@dataclass(frozen=True)
class D42UnifiedShrinkageLDAState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    log_diag_fp32: np.ndarray
    coef1_qint8: np.ndarray
    coef2_qint8: np.ndarray
    scale1_fp16: np.ndarray
    scale2_fp16: np.ndarray
    intercept_fp16: np.ndarray
    coef_fp32: np.ndarray
    intercept_fp32: np.ndarray
    covariance_policy: str

    def __post_init__(self) -> None:
        count = len(self.classes)
        new_count = count - int(self.old_class_count)
        is_int8 = self.schema == SCHEMA_INT8
        is_fp32 = self.schema == SCHEMA_FP32
        if (
            not (is_int8 or is_fp32)
            or not 2 <= int(self.old_class_count) <= 20
            or new_count not in (0,) + ALLOWED_NEW_CLASS_COUNTS
            or len(set(self.classes)) != count
            or any(not value for value in self.classes)
            or self.log_diag_fp32.dtype != np.float32
            or self.log_diag_fp32.shape != (FEATURE_DIM,)
            or not np.isfinite(self.log_diag_fp32).all()
            or self.covariance_policy
            not in {
                "sklearn_lsqr_auto_shrinkage_equal_prior",
                "unit_covariance_equal_prior_nearest_centroid",
            }
        ):
            raise D42UnifiedShrinkageLDAError("D42 state drift")
        if is_int8:
            valid = (
                self.coef1_qint8.dtype == np.int8
                and self.coef1_qint8.shape == (count, FEATURE_DIM)
                and self.coef2_qint8.dtype == np.int8
                and self.coef2_qint8.shape == (count, FEATURE_DIM)
                and self.scale1_fp16.dtype == np.float16
                and self.scale1_fp16.shape == (count, len(BLOCK_SLICES))
                and self.scale2_fp16.dtype == np.float16
                and self.scale2_fp16.shape == (count, len(BLOCK_SLICES))
                and self.intercept_fp16.dtype == np.float16
                and self.intercept_fp16.shape == (count,)
                and self.coef_fp32.shape == (0, FEATURE_DIM)
                and self.intercept_fp32.shape == (0,)
                and np.isfinite(self.scale1_fp16).all()
                and np.isfinite(self.scale2_fp16).all()
                and np.isfinite(self.intercept_fp16).all()
                and bool(np.all(self.scale1_fp16 > 0))
                and bool(np.all(self.scale2_fp16 > 0))
            )
        else:
            valid = (
                self.coef1_qint8.shape == (0, FEATURE_DIM)
                and self.coef2_qint8.shape == (0, FEATURE_DIM)
                and self.scale1_fp16.shape == (0, len(BLOCK_SLICES))
                and self.scale2_fp16.shape == (0, len(BLOCK_SLICES))
                and self.intercept_fp16.shape == (0,)
                and self.coef_fp32.dtype == np.float32
                and self.coef_fp32.shape == (count, FEATURE_DIM)
                and self.intercept_fp32.dtype == np.float32
                and self.intercept_fp32.shape == (count,)
                and np.isfinite(self.coef_fp32).all()
                and np.isfinite(self.intercept_fp32).all()
            )
        if not valid:
            raise D42UnifiedShrinkageLDAError("D42 state storage drift")
        for name, dtype in (
            ("log_diag_fp32", np.float32),
            ("coef1_qint8", np.int8),
            ("coef2_qint8", np.int8),
            ("scale1_fp16", np.float16),
            ("scale2_fp16", np.float16),
            ("intercept_fp16", np.float16),
            ("coef_fp32", np.float32),
            ("intercept_fp32", np.float32),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def is_int8(self) -> bool:
        return self.schema == SCHEMA_INT8

    @property
    def registry_state_bytes(self) -> int:
        metadata = {
            "classes": list(self.classes),
            "covariance_policy": self.covariance_policy,
            "old_class_count": int(self.old_class_count),
            "schema": self.schema,
        }
        return len(
            json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

    @property
    def persistent_state_bytes(self) -> int:
        arrays = (
            self.log_diag_fp32,
            self.coef1_qint8,
            self.coef2_qint8,
            self.scale1_fp16,
            self.scale2_fp16,
            self.intercept_fp16,
            self.coef_fp32,
            self.intercept_fp32,
        )
        return int(sum(value.nbytes for value in arrays) + self.registry_state_bytes)


@dataclass(frozen=True)
class D42UnifiedShrinkageLDAResult:
    before_state: D42UnifiedShrinkageLDAState
    state: D42UnifiedShrinkageLDAState
    matched_fp32_before_state: D42UnifiedShrinkageLDAState
    matched_fp32_state: D42UnifiedShrinkageLDAState
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _compile_state(
    classes: tuple[str, ...],
    old_class_count: int,
    log_diag: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    covariance_policy: str,
    *,
    precision: str,
) -> tuple[D42UnifiedShrinkageLDAState, dict[str, float]]:
    normalized_precision = str(precision).lower()
    if normalized_precision == "int8":
        code1, code2, scale1, scale2, decoded = _quantize_coefficients(
            coefficients
        )
        intercept16 = np.asarray(intercept, dtype=np.float16)
        if not np.isfinite(intercept16).all():
            raise D42UnifiedShrinkageLDAError("D42 intercept FP16 overflow")
        state = D42UnifiedShrinkageLDAState(
            schema=SCHEMA_INT8,
            classes=classes,
            old_class_count=old_class_count,
            log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
            coef1_qint8=code1,
            coef2_qint8=code2,
            scale1_fp16=scale1,
            scale2_fp16=scale2,
            intercept_fp16=intercept16,
            coef_fp32=np.zeros((0, FEATURE_DIM), dtype=np.float32),
            intercept_fp32=np.zeros(0, dtype=np.float32),
            covariance_policy=covariance_policy,
        )
        coef_error = np.abs(decoded - np.asarray(coefficients, dtype=np.float32))
        intercept_error = np.abs(
            intercept16.astype(np.float32) - np.asarray(intercept, dtype=np.float32)
        )
        audit = {
            "coefficient_quantization_error_mean": float(np.mean(coef_error)),
            "coefficient_quantization_error_max": float(np.max(coef_error)),
            "intercept_quantization_error_mean": float(np.mean(intercept_error)),
            "intercept_quantization_error_max": float(np.max(intercept_error)),
        }
        return state, audit
    if normalized_precision != "fp32":
        raise D42UnifiedShrinkageLDAError("D42 compile precision drift")
    state = D42UnifiedShrinkageLDAState(
        schema=SCHEMA_FP32,
        classes=classes,
        old_class_count=old_class_count,
        log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
        coef1_qint8=np.zeros((0, FEATURE_DIM), dtype=np.int8),
        coef2_qint8=np.zeros((0, FEATURE_DIM), dtype=np.int8),
        scale1_fp16=np.zeros((0, len(BLOCK_SLICES)), dtype=np.float16),
        scale2_fp16=np.zeros((0, len(BLOCK_SLICES)), dtype=np.float16),
        intercept_fp16=np.zeros(0, dtype=np.float16),
        coef_fp32=np.asarray(coefficients, dtype=np.float32),
        intercept_fp32=np.asarray(intercept, dtype=np.float32),
        covariance_policy=covariance_policy,
    )
    return state, {
        "coefficient_quantization_error_mean": 0.0,
        "coefficient_quantization_error_max": 0.0,
        "intercept_quantization_error_mean": 0.0,
        "intercept_quantization_error_max": 0.0,
    }


def decode_d42_coefficients(state: D42UnifiedShrinkageLDAState) -> np.ndarray:
    if not isinstance(state, D42UnifiedShrinkageLDAState):
        raise D42UnifiedShrinkageLDAError("D42 decode state drift")
    if state.is_int8:
        values = _decode_coefficients(
            state.coef1_qint8,
            state.coef2_qint8,
            state.scale1_fp16,
            state.scale2_fp16,
        )
    else:
        values = state.coef_fp32
    return _readonly(values, np.float32)


def _state_snapshot(state: D42UnifiedShrinkageLDAState) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(value).tobytes()
        for value in (
            state.log_diag_fp32,
            state.coef1_qint8,
            state.coef2_qint8,
            state.scale1_fp16,
            state.scale2_fp16,
            state.intercept_fp16,
            state.coef_fp32,
            state.intercept_fp32,
        )
    )


def _lda_fit_macs(row_count: int, class_count: int) -> int:
    dimension = FEATURE_DIM
    return int(
        row_count * dimension * dimension
        + dimension * dimension * dimension
        + class_count * dimension * dimension
    )


def fit_d42_unified_shrinkage_lda(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_classes: Sequence[str],
    *,
    seed: int,
    device: torch.device | str = "cpu",
    config: D42UnifiedShrinkageLDAConfig | None = None,
) -> D42UnifiedShrinkageLDAResult:
    _require_sklearn_runtime()
    locked = config or D42UnifiedShrinkageLDAConfig()
    del locked
    old_rows, old_targets, old_registry, old_k = _support(
        old_support_features, old_support_labels, old_classes, "old support"
    )
    if not 2 <= len(old_registry) <= 20:
        raise D42UnifiedShrinkageLDAError("D42 old class closure drift")
    runtime_device = torch.device(device)
    log_diag, trace, metric_resource = _fit_old_only_b3_metric(
        old_rows,
        old_targets,
        len(old_registry),
        seed=int(seed),
        device=runtime_device,
    )
    if (
        len(trace) != METRIC_EPOCHS
        or [int(row.get("optimizer_step", -1)) for row in trace]
        != list(range(1, METRIC_EPOCHS + 1))
    ):
        raise D42UnifiedShrinkageLDAError("D42 old-only B20 lifecycle drift")
    old_transformed = _transform(old_rows, log_diag)
    before_coef, before_intercept, before_lda = _fit_equal_prior_lda(
        old_transformed, old_targets, len(old_registry), old_k
    )
    before_state, before_quant = _compile_state(
        old_registry,
        len(old_registry),
        log_diag,
        before_coef,
        before_intercept,
        str(before_lda["covariance_policy"]),
        precision="int8",
    )
    matched_before, _ = _compile_state(
        old_registry,
        len(old_registry),
        log_diag,
        before_coef,
        before_intercept,
        str(before_lda["covariance_policy"]),
        precision="fp32",
    )
    before_snapshot = _state_snapshot(before_state)
    matched_before_snapshot = _state_snapshot(matched_before)

    # Stage2-C begins here.  No new-support object is read above this line.
    new_rows, new_targets, new_registry, new_k = _support(
        new_support_features, new_support_labels, new_classes, "new support"
    )
    if (
        len(new_registry) not in ALLOWED_NEW_CLASS_COUNTS
        or set(old_registry) & set(new_registry)
        or old_k != new_k
    ):
        raise D42UnifiedShrinkageLDAError("D42 class/K closure drift")

    all_registry = old_registry + new_registry
    all_rows = np.concatenate([old_rows, new_rows], axis=0).astype(np.float32)
    all_targets = np.concatenate(
        [old_targets, new_targets + len(old_registry)], axis=0
    )
    all_transformed = _transform(all_rows, log_diag)
    final_coef, final_intercept, final_lda = _fit_equal_prior_lda(
        all_transformed, all_targets, len(all_registry), old_k
    )
    state, final_quant = _compile_state(
        all_registry,
        len(old_registry),
        log_diag,
        final_coef,
        final_intercept,
        str(final_lda["covariance_policy"]),
        precision="int8",
    )
    matched_state, _ = _compile_state(
        all_registry,
        len(old_registry),
        log_diag,
        final_coef,
        final_intercept,
        str(final_lda["covariance_policy"]),
        precision="fp32",
    )
    if (
        before_snapshot != _state_snapshot(before_state)
        or matched_before_snapshot != _state_snapshot(matched_before)
    ):
        raise D42UnifiedShrinkageLDAError("D42 before state mutated during Stage2-C")

    before_int8_scores = score_d42_unified_shrinkage_lda(before_state, old_rows)
    before_fp32_scores = score_d42_unified_shrinkage_lda(matched_before, old_rows)
    final_int8_scores = score_d42_unified_shrinkage_lda(state, all_rows)
    final_fp32_scores = score_d42_unified_shrinkage_lda(matched_state, all_rows)
    before_argmax_changes = int(
        np.sum(
            np.argmax(before_int8_scores, axis=1)
            != np.argmax(before_fp32_scores, axis=1)
        )
    )
    final_argmax_changes = int(
        np.sum(
            np.argmax(final_int8_scores, axis=1)
            != np.argmax(final_fp32_scores, axis=1)
        )
    )
    before_score_error = float(
        np.max(np.abs(before_int8_scores - before_fp32_scores))
    )
    final_score_error = float(
        np.max(np.abs(final_int8_scores - final_fp32_scores))
    )
    trainable_parameters = FEATURE_DIM * (1 + len(old_registry))
    metric_macs = int(metric_resource["estimated_adaptation_macs"])
    lda_macs = _lda_fit_macs(len(old_rows), len(old_registry)) + _lda_fit_macs(
        len(all_rows), len(all_registry)
    )
    geometry = {
        "schema": "cvs.phase2.d42.unified_shrinkage_lda_geometry.v1",
        "feature_geometry": "exact_D38_B3_288d",
        "metric_source": "D42_private_old_only_exact_D38_fullbatch_B20_formula",
        "old_only_metric_helper_called_once": True,
        "old_only_metric_new_support_argument_count": 0,
        "before_materialized_pre_stage2c": True,
        "before_materialization_optimizer_step": METRIC_EPOCHS,
        "metric_frozen_during_stage2c": True,
        "stage2b_classifier": "old_only_unified_auto_shrinkage_lda",
        "stage2c_classifier": "all_registry_unified_auto_shrinkage_lda",
        "stage2c_refits_old_and_new_jointly": True,
        "lda_solver": "lsqr",
        "shrinkage": "auto",
        "sklearn_runtime_version": SKLEARN_RUNTIME_VERSION,
        "sklearn_runtime_version_lock_pass": True,
        "prior_policy": "equal_1_over_registered_class_count",
        "lda_coefficient_semantics": (
            "precision_weighted_target_prototype_w_c_equals_sigma_inverse_mu_c"
        ),
        "before_covariance_audit": dict(before_lda),
        "final_covariance_audit": dict(final_lda),
        "k1_unit_covariance_fallback": bool(
            before_lda["unit_covariance_fallback"]
            or final_lda["unit_covariance_fallback"]
        ),
        "k_le2_unit_covariance_fallback": bool(
            before_lda["unit_covariance_fallback"]
            or final_lda["unit_covariance_fallback"]
        ),
        "class_id_specific_branch": False,
        "label_permutation_equivariant": True,
        "formal_coefficient_storage": "three_block_two_level_residual_int8",
        "formal_old_target_vectors_residual_int8": True,
        "formal_new_target_vectors_residual_int8": True,
        "formal_target_vector_semantics": "precision_weighted_target_prototype",
        "formal_coefficient_shape": [len(all_registry), FEATURE_DIM],
        "formal_coefficient_dtype": "int8_two_level_codes_fp16_block_scales",
        "formal_intercept_storage": "fp16",
        "formal_intercept_shape": [len(all_registry)],
        "formal_intercept_dtype": "float16",
        "fp32_target_coefficient_stored_in_formal_state": False,
        "class_means_persisted_in_formal_state": False,
        "shared_covariance_persisted_in_formal_state": False,
        "matched_fp32_shared_lda_solution": True,
        "target_old_int8_coefficient_used_for_formal_prediction": True,
        "target_new_int8_coefficient_used_for_formal_prediction": True,
        "before_state_immutable_during_stage2c": True,
        "ground_int8_component_input_count": 0,
        "ground_int8_update_access": False,
        "support_view_count": 1,
        "physical_support_observation_multiplicity": 1,
        "query_view": "full_288d_only",
        "query_view_count": 1,
        "query_rows_used": 0,
        **before_quant,
        "before_coefficient_quantization_error_mean": before_quant[
            "coefficient_quantization_error_mean"
        ],
        "before_coefficient_quantization_error_max": before_quant[
            "coefficient_quantization_error_max"
        ],
        "before_intercept_quantization_error_mean": before_quant[
            "intercept_quantization_error_mean"
        ],
        "before_intercept_quantization_error_max": before_quant[
            "intercept_quantization_error_max"
        ],
        "final_coefficient_quantization_error_mean": final_quant[
            "coefficient_quantization_error_mean"
        ],
        "final_coefficient_quantization_error_max": final_quant[
            "coefficient_quantization_error_max"
        ],
        "final_intercept_quantization_error_mean": final_quant[
            "intercept_quantization_error_mean"
        ],
        "final_intercept_quantization_error_max": final_quant[
            "intercept_quantization_error_max"
        ],
        "before_support_score_max_abs_error": before_score_error,
        "final_support_score_max_abs_error": final_score_error,
        "int8_vs_fp32_before_support_argmax_change_count": before_argmax_changes,
        "int8_vs_fp32_final_support_argmax_change_count": final_argmax_changes,
    }
    resource = {
        "schema": "cvs.phase2.d42.unified_shrinkage_lda_resource.v1",
        "support_only": True,
        "trainable_parameters": int(trainable_parameters),
        "trainable_parameter_cap": MAX_TRAINABLE_PARAMETERS,
        "trainable_parameter_cap_pass": trainable_parameters
        <= MAX_TRAINABLE_PARAMETERS,
        "metric_epochs": METRIC_EPOCHS,
        "adaptation_epochs": METRIC_EPOCHS,
        "adaptation_epoch_cap": 30,
        "adaptation_epoch_cap_pass": METRIC_EPOCHS <= 30,
        "metric_optimizer_steps": METRIC_EPOCHS,
        "stage2b_optimizer_steps": METRIC_EPOCHS,
        "stage2c_optimizer_steps": 0,
        "lda_optimizer_steps": 0,
        "optimizer_steps": METRIC_EPOCHS,
        "optimizer_step_cap": 50,
        "optimizer_step_cap_pass": METRIC_EPOCHS <= 50,
        "lda_closed_form_fit_count": 2,
        "sklearn_runtime_version": SKLEARN_RUNTIME_VERSION,
        "sklearn_runtime_version_lock_pass": True,
        "persistent_state_bytes": state.persistent_state_bytes,
        "registry_state_bytes": state.registry_state_bytes,
        "persistent_state_cap_bytes": MAX_PERSISTENT_STATE_BYTES,
        "persistent_state_cap_pass": state.persistent_state_bytes
        <= MAX_PERSISTENT_STATE_BYTES,
        "estimated_metric_adaptation_macs": metric_macs,
        "estimated_lda_fit_macs": lda_macs,
        "estimated_adaptation_macs": metric_macs + lda_macs,
        "estimated_macs_per_query": int(
            FEATURE_DIM + 2 * FEATURE_DIM * len(all_registry)
        ),
        "support_view_count": 1,
        "physical_support_observation_multiplicity": 1,
        "query_view": "full_288d_only",
        "query_view_count": 1,
        "dense_query_graph_bytes": 0,
        "query_dependent_batch_optimization": False,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_features_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "clean_sample_access": False,
        "source_sample_access": False,
        "ground_int8_component_input_count": 0,
        "ground_int8_update_access": False,
        "formal_target_vectors_int8_no_fp32_sidecar": True,
        "resident_fp32_target_coefficient_count": 0,
        "old_k_shot": old_k,
        "new_k_shot": new_k,
        "registered_class_count": len(all_registry),
        "runtime_device": str(runtime_device),
        "peak_cuda_memory_bytes": int(
            metric_resource["peak_cuda_memory_bytes"]
        ),
        "peak_cuda_memory_scope": "metric_fit_cuda_allocations_only",
        "host_fp64_covariance_peak_memory_bytes": None,
        "host_fp64_covariance_peak_memory_measured": False,
        "int8_vs_fp32_before_support_argmax_change_count": before_argmax_changes,
        "int8_vs_fp32_final_support_argmax_change_count": final_argmax_changes,
    }
    return D42UnifiedShrinkageLDAResult(
        before_state=before_state,
        state=state,
        matched_fp32_before_state=matched_before,
        matched_fp32_state=matched_state,
        training_trace=trace,
        geometry_audit=geometry,
        resource_audit=resource,
    )


def score_d42_unified_shrinkage_lda(
    state: D42UnifiedShrinkageLDAState, features: np.ndarray
) -> np.ndarray:
    if not isinstance(state, D42UnifiedShrinkageLDAState):
        raise D42UnifiedShrinkageLDAError("D42 score state drift")
    transformed = _transform(features, state.log_diag_fp32)
    coefficients = decode_d42_coefficients(state)
    intercept = (
        state.intercept_fp16.astype(np.float32)
        if state.is_int8
        else state.intercept_fp32
    )
    scores = np.stack(
        [
            np.asarray(row @ coefficients.T + intercept, dtype=np.float32)
            for row in transformed
        ]
    )
    if not np.isfinite(scores).all():
        raise D42UnifiedShrinkageLDAError("D42 score became non-finite")
    return _readonly(scores, np.float32)


def predict_d42_unified_shrinkage_lda(
    state: D42UnifiedShrinkageLDAState, features: np.ndarray
) -> np.ndarray:
    return np.asarray(state.classes)[
        np.argmax(score_d42_unified_shrinkage_lda(state, features), axis=1)
    ]


def pairwise_support_diagnostics_d42(
    state: D42UnifiedShrinkageLDAState,
    held_features: np.ndarray,
    held_labels: Sequence[str],
    physical_ids: Sequence[str],
    *,
    scenario: str,
    outer_fold: int,
    physical_ranks: Sequence[int],
) -> list[dict[str, Any]]:
    labels = tuple(str(value) for value in held_labels)
    ids = tuple(str(value) for value in physical_ids)
    ranks = tuple(int(value) for value in physical_ranks)
    scores = score_d42_unified_shrinkage_lda(state, held_features)
    old_count = int(state.old_class_count)
    if (
        len(state.classes) == old_count
        or len(labels) != len(scores)
        or len(ids) != len(scores)
        or len(ranks) != len(scores)
        or len(set(ids)) != len(ids)
        or any(label not in state.classes for label in labels)
        or not str(scenario)
    ):
        raise D42UnifiedShrinkageLDAError("D42 pairwise diagnostic closure drift")
    output: list[dict[str, Any]] = []
    for row, truth, physical_id, rank in zip(
        scores, labels, ids, ranks, strict=True
    ):
        truth_index = state.classes.index(truth)
        truth_is_old = truth_index < old_count
        top_old_index = int(np.argmax(row[:old_count]))
        top_new_index = old_count + int(np.argmax(row[old_count:]))
        old_to_new_margin: float | None = None
        new_to_old_margin: float | None = None
        new_new_margin: float | None = None
        top_competing_new_handle: str | None = None
        top_competing_new_score: float | None = None
        if truth_is_old:
            old_to_new_margin = float(row[truth_index] - row[top_new_index])
        else:
            new_to_old_margin = float(row[truth_index] - row[top_old_index])
            competing_new = np.array(row[old_count:], copy=True)
            competing_new[truth_index - old_count] = -np.inf
            competitor_index = old_count + int(np.argmax(competing_new))
            new_new_margin = float(row[truth_index] - row[competitor_index])
            top_competing_new_handle = state.classes[competitor_index]
            top_competing_new_score = float(row[competitor_index])
        output.append(
            {
                "scenario": str(scenario),
                "outer_fold": int(outer_fold),
                "physical_rank": rank,
                "physical_sample_id": physical_id,
                "true_handle": truth,
                "true_role": "old" if truth_is_old else "new",
                "true_score": float(row[truth_index]),
                "top_old_handle": state.classes[top_old_index],
                "top_old_score": float(row[top_old_index]),
                "top_new_handle": state.classes[top_new_index],
                "top_new_score": float(row[top_new_index]),
                "top_competing_new_handle": top_competing_new_handle,
                "top_competing_new_score": top_competing_new_score,
                "old_to_new_margin": old_to_new_margin,
                "new_to_old_margin": new_to_old_margin,
                "new_new_margin": new_new_margin,
                "query_rows_used": 0,
            }
        )
    return output


__all__ = [
    "ALLOWED_NEW_CLASS_COUNTS",
    "BLOCK_DIMS",
    "D42UnifiedShrinkageLDAConfig",
    "D42UnifiedShrinkageLDAError",
    "D42UnifiedShrinkageLDAResult",
    "D42UnifiedShrinkageLDAState",
    "FEATURE_DIM",
    "METRIC_EPOCHS",
    "decode_d42_coefficients",
    "fit_d42_unified_shrinkage_lda",
    "pairwise_support_diagnostics_d42",
    "predict_d42_unified_shrinkage_lda",
    "score_d42_unified_shrinkage_lda",
]
