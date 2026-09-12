"""F0-F3 deployable affine-state compilation for the Stage2 ablation.

The compiler starts from an already fitted affine head ``coef @ x + bias``.
It neither fits from data nor accepts labels, truth, or query-side state.  F2
and F3 keep only INT8 coefficient layers and FP16 per-class-by-block scales;
decoded FP32 coefficients are transient values and are never part of the
persistent state.

The scoring path deliberately decodes to FP32 before the dot product.  It
therefore proves storage compression only, not INT8 inference acceleration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


F0 = "P2-F0"
F1 = "P2-F1"
F2 = "P2-F2"
F3 = "P2-F3"
SUPPORTED_ARMS = (F0, F1, F2, F3)

QUANTIZATION_RECEIPT_SCHEMA = (
    "cvs.full_ablation.phase2.quantization_receipt.v1"
)
RESOURCE_SCHEMA = "cvs.stage2.ablation.affine_resource.v1"
DECODE_COST_SCHEMA = "cvs.stage2.ablation.affine_decode_cost.v1"

_INT8_DENOMINATOR = 127.0
_DEFAULT_BLOCK_SIZES = (160, 96, 32)


class Stage2AblationQuantizationError(ValueError):
    """Raised when an affine state cannot be compiled or evaluated safely."""


def _readonly(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.isfinite(result).all():
        raise Stage2AblationQuantizationError(
            f"value cannot be represented as finite {result.dtype}"
        )
    result.setflags(write=False)
    return result


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or min(result.shape, default=0) <= 0:
        raise Stage2AblationQuantizationError(
            f"{name} must be a nonempty two-dimensional array"
        )
    if not np.issubdtype(result.dtype, np.number):
        raise Stage2AblationQuantizationError(f"{name} must be numeric")
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise Stage2AblationQuantizationError(f"{name} must be finite")
    return result


def _finite_bias(value: Any, *, class_count: int) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (class_count,):
        raise Stage2AblationQuantizationError(
            f"bias must have shape ({class_count},)"
        )
    if not np.issubdtype(result.dtype, np.number):
        raise Stage2AblationQuantizationError("bias must be numeric")
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise Stage2AblationQuantizationError("bias must be finite")
    return result


def _block_offsets(
    feature_dim: int, block_sizes: Sequence[int] | None
) -> tuple[int, ...]:
    sizes = (
        _DEFAULT_BLOCK_SIZES
        if block_sizes is None and feature_dim == sum(_DEFAULT_BLOCK_SIZES)
        else (feature_dim,)
        if block_sizes is None
        else tuple(int(value) for value in block_sizes)
    )
    if not sizes or any(value <= 0 for value in sizes):
        raise Stage2AblationQuantizationError(
            "block_sizes must contain positive integers"
        )
    if sum(sizes) != feature_dim:
        raise Stage2AblationQuantizationError(
            "block_sizes must sum to the affine feature dimension"
        )
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return tuple(offsets)


def _quantize_layer(
    coefficient: np.ndarray, block_offsets: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize one coefficient layer and return INT8, FP16 scale, decoded."""

    class_count = coefficient.shape[0]
    block_count = len(block_offsets) - 1
    quantized = np.zeros(coefficient.shape, dtype=np.int8)
    scales = np.empty((class_count, block_count), dtype=np.float16)
    decoded = np.empty(coefficient.shape, dtype=np.float32)
    minimum_scale = np.nextafter(np.float16(0), np.float16(1))

    for class_index in range(class_count):
        for block_index, (start, stop) in enumerate(
            zip(block_offsets[:-1], block_offsets[1:])
        ):
            block = coefficient[class_index, start:stop]
            maximum = float(np.max(np.abs(block)))
            if maximum == 0.0:
                stored_scale = np.float16(1.0)
                qblock = np.zeros(block.shape, dtype=np.int8)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    stored_scale = np.float16(
                        maximum / _INT8_DENOMINATOR
                    )
                if not np.isfinite(stored_scale):
                    raise Stage2AblationQuantizationError(
                        "coefficient block cannot be represented by an FP16 scale"
                    )
                if stored_scale == 0.0:
                    stored_scale = minimum_scale
                qblock = np.rint(
                    np.clip(
                        block / np.float32(stored_scale),
                        -_INT8_DENOMINATOR,
                        _INT8_DENOMINATOR,
                    )
                ).astype(np.int8)
            quantized[class_index, start:stop] = qblock
            scales[class_index, block_index] = stored_scale
            decoded[class_index, start:stop] = (
                qblock.astype(np.float32) * np.float32(stored_scale)
            )

    return quantized, scales, decoded


@dataclass(frozen=True)
class CompiledAffineState:
    """Persistent affine state for one F0-F3 arm."""

    arm_id: str
    class_count: int
    feature_dim: int
    block_offsets: tuple[int, ...]
    coefficient_layers: tuple[np.ndarray, ...]
    scale_layers: tuple[np.ndarray, ...]
    bias: np.ndarray

    def __post_init__(self) -> None:
        if self.arm_id not in SUPPORTED_ARMS:
            raise Stage2AblationQuantizationError(
                f"unsupported quantization arm: {self.arm_id!r}"
            )
        if self.class_count <= 0 or self.feature_dim <= 0:
            raise Stage2AblationQuantizationError(
                "compiled affine dimensions must be positive"
            )
        if (
            len(self.block_offsets) < 2
            or self.block_offsets[0] != 0
            or self.block_offsets[-1] != self.feature_dim
            or any(
                right <= left
                for left, right in zip(
                    self.block_offsets[:-1], self.block_offsets[1:]
                )
            )
        ):
            raise Stage2AblationQuantizationError(
                "compiled affine block offsets are invalid"
            )
        expected_shape = (self.class_count, self.feature_dim)
        expected_scale_shape = (
            self.class_count,
            len(self.block_offsets) - 1,
        )
        for layer in self.coefficient_layers:
            if layer.shape != expected_shape or layer.flags.writeable:
                raise Stage2AblationQuantizationError(
                    "coefficient layer shape or immutability drift"
                )
        for scale in self.scale_layers:
            if (
                scale.shape != expected_scale_shape
                or scale.dtype != np.float16
                or scale.flags.writeable
            ):
                raise Stage2AblationQuantizationError(
                    "scale layer shape, dtype, or immutability drift"
                )
        if self.bias.shape != (self.class_count,) or self.bias.flags.writeable:
            raise Stage2AblationQuantizationError(
                "bias shape or immutability drift"
            )

        expected = {
            F0: ((np.dtype(np.float32),), (), np.dtype(np.float32)),
            F1: ((np.dtype(np.float16),), (), np.dtype(np.float16)),
            F2: ((np.dtype(np.int8),), (np.dtype(np.float16),), np.dtype(np.float16)),
            F3: (
                (np.dtype(np.int8), np.dtype(np.int8)),
                (np.dtype(np.float16), np.dtype(np.float16)),
                np.dtype(np.float16),
            ),
        }[self.arm_id]
        coefficient_dtypes, scale_dtypes, bias_dtype = expected
        if tuple(layer.dtype for layer in self.coefficient_layers) != coefficient_dtypes:
            raise Stage2AblationQuantizationError(
                "coefficient storage dtype does not match the arm"
            )
        if tuple(scale.dtype for scale in self.scale_layers) != scale_dtypes:
            raise Stage2AblationQuantizationError(
                "scale storage dtype does not match the arm"
            )
        if self.bias.dtype != bias_dtype:
            raise Stage2AblationQuantizationError(
                "bias storage dtype does not match the arm"
            )

    @property
    def state_bytes(self) -> int:
        """Exact persistent ndarray bytes; transient decoded arrays are excluded."""

        arrays = (*self.coefficient_layers, *self.scale_layers, self.bias)
        return int(sum(value.nbytes for value in arrays))

    @property
    def coefficient_state_bytes(self) -> int:
        return int(sum(value.nbytes for value in self.coefficient_layers))

    @property
    def scale_state_bytes(self) -> int:
        return int(sum(value.nbytes for value in self.scale_layers))

    @property
    def bias_state_bytes(self) -> int:
        return int(self.bias.nbytes)

    @property
    def has_fp32_coefficient_sidecar(self) -> bool:
        """Always false: F0's FP32 coefficient is its primary reference state."""

        return False


def compile_affine_state(
    coefficient: Any,
    bias: Any,
    *,
    arm_id: str,
    block_sizes: Sequence[int] | None = None,
) -> CompiledAffineState:
    """Compile an already-fitted affine head into one frozen F0-F3 state."""

    if arm_id not in SUPPORTED_ARMS:
        raise Stage2AblationQuantizationError(
            f"unsupported quantization arm: {arm_id!r}"
        )
    reference = _finite_matrix(coefficient, name="coefficient")
    class_count, feature_dim = reference.shape
    reference_bias = _finite_bias(bias, class_count=class_count)
    offsets = _block_offsets(feature_dim, block_sizes)

    if arm_id == F0:
        coefficient_layers = (_readonly(reference, np.dtype(np.float32)),)
        scale_layers: tuple[np.ndarray, ...] = ()
        stored_bias = _readonly(reference_bias, np.dtype(np.float32))
    elif arm_id == F1:
        coefficient_layers = (_readonly(reference, np.dtype(np.float16)),)
        scale_layers = ()
        stored_bias = _readonly(reference_bias, np.dtype(np.float16))
    else:
        primary, primary_scale, decoded_primary = _quantize_layer(
            reference, offsets
        )
        coefficient_layers = (_readonly(primary, np.dtype(np.int8)),)
        scale_layers = (_readonly(primary_scale, np.dtype(np.float16)),)
        if arm_id == F3:
            residual = reference - decoded_primary
            second, second_scale, _decoded_second = _quantize_layer(
                residual, offsets
            )
            coefficient_layers = (
                *coefficient_layers,
                _readonly(second, np.dtype(np.int8)),
            )
            scale_layers = (
                *scale_layers,
                _readonly(second_scale, np.dtype(np.float16)),
            )
        # The manuscript locks the complete-reference F3 bias to FP16.
        stored_bias = _readonly(reference_bias, np.dtype(np.float16))

    return CompiledAffineState(
        arm_id=arm_id,
        class_count=class_count,
        feature_dim=feature_dim,
        block_offsets=offsets,
        coefficient_layers=coefficient_layers,
        scale_layers=scale_layers,
        bias=stored_bias,
    )


def decode_affine_state(
    state: CompiledAffineState,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one state to transient FP32 coefficient and bias arrays."""

    if not isinstance(state, CompiledAffineState):
        raise Stage2AblationQuantizationError(
            "state must be a CompiledAffineState"
        )
    if state.arm_id in {F0, F1}:
        coefficient = state.coefficient_layers[0].astype(np.float32, copy=True)
    else:
        coefficient = np.zeros(
            (state.class_count, state.feature_dim), dtype=np.float32
        )
        for quantized, scales in zip(
            state.coefficient_layers, state.scale_layers
        ):
            for block_index, (start, stop) in enumerate(
                zip(state.block_offsets[:-1], state.block_offsets[1:])
            ):
                coefficient[:, start:stop] += (
                    quantized[:, start:stop].astype(np.float32)
                    * scales[:, block_index].astype(np.float32)[:, None]
                )
    bias = state.bias.astype(np.float32, copy=True)
    return coefficient, bias


def _query_features(
    value: Any, *, feature_dim: int
) -> tuple[np.ndarray, bool]:
    result = np.asarray(value)
    squeezed = result.ndim == 1
    if squeezed:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != feature_dim or result.shape[0] <= 0:
        raise Stage2AblationQuantizationError(
            f"query_features must have shape (N, {feature_dim})"
        )
    if not np.issubdtype(result.dtype, np.number):
        raise Stage2AblationQuantizationError(
            "query_features must be numeric"
        )
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise Stage2AblationQuantizationError(
            "query_features must be finite"
        )
    return result, squeezed


def score_affine_state(
    state: CompiledAffineState, query_features: Any
) -> np.ndarray:
    """Decode transiently and score each query against every class."""

    features, squeezed = _query_features(
        query_features, feature_dim=state.feature_dim
    )
    coefficient, bias = decode_affine_state(state)
    scores = features @ coefficient.T + bias[None, :]
    return scores[0] if squeezed else scores


def quantization_receipt(
    state: CompiledAffineState,
    *,
    reference_coefficient: Any,
    reference_bias: Any,
    query_features: Any,
) -> Mapping[str, Any]:
    """Compare one compiled state with its FP32 affine reference."""

    coefficient = _finite_matrix(
        reference_coefficient, name="reference_coefficient"
    )
    if coefficient.shape != (state.class_count, state.feature_dim):
        raise Stage2AblationQuantizationError(
            "reference coefficient shape does not match the compiled state"
        )
    bias = _finite_bias(reference_bias, class_count=state.class_count)
    features, _squeezed = _query_features(
        query_features, feature_dim=state.feature_dim
    )
    reference_scores = features @ coefficient.T + bias[None, :]
    compiled_scores = np.asarray(
        score_affine_state(state, features), dtype=np.float32
    )
    error = np.abs(compiled_scores - reference_scores)
    reference_prediction = np.argmax(reference_scores, axis=1)
    compiled_prediction = np.argmax(compiled_scores, axis=1)
    flip_rate = float(np.mean(reference_prediction != compiled_prediction))
    receipt = {
        "schema": QUANTIZATION_RECEIPT_SCHEMA,
        "max_logit_abs_error": float(np.max(error)),
        "mean_logit_abs_error": float(np.mean(error)),
        "argmax_flip_rate": flip_rate,
        "prediction_agreement_rate": float(1.0 - flip_rate),
    }
    return MappingProxyType(receipt)


def decode_cost(state: CompiledAffineState) -> Mapping[str, Any]:
    """Return deterministic scalar-operation counts for transient decoding."""

    coefficient_count = state.class_count * state.feature_dim
    if state.arm_id == F0:
        multiply_count = 0
        add_count = 0
        int8_cast_count = 0
        fp16_cast_count = 0
    elif state.arm_id == F1:
        multiply_count = 0
        add_count = 0
        int8_cast_count = 0
        fp16_cast_count = coefficient_count + state.class_count
    elif state.arm_id == F2:
        multiply_count = coefficient_count
        add_count = 0
        int8_cast_count = coefficient_count
        fp16_cast_count = state.scale_layers[0].size + state.class_count
    else:
        multiply_count = 2 * coefficient_count
        add_count = coefficient_count
        int8_cast_count = 2 * coefficient_count
        fp16_cast_count = (
            sum(value.size for value in state.scale_layers)
            + state.class_count
        )
    payload = {
        "schema": DECODE_COST_SCHEMA,
        "coefficient_count": coefficient_count,
        "int8_to_fp32_cast_count": int(int8_cast_count),
        "fp16_to_fp32_cast_count": int(fp16_cast_count),
        "storage_to_fp32_cast_count": int(
            int8_cast_count + fp16_cast_count
        ),
        "scale_multiply_count": int(multiply_count),
        "residual_add_count": int(add_count),
        "decoded_fp32_temporary_bytes": int(
            coefficient_count * np.dtype(np.float32).itemsize
        ),
    }
    return MappingProxyType(payload)


def resource_report(
    state: CompiledAffineState,
    *,
    query_feature: Any,
    latency_repeats: int = 20,
    latency_warmup: int = 3,
) -> Mapping[str, Any]:
    """Measure the batch-1 decode-plus-FP32-dot path and report state bytes."""

    repeats = int(latency_repeats)
    warmup = int(latency_warmup)
    if repeats <= 0 or warmup < 0:
        raise Stage2AblationQuantizationError(
            "latency repeats must be positive and warmup must be nonnegative"
        )
    feature, _squeezed = _query_features(
        query_feature, feature_dim=state.feature_dim
    )
    if feature.shape[0] != 1:
        raise Stage2AblationQuantizationError(
            "resource_report requires exactly one batch-1 query feature"
        )

    for _ in range(warmup):
        score_affine_state(state, feature)
    observations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        score_affine_state(state, feature)
        observations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    observations_array = np.asarray(observations, dtype=np.float64)
    cost = decode_cost(state)
    payload = {
        "schema": RESOURCE_SCHEMA,
        "arm_id": state.arm_id,
        "state_bytes": state.state_bytes,
        "coefficient_state_bytes": state.coefficient_state_bytes,
        "scale_state_bytes": state.scale_state_bytes,
        "bias_state_bytes": state.bias_state_bytes,
        "has_fp32_coefficient_sidecar": state.has_fp32_coefficient_sidecar,
        "decode_cost": dict(cost),
        "batch1_latency_ms": float(np.median(observations_array)),
        "batch1_latency_mean_ms": float(np.mean(observations_array)),
        "batch1_latency_repeats": repeats,
        "latency_scope": "transient_decode_plus_fp32_dot_plus_bias",
        "query_head_mac": int(state.class_count * state.feature_dim),
        "integer_kernel_used": False,
        "formal_int8_acceleration_claim_allowed": False,
        "deployment_claim": "storage_compression_only",
    }
    if not all(
        math.isfinite(float(payload[field])) and float(payload[field]) >= 0.0
        for field in ("batch1_latency_ms", "batch1_latency_mean_ms")
    ):
        raise Stage2AblationQuantizationError(
            "batch-1 latency measurement is invalid"
        )
    return MappingProxyType(payload)


__all__ = [
    "CompiledAffineState",
    "DECODE_COST_SCHEMA",
    "F0",
    "F1",
    "F2",
    "F3",
    "QUANTIZATION_RECEIPT_SCHEMA",
    "RESOURCE_SCHEMA",
    "SUPPORTED_ARMS",
    "Stage2AblationQuantizationError",
    "compile_affine_state",
    "decode_affine_state",
    "decode_cost",
    "quantization_receipt",
    "resource_report",
    "score_affine_state",
]
