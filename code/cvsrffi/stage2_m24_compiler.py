"""Compact persistent inference state and quantization audit for M2.4."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_ablation_quantization import (
    F0,
    F3,
    CompiledAffineState,
    compile_affine_state,
    decode_affine_state,
    score_affine_state,
)


INFERENCE_SCHEMA = "cvs.erbt_idr.m24.inference_state.v1"


def margin_normalized_quantization_audit(reference_scores: Any, compiled_scores: Any) -> dict[str, float]:
    reference = np.asarray(reference_scores, dtype=np.float64)
    compiled = np.asarray(compiled_scores, dtype=np.float64)
    if reference.shape != compiled.shape or reference.ndim != 2 or reference.shape[1] < 2:
        raise ValueError("score matrices must have the same N x C shape with C >= 2")
    ordered = np.sort(reference, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    maximum_error = np.max(np.abs(reference - compiled), axis=1)
    ratio = maximum_error / (margin + 1.0e-12)
    return {
        "r_p50": float(np.percentile(ratio, 50)),
        "r_p95": float(np.percentile(ratio, 95)),
        "r_p99": float(np.percentile(ratio, 99)),
        "r_max": float(np.max(ratio)),
        "fraction_r_gt_0_1": float(np.mean(ratio > 0.1)),
        "fraction_r_gt_0_5": float(np.mean(ratio > 0.5)),
        "max_logit_abs_error": float(np.max(np.abs(reference - compiled))),
    }


@dataclass(frozen=True)
class M24InferenceState:
    schema: str
    classes: tuple[str, ...]
    compiled_affine_state: CompiledAffineState
    input_log_diag_fp32: np.ndarray
    domain_digest: str
    config_hash: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema != INFERENCE_SCHEMA or len(self.classes) != self.compiled_affine_state.class_count:
            raise ValueError("M2.4 inference state identity drift")
        log_diag = np.asarray(self.input_log_diag_fp32, dtype=np.float32)
        if log_diag.shape not in {(0,), (self.compiled_affine_state.feature_dim,)} or not np.isfinite(log_diag).all():
            raise ValueError("M2.4 frozen input metric drift")
        log_diag = np.array(log_diag, copy=True)
        log_diag.setflags(write=False)
        object.__setattr__(self, "input_log_diag_fp32", log_diag)
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def score(self, features: Any) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float32)
        if self.input_log_diag_fp32.size:
            if rows.ndim != 2 or rows.shape[1] != self.compiled_affine_state.feature_dim:
                raise ValueError("M2.4 query feature geometry drift")
            rows = rows * np.exp(self.input_log_diag_fp32)[None, :]
            norm = np.linalg.norm(rows, axis=1, keepdims=True)
            if not np.isfinite(norm).all() or np.any(norm <= 1.0e-12):
                raise ValueError("M2.4 frozen input metric is degenerate")
            rows = np.asarray(rows / norm, dtype=np.float32)
        return np.asarray(score_affine_state(self.compiled_affine_state, rows), dtype=np.float32)

    def predict(self, features: Any) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(features), axis=-1)]


def compile_m24_head(
    coefficient: Any,
    bias: Any,
    *,
    classes: Sequence[str],
    domain_digest: str,
    config_hash: str,
    support_features: Any,
    transient_workspace_bytes: int,
    block_sizes: Sequence[int] | None = None,
    input_log_diag: Any | None = None,
    compile_arm: str = F3,
    frozen_compiled_affine_state: CompiledAffineState | None = None,
) -> tuple[M24InferenceState, dict[str, int], dict[str, float]]:
    reference_coefficient = np.asarray(coefficient, dtype=np.float32)
    reference_bias = np.asarray(bias, dtype=np.float32)
    support = np.asarray(support_features, dtype=np.float32)
    frozen_log_diag = np.empty(0, dtype=np.float32) if input_log_diag is None else np.asarray(input_log_diag, dtype=np.float32)
    if frozen_log_diag.size:
        if frozen_log_diag.shape != (reference_coefficient.shape[1],) or not np.isfinite(frozen_log_diag).all():
            raise ValueError("input_log_diag must match the affine feature dimension")
        support = support * np.exp(frozen_log_diag)[None, :]
        norm = np.linalg.norm(support, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or np.any(norm <= 1.0e-12):
            raise ValueError("input_log_diag produces degenerate support")
        support = np.asarray(support / norm, dtype=np.float32)
    if frozen_compiled_affine_state is None:
        compiled = compile_affine_state(
            reference_coefficient,
            reference_bias,
            arm_id=compile_arm,
            block_sizes=block_sizes,
        )
    else:
        source = frozen_compiled_affine_state
        feature_dim = reference_coefficient.shape[1]
        if feature_dim not in source.block_offsets[1:]:
            raise ValueError("frozen affine prefix must end on a source block boundary")
        source_coefficient, source_bias = decode_affine_state(source)
        if (
            source.class_count != reference_coefficient.shape[0]
            or source.feature_dim < feature_dim
            or not np.array_equal(source_coefficient[:, :feature_dim], reference_coefficient)
            or not np.array_equal(source_bias, reference_bias)
        ):
            raise ValueError("frozen affine source does not match the supplied reference")
        block_count = source.block_offsets.index(feature_dim)

        def readonly_prefix(value: np.ndarray, *, columns: int | None = None) -> np.ndarray:
            result = np.array(
                value if columns is None else value[:, :columns], copy=True, order="C"
            )
            result.setflags(write=False)
            return result

        compiled = CompiledAffineState(
            arm_id=source.arm_id,
            class_count=source.class_count,
            feature_dim=feature_dim,
            block_offsets=tuple(source.block_offsets[: block_count + 1]),
            coefficient_layers=tuple(
                readonly_prefix(layer, columns=feature_dim)
                for layer in source.coefficient_layers
            ),
            scale_layers=tuple(
                readonly_prefix(scale, columns=block_count)
                for scale in source.scale_layers
            ),
            bias=readonly_prefix(source.bias),
        )
    reference_scores = support @ reference_coefficient.T + reference_bias[None, :]
    compiled_scores = np.asarray(score_affine_state(compiled, support), dtype=np.float32)
    quantization = margin_normalized_quantization_audit(reference_scores, compiled_scores)
    audit = {
        "quantization": quantization,
        "has_fp32_coefficient_sidecar": False,
        "compiled_storage_arm": compiled.arm_id,
        "frozen_source_prefix_reused": frozen_compiled_affine_state is not None,
    }
    state = M24InferenceState(
        schema=INFERENCE_SCHEMA,
        classes=tuple(str(item) for item in classes),
        compiled_affine_state=compiled,
        input_log_diag_fp32=frozen_log_diag,
        domain_digest=str(domain_digest),
        config_hash=str(config_hash),
        audit=audit,
    )
    metadata_bytes = sum(len(value.encode("utf-8")) for value in (state.schema, *state.classes, state.domain_digest, state.config_hash))
    resource = {
        "compiled_inference_state_bytes": int(compiled.state_bytes + frozen_log_diag.nbytes + metadata_bytes),
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": int(transient_workspace_bytes),
    }
    return state, resource, quantization


__all__ = ["INFERENCE_SCHEMA", "M24InferenceState", "compile_m24_head", "margin_normalized_quantization_audit"]
