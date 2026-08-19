"""Compact persistent inference state and quantization audit for M2.4."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_ablation_quantization import F3, CompiledAffineState, compile_affine_state, score_affine_state


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
    domain_digest: str
    config_hash: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema != INFERENCE_SCHEMA or len(self.classes) != self.compiled_affine_state.class_count:
            raise ValueError("M2.4 inference state identity drift")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))

    def score(self, features: Any) -> np.ndarray:
        return np.asarray(score_affine_state(self.compiled_affine_state, features), dtype=np.float32)

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
) -> tuple[M24InferenceState, dict[str, int], dict[str, float]]:
    reference_coefficient = np.asarray(coefficient, dtype=np.float32)
    reference_bias = np.asarray(bias, dtype=np.float32)
    support = np.asarray(support_features, dtype=np.float32)
    compiled = compile_affine_state(reference_coefficient, reference_bias, arm_id=F3, block_sizes=block_sizes)
    reference_scores = support @ reference_coefficient.T + reference_bias[None, :]
    compiled_scores = np.asarray(score_affine_state(compiled, support), dtype=np.float32)
    quantization = margin_normalized_quantization_audit(reference_scores, compiled_scores)
    audit = {"quantization": quantization, "has_fp32_coefficient_sidecar": False}
    state = M24InferenceState(
        schema=INFERENCE_SCHEMA,
        classes=tuple(str(item) for item in classes),
        compiled_affine_state=compiled,
        domain_digest=str(domain_digest),
        config_hash=str(config_hash),
        audit=audit,
    )
    metadata_bytes = sum(len(value.encode("utf-8")) for value in (state.schema, *state.classes, state.domain_digest, state.config_hash))
    resource = {
        "compiled_inference_state_bytes": int(compiled.state_bytes + metadata_bytes),
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": int(transient_workspace_bytes),
    }
    return state, resource, quantization


__all__ = ["INFERENCE_SCHEMA", "M24InferenceState", "compile_m24_head", "margin_normalized_quantization_audit"]
