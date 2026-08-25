"""True-dimension D92 feature integration for the M2.9 TASR48 screen."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import threading
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.stage2_m29_tasr import (
    Phase1TASRBundle,
    TargetSpectralCalibration,
    transform_tasr48,
)
from cvsrffi.stage2_m24_compiler import M24InferenceState, compile_m24_head


FFT_ALPHA4 = "M29-FFT96-A4"
FFT_ALPHA1 = "M29-FFT96-A1"
FFT_ALPHA05 = "M29-FFT96-A05"
IDENTITY_ONLY = "M29-IDENTITY160"
TASR_ALPHA1 = "M29-TASR48-A1"
M29_ARMS = (FFT_ALPHA4, FFT_ALPHA1, FFT_ALPHA05, IDENTITY_ONLY, TASR_ALPHA1)

_ARM_ALPHA = {FFT_ALPHA4: 4.0, FFT_ALPHA1: 1.0, FFT_ALPHA05: 0.5, TASR_ALPHA1: 1.0}
_GEOMETRY_LOCK = threading.RLock()
_EPS = 1.0e-12


def _rows(value: Any, dimension: int, name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != dimension or len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be finite nonempty N x {dimension}")
    return rows


def _unit(value: Any) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= _EPS):
        raise ValueError("feature block is degenerate")
    return rows / norm


def arm_block_dims(arm: str) -> tuple[int, ...]:
    if arm == IDENTITY_ONLY:
        return (160,)
    if arm in {FFT_ALPHA4, FFT_ALPHA1, FFT_ALPHA05}:
        return (160, 96)
    if arm == TASR_ALPHA1:
        return (160, 48)
    raise ValueError(f"unknown M2.9 arm: {arm}")


def make_m29_features(
    identity160: Any,
    fft96: Any,
    *,
    arm: str,
    tasr_bundle: Phase1TASRBundle | None = None,
    calibration: TargetSpectralCalibration | None = None,
) -> np.ndarray:
    """Build the exact active representation; no zero padding is permitted."""

    identity = _unit(_rows(identity160, 160, "identity160"))
    fft = _rows(fft96, 96, "FFT96")
    if len(identity) != len(fft):
        raise ValueError("identity/FFT row count drift")
    if arm == IDENTITY_ONLY:
        return np.asarray(identity, dtype=np.float32)
    if arm in {FFT_ALPHA4, FFT_ALPHA1, FFT_ALPHA05}:
        auxiliary = _unit(fft)
    elif arm == TASR_ALPHA1:
        if tasr_bundle is None or calibration is None:
            raise ValueError("TASR48 requires a frozen bundle and calibration")
        auxiliary = transform_tasr48(fft, calibration, tasr_bundle).astype(np.float64)
    else:
        raise ValueError(f"unknown M2.9 arm: {arm}")
    joined = np.concatenate([identity, _ARM_ALPHA[arm] * auxiliary], axis=1)
    return np.asarray(_unit(joined), dtype=np.float32)


@contextmanager
def d92_feature_geometry(block_dims: Sequence[int]) -> Iterator[None]:
    """Temporarily bind legacy D92 helpers to one true feature geometry."""

    dims = tuple(int(value) for value in block_dims)
    if not dims or dims[0] != 160 or any(value <= 0 for value in dims):
        raise ValueError("D92 feature geometry must start with identity160")
    offsets = np.cumsum((0, *dims)).tolist()
    slices = tuple(slice(offsets[index], offsets[index + 1]) for index in range(len(dims)))
    with _GEOMETRY_LOCK:
        original = (d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS)
        d42.FEATURE_DIM = int(sum(dims))
        d42.BLOCK_SLICES = slices
        d42.BLOCK_DIMS = dims
        try:
            yield
        finally:
            d42.FEATURE_DIM, d42.BLOCK_SLICES, d42.BLOCK_DIMS = original


def _registry(labels: Any, classes: Sequence[str], name: str) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, int]:
    registry = tuple(str(value) for value in classes)
    rows = np.asarray(labels).astype(str)
    if not registry or len(set(registry)) != len(registry) or set(rows.tolist()) != set(registry):
        raise ValueError(f"{name} registry drift")
    lookup = {value: index for index, value in enumerate(registry)}
    targets = np.asarray([lookup[value] for value in rows], dtype=np.int64)
    counts = np.bincount(targets, minlength=len(registry))
    if np.any(counts <= 0) or len(set(counts.tolist())) != 1:
        raise ValueError(f"{name} must be balanced K-shot")
    return rows, registry, targets, int(counts[0])


@dataclass(frozen=True)
class M29D92State:
    arm: str
    classes: tuple[str, ...]
    old_class_count: int
    tasr_bundle: Phase1TASRBundle | None
    calibration: TargetSpectralCalibration | None
    inference: M24InferenceState
    audit: Mapping[str, Any]
    resource: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm not in M29_ARMS or len(self.classes) < 2 or self.old_class_count != 6:
            raise ValueError("M2.9 D92 state identity drift")
        if self.arm == TASR_ALPHA1 and (self.tasr_bundle is None or self.calibration is None):
            raise ValueError("M2.9 TASR state is incomplete")
        if self.inference.compiled_affine_state.feature_dim != sum(arm_block_dims(self.arm)):
            raise ValueError("M2.9 compiled feature geometry drift")
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))
        object.__setattr__(self, "resource", MappingProxyType(dict(self.resource)))

    def score(self, identity160: Any, fft96: Any) -> np.ndarray:
        features = make_m29_features(
            identity160,
            fft96,
            arm=self.arm,
            tasr_bundle=self.tasr_bundle,
            calibration=self.calibration,
        )
        return self.inference.score(features)

    def predict(self, identity160: Any, fft96: Any) -> np.ndarray:
        return np.asarray(self.classes)[np.argmax(self.score(identity160, fft96), axis=1)]


def fit_m29_d92(
    *,
    arm: str,
    old_identity160: Any,
    old_fft96: Any,
    old_labels: Any,
    old_classes: Sequence[str],
    new_identity160: Any | None = None,
    new_fft96: Any | None = None,
    new_labels: Any | None = None,
    new_classes: Sequence[str] = (),
    tasr_bundle: Phase1TASRBundle | None = None,
    seed: int,
    device: Any = "cpu",
) -> M29D92State:
    """Fit one truth-blind D92 state from balanced target support only."""

    from cvsrffi import stage2_ablation_executors as executors

    old_identity = _rows(old_identity160, 160, "old identity160")
    old_fft = _rows(old_fft96, 96, "old FFT96")
    old_label_rows, old_registry, old_targets, old_k = _registry(old_labels, old_classes, "old support")
    if len(old_registry) != 6 or len(old_identity) != len(old_fft) or len(old_identity) != len(old_label_rows):
        raise ValueError("M2.9 requires the frozen six-old-class support geometry")
    if arm == TASR_ALPHA1 and (tasr_bundle is None or tuple(tasr_bundle.class_registry) != old_registry):
        raise ValueError("TASR Phase1 bundle/old-class registry drift")

    classes = old_registry
    labels = old_label_rows
    targets = old_targets
    identity = old_identity
    fft = old_fft
    new_registry = tuple(str(value) for value in new_classes)
    if new_registry:
        new_identity = _rows(new_identity160, 160, "new identity160")
        new_fft = _rows(new_fft96, 96, "new FFT96")
        new_label_rows, new_registry, new_targets, new_k = _registry(new_labels, new_registry, "new support")
        if old_k != new_k or set(old_registry) & set(new_registry) or len(new_identity) != len(new_fft) or len(new_identity) != len(new_label_rows):
            raise ValueError("M2.9 old/new support geometry drift")
        identity = np.concatenate([old_identity, new_identity], axis=0)
        fft = np.concatenate([old_fft, new_fft], axis=0)
        labels = np.concatenate([old_label_rows, new_label_rows])
        targets = np.concatenate([old_targets, new_targets + len(old_registry)])
        classes = old_registry + new_registry

    calibration = None
    if arm == TASR_ALPHA1:
        from cvsrffi.stage2_m29_tasr import estimate_target_spectral_calibration

        calibration = estimate_target_spectral_calibration(fft, labels, tasr_bundle)
    features = make_m29_features(
        identity,
        fft,
        arm=arm,
        tasr_bundle=tasr_bundle,
        calibration=calibration,
    )
    old_features = features[: len(old_identity)]
    dims = arm_block_dims(arm)
    started = time.perf_counter()
    with d92_feature_geometry(dims):
        log_diag, trace, metric_resource = executors._metric(
            old_features,
            old_targets,
            len(old_registry),
            enabled=True,
            seed=int(seed),
            device=device,
        )
        transformed = d42._transform(features, log_diag)
        fit, method = executors._component_builder(
            "P2-B0",
            ground_basis=np.empty((160, 0), dtype=np.float64),
            ground_weights=np.empty(0, dtype=np.float64),
            ground_audit={},
        )
        coefficient, intercept, fit_audit = executors._fit_with_fp32_centering_audit(
            fit, transformed, targets, len(classes), old_k
        )
    fit_seconds = time.perf_counter() - started
    config = {
        "arm": arm,
        "block_dims": list(dims),
        "classes": list(classes),
        "old_class_count": len(old_registry),
        "seed": int(seed),
        "tasr_component_id": None if tasr_bundle is None else tasr_bundle.component_id,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    inference, compile_resource, compile_audit = compile_m24_head(
        coefficient,
        intercept,
        classes=classes,
        domain_digest=config_hash,
        config_hash=config_hash,
        support_features=features,
        transient_workspace_bytes=int(transformed.nbytes + coefficient.nbytes + intercept.nbytes),
        block_sizes=dims,
        input_log_diag=log_diag,
    )
    bundle_bytes = 0 if tasr_bundle is None else int(tasr_bundle.state_bytes)
    calibration_bytes = 0 if calibration is None else int(calibration.delta.nbytes + calibration.shrinkage.nbytes)
    audit = {
        **dict(fit_audit),
        "numerical_method": method,
        "feature_geometry": list(dims),
        "true_feature_dimension": int(sum(dims)),
        "zero_padding_used": False,
        "support_only": True,
        "query_rows_used": 0,
        "tasr_calibration_frozen": calibration is not None,
        "compiler": compile_audit,
    }
    resource = {
        **dict(metric_resource),
        **dict(compile_resource),
        "fit_seconds": float(fit_seconds),
        "optimizer_steps": len(trace),
        "phase1_bundle_bytes": bundle_bytes,
        "target_calibration_bytes": calibration_bytes,
        "total_deployment_state_bytes": int(compile_resource["compiled_inference_state_bytes"] + bundle_bytes + calibration_bytes),
    }
    return M29D92State(
        arm=arm,
        classes=classes,
        old_class_count=len(old_registry),
        tasr_bundle=tasr_bundle,
        calibration=calibration,
        inference=inference,
        audit=audit,
        resource=resource,
    )


__all__ = [
    "FFT_ALPHA4",
    "FFT_ALPHA1",
    "FFT_ALPHA05",
    "IDENTITY_ONLY",
    "TASR_ALPHA1",
    "M29_ARMS",
    "M29D92State",
    "arm_block_dims",
    "d92_feature_geometry",
    "fit_m29_d92",
    "make_m29_features",
]
