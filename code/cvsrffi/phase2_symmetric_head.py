"""Minimal role-symmetric support head for the sealed Phase2 runtime.

Hyperparameters are selected on source validation and supplied as a locked
configuration.  Target support is used only to fit class-symmetric moments and
prototypes; query rows, roles, quotas, and labels are never accepted here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


EPS = 1.0e-8
PROTOTYPE_RULES = {"mean", "trimmed20", "medoid", "consensus67"}
BASE_SELECTED_KEYS = {
    "use_alignment",
    "prototype_rule",
    "ridge",
    "gram_mix",
    "uncertainty_penalty",
}
EVIDENCE_SELECTED_KEY = "evidence_calibration"
EVIDENCE_CALIBRATION_KEYS = {
    "mode",
    "negative_quantile",
    "prior_physical_shots",
    "scale_floor",
    "inverse_scale_cap",
}


def _normalize(rows: np.ndarray) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), EPS)


def _alignment(
    observations: np.ndarray,
    *,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(observations, dtype=np.float32).reshape(-1, observations.shape[-1])
    mean = rows.mean(axis=0).astype(np.float32)
    std = np.maximum(rows.std(axis=0), float(variance_floor)).astype(np.float32)
    reference_mean = np.asarray(source_mean, dtype=np.float32).reshape(-1)
    reference_std = np.maximum(
        np.asarray(source_std, dtype=np.float32).reshape(-1), float(variance_floor)
    )
    if reference_mean.shape != mean.shape or reference_std.shape != std.shape:
        raise ValueError("source feature statistics do not match support dimension")
    scale = (reference_std / std).astype(np.float32)
    bias = (reference_mean - mean * scale).astype(np.float32)
    return scale, bias


def _apply_alignment(
    rows: np.ndarray, scale: np.ndarray | None, bias: np.ndarray | None
) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float32)
    if scale is None or bias is None:
        return value
    return (value * scale + bias).astype(np.float32)


def _prototypes(observations: np.ndarray, *, rule: str) -> np.ndarray:
    values = _normalize(np.asarray(observations, dtype=np.float32))
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("support observations must have shape [V,C,D]")
    if rule not in PROTOTYPE_RULES:
        raise ValueError(f"unsupported locked prototype rule: {rule}")
    result: list[np.ndarray] = []
    for class_index in range(values.shape[1]):
        rows = values[:, class_index, :]
        if rule == "mean":
            center = rows.mean(axis=0)
        elif rule == "trimmed20":
            initial = _normalize(rows.mean(axis=0, keepdims=True))[0]
            keep = np.argsort(rows @ initial)[-max(2, int(np.ceil(0.8 * len(rows)))) :]
            center = rows[keep].mean(axis=0)
        elif rule == "medoid":
            center = rows[int(np.argmax((rows @ rows.T).mean(axis=1)))]
        else:
            similarities = rows @ rows.T
            keep = np.argsort(similarities.mean(axis=1))[
                -max(2, int(np.ceil((2.0 / 3.0) * len(rows)))) :
            ]
            center = rows[keep].mean(axis=0)
        result.append(_normalize(np.asarray(center)[None, :])[0])
    return np.stack(result).astype(np.float32)


def _score_transform(prototypes: np.ndarray, *, ridge: float | None, mix: float) -> np.ndarray:
    banks = _normalize(prototypes)
    if ridge is None:
        return np.eye(len(banks), dtype=np.float32)
    ridge_value = float(ridge)
    mix_value = float(mix)
    if ridge_value <= 0.0 or not 0.0 <= mix_value <= 1.0:
        raise ValueError("locked ridge/Gram mix is invalid")
    gram = banks @ banks.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram.astype(np.float64))
    gains = np.clip(1.0 / (eigenvalues + ridge_value), 0.5, 2.0)
    inverse = (eigenvectors * gains[None, :]) @ eigenvectors.T
    identity = np.eye(len(banks), dtype=np.float64)
    return ((1.0 - mix_value) * identity + mix_value * inverse).astype(np.float32)


def _class_bias(
    observations: np.ndarray, prototypes: np.ndarray, *, penalty: float
) -> np.ndarray:
    weight = float(penalty)
    if weight < 0.0 or not np.isfinite(weight):
        raise ValueError("locked uncertainty penalty is invalid")
    values = _normalize(observations)
    banks = _normalize(prototypes)
    within = np.sum(values * banks[None, :, :], axis=-1)
    return (weight * np.maximum(0.0, 1.0 - within.mean(axis=0))).astype(np.float32)


def _validate_evidence_calibration(config: Mapping[str, Any]) -> dict[str, float | str]:
    if set(config) != EVIDENCE_CALIBRATION_KEYS:
        raise ValueError("locked evidence-calibration exact schema drift")
    mode = config.get("mode")
    if mode != "robust_lopo_class_symmetric":
        raise ValueError("unsupported locked evidence-calibration mode")
    quantile = config.get("negative_quantile")
    prior = config.get("prior_physical_shots")
    floor = config.get("scale_floor")
    inverse_cap = config.get("inverse_scale_cap")
    for name, value in (
        ("negative_quantile", quantile),
        ("prior_physical_shots", prior),
        ("scale_floor", floor),
        ("inverse_scale_cap", inverse_cap),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"locked evidence-calibration value is invalid: {name}")
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("locked evidence negative quantile must be in [0.5,1)")
    if float(prior) <= 0.0:
        raise ValueError("locked evidence prior must be positive")
    if not 0.0 < float(floor) <= 2.0:
        raise ValueError("locked evidence scale floor must be in (0,2]")
    if float(inverse_cap) < 1.0 or not np.isfinite(float(inverse_cap)):
        raise ValueError("locked evidence inverse-scale cap must be finite and >=1")
    return {
        "mode": str(mode),
        "negative_quantile": float(quantile),
        "prior_physical_shots": float(prior),
        "scale_floor": float(floor),
        "inverse_scale_cap": float(inverse_cap),
    }


def _fp16_ceil(value: float) -> np.float32:
    """Return the smallest FP16 round-trip value that is not below ``value``."""

    rounded = np.float16(value)
    if float(rounded) < float(value):
        rounded = np.nextafter(rounded, np.float16(np.inf))
    return np.float32(rounded)


def _evidence_calibration(
    observations: np.ndarray,
    prototypes: np.ndarray,
    *,
    physical_shots_per_class: int,
    prototype_rule: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a query-free, class-permutation-equivariant support score normalizer."""

    locked = _validate_evidence_calibration(config)
    raw_values = np.asarray(observations, dtype=np.float32)
    raw_banks = np.asarray(prototypes, dtype=np.float32)
    if not np.isfinite(raw_values).all() or not np.isfinite(raw_banks).all():
        raise ValueError("evidence calibration support state contains non-finite values")
    values = _normalize(raw_values)
    banks = _normalize(raw_banks)
    shots = int(physical_shots_per_class)
    view_count, class_count, _feature_dim = values.shape
    if view_count != 3 * shots:
        raise ValueError("evidence calibration requires three scenario views per shot")

    observation_indices = np.arange(view_count, dtype=np.int64)
    physical_indices = observation_indices % shots
    positives: list[float] = []
    negatives: list[float] = []
    fold_mode = "leave_one_view_out" if shots == 1 else "leave_one_physical_out"
    if class_count == 1:
        diagnostics = {
            "adaptation_type": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
            "mode": str(locked["mode"]),
            "fold_mode": fold_mode,
            "fallback": "single_registered_class_identity",
            "physical_shots_per_class": shots,
            "registered_class_count": 1,
            "negative_quantile": float(locked["negative_quantile"]),
            "prior_physical_shots": float(locked["prior_physical_shots"]),
            "scale_floor": float(locked["scale_floor"]),
            "inverse_scale_cap": float(locked["inverse_scale_cap"]),
            "shrinkage_weight": 0.0,
            "trainable_parameters": 0,
            "adapt_epochs": 0,
            "optimizer_steps": 0,
            "additional_backbone_forwards": 0,
            "state_bytes_fp16": int(2 * np.dtype(np.float16).itemsize),
            "live_array_bytes_fp32": int(2 * np.dtype(np.float32).itemsize),
            "quantized_inverse_scale_max": 1.0,
        }
        return (
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            diagnostics,
        )
    for class_index in range(class_count):
        class_rows = values[:, class_index, :]
        held_out_scores: list[float] = []
        for observation_index in range(view_count):
            if shots == 1:
                keep = observation_indices != observation_index
            else:
                keep = physical_indices != physical_indices[observation_index]
            if int(np.sum(keep)) < 1:
                raise ValueError("evidence calibration has no held-in support observation")
            held_in = class_rows[keep][:, None, :]
            held_out_prototype = _prototypes(held_in, rule=prototype_rule)[0]
            held_out_scores.append(
                float(
                    np.clip(
                        np.dot(class_rows[observation_index], held_out_prototype),
                        -1.0,
                        1.0,
                    )
                )
            )
        positives.append(float(np.median(np.asarray(held_out_scores, dtype=np.float32))))

        other_rows = values[:, np.arange(class_count) != class_index, :].reshape(
            -1, values.shape[-1]
        )
        negative_scores = np.clip(other_rows @ banks[class_index], -1.0, 1.0)
        negatives.append(
            float(
                np.quantile(
                    negative_scores,
                    float(locked["negative_quantile"]),
                    method="higher",
                )
            )
        )

    positive = np.asarray(positives, dtype=np.float32)
    negative = np.asarray(negatives, dtype=np.float32)
    raw_gap = (positive - negative).astype(np.float32)
    shrinkage_weight = float(shots) / (
        float(shots) + float(locked["prior_physical_shots"])
    )
    global_negative = float(np.median(negative))
    global_gap = float(np.median(raw_gap))
    calibrated_bias = (
        shrinkage_weight * negative + (1.0 - shrinkage_weight) * global_negative
    ).astype(np.float16).astype(np.float32)
    minimum_scale = float(
        _fp16_ceil(
            max(
                float(locked["scale_floor"]),
                1.0 / float(locked["inverse_scale_cap"]),
            )
        )
    )
    calibrated_scale = np.maximum(
        shrinkage_weight * raw_gap + (1.0 - shrinkage_weight) * global_gap,
        minimum_scale,
    ).astype(np.float16).astype(np.float32)
    if not np.isfinite(calibrated_bias).all() or not np.isfinite(calibrated_scale).all():
        raise ValueError("evidence calibration produced non-finite state")
    diagnostics = {
        "adaptation_type": "EVAL_ONLY_CLOSED_FORM_ADAPTATION",
        "mode": str(locked["mode"]),
        "fold_mode": fold_mode,
        "physical_shots_per_class": shots,
        "registered_class_count": class_count,
        "negative_quantile": float(locked["negative_quantile"]),
        "prior_physical_shots": float(locked["prior_physical_shots"]),
        "scale_floor": float(locked["scale_floor"]),
        "inverse_scale_cap": float(locked["inverse_scale_cap"]),
        "shrinkage_weight": shrinkage_weight,
        "raw_positive_median": float(np.median(positive)),
        "raw_negative_median": global_negative,
        "raw_gap_median": global_gap,
        "raw_positive_by_class": positive.astype(float).tolist(),
        "raw_negative_by_class": negative.astype(float).tolist(),
        "raw_gap_by_class": raw_gap.astype(float).tolist(),
        "calibrated_bias_by_class": calibrated_bias.astype(float).tolist(),
        "calibrated_scale_by_class": calibrated_scale.astype(float).tolist(),
        "calibrated_scale_min": float(np.min(calibrated_scale)),
        "calibrated_scale_max": float(np.max(calibrated_scale)),
        "quantized_inverse_scale_max": float(np.max(1.0 / calibrated_scale)),
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "optimizer_steps": 0,
        "additional_backbone_forwards": 0,
        "state_bytes_fp16": int(2 * class_count * np.dtype(np.float16).itemsize),
        "live_array_bytes_fp32": int(
            calibrated_bias.nbytes + calibrated_scale.nbytes
        ),
    }
    return calibrated_bias, calibrated_scale, diagnostics


def fit_locked_symmetric_head(
    support_observations: np.ndarray,
    *,
    physical_shots_per_class: int,
    selected: Mapping[str, Any],
    source_mean: np.ndarray,
    source_std: np.ndarray,
    variance_floor: float = 0.05,
) -> dict[str, Any]:
    values = np.asarray(support_observations, dtype=np.float32)
    shots = int(physical_shots_per_class)
    if values.ndim != 3 or shots < 1 or values.shape[0] != 3 * shots:
        raise ValueError("formal support head requires exactly three scenario views per shot")
    selected_keys = set(selected)
    evidence_config = selected.get(EVIDENCE_SELECTED_KEY)
    if selected_keys == BASE_SELECTED_KEYS:
        evidence_config = None
    elif selected_keys == BASE_SELECTED_KEYS | {EVIDENCE_SELECTED_KEY}:
        if not isinstance(evidence_config, Mapping):
            raise ValueError("locked evidence-calibration config must be an object")
        _validate_evidence_calibration(evidence_config)
        if (
            selected["use_alignment"] is not False
            or selected["ridge"] is not None
            or float(selected["gram_mix"]) != 0.0
            or float(selected["uncertainty_penalty"]) != 0.0
        ):
            raise ValueError(
                "evidence calibration requires no alignment, identity Gram transform, and zero uncertainty penalty"
            )
    else:
        raise ValueError("locked symmetric-head exact schema drift")
    scale: np.ndarray | None = None
    bias: np.ndarray | None = None
    if selected["use_alignment"] is True:
        scale, bias = _alignment(
            values,
            source_mean=source_mean,
            source_std=source_std,
            variance_floor=float(variance_floor),
        )
    elif selected["use_alignment"] is not False:
        raise ValueError("locked use_alignment must be boolean")
    aligned = _apply_alignment(values, scale, bias)
    prototypes = _prototypes(aligned, rule=str(selected["prototype_rule"]))
    transform = _score_transform(
        prototypes, ridge=selected["ridge"], mix=float(selected["gram_mix"])
    )
    class_bias = _class_bias(
        aligned, prototypes, penalty=float(selected["uncertainty_penalty"])
    )
    evidence_bias: np.ndarray | None = None
    evidence_scale: np.ndarray | None = None
    evidence_diagnostics: dict[str, Any] | None = None
    if evidence_config is not None:
        evidence_bias, evidence_scale, evidence_diagnostics = _evidence_calibration(
            aligned,
            prototypes,
            physical_shots_per_class=shots,
            prototype_rule=str(selected["prototype_rule"]),
            config=evidence_config,
        )
    # Formal deployment state is FP16 payload reloaded to FP32 scoring.
    def q(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else value.astype(np.float16).astype(np.float32)

    return {
        "prototypes": q(prototypes),
        "score_transform": q(transform),
        "class_bias": q(class_bias),
        "alignment_scale": q(scale),
        "alignment_bias": q(bias),
        "evidence_bias": q(evidence_bias),
        "evidence_scale": q(evidence_scale),
        "evidence_diagnostics": evidence_diagnostics,
        "physical_shots_per_class": shots,
    }


def _score_locked_symmetric_head_components(
    features: np.ndarray, head: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float32)
    prototypes = np.asarray(head["prototypes"], dtype=np.float32)
    scale = head.get("alignment_scale")
    bias = head.get("alignment_bias")
    aligned = _apply_alignment(
        rows,
        None if scale is None else np.asarray(scale, dtype=np.float32),
        None if bias is None else np.asarray(bias, dtype=np.float32),
    )
    if aligned.shape[-1] != prototypes.shape[1]:
        raise ValueError("query feature dimension does not match locked head")
    flat = _normalize(aligned.reshape(-1, aligned.shape[-1]))
    cosine_scores = flat @ _normalize(prototypes).T
    transform = np.asarray(head["score_transform"], dtype=np.float32)
    class_bias = np.asarray(head["class_bias"], dtype=np.float32)[None, :]
    raw_scores = cosine_scores @ transform - class_bias
    evidence_bias = head.get("evidence_bias")
    evidence_scale = head.get("evidence_scale")
    if (evidence_bias is None) != (evidence_scale is None):
        raise ValueError("locked evidence-calibration state is incomplete")
    if evidence_bias is not None:
        calibration_bias = np.asarray(evidence_bias, dtype=np.float32).reshape(-1)
        calibration_scale = np.asarray(evidence_scale, dtype=np.float32).reshape(-1)
        if (
            calibration_bias.shape != (len(prototypes),)
            or calibration_scale.shape != (len(prototypes),)
            or not np.isfinite(calibration_bias).all()
            or not np.isfinite(calibration_scale).all()
            or np.any(calibration_scale <= 0.0)
        ):
            raise ValueError("locked evidence-calibration state is invalid")
        cosine_scores = (
            cosine_scores - calibration_bias[None, :]
        ) / calibration_scale[None, :]
    scores = cosine_scores @ transform - class_bias
    output_shape = (*aligned.shape[:-1], len(prototypes))
    return (
        scores.reshape(output_shape).astype(np.float32),
        raw_scores.reshape(output_shape).astype(np.float32),
    )


def score_locked_symmetric_head(features: np.ndarray, head: Mapping[str, Any]) -> np.ndarray:
    scores, _raw_scores = _score_locked_symmetric_head_components(features, head)
    return scores


def score_locked_symmetric_head_with_raw(
    features: np.ndarray, head: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return EvidenceNorm scores and the pre-EvidenceNorm gate score stream."""

    return _score_locked_symmetric_head_components(features, head)


__all__ = [
    "fit_locked_symmetric_head",
    "score_locked_symmetric_head",
    "score_locked_symmetric_head_with_raw",
]
