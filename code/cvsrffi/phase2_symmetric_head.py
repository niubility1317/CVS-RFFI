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
    required = {
        "use_alignment",
        "prototype_rule",
        "ridge",
        "gram_mix",
        "uncertainty_penalty",
    }
    if set(selected) != required:
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
    # Formal deployment state is FP16 payload reloaded to FP32 scoring.
    def q(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else value.astype(np.float16).astype(np.float32)

    return {
        "prototypes": q(prototypes),
        "score_transform": q(transform),
        "class_bias": q(class_bias),
        "alignment_scale": q(scale),
        "alignment_bias": q(bias),
        "physical_shots_per_class": shots,
    }


def score_locked_symmetric_head(features: np.ndarray, head: Mapping[str, Any]) -> np.ndarray:
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
    scores = flat @ _normalize(prototypes).T
    scores = scores @ np.asarray(head["score_transform"], dtype=np.float32)
    scores -= np.asarray(head["class_bias"], dtype=np.float32)[None, :]
    return scores.reshape(*aligned.shape[:-1], len(prototypes)).astype(np.float32)


__all__ = ["fit_locked_symmetric_head", "score_locked_symmetric_head"]
