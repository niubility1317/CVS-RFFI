from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


NA_VALUE = "N/A"
REQUIRED_DIAGNOSTICS = (
    "zf_tx_probe",
    "zf_domain_probe",
    "zn_domain_probe",
    "zn_tx_probe",
    "zs_content_probe",
    "clean_leo_zf_distance",
    "same_tx_zf_distance",
    "drop_f_residual_gap",
    "transplant_target_id",
    "transplant_preserve_s",
    "transplant_preserve_n",
    "gram_condition",
    "effective_rank",
    "fisher_coverage",
    "train_time_s",
    "peak_vram_mb",
    "latency_ms",
)


def _unavailable(metrics: dict[str, Any], name: str, reason: str) -> None:
    metrics[name] = NA_VALUE
    metrics[f"{name}_reason"] = str(reason)


def _finite_scalar(value: Any) -> float | None:
    try:
        tensor = torch.as_tensor(value).detach().float().reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        return None
    return float(tensor.mean().cpu())


def _feature_matrix(value: Any) -> torch.Tensor | None:
    if isinstance(value, Mapping):
        parts = [_feature_matrix(value[key]) for key in sorted(value)]
        if not parts or any(part is None for part in parts):
            return None
        return torch.cat([part for part in parts if part is not None], dim=1)
    if not torch.is_tensor(value):
        return None
    matrix = value.detach().float().cpu()
    if matrix.ndim < 2 or matrix.size(0) == 0:
        return None
    if matrix.ndim > 2:
        matrix = matrix.mean(dim=tuple(range(1, matrix.ndim - 1)))
    matrix = matrix.reshape(matrix.size(0), -1)
    return matrix if bool(torch.isfinite(matrix).all()) else None


def _linear_probe_accuracy(
    features: Any,
    labels: Any,
    train_mask: Any,
    eval_mask: Any,
) -> tuple[float | None, str | None]:
    matrix = _feature_matrix(features)
    if matrix is None or not torch.is_tensor(labels):
        return None, "detached feature matrix or labels are unavailable"
    target = labels.detach().long().cpu().reshape(-1)
    train = torch.as_tensor(train_mask, dtype=torch.bool).detach().cpu().reshape(-1)
    evaluate = torch.as_tensor(eval_mask, dtype=torch.bool).detach().cpu().reshape(-1)
    if target.numel() != matrix.size(0) or train.numel() != matrix.size(0) or evaluate.numel() != matrix.size(0):
        return None, "probe features, labels, and split masks have inconsistent rows"
    if not bool(train.any()) or not bool(evaluate.any()):
        return None, "independent probe train/eval split is empty"
    if bool((target < 0).any()):
        return None, "probe labels contain hidden or invalid values"
    classes = torch.unique(target[train], sorted=True)
    if classes.numel() < 2 or not bool(torch.isin(target[evaluate], classes).all()):
        return None, "probe train split does not cover at least two eval classes"
    class_index = {int(label): index for index, label in enumerate(classes.tolist())}
    y_train = torch.zeros(int(train.sum()), classes.numel(), dtype=torch.float32)
    for row, label in enumerate(target[train].tolist()):
        y_train[row, class_index[int(label)]] = 1.0
    x_train = matrix[train]
    x_eval = matrix[evaluate]
    x_train = torch.cat((x_train, torch.ones(x_train.size(0), 1)), dim=1)
    x_eval = torch.cat((x_eval, torch.ones(x_eval.size(0), 1)), dim=1)
    ridge = 1e-4 * torch.eye(x_train.size(1), dtype=x_train.dtype)
    weights = torch.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ y_train)
    predicted = classes[(x_eval @ weights).argmax(dim=1)]
    return float((predicted == target[evaluate]).float().mean()), None


def _mean_l2(left: Any, right: Any) -> float | None:
    a = _feature_matrix(left)
    b = _feature_matrix(right)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(torch.linalg.vector_norm(a - b, dim=1).mean())


def _spectral_diagnostics(gram: Any) -> tuple[float | None, float | None]:
    if not torch.is_tensor(gram):
        return None, None
    matrix = gram.detach().float().cpu()
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or matrix.size(-1) != matrix.size(-2) or not bool(torch.isfinite(matrix).all()):
        return None, None
    singular = torch.linalg.svdvals(matrix)
    maximum = singular.max(dim=-1).values
    positive = torch.where(singular > 1e-8, singular, torch.full_like(singular, float("inf")))
    minimum = positive.min(dim=-1).values
    valid = torch.isfinite(minimum) & (maximum > 0)
    condition = float((maximum[valid] / minimum[valid]).mean()) if bool(valid.any()) else None
    probabilities = singular / singular.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    effective_rank = float(entropy.exp().mean())
    return condition, effective_rank


def compute_fcr_diagnostics(
    artifacts: Mapping[str, Any],
    *,
    resources: Mapping[str, Any] | None,
    row_id: str,
) -> dict[str, Any]:
    """Aggregate detached, training-external FCR probes into one JSON row."""

    metrics: dict[str, Any] = {"row_id": str(row_id), "schema": "adv3b02_fcr_diagnostics:v1"}
    resources = {} if resources is None else resources
    train_mask = artifacts.get("probe_train_mask")
    eval_mask = artifacts.get("probe_eval_mask")
    probes = {
        "zf_tx_probe": (artifacts.get("z_f_id"), artifacts.get("tx_labels")),
        "zf_domain_probe": (artifacts.get("z_f_id"), artifacts.get("domain_labels")),
        "zn_domain_probe": (artifacts.get("z_n"), artifacts.get("domain_labels")),
        "zn_tx_probe": (artifacts.get("z_n"), artifacts.get("tx_labels")),
        "zs_content_probe": (artifacts.get("z_s"), artifacts.get("content_labels")),
    }
    with torch.no_grad():
        for name, (features, labels) in probes.items():
            value, reason = _linear_probe_accuracy(features, labels, train_mask, eval_mask)
            if value is None:
                _unavailable(metrics, name, reason or "probe capability is unavailable")
            else:
                metrics[name] = value

        distance = _mean_l2(artifacts.get("clean_z_f_id"), artifacts.get("leo_z_f_id"))
        if distance is None:
            _unavailable(metrics, "clean_leo_zf_distance", "matched clean/LEO z_f artifacts are unavailable")
        else:
            metrics["clean_leo_zf_distance"] = distance

        same_tx = _mean_l2(artifacts.get("same_tx_zf_left"), artifacts.get("same_tx_zf_right"))
        if same_tx is None:
            _unavailable(metrics, "same_tx_zf_distance", "strict same-TX different-content pair is unavailable")
        else:
            metrics["same_tx_zf_distance"] = same_tx

        full_error = _finite_scalar(artifacts.get("drop_f_error_full"))
        without_error = _finite_scalar(artifacts.get("drop_f_error_without"))
        if full_error is None or without_error is None:
            _unavailable(metrics, "drop_f_residual_gap", "matched full/drop-f residual artifacts are unavailable")
        else:
            metrics["drop_f_residual_gap"] = without_error - full_error

        transplant = artifacts.get("strict_transplant")
        transplant_names = {
            "transplant_target_id": "target_id",
            "transplant_preserve_s": "preserve_s",
            "transplant_preserve_n": "preserve_n",
        }
        for name, source_key in transplant_names.items():
            value = _finite_scalar(transplant.get(source_key)) if isinstance(transplant, Mapping) else None
            if value is None:
                _unavailable(metrics, name, "strict fingerprint pair/transplant capability is unavailable")
            else:
                metrics[name] = value

        condition, rank = _spectral_diagnostics(artifacts.get("gram"))
        if condition is None:
            _unavailable(metrics, "gram_condition", "finite square Gram artifact is unavailable")
        else:
            metrics["gram_condition"] = condition
        if rank is None:
            _unavailable(metrics, "effective_rank", "finite square Gram artifact is unavailable")
        else:
            metrics["effective_rank"] = rank

        coverage = _finite_scalar(artifacts.get("fisher_coverage"))
        if coverage is None:
            _unavailable(metrics, "fisher_coverage", "Fisher excitation coverage artifact is unavailable")
        else:
            metrics["fisher_coverage"] = coverage

        for name in ("train_time_s", "peak_vram_mb", "latency_ms"):
            value = _finite_scalar(resources.get(name))
            if value is None:
                _unavailable(metrics, name, f"resource measurement {name} is unavailable")
            else:
                metrics[name] = value

    for name in REQUIRED_DIAGNOSTICS:
        if name not in metrics:
            _unavailable(metrics, name, "diagnostic was not produced")
        value = metrics[name]
        if isinstance(value, float) and not math.isfinite(value):
            _unavailable(metrics, name, "diagnostic is non-finite")
    return metrics


def write_fcr_diagnostics_json(path: str | Path, metrics: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
