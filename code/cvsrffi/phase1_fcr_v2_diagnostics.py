from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from cvsrffi.phase1_fcr_diagnostics import compute_fcr_diagnostics


NA_VALUE = "N/A"
REQUIRED_V2_DIAGNOSTICS = (
    "pair_count",
    "pair_coverage",
    "same_tx_cross_domain_pair_count",
    "same_tx_cross_domain_pair_coverage",
    "eta_valid_coverage",
    "eta_component_error",
    "decoder_nuisance_sensitivity",
    "swap_output_delta",
    "z_tx_state_tx_probe",
    "grad_backbone_to_total_ratio",
    "grad_aux_to_backbone_ratio",
    "grad_domain_to_total_ratio",
    "grad_clean_leo_cosine",
    "epoch_time_s",
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
        matrix = matrix.reshape(matrix.size(0), -1)
    else:
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


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _finite_scalar(numerator)
    bottom = _finite_scalar(denominator)
    if top is None or bottom is None or abs(bottom) <= 1.0e-12:
        return None
    return float(top / bottom)


def _eta_metrics(eta_pred: Any, eta_target: Any) -> tuple[float | None, float | None, list[float] | None]:
    if not torch.is_tensor(eta_pred) or not torch.is_tensor(eta_target):
        return None, None, None
    pred = eta_pred.detach().float().cpu()
    target = eta_target.detach().float().cpu()
    if pred.shape != target.shape or pred.ndim != 2 or pred.size(0) == 0:
        return None, None, None
    valid = torch.isfinite(pred).all(dim=1) & torch.isfinite(target).all(dim=1)
    if not bool(valid.any()):
        return 0.0, None, None
    delta = (pred[valid] - target[valid]).abs()
    return (
        float(valid.float().mean()),
        float(delta.mean()),
        [float(value) for value in delta.mean(dim=0).tolist()],
    )


def _pair_summary(tx_labels: Any, domain_labels: Any, matched_count: int) -> tuple[int | None, float | None, int | None, float | None]:
    if not torch.is_tensor(tx_labels) or not torch.is_tensor(domain_labels):
        return matched_count, None, None, None
    tx = tx_labels.detach().long().cpu().reshape(-1)
    domain = domain_labels.detach().long().cpu().reshape(-1)
    row_count = min(int(tx.numel()), int(domain.numel()), int(matched_count))
    if row_count < 1:
        return 0, None, 0, None
    tx = tx[:row_count]
    domain = domain[:row_count]
    index = torch.arange(row_count)
    upper = index.view(-1, 1) < index.view(1, -1)
    valid = tx.view(-1, 1).ge(0) & domain.view(-1, 1).ge(0)
    same_tx_cross_domain = tx.view(-1, 1).eq(tx.view(1, -1)) & (~domain.view(-1, 1).eq(domain.view(1, -1))) & upper & valid
    denom = max(1, int((upper & valid).sum().item()))
    pair_count = row_count
    pair_coverage = 1.0
    strict_pair_count = int(same_tx_cross_domain.sum().item())
    strict_pair_coverage = float(strict_pair_count / denom)
    return pair_count, pair_coverage, strict_pair_count, strict_pair_coverage


def _per_tx_source_metrics(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    labels = artifacts.get("tx_labels")
    clean = artifacts.get("clean_z_f_id")
    leo = artifacts.get("leo_z_f_id")
    if not torch.is_tensor(labels):
        return {}
    tx = labels.detach().long().cpu().reshape(-1)
    clean_matrix = _feature_matrix(clean)
    leo_matrix = _feature_matrix(leo)
    out: dict[str, Any] = {}
    for label in torch.unique(tx, sorted=True).tolist():
        mask = tx.eq(int(label))
        entry: dict[str, Any] = {"count": int(mask.sum().item())}
        if clean_matrix is not None and leo_matrix is not None and clean_matrix.shape == leo_matrix.shape and mask.numel() == clean_matrix.size(0):
            entry["clean_leo_zf_distance"] = float(
                torch.linalg.vector_norm(clean_matrix[mask] - leo_matrix[mask], dim=1).mean()
            )
        else:
            entry["clean_leo_zf_distance"] = NA_VALUE
            entry["clean_leo_zf_distance_reason"] = "matched clean/LEO z_f features are unavailable"
        out[str(int(label))] = entry
    return out


def collect_fcr_v2_diagnostics(
    artifacts: Mapping[str, Any],
    *,
    resources: Mapping[str, Any] | None,
    row_id: str,
) -> dict[str, Any]:
    metrics = compute_fcr_diagnostics(artifacts, resources=resources, row_id=row_id)
    metrics["schema"] = "adv3b02_fcr_diagnostics:v2"
    resources = {} if resources is None else resources

    pair_counts = {
        str(name): max(0, int(value))
        for name, value in dict(resources.get("pair_counts", {})).items()
    }
    pair_opportunities = {
        str(name): max(0, int(value))
        for name, value in dict(resources.get("pair_opportunities", {})).items()
    }
    if not pair_counts:
        _unavailable(metrics, "pair_count", "training PairBuilder counts are unavailable")
        _unavailable(metrics, "pair_coverage", "training PairBuilder opportunities are unavailable")
        _unavailable(metrics, "same_tx_cross_domain_pair_count", "training nuisance-pair count is unavailable")
        _unavailable(metrics, "same_tx_cross_domain_pair_coverage", "training nuisance-pair opportunities are unavailable")
    else:
        total_pairs = sum(pair_counts.values())
        total_opportunities = sum(pair_opportunities.values())
        metrics["pair_count"] = int(total_pairs)
        metrics["pair_coverage"] = (
            float(total_pairs / total_opportunities) if total_opportunities > 0 else 0.0
        )
        nuisance_count = int(pair_counts.get("nuisance", 0))
        nuisance_opportunities = int(pair_opportunities.get("nuisance", 0))
        metrics["same_tx_cross_domain_pair_count"] = nuisance_count
        metrics["same_tx_cross_domain_pair_coverage"] = (
            float(nuisance_count / nuisance_opportunities)
            if nuisance_opportunities > 0
            else 0.0
        )
        metrics["pair_counts_by_axis"] = pair_counts
        metrics["pair_coverage_by_axis"] = {
            name: (
                float(pair_counts.get(name, 0) / opportunities)
                if opportunities > 0
                else 0.0
            )
            for name, opportunities in pair_opportunities.items()
        }

    eta_counts = [float(value) for value in resources.get("eta_valid_count_by_dim", ())]
    eta_error_sums = [
        float(value) for value in resources.get("eta_absolute_error_sum_by_dim", ())
    ]
    eta_opportunities = int(resources.get("eta_component_opportunities", 0) or 0)
    if eta_counts and len(eta_counts) == len(eta_error_sums) and eta_opportunities > 0:
        eta_coverage = float(sum(eta_counts) / (eta_opportunities * len(eta_counts)))
        eta_by_dim = [
            (error_sum / count if count > 0.0 else NA_VALUE)
            for error_sum, count in zip(eta_error_sums, eta_counts)
        ]
        finite_eta = [
            float(value)
            for value in eta_by_dim
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        eta_error = float(sum(finite_eta) / len(finite_eta)) if finite_eta else None
    else:
        eta_coverage, eta_error, eta_by_dim = _eta_metrics(
            artifacts.get("eta_pred"), artifacts.get("eta_target")
        )
    if eta_coverage is None:
        _unavailable(metrics, "eta_valid_coverage", "eta prediction or target artifact is unavailable")
        _unavailable(metrics, "eta_component_error", "eta prediction or target artifact is unavailable")
    else:
        metrics["eta_valid_coverage"] = float(eta_coverage)
        if eta_error is None:
            _unavailable(metrics, "eta_component_error", "eta prediction rows are present but no finite paired values exist")
        else:
            metrics["eta_component_error"] = float(eta_error)
            if eta_by_dim is not None:
                metrics["eta_component_error_by_dim"] = eta_by_dim

    nuisance_sensitivity = _mean_l2(artifacts.get("decode_full"), artifacts.get("decode_zero_nuisance"))
    if nuisance_sensitivity is None:
        _unavailable(metrics, "decoder_nuisance_sensitivity", "matched full and zero-nuisance decoder outputs are unavailable")
    else:
        metrics["decoder_nuisance_sensitivity"] = nuisance_sensitivity

    swap_delta = _mean_l2(artifacts.get("decode_full"), artifacts.get("decode_swap"))
    if swap_delta is None:
        _unavailable(metrics, "swap_output_delta", "matched full and swapped decoder outputs are unavailable")
    else:
        metrics["swap_output_delta"] = swap_delta

    z_tx_state_probe, z_tx_state_reason = _linear_probe_accuracy(
        artifacts.get("z_tx_state"),
        artifacts.get("tx_labels"),
        artifacts.get("probe_train_mask"),
        artifacts.get("probe_eval_mask"),
    )
    if z_tx_state_probe is None:
        _unavailable(metrics, "z_tx_state_tx_probe", z_tx_state_reason or "z_tx_state probe artifacts are unavailable")
    else:
        metrics["z_tx_state_tx_probe"] = z_tx_state_probe

    for name, numerator, denominator, reason in (
        ("grad_backbone_to_total_ratio", resources.get("grad_backbone"), resources.get("grad_total"), "gradient total or backbone norm is unavailable"),
        ("grad_aux_to_backbone_ratio", resources.get("grad_aux"), resources.get("grad_backbone"), "gradient backbone or aux norm is unavailable"),
        ("grad_domain_to_total_ratio", resources.get("grad_domain"), resources.get("grad_total"), "gradient total or domain norm is unavailable"),
    ):
        ratio = _ratio(numerator, denominator)
        if ratio is None:
            _unavailable(metrics, name, reason)
        else:
            metrics[name] = ratio

    grad_cosine = _finite_scalar(resources.get("grad_clean_leo_cosine"))
    if grad_cosine is None:
        _unavailable(metrics, "grad_clean_leo_cosine", "clean/LEO gradient cosine was not collected during detached finalization")
    else:
        metrics["grad_clean_leo_cosine"] = grad_cosine

    epoch_time = _finite_scalar(resources.get("epoch_time_s"))
    if epoch_time is None:
        _unavailable(metrics, "epoch_time_s", "epoch time is unavailable")
    else:
        metrics["epoch_time_s"] = epoch_time

    energy_ratio = _finite_scalar((artifacts.get("response_quality") or {}).get("energy_ratio") if isinstance(artifacts.get("response_quality"), Mapping) else None)
    state_norm = _finite_scalar((artifacts.get("response_quality") or {}).get("state_norm") if isinstance(artifacts.get("response_quality"), Mapping) else None)
    if energy_ratio is not None:
        metrics["response_energy_ratio"] = energy_ratio
    if state_norm is not None:
        metrics["response_state_norm"] = state_norm

    metrics["per_tx_source_metrics"] = _per_tx_source_metrics(artifacts)
    configured_lambdas = [str(name) for name in resources.get("configured_lambdas", ())]
    effective_weights = {
        str(name): float(value)
        for name, value in dict(resources.get("effective_weights", {})).items()
    }
    nonzero_steps = {
        str(name): int(value)
        for name, value in dict(resources.get("nonzero_loss_steps", {})).items()
    }
    gradient_ratios = {
        str(name): float(value)
        for name, value in dict(resources.get("gradient_ratios_to_identity_ce", {})).items()
    }
    actual_active: list[str] = []
    mechanism_status: dict[str, str] = {}
    for name in configured_lambdas:
        evidence = (
            effective_weights.get(name, 0.0) > 0.0
            and nonzero_steps.get(name, 0) > 0
            and gradient_ratios.get(name, 0.0) > 0.0
        )
        if evidence:
            actual_active.append(name)
            mechanism_status[name] = "ACTIVATED_WITH_LOSS_AND_GRADIENT_EVIDENCE"
        else:
            mechanism_status[name] = "MECHANISM_NOT_ACTIVATED:no_nonzero_loss_or_gradient_evidence"
    metrics["activation_state"] = {
        "configured_lambdas": configured_lambdas,
        "actual_active_lambdas": actual_active,
        "mechanism_status": mechanism_status,
        "effective_weights": effective_weights,
        "nonzero_loss_steps": nonzero_steps,
        "gradient_ratios_to_identity_ce": gradient_ratios,
        "capability_reasons": {
            str(name): str(reason)
            for name, reason in dict(resources.get("capability_reasons", {})).items()
        },
        "decoder_mode": str(artifacts.get("decoder_mode", "")),
    }

    for name in REQUIRED_V2_DIAGNOSTICS:
        if name not in metrics:
            _unavailable(metrics, name, "diagnostic was not produced")
        value = metrics[name]
        if isinstance(value, float) and not math.isfinite(value):
            _unavailable(metrics, name, "diagnostic is non-finite")
    return metrics


def write_fcr_v2_diagnostics(
    path: str | Path,
    row_id: str,
    artifacts: Mapping[str, Any],
    *,
    resources: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_fcr_v2_diagnostics(artifacts, resources=resources, row_id=row_id)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
