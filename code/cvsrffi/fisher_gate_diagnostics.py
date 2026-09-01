from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import torch

from cvsrffi.tensors import unpack_batch


_PER_SAMPLE_KEYS = ("weights", "null_weight", "q_sample", "entropy", "I", "D", "S", "U")


def _validated(
    diagnostics: Mapping[str, torch.Tensor], branch_names: Sequence[str]
) -> Dict[str, torch.Tensor]:
    branch_names = tuple(branch_names)
    missing = [key for key in _PER_SAMPLE_KEYS if key not in diagnostics]
    if missing:
        raise KeyError(f"missing gate diagnostics: {missing}")
    values = {
        key: torch.as_tensor(diagnostics[key]).detach().float().cpu()
        for key in _PER_SAMPLE_KEYS
    }
    weights = values["weights"]
    if weights.dim() != 2 or weights.size(1) != len(branch_names):
        raise ValueError("weights must have shape [N,branch_count]")
    rows = int(weights.size(0))
    for key in ("I", "D", "S", "U"):
        if values[key].shape != weights.shape:
            raise ValueError(f"{key} must match weights")
    for key in ("null_weight", "q_sample", "entropy"):
        if values[key].shape != (rows,):
            raise ValueError(f"{key} must have shape [N]")
    for key, value in values.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    return values


def _named_mean(value: torch.Tensor, branch_names: Sequence[str]) -> Dict[str, float]:
    mean = value.mean(dim=0)
    return {name: float(mean[index].item()) for index, name in enumerate(branch_names)}


def summarize_gate_by_group(
    diagnostics: Mapping[str, torch.Tensor],
    *,
    groups: Sequence[object],
    branch_names: Sequence[str],
    correct: Optional[Sequence[bool]] = None,
) -> Dict[str, Dict[str, object]]:
    """Return same-row gate/evidence summaries without pooling scenarios."""

    branch_names = tuple(branch_names)
    values = _validated(diagnostics, branch_names)
    if len(groups) != values["weights"].size(0):
        raise ValueError("groups must match the diagnostic row count")
    correct_tensor = None
    if correct is not None:
        correct_tensor = torch.as_tensor(correct, dtype=torch.bool)
        if correct_tensor.shape != values["q_sample"].shape:
            raise ValueError("correct must have shape [N]")
    ordered_groups = tuple(dict.fromkeys(str(group) for group in groups))
    group_strings = [str(group) for group in groups]
    summary: Dict[str, Dict[str, object]] = {}
    for group in ordered_groups:
        mask = torch.tensor([value == group for value in group_strings], dtype=torch.bool)
        weights = values["weights"][mask]
        q_sample = values["q_sample"][mask]
        conditional = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        row: Dict[str, object] = {
            "count": int(mask.sum().item()),
            "weight_mean": _named_mean(weights, branch_names),
            "conditional_weight_mean": _named_mean(conditional, branch_names),
            "null_mean": float(values["null_weight"][mask].mean().item()),
            "q_mean": float(q_sample.mean().item()),
            "q_p05": float(torch.quantile(q_sample, 0.05).item()),
            "q_p50": float(torch.quantile(q_sample, 0.50).item()),
            "q_p95": float(torch.quantile(q_sample, 0.95).item()),
            "entropy_mean": float(values["entropy"][mask].mean().item()),
        }
        for key in ("I", "D", "S", "U"):
            row[f"{key}_mean"] = _named_mean(values[key][mask], branch_names)
        if correct_tensor is not None:
            row["accuracy"] = float(correct_tensor[mask].float().mean().item())
        summary[group] = row
    return summary


def detect_gate_collapse(
    diagnostics: Mapping[str, torch.Tensor],
    *,
    branch_names: Sequence[str],
    starvation_floor: float = 0.02,
    hard_share: float = 0.95,
    hard_rate_limit: float = 0.80,
    null_saturation: float = 0.90,
) -> Dict[str, object]:
    """Detect branch starvation, premature top-1 routing and null saturation."""

    branch_names = tuple(branch_names)
    values = _validated(diagnostics, branch_names)
    weights = values["weights"].clamp_min(0.0)
    conditional = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    usage = weights.mean(dim=0)
    starved = [
        name
        for index, name in enumerate(branch_names)
        if float(usage[index].item()) < float(starvation_floor)
    ]
    hard_rate = float((conditional.max(dim=1).values >= float(hard_share)).float().mean().item())
    normalized_entropy = -(
        conditional * conditional.clamp_min(1e-8).log()
    ).sum(dim=1) / torch.log(conditional.new_tensor(float(len(branch_names))))
    null_rate = float(
        (values["null_weight"] >= float(null_saturation)).float().mean().item()
    )
    warnings = []
    if starved:
        warnings.append("BRANCH_STARVATION")
    if hard_rate > float(hard_rate_limit):
        warnings.append("OVER_HARD_ROUTING")
    if null_rate > 0.50:
        warnings.append("NULL_SATURATION")
    return {
        "collapsed": bool(warnings),
        "warnings": warnings,
        "starved_branches": starved,
        "usage_mean": {
            name: float(usage[index].item()) for index, name in enumerate(branch_names)
        },
        "hard_routing_rate": hard_rate,
        "conditional_entropy_mean": float(normalized_entropy.mean().item()),
        "null_saturation_rate": null_rate,
    }


def _ridge_fit(features: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    design = torch.cat([features, torch.ones(features.size(0), 1)], dim=1)
    gram = design.T @ design
    penalty = torch.eye(gram.size(0), dtype=gram.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    return torch.linalg.solve(gram + penalty, design.T @ targets)


def _ridge_predict(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    design = torch.cat([features, torch.ones(features.size(0), 1)], dim=1)
    return design @ coefficients


def _log_contrast(gate_features: torch.Tensor) -> torch.Tensor:
    logged = gate_features.clamp_min(1e-6).log()
    return logged - logged.mean(dim=1, keepdim=True)


def _fold_ids(labels: torch.Tensor, folds: int) -> torch.Tensor:
    fold_ids = torch.empty_like(labels)
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).reshape(-1)
        fold_ids[indices] = torch.arange(indices.numel()) % int(folds)
    return fold_ids


def controlled_receiver_probe(
    gate_features: torch.Tensor,
    receivers: torch.Tensor,
    controls: torch.Tensor,
    *,
    folds: int = 5,
    ridge: float = 1e-2,
) -> Dict[str, float]:
    """Probe receiver identity after train-fold-only control of signal quality."""

    gate_features = torch.as_tensor(gate_features).detach().float().cpu()
    receivers = torch.as_tensor(receivers).detach().long().cpu().reshape(-1)
    controls = torch.as_tensor(controls).detach().float().cpu()
    if gate_features.dim() != 2 or controls.dim() != 2:
        raise ValueError("gate_features and controls must be matrices")
    if gate_features.size(0) != receivers.numel() or controls.size(0) != receivers.numel():
        raise ValueError("probe inputs must share row count")
    classes, encoded = torch.unique(receivers, sorted=True, return_inverse=True)
    if classes.numel() < 2:
        raise ValueError("receiver probe requires at least two receivers")
    folds = int(folds)
    if folds < 2 or min(int((encoded == value).sum()) for value in range(classes.numel())) < folds:
        raise ValueError("each receiver must have at least one row per fold")
    targets = torch.nn.functional.one_hot(encoded, num_classes=classes.numel()).float()
    contrast = _log_contrast(gate_features)
    fold_ids = _fold_ids(encoded, folds)
    raw_correct = controlled_correct = total = 0
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        raw_coefficients = _ridge_fit(contrast[train], targets[train], ridge)
        raw_prediction = _ridge_predict(contrast[test], raw_coefficients).argmax(dim=1)

        control_coefficients = _ridge_fit(controls[train], contrast[train], ridge)
        train_residual = contrast[train] - _ridge_predict(
            controls[train], control_coefficients
        )
        test_residual = contrast[test] - _ridge_predict(
            controls[test], control_coefficients
        )
        controlled_coefficients = _ridge_fit(train_residual, targets[train], ridge)
        controlled_prediction = _ridge_predict(
            test_residual, controlled_coefficients
        ).argmax(dim=1)
        raw_correct += int((raw_prediction == encoded[test]).sum().item())
        controlled_correct += int((controlled_prediction == encoded[test]).sum().item())
        total += int(test.sum().item())
    majority = float(torch.bincount(encoded).max().item()) / float(encoded.numel())
    raw_accuracy = raw_correct / float(total)
    controlled_accuracy = controlled_correct / float(total)
    return {
        "sample_count": float(total),
        "receiver_count": float(classes.numel()),
        "majority_accuracy": majority,
        "raw_accuracy": raw_accuracy,
        "controlled_accuracy": controlled_accuracy,
        "raw_excess_accuracy": raw_accuracy - majority,
        "controlled_excess_accuracy": controlled_accuracy - majority,
    }


def expected_behavior_checks(
    diagnostics: Mapping[str, torch.Tensor],
    *,
    conditions: Sequence[object],
    branch_names: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    """Evaluate the report's directional examples without imposing top-1 routes."""

    branch_names = tuple(branch_names)
    values = _validated(diagnostics, branch_names)
    if len(conditions) != values["weights"].size(0):
        raise ValueError("conditions must match diagnostic rows")
    condition_strings = [str(value) for value in conditions]

    def mean(condition: str, key: str, branch: Optional[str] = None) -> float:
        mask = torch.tensor(
            [value == condition for value in condition_strings], dtype=torch.bool
        )
        if not bool(mask.any()):
            raise ValueError(f"missing expected-behavior condition {condition}")
        value = values[key][mask]
        if branch is not None:
            value = value[:, branch_names.index(branch)]
        return float(value.mean().item())

    low_null = mean("low_snr", "null_weight")
    reference_null = sum(
        mean(condition, "null_weight")
        for condition in ("bpsk", "rich_qam", "phase_stable")
    ) / 3.0
    qam_iq_pa = mean("rich_qam", "weights", "raw") + mean(
        "rich_qam", "weights", "pa"
    )
    bpsk_iq_pa = mean("bpsk", "weights", "raw") + mean(
        "bpsk", "weights", "pa"
    )
    cycle_phase = mean("cycle_slip", "weights", "phase")
    stable_phase = mean("phase_stable", "weights", "phase")
    return {
        "low_snr_null_above_clean": {
            "passed": low_null > reference_null,
            "observed": low_null,
            "reference": reference_null,
        },
        "rich_qam_iq_pa_above_bpsk": {
            "passed": qam_iq_pa > bpsk_iq_pa,
            "observed": qam_iq_pa,
            "reference": bpsk_iq_pa,
        },
        "cycle_slip_phase_below_stable": {
            "passed": cycle_phase < stable_phase,
            "observed": cycle_phase,
            "reference": stable_phase,
        },
    }


@torch.no_grad()
def collect_model_gate_diagnostics(
    model,
    loader,
    device,
    *,
    max_batches: int = 0,
    input_transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    metadata_fn: Optional[Callable[[object, int], Mapping[str, torch.Tensor]]] = None,
) -> Dict[str, object]:
    """Collect label-free physical evidence plus separate evaluation outcomes."""

    was_training = bool(model.training)
    model.eval()
    collected = {key: [] for key in _PER_SAMPLE_KEYS}
    correct = []
    labels = []
    metadata: Dict[str, list] = {}
    branch_names = None
    try:
        for batch_index, batch in enumerate(loader):
            x, y, extra = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if input_transform is not None:
                x = input_transform(x, int(batch_index))
            output = model(
                x,
                y_tx=None,
                grl_lambda=1.0,
                return_aux=True,
                return_physical_gate_diag=True,
            )
            identity_output = output.get("aux_id", output)
            gate_diagnostics = identity_output.get("physical_gate_diag")
            if not isinstance(gate_diagnostics, Mapping) or "per_sample" not in gate_diagnostics:
                raise RuntimeError("model did not return per-sample NMFDU diagnostics")
            current_names = tuple(gate_diagnostics["branch_names"])
            if branch_names is None:
                branch_names = current_names
            elif current_names != branch_names:
                raise RuntimeError("NMFDU branch order changed during collection")
            per_sample = gate_diagnostics["per_sample"]
            for key in _PER_SAMPLE_KEYS:
                collected[key].append(per_sample[key].detach().float().cpu())
            prediction = output["tx_logits"].argmax(dim=1)
            correct.append((prediction == y).detach().cpu())
            labels.append(y.detach().cpu())
            if metadata_fn is not None:
                batch_metadata = metadata_fn(extra, int(y.numel()))
                for key, value in batch_metadata.items():
                    tensor = torch.as_tensor(value).detach().cpu()
                    if tensor.size(0) != y.numel():
                        raise ValueError(f"metadata {key} does not match batch size")
                    metadata.setdefault(str(key), []).append(tensor)
            if int(max_batches) > 0 and batch_index + 1 >= int(max_batches):
                break
    finally:
        model.train(was_training)
    if not labels:
        raise ValueError("diagnostic loader produced no rows")
    return {
        "branch_names": branch_names,
        "diagnostics": {
            key: torch.cat(parts, dim=0) for key, parts in collected.items()
        },
        "correct": torch.cat(correct, dim=0),
        "labels": torch.cat(labels, dim=0),
        "metadata": {
            key: torch.cat(parts, dim=0) for key, parts in metadata.items()
        },
    }
