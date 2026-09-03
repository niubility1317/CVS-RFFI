from __future__ import annotations

from typing import Any

import torch


def continuous_unlabeled_trust(
    *,
    recoverability: torch.Tensor,
    view_js: torch.Tensor,
    temporal_inconsistency: torch.Tensor,
    prototype_margin: torch.Tensor,
) -> torch.Tensor:
    tensors = [torch.as_tensor(value, dtype=torch.float32).reshape(-1) for value in (
        recoverability,
        view_js,
        temporal_inconsistency,
        prototype_margin,
    )]
    if len({tensor.numel() for tensor in tensors}) != 1:
        raise ValueError("unlabeled trust inputs must align")
    recoverability_t, view_js_t, temporal_t, margin_t = tensors
    agreement = torch.exp(-view_js_t.clamp_min(0.0))
    temporal = torch.exp(-temporal_t.clamp_min(0.0))
    margin = margin_t.clamp(0.0, 1.0)
    return (recoverability_t.clamp(0.0, 1.0) * agreement * temporal * margin).pow(0.25).clamp(0.0, 1.0)


def classify_unlabeled_trust(
    *,
    trust: torch.Tensor,
    predicted_class: torch.Tensor,
    receiver_bin: torch.Tensor,
    severity_bin: torch.Tensor,
    core_threshold: float,
    irrecoverable_threshold: float,
    max_core_per_group: int,
) -> dict[str, Any]:
    trust = trust.reshape(-1).float()
    predicted_class = predicted_class.reshape(-1).long()
    receiver_bin = receiver_bin.reshape(-1).long()
    severity_bin = severity_bin.reshape(-1).long()
    if not (trust.numel() == predicted_class.numel() == receiver_bin.numel() == severity_bin.numel()):
        raise ValueError("unlabeled trust routing inputs must align")
    if not 0.0 <= float(irrecoverable_threshold) < float(core_threshold) <= 1.0:
        raise ValueError("trust thresholds must satisfy 0 <= irrecoverable < core <= 1")
    if int(max_core_per_group) < 1:
        raise ValueError("max_core_per_group must be positive")
    core_candidate = trust >= float(core_threshold)
    core = torch.zeros_like(core_candidate)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index in core_candidate.nonzero(as_tuple=False).reshape(-1).tolist():
        key = (
            int(predicted_class[index]),
            int(receiver_bin[index]),
            int(severity_bin[index]),
        )
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: float(trust[index]), reverse=True)
        core[ordered[: int(max_core_per_group)]] = True
    irrecoverable = trust <= float(irrecoverable_threshold)
    ambiguous = ~(core | irrecoverable)
    return {
        "core": core,
        "ambiguous": ambiguous,
        "irrecoverable": irrecoverable,
        "core_coverage": core.float().mean(),
    }
