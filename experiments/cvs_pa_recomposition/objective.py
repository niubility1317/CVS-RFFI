"""Recombine the deployed PA increment within labelled source TX/day cells.

This is a feature-space training objective prototype, not a hardware or
same-waveform counterfactual.  No network, classifier, or trainable state is
created here.  The caller supplies the actual deployed increments with gains
already detached; features remain differentiable and only class weights detach.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch.nn import functional as F

SOURCE_RX = frozenset((1, 3, 4, 6, 8))
SOURCE_DAY = frozenset((1, 2, 3))
MODES = frozenset(("ZERO", "SAME_RX", "CROSS_RX"))
DEFAULT_HEAD_NORMALIZE_EPS = 1e-4
INTEGER_DTYPES = frozenset((torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))


def _integer_metadata(value: torch.Tensor, count: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != count:
        raise ValueError(f"{name} must be a one-dimensional tensor with one value per sample")
    if value.dtype not in INTEGER_DTYPES:
        raise ValueError(f"{name} must contain integer metadata")
    return value.cpu().to(torch.int64)


def _ids(value: Sequence[str | int] | torch.Tensor, count: int) -> tuple[str | int, ...]:
    if isinstance(value, torch.Tensor):
        values = tuple(_integer_metadata(value, count, "physical_id").tolist())
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError("physical_id must contain one opaque ID per sample")
        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError("physical_id must contain one opaque ID per sample") from error
    if len(values) != count:
        raise ValueError("physical_id count differs from feature count")
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) or value == "" for value in values):
        raise ValueError("physical_id entries must be nonempty strings or integer IDs")
    return values


def _validate(a, b, c, weight, scale, y, rx, day, physical_id, mode, alpha, coefficient, role, seed):
    if role != "L_s":
        raise ValueError("PA recomposition permits L_s only; U/query/validation roles are forbidden")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer for the private CPU generator")
    for name, value in (("alpha", alpha), ("coefficient", coefficient)):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= alpha <= 1 or coefficient < 0:
        raise ValueError("alpha must be in [0,1] and coefficient must be nonnegative")
    if isinstance(scale, torch.Tensor):
        if scale.numel() != 1 or scale.requires_grad:
            raise ValueError("scale must be a fixed scalar, normally 30")
        scale = float(scale.item())
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive, normally 30")
    if not isinstance(a, torch.Tensor) or a.ndim != 2 or a.shape[1] == 0 or not a.is_floating_point():
        raise ValueError("a must be a real floating [batch,feature] tensor")
    for name, value in (("a", a), ("b", b), ("c", c)):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != a.shape:
            raise ValueError(f"{name} must be dense and have the same shape as a")
        if value.dtype != a.dtype or value.device != a.device or not value.is_floating_point():
            raise ValueError(f"{name} device/dtype must match a")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains nonfinite features")
    if not isinstance(weight, torch.Tensor) or weight.layout != torch.strided or weight.shape != (6, a.shape[1]):
        raise ValueError("weight must be the existing six-class [6,feature] classifier weight")
    if weight.dtype != a.dtype or weight.device != a.device or not bool(torch.isfinite(weight).all().item()):
        raise ValueError("weight must have matching device/dtype and finite values")
    count = a.shape[0]
    labels = _integer_metadata(y, count, "y")
    receivers = _integer_metadata(rx, count, "rx")
    days = _integer_metadata(day, count, "day")
    if any(not 0 <= value < 6 for value in labels.tolist()):
        raise ValueError("y must be legal true L_s labels in the six-class mapping [0,5]")
    if any(value not in SOURCE_RX for value in receivers.tolist()):
        raise ValueError("rx must be a source receiver in {1,3,4,6,8}")
    if any(value not in SOURCE_DAY for value in days.tolist()):
        raise ValueError("day must be a source day in {1,2,3}")
    ids = _ids(physical_id, count)
    metadata_by_id = {}
    for identifier, label, receiver, sample_day in zip(ids, labels.tolist(), receivers.tolist(), days.tolist()):
        metadata = (label, receiver, sample_day)
        if identifier in metadata_by_id and metadata_by_id[identifier] != metadata:
            raise ValueError("the same physical_id has conflicting TX/RX/day metadata")
        metadata_by_id[identifier] = metadata
    return labels, receivers, days, ids, float(scale)


def _plan(labels, receivers, days, ids, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cells: dict[tuple[int, int, int], dict[str | int, list[int]]] = {}
    for index, (label, receiver, sample_day, identifier) in enumerate(zip(labels.tolist(), receivers.tolist(), days.tolist(), ids)):
        cells.setdefault((label, sample_day, receiver), {}).setdefault(identifier, []).append(index)
    groups = sorted({(label, sample_day) for label, sample_day, _ in cells})
    cohort, same_donors, cross_donors = [], [], []
    cell_counts = []
    excluded_duplicate = excluded_unbalanced = excluded_insufficient = 0
    for label, sample_day in groups:
        keys = sorted(key for key in cells if key[:2] == (label, sample_day))
        qualified = [key for key in keys if len(cells[key]) >= 2]
        retained = len(qualified) >= 2
        per_cell_count = min(len(cells[key]) for key in qualified) if retained else 0
        selected_cells = []
        for key in keys:
            packets = cells[key]
            duplicate_count = sum(len(indices) - 1 for indices in packets.values())
            excluded_duplicate += duplicate_count
            selected = []
            if retained and key in qualified:
                packet_items = list(packets.values())
                order = torch.randperm(len(packet_items), generator=generator).tolist()
                for packet_position in order[:per_cell_count]:
                    indices = packet_items[packet_position]
                    representative = int(torch.randint(len(indices), (1,), generator=generator).item())
                    selected.append(indices[representative])
                excluded_unbalanced += len(packets) - per_cell_count
                selected_cells.append(selected)
                reason = None
            else:
                excluded_insufficient += len(packets)
                reason = "fewer_than_two_distinct_packets" if len(packets) < 2 else "fewer_than_two_qualified_RX_cells"
            cell_counts.append({
                "TX_label": key[0], "day": key[1], "RX": key[2],
                "batch_count": sum(len(indices) for indices in packets.values()),
                "distinct_physical_count": len(packets), "cohort_count": len(selected),
                "excluded_duplicate_count": duplicate_count,
                "excluded_unbalanced_count": len(packets) - len(selected) if selected else 0,
                "exclusion_reason": reason,
            })
        for cell_position, members in enumerate(selected_cells):
            next_cell = selected_cells[(cell_position + 1) % len(selected_cells)]
            for position, anchor in enumerate(members):
                cohort.append(anchor)
                same_donors.append(members[(position + 1) % len(members)])
                cross_donors.append(next_cell[position])
    anchors = torch.tensor(cohort, dtype=torch.int64)
    same_pairs = torch.stack((anchors, torch.tensor(same_donors, dtype=torch.int64)), dim=1)
    cross_pairs = torch.stack((anchors, torch.tensor(cross_donors, dtype=torch.int64)), dim=1)
    return anchors, same_pairs, cross_pairs, {
        "batch_count": len(ids), "cohort_count": len(cohort),
        "eligible_anchor_count": len(cohort),
        "coverage": len(cohort) / len(ids) if ids else 0.0,
        "excluded_duplicate_physical_count": excluded_duplicate,
        "excluded_unbalanced_count": excluded_unbalanced,
        "excluded_insufficient_count": excluded_insufficient,
        "cell_counts": cell_counts,
    }


def _logits(features, normalized_weight, scale):
    return F.linear(F.normalize(features, p=2, dim=1, eps=DEFAULT_HEAD_NORMALIZE_EPS), normalized_weight) * scale


def _metrics(logits, labels):
    with torch.no_grad():
        true = logits.gather(1, labels[:, None]).squeeze(1)
        alternatives = logits.clone()
        alternatives.scatter_(1, labels[:, None], -float("inf"))
        return {
            "true_margin_mean": float((true - alternatives.max(dim=1).values).mean().item()),
            "true_probability_mean": float(logits.softmax(dim=1).gather(1, labels[:, None]).mean().item()),
        }


def _paired_objective(a, b, c, normalized_weight, scale, labels, pairs, alpha, cohort):
    anchor, donor = pairs[:, 0].to(a.device), pairs[:, 1].to(a.device)
    # The residual form is algebraically the specified PA-only interpolation;
    # alpha=0 and identical PA increments exactly retain deployed features.
    replacement = alpha * (b[donor] - b[anchor])
    forward = a[anchor] + b[anchor] + c[anchor] + replacement
    reverse = a[donor] + b[donor] + c[donor] - replacement
    donor_position = {int(index): position for position, index in enumerate(pairs[:, 1].tolist())}
    reorder = torch.tensor([donor_position[int(index)] for index in cohort.tolist()], device=a.device)
    # Align both directions by the same original endpoint order before GEMM
    # and CE reduction.  This preserves exposure and avoids permutation-related
    # floating-point differences in the alpha=0 / identical-b identities.
    forward_logits = _logits(forward, normalized_weight, scale)
    reverse_logits = _logits(reverse[reorder], normalized_weight, scale)
    forward_labels = labels[pairs[:, 0]].to(a.device)
    reverse_labels = labels[pairs[:, 1]].to(a.device)[reorder]
    forward_ce = F.cross_entropy(forward_logits, forward_labels, reduction="none", label_smoothing=0.0)
    reverse_ce = F.cross_entropy(reverse_logits, reverse_labels, reduction="none", label_smoothing=0.0)
    ce = ((forward_ce + reverse_ce) * 0.5).mean()
    metrics = _metrics(torch.cat((forward_logits, reverse_logits)), torch.cat((forward_labels, reverse_labels)))
    with torch.no_grad():
        norms = torch.linalg.vector_norm(replacement, dim=1)
        metrics.update({
            "CE": float(ce.item()),
            "PA_replacement_norm_mean": float(norms.mean().item()),
            "PA_replacement_norm_rms": float(norms.square().mean().sqrt().item()),
            "PA_replacement_norm_max": float(norms.max().item()),
        })
    return ce, metrics, {"forward": forward, "reverse": reverse}


def recomposition(a, b, c, weight, scale, y, rx, day, physical_id, mode,
                  alpha=0.2, coefficient=0.1, role="L_s", seed=20261014):
    """Return weighted loss, compact JSON summary, and optional in-memory detail.

    Each TX/day keeps RX cells with >=2 distinct physical packets, and requires
    >=2 such cells.  The cells privately sample the same minimum packet count.
    SAME_RX uses a within-cell cyclic donor; CROSS_RX uses the next RX cell at
    the same position.  Every cohort packet is an anchor and donor exactly once
    in both modes.  Symmetric CE therefore has identical endpoint exposure and
    alpha=0 recovers the cohort's ordinary deployed CE.  ZERO executes the same
    recomposition/diagnostics computation and returns an exactly zero objective.

    Defaults alpha=.2 and coefficient=.1 are fixed prototype choices, without
    claims of efficacy.  The normally-30 cosine scale is fixed, with no bias or
    label smoothing.  Only class W detaches here; deployed gains must already
    be detached by the caller.  Metadata is asserted by the caller to belong to
    legal L_s; this function cannot verify an external dataset's provenance.
    """
    labels, receivers, days, ids, scale = _validate(a, b, c, weight, scale, y, rx, day, physical_id, mode, alpha, coefficient, role, seed)
    cohort, same_pairs, cross_pairs, counts = _plan(labels, receivers, days, ids, seed)
    summary: dict[str, Any] = {
        "mode": mode, "role": role, "alpha": float(alpha), "coefficient": float(coefficient),
        "scale": scale, "seed": seed, "counts": counts,
        "paired_metadata": {
            "same_TX": True, "same_day": True, "self_pair_count": 0,
            "same_physical_pair_count": 0, "same_RX_mode_equal_RX": True,
            "cross_RX_mode_different_RX": True, "shared_anchor_cohort": True,
            "anchor_uses_per_packet_per_mode": 1 if cohort.numel() else 0,
            "donor_uses_per_packet_per_mode": 1 if cohort.numel() else 0,
            "symmetric_endpoint_uses_per_packet_per_mode": 2 if cohort.numel() else 0,
            "pair_count_per_mode": int(cohort.numel()),
            "symmetric_prediction_count_per_mode": 2 * int(cohort.numel()),
            "computed_prediction_count_all_modes": 5 * int(cohort.numel()),
        },
    }
    if not cohort.numel():
        zero = (a * 0).sum() + (b * 0).sum() + (c * 0).sum()
        summary.update({"deployment_CE": None, "weighted_loss": 0.0, "SAME_RX": None, "CROSS_RX": None})
        return {"weighted_loss": zero, "summary": summary, "tensors": {"cohort": cohort, "SAME_RX_pairs": same_pairs, "CROSS_RX_pairs": cross_pairs}}
    normalized_weight = F.normalize(weight.detach(), p=2, dim=1, eps=DEFAULT_HEAD_NORMALIZE_EPS)
    deployed = a[cohort] + b[cohort] + c[cohort]
    baseline_logits = _logits(deployed, normalized_weight, scale)
    baseline_labels = labels[cohort].to(a.device)
    deployment_ce = F.cross_entropy(baseline_logits, baseline_labels, reduction="none", label_smoothing=0.0).mean()
    same_ce, same_metrics, same_features = _paired_objective(a, b, c, normalized_weight, scale, labels, same_pairs, alpha, cohort)
    cross_ce, cross_metrics, cross_features = _paired_objective(a, b, c, normalized_weight, scale, labels, cross_pairs, alpha, cohort)
    selected_ce = same_ce if mode == "SAME_RX" else cross_ce
    weighted_loss = selected_ce * (0.0 if mode == "ZERO" else float(coefficient))
    if not bool(torch.isfinite(weighted_loss).item()):
        raise ValueError("PA recomposition produced a nonfinite loss")
    baseline_metrics = _metrics(baseline_logits, baseline_labels)
    for metrics in (same_metrics, cross_metrics):
        metrics["CE_delta_from_deployment"] = metrics["CE"] - float(deployment_ce.item())
    summary.update({
        "deployment_CE": float(deployment_ce.item()), "deployment": baseline_metrics,
        "weighted_loss": float(weighted_loss.item()), "SAME_RX": same_metrics, "CROSS_RX": cross_metrics,
    })
    return {
        "weighted_loss": weighted_loss, "summary": summary,
        # Keep pair lists out of per-step JSON.  These details are for one-shot
        # checks; feature tensors retain their graph and should not be retained.
        "tensors": {"cohort": cohort, "SAME_RX_pairs": same_pairs, "CROSS_RX_pairs": cross_pairs,
                    "deployment_CE": deployment_ce, "SAME_RX_CE": same_ce, "CROSS_RX_CE": cross_ce,
                    "SAME_RX_features": same_features, "CROSS_RX_features": cross_features},
    }
