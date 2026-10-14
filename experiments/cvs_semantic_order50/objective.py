"""Source-only semantic ordering and ordinary SupCon loss prototypes.

The input is the existing deployed joint160 graph from one labelled source
batch.  This module owns no model, classifier, optimizer or persistent state.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import torch
from torch.nn import functional as F

SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAY = (1, 2, 3)
MODES = ("ZERO", "SUPCON", "ORDER")
NORMALIZE_EPS = 1e-4
INTEGER_DTYPES = tuple(dtype for dtype in (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    getattr(torch, "uint16", None), getattr(torch, "uint32", None), getattr(torch, "uint64", None),
) if dtype is not None)


def _metadata(value, count, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != count:
        raise ValueError(f"{name} must be a rank-one tensor with one item per input packet")
    if value.dtype not in INTEGER_DTYPES:
        raise ValueError(f"{name} must contain visible integer metadata")
    return value.cpu().to(torch.int64)


def _physical_ids(value: Sequence[str | int] | torch.Tensor, count):
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.numel() != count or value.dtype not in INTEGER_DTYPES:
            raise ValueError("physical_id must be a rank-one integer tensor with one item per input")
        identifiers = tuple(value.cpu().tolist())
    else:
        if isinstance(value, (str, bytes)):
            raise ValueError("physical_id must be a sequence with one opaque ID per input")
        try:
            identifiers = tuple(value)
        except TypeError as error:
            raise ValueError("physical_id must contain one opaque ID per input") from error
    if len(identifiers) != count:
        raise ValueError("physical_id length differs from input count")
    if any(isinstance(identifier, bool) or not isinstance(identifier, (int, str)) or identifier == "" for identifier in identifiers):
        raise ValueError("physical_id entries must be integer IDs or nonempty strings")
    return identifiers


def _validate(z, y, rx, day, physical_id, mode, coefficient, margin, temperature, role):
    if role != "L_s":
        raise ValueError("semantic auxiliary objectives permit visible L_s only, not U/V/query")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not isinstance(z, torch.Tensor) or z.layout != torch.strided or z.ndim != 2 or z.shape[1] != 160 or not z.is_floating_point():
        raise ValueError("z must be a real dense rank-two N x 160 deployed joint tensor")
    if not bool(torch.isfinite(z).all().item()):
        raise ValueError("z contains nonfinite deployed features")
    for name, value in (("coefficient", coefficient), ("margin", margin), ("temperature", temperature)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite fixed scalar")
    if coefficient < 0 or margin < 0 or temperature <= 0:
        raise ValueError("coefficient/margin must be nonnegative and temperature positive")
    count = z.shape[0]
    labels = _metadata(y, count, "y")
    receivers = _metadata(rx, count, "rx")
    days = _metadata(day, count, "day")
    if any(label < 0 or label >= 6 for label in labels.tolist()):
        raise ValueError("y must contain visible true L_s six-class labels in [0,5]")
    if any(receiver not in SOURCE_RX for receiver in receivers.tolist()):
        raise ValueError("rx must be an allowed source receiver in {1,3,4,6,8}")
    if any(sample_day not in SOURCE_DAY for sample_day in days.tolist()):
        raise ValueError("day must be an allowed source day in {1,2,3}")
    identifiers = _physical_ids(physical_id, count)
    return labels, receivers, days, identifiers


def _triplets(labels, receivers, days):
    """Enumerate all legal triples in O(N^2 + triplet_count) memory."""
    count = labels.numel()
    index = torch.arange(count)
    distinct = index[:, None] != index[None, :]
    positives = (labels[:, None] == labels[None, :]) & (days[:, None] == days[None, :]) & (receivers[:, None] != receivers[None, :]) & distinct
    reference_negatives = (labels[:, None] != labels[None, :]) & (receivers[:, None] == receivers[None, :]) & (days[:, None] == days[None, :]) & distinct
    parts = []
    for positive in range(count):
        anchors = torch.nonzero(positives[:, positive], as_tuple=False).flatten()
        negatives = torch.nonzero(reference_negatives[positive], as_tuple=False).flatten()
        if anchors.numel() and negatives.numel():
            parts.append(torch.stack((anchors.repeat_interleave(negatives.numel()),
                                      torch.full((anchors.numel() * negatives.numel(),), positive, dtype=torch.long),
                                      negatives.repeat(anchors.numel())), dim=1))
    triples = torch.cat(parts) if parts else torch.empty((0, 3), dtype=torch.long)
    return triples, int(positives.sum().item()), int(reference_negatives.sum().item())


def _supcon(cosine, labels, temperature, zero):
    count = labels.numel()
    if count < 2:
        return zero, torch.empty(0, dtype=torch.long), 0
    same_class = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(count, dtype=torch.bool)
    positive = same_class & off_diagonal
    positive_count = positive.sum(dim=1)
    anchors = torch.nonzero(positive_count > 0, as_tuple=False).flatten()
    if not anchors.numel():
        return zero, anchors, 0
    logits = cosine / temperature
    # All non-self packets remain in each denominator, including positives.
    log_denominator = torch.logsumexp(logits.masked_fill(~off_diagonal.to(cosine.device), -float("inf")), dim=1)
    log_probability = logits - log_denominator[:, None]
    per_anchor = -log_probability.masked_fill(~positive.to(cosine.device), 0.0).sum(dim=1) / positive_count.to(cosine.device).clamp_min(1)
    loss = per_anchor[anchors.to(cosine.device)].mean()
    return loss, anchors, int(positive.sum().item())


def _histogram(values, allowed):
    counts = Counter(values.tolist())
    return {str(value): counts[value] for value in allowed}


def _exposure(labels, receivers, days, triples, supcon_anchors):
    class_exposure = []
    for label in range(6):
        class_exposure.append({
            "TX_label": label, "cohort_packets": int((labels == label).sum().item()),
            "ORDER_anchor_endpoints": int((labels[triples[:, 0]] == label).sum().item()),
            "ORDER_positive_endpoints": int((labels[triples[:, 1]] == label).sum().item()),
            "ORDER_negative_endpoints": int((labels[triples[:, 2]] == label).sum().item()),
            "SUPCON_anchors": int((labels[supcon_anchors] == label).sum().item()),
        })
    reference_groups = []
    for receiver in SOURCE_RX:
        for sample_day in SOURCE_DAY:
            mask = (receivers == receiver) & (days == sample_day)
            if not bool(mask.any()):
                continue
            ref = triples[:, 1]
            reference_mask = (receivers[ref] == receiver) & (days[ref] == sample_day)
            reference_groups.append({
                "RX": receiver, "day": sample_day, "cohort_packets": int(mask.sum().item()),
                "class_counts": _histogram(labels[mask], range(6)),
                "ORDER_reference_packets": int(torch.unique(ref[reference_mask]).numel()),
                "ORDER_triplet_count": int(reference_mask.sum().item()),
            })
    return class_exposure, reference_groups


def objective(z, y, rx, day, physical_id, mode, *, coefficient=0.1,
              margin=0.1, temperature=0.1, role="L_s"):
    """Return weighted loss, compact JSON summary and in-memory cohort indices.

    Every occurrence of a repeated physical ID is removed from this auxiliary
    cohort.  The caller retains its ordinary main L CE on the full batch.  ORDER
    averages every legal (i,p,n), including satisfied zero terms: i/p share TX
    and day but differ RX; n differs TX and shares p's RX/day.  The feature graph
    remains attached.  At margin + cos(i,n) - cos(i,p) <= 0, ReLU's derivative is
    zero.  No hard mining, bank, active-only denominator or extra weights exist.

    SUPCON uses all non-self cohort packets in the denominator and same-TX
    positives, then averages anchors with positives.  Its endpoint exposure
    differs from ORDER by definition; matching the full L batch/IQ/update
    schedule belongs to the caller.  ZERO computes the same diagnostics and
    returns exact differentiable zero.  Legal source provenance/visible labels
    are assertions by the caller; no external provenance can be proven here.
    """
    labels, receivers, days, identifiers = _validate(z, y, rx, day, physical_id, mode, coefficient, margin, temperature, role)
    identifier_count = Counter(identifiers)
    cohort = torch.tensor([index for index, identifier in enumerate(identifiers) if identifier_count[identifier] == 1], dtype=torch.long)
    cohort_labels, cohort_rx, cohort_days = labels[cohort], receivers[cohort], days[cohort]
    features = F.normalize(z.index_select(0, cohort.to(z.device)), p=2, dim=1, eps=NORMALIZE_EPS)
    cosine = features @ features.T
    # Multiplication before summation avoids finite-input sum overflow while
    # preserving a differentiable exact zero for empty/no-relation cases.
    zero = (z * 0.0).sum()
    triples, positive_pair_count, negative_pair_count = _triplets(cohort_labels, cohort_rx, cohort_days)
    if triples.numel():
        local = triples.to(z.device)
        positive_cos = cosine[local[:, 0], local[:, 1]]
        negative_cos = cosine[local[:, 0], local[:, 2]]
        relation = margin + negative_cos - positive_cos
        order_loss = F.relu(relation).mean()
        unsatisfied = int((relation > 0).sum().item())
        violations = int((negative_cos > positive_cos).sum().item())
        ties = int((negative_cos == positive_cos).sum().item())
    else:
        order_loss, unsatisfied, violations, ties = zero, 0, 0, 0
    supcon_loss, supcon_anchors, supcon_positive_pair_count = _supcon(cosine, cohort_labels, temperature, zero)
    selected = order_loss if mode != "SUPCON" else supcon_loss
    weighted_loss = selected * (0.0 if mode == "ZERO" else float(coefficient))
    if not all(bool(torch.isfinite(loss).item()) for loss in (order_loss, supcon_loss, weighted_loss)):
        raise ValueError("semantic objective produced a nonfinite loss")
    count = triples.shape[0]
    class_exposure, reference_groups = _exposure(cohort_labels, cohort_rx, cohort_days, triples, supcon_anchors)
    summary = {
        "mode": mode, "role": role, "coefficient": float(coefficient),
        "margin": float(margin), "temperature": float(temperature), "normalize_eps": NORMALIZE_EPS,
        "input_count": z.shape[0], "unique_eligible_count": cohort.numel(),
        "duplicate_excluded_count": z.shape[0] - cohort.numel(),
        "repeated_physical_ID_count": sum(count > 1 for count in identifier_count.values()),
        "triplet_count": count, "ORDER_anchor_count": torch.unique(triples[:, 0]).numel(),
        "ORDER_has_relations": bool(count), "ORDER_has_active_relations": bool(unsatisfied),
        "legal_positive_pair_count": positive_pair_count, "legal_reference_negative_pair_count": negative_pair_count,
        "ordering_violations": violations, "ordering_tie_count": ties,
        "ordering_violation_rule": "cos(i,n) > cos(i,p)",
        "satisfied_margin_count": count - unsatisfied, "unsatisfied_margin_count": unsatisfied,
        "unsatisfied_margin_rate": unsatisfied / count if count else None,
        "ORDER_loss": float(order_loss.item()), "SUPCON_loss": float(supcon_loss.item()),
        "weighted_selected_loss": float(weighted_loss.item()),
        "SUPCON_anchor_count": supcon_anchors.numel(), "SUPCON_positive_pair_count": supcon_positive_pair_count,
        "source_RX_counts": _histogram(receivers, SOURCE_RX), "source_day_counts": _histogram(days, SOURCE_DAY),
        "cohort_RX_counts": _histogram(cohort_rx, SOURCE_RX), "cohort_day_counts": _histogram(cohort_days, SOURCE_DAY),
        "class_exposure": class_exposure, "domain_reference_groups": reference_groups,
        "exposure_note": "ORDER averages legal triplets; SUPCON averages anchors with same-TX positives. Endpoint exposure differs; full L/IQ/update scheduling is matched by the caller.",
        "class_W_dependency": False, "extra_model": False, "optimizer_state": False,
        "feature_endpoints_detached": False, "global_RNG_consumed": False,
        "external_source_provenance": "asserted_by_caller_not_proven_by_pure_function",
    }
    return {
        "weighted_loss": weighted_loss, "summary": summary,
        "tensors": {"cohort_indices": cohort, "ORDER_triplets": cohort[triples],
                    "SUPCON_anchor_indices": cohort[supcon_anchors]},
    }
