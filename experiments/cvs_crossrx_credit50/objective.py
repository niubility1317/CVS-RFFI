"""Current-source anchor-only ORDER loss and frozen five-RX combination.

Input is the actual SIG_TIME sequence_pool.output_norm pre-gain160 graph.
This module has no classifier, projection, gains, model/optimizer state or RNG.
Actual AdamW increment credit and same-batch other-RX default-W_pre utility are
provided by the caller; no utility is inferred from this pure loss function.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import torch
from torch.nn import functional as F

from ..cvs_semantic_order50.objective import (
    NORMALIZE_EPS, SOURCE_DAY, SOURCE_RX, _exposure, _histogram,
    _supcon, _triplets, _validate,
)


LAYER_KIND = "SIG_TIME_pre_gain160"
COEFFICIENT = .1
MARGIN = .1
TEMPERATURE = .1


def _boundary(role, layer_kind):
    if role != "L_s":
        raise ValueError("cross-RX ordering permits true visible source L_s only")
    if layer_kind != LAYER_KIND:
        raise ValueError("layer_kind must be SIG_TIME_pre_gain160 sequence_pool.output_norm")


def _rx_mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain exactly the five source RX entries")
    keys = set(value)
    if keys == set(SOURCE_RX):
        return {str(rx): value[rx] for rx in SOURCE_RX}
    if keys == {str(rx) for rx in SOURCE_RX}:
        return {str(rx): value[str(rx)] for rx in SOURCE_RX}
    raise ValueError(f"{name} must contain exactly RX1/3/4/6/8 with all integer or all string keys")


def combine(loss_by_RX, gate_by_RX, *, role="L_s", layer_kind=LAYER_KIND):
    """Sum fixed ORDER RX contributions with frozen Boolean accept/reject gates.

    ``loss_by_RX`` has five attached scalars already divided by the FULL legal
    triplet count. ``gate_by_RX`` is a complete five-RX Python-bool mapping or
    a CPU bool[5] tensor ordered RX1/3/4/6/8. No active-count reweighting occurs.
    All-False returns an exact differentiable zero, preserving a zero gradient.
    The gates are caller-supplied actual-credit decisions, not confidence,
    pseudo-label, phase or endpoint-agreement gates manufactured by this module.
    """
    _boundary(role, layer_kind)
    losses = _rx_mapping(loss_by_RX, "loss_by_RX")
    if isinstance(gate_by_RX, torch.Tensor):
        if gate_by_RX.device.type != "cpu" or gate_by_RX.shape != (5,) or gate_by_RX.dtype != torch.bool:
            raise ValueError("gate_by_RX tensor must be a frozen CPU bool[5] in source RX order")
        gates = {str(rx): value for rx, value in zip(SOURCE_RX, gate_by_RX.tolist())}
    else:
        gates = _rx_mapping(gate_by_RX, "gate_by_RX")
    if any(not isinstance(value, bool) for value in gates.values()):
        raise ValueError("every frozen RX gate must be Boolean, not a probability or numeric weight")
    prototype = losses["1"]
    if not isinstance(prototype, torch.Tensor):
        raise ValueError("loss_by_RX entries must be finite attached floating scalar tensors")
    for value in losses.values():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.ndim != 0 or not value.is_floating_point()
                or value.device != prototype.device or value.dtype != prototype.dtype
                or not bool(torch.isfinite(value)) or bool(value < 0)):
            raise ValueError("loss_by_RX entries must be nonnegative finite scalar tensors of one device/dtype")
    result = torch.stack([losses[str(rx)] * float(gates[str(rx)]) for rx in SOURCE_RX]).sum()
    if not bool(torch.isfinite(result)):
        raise ValueError("combined RX loss must be finite")
    return result


def objective(z, y, rx, day, physical_id, mode, *, coefficient=COEFFICIENT,
              margin=MARGIN, temperature=TEMPERATURE, role="L_s", layer_kind=LAYER_KIND):
    """Return pure selected loss, attached ORDER RX losses and cohort diagnostics.

    Every duplicate physical ID occurrence is excluded only from the auxiliary
    cohort; ordinary main L CE remains caller-owned on the original full batch.
    All legal triplets are enumerated: i/p same true TX/day, different RX;
    n differs TX and has p's RX/day. Positive and negative are detached CURRENT
    source endpoints, not EMA targets. Anchor remains attached. At satisfied
    margin, ReLU stops the derivative. No mining or active-only denominator.

    Each RX contribution is .1*sum(ReLU relations anchored at RX)/FULL triplet
    count, including all satisfied terms in that count. Their sum is ungated
    anchor-only ORDER. ``loss_by_RX`` ALWAYS contains these attached ORDER
    contributions, including when selected mode ZERO or SUPCON is requested.
    ZERO computes the same diagnostics then returns exact graph zero. SUPCON is
    ordinary full same-class contrastive loss with ALL endpoint graphs attached;
    its endpoint exposure differs from ORDER by definition. ``combine`` applies
    a later caller credit decision to ORDER; it never renormalizes accepted RX.

    Layer provenance and true-visible source labels are caller assertions. The
    pure function validates legal metadata/layer declaration but cannot inspect
    the upstream model graph to prove it is sequence_pool.output_norm before
    gain. No W, gains, model, utility, labels from U, or new parameters are inputs.
    Fixed coefficient/margin/temperature are .1; this is not a parameter scan.
    """
    _boundary(role, layer_kind)
    labels, receivers, days, identifiers = _validate(
        z, y, rx, day, physical_id, mode, coefficient, margin, temperature, role
    )
    if coefficient != COEFFICIENT or margin != MARGIN or temperature != TEMPERATURE:
        raise ValueError("this candidate fixes coefficient, margin and SupCon temperature at .1")
    identifier_count = Counter(identifiers)
    cohort = torch.tensor([index for index, identifier in enumerate(identifiers)
                           if identifier_count[identifier] == 1], dtype=torch.long)
    cohort_labels, cohort_rx, cohort_days = labels[cohort], receivers[cohort], days[cohort]
    features = F.normalize(z.index_select(0, cohort.to(z.device)), dim=1, eps=NORMALIZE_EPS)
    anchor_donor_cosine = features @ features.detach().T
    zero = (z * 0.).sum()
    triples, positive_pair_count, negative_pair_count = _triplets(cohort_labels, cohort_rx, cohort_days)
    count = triples.shape[0]
    if count:
        local = triples.to(z.device)
        positive_cos = anchor_donor_cosine[local[:, 0], local[:, 1]]
        negative_cos = anchor_donor_cosine[local[:, 0], local[:, 2]]
        relation = margin + negative_cos - positive_cos
        terms = F.relu(relation)
        order_loss = terms.sum() / count
        unsatisfied = int((relation > 0).sum())
        violations = int((negative_cos > positive_cos).sum())
        ties = int((negative_cos == positive_cos).sum())
    else:
        terms = features.new_empty((0,))
        relation = features.new_empty((0,))
        order_loss, unsatisfied, violations, ties = zero, 0, 0, 0
    loss_by_RX = {}
    rx_summary = {}
    for receiver in SOURCE_RX:
        selected = cohort_rx[triples[:, 0]] == receiver
        rx_count = int(selected.sum())
        active = int((relation[selected.to(z.device)] > 0).sum()) if count else 0
        rx_loss = coefficient * terms[selected.to(z.device)].sum() / count if count else zero
        loss_by_RX[str(receiver)] = rx_loss
        rx_summary[str(receiver)] = {
            "input_count": int((receivers == receiver).sum()),
            "cohort_count": int((cohort_rx == receiver).sum()),
            "triplet_count": rx_count,
            "anchor_count": int(torch.unique(triples[selected, 0]).numel()),
            "active_triplet_count": active, "satisfied_triplet_count": rx_count - active,
            "active_triplet_rate": active / rx_count if rx_count else None,
            "weighted_ORDER_contribution": float(rx_loss.detach()),
            "full_triplet_denominator": count,
        }
    ungated_order = combine(loss_by_RX, {receiver: True for receiver in SOURCE_RX},
                            role=role, layer_kind=layer_kind)
    supcon_loss, supcon_anchors, supcon_positive_pair_count = _supcon(
        features @ features.T, cohort_labels, temperature, zero
    )
    weighted_loss = zero if mode == "ZERO" else coefficient * supcon_loss if mode == "SUPCON" else ungated_order
    if not all(bool(torch.isfinite(value)) for value in (order_loss, supcon_loss, weighted_loss)):
        raise ValueError("cross-RX objective produced a nonfinite loss")
    class_exposure, reference_groups = _exposure(cohort_labels, cohort_rx, cohort_days, triples, supcon_anchors)
    summary = {
        "mode": mode, "role": role, "layer_kind": layer_kind,
        "coefficient": coefficient, "margin": margin, "temperature": temperature,
        "normalize_eps": NORMALIZE_EPS, "input_count": len(z),
        "unique_eligible_count": len(cohort), "duplicate_excluded_count": len(z) - len(cohort),
        "cohort_coverage": len(cohort) / len(z) if len(z) else 0.,
        "repeated_physical_ID_count": sum(value > 1 for value in identifier_count.values()),
        "triplet_count": count, "ORDER_anchor_count": int(torch.unique(triples[:, 0]).numel()),
        "ORDER_has_relations": bool(count), "ORDER_has_active_relations": bool(unsatisfied),
        "legal_positive_pair_count": positive_pair_count, "legal_reference_negative_pair_count": negative_pair_count,
        "unsatisfied_margin_count": unsatisfied, "satisfied_margin_count": count - unsatisfied,
        "unsatisfied_margin_rate": unsatisfied / count if count else None,
        "ordering_violations": violations, "ordering_tie_count": ties,
        "ORDER_loss": float(order_loss.detach()), "SUPCON_loss": float(supcon_loss.detach()),
        "ungated_weighted_ORDER_loss": float(ungated_order.detach()),
        "weighted_selected_loss": float(weighted_loss.detach()),
        "SUPCON_anchor_count": int(supcon_anchors.numel()), "SUPCON_positive_pair_count": supcon_positive_pair_count,
        "RX": rx_summary, "source_RX_counts": _histogram(receivers, SOURCE_RX),
        "source_day_counts": _histogram(days, SOURCE_DAY), "cohort_RX_counts": _histogram(cohort_rx, SOURCE_RX),
        "cohort_day_counts": _histogram(cohort_days, SOURCE_DAY),
        "class_exposure": class_exposure, "domain_reference_groups": reference_groups,
        "ORDER_denominator": "full legal triplet count including all satisfied terms; each RX contributes its sum over this same denominator",
        "ORDER_endpoint_graph": "anchor attached; positive/negative detached current source features; not EMA targets",
        "loss_by_RX_objective": "ungated attached weighted anchor-only ORDER in all selected modes",
        "exposure_note": "ordinary SUPCON attaches all endpoints and averages same-class-positive anchors; exposure differs from ORDER, not exact endpoint matching or a new contrastive invention",
        "credit_utility": "provided later by caller using actual AdamW and same-batch other-RX source L default W_pre; not inferred here",
        "pseudo_label_or_phase_gate_changes": False,
        "class_W_dependency": False, "gain_input_dependency": False,
        "extra_model": False, "optimizer_state": False, "new_state_count": 0, "new_parameters": False,
        "global_RNG_consumed": False,
        "external_source_and_layer_provenance": "asserted_by_caller_not_proven_by_pure_function",
    }
    return {"weighted_loss": weighted_loss, "loss_by_RX": loss_by_RX, "summary": summary,
            "tensors": {"cohort_indices": cohort, "ORDER_triplets": cohort[triples],
                        "SUPCON_anchor_indices": cohort[supcon_anchors],
                        "ORDER_active": (relation > 0).detach(), "ORDER_relation": relation.detach()}}
