"""Synthetic CPU checks only; no dataset, checkpoint, model or launch access."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from .objective import objective


def _fixture():
    generator = torch.Generator(device="cpu").manual_seed(7143)
    labels, receivers, days = [], [], []
    for label in range(6):
        for receiver, sample_day in ((1, 1), (3, 1), (4, 2), (6, 2)):
            labels.append(label); receivers.append(receiver); days.append(sample_day)
    labels.extend((0, 0)); receivers.extend((3, 3)); days.extend((1, 1))
    ids = [f"p{index}" for index in range(len(labels))]
    labels.append(labels[2]); receivers.append(receivers[2]); days.append(days[2]); ids.append(ids[2])
    z = torch.randn(len(labels), 160, generator=generator, dtype=torch.float64)
    return z, torch.tensor(labels), torch.tensor(receivers), torch.tensor(days), ids


def _raises(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected malformed-input or source-role rejection")


def _rng():
    return random.getstate(), torch.get_rng_state().clone(), np.random.get_state()


def _rng_equal(left, right):
    return left[0] == right[0] and torch.equal(left[1], right[1]) and left[2][0] == right[2][0] and np.array_equal(left[2][1], right[2][1]) and left[2][2:] == right[2][2:]


def run_checks():
    rng_before = _rng()
    z, y, rx, day, ids = _fixture()
    result = objective(z, y, rx, day, ids, "ORDER")
    cohort = result["tensors"]["cohort_indices"].tolist()
    assert result["summary"]["input_count"] == 27
    assert result["summary"]["unique_eligible_count"] == 25
    assert result["summary"]["duplicate_excluded_count"] == 2
    assert 2 not in cohort and 26 not in cohort
    id_counts = Counter(ids)
    assert all(id_counts[ids[index]] == 1 for index in cohort)
    expected = {(i, p, n) for i in cohort for p in cohort for n in cohort
                if len({ids[i], ids[p], ids[n]}) == 3
                and y[i] == y[p] and day[i] == day[p] and rx[i] != rx[p]
                and y[n] != y[p] and rx[n] == rx[p] and day[n] == day[p]}
    triples = result["tensors"]["ORDER_triplets"].tolist()
    assert set(map(tuple, triples)) == expected and len(triples) == len(expected)
    assert result["summary"]["triplet_count"] == len(expected) > 0
    for i, p, n in triples:
        assert y[i] == y[p] and day[i] == day[p] and rx[i] != rx[p]
        assert y[n] != y[p] and rx[n] == rx[p] and day[n] == day[p]
    normalized = F.normalize(z, dim=1, eps=1e-4)
    direct_relations = torch.stack([0.1 + normalized[i] @ normalized[n] - normalized[i] @ normalized[p] for i, p, n in triples])
    expected_order = direct_relations.relu().mean()
    assert torch.allclose(result["weighted_loss"], expected_order * 0.1, atol=1e-14, rtol=0)
    assert result["summary"]["unsatisfied_margin_count"] == int((direct_relations > 0).sum())
    assert result["summary"]["satisfied_margin_count"] + result["summary"]["unsatisfied_margin_count"] == len(expected)
    # Same source identity, all legal relations satisfied: true stopped derivative.
    identity = torch.zeros(len(ids), 160, dtype=torch.float64)
    identity[torch.arange(len(ids)), y] = 1.0
    identity.requires_grad_()
    satisfied = objective(identity, y, rx, day, ids, "ORDER")
    satisfied["weighted_loss"].backward()
    assert satisfied["summary"]["triplet_count"] > 0
    assert satisfied["summary"]["unsatisfied_margin_count"] == 0
    assert satisfied["weighted_loss"].item() == 0.0
    assert bool((identity.grad == 0).all())
    # A violating, noncollinear relation has gradient on all three endpoints.
    active_features = torch.zeros(3, 160, dtype=torch.float64)
    active_features[0, 0] = 1; active_features[1, 1] = 1
    active_features[2, 0] = 0.8; active_features[2, 2] = 0.6
    active_features.requires_grad_()
    active_y, active_rx, active_day = torch.tensor([0, 0, 1]), torch.tensor([1, 3, 3]), torch.ones(3, dtype=torch.long)
    active_ids = ["anchor", "positive", "negative"]
    active = objective(active_features, active_y, active_rx, active_day, active_ids, "ORDER")
    assert active["tensors"]["ORDER_triplets"].tolist() == [[0, 1, 2]]
    assert abs(active["weighted_loss"].item() - 0.09) < 1e-14
    assert active["summary"]["unsatisfied_margin_count"] == active["summary"]["ordering_violations"] == 1
    active["weighted_loss"].backward()
    endpoint_norms = active_features.grad.norm(dim=1)
    assert bool(torch.isfinite(active_features.grad).all()) and bool((endpoint_norms > 0).all())
    zero_features = active_features.detach().clone().requires_grad_()
    zero = objective(zero_features, active_y, active_rx, active_day, active_ids, "ZERO")
    zero["weighted_loss"].backward()
    assert zero["weighted_loss"].item() == 0.0 and bool((zero_features.grad == 0).all())
    assert zero["summary"]["ORDER_loss"] > 0 and zero["summary"]["SUPCON_loss"] > 0
    # SupCon independent per-anchor formula: all nonself packets denominator,
    # same-label positives across RX/day and mean only anchors with positives.
    supcon = objective(z, y, rx, day, ids, "SUPCON")
    rows = []
    for i in cohort:
        positives = [j for j in cohort if j != i and y[j] == y[i]]
        if positives:
            denominator = torch.logsumexp(torch.stack([normalized[i] @ normalized[j] / 0.1 for j in cohort if j != i]), dim=0)
            rows.append(torch.stack([denominator - normalized[i] @ normalized[j] / 0.1 for j in positives]).mean())
    expected_supcon = torch.stack(rows).mean()
    assert torch.allclose(supcon["weighted_loss"], expected_supcon * 0.1, atol=1e-14, rtol=0)
    assert supcon["summary"]["SUPCON_positive_pair_count"] > supcon["summary"]["legal_positive_pair_count"]
    stable_input = active_features.detach().clone().to(torch.float32).requires_grad_()
    stable = objective(stable_input, active_y, active_rx, active_day, active_ids, "SUPCON")
    stable["weighted_loss"].backward()
    assert bool(torch.isfinite(stable["weighted_loss"])) and bool(torch.isfinite(stable_input.grad).all())
    # Permuting numeric class labels does not change either loss or relation set.
    permutation = torch.tensor([3, 5, 1, 4, 0, 2])
    for mode in ("ZERO", "ORDER", "SUPCON"):
        original = objective(z, y, rx, day, ids, mode)
        permuted = objective(z, permutation[y], rx, day, ids, mode)
        assert torch.equal(original["weighted_loss"], permuted["weighted_loss"])
        assert torch.equal(original["tensors"]["ORDER_triplets"], permuted["tensors"]["ORDER_triplets"])
        json.dumps(original["summary"], allow_nan=False)
    # Empty, single packet, no legal contexts, and duplicates-only cases.
    for mode in ("ZERO", "ORDER", "SUPCON"):
        for count, identifiers in ((0, []), (1, ["only"]), (2, ["repeat", "repeat"])):
            empty = torch.zeros(count, 160, dtype=torch.float64, requires_grad=True)
            answer = objective(empty, torch.arange(count) % 6, torch.ones(count, dtype=torch.long), torch.ones(count, dtype=torch.long), identifiers, mode)
            assert answer["weighted_loss"].item() == 0.0 and answer["summary"]["triplet_count"] == 0
            assert answer["summary"]["unsatisfied_margin_rate"] is None
            answer["weighted_loss"].backward()
            assert empty.grad is not None and bool((empty.grad == 0).all())
    no_relation = objective(z[:3], torch.tensor([0, 1, 2]), torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.long), [1, 2, 3], "ORDER")
    assert no_relation["weighted_loss"].item() == 0 and not no_relation["summary"]["ORDER_has_relations"]
    # Integer opaque IDs are supported, with the same duplicate exclusion rule.
    int_ids = torch.tensor([int(identifier[1:]) for identifier in ids])
    integer_result = objective(z, y, rx, day, int_ids, "ORDER")
    assert torch.equal(integer_result["weighted_loss"], result["weighted_loss"])
    assert torch.equal(integer_result["tensors"]["cohort_indices"], result["tensors"]["cohort_indices"])
    if hasattr(torch, "uint64"):
        unsigned_ids = torch.tensor([int(identifier[1:]) + (1 << 63) for identifier in ids], dtype=torch.uint64)
        unsigned_result = objective(z, y, rx, day, unsigned_ids, "ORDER")
        assert torch.equal(unsigned_result["weighted_loss"], result["weighted_loss"])
        assert torch.equal(unsigned_result["tensors"]["cohort_indices"], result["tensors"]["cohort_indices"])
    for role in ("U_s", "U", "V", "query"):
        _raises(lambda role=role: objective(z, y, rx, day, ids, "ORDER", role=role))
    _raises(lambda: objective(z, None, rx, day, ids, "ORDER"))
    bad_y = y.clone(); bad_y[0] = -1
    _raises(lambda: objective(z, bad_y, rx, day, ids, "ORDER"))
    bad_y[0] = 6
    _raises(lambda: objective(z, bad_y, rx, day, ids, "ORDER"))
    _raises(lambda: objective(z, y.to(torch.float32), rx, day, ids, "ORDER"))
    _raises(lambda: objective(z, y[:, None], rx, day, ids, "ORDER"))
    bad_rx = rx.clone(); bad_rx[0] = 0
    _raises(lambda: objective(z, y, bad_rx, day, ids, "ORDER"))
    bad_day = day.clone(); bad_day[0] = 0
    _raises(lambda: objective(z, y, rx, bad_day, ids, "ORDER"))
    _raises(lambda: objective(z, y, rx[:-1], day, ids, "ORDER"))
    _raises(lambda: objective(z, y, rx, day, ids[:-1], "ORDER"))
    _raises(lambda: objective(z[:, :159], y, rx, day, ids, "ORDER"))
    nonfinite = z.clone(); nonfinite[0, 0] = float("nan")
    _raises(lambda: objective(nonfinite, y, rx, day, ids, "ORDER"))
    nonfinite[0, 0] = float("inf")
    _raises(lambda: objective(nonfinite, y, rx, day, ids, "ORDER"))
    _raises(lambda: objective(z, y, rx, day, ids, "ORDER", temperature=0))
    _raises(lambda: objective(z, y, rx, day, ids, "INVALID"))
    assert _rng_equal(rng_before, _rng())
    return {
        "status": "VERIFIED", "environment": {"torch": torch.__version__, "device": "cpu", "dtype": "float64 + float32 SupCon stability probe"},
        "checks": {"every_legal_triplet_exact_enumeration": True, "reference_RX_day_matching": True,
                   "all_satisfied_derivative_exact_zero": True, "violating_derivative_active": True,
                   "ZERO_exact_zero": True, "duplicates_all_occurrences_excluded": True,
                   "no_global_RNG_consumption": True, "label_permutation_invariance": True,
                   "ordinary_SUPCON_formula": True, "stable_SUPCON_finite": True,
                   "empty_no_signal_exact_zero": True, "source_role_and_malformed_metadata_negative_cases": True},
        "active_ORDER_endpoint_gradient_norms": endpoint_norms.tolist(),
        "satisfied_ORDER_gradient_norm": float(identity.grad.norm().item()),
        "ZERO_gradient_norm": float(zero_features.grad.norm().item()),
        "source_shaped_counts": {key: result["summary"][key] for key in (
            "input_count", "unique_eligible_count", "duplicate_excluded_count", "triplet_count",
            "ORDER_anchor_count", "satisfied_margin_count", "unsatisfied_margin_count", "SUPCON_anchor_count")},
        "source_shaped_summary": result["summary"],
        "scientific_status": "SYNTHETIC_MATHEMATICAL_CHECK_ONLY_NOT_REAL_TRAINING_OR_EFFICACY",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args()
    checked = run_checks()
    display = {key: value for key, value in checked.items() if key != "source_shaped_summary"} if arguments.compact else checked
    print(json.dumps(display, ensure_ascii=False, allow_nan=False, indent=2))
