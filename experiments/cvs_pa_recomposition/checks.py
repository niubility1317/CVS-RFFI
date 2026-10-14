"""CPU-only mathematical checks; no data, checkpoints, or training launches."""

from __future__ import annotations

import json
import random
import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .objective import recomposition


def _fixture():
    generator = torch.Generator(device="cpu").manual_seed(2714)
    receiver = [1] * 3 + [3] * 4 + [4] * 5 + [6] * 2 + [8] * 3 + [1] * 2 + [3]
    labels = [0] * 12 + [2] * 5 + [5] * 3
    days = [1] * 12 + [2] * 5 + [3] * 3
    ids = [f"packet_{index}" for index in range(20)]
    # A second view of one physical source packet must not become a donor.
    receiver.append(1); labels.append(0); days.append(1); ids.append(ids[0])
    size = len(labels)
    features = [torch.randn(size, 12, generator=generator, dtype=torch.float64) for _ in range(3)]
    weight = torch.randn(6, 12, generator=generator, dtype=torch.float64).requires_grad_()
    return features, weight, torch.tensor(labels), torch.tensor(receiver), torch.tensor(days), ids


def _call(features, weight, labels, receivers, days, ids, mode, **kwargs):
    return recomposition(*features, weight, 30.0, labels, receivers, days, ids, mode, **kwargs)


def _raises(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected a source-contract or mathematical input failure")


def run_checks():
    features, weight, labels, receivers, days, ids = _fixture()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    same = _call(features, weight, labels, receivers, days, ids, "SAME_RX")
    cross = _call(features, weight, labels, receivers, days, ids, "CROSS_RX")
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert same["summary"]["counts"] == cross["summary"]["counts"]
    cohort = same["tensors"]["cohort"]
    assert torch.equal(cohort, cross["tensors"]["cohort"])
    assert cohort.numel() == 13
    counts = same["summary"]["counts"]
    assert counts["excluded_duplicate_physical_count"] == 1
    assert counts["excluded_unbalanced_count"] == 4
    assert counts["excluded_insufficient_count"] == 3
    assert counts["cohort_count"] + counts["excluded_duplicate_physical_count"] + counts["excluded_unbalanced_count"] + counts["excluded_insufficient_count"] == labels.numel()
    for mode in ("SAME_RX", "CROSS_RX"):
        assert torch.equal(same["tensors"][f"{mode}_pairs"], cross["tensors"][f"{mode}_pairs"])
        pairs = same["tensors"][f"{mode}_pairs"]
        anchors, donors = pairs[:, 0], pairs[:, 1]
        assert torch.equal(labels[anchors], labels[donors])
        assert torch.equal(days[anchors], days[donors])
        assert bool((anchors != donors).all())
        assert all(ids[i] != ids[j] for i, j in pairs.tolist())
        assert Counter(anchors.tolist()) == Counter(cohort.tolist()) == Counter(donors.tolist())
        assert all(count == 1 for count in Counter(anchors.tolist()).values())
        assert bool(((receivers[anchors] == receivers[donors]) if mode == "SAME_RX" else (receivers[anchors] != receivers[donors])).all())
        zero_alpha = _call(features, weight, labels, receivers, days, ids, mode, alpha=0.0)
        assert torch.equal(zero_alpha["tensors"][f"{mode}_CE"], zero_alpha["tensors"]["deployment_CE"])
        assert zero_alpha["summary"][mode]["PA_replacement_norm_max"] == 0
    identical_b = features[1][0].expand_as(features[1]).clone()
    identical = [features[0], identical_b, features[2]]
    identical_result = _call(identical, weight, labels, receivers, days, ids, "CROSS_RX")
    for mode in ("SAME_RX", "CROSS_RX"):
        assert torch.equal(identical_result["tensors"][f"{mode}_CE"], identical_result["tensors"]["deployment_CE"])
        assert identical_result["summary"][mode]["PA_replacement_norm_max"] == 0
    # Independently verify cosine CE and PA-only forward/reverse feature math.
    pairs = cross["tensors"]["CROSS_RX_pairs"]
    i, j = pairs[:, 0], pairs[:, 1]
    expected_forward = features[0][i] + 0.8 * features[1][i] + 0.2 * features[1][j] + features[2][i]
    expected_reverse = features[0][j] + 0.8 * features[1][j] + 0.2 * features[1][i] + features[2][j]
    assert torch.allclose(cross["tensors"]["CROSS_RX_features"]["forward"], expected_forward, atol=1e-14, rtol=0)
    assert torch.allclose(cross["tensors"]["CROSS_RX_features"]["reverse"], expected_reverse, atol=1e-14, rtol=0)
    expected_logits = F.linear(F.normalize(features[0][cohort] + features[1][cohort] + features[2][cohort], dim=1, eps=1e-4), F.normalize(weight.detach(), dim=1, eps=1e-4)) * 30
    expected_ce = F.cross_entropy(expected_logits, labels[cohort], reduction="none").mean()
    assert torch.equal(expected_ce, cross["tensors"]["deployment_CE"])
    tiny_features = [feature * 1e-8 for feature in features]
    tiny_weight = weight * 1e-8
    tiny = _call(tiny_features, tiny_weight, labels, receivers, days, ids, "CROSS_RX", alpha=0.0)
    tiny_joint = sum(feature[cohort] for feature in tiny_features)
    tiny_default_logits = F.linear(F.normalize(tiny_joint, dim=1, eps=1e-4), F.normalize(tiny_weight.detach(), dim=1, eps=1e-4)) * 30
    tiny_default_ce = F.cross_entropy(tiny_default_logits, labels[cohort], reduction="none").mean()
    assert torch.equal(tiny_default_ce, tiny["tensors"]["deployment_CE"])
    assert torch.equal(tiny["tensors"]["SAME_RX_CE"], tiny_default_ce)
    assert torch.equal(tiny["tensors"]["CROSS_RX_CE"], tiny_default_ce)
    wrong_eps_logits = F.linear(F.normalize(tiny_joint, dim=1, eps=1e-12), F.normalize(tiny_weight.detach(), dim=1, eps=1e-12)) * 30
    assert not torch.allclose(tiny_default_logits, wrong_eps_logits)
    gradients = {}
    for mode in ("SAME_RX", "CROSS_RX", "ZERO"):
        raw = [feature.clone().requires_grad_() for feature in features]
        gains = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (0.7, 0.9, 0.5)]
        deployed = [feature * gain.detach() for feature, gain in zip(raw, gains)]
        for feature in deployed:
            feature.retain_grad()
        result = _call(deployed, weight, labels, receivers, days, ids, mode)
        assert torch.equal(result["tensors"]["cohort"], cohort)
        assert result["summary"]["paired_metadata"]["computed_prediction_count_all_modes"] == 5 * cohort.numel()
        result["weighted_loss"].backward()
        norms = [float(feature.grad.norm().item()) for feature in deployed]
        assert all(bool(torch.isfinite(feature.grad).all()) for feature in deployed)
        assert weight.grad is None and all(gain.grad is None for gain in gains)
        if mode == "ZERO":
            assert result["weighted_loss"].item() == 0.0 and all(norm == 0.0 for norm in norms)
        else:
            assert all(norm > 0 for norm in norms)
            assert all(feature.grad is not None and bool(torch.isfinite(feature.grad).all()) and feature.grad.norm() > 0 for feature in raw)
        gradients[mode] = {"a_norm": norms[0], "b_norm": norms[1], "c_norm": norms[2], "classW_grad": None, "gain_grads": [None, None, None]}
        json.dumps(result["summary"], allow_nan=False)
    # No eligible cells, including a wholly empty batch, returns exact zero.
    for selection in (slice(17, 20), slice(0, 0)):
        empty_features = [feature[selection].clone().requires_grad_() for feature in features]
        empty_result = _call(empty_features, weight, labels[selection], receivers[selection], days[selection], ids[selection], "CROSS_RX")
        assert empty_result["weighted_loss"].item() == 0.0
        assert empty_result["summary"]["counts"]["coverage"] == 0.0
        empty_result["weighted_loss"].backward()
        assert all(feature.grad is not None and bool((feature.grad == 0).all()) for feature in empty_features)
    # Same-physical duplicates cannot fake a two-packet RX cell.
    duplicate_only = [feature[:4] for feature in features]
    duplicate_result = _call(duplicate_only, weight, torch.zeros(4, dtype=torch.long), torch.tensor([1, 1, 3, 3]), torch.ones(4, dtype=torch.long), ["p1", "p1", "p3", "p3"], "CROSS_RX")
    assert duplicate_result["summary"]["counts"]["cohort_count"] == 0
    for role in ("U_s", "query", "V"):
        _raises(lambda role=role: _call(features, weight, labels, receivers, days, ids, "CROSS_RX", role=role))
    bad_labels = labels.clone(); bad_labels[0] = 6
    _raises(lambda: _call(features, weight, bad_labels, receivers, days, ids, "SAME_RX"))
    _raises(lambda: _call(features, weight, labels.to(torch.float64), receivers, days, ids, "SAME_RX"))
    bad_rx = receivers.clone(); bad_rx[0] = 0
    _raises(lambda: _call(features, weight, labels, bad_rx, days, ids, "CROSS_RX"))
    bad_day = days.clone(); bad_day[0] = 0
    _raises(lambda: _call(features, weight, labels, receivers, bad_day, ids, "CROSS_RX"))
    conflict = list(ids); conflict[3] = ids[0]
    _raises(lambda: _call(features, weight, labels, receivers, days, conflict, "CROSS_RX"))
    _raises(lambda: _call(features, weight, labels, receivers, days, ids[:-1], "CROSS_RX"))
    _raises(lambda: _call(features, weight, labels, receivers, days, ids, "CROSS_RX", alpha=-0.1))
    nonfinite = [feature.clone() for feature in features]; nonfinite[1][0, 0] = float("nan")
    _raises(lambda: _call(nonfinite, weight, labels, receivers, days, ids, "CROSS_RX"))
    _raises(lambda: _call(features, weight, labels, receivers, days, ids, "INVALID"))
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    return {
        "status": "VERIFIED", "environment": {"torch": torch.__version__, "device": "cpu", "dtype": "float64"},
        "checks": {"alpha0_deployment_CE": True, "identical_b_deployment_CE": True,
                   "tiny_norm_default_head_eps1e4": True,
                   "same_TX_day_physical_boundaries": True, "equal_endpoint_exposure": True,
                   "source_permissions_negative_cases": True, "private_rng_unchanged": True,
                   "ZERO_exact_zero": True, "empty_exact_zero": True, "duplicate_physical_not_counted_twice": True},
        "gradient_norms": gradients, "same_RX": same["summary"], "cross_RX": cross["summary"],
        "scientific_status": "CPU_MATHEMATICAL_PROTOTYPE_ONLY_NOT_RELEASED_OR_EFFICACY_EVIDENCE",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args()
    result = run_checks()
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    display = {key: value for key, value in result.items() if key not in ("same_RX", "cross_RX")} if arguments.compact else result
    print(json.dumps(display, ensure_ascii=False, allow_nan=False, indent=2))
