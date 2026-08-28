import pytest
import torch

from cvsrffi.muse_ssdg import (
    MUSETrainingHeads,
    align_source_domain_prior,
    compute_muse_reliability,
    geometric_fuse_probabilities,
    js_head_disagreement,
    route_fasttrust,
    route_muse_reliability,
)


def test_local_probability_keeps_finite_gradients_when_amp_probabilities_underflow():
    heads = MUSETrainingHeads(4, 4, 3, 1, 2).half()
    with torch.no_grad():
        heads.shared_projection.weight.zero_()
        heads.shared_classifier.weight.zero_()
        heads.shared_classifier.bias.copy_(
            torch.tensor([0.0, -20.0, -40.0], dtype=torch.float16)
        )
        heads.domain_delta_left.zero_()
        heads.domain_delta_right.zero_()

    features = torch.ones(1, 4, dtype=torch.float16, requires_grad=True)
    probability = heads.local_prob(features, torch.tensor([0]))
    loss = torch.nn.functional.nll_loss(
        probability.clamp_min(1e-8).log(),
        torch.tensor([0]),
    )
    loss.backward()

    assert probability.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in heads.parameters()
    )


def test_three_head_fusion_is_normalized_and_routing_is_a_partition():
    p0 = torch.tensor([[0.80, 0.15, 0.05], [0.40, 0.35, 0.25]])
    p1 = torch.tensor([[0.75, 0.20, 0.05], [0.38, 0.37, 0.25]])
    p2 = torch.tensor([[0.85, 0.10, 0.05], [0.34, 0.33, 0.33]])
    fused = geometric_fuse_probabilities([p0, p1, p2], [0.50, 0.25, 0.25])
    assert torch.allclose(fused.sum(1), torch.ones(2), atol=1e-6)
    reliability = torch.tensor([0.91, 0.52, 0.18])
    route = route_muse_reliability(reliability, high_threshold=0.80, low_threshold=0.30)
    stacked = torch.stack([route.high, route.mid, route.low]).int().sum(0)
    assert stacked.tolist() == [1, 1, 1]


def test_default_fusion_weights_match_documented_explicit_weights():
    heads = [
        torch.tensor([[0.80, 0.15, 0.05]]),
        torch.tensor([[0.75, 0.20, 0.05]]),
        torch.tensor([[0.85, 0.10, 0.05]]),
    ]
    default = geometric_fuse_probabilities(heads)
    explicit = geometric_fuse_probabilities(heads, [0.50, 0.25, 0.25])
    assert torch.allclose(default, explicit, atol=1e-7)


def test_fusion_sanitizes_nonfinite_probabilities_and_rejects_zero_weight_sum():
    probabilities = [
        torch.tensor([[float("nan"), float("inf"), -1.0]]),
        torch.tensor([[0.2, 0.3, 0.5]]),
    ]
    fused = geometric_fuse_probabilities(probabilities, [1.0, 1.0])
    assert torch.isfinite(fused).all()
    assert torch.all(fused >= 0.0)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(1), atol=1e-6)

    with pytest.raises(ValueError, match="weight"):
        geometric_fuse_probabilities(probabilities, [0.0, 0.0])


def test_fp16_zero_and_nonfinite_probability_heads_stay_finite():
    bad_heads = [
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float16),
        torch.tensor([[float("nan"), float("-inf"), 0.0]], dtype=torch.float16),
    ]
    fused = geometric_fuse_probabilities(bad_heads, [0.5, 0.5])
    disagreement = js_head_disagreement(bad_heads)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(disagreement).all()
    assert torch.allclose(fused.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_source_domain_prior_alignment_clips_ratio_and_normalizes():
    probability = torch.tensor([[0.2, 0.3, 0.5]])
    domain_prior = torch.tensor([0.01, 0.99, 0.50])
    global_prior = torch.tensor([0.50, 0.25, 0.25])
    aligned = align_source_domain_prior(
        probability,
        domain_prior,
        global_prior,
        gamma=1.0,
        ratio_clip=(0.5, 2.0),
    )
    expected = torch.tensor([[0.50, 0.1875, 0.3125]])
    assert torch.allclose(aligned, expected, atol=1e-6)
    assert torch.allclose(aligned.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_js_head_disagreement_is_zero_for_identical_heads_and_finite_for_bad_input():
    head = torch.tensor([[0.8, 0.15, 0.05]])
    assert torch.allclose(js_head_disagreement([head, head]), torch.zeros(1), atol=1e-7)

    disagreement = js_head_disagreement(
        [
            torch.tensor([[float("nan"), 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ]
    )
    assert torch.isfinite(disagreement).all()
    assert disagreement.item() > 0.0


def test_reliability_decreases_when_head_disagreement_increases():
    stable = compute_muse_reliability(
        confidence=torch.tensor([0.9]),
        margin=torch.tensor([0.5]),
        js=torch.tensor([0.01]),
        proto_distance=torch.tensor([0.1]),
        stability=torch.tensor([1.0]),
    )
    disputed = compute_muse_reliability(
        confidence=torch.tensor([0.9]),
        margin=torch.tensor([0.5]),
        js=torch.tensor([0.30]),
        proto_distance=torch.tensor([0.1]),
        stability=torch.tensor([1.0]),
    )
    assert stable.item() > disputed.item()


def test_reliability_keeps_decreasing_for_js_values_above_one():
    lower = compute_muse_reliability(0.9, 0.5, 1.0, 0.1, 1.0)
    higher = compute_muse_reliability(0.9, 0.5, 1.5, 0.1, 1.0)
    assert lower.item() > higher.item()


def test_reliability_is_bounded_and_monotone_for_each_evidence_axis():
    baseline = compute_muse_reliability(0.5, 0.5, 0.1, 0.1, 0.5)
    assert 0.0 <= baseline.item() <= 1.0
    assert compute_muse_reliability(0.8, 0.5, 0.1, 0.1, 0.5) > baseline
    assert compute_muse_reliability(0.5, 0.8, 0.1, 0.1, 0.5) > baseline
    assert compute_muse_reliability(0.5, 0.5, 0.3, 0.1, 0.5) < baseline
    assert compute_muse_reliability(0.5, 0.5, 0.1, 0.8, 0.5) < baseline
    assert compute_muse_reliability(0.5, 0.5, 0.1, 0.1, 0.8) > baseline

    abnormal = compute_muse_reliability(
        torch.tensor([float("nan"), float("inf"), -1.0]),
        torch.tensor([0.5, 2.0, -1.0]),
        torch.tensor([0.1, float("nan"), float("inf")]),
        torch.tensor([0.1, float("inf"), -1.0]),
        torch.tensor([1.0, float("nan"), 2.0]),
    )
    assert torch.isfinite(abnormal).all()
    assert torch.all((abnormal >= 0.0) & (abnormal <= 1.0))


def test_routing_uses_high_inclusive_low_exclusive_boundaries():
    route = route_muse_reliability(
        torch.tensor([0.80, 0.30, 0.299999, 1.2, float("nan")]),
        high_threshold=0.80,
        low_threshold=0.30,
    )
    assert route.high.dtype == torch.bool
    assert route.mid.dtype == torch.bool
    assert route.low.dtype == torch.bool
    assert route.high.tolist() == [True, False, False, True, False]
    assert route.mid.tolist() == [False, True, False, False, False]
    assert route.low.tolist() == [False, False, True, False, True]
    assert torch.all(
        torch.stack([route.high, route.mid, route.low]).int().sum(dim=0) == 1
    )


def test_fasttrust_hard_requires_stability_three_head_agreement_and_balanced_cap():
    reliability = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.60, 0.55])
    stable = torch.tensor([True, True, True, True, False, True, True, True])
    predictions = (
        torch.tensor([0, 0, 0, 1, 1, 1, 0, 1]),
        torch.tensor([0, 0, 0, 1, 1, 0, 0, 1]),
        torch.tensor([0, 0, 0, 1, 1, 1, 0, 1]),
    )
    evidence = [torch.nn.functional.one_hot(row, num_classes=2).float() for row in predictions]

    route = route_fasttrust(
        reliability,
        stable,
        evidence,
        high_threshold=0.80,
        low_threshold=0.30,
        hard_max_fraction=0.25,
        identity_max_fraction=0.50,
    )

    assert route.agreement.tolist() == [True, True, True, True, True, False, True, True]
    assert route.hard.tolist() == [True, False, False, True, False, False, False, False]
    assert int(route.hard.sum()) == 2
    assert int((route.hard | route.soft | route.candidate).sum()) == 4
    assert not route.hard[4]
    assert not route.hard[5]
    assert torch.all(
        torch.stack([route.hard, route.soft, route.candidate, route.no_identity])
        .int()
        .sum(0)
        == 1
    )


def test_fasttrust_routing_is_deterministic_for_ties_and_handles_empty_batches():
    evidence = [torch.tensor([[0.8, 0.2]] * 8)] * 3
    first = route_fasttrust(
        torch.full((8,), 0.9),
        torch.ones(8, dtype=torch.bool),
        evidence,
        high_threshold=0.8,
        low_threshold=0.3,
    )
    second = route_fasttrust(
        torch.full((8,), 0.9),
        torch.ones(8, dtype=torch.bool),
        evidence,
        high_threshold=0.8,
        low_threshold=0.3,
    )
    assert torch.equal(first.hard, second.hard)
    assert first.hard.nonzero().flatten().tolist() == [0, 1]

    empty = route_fasttrust(
        torch.empty(0),
        torch.empty(0, dtype=torch.bool),
        [torch.empty(0, 2)] * 3,
        high_threshold=0.8,
        low_threshold=0.3,
    )
    assert empty.hard.numel() == 0
    assert empty.no_identity.numel() == 0


def test_fasttrust_no_class_balanced_cap_keeps_global_reliability_order():
    reliability = torch.tensor([0.99, 0.98, 0.97, 0.96])
    stable = torch.ones(4, dtype=torch.bool)
    predictions = torch.tensor([0, 0, 0, 1])
    evidence = [torch.nn.functional.one_hot(predictions, num_classes=2).float()] * 3

    balanced = route_fasttrust(
        reliability,
        stable,
        evidence,
        high_threshold=0.8,
        low_threshold=0.3,
        hard_max_fraction=0.5,
        identity_max_fraction=0.5,
        class_balanced_cap=True,
    )
    global_only = route_fasttrust(
        reliability,
        stable,
        evidence,
        high_threshold=0.8,
        low_threshold=0.3,
        hard_max_fraction=0.5,
        identity_max_fraction=0.5,
        class_balanced_cap=False,
    )

    assert balanced.hard.tolist() == [True, False, False, True]
    assert global_only.hard.tolist() == [True, True, False, False]


def test_fasttrust_hard_only_no_fill_never_backfills_mid_or_low_reliability():
    reliability = torch.tensor([0.99, 0.98, 0.79, 0.70, 0.40, 0.29, 0.20, 0.10])
    stable = torch.ones(8, dtype=torch.bool)
    predictions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    evidence = [torch.nn.functional.one_hot(predictions, num_classes=2).float()] * 3

    route = route_fasttrust(
        reliability,
        stable,
        evidence,
        high_threshold=0.80,
        low_threshold=0.30,
        hard_max_fraction=0.50,
        identity_max_fraction=1.00,
        class_balanced_cap=True,
        hard_only_no_fill=True,
    )

    assert route.hard.tolist() == [True, True, False, False, False, False, False, False]
    assert not bool(route.soft.any())
    assert not bool(route.candidate.any())
    assert route.no_identity.tolist() == [False, False, True, True, True, True, True, True]
