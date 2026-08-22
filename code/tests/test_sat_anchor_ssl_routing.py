import torch

from cvsrffi.muse_ssdg import (
    calibrate_sat_anchor_thresholds,
    fuse_anchor_ema_probabilities,
    route_sat_anchor_trusted,
)


def test_anchor_ema_fusion_uses_both_teachers_and_normalizes_rows():
    anchor = torch.tensor([[0.81, 0.19], [0.25, 0.75]])
    ema = torch.tensor([[0.64, 0.36], [0.49, 0.51]])

    fused = fuse_anchor_ema_probabilities(anchor, ema, beta=0.5)

    assert torch.allclose(fused.sum(dim=1), torch.ones(2), atol=1e-7)
    assert fused.argmax(dim=1).tolist() == [0, 1]
    assert not torch.allclose(fused, anchor)
    assert not torch.allclose(fused, ema)


def test_trusted_route_requires_agreement_confidence_and_margin_without_fill():
    anchor = torch.tensor(
        [
            [0.95, 0.05],
            [0.10, 0.90],
            [0.80, 0.20],
            [0.92, 0.08],
        ]
    )
    ema = torch.tensor(
        [
            [0.90, 0.10],
            [0.85, 0.15],
            [0.72, 0.28],
            [0.88, 0.12],
        ]
    )

    route = route_sat_anchor_trusted(
        anchor,
        ema,
        confidence_thresholds=torch.tensor([0.85, 0.85]),
        margin_thresholds=torch.tensor([0.70, 0.70]),
        hard_max_fraction=1.0,
        fill_to_fraction=0.0,
    )

    assert route.agreement.tolist() == [True, False, True, True]
    assert route.strict.tolist() == [True, False, False, True]
    assert torch.equal(route.trusted, route.strict)
    assert route.no_identity.tolist() == [False, True, True, False]


def test_empty_strict_route_stays_empty_instead_of_filling_a_quota():
    anchor = torch.tensor([[0.55, 0.45], [0.40, 0.60]])
    ema = anchor.clone()

    route = route_sat_anchor_trusted(
        anchor,
        ema,
        confidence_thresholds=torch.tensor([0.99, 0.99]),
        margin_thresholds=torch.tensor([0.90, 0.90]),
        hard_max_fraction=0.25,
        fill_to_fraction=0.0,
    )

    assert route.trusted.sum().item() == 0
    assert route.no_identity.all()


def test_fixed_fill_is_explicit_control_and_not_the_adaptive_default():
    anchor = torch.tensor([[0.70, 0.30], [0.69, 0.31], [0.32, 0.68], [0.33, 0.67]])
    ema = anchor.clone()
    kwargs = dict(
        confidence_thresholds=torch.tensor([0.99, 0.99]),
        margin_thresholds=torch.tensor([0.90, 0.90]),
        hard_max_fraction=1.0,
    )

    adaptive = route_sat_anchor_trusted(anchor, ema, fill_to_fraction=0.0, **kwargs)
    fixed = route_sat_anchor_trusted(anchor, ema, fill_to_fraction=0.5, **kwargs)

    assert adaptive.trusted.sum().item() == 0
    assert fixed.strict.sum().item() == 0
    assert fixed.trusted.sum().item() == 2
    assert fixed.filled.sum().item() == 2


def test_class_receiver_cap_prevents_one_receiver_from_dominating_a_class():
    anchor = torch.tensor(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.05, 0.95],
            [0.06, 0.94],
        ]
    )
    ema = anchor.clone()
    receivers = torch.tensor([0, 0, 0, 1, 0, 1])

    route = route_sat_anchor_trusted(
        anchor,
        ema,
        confidence_thresholds=torch.tensor([0.80, 0.80]),
        margin_thresholds=torch.tensor([0.60, 0.60]),
        receivers=receivers,
        hard_max_fraction=0.5,
        class_balanced_cap=True,
        receiver_balanced_cap=True,
    )

    selected = route.trusted.nonzero(as_tuple=False).flatten().tolist()
    assert selected == [0, 3, 4]
    assert route.receiver_cap.tolist() == [True, False, False, True, True, False]


def test_vcal_calibration_is_class_complete_and_bounds_selected_error():
    probability = torch.tensor(
        [
            [0.99, 0.01],
            [0.90, 0.10],
            [0.85, 0.15],
            [0.05, 0.95],
            [0.15, 0.85],
            [0.20, 0.80],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    thresholds = calibrate_sat_anchor_thresholds(
        probability,
        labels,
        num_classes=2,
        epsilon=0.0,
    )

    predicted = probability.argmax(dim=1)
    top2 = probability.topk(2, dim=1).values
    selected = (
        probability.max(dim=1).values >= thresholds.confidence[predicted]
    ) & ((top2[:, 0] - top2[:, 1]) >= thresholds.margin[predicted])
    assert thresholds.confidence.shape == (2,)
    assert thresholds.margin.shape == (2,)
    assert selected.any()
    assert torch.equal(predicted[selected], labels[selected])
