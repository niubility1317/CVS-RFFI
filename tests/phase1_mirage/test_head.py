"""Behavioral checks for the role-blind MIRAGE open-world decision head."""

from __future__ import annotations

import copy
import importlib
import math

import pytest
import torch
import torch.nn.functional as functional


def _head_api():
    """Import the Task 5 boundary inside tests so RED proves it is absent."""

    try:
        module = importlib.import_module("cvsrffi.phase1_mirage.head")
    except ModuleNotFoundError as error:
        if error.name == "cvsrffi.phase1_mirage.head":
            pytest.fail("missing MIRAGE open-world geometry head module")
        raise
    return (
        module.DEFER_LABEL,
        module.UNKNOWN_LABEL,
        module.DecisionThresholds,
        module.MIRAGEOpenHead,
        module.OpenHeadOutput,
        module.decide,
    )


def _normalized(rows: list[list[float]]) -> torch.Tensor:
    return functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=1)


def _controlled_head(MIRAGEOpenHead, *, covariance_rank: int = 0):
    """Build hand-checkable geometry for behavior tests without any truth input."""

    head = MIRAGEOpenHead(num_classes=3, feature_dim=4, covariance_rank=covariance_rank)
    with torch.no_grad():
        head.prototypes.copy_(
            torch.tensor(
                [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, -5.0, 0.0]]
            )
        )
        head.log_radius.fill_(-3.0)
        head.log_diag_precision.zero_()
        if head.low_rank_factor is not None:
            head.low_rank_factor.zero_()
        head.log_risk_weights.zero_()
        head.risk_bias.fill_(-4.0)
    return head


@pytest.mark.parametrize("covariance_rank", (0, 8))
def test_head_normalizes_geometry_masks_proxy_rows_and_returns_finite_outputs(covariance_rank: int):
    """Catch missing row normalization, invalid rank support, or proxy-row leakage."""

    _, _, _, MIRAGEOpenHead, _, _ = _head_api()
    head = _controlled_head(MIRAGEOpenHead, covariance_rank=covariance_rank)
    z_id = _normalized([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    output = head(z_id, class_mask=torch.tensor([True, False, True]))

    assert output.class_scores.shape == (3, 3)
    assert output.class_distances.shape == (3, 3)
    assert output.radius_margins.shape == (3, 3)
    assert output.energy.shape == (3,)
    assert output.unknown_risk.shape == (3,)
    assert torch.isfinite(output.class_scores).all()
    assert torch.isfinite(output.class_distances).all()
    assert torch.isfinite(output.radius_margins).all()
    assert torch.isfinite(output.energy).all()
    assert torch.isfinite(output.unknown_risk).all()
    assert torch.all(output.class_scores[:, 1] < -1e5)
    assert torch.all(output.class_distances[:, 1] > 1e5)
    assert torch.all(output.radius_margins[:, 1] > 1e5)

    expected_radius = math.log1p(math.exp(-3.0))
    expected_far_distance = 4.0 * (math.log(2.0) + 1e-4)
    assert torch.allclose(output.class_distances[0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(output.radius_margins[0, 0], torch.tensor(-expected_radius), atol=1e-6)
    assert torch.allclose(output.class_distances[1, 0], torch.tensor(expected_far_distance), atol=1e-5)
    assert output.radius_margins[1, 0] > 0.0
    assert torch.all((output.unknown_risk >= 0.0) & (output.unknown_risk <= 1.0))


def test_masked_proxy_row_cannot_change_energy_or_unknown_risk():
    """Catch a masked proxy class leaking into energy or minimum-distance risk."""

    _, _, _, MIRAGEOpenHead, _, _ = _head_api()
    head = _controlled_head(MIRAGEOpenHead)
    z_id = _normalized([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0]])
    mask = torch.tensor([True, False, True])
    before = head(z_id, class_mask=mask)

    with torch.no_grad():
        head.prototypes[1].fill_(123.0)
        head.log_radius[1] = 12.0
        head.log_diag_precision[1].fill_(12.0)
        if head.low_rank_factor is not None:
            head.low_rank_factor[1].fill_(12.0)
    after = head(z_id, class_mask=mask)

    assert torch.allclose(after.class_scores, before.class_scores)
    assert torch.allclose(after.energy, before.energy)
    assert torch.allclose(after.unknown_risk, before.unknown_risk)


def test_unknown_risk_increases_when_registered_geometry_is_farther_outside():
    """Catch a negative or missing distance/radius/energy contribution to risk."""

    _, _, _, MIRAGEOpenHead, _, _ = _head_api()
    head = MIRAGEOpenHead(num_classes=1, feature_dim=4, covariance_rank=0)
    with torch.no_grad():
        head.prototypes.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        head.log_radius.fill_(-3.0)
        head.log_diag_precision.zero_()
        head.log_risk_weights.zero_()
        head.risk_bias.zero_()

    output = head(_normalized([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    assert output.class_distances[1, 0] > output.class_distances[0, 0]
    assert output.radius_margins[1, 0] > output.radius_margins[0, 0]
    assert output.energy[1] > output.energy[0]
    assert output.unknown_risk[1] > output.unknown_risk[0]


def test_class_row_permutation_only_permutates_class_outputs_and_label_mapping():
    """Catch class-index dependence in energy, risk, or registered predictions."""

    DEFER_LABEL, _, DecisionThresholds, MIRAGEOpenHead, _, decide = _head_api()
    head = _controlled_head(MIRAGEOpenHead, covariance_rank=8)
    with torch.no_grad():
        head.log_radius.fill_(10.0)
        head.log_diag_precision.copy_(
            torch.tensor(
                [[-1.0, -0.5, 0.0, 0.5], [0.5, 0.0, -0.5, -1.0], [0.0, 0.5, -1.0, -0.5]]
            )
        )
        head.low_rank_factor.copy_(torch.arange(96, dtype=torch.float32).reshape(3, 4, 8) / 1_000.0)
    permutation = torch.tensor([2, 0, 1])
    permuted_head = copy.deepcopy(head)
    with torch.no_grad():
        permuted_head.prototypes.copy_(head.prototypes[permutation])
        permuted_head.log_radius.copy_(head.log_radius[permutation])
        permuted_head.log_diag_precision.copy_(head.log_diag_precision[permutation])
        permuted_head.low_rank_factor.copy_(head.low_rank_factor[permutation])

    z_id = _normalized(
        [[1.0, 0.2, 0.1, 0.1], [0.1, 1.0, -0.3, 0.2], [-0.2, 0.1, -1.0, 0.3], [0.2, 0.2, 0.1, 1.0]]
    )
    original = head(z_id)
    permuted = permuted_head(z_id)

    assert torch.allclose(permuted.class_scores, original.class_scores[:, permutation])
    assert torch.allclose(permuted.class_distances, original.class_distances[:, permutation])
    assert torch.allclose(permuted.radius_margins, original.radius_margins[:, permutation])
    assert torch.allclose(permuted.energy, original.energy)
    assert torch.allclose(permuted.unknown_risk, original.unknown_risk)

    thresholds = DecisionThresholds(tau_q=0.0, tau_reg=1.0, tau_unk=1.0)
    original_decision = decide(original, quality=torch.ones(z_id.shape[0]), thresholds=thresholds)
    permuted_decision = decide(permuted, quality=torch.ones(z_id.shape[0]), thresholds=thresholds)
    assert not bool(original_decision.deferred.any())
    assert not bool(permuted_decision.deferred.any())
    assert not bool((original_decision.labels == DEFER_LABEL).any())
    assert torch.equal(permutation[permuted_decision.labels], original_decision.labels)


def test_decision_uses_quality_then_registered_unknown_and_defer_states():
    """Catch treating defer as unknown or ignoring geometric eligibility rules."""

    DEFER_LABEL, UNKNOWN_LABEL, DecisionThresholds, _, OpenHeadOutput, decide = _head_api()
    output = OpenHeadOutput(
        class_scores=torch.tensor(
            [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]
        ),
        class_distances=torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 0.0]]
        ),
        radius_margins=torch.tensor(
            [[-0.1, 0.2], [-0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, -0.1]]
        ),
        energy=torch.zeros(5),
        unknown_risk=torch.tensor([0.1, 0.9, 0.9, 0.5, 0.1]),
    )
    thresholds = DecisionThresholds(tau_q=0.5, tau_reg=0.2, tau_unk=0.8)

    decision = decide(output, quality=torch.tensor([0.9, 0.1, 0.9, 0.9, 0.9]), thresholds=thresholds)

    assert torch.equal(decision.labels, torch.tensor([0, DEFER_LABEL, UNKNOWN_LABEL, DEFER_LABEL, DEFER_LABEL]))
    assert torch.equal(decision.registered, torch.tensor([True, False, False, False, False]))
    assert torch.equal(decision.explicit_unknown, torch.tensor([False, False, True, False, False]))
    assert torch.equal(decision.deferred, torch.tensor([False, True, False, True, True]))


def test_head_and_decision_reject_malformed_embeddings_masks_thresholds_quality_and_risk():
    """Catch invalid geometry/decision inputs before a prediction is emitted."""

    _, _, DecisionThresholds, MIRAGEOpenHead, OpenHeadOutput, decide = _head_api()
    head = MIRAGEOpenHead(num_classes=2, feature_dim=4)
    normalized = _normalized([[1.0, 0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match=r"\[B, D\]"):
        head(normalized.unsqueeze(0))
    with pytest.raises(ValueError, match="normalized"):
        head(torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        head(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="class_mask"):
        head(normalized, class_mask=torch.tensor([True]))
    with pytest.raises(ValueError, match="at least one"):
        head(normalized, class_mask=torch.tensor([False, False]))
    with pytest.raises(ValueError, match="tau_reg"):
        DecisionThresholds(tau_q=0.5, tau_reg=0.9, tau_unk=0.1)
    with pytest.raises(ValueError, match="finite"):
        DecisionThresholds(tau_q=float("nan"), tau_reg=0.2, tau_unk=0.8)

    malformed_output = OpenHeadOutput(
        class_scores=torch.tensor([[0.9, 0.1]]),
        class_distances=torch.tensor([[0.0, 1.0]]),
        radius_margins=torch.tensor([[-0.1, 0.1]]),
        energy=torch.tensor([0.0]),
        unknown_risk=torch.tensor([float("nan")]),
    )
    with pytest.raises(ValueError, match="unknown_risk"):
        decide(malformed_output, quality=torch.tensor([0.9]), thresholds=DecisionThresholds(0.5, 0.2, 0.8))
    with pytest.raises(ValueError, match="quality"):
        decide(
            OpenHeadOutput(
                class_scores=torch.tensor([[0.9, 0.1]]),
                class_distances=torch.tensor([[0.0, 1.0]]),
                radius_margins=torch.tensor([[-0.1, 0.1]]),
                energy=torch.tensor([0.0]),
                unknown_risk=torch.tensor([0.1]),
            ),
            quality=torch.tensor([1.1]),
            thresholds=DecisionThresholds(0.5, 0.2, 0.8),
        )


def test_head_accepts_task4_normalized_identity_embeddings():
    """Catch a head boundary that rejects the frozen Task 4 ``z_id`` output."""

    _, _, _, MIRAGEOpenHead, _, _ = _head_api()
    from cvsrffi.phase1_mirage.model import MIRAGEEncoder

    z_id = MIRAGEEncoder()(torch.randn(2, 2, 256)).z_id
    output = MIRAGEOpenHead(num_classes=3, feature_dim=160, covariance_rank=8)(z_id)

    assert output.class_scores.shape == (2, 3)
    assert torch.isfinite(output.unknown_risk).all()
