import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.ccoi_pa import (  # noqa: E402
    CCOIPASidecar,
    HeldoutChallengePredictor,
    OperatorPool,
    PAConditionalResponseHead,
    nonoverlap_holdout_masks,
    raw_intersection_count,
    raw_support_holdout_masks,
)


def test_conditional_response_matches_challenge_token_geometry():
    head = PAConditionalResponseHead(pa_channels=12, q_dim=8, response_dim=10)
    pa_map = torch.randn(3, 12, 7)
    q = torch.randn(3, 13, 8)

    conditioned = head(pa_map, q, conditioned=True)
    constant = head(pa_map, q, conditioned=False)

    assert conditioned.shape == (3, 13, 10)
    assert constant.shape == conditioned.shape
    assert not torch.equal(conditioned, constant)


def test_operator_pool_is_permutation_invariant_and_all_invalid_is_safe():
    pool = OperatorPool(response_dim=10, q_dim=8, operator_dim=6)
    response = torch.randn(2, 13, 10)
    q = torch.randn(2, 13, 8)
    mask = torch.ones(2, 13, dtype=torch.bool)
    perm = torch.randperm(13)

    out1 = pool(response, q, mask)
    out2 = pool(response[:, perm], q[:, perm], mask[:, perm])
    empty = pool(response, q, torch.zeros_like(mask))

    torch.testing.assert_close(out1.theta, out2.theta)
    torch.testing.assert_close(out1.coverage, torch.ones(2))
    torch.testing.assert_close(empty.theta, torch.zeros_like(empty.theta))
    assert torch.isfinite(empty.entropy).all()


def test_holdout_anchors_do_not_overlap_in_raw_sample_ranges():
    support, holdout = nonoverlap_holdout_masks(13, token_length=64, stride=16, fold=0)

    assert holdout.any() and support.any()
    assert raw_intersection_count(support, holdout, token_length=64, stride=16) == 0


def test_c1_constant_control_does_not_feed_real_q_to_operator():
    sidecar = CCOIPASidecar(pa_channels=4, num_classes=3, q_dim=32)
    out = sidecar(torch.randn(3, 2, 256), torch.randn(3, 4, 11), conditioned=False)

    expected = sidecar.response_head.constant_condition.view(1, 1, -1).expand_as(out["q"])
    torch.testing.assert_close(out["condition_q"], expected)
    assert not torch.equal(out["q"], out["condition_q"])


def test_raw_support_and_holdout_views_are_disjoint():
    support, holdout = raw_support_holdout_masks(
        signal_length=256,
        token_count=13,
        token_length=64,
        stride=16,
        fold=0,
    )

    assert support.any() and holdout.any()
    assert not torch.logical_and(support, holdout).any()
    assert int(support.sum() + holdout.sum()) == 256


def test_heldout_target_is_detached_from_frozen_pa_map():
    predictor = HeldoutChallengePredictor(operator_dim=6, q_dim=8, target_dim=12)
    theta = torch.randn(2, 6, requires_grad=True)
    q_holdout = torch.randn(2, 1, 8, requires_grad=True)
    frozen_target = torch.randn(2, 1, 12, requires_grad=True).detach()

    loss = (predictor(theta, q_holdout) - frozen_target).square().mean()
    loss.backward()

    assert theta.grad is not None
    assert q_holdout.grad is not None
    assert frozen_target.grad is None
