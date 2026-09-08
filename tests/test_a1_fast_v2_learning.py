"""Source-only synthetic checks; all models start from fresh initialization."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from SSDG import train_ssdg as train
from model import CosFaceHead
from cvsrffi.daot_training import compute_daot_batch_objective
from cvsrffi.orbit_teacher import EMALossScaleNormalizer


def objective(coverage, flag, normalizer, shift=0.0):
    logits = (torch.tensor([[.1, .3, -.2]]).repeat(4, 1) +
              torch.tensor([shift, 0., 0.])).requires_grad_()
    features = torch.tensor([[1., .2, .3]]).repeat(4, 1).requires_grad_()
    teacher_logits = torch.tensor([[2., 0., 0.]]).repeat(4, 1).requires_grad_()
    other_logits = teacher_logits.detach().clone()
    other_logits[coverage:] = torch.tensor([0., 2., 0.])
    other_logits.requires_grad_()
    teacher_features = torch.tensor([[.8, .1, .5]]).repeat(4, 1).requires_grad_()
    output = compute_daot_batch_objective(
        student_clean={"z_id": features, "tx_logits": logits},
        student_channel={"z_id": features, "tx_logits": logits},
        teacher_views=[{"z_id": teacher_features, "tx_logits": teacher_logits},
                       {"z_id": teacher_features, "tx_logits": other_logits}],
        reliability=torch.ones(4, 2), importance=torch.ones(4, 2),
        recoverability=torch.ones(4), orbit_scale=.7, tangent_scale=0.,
        weights={"orbit_logit": 1.}, coverage_floor=.15, huber_beta_min=.3,
        temperature=1., loss_normalizer=normalizer, logit_coverage_weighting=flag,
    )
    return output, logits, teacher_logits, other_logits, teacher_features


@pytest.mark.parametrize("coverage", [1, 2, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_coverage_applied_after_normalization_scales_loss_and_gradient(coverage, batched):
    normalizers = [EMALossScaleNormalizer(batched_readback=batched) for _ in range(2)]
    for shift in (0., .4):
        baseline, student_base, *_ = objective(coverage, False, normalizers[0], shift)
        weighted, student_weighted, *teachers = objective(coverage, True, normalizers[1], shift)
        expected_q = (coverage / 4.) * F.softmax(torch.tensor([2., 0., 0.]), dim=0).max()
        q = weighted["diagnostics"]["effective_logit_coverage"]
        torch.testing.assert_close(q, expected_q)
        assert not q.requires_grad
        assert normalizers[0].state_dict() == normalizers[1].state_dict()
        assert torch.equal(baseline["components"]["orbit_logit"], weighted["components"]["orbit_logit"])
        assert torch.equal(baseline["normalized_components"]["orbit_logit"],
                           weighted["normalized_components"]["orbit_logit"])
        torch.testing.assert_close(weighted["loss"], baseline["loss"] * expected_q)
        torch.testing.assert_close(weighted["diagnostics"]["weighted_logit_contribution"], weighted["loss"].detach())
        baseline["loss"].backward()
        weighted["loss"].backward()
        assert student_base.grad.norm() > 0
        torch.testing.assert_close(student_weighted.grad, student_base.grad * expected_q)
        assert all(teacher.grad is None for teacher in teachers)


def test_no_consensus_has_zero_finite_loss_and_gradient():
    result, logits, *_ = objective(0, True, EMALossScaleNormalizer())
    assert result["diagnostics"]["effective_logit_coverage"].item() == 0
    assert result["loss"].item() == 0
    result["loss"].backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_versioned_ema_refreshes_real_cosface_cache_and_copies_buffers():
    teacher = CosFaceHead(3, 2).eval()
    student = CosFaceHead(3, 2).eval()
    teacher.register_buffer("counter", torch.tensor(1))
    student.register_buffer("counter", torch.tensor(9))
    with torch.no_grad():
        teacher.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        student.weight.copy_(torch.tensor([[0., 1., 0.], [0., 0., 1.]]))
        x = torch.tensor([[1., 2., 3.], [3., 1., 2.]])
        before = teacher(x).clone()
        old_version = teacher.weight._version
        old_cache_key = teacher._norm_weight_cache_key
        original_weight = teacher.weight.clone()
        train._update_ema_model(teacher, student, .6, versioned=True)
        expected_weight = original_weight.mul(.6).add(student.weight, alpha=.4)
        assert torch.equal(teacher.weight, expected_weight)
        assert teacher.weight._version > old_version
        actual = teacher(x)
        assert teacher._norm_weight_cache_key != old_cache_key
        assert not torch.equal(before, actual)
        teacher._norm_weight_cache_key = None
        forced_refresh = teacher(x)
        assert torch.equal(actual, forced_refresh)
        assert teacher.counter.item() == 9


def test_startup_ema_first_update_is_student_then_successful_step_average():
    # The initial teacher is an in-memory copy of this run's random student.
    student = torch.nn.Linear(1, 1, bias=False)
    teacher = deepcopy(student)
    successful = 0
    seen = []
    for value, applied in [(2., True), (99., False), (4., True), (9., True)]:
        with torch.no_grad():
            student.weight.fill_(value)
        if applied:
            successful += 1
            decay = train._effective_a1_ema_decay(.99, successful, True)
            assert decay == (successful - 1) / successful
            train._update_ema_model(teacher, student, decay, versioned=True)
            seen.append(value)
        torch.testing.assert_close(teacher.weight, torch.tensor([[sum(seen) / len(seen)]]))
    assert successful == 3
    assert train._effective_a1_ema_decay(.5, 10, True) == .5
    assert train._effective_a1_ema_decay(.9, 0, False) == .9
    with pytest.raises(ValueError, match="positive"):
        train._effective_a1_ema_decay(.9, 0, True)
    with pytest.raises(ValueError, match="decay"):
        train._effective_a1_ema_decay(1.1, 1, True)
