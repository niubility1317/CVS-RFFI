import pytest
import torch

from cvsrffi.pair_reform import (
    asymmetric_pair_loss, asymmetric_teacher_target, classification_confidence,
    cosine_safety_radius, physical_reliability, point_pair_loss, safe_pair_loss,
    unified_soft_ce,
)


def test_safety_radius_geometry_and_detachment():
    teacher = torch.tensor([[1., 0.]], requires_grad=True)
    weights = torch.tensor([[3., 0.], [0., 7.]], requires_grad=True)
    radius, valid = cosine_safety_radius(teacher, weights, torch.tensor([0]))
    torch.testing.assert_close(radius, torch.tensor([.5 / 2 ** .5]))
    assert valid.tolist() == [True] and not radius.requires_grad
    student = torch.tensor([[0., 1.]], requires_grad=True)
    loss = safe_pair_loss(student, teacher, weights, torch.tensor([0]), torch.ones(1))
    torch.testing.assert_close(loss, torch.tensor(2. - .125))
    loss.backward()
    assert student.grad is not None and teacher.grad is None and weights.grad is None


def test_inside_radius_keeps_class_and_has_zero_loss():
    weights = torch.eye(2)
    teacher = torch.tensor([[1., 0.]])
    student = torch.tensor([[.98, .2]], requires_grad=True)
    assert (student @ weights.T).argmax(1).item() == 0
    assert safe_pair_loss(student, teacher, weights, torch.tensor([0]), torch.ones(1)) == 0


def test_invalid_anchors_and_heads_are_excluded():
    weights = torch.eye(2)
    anchors = torch.tensor([[0., 1.], [0., 0.], [float('nan'), 1.], [1., 1.]])
    radius, valid = cosine_safety_radius(anchors, weights, torch.zeros(4, dtype=torch.long))
    assert not valid.any() and torch.equal(radius, torch.zeros(4))
    _, valid = cosine_safety_radius(torch.tensor([[1., 0.]]), torch.ones(2, 2), torch.tensor([0]))
    assert not valid.any()
    for kwargs in ({'head_kind': 'mlp'}, {'bias': torch.zeros(2)}):
        with pytest.raises(ValueError):
            cosine_safety_radius(anchors, weights, torch.zeros(4, dtype=torch.long), **kwargs)
    loss = safe_pair_loss(torch.ones(4, 2), anchors, weights, torch.zeros(4, dtype=torch.long), torch.ones(4))
    assert torch.isfinite(loss) and loss == 0


def test_uniform_classification_retains_physical_feature_learning():
    probabilities = torch.full((2, 3), 1 / 3)
    q = classification_confidence(probabilities)
    assert torch.equal(q, torch.zeros(2))
    phys, valid = physical_reliability(torch.ones(2), torch.ones(2, dtype=torch.bool))
    student = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    teacher = torch.tensor([[1., 0.], [1., 0.]])
    assert point_pair_loss(student, teacher, phys, tolerance=.2) > 0
    assert unified_soft_ce(torch.zeros(2, 3), probabilities, phys, q) == 0
    assert valid.all() and torch.equal(phys, torch.ones(2))


def test_fixed_batch_denominator_and_teacher_stop_gradient():
    teacher = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    student = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    full = point_pair_loss(student, teacher, torch.ones(2))
    half = point_pair_loss(student, teacher, torch.tensor([1., 0.]))
    torch.testing.assert_close(half, full / 2)
    half.backward()
    assert teacher.grad is None
    logits = torch.zeros(2, 2, requires_grad=True)
    probs = torch.tensor([[.9, .1], [.8, .2]], requires_grad=True)
    full = unified_soft_ce(logits, probs, torch.ones(2), torch.ones(2))
    weak = unified_soft_ce(logits, probs, torch.ones(2), torch.full((2,), .1))
    torch.testing.assert_close(weak, full * .1)
    weak.backward()
    assert probs.grad is None


def test_unknown_quality_is_explicit_and_cache_cannot_enter():
    values = torch.tensor([.05, float('nan'), 1.])
    valid = torch.tensor([True, False, False])
    phys, known = physical_reliability(values, valid)
    torch.testing.assert_close(phys, torch.tensor([.05, .5, .5]))
    assert known.tolist() == [True, False, False]
    phys, _ = physical_reliability(values, valid, unknown_policy='zero')
    torch.testing.assert_close(phys, torch.tensor([.05, 0., 0.]))


def test_asymmetric_target_is_clean_dominant_and_detached():
    clean = torch.tensor([[1., 0.]], requires_grad=True)
    leo = torch.tensor([[0., 1.]], requires_grad=True)
    target = asymmetric_teacher_target(clean, leo, leo_mix=.25)
    assert target[0, 0] > target[0, 1] and not target.requires_grad
    torch.testing.assert_close(asymmetric_teacher_target(clean), clean.detach())
    with pytest.raises(ValueError):
        asymmetric_teacher_target(clean, leo, leo_mix=.6)
    student = leo.detach().clone().requires_grad_()
    asymmetric_pair_loss(student, clean, torch.ones(1), leo_teacher=leo, leo_mix=.25).backward()
    assert student.grad is not None and clean.grad is None and leo.grad is None


def test_js_strength_is_configurable_and_not_fourth_root():
    p = torch.tensor([[.99, .01]])
    other = p.flip(1)
    base = classification_confidence(p, other, js_scale=0.)
    attenuated = classification_confidence(p, other, js_scale=4.)
    assert attenuated < base * .2


def test_safety_uses_fixed_batch_denominator_and_scale_invariant_planes():
    teacher = torch.tensor([[1., 0.], [0., 1.]])
    student = teacher.flip(1)
    labels = torch.zeros(2, dtype=torch.long)
    radius, valid = cosine_safety_radius(teacher, torch.eye(2), labels)
    scaled_radius, scaled_valid = cosine_safety_radius(teacher, torch.diag(torch.tensor([100., .01])), labels)
    torch.testing.assert_close(radius, scaled_radius)
    assert torch.equal(valid, scaled_valid)
    both = safe_pair_loss(student, teacher, torch.eye(2), labels, torch.ones(2))
    one = safe_pair_loss(student[:1], teacher[:1], torch.eye(2), labels[:1], torch.ones(1))
    torch.testing.assert_close(both, one / 2)


def test_classification_target_requires_real_normalized_distribution():
    with pytest.raises(ValueError):
        unified_soft_ce(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError):
        asymmetric_teacher_target(torch.ones(1, 2), leo_mix=.1)
