"""Tensor-only checks for protected160 prototypes and soft source geometry."""
import inspect

import pytest
import torch
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_protected_open_loss as loss


def support_fixture():
    z = torch.zeros(24, 160)
    z[torch.arange(24), torch.arange(24)] = 1.
    return (z, torch.arange(6).repeat_interleave(4), torch.tensor([1, 3, 2, 4] * 6),
            torch.tensor([3, 1, 2, 3] * 6))


def neighbor_fixture():
    teacher = torch.zeros(6, 160, dtype=torch.float64)
    teacher[:, :3] = torch.tensor([[1., 0., 0.], [.8, .6, 0.], [.6, .8, 0.],
                                  [0., .8, .6], [.3, .4, .8], [.2, -.5, .7]], dtype=torch.float64)
    student = teacher.clone()
    student[0, :3] += torch.tensor([-.3, .6, .2])
    student[2, :3] += torch.tensor([.2, -.4, .5])
    rx = torch.tensor([1, 3, 3, 8, 8, 8])
    day = torch.tensor([1, 2, 2, 3, 3, 3])
    ids = torch.arange(6)
    return student, teacher, rx, day, ids


def manual_neighbor(student, teacher, rx, day, ids):
    s, t = F.normalize(student, dim=1), F.normalize(teacher.detach(), dim=1)
    terms = []
    keys = list(zip(rx.tolist(), day.tolist()))
    for i in range(len(s)):
        eligible = [j for j in range(len(s)) if rx[i] != rx[j] and day[i] != day[j] and ids[i] != ids[j]]
        weight = torch.tensor([1. / sum(keys[k] == keys[j] for k in eligible) for j in eligible], dtype=s.dtype)
        a = (t[i] @ t[eligible].T) / .1 + weight.log()
        b = (s[i] @ s[eligible].T) / .1 + weight.log()
        terms.append((a.softmax(0) * (a.log_softmax(0) - b.log_softmax(0))).sum())
    terms = torch.stack(terms)
    grouped = torch.stack([terms[torch.tensor([key == group for key in keys])].mean() for group in sorted(set(keys))])
    return grouped.mean(), terms.mean()


def test_query_prototypes_exclude_same_RX_OR_day_and_preserve_support_gradients():
    z, y, rx, day = support_fixture()
    z.requires_grad_()
    p = loss.query_prototypes(z, y, rx, day, torch.tensor([1]), torch.tensor([1]))
    assert p.shape == (1, 6, 160)
    expected = F.normalize(z.detach().reshape(6, 4, 160)[:, 2:].mean(1), dim=1)
    assert torch.equal(p[0], expected)
    p.sum().backward()
    assert torch.count_nonzero(z.grad.reshape(6, 4, 160)[:, :2]) == 0
    assert z.grad.reshape(6, 4, 160)[:, 2:].abs().sum() > 0


def test_query_prototype_missing_class_or_cross_domain_support_fails():
    z, y, rx, day = support_fixture()
    for labels, sr, sd in ((y.clamp_max(4), rx, day), (y, torch.ones_like(rx), day),
                           (y, rx, torch.ones_like(day))):
        with pytest.raises(ValueError):
            loss.query_prototypes(z, labels, sr, sd, torch.tensor([1]), torch.tensor([1]))
    with pytest.raises(ValueError):
        loss.query_prototypes(z, y.float(), rx, day, torch.tensor([1]), torch.tensor([1]))


def test_membership_balances_L_groups_selects_only_registered_slots_and_OE_uses_all_six():
    relation = torch.full((4, 6), 1000.)
    relation[:, 0] = torch.tensor([2., -1., 0., 1.])
    relation[:, 1] = relation[:, 0] - 2.
    relation.requires_grad_()
    result = loss.member_loss(relation, torch.tensor([0, 0, 1, 3]), torch.tensor([0, 1]))
    expected = .5 * (F.softplus(-relation[:3, 0]).mean() + F.softplus(relation[3:, 0]).mean())
    assert torch.allclose(result['loss'], expected)
    assert (result['known_count'], result['unknown_count'], result['selected_class_count']) == (3, 1, 2)
    result['loss'].backward()
    assert torch.count_nonzero(relation.grad[:, 2:]) == 0
    assert relation.grad[:, :2].abs().sum() > 0
    oe = torch.arange(18, dtype=torch.float32).reshape(3, 6).requires_grad_()
    rejected = loss.oe_member_loss(oe)
    assert torch.equal(rejected['loss'], F.softplus(oe.max(1).values).mean())
    assert rejected['count'] == 3
    assert list(inspect.signature(loss.oe_member_loss).parameters) == ['oe_relation']


def test_member_requires_2_to_5_unique_old_slots_and_both_simulated_groups():
    relation = torch.zeros(4, 6)
    for labels, selected in ((torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1])),
                             (torch.tensor([3, 3, 3, 3]), torch.tensor([0, 1])),
                             (torch.tensor([0, 1, 2, -1]), torch.tensor([0, 1])),
                             (torch.tensor([0, 1, 2, 3]), torch.tensor([0])),
                             (torch.tensor([0, 1, 2, 3]), torch.arange(6)),
                             (torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0]))):
        with pytest.raises(ValueError):
            loss.member_loss(relation, labels, selected)


def test_soft_neighbor_equal_geometry_zero_KL_teacher_detached_and_student_perturbation_has_grad():
    s, teacher, rx, day, ids = neighbor_fixture()
    teacher.requires_grad_()
    equal = teacher.detach().clone().requires_grad_()
    zero = loss.soft_neighbor_loss(equal, teacher, rx, day, ids)
    assert zero['loss'].abs() < 1e-12
    s.requires_grad_()
    result = loss.soft_neighbor_loss(s, teacher, rx, day, ids)
    result['loss'].backward()
    assert result['loss'] > 0 and s.grad.abs().sum() > 0 and torch.isfinite(s.grad).all()
    assert teacher.grad is None
    assert result['candidate_domain_count_min'] == 2


def test_soft_neighbor_cell_weights_match_manual_KL_and_row_domain_permutations():
    s, teacher, rx, day, ids = neighbor_fixture()
    expected, sample_weighted = manual_neighbor(s, teacher, rx, day, ids)
    actual = loss.soft_neighbor_loss(s, teacher, rx, day, ids)['loss']
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(expected, sample_weighted, atol=1e-6, rtol=1e-6)
    order = torch.tensor([5, 2, 0, 4, 1, 3])
    row = loss.soft_neighbor_loss(s[order], teacher[order], rx[order], day[order], ids[order])['loss']
    relabeled_rx = torch.tensor([5, 8, 8, 3, 3, 3])
    relabeled_day = torch.tensor([9, 4, 4, 7, 7, 7])
    domain = loss.soft_neighbor_loss(s, teacher, relabeled_rx, relabeled_day, ids * 17 + 4)['loss']
    assert torch.allclose(actual, row, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual, domain, atol=1e-12, rtol=1e-12)


def test_soft_neighbor_duplicate_ID_mask_and_insufficient_candidate_cells():
    s, teacher, rx, day, ids = neighbor_fixture()
    ids[3] = ids[1]  # Cross-domain copies must be excluded without emptying a candidate cell.
    actual = loss.soft_neighbor_loss(s, teacher, rx, day, ids)
    expected, _ = manual_neighbor(s, teacher, rx, day, ids)
    assert torch.allclose(actual['loss'], expected, atol=1e-12, rtol=1e-12)
    for bad_rx, bad_day, bad_ids in ((torch.ones_like(rx), day, ids),
                                    (rx, torch.ones_like(day), ids),
                                    (rx, day, torch.zeros_like(ids))):
        with pytest.raises(ValueError):
            loss.soft_neighbor_loss(s, teacher, bad_rx, bad_day, bad_ids)
    with pytest.raises(ValueError):
        loss.soft_neighbor_loss(s[:3], teacher[:3], rx[:3], day[:3], torch.arange(3))


def test_finite_and_zero_vectors_are_explicit_without_NaN():
    support = torch.zeros(12, 160)
    y = torch.arange(6).repeat_interleave(2)
    rx, day = torch.tensor([2, 3] * 6), torch.tensor([2, 3] * 6)
    p = loss.query_prototypes(support, y, rx, day, torch.tensor([1]), torch.tensor([1]))
    assert torch.equal(p, torch.zeros_like(p))
    z = torch.zeros(3, 160, requires_grad=True)
    t = torch.zeros_like(z, requires_grad=True)
    result = loss.soft_neighbor_loss(z, t, torch.arange(3), torch.arange(3), torch.arange(3))
    assert result['loss'] == 0 and torch.isfinite(result['loss'])
    result['loss'].backward()
    assert torch.isfinite(z.grad).all() and t.grad is None
    bad = z.detach().clone()
    bad[0, 0] = float('nan')
    with pytest.raises(ValueError):
        loss.soft_neighbor_loss(bad, t, torch.arange(3), torch.arange(3), torch.arange(3))
    with pytest.raises(ValueError):
        loss.oe_member_loss(torch.full((2, 6), float('inf')))
