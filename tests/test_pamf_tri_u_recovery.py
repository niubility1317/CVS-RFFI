import torch

from experiments.pamf_physics_tri.pairing import BRANCHES
from experiments.pamf_physics_tri.recovery import (
    RejectedURecovery, branch_certificate, candidate_classes, candidate_mass_loss,
)


def test_certificate_uses_other_receiver_and_checks_same_pseudo_class_under_echo():
    labels = torch.tensor([0, 1, 0, 1])
    labeled_rx = torch.tensor([1, 1, 3, 3])
    query_rx = torch.tensor([1, 1])
    pseudo = torch.tensor([1, 1])
    # Same-RX L would vote for class 0. The independent RX votes for class 1.
    raw = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 0.], [1., 0.]])
    echo = raw.clone()
    echo[-1] = torch.tensor([0., 1.])
    clean_paths = dict.fromkeys(BRANCHES, raw)
    echo_paths = dict.fromkeys(BRANCHES, echo)
    ordinary, stable = branch_certificate(clean_paths, echo_paths, labels, labeled_rx, query_rx, pseudo, 2, 4)
    assert ordinary.tolist() == [True, True]
    assert stable.tolist() == [True, False]
    ordinary, stable = branch_certificate(clean_paths, echo_paths, labels, torch.ones_like(labeled_rx),
                                          query_rx, pseudo, 2, 4)
    assert not ordinary.any() and not stable.any()


def test_recovery_selection_keeps_original_hard_pseudo_threshold():
    obj = RejectedURecovery.__new__(RejectedURecovery)
    obj.recipe = dict(minimum_confidence=.5)
    obj.settings = dict(threshold=.95)
    scores = torch.tensor([.98, .95, .94, .5, .49])
    assert obj.accept_pseudo(None, None, scores, None).tolist() == [True, True, False, False, False]
    assert obj.candidates.tolist() == [False, False, True, True, False]


def test_partial_class_evidence_preserves_teacher_and_excludes_unsupported_logits():
    raw = torch.tensor([[[.9, .8, .7, .1, .3, 0.]]]).expand(1, 5, 6).clone()
    echo = raw.clone()
    echo[:, :, 1] = .2
    teacher = torch.tensor([4])
    ordinary, physical = candidate_classes(raw, echo, teacher)
    assert ordinary.tolist() == [[True, True, False, False, True, False]]
    assert physical.tolist() == [[True, False, True, False, True, False]]

    logits = torch.tensor([[.9, .8, .7, .1, .3, 0.]], requires_grad=True)
    loss = candidate_mass_loss(logits, physical)
    torch.testing.assert_close(loss, -logits.softmax(1)[physical].sum().log().view(1))
    loss.sum().backward()
    assert (logits.grad[physical] < 0).all() and (logits.grad[~physical] > 0).all()
    assert torch.equal(candidate_mass_loss(logits, torch.ones_like(physical)), torch.zeros(1))
