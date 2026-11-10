import torch

from experiments.pamf_physics_tri.pairing import BRANCHES
from experiments.pamf_physics_tri.recovery import RejectedURecovery, branch_certificate


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
