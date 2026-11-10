import pytest
import torch
from torch.nn.functional import normalize

from experiments.pamf_physics_tri.pairing import BRANCHES, BranchPairedREPL, branch_partners
from experiments.pamf_ssl_research.u_mixstyle_objective import partner_eligibility


def test_echo_discount_can_reorder_only_eligible_cross_receiver_neighbors():
    z = torch.tensor([[1., 0., 0.], [.7, 0., .7141428], [.6, .8, 0.], [1., 0., 0.],
                      [1., 0., 0.], [1., 0., 0.]])
    pseudo = torch.tensor([0, 0, 0, 0, 0, 1])
    rx = torch.tensor([1, 3, 4, 1, 3, 4])
    accepted = torch.tensor([True, True, True, True, False, True])
    eligible = partner_eligibility(pseudo, accepted, rx)
    assert eligible[0].tolist() == [False, True, True, False, False, False]
    selected = eligible.any(1)
    assert selected.tolist() == [True, True, True, True, False, False]
    directions = torch.zeros_like(z)
    directions[[0, 2]] = normalize((z[0] - z[2])[None, :])
    branch = {name: z[selected] for name in BRANCHES}
    tangent = {name: directions[selected] for name in BRANCHES}
    valid_pairs = eligible[selected][:, selected]
    ordinary, changed, discount = branch_partners(branch, tangent, valid_pairs, 0.)
    assert ordinary[0].item() == 1 and changed == 0 and discount == 0.
    physical, changed, discount = branch_partners(branch, tangent, valid_pairs, .5)
    assert physical[0].item() == 2 and changed > 0 and discount > 0.
    assert valid_pairs[torch.arange(len(physical)), physical].all()


def test_u_plan_fails_if_it_changes_eligibility_or_reads_visible_labels():
    objective = BranchPairedREPL.__new__(BranchPairedREPL)
    labels = torch.tensor([-1, -1])
    accepted = torch.tensor([True, True])
    objective.validate_unlabeled_batch((None, labels, None, {'tx_label_visible': ~accepted}))
    with pytest.raises(ValueError, match='hidden-label'):
        objective.validate_unlabeled_batch((None, labels, None, {'tx_label_visible': accepted}))
    with pytest.raises(ValueError, match='U truth exposed'):
        objective.validate_unlabeled_batch((None, labels + 1, None, {'tx_label_visible': ~accepted}))
    partner = torch.tensor([1, 0])
    objective.pair_plan = partner, accepted
    with pytest.raises(ValueError, match='eligibility differs'):
        objective.select_style_partners(partner, ~accepted)
