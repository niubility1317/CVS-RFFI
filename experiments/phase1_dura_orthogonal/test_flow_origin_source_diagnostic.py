import torch

from experiments.phase1_dura_orthogonal.flow_origin_source_diagnostic import source_split, velocity_error


def test_source_partition_keeps_all_three_views_of_each_id_disjoint():
    fit, heldout = source_split(total=12, fit=5, heldout=3)
    assert torch.equal(fit, torch.tensor([0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28]))
    assert torch.equal(heldout, torch.tensor([5, 6, 7, 17, 18, 19, 29, 30, 31]))
    assert not set((fit % 12).tolist()) & set((heldout % 12).tolist())


def test_velocity_error_is_zero_for_exact_source_velocity():
    clean = torch.ones(3, 2, 256)
    anchor = torch.zeros_like(clean)
    assert torch.equal(velocity_error(clean-anchor, anchor, clean), torch.zeros(3))
