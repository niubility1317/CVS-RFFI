import pytest
import torch
from experiments.phase1_dura_orthogonal.variance_separability_source_probe import pair_auc


def test_auc_handles_order_and_ties_without_label_inversion():
    assert pair_auc(torch.tensor([0., 1.]), torch.tensor([1., 2.])) == pytest.approx(.875)
    assert pair_auc(torch.tensor([1., 2.]), torch.tensor([0., 1.])) == pytest.approx(.125)
    assert pair_auc(torch.tensor([1.]), torch.tensor([1.])) == pytest.approx(.5)


def test_auc_requires_nonempty_finite_source_groups():
    with pytest.raises(ValueError):
        pair_auc(torch.tensor([]), torch.tensor([1.]))
    with pytest.raises(ValueError):
        pair_auc(torch.tensor([float('nan')]), torch.tensor([1.]))
