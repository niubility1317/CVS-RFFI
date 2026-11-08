import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.signal_balanced_mix_loss import receiver_weights, balanced_oracle_loss


def test_receiver_weights_give_equal_total_mass():
    rx=np.array([1,1,1,3,4,4,6,8,8,8,8])
    w=receiver_weights(rx).numpy()
    totals=[w[rx==i].sum() for i in (1,3,4,6,8)]
    np.testing.assert_allclose(totals,np.full(5,len(rx)/5),rtol=1e-6)
    with pytest.raises(ValueError):
        receiver_weights(np.array([1,3,4]))


def test_oracle_loss_prefers_fitting_clean_and_has_finite_gradients():
    anchor=torch.zeros(5,2,4)
    proposal=torch.ones_like(anchor)
    clean=torch.full_like(anchor,.3)
    weights=torch.ones(5)
    alpha=torch.full((5,1,1),.3,requires_grad=True)
    aligned=balanced_oracle_loss(anchor+alpha*(proposal-anchor),anchor,proposal,clean,weights)
    bad=balanced_oracle_loss(anchor+.9*(proposal-anchor),anchor,proposal,clean,weights)
    assert aligned < bad
    aligned.backward()
    assert torch.isfinite(alpha.grad).all()
    same=balanced_oracle_loss(anchor,anchor,anchor,clean,weights)
    assert torch.isfinite(same)
