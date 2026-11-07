import pytest
import torch

from experiments.phase1_dura_orthogonal.large_source_tail import tail_iq_penalty


def test_loss_adds_worst_quarter_of_normalized_iq_error():
    clean=torch.ones(8,2,256)
    output=(clean+torch.arange(8,dtype=torch.float32)[:,None,None]).requires_grad_()
    loss=tail_iq_penalty(output,clean)
    assert float(loss.detach())==pytest.approx(.5*(36+49)/2)
    loss.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


def test_zero_error_adds_no_loss_or_gradient():
    clean=torch.ones(8,2,256)
    output=clean.clone().requires_grad_()
    loss=tail_iq_penalty(output,clean)
    loss.backward()
    assert loss.item()==0 and torch.count_nonzero(output.grad)==0


def test_odd_batch_uses_ceiling_for_tail_count():
    clean=torch.ones(5,2,256)
    output=clean+torch.arange(5,dtype=torch.float32)[:,None,None]
    extra=tail_iq_penalty(output,clean)
    assert extra.item()==pytest.approx(.5*(9+16)/2)
