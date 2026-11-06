import pytest
import torch
from experiments.phase1_dura_orthogonal.gradient_alignment_source_probe import alignment


def test_alignment_shared_parameters_and_sign():
    a = (torch.tensor([1., 2.]), None, torch.tensor([3.]))
    b = (torch.tensor([-1., -2.]), torch.tensor([9.]), None)
    result = alignment(a, b)
    assert result['shared_tensors'] == 1
    assert result['cosine'] == pytest.approx(-1.)
    assert result['iq_norm'] == pytest.approx(5 ** .5)
    assert result['action_norm'] == pytest.approx(5 ** .5)


def test_alignment_rejects_no_shared_or_zero_gradients():
    with pytest.raises(ValueError):
        alignment((None,), (torch.ones(1),))
    with pytest.raises(ValueError):
        alignment((torch.zeros(1),), (torch.ones(1),))
