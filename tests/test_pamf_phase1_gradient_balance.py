import torch
import pytest
from experiments.pamf_phase1_flow.gradient_balance import balance_gradients


def test_conflicting_auxiliary_is_projected_and_capped():
    ref=[torch.tensor([1.,0.]),torch.tensor([0.])]
    aux=[torch.tensor([-2.,10.]),torch.tensor([0.])]
    result,meta=balance_gradients(ref,aux)
    added=[a-b for a,b in zip(result,ref)]
    assert meta['projected'] and abs(float(sum((a*b).sum() for a,b in zip(ref,added))))<1e-6
    assert float(sum((a*a).sum() for a in added).sqrt())<=1.000001


def test_zero_reference_suppresses_auxiliary():
    result,meta=balance_gradients([torch.zeros(2)],[torch.tensor([1.,3.])])
    assert torch.equal(result[0],torch.zeros(2)) and meta['scale']==0.


def test_extreme_finite_gradients_are_rejected_instead_of_nan():
    with pytest.raises(FloatingPointError):
        balance_gradients([torch.tensor([1e20])],[torch.tensor([-1e20])])


@pytest.mark.parametrize('ratio',[float('nan'),float('inf'),float('-inf')])
def test_nonfinite_ratio_is_rejected(ratio):
    with pytest.raises(ValueError):balance_gradients([torch.ones(1)],[torch.ones(1)],max_aux_ratio=ratio)
