import torch
from experiments.pamf_phase3_restoration.response_epoch10_diagnostic import guard_flags


def test_distinguishes_nonfinite_loss_and_oversized_residual():
    assert guard_flags(torch.tensor(1.),torch.tensor([1e-10,1e-9])) == []
    assert guard_flags(torch.tensor(float('nan')),torch.tensor([1e-9])) == ['NONFINITE_LOSS']
    assert guard_flags(torch.tensor(1.),torch.tensor([1e-9,1.1e-8])) == ['POSTERIOR_RESIDUAL_EXCEEDS_LIMIT']
    assert guard_flags(torch.tensor(float('inf')),torch.tensor([2e-8])) == ['NONFINITE_LOSS','POSTERIOR_RESIDUAL_EXCEEDS_LIMIT']


def test_nonfinite_residual_is_reported_explicitly():
    assert guard_flags(torch.tensor(1.),torch.tensor([float('nan')])) == ['NONFINITE_RESIDUAL']
