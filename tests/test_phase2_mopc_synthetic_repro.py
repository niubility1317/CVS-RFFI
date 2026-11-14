import torch
from experiments.pamf_phase2_practical.group2_mopc_synthetic_diagnostic import finite_absmax


def test_synthetic_statistics_report_nonfinite_without_serializing_nan():
    assert finite_absmax(None) is None
    assert finite_absmax(torch.tensor([1.,-3.]))==3.
    assert finite_absmax(torch.tensor([float('inf')])) is None
