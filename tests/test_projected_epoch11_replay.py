import pytest
import torch
from experiments.pamf_phase3_restoration.projected_epoch11_replay import spike_indices


def test_projected_spike_threshold_and_invalid_values():
    assert spike_indices(torch.tensor([0.,100.,100.01,500.])).tolist()==[2,3]
    with pytest.raises(ValueError):spike_indices(torch.tensor([float('nan')]))
    with pytest.raises(ValueError):spike_indices(torch.tensor([-1.]))
