import math
import pytest
import torch
from experiments.pamf_phase3_restoration.projected_spike_trace import tensor_stats


def test_tensor_statistics_and_nonfinite_rejection():
    value=tensor_stats(torch.tensor([3.,4.]))
    assert value['norm']==5. and value['rms']==pytest.approx(math.sqrt(12.5)) and value['max_abs']==4.
    with pytest.raises(ValueError):tensor_stats(torch.tensor([float('inf')]))
