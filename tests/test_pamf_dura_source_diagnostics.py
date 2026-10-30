import numpy as np
import pytest
import torch

from experiments.pamf_dura_diagnostics.runner import action_from_arrays, source_indices, waveform_metrics
from experiments.pamf_uncertainty_restore.models import weighted_map


def test_source_roles_and_deterministic_position_sampling():
    roles = np.array(['L_s']*6300+['U_s']*56700)
    selected = source_indices(roles)
    assert len(np.unique(selected)) == 512
    assert (roles[selected] == 'L_s').sum() == 128
    assert np.array_equal(selected, source_indices(roles))
    with pytest.raises(ValueError):
        source_indices(np.array(['V']*63000))
    with pytest.raises(ValueError):
        source_indices(roles[:-1])


def test_oracle_action_and_map_identity_matches_analytic_solution():
    kernel = torch.zeros(3, 2, 64)
    kernel[:, 0, 0] = 1
    coefficients = torch.zeros(3, 4, 256)
    coefficients[:, 0] = 1
    action = action_from_arrays(kernel, coefficients)
    torch.manual_seed(7)
    observed = torch.randn(3, 256, dtype=torch.complex64)
    prior = torch.randn_like(observed)
    variance = torch.tensor([.001, .01, .1])
    tau2 = .02
    result = weighted_map(action, observed, prior, variance, prior_variance=tau2, steps=2)
    expected = (tau2*observed+variance[:, None]*prior)/(tau2+variance[:, None])
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)


def test_diagnostic_metrics_handle_weak_bins_without_nonfinite_weights():
    clean = torch.zeros(2, 2, 256)
    estimate = clean.clone()
    estimate[..., -1] = 1
    metrics = waveform_metrics(estimate, clean)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert torch.equal(metrics['prefix232_nmse'], torch.zeros(2))
    assert (metrics['tail24_nmse'] > 0).all()
