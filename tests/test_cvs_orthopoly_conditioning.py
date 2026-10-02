import pytest
import torch

from experiments.cvs_orthopoly_identity.model import packet_basis
from experiments.cvs_orthopoly_identity.conditioning import (
    inverse_transport, original_coordinates, restore_coordinates, shared_inverse_fit,
    public_conditioning_probe)


@pytest.mark.parametrize('lag', [0, 4])
def test_exact_transport_preserves_coordinate_values_and_input_jacobian(lag):
    generator = torch.Generator().manual_seed(61)
    z = torch.randn(3, 2, 256, generator=generator, dtype=torch.float64, requires_grad=True)
    basis, moments = packet_basis(z, lag, True)
    restored = restore_coordinates(basis, inverse_transport(moments))
    original = original_coordinates(z, lag)
    assert (restored - original).abs().max() < 1e-12
    coefficients = torch.randn(restored.shape, generator=generator, dtype=torch.float64)
    a = torch.autograd.grad((restored * coefficients).sum(), z, retain_graph=True)[0]
    b = torch.autograd.grad((original * coefficients).sum(), z)[0]
    assert (a - b).abs().max() < 2e-11


@pytest.mark.parametrize('lag', [0, 4])
@pytest.mark.parametrize('amplitude', [0., 1e-15, 1.])
def test_transport_degenerate_packets_remain_finite(lag, amplitude):
    z = torch.zeros(2, 2, 256, dtype=torch.float64)
    z[:, 0] = amplitude
    basis, moments = packet_basis(z, lag, True)
    result = restore_coordinates(basis, inverse_transport(moments))
    assert torch.isfinite(result).all()
    assert (result - original_coordinates(z, lag)).abs().max() < 1e-12


def test_distinct_packet_moments_require_distinct_inverse_charts():
    generator = torch.Generator().manual_seed(17)
    z = torch.randn(2, 2, 256, generator=generator, dtype=torch.float64)
    z[1] *= 2
    basis, moments = packet_basis(z, 0, True)
    transport = inverse_transport(moments)
    assert (transport[0] - transport[1]).abs().max() > .1
    result = shared_inverse_fit(basis, original_coordinates(z, 0))
    assert min(result['per_delay_relative_residual']) > .05


def test_probe_uses_public_iq_and_disposable_convolution_without_updates():
    result = public_conditioning_probe()
    assert result['synthetic_only'] and len(result['records']) == 2
    assert all(result[k] is False for k in ('formal_data_access', 'source_checkpoint_access',
        'target_query_access', 'target_truth_access', 'classifier_labels_used', 'model_updated', 'source_selection_changed'))
    for row in result['records']:
        assert row['packets'] == 30
        assert row['restored_coordinates_max_error'] < 1e-12
        assert row['scratch_convolution_max_error'] < 1e-12
        assert max(r['inverse_matrix_range_max'] for r in row['moment_ranges']) > 1e-4
    assert result['conclusions']['cause_of_source_regression_proven'] is False
    assert result['conclusions']['exact_inverse_alone_is_a_new_performance_improvement'] is False


def test_transport_rejects_cross_packet_shapes():
    with pytest.raises(ValueError):
        restore_coordinates(torch.zeros(2, 2, 12, 256), torch.zeros(1, 4, 3, 3))
