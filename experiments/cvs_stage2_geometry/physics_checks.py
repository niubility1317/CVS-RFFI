"""Synthetic fixed-physics checks; no model, source/target I/O or fitting."""
import argparse
import json
import math

import torch

from .physics_audit import (PhysicalAudit, algebraic_audit, fourier_coefficients,
                            sample_series, template, projective_time, repeat_energy,
                            SAMPLE_RATE, PERIOD)


def _iq(z):
    return torch.stack((z.real, z.imag), 1)


def _linear(z, h1, h3):
    def shift(delay):
        return torch.nn.functional.pad(z[..., :-delay], (delay, 0))
    return z + h1 * shift(1) + h3 * shift(3)


def run(device='cpu'):
    audit = PhysicalAudit(device)
    evidence = algebraic_audit(device)
    assert evidence['template_conjugate_delay_error'] < 1e-12
    assert evidence['discrete_orders_135_rank'] == evidence['discrete_four_repeats_rank'] == 3
    assert 54 < evidence['discrete_orders_135_condition'] < 55
    assert evidence['sideband_ratio_error'] < 1e-13
    assert evidence['sideband_orders_35_complex_rank'] == 1
    assert all(v['rank'] == 1 for v in evidence['residual_svd'].values())
    assert evidence['sampled_cubic_sideband_alias_error'] > 1e-3
    bound = evidence['linear_action_bound']
    assert bound['checked_singular_min'] >= bound['lower'] - 1e-12
    assert bound['checked_singular_max'] <= bound['upper'] + 1e-12
    # Dense oversampling resolves fifth-order extent (+/-30 harmonics), before
    # receiver masking. It independently checks the Fourier-convolution path.
    dense_n = torch.arange(320, device=device, dtype=torch.float64) / 16
    dense_s = template(dense_n)
    oversampled_errors = {}
    antialias_errors = {}
    for order in (3, 5):
        powers = dense_s * dense_s.abs().pow(order - 1)
        analytic = sample_series(fourier_coefficients(order), dense_n)
        error = float((powers - analytic).abs().max())
        assert error < 1e-11
        oversampled_errors[str(order)] = error
        harmonic_grid = torch.fft.fftfreq(320, device=device, dtype=torch.float64) * 320
        receiver_filtered = torch.fft.ifft(torch.fft.fft(powers) * (harmonic_grid.abs() < 8))
        decimated = receiver_filtered[::16]
        direct_physical = sample_series(fourier_coefficients(order, True),
                                        torch.arange(20, device=device, dtype=torch.float64))
        antialias_error = float((decimated - direct_physical).abs().max())
        assert antialias_error < 1e-11
        antialias_errors[str(order)] = antialias_error
    n = torch.arange(256, device=device, dtype=torch.float64)
    steady = template(n)[None]
    raw_before = steady.clone()
    baseline = audit(_iq(steady))
    assert torch.equal(steady, raw_before)
    assert baseline['features'].shape == (1, 42)
    assert not baseline['contract']['candidate_window_assumed_STF']
    assert not baseline['contract']['tail_is_complete_symbol']
    assert float(baseline['stats']['temporal_distance_from_steady'].max()) < 1e-10
    assert float(baseline['stats']['nonlinear_80_residual_power'].max()) < 1e-24
    assert abs(float(baseline['stats']['w80_160_template_correlation'][0]) - 1) < 1e-12
    assert abs(float(baseline['stats']['early_reference_cycle_correlation'][0]) - 1) < 1e-12
    # A fractional template delay remains an LTI tone phase change. It does
    # not create nonlinear evidence, while the timing quality reports it.
    delayed = audit(_iq(template(n, .3125)[None]))
    assert float(delayed['stats']['w80_160_template_correlation'][0]) > 1 - 1e-12
    assert float(delayed['stats']['w80_160_best_public_delay_samples'][0]) == .3125
    assert float(delayed['stats']['nonlinear_80_residual_power'][0]) < 1e-23
    source_stability = audit.linear_stability(_iq(steady), baseline)
    assert source_stability['contract']['target_access'] is False
    assert all(v.shape == (1,) and torch.isfinite(v).all() for v in source_stability['stats'].values())
    assert float(source_stability['stats']['controlled_L_corner0_temporal_feature_l2'][0]) > 0
    # Exact nonzero shared gain invariance in the specified per-tone model.
    generator = torch.Generator().manual_seed(9821)
    pair = torch.complex(torch.randn(6, 2, 12, generator=generator, dtype=torch.float64),
                         torch.randn(6, 2, 12, generator=generator, dtype=torch.float64)).to(device)
    gains = torch.complex(torch.randn(6, 12, generator=generator, dtype=torch.float64),
                          torch.randn(6, 12, generator=generator, dtype=torch.float64)).to(device)
    a, valid, _ = projective_time(pair)
    b, valid2, _ = projective_time(pair * gains[:, None])
    gain_error = float((a - b).abs().max())
    assert valid.all() and valid2.all() and gain_error < 1e-12
    # A steady waveform filtered with its actual periodic prehistory obeys the
    # common-gain hypothesis. Causal zero extension of a captured window does not.
    h1, h3 = .06 + .06j, -.06 + .06j
    ideal_l = (template(n) + h1 * template(n - 1) + h3 * template(n - 3))[None]
    causal_l = _linear(steady, h1, h3)
    ideal_features = audit(_iq(ideal_l))['features'][:, :36]
    causal_features = audit(_iq(causal_l))['features'][:, :36]
    ideal_error = float((ideal_features - baseline['features'][:, :36]).abs().max())
    causal_error = float((causal_features - baseline['features'][:, :36]).abs().max())
    assert ideal_error < 1e-11 and causal_error > 1e-5
    impulse = torch.zeros_like(steady)
    impulse[:, 79] = 1
    commutator = _linear(impulse, h1, h3)[:, 80:160] - _linear(impulse[:, 80:160], h1, h3)
    boundary_norm = float(commutator.norm())
    assert boundary_norm > .08
    # One reference CFO removes a truly common phase slope. An early-only
    # slope remains visible; no per-window refit silently erases its dynamics.
    omega = 2 * math.pi * 2000 / SAMPLE_RATE
    common_cfo = audit(_iq(steady * torch.exp(1j * omega * n)[None]))
    common_error = float((common_cfo['features'][:, :36] - baseline['features'][:, :36]).abs().max())
    assert common_error < 1e-11
    assert abs(float(common_cfo['stats']['reference_cfo_hz'][0]) - 2000) < 1e-7
    early = steady.clone()
    early[:, :80] *= torch.exp(1j * omega * n[:80])
    early_output = audit(_iq(early))
    early_difference = float((early_output['features'][:, :36] - baseline['features'][:, :36]).abs().max())
    assert abs(float(early_output['stats']['reference_cfo_hz'][0])) < 1e-7 and early_difference > .001
    # Unknown extension may fail STF tests without entering default features.
    extension = steady.clone()
    extension[:, 160:240] = 0
    extension[:, 240:256] *= 2
    extension_output = audit(_iq(extension))
    assert torch.equal(extension_output['features'], baseline['features'])
    assert float(extension_output['stats']['w160_240_power'][0]) == 0
    assert float(extension_output['stats']['tail16_power_ratio'][0]) > float(baseline['stats']['tail16_power_ratio'][0])
    zero = audit(torch.zeros(2, 2, 256, device=device))
    assert not zero['features'].any() and not zero['stats']['reference_cfo_valid'].any()
    assert not zero['stats']['temporal_valid_tone_fraction'].any()
    # Negative repeat-energy estimates are evidence, never clamped away.
    alternating = torch.tensor([1., -1., 1., -1.], device=device, dtype=torch.float64)[None, :, None]
    alternating = alternating.expand(1, 4, 20).to(torch.complex128)
    negative = float(repeat_energy(alternating)['repeat_energy'][0])
    assert abs(negative + 1 / 3) < 1e-12
    signal = torch.complex(torch.randn(20, generator=generator, dtype=torch.float64),
                           torch.randn(20, generator=generator, dtype=torch.float64)).to(device)
    noise = torch.complex(torch.randn(4096, 4, 20, generator=generator, dtype=torch.float64),
                          torch.randn(4096, 4, 20, generator=generator, dtype=torch.float64)).to(device) * .05
    repeated = repeat_energy(signal[None, None] + noise)
    repeated_error = abs(float(repeated['repeat_energy'].mean() - signal.abs().square().mean()))
    assert repeated_error < .002
    return dict(status='PASS', device=str(device), algebra=evidence,
                oversampled_convolution_errors=oversampled_errors,
                oversample_filter_then_decimate_errors=antialias_errors,
                common_gain_projector_error=gain_error, periodic_linear_temporal_error=ideal_error,
                zero_extended_linear_temporal_change=causal_error, window_linear_commutator_norm=boundary_norm,
                common_cfo_temporal_error=common_error, early_only_cfo_preserved_change=early_difference,
                extension_excluded_from_features=True, zero_energy_finite_and_masked=True,
                negative_repeat_energy_preserved=negative, iid_repeat_energy_mean_error=repeated_error,
                actual_received_innovation_not_claimed_thermal_noise=True, physical_feature_dimension=42,
                controlled_actual_packet_L_interface=True,
                target_access=False, model_training=False, input_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.device), allow_nan=False), flush=True)
