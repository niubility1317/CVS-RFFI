import pytest
import torch

from cvsrffi.deployment_orbit import (
    PAIR_REFORM_NUISANCE_NAMES, PAIR_REFORM_PROBE_NAMES,
    apply_physical_probe, apply_pair_reform_nuisance, physical_probe_registry,
    physical_reliability, sample_pair_reform_probe,
)


@pytest.mark.parametrize('name,amount', [('sto', .02), ('sampling_clock', 200.)])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_time_boundary_continuity(name, amount, sign):
    x = torch.arange(1, 34, dtype=torch.float64).repeat(2, 2, 1)
    diffs = []
    for step in [amount, amount / 2, amount / 4]:
        y = apply_physical_probe(x, name=name, amount=sign * step, sample_rate_hz=1e6)
        diffs.append((y-x).square().sum())
    assert torch.allclose(diffs[0] / diffs[1], torch.tensor(4., dtype=x.dtype), rtol=1e-6)
    assert torch.allclose(diffs[1] / diffs[2], torch.tensor(4., dtype=x.dtype), rtol=1e-6)
    assert torch.equal(apply_physical_probe(x, name=name, amount=0., sample_rate_hz=1e6), x)


def test_registry_unique_truthful_and_no_opposite_cfo_route():
    registry = physical_probe_registry()
    assert 'pa' not in registry and 'dac_nonlinearity' not in registry
    assert 'clock_skew' not in registry and 'iq_phase' not in registry
    assert registry['sampling_clock'].unit == 'ppm'
    assert registry['common_phase'].kind == registry['net_cfo'].kind == 'mixed'
    assert set(PAIR_REFORM_PROBE_NAMES) == {'amplitude_nonlinearity', 'iq_gain'}
    assert 'net_cfo' not in PAIR_REFORM_NUISANCE_NAMES
    assert all(s.location == 'received_iq_post_channel' for s in registry.values())


def test_independent_noise_is_fixed_and_linear_not_signal_residual():
    x = torch.ones(2, 2, 32, dtype=torch.float64)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(4), dtype=x.dtype)
    y = apply_physical_probe(x, name='noise_strength', amount=.2, sample_rate_hz=1e6, fixed_noise=noise)
    assert torch.allclose(y-x, .2 * noise)
    assert torch.equal(y, apply_physical_probe(x, name='noise_strength', amount=.2, sample_rate_hz=1e6, fixed_noise=noise))
    with pytest.raises(ValueError, match='fixed_noise'):
        apply_physical_probe(x, name='noise_strength', amount=.2, sample_rate_hz=1e6)


def test_unknown_metadata_is_not_high_or_low_quality():
    score, valid = physical_reliability({}, batch_size=2, device=torch.device('cpu'))
    assert not valid.any() and torch.isnan(score).all()
    score, valid = physical_reliability({'snr_db': [40., float('nan')]}, batch_size=2, device=torch.device('cpu'))
    assert valid.tolist() == [True, False]
    assert score[0] > .9 and torch.isnan(score[1])
    score, valid = physical_reliability({'snr_db': [-30., 40.], 'clip_ratio': [0., 0.]}, batch_size=2, device=torch.device('cpu'))
    assert valid.all() and score[0] < score[1]


def test_no_periodic_wrap_and_replay_and_nuisance_identity():
    x = torch.zeros(12, 2, 32)
    x[:, :, -1] = 1.
    y = apply_physical_probe(x, name='sto', amount=.1, sample_rate_hz=1e6)
    assert torch.equal(y[:, :, 0], torch.zeros_like(y[:, :, 0]))
    a = sample_pair_reform_probe(x, seed=3, strength=.03, sample_rate_hz=1e6)
    b = sample_pair_reform_probe(x, seed=3, strength=.03, sample_rate_hz=1e6)
    assert all(torch.equal(i, j) for i, j in zip(a, b))
    assert a[1].max() < len(PAIR_REFORM_PROBE_NAMES)
    assert torch.equal(apply_pair_reform_nuisance(x, torch.zeros(12, 5), delta=.05, sample_rate_hz=1e6, fixed_noise=torch.ones_like(x)), x)


@pytest.mark.parametrize('name', tuple(physical_probe_registry()))
def test_every_audited_operator_has_zero_identity_and_finite_backward(name):
    x = torch.randn(2, 2, 16, dtype=torch.float64, requires_grad=True)
    noise = torch.randn_like(x)
    zero = apply_physical_probe(x, name=name, amount=0., sample_rate_hz=1e6, fixed_noise=noise)
    assert torch.equal(zero, x)
    y = apply_physical_probe(x, name=name, amount=.01, sample_rate_hz=1e6, fixed_noise=noise)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_field_does_not_turn_into_zero_error_evidence():
    score, valid = physical_reliability({'clip_ratio': [float('inf'), -1.]}, batch_size=2, device=torch.device('cpu'))
    assert not valid.any() and torch.isnan(score).all()
    score, valid = physical_reliability({'snr_db': [40., 30., 20.]}, batch_size=2, device=torch.device('cpu'))
    assert not valid.any() and torch.isnan(score).all()
