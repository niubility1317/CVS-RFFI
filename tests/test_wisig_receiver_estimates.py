import torch

from experiments.pamf_unrolled_restore.wisig_receiver_estimates import receive_coordinates, guard_features, guard_score


def test_periodic_preamble_is_estimated_from_received_window():
    torch.manual_seed(9)
    period = torch.randn(3, 20, dtype=torch.complex64)
    received = period.repeat(1, 13)[:, :256]
    iq = torch.stack((received.real, received.imag), dim=1)
    normalized, scale, quality, variance = receive_coordinates(iq)
    assert normalized.shape == (3, 256)
    assert torch.allclose(normalized * scale, received, atol=1e-6)
    assert torch.all(quality > .999)
    assert torch.all(variance < 1e-8)


def test_missing_or_malformed_iq_is_rejected():
    try:
        receive_coordinates(torch.zeros(4, 2, 128))
    except ValueError:
        pass
    else:
        raise AssertionError('Expected 256 observed complex samples')


def test_wisig_guard_features_use_the_visible_periodic_preamble():
    torch.manual_seed(19)
    wave = torch.randn(5, 20, dtype=torch.complex64).repeat(1, 13)[:, :256]
    clean = torch.stack((wave.real, wave.imag), dim=1)
    noisy = clean + torch.randn_like(clean) * .25
    a, b = guard_features(clean), guard_features(noisy)
    assert a.shape == (5, 8) and b.shape == (5, 8)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (b[:, ::2] > a[:, ::2]).all()


def test_guard_score_uses_frozen_source_fit_coefficients():
    iq = torch.randn(3, 2, 256)
    record = dict(midpoint=[0.] * 8, dispersion=[2.] * 8, weight=[1.] * 8, offset=3.)
    expected = (guard_features(iq) / 2).clamp(-10, 10).sum(1) + 3.
    assert torch.allclose(guard_score(iq, record), expected)
