"""Check the phase recovery design on synthetic arrays without data or models."""

import numpy as np


def angular_curvature(wave, lag):
    unit = wave / np.maximum(np.abs(wave), 1e-12)
    return unit[..., 2*lag:] * unit[..., :-2*lag] * unit[..., lag:-lag].conj() ** 2


def angular_projection(proposed, wave, window, time):
    power = np.abs(wave) ** 2
    weights = power * window
    total = weights.sum(-1, keepdims=True)
    safe_total = np.maximum(total, 1e-12)
    centered_time = time - (weights * time).sum(-1, keepdims=True) / safe_total
    time_energy = (weights * centered_time ** 2).sum(-1, keepdims=True)
    offset = (proposed * weights).sum(-1, keepdims=True) / safe_total
    slope = (proposed * weights * centered_time).sum(-1, keepdims=True) / np.maximum(time_energy, 1e-12)
    phase = window * (proposed - offset - slope * centered_time)
    phase = np.where((total > 1e-12) & (time_energy > 1e-12), phase, 0.)
    power_ratio = (power * phase ** 2).sum((1, 2), keepdims=True) / np.maximum(power.sum((1, 2), keepdims=True), 1e-12)
    gain = np.minimum(1., .10 / np.maximum(np.sqrt(power_ratio), 1e-12))
    gain = np.minimum(gain, .35 / np.maximum(np.abs(phase).max((1, 2), keepdims=True), 1e-12))
    return gain * phase


def main():
    rng = np.random.default_rng(20261124)
    sample = np.arange(256)
    time = (sample - 127.5) / 127.5
    distance = np.minimum(np.abs(sample - 127.5), 63.5)
    edge = .5 - .5*np.cos(np.pi*np.clip((distance - 47.5)/16, 0, 1))
    middle = np.roll(edge, 128)
    assert np.count_nonzero(edge) == np.count_nonzero(middle)
    np.testing.assert_allclose(np.linalg.norm(edge), np.linalg.norm(middle), rtol=0, atol=1e-12)

    shape = (32, 6, len(sample))
    wave = (rng.uniform(.01, 2.5, shape) * np.exp(1j*rng.normal(size=shape))).astype(np.complex64)
    gauge = np.exp(1j*(rng.normal(size=(32, 6, 1)) + rng.normal(size=(32, 6, 1))*time))
    curvature_error = max(np.max(np.abs(angular_curvature(wave*gauge, lag)-angular_curvature(wave, lag)))
                          for lag in (1,4,12))
    assert curvature_error < 1e-6
    proposed = rng.normal(size=shape)
    phase = angular_projection(proposed, wave, edge, time)
    assert not phase[..., edge == 0].any()
    power = np.abs(wave) ** 2
    for basis in (1, time):
        assert np.max(np.abs((power*phase*basis).sum(-1))) < 1e-9
    recovered = (wave*np.exp(1j*phase)).astype(np.complex64)
    amplitude_error = np.max(np.abs(np.abs(wave) - np.abs(recovered)))
    dose = np.max(np.sqrt((np.abs(recovered-wave) ** 2).sum((1,2))/power.sum((1,2))))
    assert amplitude_error < 1e-6 and dose <= .10 + 1e-6
    sparse = np.zeros_like(wave)
    sparse[..., 0] = 1.
    assert not angular_projection(proposed, sparse, edge, time).any()
    assert not angular_projection(proposed, np.zeros_like(wave), edge, time).any()
    recovered[::2] = wave[::2]
    assert np.array_equal(recovered[::2], wave[::2])
    print(f'PASS synthetic {len(wave)} packets, amplitude error {amplitude_error:.3g}, '
          f'curvature error {curvature_error:.3g}, largest RMS dose {dose:.6f}')


if __name__ == '__main__':
    main()
