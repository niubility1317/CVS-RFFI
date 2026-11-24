import numpy as np

from experiments.pamf_unrolled_restore.wisig_joint_preamble import fit, waveform
from experiments.pamf_unrolled_restore.wisig_smooth_preamble import (
    FIXED, MODES, SMOOTH, TIME, aligned, blend_weights, choose, estimate, observations, readouts,
)
from experiments.pamf_unrolled_restore.wisig_timing import NOMINAL


def synthetic():
    shifts = np.asarray([-28, -12, 12, 28])
    bank = waveform(TIME[None] - NOMINAL - shifts[:, None])
    signal = ((1+1j)*bank).astype(np.complex64)
    return shifts, signal, np.stack((signal.real, signal.imag), axis=1)


def test_fixed_same_samples_matches_registered_response():
    shifts, signal, _ = synthetic()
    apparent, predictive, _ = estimate(signal, FIXED)
    old = fit(signal, 'joint_channel')
    np.testing.assert_array_equal(choose(apparent)[0], old.delta)
    np.testing.assert_allclose(choose(apparent)[1], old.margin, atol=1e-6)
    np.testing.assert_array_equal(choose(predictive)[0], shifts)


def test_smooth_shift_is_stable_under_the_supported_gain_model():
    shifts, signal, _ = synthetic()
    gain = blend_weights(TIME) @ np.asarray([.8+.8j, 1.1-.1j, .5+.4j, 1.4-.8j], dtype=np.complex64)
    apparent, predictive, _ = estimate(signal*gain, SMOOTH)
    np.testing.assert_array_equal(choose(apparent)[0], shifts)
    np.testing.assert_array_equal(choose(predictive)[0], shifts)


def test_gate_bypass_and_both_boundaries_keep_measured_overlap():
    _, _, iq = synthetic()
    bypass, _ = readouts(iq, dict.fromkeys(MODES, np.inf))
    assert tuple(bypass) == MODES
    assert all(np.array_equal(wave, iq) for wave in bypass.values())

    raw, info, fits = observations(iq)
    shifts, _, response = fits['smooth_held']
    predicted = aligned(raw, shifts, info.coarse_rad_per_sample, response, varying=True)
    mirrored = aligned(raw, shifts, info.coarse_rad_per_sample, response, reflect=True)
    coordinates = TIME[None] + shifts[:, None]
    valid = (coordinates >= 0) & (coordinates < len(TIME))
    rows, cols = np.nonzero(valid)
    np.testing.assert_array_equal(predicted[rows, cols], raw[rows, coordinates[rows, cols]])
    np.testing.assert_array_equal(predicted[valid], mirrored[valid])
    assert np.isfinite(predicted).all() and np.any(predicted[~valid] != mirrored[~valid])
