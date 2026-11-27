import numpy as np

from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER
from experiments.pamf_lora_main.w24_practical_direction_factor import interventions


def test_regions_hold_the_other_factor_fixed():
    received = np.zeros((2, 8, CENTER+48), dtype=np.complex64)
    learned_edit = received.copy()
    true_edit = received.copy()
    learned_edit[0, :, :CENTER] = 2j
    learned_edit[0, :, CENTER:] = 4
    true_edit[0, :, :CENTER] = 3
    true_edit[0, :, CENTER:] = 2+2j
    learned_edit[1] = 0.25j

    direct, frozen, amplitude, direction, both, projected = interventions(received, learned_edit, true_edit)
    np.testing.assert_array_equal(direct, received)
    np.testing.assert_array_equal(frozen, learned_edit)
    np.testing.assert_array_equal(amplitude[0, :, :CENTER], 3j)
    np.testing.assert_allclose(amplitude[0, :, CENTER:], np.sqrt(8))
    np.testing.assert_array_equal(direction[0, :, :CENTER], 2)
    np.testing.assert_allclose(direction[0, :, CENTER:], (2+2j)*np.sqrt(2))
    np.testing.assert_array_equal(both[0], true_edit[0])
    np.testing.assert_array_equal(projected[0, :, :CENTER], 0)
    np.testing.assert_array_equal(projected[0, :, CENTER:], 2)
    for wave in (amplitude, direction, both, projected):
        np.testing.assert_array_equal(wave[1], learned_edit[1])


def test_true_both_recovers_noise_reference_when_learned_edit_is_zero():
    observed = np.zeros((1, 8, CENTER+48), dtype=np.complex64)
    signal = np.ones_like(observed)
    direct, frozen, amplitude, direction, both, projected = interventions(observed, observed, signal)
    for wave in (direct, frozen, amplitude, direction, projected):
        np.testing.assert_array_equal(wave, observed)
    np.testing.assert_array_equal(both, signal)
