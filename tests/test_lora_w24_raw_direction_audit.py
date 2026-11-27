import numpy as np

from experiments.pamf_lora_main import w24_raw_direction_audit as audit


def test_equal_edit_dose_and_nonnegative_gain_in_linear_w24_coordinates(monkeypatch):
    # The deployed aperture is bound from the W24 archive during the real smoke.
    def fixture_aperture(raw):
        return raw.reshape(-1, 8, 1024)[..., 32:896:2]

    monkeypatch.setattr(audit, 'native_aperture', fixture_aperture)
    rng = np.random.default_rng(79)
    capture = (rng.normal(size=(3, 8192)) + 1j*rng.normal(size=(3, 8192))).astype(np.complex64)
    noise = capture.copy() * .4
    noise[2] = 0
    signal = capture - noise
    learned = np.stack((-.25*noise[0], .5*noise[1]+.3j*noise[1], .7j*capture[2]))
    recovered = capture + learned
    (gain_only, same_dose, exact, preserved), stats = audit.edits(capture, recovered, signal)
    np.testing.assert_allclose(stats['learned_best_nonnegative_gain'][:2], [4., 0.], atol=1e-5)
    np.testing.assert_allclose(gain_only[0], signal[0], atol=1e-6)
    np.testing.assert_array_equal(gain_only[1], capture[1])
    np.testing.assert_allclose(same_dose[0], recovered[0], atol=1e-6)
    np.testing.assert_array_equal(exact, signal)
    np.testing.assert_allclose(preserved[1], signal[1]+.3j*noise[1], atol=1e-6)
    np.testing.assert_array_equal(np.stack((gain_only, same_dose, exact, preserved))[:, 2],
                                  np.repeat(capture[None, 2], 4, axis=0))
    correction = fixture_aperture(same_dose-capture)[..., :audit.CENTER]
    required = fixture_aperture(recovered-capture)[..., :audit.CENTER]
    np.testing.assert_allclose(np.linalg.norm(correction[1]), np.linalg.norm(required[1]), rtol=1e-6)
