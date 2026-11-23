import numpy as np

from experiments.cvs_rff_physics.known_excitation import template
from experiments.pamf_unrolled_restore.wisig_timing import estimate_timing, long_template, realign


def test_short_to_long_transition_identifies_both_directions_and_on_time():
    expected = np.array([-31, -15, 0, 8, 12, 23, 35])
    coordinate = np.arange(256) - expected[:, None]
    wave = np.where(coordinate < 200, template(coordinate), long_template(coordinate - 200))
    cfo = np.linspace(-.01, .01, len(expected))
    capture = wave * np.exp(1j * cfo[:, None] * np.arange(256))
    estimated, margin, _ = estimate_timing(capture)
    np.testing.assert_array_equal(np.where(margin > 0, estimated, 0), expected)


def test_realign_preserves_observed_overlap_and_untouched_rows():
    rng = np.random.default_rng(83)
    z = rng.standard_normal((3, 256)) + 1j * rng.standard_normal((3, 256))
    delta = np.array([-40, 0, 40])
    for aligned in (realign(z, delta), realign(z, delta, protocol=True, cfo=np.zeros(3))):
        np.testing.assert_array_equal(aligned[0, 40:], z[0, :-40])
        np.testing.assert_array_equal(aligned[1], z[1])
        np.testing.assert_array_equal(aligned[2, :-40], z[2, 40:])
        assert np.isfinite(aligned).all()
