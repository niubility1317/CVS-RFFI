import numpy as np

from experiments.pamf_unrolled_restore.wisig_timing_diagnostic import oracle_readouts, paired_delay


def test_isolated_pair_timing_direction_and_edge_replacement():
    rng = np.random.default_rng(684)
    ref = (rng.standard_normal((3, 256)) + 1j * rng.standard_normal((3, 256))).astype(np.complex64)
    delay = np.array([-50, 0, 41])
    samples = np.arange(256) - delay[:, None]
    observed = np.take_along_axis(ref, np.clip(samples, 0, 255), axis=1) * np.complex64(.8j)
    detected, after, before = paired_delay(observed, ref)
    np.testing.assert_array_equal(detected, delay)
    np.testing.assert_allclose(after, 1., atol=3e-7)
    assert before[0] < .2 and before[2] < .2
    (overlap, exact_tail, exact_gain), stats = oracle_readouts(observed, ref)
    np.testing.assert_array_equal(stats['paired_delta'], delay)
    valid = ((np.arange(256) + delay[:, None]) >= 0) & ((np.arange(256) + delay[:, None]) < 256)
    np.testing.assert_array_equal(exact_tail[valid], overlap[valid])
    np.testing.assert_array_equal(exact_tail[~valid], ref[~valid])
    np.testing.assert_allclose(exact_gain, ref, atol=1e-6)
