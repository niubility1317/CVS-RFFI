import numpy as np

from experiments.pamf_unrolled_restore.wisig_overlap_isolation import extend_hybrids, isolated_views


def test_swap_matrix_replaces_only_intended_coordinate_or_component():
    raw = np.array([[1j, 2.+1j, 1., 0.]], dtype=np.complex64)
    r69 = np.array([[3., 0., .4j, 1.-1j]], dtype=np.complex64)
    valid = np.array([[False, True, True, True]])
    edge, overlap, phase, amplitude = extend_hybrids(raw, r69, valid)
    np.testing.assert_array_equal(edge[valid], raw[valid])
    np.testing.assert_array_equal(edge[~valid], r69[~valid])
    np.testing.assert_array_equal(overlap[valid], r69[valid])
    np.testing.assert_array_equal(overlap[~valid], raw[~valid])
    np.testing.assert_allclose(np.abs(phase[valid]), np.abs(raw[valid]), atol=1e-7)
    np.testing.assert_allclose(np.abs(amplitude[valid]), np.abs(r69[valid]), atol=1e-7)
    np.testing.assert_array_equal(phase[~valid], r69[~valid])
    np.testing.assert_array_equal(amplitude[~valid], r69[~valid])


def test_shift_coordinates_and_source_edges():
    rng = np.random.default_rng(2)
    x = (rng.normal(size=(3, 256)) + 1j*rng.normal(size=(3, 256))).astype(np.complex64)
    y = x.copy()
    y[0] = np.roll(x[0], 9)
    y[1] = np.roll(x[1], -12)
    restore = 2. * x
    (aligned, edges, overlap, source_edges, scalar, _, _), tags = isolated_views(y, x, restore)
    np.testing.assert_array_equal(tags['paired_delta'], [9, -12, 0])
    np.testing.assert_array_equal(tags['r69_delay'], 0)
    np.testing.assert_array_equal(tags['missing'], [9, 12, 0])
    np.testing.assert_allclose(edges[0, -9:], restore[0, -9:])
    np.testing.assert_allclose(overlap[1, 12:], restore[1, 12:])
    np.testing.assert_allclose(aligned[1, 12:], x[1, 12:])
    np.testing.assert_allclose(source_edges, x)
    np.testing.assert_allclose(scalar, x, atol=1e-6)
