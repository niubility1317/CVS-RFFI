import numpy as np

from experiments.pamf_phase3_restoration.source_receiver_priors import fit_prior, select_receivers


def test_matched_receiver_sets_preserve_roles_and_holdout():
    rx = np.repeat([1, 3, 4, 6, 8], 12600)
    day = np.tile(np.repeat([1, 2, 3], 4200), 5)
    roles = np.tile(np.concatenate((np.repeat('L_s', 420), np.repeat('U_s', 3780))), 15)
    chosen = select_receivers(rx, day, roles)
    for held, (same, other) in chosen.items():
        assert len(same) == len(other) == 12600
        assert (rx[same] == held).all() and (rx[other] != held).all()
        assert (roles[same] == 'L_s').sum() == (roles[other] == 'L_s').sum() == 1260
        assert len(np.unique(other)) == 12600
    repeated = select_receivers(rx, day, roles)
    assert all(np.array_equal(chosen[r][1], repeated[r][1]) for r in chosen)


def test_fit_uses_only_selected_iq_and_matches_population_shrinkage():
    data = np.random.default_rng(41).normal(size=(17, 2, 256)).astype(np.float32)
    selected = np.asarray([0, 2, 4, 6, 8, 10, 12, 14])
    mean, covariance = fit_prior(data, selected)
    x = data[selected].astype(np.float64).reshape(len(selected), 512)
    centered = x-x.mean(0)
    expected = centered.T@centered/len(selected)
    expected = .999*expected + .001*np.trace(expected)/512*np.eye(512)
    np.testing.assert_array_equal(mean, x.mean(0).astype(np.float32))
    np.testing.assert_array_equal(covariance, expected.astype(np.float32))
    data[1::2] = 1e6
    changed = fit_prior(data, selected)
    assert all(np.array_equal(a,b) for a,b in zip((mean,covariance), changed))
