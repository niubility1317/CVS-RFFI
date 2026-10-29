import numpy as np
import pytest

from experiments.pamf_phase3_open_world import star_relation_oe_eval as e


def known_fixture():
    y = np.repeat(np.arange(6), 300).astype(np.int64)
    rx = np.tile(np.repeat(e.common.RECEIVERS, 10), 36).astype(np.int64)
    scene = np.tile(np.repeat(np.asarray(e.common.SCENES), 50), 6)
    ids = np.asarray(['known-'+str(i) for i in range(len(y))])
    return ids, y, y.copy(), y.copy(), rx, scene


def test_entropy_is_stable_and_identity_independent():
    z = np.asarray([[0.]*6, [10000., -10000., -10000., -10000., -10000., -10000.]])
    assert np.allclose(e.entropy_score(z), [0., 1.], atol=1e-14)
    assert np.allclose(e.entropy_score(z+123.), e.entropy_score(z))
    assert np.allclose(e.entropy_score(z[:, ::-1]), e.entropy_score(z))
    with pytest.raises(ValueError):
        e.entropy_score(np.asarray([[np.nan]*6]))


def test_only_seven_groups_are_hard_and_ties_stay_complete():
    ids, y, c, b, rx, scene = known_fixture()
    values = np.full(len(y), .9)
    # One observation of each true TX in one RX/scene: 2% of that scene,
    # 10% of that RX/scene cell. The latter is explicitly diagnostic only.
    for label in range(6):
        i = np.flatnonzero((y == label) & (rx == 1) & (scene == e.common.SCENES[0]))[0]
        values[i] = .1
    tau, r = e.calibrate_known(ids, y, c, b, values, rx, scene)
    assert tau == .9 and r['constraints_passed']
    assert int((values < tau).sum()) == 6
    assert len(r['hard_constraint_groups']) == 7 and len(r['groups']) == 42
    assert not r['groups']['rxscene/1/'+e.common.SCENES[0]]['constraints_passed']
    assert r['shared_predicted_class_tail_constraint'] is False


def test_infeasible_candidate_accepts_all_without_hiding_failure():
    ids, y, c, b, rx, scene = known_fixture()
    c[y == 0] = 1
    values = np.linspace(.1, .9, len(y))
    tau, r = e.calibrate_known(ids, y, c, b, values, rx, scene)
    assert tau == values.min() and r['zero_rejection_fallback']
    assert not r['constraints_passed'] and np.all(values >= tau)


def test_known_calibration_rejects_hidden_unknown_labels_and_invalid_scores():
    ids, y, c, b, rx, scene = known_fixture()
    y[0] = 6
    with pytest.raises(ValueError):
        e.calibrate_known(ids, y, c, b, np.ones(len(y)), rx, scene)

