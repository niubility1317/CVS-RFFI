import numpy as np
import pytest
from experiments.pamf_phase3_open_world import source_spectral_calibration as s


def fixture():
    rng = np.random.default_rng(62031)
    x = rng.normal(size=(30, 2, 256)).astype(np.float32)
    rx = np.repeat(np.asarray(s.RX, np.int64), 6)
    day = np.tile(np.asarray([1, 1, 2, 2, 3, 3], np.int64), 5)
    return x, rx, day


def test_frozen_spectral_application_is_row_local_and_identity():
    x, _, _ = fixture()
    gain = np.ones(256, np.float64)
    np.testing.assert_allclose(s.apply(x, gain), x, atol=1e-7, rtol=0)
    gain[20:40] = .5
    gain[100:120] = 2
    together = s.apply(x, gain)
    np.testing.assert_array_equal(together[3], s.apply(x[3:4], gain)[0])
    np.testing.assert_array_equal(together[3], s.apply(np.concatenate([x[3:4], x[:2]]), gain)[0])


def test_fitting_ignores_later_records_and_references_exclude_held_receiver():
    x, rx, day = fixture()
    state = s.fit(x, rx, day)
    changed = x.copy()
    changed[day != 1] *= 20
    again = s.fit(changed, rx, day)
    for key in state:
        np.testing.assert_array_equal(state[key], again[key])
    assert state['gains'].shape == (5, 256)
    assert np.all((state['gains'] >= .5) & (state['gains'] <= 2))
    changed = x.copy()
    changed[(rx == 1) & (day == 1), 0, :64] *= 5
    other = s.fit(changed, rx, day)
    np.testing.assert_array_equal(state['references'][0], other['references'][0])
    assert not np.array_equal(state['gains'][0], other['gains'][0])


@pytest.mark.parametrize('issue', ['nan', 'zero', 'shape', 'gain', 'missing_early', 'unknown_receiver'])
def test_unusable_inputs_fail_explicitly(issue):
    x, rx, day = fixture()
    with pytest.raises(ValueError):
        if issue == 'nan':
            x[0, 0, 0] = np.nan
            s.mean_spectrum(x)
        elif issue == 'zero':
            s.mean_spectrum(np.zeros_like(x))
        elif issue == 'shape':
            s.apply(x[:, :, :255], np.ones(256))
        elif issue == 'gain':
            s.apply(x, np.full(256, 2.001))
        elif issue == 'missing_early':
            day[(rx == 1) & (day == 1)] = 2
            s.fit(x, rx, day)
        else:
            rx[0] = 34
            s.fit(x, rx, day)


def test_config_rejects_query_input_and_target_ancestor():
    c = dict(run_id=s.RUN_ID, code_commit='a'*40, IQ_root='/source/cache', output_root='/new/diagnostic', batch_size=128,
        backend_config=dict(dura_checkpoint='/G/final.pt', dura_release='/G/code', train_config=dict(run_id='20261030-phase3-star-signal-open-s2026092701-r03',
            code_commit='dc01a402582f93e452dbaa3ef59a9fbb44151818', arm='E_SIGNAL', IQ_root='/source/cache', output_root='/r03/train', native_config=dict(source_output='/F0/source'))))
    s.validate_config(c)
    c['Q_root'] = '/target/query'
    with pytest.raises(ValueError):
        s.validate_config(c)
    del c['Q_root']
    c['backend_config']['train_config']['run_id'] = 'target-adapted'
    with pytest.raises(ValueError):
        s.validate_config(c)


def test_source_screen_cannot_pass_by_collapsing_identity():
    n = 6 * 5 * 3
    y = np.repeat(np.arange(6, dtype=np.int64), 15)
    rx = np.tile(np.repeat(np.asarray(s.RX, np.int64), 3), 6)
    day = np.tile(np.arange(1, 4, dtype=np.int64), 30)
    logits = np.eye(6, dtype=np.float32)[y] * 5
    z = np.pad(np.eye(6, dtype=np.float32)[y], ((0, 0), (0, 154)))
    rows = dict(logits=np.stack((logits, np.roll(logits, 1, axis=1)), axis=1), F0_unit=np.stack((z, z), 1), r03_unit=np.stack((z, z), 1))
    result = s.summarize(rows, dict(y=y, rx_i=rx, day_i=day))
    assert result['paired_later']['correct_lost'] == n * 2 // 3
    assert not result['source_identity_protected']
    assert not result['source_screen_passed']
    assert result['R_U'] is None and result['full_goal_status'] == 'ACTIVE'
