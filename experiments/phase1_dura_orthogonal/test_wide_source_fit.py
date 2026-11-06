import numpy as np
import pytest

from experiments.phase1_dura_orthogonal.train_source import FIT_SIZE, WIDE_FIT_SIZE, select_source_fit


def test_wide_source_fit_is_deterministic_and_excludes_held_out_ids():
    n = 50000
    rx = np.ones(n, dtype=np.int32)
    day = np.ones(n, dtype=np.int32)
    roles = np.full(n, 'U_s')
    roles[:5000] = 'L_s'
    rx[1000:2000], day[1000:2000] = 8, 3
    eval_ids = np.arange(3000, 3672)
    fit = select_source_fit(rx, day, roles, eval_ids, WIDE_FIT_SIZE)
    assert len(fit) == WIDE_FIT_SIZE and len(np.unique(fit)) == WIDE_FIT_SIZE
    assert not np.isin(fit, eval_ids).any()
    assert not ((rx[fit] == 8) & (day[fit] == 3)).any()
    np.testing.assert_array_equal(fit, select_source_fit(rx, day, roles, eval_ids, WIDE_FIT_SIZE))
    assert len(select_source_fit(rx, day, roles, eval_ids, FIT_SIZE)) == FIT_SIZE


def test_wide_source_fit_rejects_invalid_eligible_roles():
    roles = np.full(40000, 'U_s')
    roles[20] = 'V'
    with pytest.raises(ValueError):
        select_source_fit(np.ones(40000), np.ones(40000), roles, np.array([], dtype=int), WIDE_FIT_SIZE)
