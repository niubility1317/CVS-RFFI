import numpy as np
import pytest

from experiments.pamf_lora_main.dura_real_gain_wisig_full_source import select_fresh_rows


def test_selects_balanced_fresh_source_ids_without_using_labels():
    ids = np.arange(1500).astype(str)
    rx = np.ones(1500, dtype=int)
    day = np.repeat([1, 2, 3], 500)
    previous = ids[np.r_[0:100, 500:600, 1000:1100]]
    rows = select_fresh_rows(ids, rx, day, previous)
    assert len(rows) == 300 and not np.isin(ids[rows], previous).any()
    assert all((day[rows] == d).sum() == 100 for d in (1, 2, 3))
    np.testing.assert_array_equal(rows, select_fresh_rows(ids, rx, day, previous))
    with pytest.raises(ValueError):
        select_fresh_rows(ids[:1000], rx[:1000], day[:1000], previous)
