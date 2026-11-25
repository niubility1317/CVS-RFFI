import numpy as np
import pytest

from experiments.pamf_unrolled_restore import wisig_forced_clean as cleanup


def test_clean_restoration_is_full_batch_and_reaches_reader():
    def restore(model, threshold, iq, variance, *, force):
        assert force and threshold is None and variance is None
        assert len(iq) == 3
        return iq + 4, np.ones(len(iq), dtype=bool)

    original = np.arange(3)
    result = cleanup.classify_restored(restore, lambda x: x + 1, None, original)
    assert result.tolist() == [5, 6, 7]
    assert original.tolist() == [0, 1, 2]


def test_recovered_clean_replaces_only_clean_and_requires_raw_parity():
    values = np.zeros((3, 7, 2), dtype=np.uint8)
    replay = {arm: values.copy() for arm in cleanup.METHODS}
    clean = {'raw': values[:, 0].copy(), **{
        arm: np.full((3, 2), 5, dtype=np.uint8) for arm in cleanup.ARMS}}
    merged = cleanup.merge_clean(replay, clean)
    for arm in cleanup.ARMS:
        assert np.all(merged[arm][:, 0] == 5)
        assert np.array_equal(merged[arm][:, 1:], replay[arm][:, 1:])
        assert not replay[arm].any()
    assert np.array_equal(merged['R69'], replay['R69'])
    clean['raw'][0, 0] = 1
    with pytest.raises(ValueError, match='raw'):
        cleanup.merge_clean(replay, clean)
