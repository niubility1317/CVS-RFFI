import pytest

from experiments.pamf_phase2_practical.r42_direct_c_dispatch import jobs


def test_mix01_matrix_covers_four_seeds_three_pools_seven_receivers():
    rt = {'model_seeds': [2026092701, 2026092702, 2026092703, 2026092704], 'target_receivers': ['r' + str(i) for i in range(7)],
          'capsules': ['capsules/P%02d/MIX01/r%d' % (pool, rx) for pool in range(1, 4) for rx in range(7)]}
    result = jobs(rt)
    assert len(result) == 84
    assert len({entry['key'] for entry in result}) == 84
    assert all(len(entry['points']) == 16 for entry in result)
    assert all(set(map(tuple, entry['points'])) == {(k, n) for k in (1, 5, 10, 20) for n in (2, 5, 10, 20)} for entry in result)


def test_matrix_rejects_missing_or_duplicate_capsule():
    rt = {'model_seeds': [2026092701, 2026092702, 2026092703, 2026092704], 'target_receivers': ['r' + str(i) for i in range(7)],
          'capsules': ['capsules/P01/MIX01/r0'] * 21}
    with pytest.raises(ValueError):
        jobs(rt)
