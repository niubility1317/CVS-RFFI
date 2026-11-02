import pytest
from experiments.pamf_phase2_practical.r42_direct_c_takeover import remaining, free_slots, reconcile_old


def test_remaining_excludes_completed_and_inherited_without_repeating_tasks():
    keys = {'a', 'b', 'c', 'd'}
    assert remaining(keys, {'a'}, {'b'}, {'c'}) == {'d'}
    with pytest.raises(ValueError):
        remaining(keys, {'a'}, {'a'}, set())
    with pytest.raises(ValueError):
        remaining(keys, set(), {'outside'}, set())


def test_concurrency_counts_both_old_and_new_workers_and_respects_gpu7():
    limits = {'0': 3, '1': 2, '7': 1}
    inherited = {'a': {'gpu': 0}, 'b': {'gpu': 7}}
    own = {'c': {'gpu': 0}, 'd': {'gpu': 1}}
    assert free_slots([0, 1, 7], limits, inherited, own, {0: 12000, 1: 7000, 7: 20000}) == [0, 1]
    assert free_slots([0, 1, 7], limits, inherited, own, {0: 5999, 1: 7000, 7: 20000}) == [1]


def test_reconcile_blocks_failed_or_unidentified_old_workers():
    state = [{'key': 'prediction-a', 'pid': 101, 'gpu': 0}]
    with pytest.raises(ValueError, match='without closure'):
        reconcile_old({'a', 'b'}, state, {}, {'a'}, set(), {101}, set())
    with pytest.raises(ValueError, match='Unidentified predecessor child'):
        reconcile_old({'a', 'b'}, [], {}, set(), set(), {999}, set())
    with pytest.raises(ValueError, match='without closure'):
        reconcile_old({'a', 'b'}, [], {}, {'b'}, set(), set(), set())
    reconcile_old({'a', 'b'}, state, {}, {'a'}, {'a'}, {101}, set())
    reconcile_old({'a', 'b'}, state, {'a': {'pid': 101, 'gpu': 0}}, {'a'}, set(), {101}, set())
