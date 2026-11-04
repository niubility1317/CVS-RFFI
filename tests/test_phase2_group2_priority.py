import pytest

from experiments.pamf_phase2_practical.group2_c_backbone_schedule import _priority_wait, _worker_pending


def test_other_methods_are_first_without_changing_indices_or_reused_jobs():
    jobs = [{'b_method': method, 'key': str(index)} for index, method in enumerate(
        ['radionet', 'poster', 'radionet', 'hu', 'poster', 'radionet'])]
    rt = {'b_methods': ['radionet', 'poster', 'hu'], 'deferred_b_methods': ['radionet']}
    pending = _worker_pending(rt, jobs, {2, 4})
    assert [(i, job['key']) for i, job in pending] == [(1, '1'), (3, '3'), (0, '0'), (5, '5')]
    assert _worker_pending({'b_methods': rt['b_methods']}, jobs, set()) == list(enumerate(jobs))


def test_deferred_work_waits_until_priority_workers_finish():
    jobs = [{'b_method': 'poster'}, {'b_method': 'radionet'}]
    rt = {'b_methods': ['poster', 'radionet'], 'deferred_b_methods': ['radionet']}
    pending = [(1, jobs[1])]
    assert _priority_wait(rt, jobs, pending, [(0, '0', object(), object())])
    assert not _priority_wait(rt, jobs, pending, [])
    assert not _priority_wait(rt, jobs, pending, [(1, '0', object(), object())])


@pytest.mark.parametrize('deferred', [['unknown'], ['radionet', 'radionet'], 'radionet'])
def test_invalid_priority_configuration_is_rejected(deferred):
    with pytest.raises(ValueError):
        _worker_pending({'b_methods': ['radionet'], 'deferred_b_methods': deferred}, [], set())
