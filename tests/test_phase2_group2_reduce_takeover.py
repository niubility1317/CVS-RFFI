import pytest
import json

from experiments.pamf_phase2_practical import group2_c_backbone_reduce_takeover as reduce


def runtimes():
    old = {key: key for key in reduce.SCIENCE_KEYS}
    old.update(gpu_limits={str(i): 3 if i < 7 else 2 for i in range(8)},
               minimum_free_MiB=6000, excluded_failed_indices=[36248, 36266],
               deferred_b_methods=['radionet_ada_knn'], release_root='/release', runtime_file='/release/runtime.json')
    new = dict(old, gpu_limits={str(i): 2 for i in range(8)})
    return old, new


def test_reduce_accepts_matching_science_and_strict_cap():
    old, new = runtimes()
    reduce.validate_parent(old, new)


@pytest.mark.parametrize('change', ['cap', 'quarantine', 'science', 'free'])
def test_reduce_rejects_incompatible_policy_or_inputs(change):
    old, new = runtimes()
    if change == 'cap':
        new['gpu_limits']['0'] = 3
    elif change == 'quarantine':
        new['excluded_failed_indices'] = [36248]
    elif change == 'science':
        new['model_seeds'] = [9]
    else:
        new['minimum_free_MiB'] = 0
    with pytest.raises(ValueError):
        reduce.validate_parent(old, new)


def test_owner_requires_pid_start_cwd_module_and_runtime():
    old, _ = runtimes()
    info = dict(pid=9, start_ticks=10, cwd='/release', argv=['python', '-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_resume', '--runtime', '/release/runtime.json'])
    assert reduce.owner_matches(info, old, 9, 10)
    assert not reduce.owner_matches(info, old, 9, 11)
    assert not reduce.owner_matches(dict(info, cwd='/other'), old, 9, 10)
    assert not reduce.owner_matches(dict(info, argv=['python']), old, 9, 10)


def test_quarantine_is_retained_in_grandparent(tmp_path):
    old, _ = runtimes()
    old['excluded_failed_indices'] = [0, 1]
    grandparent, parent, current = (tmp_path / name for name in ('r06', 'r07', 'r08'))
    for root in (grandparent, parent, current):
        root.mkdir()
    def save(path, data):
        path.write_text(json.dumps(data), encoding='utf-8')
    r06_rt = tmp_path/'r06.json'
    r07_rt = tmp_path/'r07.json'
    save(r06_rt, {'run_root': str(grandparent)})
    save(r07_rt, {'run_root': str(parent), 'parent_runtime_file': str(r06_rt)})
    for root, predecessor in ((parent, grandparent), (current, parent)):
        save(root/'resume.json', {'status': 'RESUMED', 'excluded': [0, 1], 'parent_root': str(predecessor)})
    old.update(run_root=str(current), parent_runtime_file=str(r07_rt))
    jobs = [{'key': 'a'}, {'key': 'b'}]
    for job in jobs:
        partial = grandparent/'predictions'/job['key']
        partial.mkdir(parents=True)
        (partial/'partial.npz').write_bytes(b'prediction')
    reduce.verify_quarantine(old, jobs)
    (parent/'predictions'/'a').mkdir(parents=True)
    with pytest.raises(ValueError):
        reduce.verify_quarantine(old, jobs)


def test_conflicting_successor_still_pauses_parent_first(tmp_path, monkeypatch):
    old, rt = runtimes()
    oldroot = tmp_path/'old'
    oldroot.mkdir()
    successor = tmp_path/'occupied'
    successor.mkdir()
    old.update(run_root=str(oldroot))
    rt.update(run_root=str(successor), parent_runtime_file='parent',
              parent_controller_pid=9, parent_start_ticks=10)
    info = dict(pid=9, start_ticks=10, cwd='/release', state='T', argv=['python', '-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_resume', '--runtime', '/release/runtime.json'])
    signals = []
    monkeypatch.setattr(reduce, 'validate_runtime', lambda _: None)
    monkeypatch.setattr(reduce, '_proc', lambda _: info)
    monkeypatch.setattr(reduce.signal, 'SIGSTOP', 19, raising=False)
    monkeypatch.setattr(reduce.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(reduce, 'read', lambda path: old if str(path) == 'parent' else
                        {'status': 'RUNNING', 'failed': [], 'excluded': 2})
    with pytest.raises(ValueError):
        reduce.dispatch(rt)
    assert signals == [(9, reduce.signal.SIGSTOP)]
    assert not list(successor.iterdir())
