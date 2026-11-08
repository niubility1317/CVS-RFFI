import pytest

from experiments.pamf_phase2_practical import group2_resume


def test_user_cancelled_radio_is_excluded_without_removing_other_methods():
    jobs = [{'profile': 'radionet_ada_knn', 'key': 'r'},
            {'profile': 'dadda_sda', 'key': 'd'},
            {'profile': 'protonet_cda', 'key': 'p'}]
    pending, cancelled = group2_resume.separate_cancelled({'cancelled_profiles': ['radionet_ada_knn']}, jobs)
    assert [x['key'] for x in pending] == ['d', 'p']
    assert [x['key'] for x in cancelled] == ['r']


def test_cancelled_scope_cannot_silently_exclude_other_methods():
    with pytest.raises(ValueError):
        group2_resume.separate_cancelled({'cancelled_profiles': ['dadda_sda']}, [])


def test_reuses_closed_jobs_skips_exact_failed_job_and_refuses_other_partial(tmp_path, monkeypatch):
    previous=tmp_path/'previous'; (previous/'predictions'/'done').mkdir(parents=True)
    (previous/'predictions'/'failed').mkdir()
    (previous/'predictions'/'done'/'complete.json').write_text('{}')
    jobs=[{'key':name,'output':str(tmp_path/'new'/name),'points':[(1,2)],'methods':['lora']}
          for name in ('done','failed','new')]
    monkeypatch.setattr(group2_resume, 'tasks', lambda rt: jobs)
    checked=[]
    monkeypatch.setattr(group2_resume, 'closed_prediction', lambda job, expected: checked.append((job['key'],expected)))
    monkeypatch.setattr(group2_resume, 'verify_reuse_binding', lambda rt, job: None)
    reused,deferred,pending=group2_resume.classify({}, previous, {'failed'})
    assert [v['key'] for v in reused]==['done']
    assert [v['key'] for v in deferred]==['failed']
    assert [v['key'] for v in pending]==['new']
    assert reused[0]['output']==str(previous/'predictions'/'done')
    assert checked==[('done',1)]
    (previous/'predictions'/'new').mkdir()
    with pytest.raises(ValueError,match='Unreconciled partial'):
        group2_resume.classify({}, previous, {'failed'})


def test_resume_rejects_changed_scientific_overrides():
    old={'data_root':'d', 'native_adaptation_overrides':{'dadda_sda':{'epochs':100}}, 'run_root':'old'}
    group2_resume.require_same_method(old, dict(old,run_root='new',previous_run_root='old',
                                                last_profiles=['radionet_ada_knn']))
    changed=dict(old,run_root='new',native_adaptation_overrides={'dadda_sda':{'epochs':1}})
    with pytest.raises(ValueError,match='identity differs'):
        group2_resume.require_same_method(old, changed)


def test_priority_only_reorders_unstarted_jobs():
    rt = {'profiles': ['proto', 'mrior', 'hu', 'radio'], 'last_profiles': ['proto', 'radio']}
    pending = [{'key': key, 'profile': profile} for key, profile in (
        ('proto-1', 'proto'), ('radio-1', 'radio'), ('mrior-1', 'mrior'),
        ('hu-1', 'hu'), ('mrior-2', 'mrior'), ('proto-2', 'proto'))]
    ordered = group2_resume.prioritize_pending(rt, pending)
    assert ordered == [pending[i] for i in (2, 3, 4, 0, 5, 1)]
    assert group2_resume.prioritize_pending({}, pending) == pending
    with pytest.raises(ValueError, match='Invalid last profiles'):
        group2_resume.prioritize_pending(dict(rt, last_profiles=['missing']), pending)


def test_chain_inventory_preserves_both_ancestors_and_quarantines_partials(tmp_path, monkeypatch):
    roots = [tmp_path/'r02', tmp_path/'r03']
    for root in roots:
        (root/'predictions').mkdir(parents=True)
    for root, key, complete in ((roots[0], 'old_done', True), (roots[1], 'new_done', True),
                                (roots[0], 'old_failure', False), (roots[1], 'new_failure', False)):
        out = root/'predictions'/key
        out.mkdir()
        (out/('complete.json' if complete else 'partial.npz')).write_text('x')
    jobs = [{'key': key, 'output': str(tmp_path/'next'/key), 'points': [(1, 2)], 'methods': ['lora']}
            for key in ('old_done', 'new_done', 'old_failure', 'new_failure', 'unstarted')]
    monkeypatch.setattr(group2_resume, 'tasks', lambda rt: jobs)
    monkeypatch.setattr(group2_resume, 'closed_prediction', lambda *args: None)
    monkeypatch.setattr(group2_resume, 'verify_reuse_binding', lambda *args: None)
    deferred = {'old_failure', 'new_failure'}
    reused, failures, pending = group2_resume.classify_chain({}, roots, deferred)
    assert [j['key'] for j in reused] == ['old_done', 'new_done']
    assert [j['key'] for j in failures] == ['old_failure', 'new_failure']
    assert [j['key'] for j in pending] == ['unstarted']
    assert failures[0]['output'].startswith(str(roots[0]))
    assert failures[1]['output'].startswith(str(roots[1]))
    (roots[1]/'predictions'/'old_done').mkdir()
    with pytest.raises(ValueError, match='Multiple ancestor outputs'):
        group2_resume.classify_chain({}, roots, deferred)


def test_chain_inventory_rejects_unrecorded_partial(tmp_path, monkeypatch):
    root = tmp_path/'r03'
    (root/'predictions'/'unexpected').mkdir(parents=True)
    jobs = [{'key': 'unexpected', 'output': 'unused', 'points': [(1, 2)], 'methods': ['lora']}]
    monkeypatch.setattr(group2_resume, 'tasks', lambda rt: jobs)
    with pytest.raises(ValueError, match='Unreconciled partial'):
        group2_resume.classify_chain({}, [root], set())


def test_dispatch_follows_parent_chain_before_selecting_pending(tmp_path, monkeypatch):
    r02, r03 = tmp_path/'r02', tmp_path/'r03'
    for root in (r02, r03):
        (root/'launches').mkdir(parents=True)
    old_runtime = str(tmp_path/'r02.json')
    parent_runtime = str(tmp_path/'r03.json')
    base = {'data_root': 'data'}
    r02_rt = dict(base, run_root=str(r02), runtime_file=old_runtime)
    r03_rt = dict(base, run_root=str(r03), runtime_file=parent_runtime,
                  previous_run_root=str(r02), previous_runtime_file=old_runtime,
                  deferred_failure_keys=['failed_r02'])
    current = dict(base, run_root=str(tmp_path/'r04'), previous_run_root=str(r03),
                   previous_runtime_file=parent_runtime,
                   deferred_failure_keys=['failed_r02', 'failed_r03'])
    records = {old_runtime: r02_rt, parent_runtime: r03_rt,
               str(r02/'owner.json'): {}, str(r03/'owner.json'): {},
               str(r02/'worker_state.json'): {'status': 'FAILED', 'active': [],
                                              'failed': [{'key': 'failed_r02'}]},
               str(r03/'worker_state.json'): {'status': 'FAILED', 'active': [],
                                              'failed': [{'key': 'failed_r03'}]}}
    monkeypatch.setattr(group2_resume, 'validate_runtime', lambda rt: None)
    monkeypatch.setattr(group2_resume, 'read', lambda path: records[str(path)])
    seen = []
    def classify(rt, roots, keys):
        seen.append((list(roots), keys))
        raise RuntimeError('stop before creating output')
    monkeypatch.setattr(group2_resume, 'classify_chain', classify, raising=False)
    with pytest.raises(RuntimeError, match='stop before creating output'):
        group2_resume.dispatch(current)
    assert seen == [([r02, r03], {'failed_r02', 'failed_r03'})]


def test_chain_inventory_requires_all_ancestor_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(group2_resume, 'tasks', lambda rt: [])
    with pytest.raises(ValueError, match='Ancestor predictions directory missing'):
        group2_resume.classify_chain({}, [tmp_path/'missing'], set())


def test_chain_inventory_preserves_known_pre_prediction_failure(tmp_path, monkeypatch):
    root = tmp_path/'r02'
    (root/'predictions'/'failure').mkdir(parents=True)
    monkeypatch.setattr(group2_resume, 'tasks', lambda rt: [
        {'key': 'failure', 'output': 'unused', 'points': [(1, 2)], 'methods': ['lora']}])
    reused, deferred, pending = group2_resume.classify_chain({}, [root], {'failure'})
    assert not reused and not pending and deferred[0]['key'] == 'failure'
