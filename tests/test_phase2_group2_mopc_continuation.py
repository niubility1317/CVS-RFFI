import json

import pytest

from experiments.pamf_phase2_practical import group2_mopc_continuation as continuation


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path, monkeypatch):
    parent = tmp_path/'r11'
    (parent/'jobs').mkdir(parents=True)
    original = dict(key='p', profile='protonet_cda', seed=7, capsule='capsules/P/M/R',
                    methods=['mopc_hr', 'orthogonal', 'lora'], points=[(5, 20)], output=str(tmp_path/'r12/predictions/p'))
    subset = dict(original, methods=['orthogonal', 'lora'], points=[[5, 20]],
                  save_B_for_mopc=True, output=str(parent/'predictions/p'))
    write(parent/'jobs/00000.json', subset)
    other = [dict(original, key=key) for key in ('old', 'failed', 'radio')]
    monkeypatch.setattr(continuation, 'tasks', lambda rt: [original]+other)
    seen = []
    monkeypatch.setattr(continuation, 'closed_prediction', lambda job, count: seen.append((job, count)))
    manifest = dict(scope='ORTHOGONAL_LORA_ONLY_MOPC_PENDING', truth_read=False, query_fit=False,
                    reused=[dict(key='old')], deferred=[dict(key='failed')],
                    cancelled=[dict(key='radio')], pending_count=1)
    return parent, original, subset, manifest, seen


def test_only_completed_subset_reuses_same_b_and_original_points(tmp_path, monkeypatch):
    parent, original, subset, manifest, seen = fixture(tmp_path, monkeypatch)
    jobs = continuation.saved_b_jobs({}, parent, manifest, [parent/'jobs/00000.json'])
    assert len(jobs) == 1 and jobs[0]['key'] == 'p' and jobs[0]['methods'] == ['mopc_hr']
    assert jobs[0]['load_B_for_mopc'] == subset['output'] and jobs[0]['trace_registration'] is True
    assert jobs[0]['output'] == original['output'] and seen == [(subset, 2)]
    assert not subset.get('load_B_for_mopc')


def test_quarantined_unknown_or_changed_subset_cannot_enter_queue(tmp_path, monkeypatch):
    parent, _, subset, manifest, _ = fixture(tmp_path, monkeypatch)
    for change in (dict(key='failed'), dict(profile='radionet_ada_knn'),
                   dict(points=[[1, 20]]), dict(output=str(tmp_path/'unrelated'))):
        write(parent/'jobs/00000.json', dict(subset, **change))
        with pytest.raises(ValueError):
            continuation.saved_b_jobs({}, parent, manifest, [parent/'jobs/00000.json'])
    with pytest.raises(ValueError, match='incomplete'):
        continuation.saved_b_jobs({}, parent, manifest, [])


def test_duplicate_subset_and_overlapping_prior_categories_rejected(tmp_path, monkeypatch):
    parent, _, subset, manifest, _ = fixture(tmp_path, monkeypatch)
    write(parent/'jobs/00001.json', subset)
    with pytest.raises(ValueError, match='Duplicate'):
        continuation.saved_b_jobs({}, parent, manifest, sorted((parent/'jobs').glob('*.json')))
    manifest['deferred'].append(dict(key='old'))
    with pytest.raises(ValueError, match='overlap'):
        continuation.saved_b_jobs({}, parent, manifest, [parent/'jobs/00000.json'])


def test_only_empty_pre_registration_identity_failure_is_recoverable(tmp_path):
    root = tmp_path/'r12'
    for name in ('jobs', 'logs', 'predictions'):
        (root/name).mkdir(parents=True)
    output = root/'predictions/p'
    output.mkdir()
    write(root/'jobs/00000.json', dict(key='p', output=str(output)))
    log = root/'logs/worker-00000.log'
    log.write_text('ValueError: B/C old physical support differs\n', encoding='utf-8')
    state = dict(status='FAILED', active=[], completed=0, failed=[dict(index=0, key='p')])
    manifest = dict(scope='MOPC_ONLY_FROM_SAVED_B')
    continuation.require_identity_failure_only(root, state, manifest)
    (output/'C-K1-N2-mopc_hr.npz').touch()
    with pytest.raises(ValueError, match='not before MoPC'):
        continuation.require_identity_failure_only(root, state, manifest)
    (output/'C-K1-N2-mopc_hr.npz').unlink()
    log.write_text('REGISTRATION_FAILURE\nValueError: B/C old physical support differs\n', encoding='utf-8')
    with pytest.raises(ValueError, match='not before MoPC'):
        continuation.require_identity_failure_only(root, state, manifest)
    log.write_text('ValueError: B/C old physical support differs\n', encoding='utf-8')
    (root/'predictions/unclassified').mkdir()
    with pytest.raises(ValueError, match='unclassified'):
        continuation.require_identity_failure_only(root, state, manifest)
