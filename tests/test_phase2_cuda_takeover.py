import json
import pytest

from experiments.pamf_phase2_practical.cuda_takeover import select_job, resolve_jobs, _check_pending


def _job(root, kind, original, dependencies=()):
    out = root/original
    return dict(id='replacement-'+original, origin='replacement', kind=kind,
        original_id=original, dependencies=list(dependencies), output=str(out),
        completion=str(out/'complete.json'),
        status='PREDICTIONS_COMPLETE' if kind == 'predict' else 'SCORED_COMPLETE')


def _complete(job):
    from pathlib import Path
    path = Path(job['completion'])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': job['status']}), encoding='utf-8')


def test_score_uses_its_actual_prediction_root(tmp_path):
    old_pred = _job(tmp_path/'old', 'predict', 'predict-pamf')
    new_pred = _job(tmp_path/'new', 'predict', 'predict-pamf')
    old_score = _job(tmp_path/'old', 'score', 'score-predict-pamf', [old_pred['id']])
    new_score = _job(tmp_path/'new', 'score', 'score-predict-pamf')
    old_by = {old_pred['id']: old_pred}
    new_by = {new_pred['original_id']: new_pred, new_score['original_id']: new_score}
    assert select_job(old_pred, old_by, new_by) is new_pred
    with pytest.raises(ValueError, match='exactly one'):
        select_job(old_score, old_by, new_by)
    _complete(old_pred)
    assert select_job(old_score, old_by, new_by) is old_score
    _complete(new_pred)
    with pytest.raises(ValueError, match='exactly one'):
        select_job(old_score, old_by, new_by)


def test_score_routes_new_prediction_to_new_root(tmp_path):
    old_pred = _job(tmp_path/'old', 'predict', 'predict-pamf')
    new_pred = _job(tmp_path/'new', 'predict', 'predict-pamf')
    old_score = _job(tmp_path/'old', 'score', 'score-predict-pamf', [old_pred['id']])
    new_score = _job(tmp_path/'new', 'score', 'score-predict-pamf')
    _complete(new_pred)
    assert select_job(old_score, {old_pred['id']: old_pred},
        {new_pred['original_id']: new_pred, new_score['original_id']: new_score}) is new_score


def test_external_work_and_old_source_remain_unchanged():
    external = dict(origin='legacy', kind='predict')
    source = dict(origin='replacement', kind='source')
    assert select_job(external, {}, {}) is external
    assert select_job(source, {}, {}) is source


@pytest.mark.parametrize('predecessor_state,expected', [('done', 'original'), ('active', 'original'),
                                                         ('pending', 'cuda')])
def test_plan_persists_correct_prediction_and_score_paths(tmp_path, predecessor_state, expected):
    old_pred = _job(tmp_path/'old', 'predict', 'predict-pamf')
    new_pred = _job(tmp_path/'new', 'predict', 'predict-pamf')
    old_score = _job(tmp_path/'old', 'score', 'score-predict-pamf', [old_pred['id']])
    new_score = _job(tmp_path/'new', 'score', 'score-predict-pamf')
    old_score['prediction'] = old_pred['output']
    new_score['prediction'] = new_pred['output']
    new_by = {new_pred['original_id']: new_pred, new_score['original_id']: new_score}
    done = {old_pred['id']} if predecessor_state == 'done' else set()
    active = {old_pred['id']: {}} if predecessor_state == 'active' else {}
    rt = {'runtime_file': 'new-runtime', 'release_root': 'new-release'}
    resolved = resolve_jobs([old_pred, old_score], done, active, new_by, rt)
    assert resolved[0]['execution_version'] == expected
    assert resolved[1]['execution_version'] == expected
    assert resolved[1]['prediction'] == resolved[0]['output']


@pytest.mark.parametrize('artifact', ['log', 'output'])
def test_unowned_started_job_is_never_relaunched(tmp_path, artifact):
    old = {'run_root': str(tmp_path/'old')}
    legacy = {'run_root': str(tmp_path/'legacy')}
    job = _job(tmp_path/'old'/'predictions', 'predict', 'predict-pamf')
    _check_pending([job], old, legacy)
    from pathlib import Path
    if artifact == 'log':
        path = Path(old['run_root'])/'logs'/(job['original_id']+'.log')
        path.parent.mkdir(parents=True)
        path.touch()
    else:
        Path(job['output']).mkdir(parents=True)
    with pytest.raises(ValueError, match='Unowned partial'):
        _check_pending([job], old, legacy)
