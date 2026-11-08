import json

import pytest

from experiments.pamf_phase3_restoration import fixed_prediction_score_recovery as recovery


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf8')


def prepare(tmp_path, monkeypatch):
    c = recovery.CASES['affine']
    run = tmp_path / 'runs' / c['parent']
    row = run / 'rows' / c['arm']
    pred = row / 'prediction'
    identity = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100,
        model_seed=2026092701, source_run=recovery.main_eval.F_RUN,
        source_row=recovery.main_eval.F_ROW, source_code_commit=recovery.main_eval.F_COMMIT)
    put(pred / 'complete.json', dict(status='ALL_PREDICTIONS_COMPLETE', run_id=c['parent'],
        row_id=c['arm'], views=list(recovery.main_eval.VIEWS), readouts=list(recovery.STREAMS),
        truth_read=False, query_fit=False, code_commit=c['commit']))
    for task, count in recovery.main_eval.COUNTS.items():
        put(pred / task / 'complete.json', dict(status='PREDICTIONS_COMPLETE', run_id=c['parent'],
            arm=c['arm'], task=task, count=count, code_commit=c['commit'], readouts=list(recovery.STREAMS),
            views=list(recovery.main_eval.VIEWS), truth_read=False, query_fit=False, model_state_updates=0,
            F0_identity=identity))
        (pred / task / 'predictions.npz').write_bytes(b'fixture')
    put(run / ('terminal-' + c['arm'] + '-score.json'), dict(exit_code=1))
    log = run / 'logs' / (c['arm'] + '-score.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("ModuleNotFoundError: No module named 'experiments.pamf_repl_view_shuffle'", encoding='utf8')
    return c, row, pred


def test_frozen_original_predictions_accepted(tmp_path, monkeypatch):
    _, _, pred = prepare(tmp_path, monkeypatch)
    assert recovery.frozen_predictions(tmp_path, 'affine') == pred


@pytest.mark.parametrize('field,value', [('truth_read', True), ('query_fit', True),
                                         ('code_commit', 'wrong'), ('views', ['clean'])])
def test_incomplete_or_contaminated_matrix_rejected(tmp_path, monkeypatch, field, value):
    _, _, pred = prepare(tmp_path, monkeypatch)
    path = pred / 'complete.json'; obj = json.loads(path.read_text(encoding='utf8'))
    obj[field] = value; put(path, obj)
    with pytest.raises(ValueError, match='frozen before truth'):
        recovery.frozen_predictions(tmp_path, 'affine')


def test_existing_score_and_other_failure_not_retried(tmp_path, monkeypatch):
    c, row, _ = prepare(tmp_path, monkeypatch)
    put(row / 'score' / 'scores.json', {})
    with pytest.raises(ValueError, match='reconciliation'):
        recovery.frozen_predictions(tmp_path, 'affine')
    (row / 'score' / 'scores.json').unlink()
    log = tmp_path / 'runs' / c['parent'] / 'logs' / (c['arm'] + '-score.log')
    log.write_text('different failure', encoding='utf8')
    with pytest.raises(ValueError, match='missing-module'):
        recovery.frozen_predictions(tmp_path, 'affine')
