import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.cvs_phase1_stack import completed_eval as e


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / 'release'; root.mkdir()
    source = tmp_path / 'source'
    monkeypatch.setattr(e, 'ROOT', root)
    rows = [dict(row_id=f'r2-a-s{i}', stage='r2', arm='a', model_seed=i) for i in range(32)]
    value = dict(run_id=e.RUN, status='FROZEN_COMPLETED_ROWS', rows=rows,
                 selection='ALL_COMPLETED_AT_SNAPSHOT', target_feedback_forbidden=True)
    e.write(root / 'frozen_rows.json', value)
    for c in rows:
        e.write(source / 'configs' / ('source-' + c['row_id'] + '.json'), c)
    return SimpleNamespace(BASE=str(source), validate=lambda c: None), value


def test_fixed_snapshot_exact_configs_and_no_metric_selection(frozen):
    d, value = frozen
    assert e.snapshot(d) == value['rows']
    c = dict(value['rows'][0], model_seed=999)
    e.write(Path(d.BASE) / 'configs' / ('source-' + c['row_id'] + '.json'), c)
    with pytest.raises(ValueError, match='Original source config'):
        e.snapshot(d)


@pytest.mark.parametrize('change', ['count', 'duplicate', 'selection', 'feedback', 'stage'])
def test_snapshot_rejects_changed_scope(frozen, change):
    d, value = frozen
    if change == 'count': value['rows'].pop()
    elif change == 'duplicate': value['rows'][-1] = value['rows'][0]
    elif change == 'selection': value['selection'] = 'BEST_TARGET_ACCURACY'
    elif change == 'feedback': value['target_feedback_forbidden'] = False
    else: value['rows'][0]['stage'] = 'r4'
    e.write(e.ROOT / 'frozen_rows.json', value)
    with pytest.raises(ValueError):
        e.snapshot(d)


@pytest.fixture
def source(tmp_path):
    c = dict(output_root=str(tmp_path / 'trained'), row_id='r2-bridge-s1')
    role = dict(schema='core90_game_source_roles_v1', num_classes=6, role_ids={'U': ['hidden-u']})
    e.write(tmp_path / 'roles.json', role)
    e.write(Path(c['output_root']) / 'source_contract.json', dict(role, native_role_comparison='EXACT_MATCH'))
    e.write(Path(c['output_root']) / 'completion.json', dict(status='SOURCE_TRAINED', config=c, epoch=200,
            logged_steps=44400, optimizer_steps=44400, target_access=False, target_evaluated=False))
    e.write(Path(c['output_root']) / 'initialization.json', dict(scratch_only=True, ancestors=[],
            checkpoint_sources=[], target_contact=False, source_roles='EXACT_MATCH'))
    def budget(logged, successful):
        if logged != 44400 or successful != 44400:
            raise ValueError('Incomplete budget')
    return SimpleNamespace(SOURCE=str(tmp_path / 'roles.json'), require_budget=budget), c


@pytest.mark.parametrize('change', [None, 'budget', 'target', 'ancestor', 'physical_ids'])
def test_completed_source_contract_and_provenance(source, change):
    d, c = source; root = Path(c['output_root'])
    if change in {'budget', 'target'}:
        p = root / 'completion.json'; value = e.read(p)
        if change == 'budget': value['optimizer_steps'] = 44399
        else: value['target_access'] = True
        e.write(p, value)
    elif change == 'ancestor':
        p = root / 'initialization.json'; value = e.read(p); value['ancestors'] = ['legacy']; e.write(p, value)
    elif change == 'physical_ids':
        p = root / 'source_contract.json'; value = e.read(p); value['role_ids']['U'] = ['other-u']; e.write(p, value)
    if change:
        with pytest.raises(ValueError): e.source_provenance(d, c)
    else:
        assert e.source_provenance(d, c) == root


@pytest.mark.parametrize('active_count,pids,free,expected', [(4, [], 20000, None), (3, [1, 2], 20000, None),
                         (0, [1], 11999, None), (0, [1], 12000, 2), (3, [], 24000, 2)])
def test_slot_enforces_four_workers_and_two_total_gpu_processes(active_count, pids, free, expected):
    active = {str(i): {} for i in range(active_count)}
    assert e.slot(active, lambda jobs: 2, lambda jobs: {2: dict(pids=set(pids), free_mb=free)}) == expected


@pytest.fixture
def predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'BASE', tmp_path / 'evaluation')
    monkeypatch.setattr(e, 'COUNT', 42)
    monkeypatch.setattr(e, 'VIEWS_ROOT', tmp_path / 'views')
    e.VIEWS_ROOT.mkdir()
    capsule = tmp_path / 'capsule'; capsule.mkdir()
    ids = np.asarray(['q' + str(i) for i in range(42)])
    np.savez(e.VIEWS_ROOT / 'index.npz', ids=ids); np.savez(capsule / 'index.npz', ids=ids)
    d = SimpleNamespace(CAPSULE=str(capsule), CLASSES=['tx' + str(i) for i in range(6)], TRUTH=str(tmp_path / 'truth.json'))
    e.write(e.VIEWS_ROOT / 'manifest.json', dict(status='VALIDATED_ONCE', count=42, scenes=list(e.SCENES),
            classes=d.CLASSES, truth_read=False, channel='residual/post_sync/noeq', source_capsule=d.CAPSULE,
            clean_ref=d.CAPSULE + '/clean.npy', rows_share_observations=True))
    rows = [dict(row_id='r3-a-s1', stage='r3', arm='a', model_seed=1),
            dict(row_id='r3-a-s2', stage='r3', arm='a', model_seed=2),
            dict(row_id='r3-b-s1', stage='r3', arm='b', model_seed=1)]
    truth = {rid: dict(label=i % 6, receiver=i % 7) for i, rid in enumerate(ids)}
    e.write(d.TRUTH, truth)
    for c in rows:
        out = e.BASE / c['row_id'] / 'prediction'
        e.write(out / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=42, views=list(e.VIEWS),
                truth_read=False, query_fit=False, inference_seconds={v: 1. for v in e.VIEWS}))
        e.write(out / 'provenance.json', dict(config=c))
        np.savez(out / 'predictions.npz', ids=ids, **{v: np.arange(42, dtype=np.int64) % 6 for v in e.VIEWS})
    monkeypatch.setattr(e, 'design', lambda: d)
    monkeypatch.setattr(e, 'snapshot', lambda design: rows)
    return d, rows, ids


@pytest.mark.parametrize('corruption', ['missing_row', 'missing_view', 'wrong_ids', 'invalid_class', 'config'])
def test_truth_stays_closed_until_every_frozen_row_and_view_is_valid(predictions, monkeypatch, corruption):
    d, rows, ids = predictions
    out = e.BASE / rows[-1]['row_id'] / 'prediction'
    if corruption == 'missing_row':
        (out / 'complete.json').unlink()
    elif corruption == 'config':
        e.write(out / 'provenance.json', dict(config={}))
    else:
        values = {v: np.zeros(42, dtype=np.int64) for v in e.VIEWS}
        if corruption == 'missing_view': values.pop(e.VIEWS[-1])
        elif corruption == 'wrong_ids': ids = ids[::-1]
        else: values[e.VIEWS[-1]][0] = 6
        np.savez(out / 'predictions.npz', ids=ids, **values)
    original = e.read
    def read(path):
        assert str(path) != d.TRUTH, 'Truth opened before complete prediction preflight'
        return original(path)
    monkeypatch.setattr(e, 'read', read)
    with pytest.raises((ValueError, FileNotFoundError)):
        e.score()


def test_score_complete_strata_independent_recount_actual_seed_counts(predictions):
    d, rows, ids = predictions
    e.score()
    result = e.read(e.BASE / 'scores.json')['results']
    assert len(result) == 3 * 7 * (1 + 7 + 6)
    assert all(r['accuracy'] == 1. for r in result)
    for r in result:
        assert np.trace(r['confusion']) == r['query_count']
    summary = e.read(e.BASE / 'summary.json')['results']
    assert len(summary) == 14
    for r in summary:
        assert r['seed_count'] == (2 if r['arm'] == 'a' else 1)
        assert r['accuracy_sd'] == (0. if r['arm'] == 'a' else None)
    done = e.read(e.BASE / 'scoring_complete.json')
    assert done['truth_last'] and done['independent_recount'] == 'VERIFIED'
    assert done['prediction_count'] == 3 * 7 * 42
    assert 'N/A' in (e.BASE / 'analysis.md').read_text(encoding='utf-8')


def test_unknown_snapshot_row_cannot_open_query(monkeypatch):
    monkeypatch.setattr(e, 'design', lambda: object())
    monkeypatch.setattr(e, 'snapshot', lambda d: [])
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Query read for unknown row'))
    with pytest.raises(ValueError, match='not in fixed'):
        e.predict('not-authorized')
