from contextlib import contextmanager

import pytest

from experiments.pamf_physics_tri_unprotected import plan, scope, source


def test_registered_rows_are_three_matched_seeds():
    rows = plan.candidate_rows()
    assert [r['model_seed'] for r in rows] == [2026092701, 2026092702, 2026092703]
    assert all(len(r['views']) == 7 and all('weak' not in v for v in r['views']) for r in rows)
    with pytest.raises(ValueError):
        plan.configuration({}, {}, '/project', 'commit')


def test_unregistered_parent_is_rejected_before_access():
    with pytest.raises(ValueError):
        source.parent(dict(model_seed=2026092701, output_root='/bad', run_id='other', row_id='wrong'))


def test_scope_keeps_original_ssl_and_removes_only_identity_projection(monkeypatch):
    from experiments.cric_dura_comparison import t1_model
    calls = []

    @contextmanager
    def noop(*args, **kwargs):
        calls.append(kwargs)
        yield

    original = lambda model, cfg: ('original', model, cfg)
    monkeypatch.setattr(t1_model, 'install', original)
    monkeypatch.setattr(scope, 'indexed_source_scope', noop)
    monkeypatch.setattr(scope, 'original_tri', noop)
    monkeypatch.setattr(scope, 'install_labeled', lambda model, cfg, **kw: (model, kw))
    with scope.installed({'x': 1}, {}):
        result = t1_model.install('net', {'x': 1})
    assert calls == [{}, {'enabled': True}]
    assert result == (('original', 'net', {'x': 1}), {'protect_identity': False})
    assert t1_model.install is original
