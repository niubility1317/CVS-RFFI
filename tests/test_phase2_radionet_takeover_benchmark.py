import pytest

from experiments.pamf_phase2_practical import group2_c_backbone_fast_takeover as takeover


def test_benchmark_gpu_selection_respects_live_parent_capacity(monkeypatch):
    rt = {'gpu_limits': {'0': 3, '7': 2}, 'minimum_free_MiB': 6000}
    monkeypatch.setattr(takeover, '_gpu_free', lambda gpu: 10000)
    children = {0: {'pid': 10, 'start_ticks': 1}, 1: {'pid': 11, 'start_ticks': 2},
                2: {'pid': 12, 'start_ticks': 3}, 3: {'pid': 13, 'start_ticks': 4}}
    assigned = {0: '0', 1: '0', 2: '0', 3: '7'}
    monkeypatch.setattr(takeover, '_proc', lambda pid: {'start_ticks': pid-9, 'state': 'R'})
    assert takeover._benchmark_free_gpu(rt, children, assigned) == '7'
    monkeypatch.setattr(takeover, '_gpu_free', lambda gpu: 5999)
    assert takeover._benchmark_free_gpu(rt, children, assigned) is None


def test_benchmark_rejects_no_gain_or_bad_equivalence():
    good = {'status': 'PASS', 'same_gradient': True, 'rng_unchanged': True,
            'old_seconds': 100., 'new_seconds': 90., 'source_only': True, 'query_read': False}
    takeover._require_radionet_gain(good)
    for change in ({'new_seconds': 96.}, {'same_gradient': False}, {'query_read': True},
                   {'old_seconds': 0.}, {'new_seconds': float('nan')}):
        with pytest.raises(ValueError):
            takeover._require_radionet_gain(dict(good, **change))
