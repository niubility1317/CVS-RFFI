import pytest

from experiments.pamf_physics_route import paired_report
from experiments.pamf_physics_tri import validation_plan


def test_multi_protocol_paired_summary_does_not_mix_same_seeds(monkeypatch):
    prepared = {
        f'{pid}-{seed}': ({'protocol_id': pid, 'model_seed': seed}, {})
        for pid in ('F1', 'F2', 'label05') for seed in (1, 2, 3)
    }
    monkeypatch.setattr(paired_report, 'prepare_row', lambda root, row: prepared[row])
    monkeypatch.setattr(paired_report, 'summarize', lambda values: [cfg['model_seed'] for cfg, _ in values])
    result = paired_report.run([('/run', row) for row in prepared])
    assert result['status'] == 'PAIRED_BY_PROTOCOL_COMPLETE'
    assert result['protocols'] == {'F1': [1, 2, 3], 'F2': [1, 2, 3], 'label05': [1, 2, 3]}


def test_paper_smoke_checks_three_view_graph_without_changing_config(monkeypatch):
    from experiments.pamf_repl_view_shuffle import source

    monkeypatch.setattr(source, 'assert_structure', lambda model, cfg: cfg['view_comparison']['view_count'])
    c = {'paper_row': {'signature_views': 3, 'permuted': True}}
    assert validation_plan.check_graph(object(), c) == 3
    assert 'view_comparison' not in c
    with pytest.raises(ValueError):
        validation_plan.check_graph(object(), {'paper_row': {'signature_views': 1, 'permuted': True}})
