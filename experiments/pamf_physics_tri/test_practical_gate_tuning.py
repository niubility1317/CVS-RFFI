import pytest

from experiments.pamf_physics_tri.practical_gate import required_votes


def test_original_gate_retains_four_votes_and_declared_variants_are_valid():
    assert required_votes({'mode': 'branch'}) == 4
    assert required_votes({'mode': 'branch', 'required_votes': 3}) == 3
    assert required_votes({'mode': 'branch', 'required_votes': 5}) == 5
    with pytest.raises(ValueError):
        required_votes({'mode': 'branch', 'required_votes': 6})


def test_gate_pilot_changes_only_vote_threshold(monkeypatch):
    from experiments.pamf_paper_completion import plan as paper
    from experiments.pamf_physics_tri import source_tuning_gate, source_tuning_plan as plan
    monkeypatch.setattr(paper, 'configuration', lambda ctx, row, project, commit: {
        'paper_row': row, 'source_contract': 'source.json', 'model_seed': row['model_seed'],
        'run_id': paper.RUN, 'row_id': row['row_id'], 'protocol_id': row['protocol_id']})
    monkeypatch.setattr(paper, 'validate_config', lambda cfg: cfg)
    with source_tuning_gate.configured():
        row = plan.candidate_rows()[0]
        cfg = plan.configuration({}, row, '/project', 'commit',
                                 {'train_ids': ['a'], 'dev_ids': ['b'], 'seed': 20261112})
        assert cfg['source_prototype']['temperature'] == .2
        assert cfg['practical_gate']['required_votes'] == 3
        plan.validate(cfg)
        cfg['practical_gate']['required_votes'] = 4
        with pytest.raises(ValueError):
            plan.validate(cfg)


def test_gate_prefix_must_have_same_source_development_partition(monkeypatch):
    from experiments.pamf_branch_training import source as parent_source
    from experiments.pamf_paper_completion import plan as paper
    from experiments.pamf_physics_tri import source_tuning_gate, source_tuning_plan as plan
    monkeypatch.setattr(paper, 'configuration', lambda ctx, row, project, commit: {
        'paper_row': row, 'source_contract': 'source.json', 'model_seed': row['model_seed'],
        'run_id': paper.RUN, 'row_id': row['row_id'], 'protocol_id': row['protocol_id']})
    monkeypatch.setattr(paper, 'validate_config', lambda cfg: cfg)
    with source_tuning_gate.configured():
        cfg = plan.configuration({}, plan.candidate_rows()[0], '/project', 'commit',
                                 {'train_ids': ['a'], 'dev_ids': ['b'], 'seed': 20261112})
        monkeypatch.setattr(parent_source, 'checked_e20_parent', lambda _: {'config': {
            'source_contract': 'source.json',
            'source_development': {'train_ids': ['b'], 'dev_ids': ['a'], 'seed': 20261112}}})
        with pytest.raises(ValueError, match='CHECKPOINT_DATA_CONTRACT_MISMATCH'):
            source_tuning_gate.parent(cfg)


def test_gate_comparison_uses_matched_source_results(tmp_path):
    import json
    from experiments.pamf_physics_tri import source_tuning_gate as gate

    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    root = tmp_path / 'runs' / gate.RUN
    baseline = tmp_path / 'runs' / gate.BASELINE_RUN
    rid = gate.rows()[0]['row_id']
    cfg = {'source_contract': 'source.json', 'source_development': {'train_ids': ['a'], 'dev_ids': ['b']}}
    put(root / 'F' / rid / 'source' / 'complete.json', {'config': cfg})
    put(baseline / 'F' / gate.BASELINE_ROW / 'source' / 'complete.json', {'config': cfg})
    put(root / 'source_scores' / rid / 'result.json', {'count': 100, 'clean_accuracy': .9, 'physical_accuracy': .8})
    put(baseline / 'source_scores' / gate.BASELINE_ROW / 'result.json', {'count': 100, 'clean_accuracy': .8, 'physical_accuracy': .8})
    gate.analyze({'run_root': str(root), 'project': str(tmp_path)})
    result = json.loads((root / 'analysis' / 'complete.json').read_text())
    assert result['source_pilot_pass'] and result['source_score_improvement'] == pytest.approx(.05)
    assert result['target_reads'] == 0


def test_prototype_weight_scales_only_the_prototype_term():
    import torch
    from experiments.pamf_physics_tri.source_prototype import weighted_prototype
    component = torch.tensor(2., requires_grad=True)
    result = weighted_prototype({'branch': component}, {'weight': .5})
    result.backward()
    assert result.item() == 1.
    assert component.grad.item() == .5
    assert weighted_prototype({'branch': component.detach()}, {}).item() == 2.
