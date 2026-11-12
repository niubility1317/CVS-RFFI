import pytest

from experiments.pamf_physics_tri.source_tuning_plan import candidate_rows, validate_candidate, configuration, validate
from experiments.pamf_physics_tri.source_tuning_selection import select


def test_first_pilot_uses_one_seed_and_three_prototype_temperatures():
    rows = candidate_rows()
    assert len(rows) == 3
    assert {r['temperature'] for r in rows} == {0.1, 0.2, 0.4}
    assert {r['setting'] for r in rows} == {'ten_class'}
    assert {r['model_seed'] for r in rows} == {2026092701}
    for row in rows:
        validate_candidate(row)


def test_unregistered_candidate_is_rejected():
    row = dict(candidate_rows()[0], temperature=0.3)
    with pytest.raises(ValueError):
        validate_candidate(row)


def test_configuration_retains_partition_and_rejects_changed_temperature(monkeypatch):
    from experiments.pamf_paper_completion import plan as paper
    monkeypatch.setattr(paper, 'configuration', lambda ctx, row, project, commit: {
        'paper_row': row, 'source_contract': 'source.json', 'model_seed': row['model_seed'],
        'run_id': paper.RUN, 'row_id': row['row_id'], 'protocol_id': row['protocol_id']})
    monkeypatch.setattr(paper, 'validate_config', lambda c: c)
    row = candidate_rows()[0]
    partition = {'train_ids': ['a'], 'dev_ids': ['b'], 'seed': 20261112}
    cfg = configuration({}, row, '/project', 'commit', partition)
    assert cfg['source_development'] == partition
    assert cfg['source_prototype']['temperature'] == 0.1
    validate(cfg)
    cfg['source_prototype']['temperature'] = 0.4
    with pytest.raises(ValueError):
        validate(cfg)


def test_configuration_rejects_wrong_contract_or_output(monkeypatch):
    from experiments.pamf_paper_completion import plan as paper
    monkeypatch.setattr(paper, 'configuration', lambda ctx, row, project, commit: {
        'paper_row': row, 'source_contract': 'source.json', 'model_seed': row['model_seed'],
        'run_id': paper.RUN, 'row_id': row['row_id'], 'protocol_id': row['protocol_id']})
    monkeypatch.setattr(paper, 'validate_config', lambda c: c)
    cfg = configuration({}, candidate_rows()[0], '/project', 'commit',
                        {'train_ids': ['a'], 'dev_ids': ['b'], 'seed': 20261112})
    cfg['protocol_id'] = 'F0'
    with pytest.raises(ValueError):
        validate(cfg)
    cfg['protocol_id'] = 'ManyRx'
    cfg['output_root'] = '/project/runs/some-other-owner/source'
    with pytest.raises(ValueError):
        validate(cfg)


def test_selection_requires_all_source_scores_and_prefers_original_on_tie():
    rows = candidate_rows()
    scores = [{'row_id': r['row_id'], 'clean_accuracy': .8, 'physical_accuracy': .7} for r in rows]
    assert select(scores)['temperature'] == .2
    with pytest.raises(ValueError):
        select(scores[:-1])
    scores[0]['physical_accuracy'] = .8
    assert select(scores)['temperature'] == .1


def test_fresh_training_rejects_full_l_checkpoint_before_creating_output(monkeypatch, tmp_path):
    from experiments.pamf_physics_tri import source_tuning
    monkeypatch.setattr(source_tuning.plan, 'validate', lambda cfg: None)
    output = tmp_path / 'new-source'
    cfg = {'output_root': str(output), 'parent_source': 'old-full-l-checkpoint'}
    with pytest.raises(ValueError, match='start from scratch'):
        source_tuning.train(cfg)
    assert not output.exists()


def test_release_preflight_is_source_only_and_compiles(monkeypatch):
    from experiments.pamf_physics_tri import source_tuning_publish
    from experiments.pamf_branch_training import publish as base

    def capture(*args, **kwargs):
        script = base.REMOTE
        compile(script.replace('PACKAGE', '{}'), '<remote-preflight>', 'exec')
        assert "ctx['handles']['ManyRx']" in script
        assert "h['tests']" not in script and "ctx['caches']['F0']" not in script
        assert 'experiments.pamf_physics_tri.source_tuning_runner' in script
        return 'captured'

    monkeypatch.setattr(base, 'publish', capture)
    assert source_tuning_publish.publish('repo', 'out', 'context', 'ssh') == 'captured'


def test_preparation_registers_only_source_jobs_with_one_shared_partition(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    import numpy as np
    from experiments.cric_dura_comparison import f_data
    from experiments.pamf_paper_completion import plan as paper
    from experiments.pamf_physics_tri import source_tuning_runner

    class Source:
        def __init__(self, root, role, contract):
            assert role == 'L_s'
            self.metadata = {'ids': np.array([f'id-{i}' for i in range(300)]),
                             'y': np.repeat(np.arange(10), 30)}

    monkeypatch.setattr(f_data, 'SourceDataset', Source)
    monkeypatch.setattr(paper, 'configuration', lambda ctx, row, project, commit: {
        'paper_row': row, 'source_contract': 'source.json', 'model_seed': row['model_seed'],
        'run_id': paper.RUN, 'row_id': row['row_id'], 'protocol_id': row['protocol_id']})
    monkeypatch.setattr(paper, 'validate_config', lambda cfg: cfg)
    contract = tmp_path / 'contract.json'
    contract.write_text('{}', encoding='utf-8')
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'handles': {'ManyRx': {'source_cache': 'unused',
                                                          'source_contract': str(contract)}}}), encoding='utf-8')
    run_root = tmp_path / 'run'
    rt = {'run_root': str(run_root), 'context': str(context), 'project': str(tmp_path),
          'code_commit': 'abc', 'frozen_sources': {}}
    jobs = source_tuning_runner.prepare(rt)
    assert len(jobs) == 6 and {j['kind'] for j in jobs} == {'train', 'source_score'}
    configs = [json.loads(Path(j['config']).read_text(encoding='utf-8')) for j in jobs if j['kind'] == 'train']
    assert all(c['source_development'] == configs[0]['source_development'] for c in configs)
    assert all(not any(c.get(k) for k in ('truth', 'tests', 'target_inputs')) for c in configs)


def test_one_percent_plan_uses_its_own_data_and_run(monkeypatch):
    from experiments.pamf_paper_completion import plan as paper
    from experiments.pamf_physics_tri import source_tuning_plan as plan, source_tuning_label01
    monkeypatch.setattr(paper, 'configuration', lambda ctx, row, project, commit: {
        'paper_row': row, 'source_contract': 'label01.json', 'model_seed': row['model_seed'],
        'run_id': paper.RUN, 'row_id': row['row_id'], 'protocol_id': row['protocol_id']})
    monkeypatch.setattr(paper, 'validate_config', lambda cfg: cfg)
    with source_tuning_label01.configured():
        rows = plan.candidate_rows()
        assert len(rows) == 3 and {r['protocol_id'] for r in rows} == {'label01'}
        for row in rows:
            cfg = plan.configuration({}, row, '/project', 'commit',
                {'train_ids': ['a'], 'dev_ids': ['b'], 'seed': 20261112})
            assert cfg['paper_row']['data_setting'] == 'labeled_1pct'
            assert cfg['run_id'] == source_tuning_label01.RUN
            plan.validate(cfg)
        scores = [{'row_id': r['row_id'], 'clean_accuracy': .8, 'physical_accuracy': .7} for r in rows]
        assert select(scores)['temperature'] == .2


def test_one_percent_release_targets_only_its_own_source_and_workers(monkeypatch):
    from experiments.pamf_physics_tri import source_tuning_publish as publisher, source_tuning_label01
    from experiments.pamf_branch_training import publish as base

    def capture(*args, **kwargs):
        script = base.REMOTE
        compile(script.replace('PACKAGE', '{}'), '<remote-preflight>', 'exec')
        assert "ctx['handles']['label01']" in script
        assert 'labeled_1pct-factorial-ssl-full-permuted-s2026092701' in script
        assert 'experiments.pamf_physics_tri.source_tuning_label01' in script
        assert "rt['allowed_gpus']=[4, 5, 7]" in script
        assert base.RUN == source_tuning_label01.RUN
        return 'captured'

    monkeypatch.setattr(base, 'publish', capture)
    assert publisher.publish('repo', 'out', 'context', 'ssh', run_id=source_tuning_label01.RUN,
        protocol_id='label01', setting='labeled_1pct',
        worker_module='experiments.pamf_physics_tri.source_tuning_label01', gpus=(4, 5, 7)) == 'captured'
