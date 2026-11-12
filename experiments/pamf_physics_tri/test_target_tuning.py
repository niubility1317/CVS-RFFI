import pytest

from experiments.pamf_physics_tri.target_tuning_plan import candidate_rows, baseline_row, configuration, validate


def test_fixed_matrix_uses_three_matched_seeds_and_only_changed_parameters():
    rows = candidate_rows()
    assert len(rows) == 33
    assert {r['setting'] for r in rows} == {'ten_class', 'labeled_1pct', 'source_rx_6'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}
    assert len({r['row_id'] for r in rows}) == len(rows)
    assert {r['temperature'] for r in rows if r['setting'] == 'source_rx_6'} == {.05, .1, .4, .8}
    assert {r['prototype_weight'] for r in rows if r['prototype_weight'] is not None} == {.5, 2.}
    assert any(r.get('required_votes') == 3 for r in rows)
    assert all(baseline_row(r)['arm'] == 'branch_prototype' for r in rows)


def test_candidate_config_changes_only_declared_parameters_and_rejects_holdout(monkeypatch):
    from experiments.pamf_physics_tri import full_matrix_plan as original
    monkeypatch.setattr(original, 'configuration', lambda ctx, old, project, commit: {
        'source_contract': 'full-source.json', 'model_seed': old['model_seed'],
        'protocol_id': old['protocol_id'], 'paper_row': {'row_id': 'original'},
        'shared_prefix': '/original/e20', 'practical_gate': {'mode': 'branch'},
        'source_prototype': {'mode': 'branch'}})
    monkeypatch.setattr(original, 'validate', lambda cfg: cfg)
    row = next(r for r in candidate_rows() if r['prototype_weight'] == 2. and r['model_seed'] == 2026092701)
    cfg = configuration({}, row, '/project', 'commit')
    assert cfg['source_prototype']['weight'] == 2.
    assert cfg['source_prototype']['temperature'] == .2
    assert cfg['shared_prefix'] == '/original/e20'
    validate(cfg)
    cfg['source_development'] = {'train_ids': ['a'], 'dev_ids': ['b']}
    with pytest.raises(ValueError):
        validate(cfg)


def test_finished_training_prediction_is_not_delayed_by_other_training():
    from experiments.pamf_branch_training.runner import choose_gpu_job
    jobs = [{'id': 'new_train', 'kind': 'train'}, {'id': 'ready_prediction', 'kind': 'predict'}]
    assert choose_gpu_job(jobs, prioritize_prediction=True)['id'] == 'ready_prediction'
    assert choose_gpu_job(jobs, prioritize_prediction=False)['id'] == 'new_train'


def test_each_full_source_row_automatically_chains_seven_view_prediction_and_own_score(monkeypatch, tmp_path):
    from experiments.pamf_physics_tri import target_tuning_plan as plan, target_tuning_runner as runner
    from experiments.pamf_repl_view_shuffle.plan import VIEWS
    from experiments.pamf_single_view_paper.plan import TASKS
    monkeypatch.setattr(plan, 'configuration', lambda ctx, row, project, commit, run_id=plan.RUN: {
        'run_id': plan.RUN, 'row_id': row['row_id'], 'paper_row': {'row_id': 'paper'},
        'source_contract': 'source.json', 'output_root': str(tmp_path / 'source' / row['row_id'])})
    handles = {pid: {'source_contract': pid + '/contract', 'classes': ['a', 'b'],
                     'tests': {task: {'views_root': f'{pid}/{task}', 'truth': f'{pid}/{task}/truth'} for task in TASKS}}
               for pid in ('ManyRx', 'label01', 'm6-natural')}
    jobs, _, configs = runner.make_jobs(str(tmp_path / 'run'), '/project', {'handles': handles}, 'commit', {})
    assert len(jobs) == 99
    row = plan.candidate_rows()[0]
    train = next(j for j in jobs if j['id'] == 'train__' + row['row_id'])
    pred = next(j for j in jobs if j['id'] == 'predict__' + row['row_id'])
    score = next(j for j in jobs if j['id'] == 'score__' + row['row_id'])
    assert pred['dependencies'] == [train['completion']]
    assert score['dependencies'] == [pred['completion']]
    assert configs[pred['id']]['views'] == list(VIEWS)
    assert configs[pred['id']]['readouts'] == ['raw']
    assert all('truth' not in item for item in configs[pred['id']]['tests'].values())


def test_summary_pools_two_test_populations_by_sample_count():
    from experiments.pamf_physics_tri.target_tuning_report import pooled
    from experiments.pamf_repl_view_shuffle.plan import VIEWS
    records = []
    for view in VIEWS:
        records += [dict(dimension='overall', readout='raw', test='cross_rx', view=view, count=2, accuracy=1.),
                    dict(dimension='overall', readout='raw', test='cross_both', view=view, count=8, accuracy=.5)]
        records.append(dict(dimension='overall', readout='raw', test='cross_date', view=view, count=100, accuracy=0.))
    result = pooled(records)
    assert result['clean'] == pytest.approx(.6)
    assert result['practical_mean'] == pytest.approx(.6)
    assert result['counts']['clean'] == 10
