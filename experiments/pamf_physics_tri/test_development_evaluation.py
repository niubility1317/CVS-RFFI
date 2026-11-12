from experiments.pamf_physics_tri.development_evaluation_plan import rows, make_jobs, source_path


def test_each_completed_model_has_own_seven_view_prediction_and_score(tmp_path):
    candidates = rows()
    assert len(candidates) == 3
    ctx = {'handles': {'ManyRx': {'source_contract': 'contract.json', 'classes': ['a', 'b'],
             'tests': {task: {'views_root': f'{task}/views', 'truth': f'{task}/truth'}
                       for task in ('cross_date', 'cross_rx', 'cross_both')}}}}
    jobs, _, configs = make_jobs(str(tmp_path), '/project', ctx, {})
    assert len(jobs) == 6
    for row in candidates:
        prediction = next(j for j in jobs if j['id'] == 'predict__' + row['row_id'])
        score = next(j for j in jobs if j['id'] == 'score__' + row['row_id'])
        pred_cfg, score_cfg = configs[prediction['id']], configs[score['id']]
        assert len(pred_cfg['views']) == 7 and pred_cfg['readouts'] == ['raw']
        assert all('truth' not in value for value in pred_cfg['tests'].values())
        assert score['dependencies'] == [prediction['completion']]
        assert score_cfg['prediction_output'] == pred_cfg['output_root']
        assert prediction['dependencies'] == [str(source_path(row, '/project') / 'complete.json')]


def test_predict_uses_direct_readout_and_restores_existing_setting(monkeypatch):
    from experiments.pamf_paper_completion import evaluation as paper
    from experiments.pamf_single_view_paper import evaluation as native
    from experiments.pamf_physics_tri.development_evaluation import predict
    before = native.READOUTS

    def fake(cfg):
        assert native.READOUTS == ('raw',)
        assert cfg['readouts'] == ['raw']
        return 'done'

    monkeypatch.setattr(paper, 'predict', fake)
    assert predict({'readouts': ['raw']}) == 'done'
    assert native.READOUTS == before
